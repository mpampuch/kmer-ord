# src/kmer_ord/utils/benchmark.py
"""Peak-RAM and timing instrumentation for pipeline stages.

BenchmarkTimer is a context manager that samples the resident set size (RSS)
of this process *and all child processes* on a background thread for the
duration of the block, and appends one row per block to a TSV log.

Peak RSS is sampled rather than derived from start/end deltas because memory
freed before the block exits (the common case for numeric pipelines) is
invisible to a delta, and child processes (ProcessPoolExecutor workers, the
Rust k-mer counter) are invisible to the parent's own RSS entirely.
"""
import contextvars
import csv
import functools
import os
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

# Columns written to benchmark_log.tsv. Kept in one place so tests and any
# downstream analysis scripts can import the schema instead of hardcoding it.
LOG_COLUMNS = [
    "timestamp",
    "git_commit",
    "script_name",
    "stage_label",
    "parent_label",
    "input_file",
    "input_file_size_bytes",
    "input_rows",
    "input_cols",
    "input_args",
    "wall_time_s",
    "cpu_time_s",
    "peak_rss_self_bytes",
    "peak_rss_children_bytes",
    "end_rss_bytes",
    "ru_maxrss_bytes",
]

# Innermost active timer label, so nested BenchmarkTimer rows can record
# which parent stage they belong to without every call site passing it.
_current_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "benchmark_parent", default=None
)

_SAMPLE_INTERVAL_S = 0.05


@functools.lru_cache(maxsize=1)
def _get_git_commit() -> str:
    """Short commit hash of the kmer-ord source tree, '-dirty' if modified.

    Resolved relative to this file (not the caller's cwd) so logs identify the
    code version even when the pipeline runs from an arbitrary directory.
    """
    src_dir = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=src_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not commit:
            return "N/A"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=src_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.SubprocessError):
        return "N/A"


def _ru_maxrss_bytes() -> int:
    """Lifetime peak RSS of this process from the kernel, normalized to bytes.

    macOS reports ru_maxrss in bytes, Linux in kilobytes. This is a
    cross-check for the sampler: it covers the whole process lifetime (not
    just the timed block) and excludes children, so it can legitimately
    exceed peak_rss_self for late-pipeline stages.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


class _PeakRssSampler:
    """Background thread tracking peak RSS of this process and its children."""

    def __init__(self, interval_s: float = _SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.peak_self = 0
        self.peak_children = 0
        self._proc = psutil.Process()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            rss = self._proc.memory_info().rss
        except psutil.Error:
            return
        self.peak_self = max(self.peak_self, rss)

        children_rss = 0
        try:
            for child in self._proc.children(recursive=True):
                try:
                    children_rss += child.memory_info().rss
                except psutil.Error:
                    # child exited between enumeration and sampling
                    continue
        except psutil.Error:
            pass
        self.peak_children = max(self.peak_children, children_rss)

    def _run(self):
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval_s)

    def start(self):
        self._sample()  # baseline sample so even instant blocks get a value
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._sample()  # final sample to catch state at block exit


class BenchmarkTimer:
    """Context manager logging wall time, CPU time and peak RSS to a TSV.

    Usage:
        with BenchmarkTimer(label="stage", input_file=path) as bt:
            ...
            bt.record_input_shape(n_rows, n_cols)  # optional
    """

    def __init__(self, label="Run", log_dir="benchmarking", script_name=None,
                 input_file=None, input_args=None):
        self.label = label
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "benchmark_log.tsv")
        self.script_name = script_name
        self.input_file = str(input_file) if input_file else None
        self.input_args = input_args
        self.input_rows = None
        self.input_cols = None

        os.makedirs(self.log_dir, exist_ok=True)

        self._sampler = None
        self._parent_token = None
        self.parent_label = None
        self.start_time = None
        self.start_cpu_time = None
        self.wall_time = None
        self.cpu_time = None
        self.peak_rss_self = None
        self.peak_rss_children = None
        self.end_rss = None

    def record_input_shape(self, n_rows: int, n_cols: int):
        """Attach input matrix dimensions (usually known only after loading)."""
        self.input_rows = n_rows
        self.input_cols = n_cols

    def __enter__(self):
        # snapshot the outer label before we become the current parent
        self.parent_label = _current_parent.get()
        self._parent_token = _current_parent.set(self.label)
        self._sampler = _PeakRssSampler()
        self._sampler.start()
        self.start_time = time.time()
        self.start_cpu_time = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.wall_time = time.time() - self.start_time
            self.cpu_time = time.process_time() - self.start_cpu_time
            self._sampler.stop()
            self.peak_rss_self = self._sampler.peak_self
            self.peak_rss_children = self._sampler.peak_children
            self.end_rss = psutil.Process().memory_info().rss
            self._log_metrics()
        finally:
            if self._parent_token is not None:
                _current_parent.reset(self._parent_token)

    def _prepare_log_file(self):
        """Make sure an existing log matches LOG_COLUMNS before appending.

        Compatible upgrades (new columns whose names are a superset of the
        old header, e.g. adding parent_label) are rewritten in place with
        N/A for missing fields so historical rows stay in the same file.
        Incompatible headers are rotated to benchmark_log_legacy_<stamp>.tsv.
        """
        with open(self.log_file, newline="") as f:
            existing_header = f.readline().rstrip("\n").rstrip("\r").split("\t")
            body = f.read()
        if existing_header == LOG_COLUMNS:
            return
        if existing_header and set(existing_header) <= set(LOG_COLUMNS):
            import io
            reader = csv.DictReader(
                io.StringIO("\t".join(existing_header) + "\n" + body),
                delimiter="\t",
            )
            rows = list(reader)
            with open(self.log_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=LOG_COLUMNS, delimiter="\t", extrasaction="ignore"
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            col: (
                                row[col]
                                if col in row and row[col] not in (None, "")
                                else "N/A"
                            )
                            for col in LOG_COLUMNS
                        }
                    )
            return
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        rotated = os.path.join(self.log_dir, f"benchmark_log_legacy_{stamp}.tsv")
        os.rename(self.log_file, rotated)

    def _log_metrics(self):
        if os.path.isfile(self.log_file):
            self._prepare_log_file()
        log_exists = os.path.isfile(self.log_file)

        input_file_size = (
            os.path.getsize(self.input_file)
            if self.input_file and os.path.exists(self.input_file)
            else "N/A"
        )

        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            if not log_exists:
                writer.writerow(LOG_COLUMNS)
            writer.writerow([
                datetime.now().isoformat(),
                _get_git_commit(),
                self.script_name or "N/A",
                self.label,
                self.parent_label or "N/A",
                self.input_file or "N/A",
                input_file_size,
                self.input_rows if self.input_rows is not None else "N/A",
                self.input_cols if self.input_cols is not None else "N/A",
                self.input_args or "N/A",
                f"{self.wall_time:.4f}",
                f"{self.cpu_time:.4f}",
                self.peak_rss_self,
                self.peak_rss_children,
                self.end_rss,
                _ru_maxrss_bytes(),
            ])
