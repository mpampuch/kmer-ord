# src/kmer_ord/utils/benchmark.py
"""Peak-RAM and timing instrumentation for pipeline stages.

BenchmarkTimer is a context manager that samples the resident set size (RSS)
of this process *and all child processes* on a background thread for the
duration of the block, and appends one row per block to benchmark_log.tsv.
A sibling benchmark_hr_log.tsv mirrors that file with byte columns scaled
to B/KB/MB/GB/TB/PB.

Peak RSS is sampled rather than derived from start/end deltas because memory
freed before the block exits (the common case for numeric pipelines) is
invisible to a delta, and child processes (ProcessPoolExecutor workers, the
Rust k-mer counter) are invisible to the parent's own RSS entirely.

peak_rss_children_bytes is the max sum of per-process RSS. That is the right
figure for one threaded child (threads share one RSS) and too high when
several processes share pages, because RSS counts those pages once per
process. peak_pss_tree_bytes is the simultaneous total of proportional set
size (PSS) for this process and its children, which splits shared pages
across sharers. It is N/A when smaps_rollup does not cover every process in
a sample (like potentially on MacOS or if there are permission issues). 
peak_cgroup_bytes is the cgroup memory high-water mark when the process is 
inside a job cgroup. This is relevant for HPC jobs and is the the quantity 
that SLURM and Nextflow report.
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
    "peak_pss_tree_bytes",
    "peak_cgroup_bytes",
    "n_children_at_peak",
    "status",
    "error",
]

# Same names as the raw log so the two TSVs line up column for column.
BYTE_COLUMNS = [column for column in LOG_COLUMNS if column.endswith("_bytes")]

# 1024-based, matching the MB/GB labels already used by run_benchmarks.
_BYTE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def format_bytes(n) -> str:
    """Scale a byte count to B/KB/MB/GB/TB/PB.

    Missing measurements stay N/A. Values under 1024 stay integer bytes;
    larger units use two decimal places (1.00 KB, 1.02 GB).
    """
    if n in (None, "", "N/A"):
        return "N/A"
    try:
        value = float(n)
    except (TypeError, ValueError):
        return str(n)
    unit_index = 0
    while abs(value) >= 1024 and unit_index < len(_BYTE_UNITS) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} B"
    return f"{value:.2f} {_BYTE_UNITS[unit_index]}"


# Innermost active timer label, so nested BenchmarkTimer rows can record
# which parent stage they belong to without every call site passing it.
_current_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "benchmark_parent", default=None
)

# 50 ms balances sampling overhead against the risk of missing short-lived
# allocation spikes; the stages being measured run for seconds to hours.
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


def _parse_smaps_rollup_pss(text: str) -> int | None:
    """Return PSS in bytes from a smaps_rollup body, or None if the field is absent."""
    for line in text.splitlines():
        if line.startswith("Pss:"):
            # "Pss:               12345 kB"
            try:
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def _pss_bytes(pid: int) -> int | None:
    """Proportional set size of one process, from Linux smaps_rollup.

    None on macOS, on older kernels, and when the process exits mid-read.
    A sample that includes any None is omitted from the PSS peak.
    """
    path = f"/proc/{pid}/smaps_rollup"
    try:
        with open(path) as handle:
            return _parse_smaps_rollup_pss(handle.read())
    except OSError:
        return None


def _cgroup_usage_path() -> str | None:
    """Usage file for this process's memory cgroup, or None at the root / off Linux.

    The root cgroup's memory.current is the whole machine, so it is not a
    per-job peak. SLURM and container jobs sit in a child cgroup.
    """
    try:
        with open("/proc/self/cgroup") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1], parts[2]
        if path in ("", "/"):
            continue
        if controllers == "":
            # cgroup v2 unified hierarchy
            candidate = f"/sys/fs/cgroup{path}/memory.current"
        elif "memory" in controllers.split(","):
            candidate = f"/sys/fs/cgroup/memory{path}/memory.usage_in_bytes"
        else:
            continue
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_cgroup_usage(path: str | None) -> int | None:
    if not path:
        return None
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _tsv_safe(text: str) -> str:
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


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
    """Background thread tracking peak RSS, PSS, and cgroup usage.

    peak_children is the max sum of child RSS. Shared pages are included in
    every process's RSS, so that sum can exceed physical RAM.
    peak_pss_tree is the max simultaneous PSS of self and children, or None
    when no sample had smaps_rollup for every process still alive.
    n_children_at_peak is how many child processes were alive on the sample
    that set peak_children, so a huge child-RSS figure can be checked against
    a process count.
    """

    def __init__(self, interval_s: float = _SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.peak_self = 0
        self.peak_children = 0
        self.peak_pss_tree = None
        self.peak_cgroup = None
        self.n_children_at_peak = 0
        self._proc = psutil.Process()
        self._cgroup_path = _cgroup_usage_path()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            rss = self._proc.memory_info().rss
        except psutil.Error:
            return
        self.peak_self = max(self.peak_self, rss)
        # A mixed RSS/PSS sum would still be stored under the PSS column.
        # Skip the peak unless every live process in this sample has PSS.
        self_pss = _pss_bytes(self._proc.pid)
        pss_complete = self_pss is not None
        tree_pss = 0 if self_pss is None else self_pss

        children_rss = 0
        n_children = 0
        try:
            children = self._proc.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                child_rss = child.memory_info().rss
            except psutil.Error:
                # child exited between enumeration and sampling
                continue
            children_rss += child_rss
            n_children += 1
            child_pss = _pss_bytes(child.pid)
            if child_pss is None:
                pss_complete = False
            else:
                tree_pss += child_pss

        if children_rss > self.peak_children:
            self.peak_children = children_rss
            self.n_children_at_peak = n_children
        if pss_complete:
            self.peak_pss_tree = (
                tree_pss if self.peak_pss_tree is None else max(self.peak_pss_tree, tree_pss)
            )

        usage = _read_cgroup_usage(self._cgroup_path)
        if usage is not None:
            self.peak_cgroup = usage if self.peak_cgroup is None else max(self.peak_cgroup, usage)

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

    Writes raw bytes to benchmark_log.tsv and a human-readable mirror to
    benchmark_hr_log.tsv in the same directory.

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
        self.peak_pss_tree = None
        self.peak_cgroup = None
        self.n_children_at_peak = None
        self.status = "ok"
        self.error = None
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
            self.peak_pss_tree = self._sampler.peak_pss_tree
            self.peak_cgroup = self._sampler.peak_cgroup
            self.n_children_at_peak = self._sampler.n_children_at_peak
            if exc_type is not None:
                self.status = "failed"
                self.error = _tsv_safe(f"{exc_type.__name__}: {exc_val}")
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
                self.peak_pss_tree if self.peak_pss_tree is not None else "N/A",
                self.peak_cgroup if self.peak_cgroup is not None else "N/A",
                self.n_children_at_peak,
                self.status,
                self.error or "N/A",
            ])
        self._write_hr_log()

    def _write_hr_log(self):
        """Rewrite benchmark_hr_log.tsv from the current raw-byte log.

        A full rewrite (not a parallel append) keeps historical rows in sync
        after an in-place schema upgrade or a legacy-header rotation.
        """
        hr_file = os.path.join(self.log_dir, "benchmark_hr_log.tsv")
        with open(self.log_file, newline="") as raw:
            reader = csv.DictReader(raw, delimiter="\t")
            rows = list(reader)
            fieldnames = reader.fieldnames or LOG_COLUMNS
        with open(hr_file, "w", newline="") as hr:
            writer = csv.DictWriter(
                hr, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                for column in BYTE_COLUMNS:
                    if column in row:
                        row[column] = format_bytes(row[column])
                writer.writerow(row)
