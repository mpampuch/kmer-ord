# tests/test_benchmark.py
"""Tests for the peak-RSS BenchmarkTimer.

The key property under test: the timer must capture the *peak* resident set
size during the block (including memory freed before the block exits, and
memory allocated by child processes), not the start->end RSS delta.
"""
import csv
import subprocess
import sys
import time

import numpy as np
import psutil

from kmer_ord.utils.benchmark import BenchmarkTimer, LOG_COLUMNS

# Big enough to stand out over sampler noise, small enough to be fast.
ALLOC_BYTES = 300 * 1024 * 1024  # 300 MiB
# Peak detection tolerance: the allocation must be visible at >= 2/3 its size.
MIN_DETECTED = ALLOC_BYTES * 2 // 3


def _hold_for_sampler():
    """Sleep long enough for the ~50 ms sampler to observe current RSS."""
    time.sleep(0.5)


def test_peak_captured_even_after_free(tmp_path):
    """Memory allocated then freed inside the block must still show in the peak."""
    start_rss = psutil.Process().memory_info().rss

    with BenchmarkTimer(label="alloc_free", log_dir=str(tmp_path)) as bt:
        arr = np.ones(ALLOC_BYTES // 8, dtype=np.float64)  # touch pages
        _hold_for_sampler()
        del arr
        time.sleep(0.2)

    assert bt.peak_rss_self >= start_rss + MIN_DETECTED, (
        f"peak_rss_self={bt.peak_rss_self} did not capture a "
        f"{ALLOC_BYTES} byte allocation over baseline {start_rss}"
    )


def test_child_process_memory_captured(tmp_path):
    """Allocations in child processes must be visible in peak_rss_children."""
    child_code = (
        "import numpy, time; "
        f"a = numpy.ones({ALLOC_BYTES} // 8, dtype=numpy.float64); "
        "time.sleep(1.5)"
    )
    with BenchmarkTimer(label="child_alloc", log_dir=str(tmp_path)) as bt:
        proc = subprocess.Popen([sys.executable, "-c", child_code])
        try:
            proc.wait(timeout=30)
        finally:
            proc.kill()

    assert bt.peak_rss_children >= MIN_DETECTED, (
        f"peak_rss_children={bt.peak_rss_children} missed a "
        f"{ALLOC_BYTES} byte child allocation"
    )


def test_log_row_written_with_schema(tmp_path):
    with BenchmarkTimer(
        label="log_test",
        log_dir=str(tmp_path),
        script_name="test_script",
        input_args="k=6",
    ) as bt:
        bt.record_input_shape(123, 456)
        time.sleep(0.1)

    log_file = tmp_path / "benchmark_log.tsv"
    assert log_file.exists()

    with open(log_file) as f:
        rows = list(csv.reader(f, delimiter="\t"))

    header, row = rows[0], rows[1]
    assert header == LOG_COLUMNS
    record = dict(zip(header, row))
    assert record["stage_label"] == "log_test"
    assert record["script_name"] == "test_script"
    assert record["input_rows"] == "123"
    assert record["input_cols"] == "456"
    assert float(record["wall_time_s"]) >= 0.1
    assert int(record["peak_rss_self_bytes"]) > 0
    assert int(record["end_rss_bytes"]) > 0
    # git commit is best-effort but should resolve inside this repo
    assert record["git_commit"] not in ("", "N/A")


def test_old_format_log_rotated_not_corrupted(tmp_path):
    """A pre-existing log with the legacy header must be rotated aside, not
    appended to (mixing schemas would corrupt the TSV)."""
    log_file = tmp_path / "benchmark_log.tsv"
    legacy_header = "timestamp\tscript_name\tinput_file\tlegacy_col\n"
    log_file.write_text(legacy_header + "old\trow\tof\tdata\n")

    with BenchmarkTimer(label="rotation_test", log_dir=str(tmp_path)):
        pass

    with open(log_file) as f:
        header = f.readline().strip().split("\t")
    assert header == LOG_COLUMNS

    rotated = list(tmp_path.glob("benchmark_log_legacy*.tsv"))
    assert len(rotated) == 1
    assert "legacy_col" in rotated[0].read_text()


def test_multiple_rows_append(tmp_path):
    for i in range(2):
        with BenchmarkTimer(label=f"run{i}", log_dir=str(tmp_path)):
            pass

    log_file = tmp_path / "benchmark_log.tsv"
    with open(log_file) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert len(rows) == 3  # header + 2 data rows
    assert rows[1][LOG_COLUMNS.index("stage_label")] == "run0"
    assert rows[2][LOG_COLUMNS.index("stage_label")] == "run1"
