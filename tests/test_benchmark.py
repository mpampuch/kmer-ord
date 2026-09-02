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
    assert record["parent_label"] == "N/A"
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


def test_compatible_schema_upgraded_in_place(tmp_path):
    """Adding parent_label must rewrite the existing log in place so
    benchmarks/benchmark_log.tsv keeps its history instead of rotating."""
    log_file = tmp_path / "benchmark_log.tsv"
    old_header = [c for c in LOG_COLUMNS if c != "parent_label"]
    old_row = {c: "old" if c != "wall_time_s" else "1.0" for c in old_header}
    old_row["stage_label"] = "bench_small_kmer_stats"
    old_row["script_name"] = "run_benchmarks"
    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_header, delimiter="\t")
        writer.writeheader()
        writer.writerow(old_row)

    with BenchmarkTimer(label="bench_small_pca_pre", log_dir=str(tmp_path),
                        script_name="run_benchmarks"):
        pass

    assert list(tmp_path.glob("benchmark_log_legacy*.tsv")) == []
    with open(log_file) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert list(rows[0].keys()) == LOG_COLUMNS
    assert rows[0]["stage_label"] == "bench_small_kmer_stats"
    assert rows[0]["parent_label"] == "N/A"
    assert rows[1]["stage_label"] == "bench_small_pca_pre"
    assert rows[1]["parent_label"] == "N/A"


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


def test_nested_timers_record_parent_label(tmp_path):
    with BenchmarkTimer(label="parent", log_dir=str(tmp_path), script_name="project"):
        with BenchmarkTimer(label="child", log_dir=str(tmp_path), script_name="project"):
            pass

    log_file = tmp_path / "benchmark_log.tsv"
    with open(log_file) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header = rows[0]
    assert header == LOG_COLUMNS
    child, parent = dict(zip(header, rows[1])), dict(zip(header, rows[2]))
    assert child["stage_label"] == "child"
    assert child["parent_label"] == "parent"
    assert parent["stage_label"] == "parent"
    assert parent["parent_label"] == "N/A"


def test_matrix_context_writes_under_output_dir(tmp_path):
    from kmer_ord.workflow.context import MatrixContext

    matrix = tmp_path / "m.tsv"
    matrix.write_text("sequence_id\tk1\nr1\t1\n")
    out = tmp_path / "run"
    ctx = MatrixContext(matrix, out, script_name="dr")
    with ctx.benchmark_timer("leaf"):
        pass

    log_file = out / "benchmarking" / "benchmark_log.tsv"
    assert log_file.exists()
    with open(log_file) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    record = dict(zip(rows[0], rows[1]))
    assert record["stage_label"] == "leaf"
    assert record["script_name"] == "dr"
    assert record["parent_label"] == "N/A"


def test_run_dr_methods_writes_parent_and_leaf_rows(tmp_path):
    import numpy as np
    from kmer_ord.dr.methods import run_dr_methods

    X = np.random.default_rng(0).normal(size=(40, 10)).astype(np.float32)
    log_dir = str(tmp_path)
    with BenchmarkTimer(
        label="dimensionality_reduction_clr",
        log_dir=log_dir,
        script_name="project",
    ):
        run_dr_methods(
            X=X,
            methods=["pca"],
            dims=2,
            seed=0,
            scale="small",
            screen_params=False,
            output_dir=tmp_path / "dr",
            normalisation="clr",
            input_name="test",
            sequence_ids=[f"s{i}" for i in range(40)],
            log_dir=log_dir,
            script_name="project",
        )

    log_file = tmp_path / "benchmark_log.tsv"
    with open(log_file) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    records = [dict(zip(rows[0], row)) for row in rows[1:]]
    labels = [r["stage_label"] for r in records]
    assert "dr_clr_pca" in labels
    assert "dimensionality_reduction_clr" in labels
    leaf = next(r for r in records if r["stage_label"] == "dr_clr_pca")
    parent = next(r for r in records if r["stage_label"] == "dimensionality_reduction_clr")
    assert leaf["parent_label"] == "dimensionality_reduction_clr"
    assert parent["parent_label"] == "N/A"
