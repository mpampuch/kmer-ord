# tests/test_run_benchmarks.py
"""Tests for the standalone benchmark runner (benchmarks/run_benchmarks.py)."""
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "run_benchmarks.py"

spec = importlib.util.spec_from_file_location("run_benchmarks", RUNNER_PATH)
run_benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_benchmarks)


def test_synthetic_matrix_deterministic(tmp_path):
    """Same seed must produce byte-identical matrices (benchmarks across
    commits are only comparable if the input data is identical)."""
    a = tmp_path / "a.tsv"
    b = tmp_path / "b.tsv"
    run_benchmarks.generate_synthetic_matrix(a, n_reads=100, n_features=20, seed=7)
    run_benchmarks.generate_synthetic_matrix(b, n_reads=100, n_features=20, seed=7)
    assert a.read_bytes() == b.read_bytes()

    c = tmp_path / "c.tsv"
    run_benchmarks.generate_synthetic_matrix(c, n_reads=100, n_features=20, seed=8)
    assert a.read_bytes() != c.read_bytes()


def test_synthetic_matrix_shape_and_dtype(tmp_path):
    import pandas as pd

    path = tmp_path / "m.tsv"
    run_benchmarks.generate_synthetic_matrix(path, n_reads=50, n_features=10, seed=1)
    df = pd.read_csv(path, sep="\t", index_col=0)
    assert df.shape == (50, 10)
    assert df.index[0] == "read_00000000"
    assert (df.values >= 0).all()


def test_all_stages_registered():
    assert set(run_benchmarks.STAGES) == set(run_benchmarks.STAGE_NAMES)


def test_run_stage_appends_to_explicit_log_dir(tmp_path):
    """The standalone harness must keep writing to the --log-dir it is given
    (benchmarks/ by default), including after the parent_label schema change."""
    matrix = tmp_path / "m.tsv"
    run_benchmarks.generate_synthetic_matrix(matrix, n_reads=40, n_features=12, seed=1)
    log_dir = tmp_path / "logs"
    workdir = tmp_path / "work"
    run_benchmarks.run_stage_in_this_process(
        stage="preprocess_clr",
        matrix_path=matrix,
        tier="small",
        log_dir=log_dir,
        workdir=workdir,
    )
    log_file = log_dir / "benchmark_log.tsv"
    assert log_file.exists()
    import csv
    from kmer_ord.utils.benchmark import LOG_COLUMNS
    with open(log_file) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert list(rows[0].keys()) == LOG_COLUMNS
    assert rows[0]["script_name"] == "run_benchmarks"
    assert rows[0]["stage_label"] == "bench_small_preprocess_clr"
    assert rows[0]["parent_label"] == "N/A"
