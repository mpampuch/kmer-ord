# tests/test_kmer_stats_golden.py
"""Golden-output regression tests for k-mer metrics.

Metric *values* are frozen from the original implementation's math (float64
reference below). Column *names* are the corrected ones from the memory
audit — the originals were misleading (`total_nonzero_kmers` was actually the
total count sum, `shannon_evenness` was raw entropy in nats):

    total_kmer_counts     row sum of counts (was: total_nonzero_kmers)
    num_nonzero_kmers     number of k-mer types present (was: num_unique_kmers)
    shannon_entropy_nats  Shannon entropy, natural log (was: shannon_evenness)
    shannon_entropy_bits  Shannon entropy, log2 (was: shannon_diversity)
    shannon_evenness      NEW: normalized entropy H / ln(S), 0-1
"""
import csv
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kmer_ord.io.kmer_stats import METRIC_COLUMNS, RunningStats, process_kmer_file

# float32 pipeline vs float64 reference
RTOL = 1e-4
ATOL = 1e-6


def write_matrix_tsv(path: Path, counts: np.ndarray) -> pd.DataFrame:
    index = [f"read_{i:05d}" for i in range(counts.shape[0])]
    columns = [f"kmer_{j:04d}" for j in range(counts.shape[1])]
    df = pd.DataFrame(counts, index=index, columns=columns)
    df.index.name = "sequence_id"
    df.to_csv(path, sep="\t")
    return df


@pytest.fixture
def matrix_file(tmp_path) -> tuple[Path, pd.DataFrame]:
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=4.0, size=(100, 12)).astype(np.uint32)
    counts[3] = 0                      # all-zero row
    counts[7, :11] = 0                 # single-kmer row (S == 1 evenness case)
    counts[7, 11] = 9
    path = tmp_path / "matrix.tsv"
    df = write_matrix_tsv(path, counts)
    return path, df


def reference_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Frozen float64 reference of the original metric math, plus the
    normalized evenness added by the audit."""
    values = df.to_numpy(dtype=np.float64)

    nonzero = (values != 0).sum(axis=1)
    row_sums = values.sum(axis=1)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    probs = values / row_sums_safe[:, None]
    positive = probs > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        nats = -np.sum(np.where(positive, probs * np.log(probs), 0.0), axis=1)
        bits = -np.sum(np.where(positive, probs * np.log2(probs), 0.0), axis=1)

    evenness = np.where(nonzero > 1, nats / np.log(np.maximum(nonzero, 2)), 1.0)

    return pd.DataFrame(
        {
            "total_kmer_counts": row_sums.astype("int64"),
            "num_nonzero_kmers": nonzero,
            "shannon_entropy_nats": nats,
            "shannon_entropy_bits": bits,
            "shannon_evenness": evenness,
        },
        index=df.index,
    )


def read_metrics(output_file: Path) -> pd.DataFrame:
    return pd.read_csv(output_file, sep="\t", index_col=0)


def assert_matches_reference(result: pd.DataFrame, expected: pd.DataFrame):
    assert list(result.columns) == METRIC_COLUMNS == list(expected.columns)
    assert list(result.index) == list(expected.index)
    for col in expected.columns:
        np.testing.assert_allclose(
            result[col].to_numpy(dtype=np.float64),
            expected[col].to_numpy(dtype=np.float64),
            rtol=RTOL, atol=ATOL, err_msg=f"metric '{col}' diverged",
        )


def test_metrics_match_reference_single_chunk(matrix_file, tmp_path):
    path, df = matrix_file
    out = tmp_path / "metrics.tsv"
    process_kmer_file(input_file=str(path), output_file=str(out), cpus=1)
    assert_matches_reference(read_metrics(out), reference_metrics(df))


def test_metrics_identical_across_chunk_sizes(matrix_file, tmp_path):
    """Chunking is a memory detail; it must never change values or row order."""
    path, df = matrix_file
    out_small = tmp_path / "small_chunks.tsv"
    out_big = tmp_path / "big_chunks.tsv"
    process_kmer_file(str(path), str(out_small), chunksize=7, cpus=1)
    process_kmer_file(str(path), str(out_big), chunksize=100_000, cpus=1)
    pd.testing.assert_frame_equal(read_metrics(out_small), read_metrics(out_big))
    assert_matches_reference(read_metrics(out_small), reference_metrics(df))


def test_metrics_identical_with_parallel_workers(matrix_file, tmp_path):
    path, df = matrix_file
    out = tmp_path / "parallel.tsv"
    process_kmer_file(str(path), str(out), chunksize=13, cpus=2)
    assert_matches_reference(read_metrics(out), reference_metrics(df))


def test_returns_output_path(matrix_file, tmp_path):
    path, _ = matrix_file
    out = tmp_path / "metrics.tsv"
    result = process_kmer_file(str(path), str(out), cpus=1)
    assert Path(result) == out


def test_streaming_memory_bounded(tmp_path):
    """The core Phase-1 property: peak allocations must scale with one chunk,
    not with the whole file (the old code did `chunks = list(reader)`).

    50,000 x 200 uint32 is ~40 MB fully materialized (double that transiently
    for the old float64 conversion); one 2,000-row chunk is ~1.6 MB. The 12 MB
    bound fails the old all-chunks-resident implementation with a wide margin
    for parser buffers, while streaming stays comfortably below it.
    """
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=4.0, size=(50_000, 200)).astype(np.uint32)
    path = tmp_path / "big.tsv"
    write_matrix_tsv(path, counts)
    del counts

    out = tmp_path / "metrics.tsv"
    tracemalloc.start()
    tracemalloc.reset_peak()
    process_kmer_file(str(path), str(out), chunksize=2_000, cpus=1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 12 * 1024 * 1024, (
        f"peak traced allocations {peak / 1e6:.1f} MB suggest the matrix is "
        "not being streamed chunk-by-chunk"
    )

    # sanity: the streamed output still covers every row
    with open(out) as f:
        n_rows = sum(1 for _ in f) - 1
    assert n_rows == 50_000


def test_running_stats_matches_numpy():
    """The dataset-wide summary uses running (Welford) accumulators instead of
    retaining all metrics; they must agree with direct numpy computation."""
    rng = np.random.default_rng(1)
    data = rng.normal(loc=5.0, scale=2.0, size=10_000)

    rs = RunningStats()
    # feed in uneven batches to exercise the merge path
    for batch in np.array_split(data, [17, 400, 401, 5000]):
        rs.update(batch)

    assert rs.n == data.size
    np.testing.assert_allclose(rs.mean, data.mean(), rtol=1e-10)
    np.testing.assert_allclose(rs.std(ddof=1), data.std(ddof=1), rtol=1e-9)
    assert rs.min == data.min()
    assert rs.max == data.max()


def test_running_stats_empty_and_single():
    rs = RunningStats()
    assert rs.n == 0
    rs.update(np.array([2.5]))
    assert rs.mean == 2.5
    assert np.isnan(rs.std(ddof=1))  # undefined for n == 1


def test_output_written_incrementally_headers_once(matrix_file, tmp_path):
    """Appending per chunk must produce exactly one header row."""
    path, _ = matrix_file
    out = tmp_path / "metrics.tsv"
    process_kmer_file(str(path), str(out), chunksize=10, cpus=1)
    with open(out) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header_count = sum(1 for r in rows if r and r[0] == "sequence_id")
    assert header_count == 1
    assert len(rows) == 101  # header + 100 reads
