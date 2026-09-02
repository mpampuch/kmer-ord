# tests/test_loader.py
"""Tests for load_matrix: the matrix must be parsed directly into float32 so
it exists in RAM exactly once (the old path loaded as int64/float64 and let
preprocess_data make a float32 copy afterwards)."""
import numpy as np
import pandas as pd
import pytest

from kmer_ord.dr.loader import load_matrix


@pytest.fixture
def matrix_tsv(tmp_path):
    rng = np.random.default_rng(3)
    counts = rng.poisson(lam=5.0, size=(30, 8)).astype(np.uint32)
    df = pd.DataFrame(
        counts,
        index=[f"read_{i:03d}" for i in range(30)],
        columns=[f"kmer_{j}" for j in range(8)],
    )
    df.index.name = "sequence_id"
    path = tmp_path / "matrix.tsv"
    df.to_csv(path, sep="\t")
    return path, df


def test_tsv_loaded_as_float32(matrix_tsv):
    path, original = matrix_tsv
    loaded = load_matrix(path)
    assert all(dt == np.float32 for dt in loaded.dtypes)
    assert list(loaded.index) == list(original.index)
    assert list(loaded.columns) == list(original.columns)
    np.testing.assert_array_equal(
        loaded.to_numpy(dtype=np.float64), original.to_numpy(dtype=np.float64)
    )


def test_csv_loaded_as_float32(matrix_tsv, tmp_path):
    _, original = matrix_tsv
    path = tmp_path / "matrix.csv"
    original.to_csv(path, sep=",")
    loaded = load_matrix(path)
    assert all(dt == np.float32 for dt in loaded.dtypes)
    np.testing.assert_array_equal(
        loaded.to_numpy(dtype=np.float64), original.to_numpy(dtype=np.float64)
    )


def test_npy_loaded_as_float32(tmp_path):
    arr = np.arange(20, dtype=np.int64).reshape(4, 5)
    path = tmp_path / "matrix.npy"
    np.save(path, arr)
    loaded = load_matrix(path)
    assert all(dt == np.float32 for dt in loaded.dtypes)
    np.testing.assert_array_equal(loaded.to_numpy(), arr.astype(np.float32))


def test_non_numeric_column_raises(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("sequence_id\tkmer_0\tkmer_1\nread_0\t3\toops\nread_1\t1\t2\n")
    with pytest.raises(ValueError):
        load_matrix(path)


def test_too_few_samples_raises(tmp_path):
    path = tmp_path / "one_row.tsv"
    path.write_text("sequence_id\tkmer_0\nread_0\t3\n")
    with pytest.raises(ValueError, match="at least 2 samples"):
        load_matrix(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_matrix(tmp_path / "nope.tsv")


def test_unsupported_format_raises(tmp_path):
    path = tmp_path / "matrix.parquet"
    path.write_text("not really parquet")
    with pytest.raises(ValueError, match="Unsupported matrix format"):
        load_matrix(path)
