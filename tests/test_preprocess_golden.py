# tests/test_preprocess_golden.py
"""Golden-output regression tests for matrix preprocessing.

The reference implementations below are frozen copies of the original
(pre-optimization) code paths. The package functions are allowed to change
*how* they compute (in-place ops, log-difference CLR, single float32 load),
but the resulting values must stay numerically identical to these references.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kmer_ord.dr.preprocess import preprocess_data, reduce_dimensions_with_pca

RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture
def kmer_counts() -> pd.DataFrame:
    """Small deterministic k-mer count matrix, including zero counts and one
    all-zero row (the edge cases the normalizations must handle)."""
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=3.0, size=(60, 25)).astype(np.int64)
    counts[5] = 0  # all-zero row: 'relative' must not divide by zero
    index = [f"read_{i:04d}" for i in range(60)]
    columns = [f"kmer_{j:03d}" for j in range(25)]
    return pd.DataFrame(counts, index=index, columns=columns)


# ---------------------------------------------------------------------------
# Frozen reference implementations (copied verbatim from the original code)
# ---------------------------------------------------------------------------

def reference_raw(df):
    return df.copy().astype(np.float32)


def reference_relative(df):
    X = df.copy().astype(np.float32)
    row_sums = X.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return X.div(row_sums, axis=0)


def reference_log(df):
    X = df.copy().astype(np.float32)
    X = np.log1p(X)
    return pd.DataFrame(X, index=df.index, columns=df.columns)


def reference_clr(df):
    X = df.copy().astype(np.float32)
    X += 1e-9
    geometric_mean = np.exp(np.mean(np.log(X), axis=1))
    return np.log(X.div(geometric_mean, axis=0))


def reference_zscore(df):
    X = df.copy().astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, index=df.index, columns=df.columns)


def reference_variance_pca(df, keep_variance):
    pca_full = PCA()
    pca_full.fit_transform(df.values)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    keep_pcs = int(np.searchsorted(cumulative, keep_variance) + 1)
    pca = PCA(n_components=keep_pcs)
    return pca.fit_transform(df.values), keep_pcs


# ---------------------------------------------------------------------------
# Normalization regression tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,reference", [
    ("raw", reference_raw),
    ("relative", reference_relative),
    ("log", reference_log),
    ("clr", reference_clr),
    ("zscore", reference_zscore),
])
def test_normalization_matches_reference(kmer_counts, method, reference):
    result = preprocess_data(kmer_counts, method)
    expected = reference(kmer_counts)

    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=RTOL, atol=ATOL,
        err_msg=f"'{method}' output diverged from the frozen reference",
    )


@pytest.mark.parametrize("method", ["raw", "relative", "log", "clr", "zscore"])
def test_normalization_preserves_ids_and_dtype(kmer_counts, method):
    result = preprocess_data(kmer_counts, method)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == list(kmer_counts.index)
    assert list(result.columns) == list(kmer_counts.columns)
    assert all(dt == np.float32 for dt in result.dtypes)


def test_normalization_does_not_mutate_input(kmer_counts):
    """The pipeline reuses the loaded matrix across normalisations, so
    preprocess_data must never modify its input in place."""
    original = kmer_counts.copy(deep=True)
    for method in ["raw", "relative", "log", "clr", "zscore"]:
        preprocess_data(kmer_counts, method)
        pd.testing.assert_frame_equal(kmer_counts, original)


def test_unknown_method_raises(kmer_counts):
    with pytest.raises(ValueError, match="Unknown normalization method"):
        preprocess_data(kmer_counts, "not_a_method")


def test_clr_accepts_float32_input(kmer_counts):
    """After the loader change the matrix arrives already as float32; CLR
    output must be identical either way."""
    as_float32 = kmer_counts.astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(preprocess_data(as_float32, "clr"), dtype=np.float64),
        np.asarray(preprocess_data(kmer_counts, "clr"), dtype=np.float64),
        rtol=RTOL, atol=ATOL,
    )


# ---------------------------------------------------------------------------
# PCA regression tests
# ---------------------------------------------------------------------------

def test_variance_threshold_pca_matches_reference(kmer_counts):
    df = preprocess_data(kmer_counts, "clr")
    expected, expected_pcs = reference_variance_pca(df, keep_variance=0.9)

    result = reduce_dimensions_with_pca(df, keep_variance=0.9)

    assert result.shape == (df.shape[0], expected_pcs)
    assert list(result.columns) == [f"PC{i+1}" for i in range(expected_pcs)]
    np.testing.assert_allclose(
        result.values, expected, rtol=1e-4, atol=1e-5,
        err_msg="variance-threshold PCA output diverged from reference",
    )


def test_fixed_pcs_pca(kmer_counts):
    df = preprocess_data(kmer_counts, "clr")
    result = reduce_dimensions_with_pca(df, keep_pcs=5)
    expected = PCA(n_components=5).fit_transform(df.values)

    assert result.shape == (df.shape[0], 5)
    assert list(result.index) == list(df.index)
    np.testing.assert_allclose(result.values, expected, rtol=1e-4, atol=1e-5)


def test_pca_requires_a_selection_argument(kmer_counts):
    df = preprocess_data(kmer_counts, "clr")
    with pytest.raises(ValueError):
        reduce_dimensions_with_pca(df)
