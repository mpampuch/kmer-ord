# tests/test_pca_pre.py
"""Tests for the IncrementalPCA option of reduce_dimensions_with_pca.

IncrementalPCA on a single batch is mathematically the same SVD as standard
PCA, so the single-batch case must match PCA outputs; the multi-batch case is
an approximation and is tested for shape, determinism, and variance capture.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from kmer_ord.dr.preprocess import preprocess_data, reduce_dimensions_with_pca


@pytest.fixture
def clr_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    counts = rng.poisson(lam=6.0, size=(80, 20)).astype(np.int64)
    df = pd.DataFrame(
        counts,
        index=[f"read_{i:03d}" for i in range(80)],
        columns=[f"kmer_{j:02d}" for j in range(20)],
    )
    return preprocess_data(df, "clr")


def test_ipca_single_batch_matches_pca(clr_matrix):
    expected = PCA(n_components=5).fit_transform(clr_matrix.values)
    result = reduce_dimensions_with_pca(
        clr_matrix, keep_pcs=5, method="ipca", batch_size=1000
    )
    assert result.shape == (80, 5)
    assert list(result.columns) == [f"PC{i+1}" for i in range(5)]
    assert list(result.index) == list(clr_matrix.index)
    np.testing.assert_allclose(result.values, expected, rtol=1e-3, atol=1e-4)


def test_ipca_multi_batch_deterministic_and_useful(clr_matrix):
    a = reduce_dimensions_with_pca(clr_matrix, keep_pcs=5, method="ipca", batch_size=16)
    b = reduce_dimensions_with_pca(clr_matrix, keep_pcs=5, method="ipca", batch_size=16)
    assert a.shape == (80, 5)
    pd.testing.assert_frame_equal(a, b)  # no hidden randomness

    # the estimated subspace must capture variance comparably to exact PCA
    pca_var = PCA(n_components=5).fit(clr_matrix.values).explained_variance_.sum()
    ipca_var = a.values.var(axis=0, ddof=1).sum()
    assert ipca_var >= 0.9 * pca_var


def test_ipca_variance_threshold(clr_matrix):
    result = reduce_dimensions_with_pca(
        clr_matrix, keep_variance=0.9, method="ipca", batch_size=1000
    )
    # single batch == exact PCA, so the component count must match the
    # standard-PCA selection for the same threshold
    full = PCA().fit(clr_matrix.values)
    expected_pcs = int(
        np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.9) + 1
    )
    assert result.shape == (80, expected_pcs)


def test_ipca_output_is_float32(clr_matrix):
    result = reduce_dimensions_with_pca(
        clr_matrix, keep_pcs=3, method="ipca", batch_size=32
    )
    assert all(dt == np.float32 for dt in result.dtypes)


def test_unknown_pca_method_raises(clr_matrix):
    with pytest.raises(ValueError, match="Unknown PCA method"):
        reduce_dimensions_with_pca(clr_matrix, keep_pcs=3, method="nope")
