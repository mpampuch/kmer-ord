"""Unit and numeric-parity tests for the sparse k-mer matrix pipeline."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from kmer_ord.io.sparse_matrix import (
    dense_npy_to_csr,
    save_sparse_matrix,
    load_sparse_matrix,
    write_matrix_tsv,
)
from kmer_ord.io.kmer_counter import canonical_kmers
from kmer_ord.io.kmer_stats import process_kmer_file
from kmer_ord.dr.preprocess import preprocess_data
from kmer_ord.dr.reduce import reduce_matrix


def _random_counts(n_rows, n_cols, density=0.2, seed=0):
    """A sparse-ish non-negative integer count matrix (dense ndarray)."""
    rng = np.random.default_rng(seed)
    dense = rng.integers(0, 50, size=(n_rows, n_cols))
    mask = rng.random((n_rows, n_cols)) < density
    dense = (dense * mask).astype(np.uint32)
    # Guarantee no all-zero row so row sums are positive (matches real reads).
    dense[np.where(dense.sum(axis=1) == 0)[0], 0] = 1
    return dense


def test_npz_roundtrip(tmp_path):
    dense = _random_counts(6, 10, seed=1)
    ids = [f"seq{i}" for i in range(dense.shape[0])]
    path = tmp_path / "m.npz"

    save_sparse_matrix(path, sparse.csr_matrix(dense), ids, kmer_length=7)
    matrix, loaded_ids, k = load_sparse_matrix(path)

    np.testing.assert_array_equal(matrix.toarray(), dense)
    assert list(loaded_ids) == ids
    assert k == 7


def test_dense_npy_to_csr_matches_and_blocks(tmp_path):
    dense = _random_counts(20, 8, seed=2)
    npy = tmp_path / "counts.npy"
    np.save(npy, dense)

    # Tiny block cap forces multiple row blocks through the streaming path.
    csr = dense_npy_to_csr(npy, max_block_bytes=8 * 4)  # ~1 row per block

    assert sparse.issparse(csr)
    np.testing.assert_array_equal(csr.toarray(), dense)


def test_write_matrix_tsv_matches_reference(tmp_path):
    k = 3
    kmer_keys = canonical_kmers(k)
    dense = _random_counts(5, len(kmer_keys), seed=3)
    ids = [f"read{i}" for i in range(dense.shape[0])]

    out = tmp_path / "matrix.tsv"
    write_matrix_tsv(out, sparse.csr_matrix(dense), ids, kmer_length=k)

    # Reference = the pipeline's original dense writer logic.
    expected_lines = ["sequence_id\t" + "\t".join(kmer_keys)]
    for i in range(dense.shape[0]):
        expected_lines.append(ids[i] + "\t" + "\t".join(map(str, dense[i])))
    expected = "\n".join(expected_lines) + "\n"

    assert out.read_text() == expected


def test_metrics_sparse_matches_table(tmp_path):
    dense = _random_counts(12, 15, seed=4)
    ids = [f"s{i}" for i in range(dense.shape[0])]

    npz = tmp_path / "m.npz"
    save_sparse_matrix(npz, sparse.csr_matrix(dense), ids, kmer_length=4)

    tsv = tmp_path / "m.tsv"
    cols = [f"c{j}" for j in range(dense.shape[1])]
    df = pd.DataFrame(dense, index=ids, columns=cols)
    df.index.name = "sequence_id"
    df.to_csv(tsv, sep="\t")

    from_npz = process_kmer_file(str(npz))
    from_tsv = process_kmer_file(str(tsv))

    assert list(from_npz.index) == list(from_tsv.index)
    for col in ["total_nonzero_kmers", "num_unique_kmers",
                "shannon_evenness", "shannon_diversity"]:
        np.testing.assert_allclose(
            from_npz[col].to_numpy(dtype=float),
            from_tsv[col].to_numpy(dtype=float),
            rtol=1e-6, atol=1e-8,
            err_msg=f"metric mismatch in {col}")


def _reference_preprocess(dense, method):
    """The original pandas normalization math, for parity checking."""
    X = dense.astype(np.float32)
    if method == "raw":
        return X
    if method == "relative":
        row_sums = X.sum(axis=1)
        row_sums[row_sums == 0] = 1
        return X / row_sums[:, None]
    if method == "log":
        return np.log1p(X)
    if method == "clr":
        X = X + 1e-9
        geo_mean = np.exp(np.mean(np.log(X), axis=1))
        return np.log(X / geo_mean[:, None])
    if method == "zscore":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(X)
    raise ValueError(method)


@pytest.mark.parametrize("method", ["raw", "relative", "log", "clr", "zscore"])
def test_preprocess_parity(method):
    dense = _random_counts(30, 12, seed=5)
    csr = sparse.csr_matrix(dense)

    result = preprocess_data(csr, method)
    reference = _reference_preprocess(dense, method)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("method", ["raw", "relative", "log"])
def test_reduce_sparse_norm_shape(method):
    dense = _random_counts(40, 60, seed=6)
    csr = sparse.csr_matrix(dense)

    reduced = reduce_matrix(csr, method, n_components=5, seed=0)

    assert reduced.shape == (40, 5)
    assert reduced.dtype == np.float32
    assert np.isfinite(reduced).all()


@pytest.mark.parametrize("method", ["clr", "zscore"])
@pytest.mark.parametrize("max_block_bytes", [1 << 30, 64])  # one block, then many
def test_reduce_gram_matches_full_pca(method, max_block_bytes):
    """The Gram-based reducer is exact PCA regardless of feature-block size, so
    it must match full-dense PCA even when forced into many tiny blocks. Compare
    via the Gram matrix, which is invariant to per-component sign/rotation."""
    from sklearn.decomposition import PCA

    dense = _random_counts(40, 60, seed=8)
    csr = sparse.csr_matrix(dense)
    n_components = 4

    reduced = reduce_matrix(csr, method, n_components=n_components,
                            max_block_bytes=max_block_bytes)
    ref = PCA(n_components=n_components).fit_transform(
        _reference_preprocess(dense, method))

    assert reduced.shape == (40, n_components)
    assert reduced.dtype == np.float32
    # Scores are stored as float32, so compare the Gram matrices at float32
    # precision (the reducer is exact PCA up to this rounding).
    g_reduced = reduced.astype(np.float64) @ reduced.astype(np.float64).T
    g_ref = ref @ ref.T
    np.testing.assert_allclose(g_reduced, g_ref, rtol=2e-3, atol=0.1)


@pytest.mark.parametrize("method", ["clr", "zscore"])
def test_reduce_many_blocks_finite(method):
    """Forcing many feature blocks still yields a finite embedding."""
    dense = _random_counts(50, 200, seed=7)
    reduced = reduce_matrix(sparse.csr_matrix(dense), method,
                            n_components=4, max_block_bytes=64)
    assert reduced.shape == (50, 4)
    assert np.isfinite(reduced).all()
