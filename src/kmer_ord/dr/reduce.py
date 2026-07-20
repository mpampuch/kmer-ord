# src/kmer_ord/dr/reduce.py
"""
Memory-bounded dimensionality pre-reduction for wide k-mer matrices.

At large k the normalized matrix would be dense and enormous (e.g. ~12 GB at
k=11), and non-linear DR (UMAP/t-SNE) cannot consume a sparse matrix directly.
This module projects the matrix onto a small number of components *before* DR
without ever materialising the full dense matrix:

- ``raw`` / ``relative`` / ``log`` preserve zeros, so they are normalized while
  sparse and fed straight into :class:`~sklearn.decomposition.TruncatedSVD`.
- ``clr`` / ``zscore`` are dense by nature (they make every entry non-zero), so
  they are reduced with an exact PCA computed from the tiny ``n x n`` Gram matrix
  ``G = Xc @ Xc.T`` accumulated one *feature* block at a time. Because each
  feature column lives entirely within one block, per-block column-centering
  equals global centering, so the result is exact PCA - but peak RAM is only one
  feature block plus the ``n x n`` Gram, never the full dense matrix or a wide
  SVD workspace.
"""
import numpy as np

CLR_PSEUDOCOUNT = 1e-9

SPARSE_PRESERVING = ("raw", "relative", "log")
DENSE_BY_NATURE = ("clr", "zscore")

DEFAULT_N_COMPONENTS = 50
# Byte budget for a single densified feature block on the clr/zscore path. The
# block width is derived from this and the row count so peak RAM stays bounded
# even for very wide (large-k) matrices.
DEFAULT_MAX_BLOCK_BYTES = 256 * 1024 * 1024


def reduce_matrix(
    matrix,
    method: str,
    n_components: int = DEFAULT_N_COMPONENTS,
    keep_variance: float | None = None,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
    seed: int = 42,
) -> np.ndarray:
    """
    Normalize and reduce a sparse k-mer matrix to ``n_components`` dimensions,
    returning a small dense ``float32`` array of shape ``(n_samples, k)``.
    """
    matrix = matrix.tocsr()
    n_samples, n_features = matrix.shape

    if method in SPARSE_PRESERVING:
        # TruncatedSVD (randomized) needs n_components strictly below n_features.
        k = max(1, min(n_components, n_features - 1, n_samples - 1))
        reduced, explained = _truncated_svd(
            _sparse_row_normalize(matrix, method), k, seed
        )
    elif method in DENSE_BY_NATURE:
        k = max(1, min(n_components, n_samples, n_features))
        reduced, explained = _gram_pca_blocked(
            matrix, method, k, max_block_bytes
        )
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if keep_variance is not None:
        reduced = _trim_by_variance(reduced, explained, keep_variance)

    return reduced.astype(np.float32, copy=False)


def _sparse_row_normalize(matrix, method):
    """Row-independent normalization that keeps the matrix sparse."""
    X = matrix.astype(np.float32)

    if method == "raw":
        return X

    if method == "relative":
        from sklearn.preprocessing import normalize

        # L1 row normalization == divide each row by its (non-negative) sum;
        # all-zero rows are left as zeros (no divide-by-zero).
        return normalize(X, norm="l1", axis=1, copy=False)

    if method == "log":
        X = X.copy()
        X.data = np.log1p(X.data)  # log1p(0) == 0, so zeros stay implicit
        return X

    raise ValueError(f"Not a sparse-preserving method: {method}")


def _truncated_svd(sparse_matrix, n_components, seed):
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced = svd.fit_transform(sparse_matrix)
    return reduced, svd.explained_variance_ratio_


def _gram_pca_blocked(matrix, method, n_components, max_block_bytes):
    """
    Exact PCA of the ``clr``/``zscore``-normalized matrix via its ``n x n`` Gram
    matrix, accumulated one feature block at a time so the full dense matrix is
    never materialised. Returns ``(scores, explained_variance_ratio)``.
    """
    n_samples, n_features = matrix.shape
    csc = matrix.tocsc()

    # Per-row / per-column auxiliaries needed to reconstruct normalized values
    # from raw counts on a feature slice (all cheap, from the sparse matrix).
    row_log_mean = _clr_row_log_mean(matrix) if method == "clr" else None
    if method == "zscore":
        col_mean, col_std = _sparse_column_stats(matrix)
    else:
        col_mean = col_std = None

    block_cols = max(1, int(max_block_bytes // (n_samples * 8)))  # float64 block

    gram = np.zeros((n_samples, n_samples), dtype=np.float64)
    for start in range(0, n_features, block_cols):
        end = min(start + block_cols, n_features)
        block = _normalized_column_block(
            csc, start, end, method, row_log_mean, col_mean, col_std)
        # Each feature column lives entirely in this block, so subtracting the
        # block's per-column mean is exact global column-centering for PCA.
        block -= block.mean(axis=0, keepdims=True)
        gram += block @ block.T

    # Eigendecomposition of the symmetric Gram matrix gives PCA scores:
    # G = U S^2 U^T, scores = U S = eigvecs * sqrt(eigvals).
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    total_variance = eigvals[eigvals > 0].sum()

    top = order[:n_components]
    top_vals = np.clip(eigvals[top], 0.0, None)
    scores = eigvecs[:, top] * np.sqrt(top_vals)

    explained = (top_vals / total_variance if total_variance > 0
                 else np.zeros_like(top_vals))
    return scores, explained


def _clr_row_log_mean(matrix):
    """Per-row mean of ``log(count + eps)`` over *all* features, from the CSR.

    This is the log of the row geometric mean used by CLR; computing it from the
    sparse data (zeros contribute ``log(eps)``) avoids densifying anything.
    """
    n_samples, n_features = matrix.shape
    row_nnz = np.diff(matrix.indptr)
    row_index = np.repeat(np.arange(n_samples), row_nnz)

    log_nonzero = np.log(matrix.data.astype(np.float64) + CLR_PSEUDOCOUNT)
    sum_log_nonzero = np.bincount(row_index, weights=log_nonzero, minlength=n_samples)

    log_eps = np.log(CLR_PSEUDOCOUNT)
    sum_log = sum_log_nonzero + (n_features - row_nnz) * log_eps
    return sum_log / n_features


def _normalized_column_block(csc, start, end, method, row_log_mean, col_mean, col_std):
    """Densify a feature block and apply the (dense-by-nature) normalization."""
    block = csc[:, start:end].toarray().astype(np.float64)

    if method == "clr":
        # clr[i, f] = log(count + eps) - row_log_mean[i]
        block = np.log(block + CLR_PSEUDOCOUNT) - row_log_mean[:, None]
        return block

    if method == "zscore":
        block -= col_mean[start:end]
        block /= col_std[start:end]
        return block

    raise ValueError(f"Not a dense-by-nature method: {method}")


def _sparse_column_stats(matrix):
    """Per-column mean and std (population, ddof=0) computed from the CSR."""
    n = matrix.shape[0]
    m = matrix.astype(np.float64)
    col_sum = np.asarray(m.sum(axis=0)).ravel()
    col_sumsq = np.asarray(m.multiply(m).sum(axis=0)).ravel()

    mean = col_sum / n
    var = col_sumsq / n - mean ** 2
    var[var < 0] = 0.0  # guard tiny negative values from floating-point error
    std = np.sqrt(var)
    std[std == 0] = 1.0  # avoid divide-by-zero for constant columns
    return mean.astype(np.float32), std.astype(np.float32)


def _trim_by_variance(reduced, explained_variance_ratio, keep_variance):
    cumulative = np.cumsum(explained_variance_ratio)
    k = int(np.searchsorted(cumulative, keep_variance) + 1)
    k = min(max(k, 1), reduced.shape[1])
    return reduced[:, :k]
