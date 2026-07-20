# src/kmer_ord/dr/preprocess.py
"""
Normalization of k-mer count matrices.

Accepts a scipy sparse matrix, numpy array, or pandas DataFrame and always
returns a ``float32`` numpy array (row = sample, column = feature). The
row-independent transforms (``raw``/``relative``/``log``/``clr``) live in
:func:`apply_row_normalization` so the dense small-k path here and the blocked
large-k path in :mod:`kmer_ord.dr.reduce` compute identical values.
"""
import numpy as np

# All supported normalization methods (used to expand the "all" shortcut).
ALL_NORMALISATIONS = ("raw", "relative", "log", "clr", "zscore")


def _to_dense_float32(matrix) -> np.ndarray:
    """Materialise a sparse/array/DataFrame input as a dense float32 array."""
    import scipy.sparse as sp

    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)

    # pandas DataFrame (has .to_numpy) or plain ndarray / array-like.
    to_numpy = getattr(matrix, "to_numpy", None)
    if callable(to_numpy):
        return to_numpy(dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def apply_row_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """
    Apply a row-independent normalization to a dense float32 array in place
    where possible. Because every transform here depends only on the values
    within a row, it is safe to apply to a block of rows in isolation, which is
    what the blocked IncrementalPCA path relies on for CLR.
    """
    if method == "raw":
        return X

    if method == "relative":
        row_sums = X.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        X /= row_sums[:, None]
        return X

    if method == "log":
        return np.log1p(X)

    if method == "clr":
        # CLR = log(x / geometric_mean(x)) = log(x) - mean(log(x)).
        # The pseudocount avoids log(0); computing in log-space avoids the
        # extra exp()/division copies the original implementation allocated.
        X += 1e-9
        log_X = np.log(X)
        log_X -= log_X.mean(axis=1, keepdims=True)
        return log_X

    raise ValueError(f"Unknown normalization method: {method}")


def preprocess_data(matrix, method: str) -> np.ndarray:
    """
    Normalize a k-mer count matrix and return a dense ``float32`` array.

    Used for the small-k (below-threshold) path; large-k inputs are reduced
    without full densification in :mod:`kmer_ord.dr.reduce`.
    """
    if method == "zscore":
        # Column standardization needs global (per-feature) statistics, so it is
        # handled separately from the row-independent transforms above.
        from sklearn.preprocessing import StandardScaler

        X = _to_dense_float32(matrix)
        return StandardScaler().fit_transform(X).astype(np.float32, copy=False)

    X = _to_dense_float32(matrix)
    return apply_row_normalization(X, method)
