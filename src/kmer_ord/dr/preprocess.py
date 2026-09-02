# src/kmer_ord/dr/preprocess.py
import pandas as pd


def preprocess_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply normalization to k-mer matrix (DataFrame of numeric k-mer counts).
    Returns a float32 DataFrame (rows = samples, columns = features),
    preserving sample IDs. The input DataFrame is never modified.

    Memory design: exactly one output-sized float32 buffer is allocated and
    every transform operates on it in place, so peak RAM at this stage is the
    input matrix plus one copy — the previous implementation allocated
    additional full-matrix temporaries per operation (worst for CLR, which
    built log/geometric-mean/divide intermediates).
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # the single allocation: this buffer becomes the returned matrix
    X = df.to_numpy(dtype=np.float32, copy=True)

    if method == "raw":
        pass

    elif method == "relative":
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # all-zero rows stay zero instead of NaN
        X /= row_sums

    elif method == "log":
        np.log1p(X, out=X)

    elif method == "clr":
        # Standard log-difference formulation of CLR:
        #   log(x / gmean(x)) == log(x) - mean(log(x))
        # Mathematically identical to the explicit geometric-mean form but
        # needs no divide/exp temporaries (verified numerically equivalent
        # and ~50% leaner in my-notes/CLR-optimizations/).
        X += np.float32(1e-9)  # pseudocount to avoid log(0)
        np.log(X, out=X)
        X -= X.mean(axis=1, keepdims=True, dtype=np.float32)

    elif method == "zscore":
        # copy=False lets sklearn scale our own buffer in place
        X = StandardScaler(copy=False).fit_transform(X)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # wraps X without copying
    return pd.DataFrame(X, index=df.index, columns=df.columns)


def reduce_dimensions_with_pca(df: pd.DataFrame,
                               keep_pcs: int | None = None,
                               keep_variance: float | None = None,
                               method: str = "pca",
                               batch_size: int | None = None) -> pd.DataFrame:
    """
    Apply PCA reduction to a DataFrame either by fixed number of PCs
    or by cumulative variance threshold. Returns DataFrame with sample IDs
    as index.

    method="pca"  — exact sklearn PCA (whole matrix at once; most RAM).
    method="ipca" — sklearn IncrementalPCA fitted and transformed in row
                    batches; peak RAM is one batch plus the model instead of
                    full-matrix float64 workspaces. Results approximate exact
                    PCA (identical when the data fits in one batch).
    """
    import numpy as np

    if keep_pcs is None and keep_variance is None:
        raise ValueError("Either keep_pcs or keep_variance must be specified.")

    if method == "pca":
        X_pca = _standard_pca(df.values, keep_pcs, keep_variance)
    elif method == "ipca":
        X_pca = _incremental_pca(df.values, keep_pcs, keep_variance, batch_size)
    else:
        raise ValueError(f"Unknown PCA method: {method} (expected 'pca' or 'ipca')")

    columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, index=df.index, columns=columns)


def _standard_pca(X, keep_pcs, keep_variance):
    import numpy as np
    from sklearn.decomposition import PCA

    if keep_pcs is None:
        # fit() only: the old code used fit_transform() here, materializing
        # the full n x n_components transformed matrix just to read the
        # explained-variance spectrum
        pca_full = PCA()
        pca_full.fit(X)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        keep_pcs = int(np.searchsorted(cumulative_variance, keep_variance) + 1)

    return PCA(n_components=keep_pcs).fit_transform(X)


def _incremental_pca(X, keep_pcs, keep_variance, batch_size):
    import numpy as np
    from sklearn.decomposition import IncrementalPCA

    n_samples, n_features = X.shape
    max_pcs = min(n_samples, n_features)

    if keep_pcs is not None:
        n_fit = min(keep_pcs, max_pcs)
    else:
        # variance threshold needs the spectrum before choosing a count, so
        # fit a capped number of components (500 is far beyond any realistic
        # cumulative-variance cutoff for k-mer matrices)
        n_fit = min(500, max_pcs)

    if batch_size is None:
        batch_size = max(2048, 5 * n_fit)
    batch_size = max(batch_size, n_fit)  # sklearn requires batch >= components

    ipca = IncrementalPCA(n_components=n_fit, batch_size=batch_size)
    ipca.fit(X)

    if keep_pcs is None:
        cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
        keep_pcs = int(np.searchsorted(cumulative_variance, keep_variance) + 1)
    keep_pcs = min(keep_pcs, n_fit)

    # transform in batches so no full-matrix float64 workspace is created
    out = np.empty((n_samples, keep_pcs), dtype=np.float32)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        out[start:stop] = ipca.transform(X[start:stop])[:, :keep_pcs]
    return out
