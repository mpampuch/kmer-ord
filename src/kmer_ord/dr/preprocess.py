# src/kmer_ord/dr/preprocess.py
import pandas as pd

# Tuple of normalization methods accepted by preprocess_data.
# "--norm all" expands to this (not to the DR-method list in dr/methods.py).
NORMALISATION_METHODS = ("raw", "relative", "log", "clr", "zscore")


def preprocess_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply normalization to k-mer matrix (DataFrame of numeric k-mer counts).
    Returns a float32 DataFrame (rows = samples, columns = features),
    preserving sample IDs. 
    
    Memory notes: The input DataFrame is never modified.
    Exactly one output-sized float32 buffer is allocated and every transform 
    operates on it in place, so peak RAM at this stage is the input matrix plus 
    one copy. The previous implementation allocated additional full-matrix temporaries 
    per operation, which was the worst for the CLR transform because it allocated a lot
    of temporary arrays.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # the single allocation: this buffer becomes the returned matrix
    # This creates one NumPy array that will become the matrix used by the rest of the code
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
        # Mathematically identical to the explicit geometric-mean form
        # but needs no divide/exp temporary arrays. Roughly halves peak 
        # RAM allocation for this transform based on benchmarks.
        X += np.float32(1e-9)  # pseudocount to avoid log(0)
        np.log(X, out=X)
        X -= X.mean(axis=1, keepdims=True, dtype=np.float32)

    elif method == "zscore":
        # copy=False lets sklearn scale our own buffer in place
        X = StandardScaler(copy=False).fit_transform(X)

    else:
        raise ValueError(
            f"Unknown normalization method: {method} "
            f"(expected one of {', '.join(NORMALISATION_METHODS)})"
        )

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

    method="pca"  — exact sklearn PCA. Loads the whole matrix at once into RAM.
    method="ipca" — sklearn IncrementalPCA. Fits and transforms in row
                    batches. So the peak RAM is one batch plus the DR operations temporary arrays.
                    Results should approximate exact PCA / be identical when the data
                    fits in one batch.
    """
    import numpy as np

    if keep_pcs is None and keep_variance is None:
        # Edited typo here to be consistent with CLI flags (was underscores before)
        raise ValueError(
            "PCA pre-reduction requires either --keep-pcs or --keep-variance "
            "to be specified."
        )

    if method == "pca":
        X_pca = _standard_pca(df.values, keep_pcs, keep_variance)
    elif method == "ipca":
        X_pca = _incremental_pca(df.values, keep_pcs, keep_variance, batch_size)
    else:
        raise ValueError(f"Unknown PCA method: {method} (expected 'pca' or 'ipca')")

    columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, index=df.index, columns=columns)


def _standard_pca(X, keep_pcs, keep_variance):
    """
    Apply standard PCA to reduce X either to a fixed number of PCs
    or to the minimum number of PCs needed to retain the requested
    cumulative variance. When keep_variance is specified, fit PCA
    first to determine the required number of components needed to 
    retain the requested cumulative variance. And do this without
    materializing the full transformed matrix to save RAM. Then 
    transform X using only those components.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    # The case where keep_variance is specified:
    if keep_pcs is None:
        # use fit() only instead of fit_transform(): 
        # This avoids creating the entire transformed dataset when the code only needs PCA's explained-variance information.
        pca_full = PCA()
        pca_full.fit(X)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        keep_pcs = int(np.searchsorted(cumulative_variance, keep_variance) + 1)

    return PCA(n_components=keep_pcs).fit_transform(X)


def _incremental_pca(X, keep_pcs, keep_variance, batch_size):
    """
    Apply Incremental PCA to reduce X using either a fixed number of
    components or a cumulative-variance threshold. Fit only a capped
    number of components when selecting by variance, then transform X
    in batches to avoid materializing a large transformed matrix into
    RAM.
    """
    import numpy as np
    from sklearn.decomposition import IncrementalPCA

    n_samples, n_features = X.shape
    # Can't have more PCs than the number of samples or features.
    max_pcs = min(n_samples, n_features)

    if keep_pcs is not None:
        # Can't have more PCs than the number of samples or features, so use the smaller of the two.
        n_fit = min(keep_pcs, max_pcs)
    else:
        # variance threshold needs the spectrum before choosing a count, so
        # fit a capped number of components (500 chosen because it's far beyond 
        # any realistic cumulative-variance cutoff).
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

    # transform in batches so no full-matrix RAM allocation is created
    out = np.empty((n_samples, keep_pcs), dtype=np.float32)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        out[start:stop] = ipca.transform(X[start:stop])[:, :keep_pcs]
    return out
