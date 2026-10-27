# src/kmer_ord/dr/preprocess.py
from dataclasses import dataclass

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


def _ipca_fit_plan(
    n_samples: int,
    n_features: int,
    keep_pcs: int | None,
    batch_size: int | None,
) -> tuple[int, int]:
    """Components to fit, and the batch size sklearn will accept.

    sklearn requires batch_size >= n_components. A missing batch size becomes
    max(2048, 5 * components). Variance selection fits at most 500 components
    before the threshold is applied; that cap does not depend on the threshold.
    """
    max_pcs = min(n_samples, n_features)
    if keep_pcs is not None:
        n_fit = min(keep_pcs, max_pcs)
    else:
        n_fit = min(500, max_pcs)
    if batch_size is None:
        batch_size = max(2048, 5 * n_fit)
    return n_fit, max(batch_size, n_fit)


def resolve_ipca_batch_size(
    n_samples: int,
    n_features: int,
    keep_pcs: int | None,
    keep_variance: float | None,
    batch_size: int | None,
) -> int:
    """Batch size IncrementalPCA will actually use.

    The variance threshold does not change the batch: variance selection
    always fits at most 500 components first. See _ipca_fit_plan.
    """
    del keep_variance
    return _ipca_fit_plan(n_samples, n_features, keep_pcs, batch_size)[1]


@dataclass(frozen=True)
class IpcaBatchReport:
    """One pass (fit or transform) over the matrix in row batches."""

    slices: tuple[slice, ...]
    batch_size: int
    # "full", "short", or "merged". Merged means sklearn folded a remainder
    # smaller than n_components into the previous batch.
    handling: str
    n_components: int
    remainder: int | None = None

    @property
    def n_batches(self) -> int:
        return len(self.slices)

    @property
    def last_rows(self) -> int:
        last = self.slices[-1]
        return last.stop - last.start


def _row_batch_slices(n_samples: int, batch_size: int) -> tuple[slice, ...]:
    """Plain tail slices. A short last batch stays short.

    This is the transform loop. It is not sklearn's fit rule: transform does
    not require each batch to hold at least n_components.
    """
    return tuple(
        slice(start, min(start + batch_size, n_samples))
        for start in range(0, n_samples, batch_size)
    )


def _classify_batches(
    slices: tuple[slice, ...],
    batch_size: int,
    n_components: int,
) -> IpcaBatchReport:
    if not slices:
        raise ValueError("IPCA batch report needs at least one row")
    last_rows = slices[-1].stop - slices[-1].start
    remainder = None
    if last_rows == batch_size:
        handling = "full"
    elif last_rows < batch_size:
        handling = "short"
    else:
        # gen_batches only absorbs one short tail onto the previous batch,
        # so the extra rows are the remainder that was too small to fit alone.
        handling = "merged"
        remainder = last_rows - batch_size
    return IpcaBatchReport(
        slices=slices,
        batch_size=batch_size,
        handling=handling,
        n_components=n_components,
        remainder=remainder,
    )


def describe_ipca_batches(
    n_samples: int,
    n_features: int,
    keep_pcs: int | None,
    keep_variance: float | None,
    batch_size: int | None,
) -> tuple[IpcaBatchReport, IpcaBatchReport]:
    """Fit and transform batch plans for the batch size IncrementalPCA will use.

    Fit slices come from sklearn.utils.gen_batches with min_batch_size set to
    the component count being fit. When the remainder is smaller than that
    count, sklearn folds it into the previous batch: the last fit batch is
    then larger than batch_size, and the fit count is one lower than the
    transform count. The variance threshold does not change this plan.
    """
    from sklearn.utils import gen_batches

    del keep_variance
    n_fit, batch_size = _ipca_fit_plan(
        n_samples, n_features, keep_pcs, batch_size
    )
    fit_slices = tuple(
        gen_batches(n_samples, batch_size, min_batch_size=n_fit)
    )
    return (
        _classify_batches(fit_slices, batch_size, n_fit),
        _classify_batches(_row_batch_slices(n_samples, batch_size), batch_size, n_fit),
    )


def format_ipca_batch_line(kind: str, report: IpcaBatchReport) -> str:
    """One console/log line: batch count, last-batch rows, and how it was handled."""
    noun = "batch" if report.n_batches == 1 else "batches"
    if report.handling == "full":
        detail = "full"
    elif report.handling == "short":
        detail = "short remainder"
    elif report.handling == "merged":
        detail = (
            f"merged (remainder {report.remainder:,} is below "
            f"{report.n_components:,} components)"
        )
    else:
        raise ValueError(f"Unknown IPCA batch handling: {report.handling}")
    return (
        f"{'ipca ' + kind:<16}  {report.n_batches:,} {noun}; "
        f"last {report.last_rows:,} rows, {detail}"
    )


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
    n_fit, batch_size = _ipca_fit_plan(
        n_samples, n_features, keep_pcs, batch_size
    )

    ipca = IncrementalPCA(n_components=n_fit, batch_size=batch_size)
    ipca.fit(X)

    if keep_pcs is None:
        cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
        keep_pcs = int(np.searchsorted(cumulative_variance, keep_variance) + 1)
    keep_pcs = min(keep_pcs, n_fit)

    # transform in batches so no full-matrix RAM allocation is created.
    # These slices stay a plain tail; they do not use the fit-time merge.
    out = np.empty((n_samples, keep_pcs), dtype=np.float32)
    for batch in _row_batch_slices(n_samples, batch_size):
        out[batch] = ipca.transform(X[batch])[:, :keep_pcs]
    return out
