import os
import math
import pandas as pd
from kmer_ord.utils.logging_utils import section, info, warn

def build_dtypes(input_file: str) -> dict:
    """Infer dtypes for chunked reading of k-mer frequency matrix."""
    with open(input_file, "r") as f:
        column_names = f.readline().strip().split("\t")
        num_columns = len(column_names)

    dtypes = {0: "str"}  # index column
    for col in range(1, num_columns - 1):
        dtypes[col] = "uint16"
    return dtypes

def calculate_kmer_metrics_chunk(kmer_df: pd.DataFrame) -> pd.DataFrame:
    """Compute k-mer metrics for a chunk of sequences."""
    import numpy as np
    numeric = kmer_df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("No numeric k-mer columns found.")

    numeric = numeric.dropna(axis=1, how="all").fillna(0)
    values = numeric.to_numpy(dtype=float)

    nonzero_mask = values != 0
    total_nonzero = nonzero_mask.sum(axis=1)
    row_sums = values.sum(axis=1)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    probs = values / row_sums_safe[:, None]
    positive = probs > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        shannon_nats = -np.sum(np.where(positive, probs * np.log(probs), 0.0), axis=1)
        shannon_bits = -np.sum(np.where(positive, probs * np.log2(probs), 0.0), axis=1)

    metrics_chunk = pd.DataFrame(
        {"total_nonzero_kmers": row_sums.astype("int64"),
         "num_unique_kmers": total_nonzero,
         "shannon_evenness": shannon_nats,
         "shannon_diversity": shannon_bits},
        index=kmer_df.index)
    return metrics_chunk

def calculate_kmer_metrics_sparse(matrix, sequence_ids) -> pd.DataFrame:
    """
    Compute the same per-sequence metrics as :func:`calculate_kmer_metrics_chunk`
    directly from a scipy CSR matrix, without densifying, upcasting to float64
    over the full matrix, or paying pandas' millions-of-columns overhead.
    """
    import numpy as np

    matrix = matrix.tocsr()
    n_rows = matrix.shape[0]

    counts = matrix.data.astype(np.float64)
    row_nnz = np.diff(matrix.indptr)                       # num_unique_kmers
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()      # total count per row

    # Map every stored (non-zero) value back to its row so we can reduce
    # per-row with bincount (robust to empty rows and trailing empty rows).
    row_index = np.repeat(np.arange(n_rows), row_nnz)
    row_sums_per_nnz = np.repeat(row_sums, row_nnz)

    if counts.size:
        probs = counts / row_sums_per_nnz
        shannon_nats = -np.bincount(
            row_index, weights=probs * np.log(probs), minlength=n_rows)
        shannon_bits = -np.bincount(
            row_index, weights=probs * np.log2(probs), minlength=n_rows)
    else:
        shannon_nats = np.zeros(n_rows)
        shannon_bits = np.zeros(n_rows)

    return pd.DataFrame(
        {"total_nonzero_kmers": row_sums.astype("int64"),
         "num_unique_kmers": row_nnz.astype("int64"),
         "shannon_evenness": shannon_nats,
         "shannon_diversity": shannon_bits},
        # Name the index "sequence_id" so the written TSV matches the dense path
        # and downstream FeatureMerge can find the id column.
        index=pd.Index(sequence_ids, name="sequence_id"))


def _metrics_from_table(input_file: str, chunksize: int, cpus: int) -> pd.DataFrame:
    """Legacy dense TSV/CSV path (chunked pandas)."""
    from concurrent.futures import ProcessPoolExecutor

    dtypes = build_dtypes(input_file)
    reader = pd.read_csv(input_file, sep="\t", index_col=0, dtype=dtypes, chunksize=chunksize)

    chunks = list(reader)

    if cpus <= 1:
        results = [calculate_kmer_metrics_chunk(chunk) for chunk in chunks]
    else:
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [executor.submit(calculate_kmer_metrics_chunk, chunk) for chunk in chunks]
            results = [f.result() for f in futures]  # collect in submission order

    return pd.concat(results)


def process_kmer_file(
    input_file: str,
    output_file: str = None,
    chunksize: int = 25000,
    cpus: int = 1,
    total_rows: int = None,) -> pd.DataFrame:
    """Process a k-mer matrix to compute metrics, optionally parallelized."""
    from pathlib import Path
    section("Calculating k-mer metrics")
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if os.path.exists(output_file):
            os.remove(output_file)

    if Path(input_file).suffix.lower() == ".npz":
        from kmer_ord.io.sparse_matrix import load_sparse_matrix

        matrix, sequence_ids, _ = load_sparse_matrix(input_file)
        combined_metrics = calculate_kmer_metrics_sparse(matrix, sequence_ids)
    else:
        combined_metrics = _metrics_from_table(input_file, chunksize, cpus)

    if output_file:
        combined_metrics.to_csv(output_file, sep="\t")

    # Dataset-wide summary
    shannon = combined_metrics["shannon_diversity"].to_numpy()
    unique_kmers = combined_metrics["num_unique_kmers"].to_numpy()
    w = 20
    info(f"{'shannon diversity':<{w}}  {'mean':<4} {shannon.mean():8.3f}  {'sd':<3} {shannon.std(ddof=1):8.3f}")
    info(f"{'shannon range':<{w}}  {'min':<4} {shannon.min():8.3f}  {'max':<3} {shannon.max():8.3f}")
    info(f"{'unique kmers':<{w}}  {'mean':<4} {unique_kmers.mean():8.1f}  {'sd':<3} {unique_kmers.std(ddof=1):8.1f}")
    info(f"{'unique kmers range':<{w}}  {'min':<4} {unique_kmers.min():8.0f}  {'max':<3} {unique_kmers.max():8.0f}")

    return combined_metrics
