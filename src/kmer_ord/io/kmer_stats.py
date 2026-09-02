# src/kmer_ord/io/kmer_stats.py
"""Per-read k-mer metrics, computed as a stream over the matrix TSV.

Memory design: the matrix is read chunk-by-chunk and each chunk's metrics are
appended to the output file immediately, so peak RAM scales with one chunk
(times the number of in-flight workers when parallel), never with the whole
matrix. Dataset-wide summary statistics are maintained with running
accumulators instead of retaining all per-read metrics.
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from kmer_ord.utils.logging_utils import section, info

# Output schema. Names follow the memory-audit corrections: the historical
# names were misleading (`total_nonzero_kmers` was really the total count sum,
# `shannon_evenness` was raw entropy in nats, `shannon_diversity` was entropy
# in bits). `shannon_evenness` is now true Pielou evenness (H / ln S).
METRIC_COLUMNS = [
    "total_kmer_counts",
    "num_nonzero_kmers",
    "shannon_entropy_nats",
    "shannon_entropy_bits",
    "shannon_evenness",
]


def build_dtypes(input_file: str) -> dict:
    """Positional dtypes for chunked reading of the k-mer frequency matrix.

    Counts are read as uint32 to match the Rust counter's output range — the
    old uint16 downcast could silently overflow for small k on long reads.
    """
    with open(input_file, "r") as f:
        num_columns = len(f.readline().strip().split("\t"))

    dtypes = {0: "str"}  # index column (sequence IDs)
    for col in range(1, num_columns):
        dtypes[col] = "uint32"
    return dtypes


class RunningStats:
    """Streaming mean/std/min/max over batches (Chan et al. parallel Welford).

    Lets the dataset-wide summary be computed without keeping every per-read
    metric in memory.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0  # sum of squared deviations from the running mean
        self.min = math.inf
        self.max = -math.inf

    def update(self, batch: np.ndarray):
        batch = np.asarray(batch, dtype=np.float64)
        n_b = batch.size
        if n_b == 0:
            return
        mean_b = batch.mean()
        m2_b = np.square(batch - mean_b).sum()

        delta = mean_b - self.mean
        n_total = self.n + n_b
        self.mean += delta * n_b / n_total
        self._m2 += m2_b + delta * delta * self.n * n_b / n_total
        self.n = n_total
        self.min = min(self.min, batch.min())
        self.max = max(self.max, batch.max())

    def std(self, ddof: int = 1) -> float:
        if self.n <= ddof:
            return math.nan
        return math.sqrt(self._m2 / (self.n - ddof))


def calculate_kmer_metrics_chunk(kmer_df: pd.DataFrame) -> pd.DataFrame:
    """Compute k-mer metrics for one chunk of sequences.

    Stats are computed in float32 (the output keeps ~3 decimals; float64
    would double the working memory) with float64 accumulators for the row
    reductions, and buffers are reused in place to minimize chunk-sized
    temporaries.
    """
    numeric = kmer_df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("No numeric k-mer columns found.")

    values = numeric.to_numpy(dtype=np.float32)

    nonzero_mask = values != 0
    num_nonzero = nonzero_mask.sum(axis=1)
    # float64 accumulator keeps integer row sums exact (< 2^53)
    row_sums = values.sum(axis=1, dtype=np.float64)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)

    # convert counts to probabilities in place: `values` is our own buffer
    values /= row_sums_safe[:, None].astype(np.float32)

    # p * ln(p) with zeros untouched (where= skips them), reusing one buffer
    plogp = np.zeros_like(values)
    np.log(values, out=plogp, where=nonzero_mask)
    plogp *= values

    shannon_nats = -plogp.sum(axis=1, dtype=np.float64)
    shannon_bits = shannon_nats / math.log(2)
    # Pielou evenness H / ln(S); defined as 1.0 when only one k-mer type is
    # present (maximum evenness of a single category, avoids ln(1)=0 division)
    shannon_evenness = np.where(
        num_nonzero > 1,
        shannon_nats / np.log(np.maximum(num_nonzero, 2)),
        1.0,
    )

    return pd.DataFrame(
        {
            "total_kmer_counts": row_sums.astype("int64"),
            "num_nonzero_kmers": num_nonzero,
            "shannon_entropy_nats": shannon_nats,
            "shannon_entropy_bits": shannon_bits,
            "shannon_evenness": shannon_evenness,
        },
        index=kmer_df.index,
    )


def process_kmer_file(
    input_file: str,
    output_file: str,
    chunksize: int = 25000,
    cpus: int = 1,
) -> Path:
    """Stream a k-mer matrix, writing per-read metrics to `output_file`.

    Returns the output path. The full metrics table is never held in memory;
    read it back from disk if needed.
    """
    section("Calculating k-mer metrics")
    if not output_file:
        raise ValueError("output_file is required (metrics are streamed to disk).")

    output_file = str(output_file)
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    dtypes = build_dtypes(input_file)
    reader = pd.read_csv(
        input_file, sep="\t", index_col=0, dtype=dtypes, chunksize=chunksize
    )

    entropy_stats = RunningStats()
    nonzero_stats = RunningStats()
    first_chunk = True

    def write_metrics(metrics: pd.DataFrame):
        nonlocal first_chunk
        metrics.to_csv(
            output_file, sep="\t", mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        entropy_stats.update(metrics["shannon_entropy_bits"].to_numpy())
        nonzero_stats.update(metrics["num_nonzero_kmers"].to_numpy())

    if cpus <= 1:
        for chunk in reader:
            write_metrics(calculate_kmer_metrics_chunk(chunk))
    else:
        # Bounded submission: at most `cpus` chunks are in flight, so parallel
        # mode costs ~cpus chunk-copies of RAM instead of the whole matrix
        # (the old code submitted every chunk up front).
        from collections import deque
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=cpus) as executor:
            in_flight = deque()
            for chunk in reader:
                in_flight.append(
                    executor.submit(calculate_kmer_metrics_chunk, chunk)
                )
                if len(in_flight) >= cpus:
                    write_metrics(in_flight.popleft().result())
            while in_flight:
                write_metrics(in_flight.popleft().result())

    # Dataset-wide summary from the running accumulators
    w = 20
    info(f"{'shannon entropy':<{w}}  {'mean':<4} {entropy_stats.mean:8.3f}  {'sd':<3} {entropy_stats.std(ddof=1):8.3f}")
    info(f"{'shannon range':<{w}}  {'min':<4} {entropy_stats.min:8.3f}  {'max':<3} {entropy_stats.max:8.3f}")
    info(f"{'nonzero kmers':<{w}}  {'mean':<4} {nonzero_stats.mean:8.1f}  {'sd':<3} {nonzero_stats.std(ddof=1):8.1f}")
    info(f"{'nonzero kmers range':<{w}}  {'min':<4} {nonzero_stats.min:8.0f}  {'max':<3} {nonzero_stats.max:8.0f}")

    return Path(output_file)
