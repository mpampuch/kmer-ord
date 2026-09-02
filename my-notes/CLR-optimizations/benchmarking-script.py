import io
import time
import tracemalloc

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

# N_RUNS = 10_000
N_RUNS = 100
EPS = 1e-9


# ============================================================
# Dataset
# ============================================================

# data_str = """sequence_id AAAAAA AAAAAC AAAAAG AAAAAT AAAACA AAAACC AAAACG AAAACT AAAAGA AAAAGC
# m84180_260101_071145_s1/243799085/ccs 5 12 5 4 19 11 5 6 8 14
# m84180_260101_071145_s1/245962504/ccs 5 7 4 0 13 10 4 3 1 5
# m84180_260101_071145_s1/251268719/ccs 6 10 6 2 7 6 3 4 3 9
# m84180_260101_071145_s1/262672757/ccs 3 4 0 1 11 6 3 1 2 5
# m84180_260101_071145_s1/244586185/ccs 13 8 9 4 10 2 4 3 6 13
# m84180_260101_071145_s1/262411371/ccs 6 6 8 6 10 3 3 4 3 12
# m84180_260101_071145_s1/245698699/ccs 5 4 5 2 4 9 1 1 4 8
# m84180_260101_071145_s1/245568929/ccs 3 10 7 6 19 8 12 4 7 13
# m84180_260101_071145_s1/262345483/ccs 0 1 1 1 0 2 3 4 3 3
# """

df = pd.read_csv(
    "/Users/markpampuch/Dropbox/KAUST/PhD/20260823_kmer-ord-local/kmer-ord/my-notes/CLR-optimizations/62_Coelastrummicroporum.hifi_reads_6mer_matrix.tsv",
    sep="\t",
    index_col=0,
).astype(np.float32)


# ============================================================
# CLR Implementations
# ============================================================

def clr_original(
    df_in: pd.DataFrame,
    eps: float = EPS,
) -> pd.DataFrame:
    """
    Original Pandas CLR implementation.

    Uses:
        geometric_mean = exp(mean(log(X)))
        CLR = log(X / geometric_mean)

    Input and calculations remain float32.
    """

    X = df_in.copy()

    X += np.float32(eps)

    geometric_mean = np.exp(
        np.mean(np.log(X), axis=1)
    )

    X = np.log(
        X.div(geometric_mean, axis=0)
    )
    
    return X


def clr_log_diff(
    df_in: pd.DataFrame,
    eps: float = EPS,
) -> pd.DataFrame:
    """
    Memory-efficient CLR implementation.

    CLR = log(X + eps) - mean(log(X + eps))

    Calculations are performed on a NumPy float32 array.
    """
    X = df_in.to_numpy(copy=True)

    X += np.float32(eps)
    np.log(X, out=X)

    X -= X.mean(axis=1, dtype=np.float32)[:, None]

    return pd.DataFrame(
        X,
        index=df_in.index,
        columns=df_in.columns,
    )

# ============================================================
# Numerical Verification
# ============================================================

def verify_outputs() -> None:
    """
    Verify that both CLR implementations produce
    numerically equivalent results.
    """

    res_orig = clr_original(df)
    res_diff = clr_log_diff(df)

    np.testing.assert_allclose(
        res_orig.values,
        res_diff.values,
        rtol=1e-5,
        atol=1e-6,
    )

    max_abs_diff = np.max(
        np.abs(
            res_orig.values.astype(np.float64)
            - res_diff.values.astype(np.float64)
        )
    )

    print("=" * 70)
    print("NUMERICAL VERIFICATION")
    print("=" * 70)
    print("✓ Original vs. Log-Diff CLR (Pandas): numerically equivalent")
    print(f"  Maximum absolute difference: {max_abs_diff:.3e}")
    print(f"  Data type: {res_orig.dtypes.iloc[0]}")


# ============================================================
# Execution-Time Benchmark
# ============================================================

def benchmark_time(
    func,
    *args,
    n_runs: int = N_RUNS,
) -> float:
    """
    Benchmark a function.

    Returns:
        Average execution time in microseconds per operation.
    """

    # Warm-up
    func(*args)

    start = time.perf_counter()

    for _ in range(n_runs):
        func(*args)

    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1e6


# ============================================================
# Memory Benchmark
# ============================================================

def measure_memory_peak(
    func,
    *args,
) -> int:
    """
    Measure peak Python-level memory allocation using tracemalloc.

    Returns:
        Peak allocated memory in bytes.
    """

    tracemalloc.start()

    try:
        tracemalloc.reset_peak()

        _ = func(*args)

        _, peak_bytes = tracemalloc.get_traced_memory()

    finally:
        tracemalloc.stop()

    return peak_bytes


# ============================================================
# Formatting Helpers
# ============================================================

def format_memory(bytes_used: int) -> str:
    """Format bytes as KiB."""

    return f"{bytes_used / 1024:.2f} KiB"


# ============================================================
# Main Benchmark
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CLR BENCHMARK")
    print("=" * 70)

    print(
        f"Dataset: {df.shape[0]} sequences × "
        f"{df.shape[1]} features"
    )

    print(f"Runs:    {N_RUNS:,}")
    print(f"Epsilon: {EPS:g}")
    print(f"dtype:   {df.dtypes.iloc[0]}")

    # --------------------------------------------------------
    # Numerical verification
    # --------------------------------------------------------

    verify_outputs()

    # --------------------------------------------------------
    # Execution time
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXECUTION TIME")
    print("=" * 70)

    t_orig = benchmark_time(
        clr_original,
        df,
    )

    t_diff = benchmark_time(
        clr_log_diff,
        df,
    )

    speedup_diff = t_orig / t_diff

    print(
        f"Original CLR (Pandas):        "
        f"{t_orig:10.2f} µs/op"
    )

    print(
        f"Log-Diff CLR (Pandas):        "
        f"{t_diff:10.2f} µs/op "
        f"({speedup_diff:.2f}x faster)"
    )

    # --------------------------------------------------------
    # Peak memory
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PEAK MEMORY")
    print("=" * 70)

    peak_orig = measure_memory_peak(
        clr_original,
        df,
    )

    peak_diff = measure_memory_peak(
        clr_log_diff,
        df,
    )

    print(
        f"Original CLR (Pandas):        "
        f"{format_memory(peak_orig):>12} "
        f"({peak_orig:,} bytes)"
    )

    print(
        f"Log-Diff CLR (Pandas):        "
        f"{format_memory(peak_diff):>12} "
        f"({peak_diff:,} bytes)"
    )

    # --------------------------------------------------------
    # Memory comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MEMORY COMPARISON")
    print("=" * 70)

    memory_ratio = peak_orig / peak_diff

    print(
        f"Log-Diff Pandas vs Original: "
        f"{memory_ratio:.2f}x lower peak allocation"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    fastest_name, fastest_time = min(
        [
            ("Original CLR (Pandas)", t_orig),
            ("Log-Diff CLR (Pandas)", t_diff),
        ],
        key=lambda x: x[1],
    )

    lowest_memory_name, lowest_memory = min(
        [
            ("Original CLR (Pandas)", peak_orig),
            ("Log-Diff CLR (Pandas)", peak_diff),
        ],
        key=lambda x: x[1],
    )

    print(
        f"Fastest implementation:       "
        f"{fastest_name} ({fastest_time:.2f} µs/op)"
    )

    print(
        f"Lowest peak allocation:        "
        f"{lowest_memory_name} "
        f"({format_memory(lowest_memory)})"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()