#!/usr/bin/env python
# benchmarks/run_benchmarks.py
"""Repeatable per-stage RAM/time benchmarks for the kmer-ord pipeline.

Usage:
    # fast per-change check on a seeded synthetic matrix
    python benchmarks/run_benchmarks.py run --tier small

    # milestone validation on the real 3.9 GB 6-mer matrix
    python benchmarks/run_benchmarks.py run --tier full

    # compare the two most recent commits present in the log
    python benchmarks/run_benchmarks.py compare
    python benchmarks/run_benchmarks.py compare --commits abc1234 def5678

Each stage runs in its own subprocess so peak RSS reflects that stage alone
(a shared process would carry allocator high-water marks between stages).
Results append to benchmarks/benchmark_log.tsv keyed by git commit, via the
same BenchmarkTimer used inside the pipeline.
"""
import argparse
import csv
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = BENCH_DIR
DATA_DIR = BENCH_DIR / "data"

# The real dataset used for milestone (tier=full) validation.
DEFAULT_FULL_MATRIX = (
    BENCH_DIR.parent
    / "my-notes"
    / "CLR-optimizations"
    / "62_Coelastrummicroporum.hifi_reads_6mer_matrix.tsv"
)

# Stage registry: name -> callable(matrix_path, workdir). Stages are
# self-contained so each can run in an isolated subprocess.
STAGE_NAMES = ["kmer_stats", "preprocess_clr", "pca_pre", "dr_pca"]

# small-tier defaults: 10k reads x 2080 features matches the shape of a
# canonical 6-mer matrix while keeping each stage run in the seconds range
SMALL_N_READS = 10_000
SMALL_N_FEATURES = 2_080


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def generate_synthetic_matrix(path: Path, n_reads: int, n_features: int,
                              seed: int) -> None:
    """Write a deterministic synthetic k-mer count matrix as TSV.

    Counts are Poisson-distributed uint32 values, written in row blocks so
    generation itself never holds the full matrix in memory.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    columns = [f"kmer_{i:05d}" for i in range(n_features)]
    block_rows = 5_000

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("sequence_id\t" + "\t".join(columns) + "\n")
        for start in range(0, n_reads, block_rows):
            rows = min(block_rows, n_reads - start)
            counts = rng.poisson(lam=5.0, size=(rows, n_features)).astype("uint32")
            ids = [f"read_{start + r:08d}" for r in range(rows)]
            lines = [
                ids[r] + "\t" + "\t".join(map(str, counts[r]))
                for r in range(rows)
            ]
            f.write("\n".join(lines) + "\n")


def synthetic_matrix_path(n_reads: int, n_features: int, seed: int) -> Path:
    return DATA_DIR / f"synthetic_{n_reads}x{n_features}_seed{seed}.tsv"


def ensure_synthetic_matrix(n_reads: int, n_features: int, seed: int) -> Path:
    path = synthetic_matrix_path(n_reads, n_features, seed)
    if not path.exists():
        print(f"[generate] {path.name} ({n_reads} x {n_features}, seed={seed})")
        generate_synthetic_matrix(path, n_reads, n_features, seed)
    return path


# ---------------------------------------------------------------------------
# Stage implementations (run inside the isolated subprocess)
# ---------------------------------------------------------------------------

def _stage_kmer_stats(matrix_path: Path, workdir: Path):
    from kmer_ord.io.kmer_stats import process_kmer_file
    process_kmer_file(
        input_file=str(matrix_path),
        output_file=str(workdir / "kmer_metrics.tsv"),
        cpus=1,
    )


def _clr_matrix(matrix_path: Path):
    from kmer_ord.dr.loader import load_matrix
    from kmer_ord.dr.preprocess import preprocess_data
    matrix = load_matrix(matrix_path)
    return preprocess_data(matrix, "clr")


def _stage_preprocess_clr(matrix_path: Path, workdir: Path):
    import numpy as np
    X = _clr_matrix(matrix_path)
    np.save(workdir / "clr.npy", X)


def _stage_pca_pre(matrix_path: Path, workdir: Path):
    import numpy as np
    from kmer_ord.dr.preprocess import reduce_dimensions_with_pca
    X = _clr_matrix(matrix_path)
    # keep_pcs=50 mirrors the recommended --pca-pre --keep-pcs 50 recipe
    reduced = reduce_dimensions_with_pca(X, keep_pcs=50)
    np.save(workdir / "pca50.npy", reduced)


def _stage_dr_pca(matrix_path: Path, workdir: Path):
    """2-D PCA embedding via the DR dispatch. PCA (not UMAP etc.) keeps the
    small tier deterministic and fast while still exercising the DR path."""
    import numpy as np
    from kmer_ord.dr.methods import _run_single_method
    X = _clr_matrix(matrix_path)
    X = np.ascontiguousarray(X, dtype=np.float32)
    embedding, _ = _run_single_method(X=X, method="pca", dims=2, seed=42)
    np.save(workdir / "dr_pca_2d.npy", embedding)


STAGES = {
    "kmer_stats": _stage_kmer_stats,
    "preprocess_clr": _stage_preprocess_clr,
    "pca_pre": _stage_pca_pre,
    "dr_pca": _stage_dr_pca,
}
assert list(STAGES) == STAGE_NAMES


def run_stage_in_this_process(stage: str, matrix_path: Path, tier: str,
                              log_dir: Path, workdir: Path):
    """Entry point of the isolated subprocess: time one stage."""
    from kmer_ord.utils.benchmark import BenchmarkTimer

    workdir.mkdir(parents=True, exist_ok=True)
    with BenchmarkTimer(
        label=f"bench_{tier}_{stage}",
        log_dir=str(log_dir),
        script_name="run_benchmarks",
        input_file=matrix_path,
        input_args=f"tier={tier}",
    ):
        STAGES[stage](matrix_path, workdir)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def cmd_run(args):
    log_dir = Path(args.log_dir)

    if args.tier == "small":
        matrix_path = ensure_synthetic_matrix(
            args.n_reads, args.n_features, args.seed
        )
    else:
        matrix_path = Path(args.full_matrix)
        if not matrix_path.exists():
            sys.exit(f"Full-tier matrix not found: {matrix_path}")

    stages = args.stages.split(",") if args.stages else STAGE_NAMES
    unknown = set(stages) - set(STAGE_NAMES)
    if unknown:
        sys.exit(f"Unknown stage(s): {sorted(unknown)}; choose from {STAGE_NAMES}")

    import tempfile
    failures = []
    for stage in stages:
        print(f"[run] tier={args.tier} stage={stage}")
        with tempfile.TemporaryDirectory(prefix=f"kmerord_bench_{stage}_") as tmp:
            result = subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "stage",
                "--name", stage,
                "--matrix", str(matrix_path),
                "--tier", args.tier,
                "--log-dir", str(log_dir),
                "--workdir", tmp,
            ])
        if result.returncode != 0:
            failures.append(stage)
            print(f"[run] stage {stage} FAILED (exit {result.returncode})")

    log_file = log_dir / "benchmark_log.tsv"
    print(f"\nResults appended to {log_file}")
    _print_latest_rows(log_file, len(stages) - len(failures))
    if failures:
        sys.exit(f"Failed stages: {failures}")


def cmd_stage(args):
    run_stage_in_this_process(
        stage=args.name,
        matrix_path=Path(args.matrix),
        tier=args.tier,
        log_dir=Path(args.log_dir),
        workdir=Path(args.workdir),
    )


def _read_log(log_file: Path) -> list[dict]:
    if not log_file.exists():
        sys.exit(f"No benchmark log at {log_file}; run benchmarks first.")
    with open(log_file) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _fmt_bytes(n: float) -> str:
    return f"{n / (1024 ** 3):.2f} GB" if n >= 1024 ** 3 else f"{n / (1024 ** 2):.1f} MB"


def _print_latest_rows(log_file: Path, n: int):
    rows = _read_log(log_file)[-n:] if n > 0 else []
    for r in rows:
        peak = int(r["peak_rss_self_bytes"]) + int(r["peak_rss_children_bytes"])
        print(
            f"  {r['stage_label']:<32} peak {_fmt_bytes(peak):>10}   "
            f"wall {float(r['wall_time_s']):8.2f}s"
        )


def cmd_compare(args):
    rows = _read_log(Path(args.log_dir) / "benchmark_log.tsv")

    if args.commits:
        commit_a, commit_b = args.commits
    else:
        # default: the two most recent distinct commits in the log,
        # compared oldest -> newest
        seen = OrderedDict()
        for r in rows:
            seen[r["git_commit"]] = None
        commits = list(seen)
        if len(commits) < 2:
            sys.exit("Need at least two distinct commits in the log to compare.")
        commit_a, commit_b = commits[-2], commits[-1]

    def latest_per_stage(commit: str) -> dict[str, dict]:
        out = {}
        for r in rows:
            if r["git_commit"].startswith(commit):
                out[r["stage_label"]] = r  # later rows overwrite: keep latest
        return out

    a_rows, b_rows = latest_per_stage(commit_a), latest_per_stage(commit_b)
    common = [s for s in a_rows if s in b_rows]
    if not common:
        sys.exit(f"No common stages between {commit_a} and {commit_b}.")

    print(f"\n{'stage':<32} {'metric':<10} {commit_a:>14} {commit_b:>14} {'delta':>9}")
    print("-" * 84)
    for stage in common:
        ra, rb = a_rows[stage], b_rows[stage]
        peak_a = int(ra["peak_rss_self_bytes"]) + int(ra["peak_rss_children_bytes"])
        peak_b = int(rb["peak_rss_self_bytes"]) + int(rb["peak_rss_children_bytes"])
        wall_a, wall_b = float(ra["wall_time_s"]), float(rb["wall_time_s"])
        peak_delta = (peak_b - peak_a) / peak_a * 100 if peak_a else 0.0
        wall_delta = (wall_b - wall_a) / wall_a * 100 if wall_a else 0.0
        print(f"{stage:<32} {'peak RAM':<10} {_fmt_bytes(peak_a):>14} "
              f"{_fmt_bytes(peak_b):>14} {peak_delta:>+8.1f}%")
        print(f"{'':<32} {'wall time':<10} {wall_a:>13.2f}s {wall_b:>13.2f}s "
              f"{wall_delta:>+8.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run benchmark stages")
    p_run.add_argument("--tier", choices=["small", "full"], default="small")
    p_run.add_argument("--stages", default=None,
                       help=f"Comma-separated subset of {STAGE_NAMES}")
    p_run.add_argument("--n-reads", type=int, default=SMALL_N_READS)
    p_run.add_argument("--n-features", type=int, default=SMALL_N_FEATURES)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--full-matrix", default=str(DEFAULT_FULL_MATRIX))
    p_run.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    p_run.set_defaults(func=cmd_run)

    # internal: executed by `run` in a fresh subprocess per stage
    p_stage = sub.add_parser("stage", help=argparse.SUPPRESS)
    p_stage.add_argument("--name", required=True, choices=STAGE_NAMES)
    p_stage.add_argument("--matrix", required=True)
    p_stage.add_argument("--tier", required=True)
    p_stage.add_argument("--log-dir", required=True)
    p_stage.add_argument("--workdir", required=True)
    p_stage.set_defaults(func=cmd_stage)

    p_cmp = sub.add_parser("compare", help="Compare two commits in the log")
    p_cmp.add_argument("--commits", nargs=2, metavar=("OLD", "NEW"), default=None)
    p_cmp.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
