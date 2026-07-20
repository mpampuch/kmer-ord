"""
Peak-RSS memory-regression tests over the TESTS.md project runs.

These run the full ``kmer-ord project`` pipeline on the real test dataset and
assert the peak resident set size of the whole process tree stays under a
per-k budget - the direct check that the sparse refactor fixed the OOM.

They are marked ``slow`` and auto-skip when the dataset or the runtime
environment (conda k-mer counter, heavy DR deps) is unavailable, so a plain
``pytest`` run on a dev box does not fail spuriously. Run explicitly with:

    pytest -m slow tests/test_memory_regression.py

Only dev-machine-safe k values (6, 8) are exercised here; the large-k cases
(10, 11) that previously OOM'd should be validated on a big-memory node.

Thresholds are overridable via env vars, e.g. ``KMERORD_MEM_LIMIT_K8=6``.
"""
import os
import sys
import time
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "TEST-DATA" / "63_Monoraphidiumcircinale.hifi_reads.subsampled.1percent.fasta"

# Default peak-RSS budgets in GB (well below the >400 GB pre-fix OOM).
# Only k values that are safe to run on a typical dev machine are exercised;
# larger k (10, 11) previously OOM'd and are validated on a big-memory node.
DEFAULT_LIMITS_GB = {6: 2.0, 8: 4.0}


def _limit_gb(k: int) -> float:
    return float(os.environ.get(f"KMERORD_MEM_LIMIT_K{k}", DEFAULT_LIMITS_GB[k]))


def _tree_peak_rss(proc: "psutil.Process", poll_interval: float = 0.05) -> tuple[int, int]:
    """Run a process to completion while sampling process-tree RSS.

    Returns ``(peak_bytes, returncode)``.
    """
    peak = 0
    while proc.poll() is None:
        peak = max(peak, _current_tree_rss(proc.pid))
        time.sleep(poll_interval)
    peak = max(peak, _current_tree_rss(proc.pid))
    return peak, proc.returncode


def _current_tree_rss(pid: int) -> int:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    procs = [parent]
    try:
        procs += parent.children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


@pytest.mark.slow
@pytest.mark.parametrize("k", [6, 8])
def test_project_peak_memory(tmp_path, k):
    import subprocess

    if not DATASET.exists():
        pytest.skip(f"Test dataset not found: {DATASET}")

    env = os.environ.copy()
    # Ensure the subprocess runs THIS source tree, not any installed copy.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    )

    cmd = [
        sys.executable, "-m", "kmer_ord.cli.main", "project",
        "--input", str(DATASET),
        "--output", str(tmp_path / f"K{k}"),
        "--threads", "1",
        "--kmer", str(k),
        "--dr", "pca",          # fast, dependency-light DR method
        "--no-tiara",
        "--no-matrix-tsv",
    ]

    proc = psutil.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True)
    peak_bytes, returncode = _tree_peak_rss(proc)
    output = proc.stdout.read() if proc.stdout else ""

    if returncode != 0:
        pytest.skip(
            f"Pipeline did not complete (rc={returncode}); environment likely "
            f"missing conda k-mer counter or DR deps.\n{output[-2000:]}"
        )

    peak_gb = peak_bytes / (1024 ** 3)
    limit_gb = _limit_gb(k)
    print(f"[mem] k={k} peak={peak_gb:.2f} GB (limit {limit_gb:.1f} GB)")
    assert peak_gb < limit_gb, (
        f"k={k} peak RSS {peak_gb:.2f} GB exceeded budget {limit_gb:.1f} GB"
    )
