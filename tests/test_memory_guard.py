# tests/test_memory_guard.py
"""Tests for the per-method DR peak-memory estimator.

The old guard was `X.nbytes * 4`, which ignores neighbor graphs and pair
tables entirely — a `n_neighbors=200` UMAP or `FP_ratio=5` PaCMAP fit on a
PCA-reduced matrix could OOM while passing the check. The estimator must
account for the method-specific structures from the memory audit.
"""
import pytest

from kmer_ord.dr.methods import estimate_peak_memory_gb

GB = 1024 ** 3

# 1M reads on a 50-PC matrix: X is only ~200 MB, but 'large'-scale neighbor
# structures are gigabytes — exactly the case the old X.nbytes*4 guard missed.
N_SEQ = 1_000_000
N_FEAT = 50
X_GB = N_SEQ * N_FEAT * 4 / GB          # ~0.19 GB
OLD_GUARD_GB = X_GB * 4                  # ~0.75 GB


@pytest.mark.parametrize("method", ["umap", "tsne", "trimap", "pacmap", "localmap"])
def test_large_scale_estimates_exceed_old_guard(method):
    """Neighbor/pair structures at 'large' presets dwarf X.nbytes*4."""
    est = estimate_peak_memory_gb(N_SEQ, N_FEAT, method, scale="large")
    assert est > OLD_GUARD_GB, (
        f"{method} estimate {est:.2f} GB does not account for its "
        f"neighbor/pair structures (old guard was {OLD_GUARD_GB:.2f} GB)"
    )


def test_estimate_includes_resident_matrix():
    """Every estimate must at least cover X itself."""
    for method in ["pca", "umap", "tsne", "trimap", "pacmap", "localmap"]:
        assert estimate_peak_memory_gb(N_SEQ, N_FEAT, method, "default") >= X_GB


def test_pca_estimate_includes_float64_copy():
    """sklearn PCA always makes a float64 working copy of float32 input."""
    est = estimate_peak_memory_gb(N_SEQ, N_FEAT, "pca", scale="default")
    float64_copy_gb = N_SEQ * N_FEAT * 8 / GB
    assert est >= X_GB + float64_copy_gb


def test_neighbor_count_scales_estimate():
    """More neighbors -> bigger graph -> bigger estimate (RAM knob per audit)."""
    small = estimate_peak_memory_gb(N_SEQ, N_FEAT, "umap", scale="small")   # 50
    large = estimate_peak_memory_gb(N_SEQ, N_FEAT, "umap", scale="large")   # 200
    assert large > small


def test_unknown_method_falls_back_conservatively():
    est = estimate_peak_memory_gb(N_SEQ, N_FEAT, "kernel_pca", scale="default")
    assert est >= OLD_GUARD_GB  # at least as strict as the old heuristic
