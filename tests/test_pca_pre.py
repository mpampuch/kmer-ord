# tests/test_pca_pre.py
"""Tests for the IncrementalPCA option of reduce_dimensions_with_pca.

IncrementalPCA on a single batch is mathematically the same SVD as standard
PCA, so the single-batch case must match PCA outputs; the multi-batch case is
an approximation and is tested for shape, determinism, and variance capture.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.utils import gen_batches

from kmer_ord.dr.preprocess import (
    describe_ipca_batches,
    format_ipca_batch_line,
    preprocess_data,
    reduce_dimensions_with_pca,
    resolve_ipca_batch_size,
)


@pytest.fixture
def clr_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    counts = rng.poisson(lam=6.0, size=(80, 20)).astype(np.int64)
    df = pd.DataFrame(
        counts,
        index=[f"read_{i:03d}" for i in range(80)],
        columns=[f"kmer_{j:02d}" for j in range(20)],
    )
    return preprocess_data(df, "clr")


def test_ipca_single_batch_matches_pca(clr_matrix):
    expected = PCA(n_components=5).fit_transform(clr_matrix.values)
    result = reduce_dimensions_with_pca(
        clr_matrix, keep_pcs=5, method="ipca", batch_size=1000
    )
    assert result.shape == (80, 5)
    assert list(result.columns) == [f"PC{i+1}" for i in range(5)]
    assert list(result.index) == list(clr_matrix.index)
    np.testing.assert_allclose(result.values, expected, rtol=1e-3, atol=1e-4)


def test_ipca_multi_batch_deterministic_and_useful(clr_matrix):
    a = reduce_dimensions_with_pca(clr_matrix, keep_pcs=5, method="ipca", batch_size=16)
    b = reduce_dimensions_with_pca(clr_matrix, keep_pcs=5, method="ipca", batch_size=16)
    assert a.shape == (80, 5)
    pd.testing.assert_frame_equal(a, b)  # no hidden randomness

    # the estimated subspace must capture variance comparably to exact PCA
    pca_var = PCA(n_components=5).fit(clr_matrix.values).explained_variance_.sum()
    ipca_var = a.values.var(axis=0, ddof=1).sum()
    assert ipca_var >= 0.9 * pca_var


def test_ipca_variance_threshold(clr_matrix):
    result = reduce_dimensions_with_pca(
        clr_matrix, keep_variance=0.9, method="ipca", batch_size=1000
    )
    # single batch == exact PCA, so the component count must match the
    # standard-PCA selection for the same threshold
    full = PCA().fit(clr_matrix.values)
    expected_pcs = int(
        np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.9) + 1
    )
    assert result.shape == (80, expected_pcs)


def test_ipca_output_is_float32(clr_matrix):
    result = reduce_dimensions_with_pca(
        clr_matrix, keep_pcs=3, method="ipca", batch_size=32
    )
    assert all(dt == np.float32 for dt in result.dtypes)


def test_unknown_pca_method_raises(clr_matrix):
    with pytest.raises(ValueError, match="Unknown PCA method"):
        reduce_dimensions_with_pca(clr_matrix, keep_pcs=3, method="nope")


def test_missing_keep_args_error_names_cli_flags(clr_matrix):
    # the error must reference the CLI flag spellings, not the Python
    # parameter names, so users can act on it directly
    with pytest.raises(ValueError, match=r"--keep-pcs.*--keep-variance"):
        reduce_dimensions_with_pca(clr_matrix)


@pytest.mark.parametrize(
    "command, base_args",
    [
        ("project", ["-i", "x.fa", "-o", "out"]),
        ("cluster", ["-i", "x.fa", "-o", "out"]),
        ("dr", ["-i", "x.tsv", "-o", "out", "-m", "umap"]),
    ],
)
def test_cli_pca_pre_requires_keep_flag(command, base_args):
    """--pca-pre without --keep-pcs/--keep-variance must fail before any work."""
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    result = CliRunner().invoke(app, [command, *base_args, "--pca-pre"])
    assert result.exit_code != 0
    message = result.output or str(result.exception)
    assert "--keep-pcs" in message and "--keep-variance" in message


def test_resolve_ipca_batch_size_default_and_floor():
    # None uses max(2048, 5 * components fitted).
    assert resolve_ipca_batch_size(80, 20, keep_pcs=5, keep_variance=None, batch_size=None) == 2048
    # an explicit batch already above the component count is kept
    assert resolve_ipca_batch_size(80, 20, keep_pcs=5, keep_variance=None, batch_size=16) == 16
    # sklearn requires batch >= components, so a too-small batch is raised
    assert resolve_ipca_batch_size(1000, 800, keep_pcs=600, keep_variance=None, batch_size=100) == 600
    # variance selection fits at most 500 components before choosing a cutoff
    assert resolve_ipca_batch_size(2000, 2000, keep_pcs=None, keep_variance=0.9, batch_size=None) == max(2048, 5 * 500)


def test_ipca_batch_report_full_short_merged_and_single():
    # Exact division: every batch, including the last, is a full batch.
    full_fit, full_transform = describe_ipca_batches(
        4096, 200, keep_pcs=100, keep_variance=None, batch_size=2048
    )
    assert list(full_fit.slices) == list(gen_batches(4096, 2048, min_batch_size=100))
    assert full_fit.n_batches == 2 and full_fit.last_rows == 2048
    assert full_fit.handling == "full"
    assert full_transform.handling == "full" and full_transform.n_batches == 2
    assert format_ipca_batch_line("fit", full_fit) == (
        "ipca fit          2 batches; last 2,048 rows, full"
    )

    # Remainder 1,808 is at least 100 components, so it stays its own short batch.
    short_fit, short_transform = describe_ipca_batches(
        10_000, 200, keep_pcs=100, keep_variance=None, batch_size=2048
    )
    assert list(short_fit.slices) == list(gen_batches(10_000, 2048, min_batch_size=100))
    assert short_fit.n_batches == 5 and short_fit.last_rows == 1808
    assert short_fit.handling == "short"
    assert short_transform.n_batches == 5 and short_transform.last_rows == 1808
    assert format_ipca_batch_line("fit", short_fit) == (
        "ipca fit          5 batches; last 1,808 rows, short remainder"
    )
    assert format_ipca_batch_line("transform", short_transform) == (
        "ipca transform    5 batches; last 1,808 rows, short remainder"
    )

    # Remainder 1,808 is below 1,900 components, so sklearn merges it into the
    # previous batch: one fewer fit batch, and that last batch is larger than
    # the batch size. Transform still keeps the short tail.
    merged_fit, merged_transform = describe_ipca_batches(
        10_000, 2000, keep_pcs=1900, keep_variance=None, batch_size=2048
    )
    assert list(merged_fit.slices) == list(gen_batches(10_000, 2048, min_batch_size=1900))
    assert merged_fit.n_batches == merged_transform.n_batches - 1
    assert merged_fit.n_batches == 4
    assert merged_fit.last_rows == 3856 and merged_fit.last_rows > 2048
    assert merged_fit.handling == "merged" and merged_fit.remainder == 1808
    assert merged_transform.n_batches == 5 and merged_transform.handling == "short"
    assert format_ipca_batch_line("fit", merged_fit) == (
        "ipca fit          4 batches; last 3,856 rows, merged "
        "(remainder 1,808 is below 1,900 components)"
    )
    assert format_ipca_batch_line("transform", merged_transform) == (
        "ipca transform    5 batches; last 1,808 rows, short remainder"
    )

    # The whole matrix is smaller than the batch size: one short batch of every row.
    single_fit, single_transform = describe_ipca_batches(
        80, 20, keep_pcs=5, keep_variance=None, batch_size=2048
    )
    assert list(single_fit.slices) == list(gen_batches(80, 2048, min_batch_size=5))
    assert single_fit.n_batches == 1 and single_fit.last_rows == 80
    assert single_fit.handling == "short"
    assert single_transform.n_batches == 1 and single_transform.last_rows == 80
    assert format_ipca_batch_line("fit", single_fit) == (
        "ipca fit          1 batch; last 80 rows, short remainder"
    )

    # Variance selection fits at most 500 components first, so the threshold
    # does not change the batch plan versus asking for those 500 components.
    variance_fit, _ = describe_ipca_batches(
        2000, 2000, keep_pcs=None, keep_variance=0.9, batch_size=None
    )
    capped_fit, _ = describe_ipca_batches(
        2000, 2000, keep_pcs=500, keep_variance=None, batch_size=None
    )
    assert variance_fit.slices == capped_fit.slices
    assert variance_fit.batch_size == max(2048, 5 * 500)
    assert list(variance_fit.slices) == list(
        gen_batches(2000, variance_fit.batch_size, min_batch_size=500)
    )


def test_pca_ignores_batch_size(clr_matrix):
    expected = reduce_dimensions_with_pca(clr_matrix, keep_pcs=3, method="pca")
    result = reduce_dimensions_with_pca(
        clr_matrix, keep_pcs=3, method="pca", batch_size=16
    )
    pd.testing.assert_frame_equal(expected, result)


def test_cli_pca_pre_method_rejects_unknown():
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    result = CliRunner().invoke(
        app,
        ["project", "-i", "x.fa", "-o", "out", "--pca-pre-method", "nope"],
    )
    assert result.exit_code != 0
    message = (result.output or str(result.exception)).lower()
    assert "nope" in message and ("pca" in message or "ipca" in message)


def test_cli_pca_pre_method_help_lists_choices():
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    result = CliRunner().invoke(app, ["project", "--help"])
    assert result.exit_code == 0
    assert "--pca-pre-method" in result.output
    assert "pca" in result.output
    assert "ipca" in result.output
    # An 80-column help table ellipsizes this flag to "--pca-pre-batch-…".
    assert "--pca-pre-batch" in result.output
