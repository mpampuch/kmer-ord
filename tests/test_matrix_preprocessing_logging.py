# tests/test_matrix_preprocessing_logging.py
"""The matrix-preprocessing step must announce itself on the console.
"""
import numpy as np
import pandas as pd
import pytest

from kmer_ord.utils.logging_utils import console
from kmer_ord.workflow.context import MatrixContext
from kmer_ord.workflow.operations import MatrixPreprocessing


@pytest.fixture
def matrix_tsv(tmp_path):
    rng = np.random.default_rng(7)
    counts = rng.poisson(lam=5.0, size=(20, 8)).astype(np.uint32)
    df = pd.DataFrame(
        counts,
        index=[f"read_{i:03d}" for i in range(20)],
        columns=[f"kmer_{j}" for j in range(8)],
    )
    df.index.name = "sequence_id"
    path = tmp_path / "matrix.tsv"
    df.to_csv(path, sep="\t")
    return path


def _run_capturing(operation, matrix_path, output_dir, force):
    context = MatrixContext(matrix_path, output_dir, force=force, script_name="test")
    with console.capture() as capture:
        operation.run(context)
    return capture.get()


def test_preprocessing_logs_section_and_input_shape(matrix_tsv, tmp_path):
    output = _run_capturing(
        MatrixPreprocessing(normalisations=("clr",)),
        matrix_tsv,
        tmp_path / "out",
        force=True,
    )
    assert "matrix preprocessing" in output
    assert "clr" in output
    assert "20 sequences" in output and "8 features" in output
    assert "normalising" in output


def test_preprocessing_logs_cached_skip(matrix_tsv, tmp_path):
    out_dir = tmp_path / "out"
    _run_capturing(
        MatrixPreprocessing(normalisations=("clr",)), matrix_tsv, out_dir, force=True
    )
    # second run must say it reused the cached matrix instead of staying silent
    output = _run_capturing(
        MatrixPreprocessing(normalisations=("clr",)), matrix_tsv, out_dir, force=False
    )
    assert "matrix preprocessing" in output
    assert "skipping clr" in output


def test_preprocessing_logs_pca_pre_result(matrix_tsv, tmp_path):
    output = _run_capturing(
        MatrixPreprocessing(normalisations=("clr",), pca_dim_red=True, keep_pcs=3),
        matrix_tsv,
        tmp_path / "out",
        force=True,
    )
    assert "pca" in output.lower()
    assert "3" in output  # retained-component count is reported


def test_norm_all_expands_to_all_normalisations(matrix_tsv, tmp_path):
    """--norm all previously hit a NameError (undefined ALL_METHODS) and,
    even if imported, would have expanded to DR method names."""
    out_dir = tmp_path / "out"
    context = MatrixContext(matrix_tsv, out_dir, force=True, script_name="test")
    MatrixPreprocessing(normalisations=("all",)).run(context)

    produced = {p.name for p in context.get("preprocessed_matrices")}
    expected = {
        f"matrix_{norm}.npy" for norm in ("raw", "relative", "log", "clr", "zscore")
    }
    assert produced == expected
