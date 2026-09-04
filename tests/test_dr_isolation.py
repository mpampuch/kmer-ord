"""Tests for DR wrapper RAM changes: merge-from-disk, apply_pca skip, method isolation.

UMAP hyperparameters are untouched. These tests cover the wrapper so a child
segfault (the Ibex UMAP death) does not discard embeddings from other methods.
"""
import os

import numpy as np
import pandas as pd
import pytest

from kmer_ord.dr.methods import (
    _run_single_method,
    merge_embedding_tsvs,
    run_dr_methods,
)


def _write_method_tsv(path, seq_ids, method, coords):
    df = pd.DataFrame(coords, columns=[f"{method}_1", f"{method}_2"])
    df.insert(0, "sequence_id", seq_ids)
    df.to_csv(path, sep="\t", index=False)


def test_merge_embedding_tsvs_single_sequence_id_column(tmp_path):
    seq_ids = ["s0", "s1", "s2"]
    pca = tmp_path / "pca.tsv"
    tsne = tmp_path / "tsne.tsv"
    _write_method_tsv(pca, seq_ids, "pca", [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    _write_method_tsv(tsne, seq_ids, "tsne", [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]])

    out = tmp_path / "merged.tsv"
    merge_embedding_tsvs([pca, tsne], out)

    df = pd.read_csv(out, sep="\t")
    assert list(df.columns) == ["sequence_id", "pca_1", "pca_2", "tsne_1", "tsne_2"]
    assert list(df["sequence_id"]) == seq_ids
    assert df["sequence_id"].tolist().count("s0") == 1
    np.testing.assert_allclose(df["pca_1"], [0.1, 0.3, 0.5])
    np.testing.assert_allclose(df["tsne_2"], [1.2, 1.4, 1.6])


def test_merge_embedding_tsvs_empty_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No embeddings"):
        merge_embedding_tsvs([], tmp_path / "merged.tsv")


def test_apply_pca_false_when_n_feat_le_100(monkeypatch):
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured[self._name] = kwargs

        def fit_transform(self, X):
            return np.zeros((X.shape[0], 2), dtype=np.float64)

    class FakeTri(FakeModel):
        _name = "trimap"

    class FakePac(FakeModel):
        _name = "pacmap"

    class FakeLocal(FakeModel):
        _name = "localmap"

    import trimap
    import pacmap
    import pacmap.pacmap as pacmap_mod

    monkeypatch.setattr(trimap, "TRIMAP", FakeTri)
    monkeypatch.setattr(pacmap, "PaCMAP", FakePac)
    monkeypatch.setattr(pacmap_mod, "LocalMAP", FakeLocal)

    X = np.zeros((20, 50), dtype=np.float32)
    for method in ("trimap", "pacmap", "localmap"):
        _run_single_method(X, method, dims=2, seed=0, scale="default", n_jobs=1)

    for name in ("trimap", "pacmap", "localmap"):
        assert captured[name].get("apply_pca") is False, name


def test_apply_pca_not_forced_when_n_feat_gt_100(monkeypatch):
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured[self._name] = kwargs

        def fit_transform(self, X):
            return np.zeros((X.shape[0], 2), dtype=np.float64)

    class FakeTri(FakeModel):
        _name = "trimap"

    import trimap

    monkeypatch.setattr(trimap, "TRIMAP", FakeTri)
    X = np.zeros((20, 200), dtype=np.float32)
    _run_single_method(X, "trimap", dims=2, seed=0, scale="default", n_jobs=1)
    assert "apply_pca" not in captured["trimap"]


def test_isolated_method_crash_does_not_abort_siblings(tmp_path, monkeypatch):
    """A child os._exit (the UMAP segfault analogue) must not drop PCA output."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 12)).astype(np.float32)
    matrix_path = tmp_path / "X.npy"
    seqid_path = tmp_path / "ids.npy"
    np.save(matrix_path, X)
    np.save(seqid_path, np.array([f"s{i}" for i in range(40)], dtype="U"))

    monkeypatch.setenv("KMER_ORD_DR_FAIL_METHOD", "umap")

    merged, graphs = run_dr_methods(
        X=None,
        methods=["umap", "pca"],
        dims=2,
        seed=0,
        scale="small",
        screen_params=False,
        output_dir=tmp_path / "dr",
        normalisation="clr",
        input_name="test",
        matrix_path=matrix_path,
        seqid_path=seqid_path,
        isolate=True,
    )

    assert merged.exists()
    df = pd.read_csv(merged, sep="\t")
    assert "pca_1" in df.columns and "pca_2" in df.columns
    assert "umap_1" not in df.columns
    assert len(df) == 40
    umap_tsv = tmp_path / "dr" / "clr" / "umap"
    pca_tsv = list((tmp_path / "dr" / "clr" / "pca").glob("*_pca_2D.tsv"))
    assert pca_tsv, "pca embedding TSV should have been written"
    assert not list(umap_tsv.glob("*_umap_2D.tsv"))
    assert os.environ.get("KMER_ORD_DR_FAIL_METHOD") == "umap"
