"""--verbose turns on library progress for UMAP and t-SNE; the default does not."""
import sys
import types

import numpy as np
import pytest
from typer.testing import CliRunner

from kmer_ord.cli.main import app
from kmer_ord.dr.methods import DR_HYPERPARAMS, _run_single_method

_runner = CliRunner()
_DR_VERBOSE_HELP = "library progress from dimensionality reduction"


class _RecordingModel:
    def __init__(self, sink, **kwargs):
        sink.clear()
        sink.update(kwargs)
        self.graph_ = None

    def fit_transform(self, X):
        return np.zeros((X.shape[0], 2), dtype=np.float64)


def _install_umap(monkeypatch, sink):
    try:
        import umap
    except ImportError:
        umap = types.ModuleType("umap")
        monkeypatch.setitem(sys.modules, "umap", umap)

    def factory(**kwargs):
        return _RecordingModel(sink, **kwargs)

    monkeypatch.setattr(umap, "UMAP", factory, raising=False)


def _install_tsne(monkeypatch, sink):
    import sklearn.manifold as manifold

    def factory(**kwargs):
        return _RecordingModel(sink, **kwargs)

    monkeypatch.setattr(manifold, "TSNE", factory)


def _matrix():
    return np.zeros((8, 4), dtype=np.float32)


def test_verbose_reaches_umap_as_true(monkeypatch):
    captured = {}
    _install_umap(monkeypatch, captured)
    _run_single_method(_matrix(), "umap", dims=2, seed=0, scale="default", verbose=True)
    assert captured["verbose"] is True


def test_verbose_reaches_tsne_as_one(monkeypatch):
    captured = {}
    _install_tsne(monkeypatch, captured)
    _run_single_method(_matrix(), "tsne", dims=2, seed=0, scale="default", verbose=True)
    assert captured["verbose"] == 1


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_default_omits_verbose(monkeypatch, method):
    captured = {}
    if method == "umap":
        _install_umap(monkeypatch, captured)
    else:
        _install_tsne(monkeypatch, captured)
    _run_single_method(_matrix(), method, dims=2, seed=0, scale="default")
    assert "verbose" not in captured


def test_preset_cannot_turn_umap_verbose_off(monkeypatch):
    captured = {}
    _install_umap(monkeypatch, captured)
    preset = dict(DR_HYPERPARAMS["umap"]["default"])
    preset["verbose"] = False
    monkeypatch.setitem(DR_HYPERPARAMS["umap"], "default", preset)
    _run_single_method(_matrix(), "umap", dims=2, seed=0, scale="default", verbose=True)
    assert captured["verbose"] is True


@pytest.mark.parametrize("command", ["project", "cluster", "dr"])
def test_help_documents_dr_verbose(command):
    result = _runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    # Rich wraps the option help inside a box, so drop the rules and
    # collapse whitespace before matching the sentence.
    text = " ".join(result.output.replace("│", " ").split())
    assert "--verbose" in text
    assert _DR_VERBOSE_HELP in text
