# tests/test_screen_memory.py
"""Phase-3 regression test: the parameter screen must not accumulate
sequence_id columns across grid combinations.

Each combination's DataFrame used to be retained wholesale for the final
density grid, duplicating an object-dtype sequence_id column (~95 bytes/row)
per combination for the entire screen — only the two float coordinate columns
are actually read by the renderer.
"""
import numpy as np
import pandas as pd

import kmer_ord.vis.embedding_plots as embedding_plots
from kmer_ord.dr.methods import _run_parameter_screen


def test_screen_retains_only_coordinate_columns(tmp_path, monkeypatch):
    captured = {}

    def fake_render(combos, **kwargs):
        captured["combos"] = combos

    # _run_parameter_screen imports the renderer at call time, so patching
    # the source module attribute intercepts it
    monkeypatch.setattr(
        embedding_plots, "render_param_screen_density_grid", fake_render
    )

    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    sequence_ids = np.array([f"read_{i:03d}" for i in range(60)])

    # t-SNE: screenable via sklearn only (no numba warm-up); 1x1 grid keeps
    # the test fast while still exercising save/track/render plumbing
    out_paths = _run_parameter_screen(
        X=X,
        method="tsne",
        dims=2,
        seed=42,
        scale="default",
        output_dir=tmp_path,
        normalisation="clr",
        input_name="testinput",
        sequence_ids=sequence_ids,
        n_jobs=1,
        values1=["tsne=5"],
        values2=["tsne=100"],
    )

    combos = captured["combos"]
    assert len(combos) == 1
    axis1_value, axis2_value, coords = combos[0]
    assert (axis1_value, axis2_value) == (5, 100)

    # the retained frame must hold ONLY the two float coordinate columns
    assert list(coords.columns) == ["tsne_1", "tsne_2"]
    assert "sequence_id" not in coords.columns
    assert all(np.issubdtype(dt, np.floating) for dt in coords.dtypes)
    assert len(coords) == 60

    # ...while the per-combination TSV on disk still carries sequence_id
    assert len(out_paths) == 1
    on_disk = pd.read_csv(out_paths[0], sep="\t")
    assert list(on_disk.columns) == ["sequence_id", "tsne_1", "tsne_2"]
    assert list(on_disk["sequence_id"]) == list(sequence_ids)
