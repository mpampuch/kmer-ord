"""Parameter-screen feature grids drawn by `kmer-ord visualise`.

The density grid is written during dimensionality reduction and is not
re-tested here. These tests cover the later step: join each screened
embedding to the features table and lay the coloured panels out on the
same hyperparameter axes.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from kmer_ord.vis.embedding_plots import (
    MAX_CATEGORICAL_AGG_BYTES,
    PARAM_SCREEN_COL_LABEL_H,
    PARAM_SCREEN_EDGE_PAD,
    PARAM_SCREEN_PADDING,
    PARAM_SCREEN_ROW_LABEL_W,
    PARAM_SCREEN_TITLE_H,
    _padded_data_range,
    _render_feature_to_image,
    load_screen_panels,
    plot_param_screen_features,
    render_param_screen_feature_grid,
    write_screen_manifest,
)


def _features_db(db_path, frame):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        frame.to_sql("features", conn, index=False)


def _embedding_tsv(path, method, sequence_ids, shift):
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    n = len(sequence_ids)
    frame = pd.DataFrame({
        "sequence_id": sequence_ids,
        f"{method}_1": rng.normal(loc=shift, size=n),
        f"{method}_2": rng.normal(loc=shift, size=n),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _screen_project(tmp_path, filenames, manifest_rows=None):
    """Project layout visualise discovers: db beside dr/<norm>/<method>/parameter_screen/."""
    project = tmp_path / "project"
    screen = project / "dr" / "clr" / "umap" / "parameter_screen"
    screen.mkdir(parents=True)
    ids = [f"read_{i:03d}" for i in range(40)]
    for name in filenames:
        _embedding_tsv(screen / name, "umap", ids, shift=0)
    if manifest_rows is not None:
        write_screen_manifest(screen, manifest_rows)
    features = pd.DataFrame({
        "sequence_id": ids,
        "header": [f"hdr_{i}" for i in range(len(ids))],
        "taxon": ["algae" if i % 2 == 0 else "bacteria" for i in range(len(ids))],
        "gc": np.linspace(0.3, 0.7, len(ids)),
    })
    _features_db(project / "kmerord.sqlite", features)
    return project, screen, ids


def _cell_center(row, col, panel):
    x = PARAM_SCREEN_ROW_LABEL_W + col * (panel + PARAM_SCREEN_PADDING) + panel // 2
    y = PARAM_SCREEN_TITLE_H + PARAM_SCREEN_COL_LABEL_H + row * (panel + PARAM_SCREEN_PADDING) + panel // 2
    return x, y


class TestManifestAndFilenames:
    def test_manifest_roundtrip_keeps_axis_values(self, tmp_path):
        screen = tmp_path / "parameter_screen"
        (screen).mkdir()
        (screen / "reads_clr_umap_n5_min0.1_2D.tsv").write_text("sequence_id\tumap_1\tumap_2\n")
        write_screen_manifest(screen, [{
            "filename": "reads_clr_umap_n5_min0.1_2D.tsv",
            # Manifest wins over the filename, so a mismatch stays 0.25.
            "axis1_name": "n_neighbors",
            "axis1_value": 5,
            "axis2_name": "min_dist",
            "axis2_value": 0.25,
        }])

        panels = load_screen_panels(screen, "umap")
        assert len(panels) == 1
        assert panels[0]["axis1_name"] == "n_neighbors"
        assert panels[0]["axis1_value"] == 5
        assert panels[0]["axis2_name"] == "min_dist"
        assert panels[0]["axis2_value"] == 0.25
        assert panels[0]["path"].name == "reads_clr_umap_n5_min0.1_2D.tsv"

    def test_legacy_filenames_parse_every_screenable_method(self, tmp_path):
        cases = [
            ("umap", "reads_clr_umap_n5_min0.1_2D.tsv", "n_neighbors", 5, "min_dist", 0.1),
            ("tsne", "reads_clr_tsne_p30_lr200_2D.tsv", "perplexity", 30, "learning_rate", 200),
            ("trimap", "reads_clr_trimap_inliers50_weighttemp1.0_2D.tsv", "n_inliers", 50, "weight_temp", 1.0),
            ("pacmap", "reads_clr_pacmap_n10_FPratio0.5_2D.tsv", "n_neighbors", 10, "FP_ratio", 0.5),
            ("localmap", "reads_clr_localmap_n10_FPratio2_2D.tsv", "n_neighbors", 10, "FP_ratio", 2),
        ]
        for method, filename, axis1_name, axis1, axis2_name, axis2 in cases:
            screen = tmp_path / method
            screen.mkdir()
            (screen / filename).write_text("sequence_id\tx\ty\n")
            panels = load_screen_panels(screen, method)
            assert len(panels) == 1
            assert panels[0]["axis1_name"] == axis1_name
            assert panels[0]["axis1_value"] == axis1
            assert panels[0]["axis2_name"] == axis2_name
            assert panels[0]["axis2_value"] == axis2

    def test_one_dimensional_filename_uses_the_held_default(self, tmp_path):
        # UMAP's default preset holds min_dist at 0.1 when the screen is 1D.
        screen = tmp_path / "parameter_screen"
        screen.mkdir()
        (screen / "reads_clr_umap_n10_2D.tsv").write_text("sequence_id\tumap_1\tumap_2\n")
        panels = load_screen_panels(screen, "umap")
        assert panels[0]["axis1_value"] == 10
        assert panels[0]["axis2_name"] == "min_dist"
        assert panels[0]["axis2_value"] == 0.1


class TestPanelBackgroundAndRange:
    def test_padded_data_range_extends_past_both_ends(self):
        lo, hi = _padded_data_range(pd.Series([0.0, 10.0]), plot_px=100, pad_px=4)
        assert lo == pytest.approx(-0.4)
        assert hi == pytest.approx(10.4)

    def test_padded_data_range_zero_span_stays_nonempty(self):
        lo, hi = _padded_data_range(pd.Series([3.0, 3.0]), plot_px=100)
        assert lo < 3.0 < hi

    def test_screen_panel_corner_is_black_and_standalone_stays_transparent(self):
        # Points sit in the middle of the padded range, so the corner is empty.
        frame = pd.DataFrame({
            "x": np.linspace(-1, 1, 40),
            "y": np.linspace(-0.2, 0.2, 40),
            "taxon": ["algae"] * 40,
        })
        black, _ = _render_feature_to_image(
            frame, "x", "y", "taxon", mode="categorical",
            width=60, height=60, background="black", range_pad_px=4,
        )
        plain, _ = _render_feature_to_image(
            frame, "x", "y", "taxon", mode="categorical",
            width=60, height=60, range_pad_px=4,
        )
        black_px = black.to_pil().convert("RGBA").getpixel((0, 0))
        plain_px = plain.to_pil().convert("RGBA").getpixel((0, 0))
        assert black_px == (0, 0, 0, 255)
        assert plain_px[3] == 0


class TestFeatureGridLayout:
    def test_legacy_filenames_land_on_the_matching_row_and_column(self, tmp_path):
        screen = tmp_path / "parameter_screen"
        screen.mkdir()
        colors = {
            (5, 0.1): (255, 0, 0, 255),
            (5, 0.5): (0, 255, 0, 255),
            (50, 0.1): (0, 0, 255, 255),
            (50, 0.5): (255, 255, 0, 255),
        }
        for n_neighbors, min_dist in colors:
            name = f"reads_clr_umap_n{n_neighbors}_min{min_dist}_2D.tsv"
            (screen / name).write_text("sequence_id\tumap_1\tumap_2\n")

        panels_meta = load_screen_panels(screen, "umap")
        axis1_vals = sorted({p["axis1_value"] for p in panels_meta})
        axis2_vals = sorted({p["axis2_value"] for p in panels_meta})
        assert axis1_vals == [5, 50]
        assert axis2_vals == [0.1, 0.5]

        panel = 20
        images = {
            (meta["axis1_value"], meta["axis2_value"]): Image.new(
                "RGBA", (panel, panel), colors[(meta["axis1_value"], meta["axis2_value"])]
            )
            for meta in panels_meta
        }
        out = tmp_path / "grid.png"
        render_param_screen_feature_grid(
            panels=images,
            axis1_vals=axis1_vals,
            axis2_vals=axis2_vals,
            axis1_name="n_neighbors",
            axis2_name="min_dist",
            method="umap",
            feature_col="taxon",
            out_path=out,
            mode="categorical",
            legend_info={"algae": "#FF0000", "bacteria": "#00FF00"},
            width=panel,
            height=panel,
        )

        img = Image.open(out).convert("RGBA")
        expected_cells = {
            (0, 0): (255, 0, 0, 255),
            (0, 1): (0, 255, 0, 255),
            (1, 0): (0, 0, 255, 255),
            (1, 1): (255, 255, 0, 255),
        }
        for (row, col), color in expected_cells.items():
            assert img.getpixel(_cell_center(row, col, panel)) == color
        # The shared legend is appended to the right of the grid, not painted over a panel.
        grid_width = PARAM_SCREEN_ROW_LABEL_W + 2 * panel + PARAM_SCREEN_PADDING
        assert img.width > grid_width
        # The last panel used to end flush with the PNG, so its right and bottom
        # borders were drawn on a pixel that does not exist.
        last_x = PARAM_SCREEN_ROW_LABEL_W + (panel + PARAM_SCREEN_PADDING)
        last_y = PARAM_SCREEN_TITLE_H + PARAM_SCREEN_COL_LABEL_H + (panel + PARAM_SCREEN_PADDING)
        assert img.width > last_x + panel
        assert img.height >= last_y + panel + PARAM_SCREEN_EDGE_PAD


class TestPlotParamScreenFeatures:
    def test_manifest_grid_writes_one_png_per_feature_and_skips_ids(self, tmp_path):
        filenames = [
            "reads_clr_umap_n5_min0.1_2D.tsv",
            "reads_clr_umap_n5_min0.5_2D.tsv",
            "reads_clr_umap_n50_min0.1_2D.tsv",
            "reads_clr_umap_n50_min0.5_2D.tsv",
        ]
        rows = []
        for name in filenames:
            # n5_min0.1 -> axis values parsed from the tokens we put in the manifest
            n_token = name.split("_n")[1].split("_")[0]
            min_token = name.split("_min")[1].split("_")[0]
            rows.append({
                "filename": name,
                "axis1_name": "n_neighbors",
                "axis1_value": int(n_token),
                "axis2_name": "min_dist",
                "axis2_value": float(min_token),
            })
        project, _, _ = _screen_project(tmp_path, filenames, manifest_rows=rows)
        output = tmp_path / "plots" / "param_screen"

        plot_param_screen_features(project / "kmerord.sqlite", output, mode="all")

        written = sorted(p.name for p in output.rglob("*.png"))
        assert written == ["umap_by_gc.png", "umap_by_taxon.png"]
        assert (output / "clr" / "umap" / "umap_by_taxon.png").is_file()
        assert (output / "clr" / "umap" / "umap_by_gc.png").stat().st_size > 0

    def test_shared_color_key_does_not_depend_on_one_panel(self, tmp_path, monkeypatch):
        import kmer_ord.vis.embedding_plots as embedding_plots

        project, screen, ids = _screen_project(tmp_path, [
            "reads_clr_umap_n5_min0.1_2D.tsv",
            "reads_clr_umap_n5_min0.5_2D.tsv",
        ], manifest_rows=[
            {"filename": "reads_clr_umap_n5_min0.1_2D.tsv", "axis1_name": "n_neighbors",
             "axis1_value": 5, "axis2_name": "min_dist", "axis2_value": 0.1},
            {"filename": "reads_clr_umap_n5_min0.5_2D.tsv", "axis1_name": "n_neighbors",
             "axis1_value": 5, "axis2_name": "min_dist", "axis2_value": 0.5},
        ])
        # The second combination only contains algae. Its own category list
        # would assign a different colour if the key were rebuilt per panel.
        algae_ids = [i for i in ids if i.endswith(("0", "2", "4", "6", "8"))]
        _embedding_tsv(screen / "reads_clr_umap_n5_min0.5_2D.tsv", "umap", algae_ids, shift=3)

        captured = []
        real = embedding_plots._render_feature_to_image

        def spy(df, xcol, ycol, feature_col, mode="categorical", width=400, height=400,
                 value_range=None, color_key=None, **kwargs):
            captured.append({
                "feature": feature_col,
                "mode": mode,
                "color_key": None if color_key is None else dict(color_key),
                "values": set(df[feature_col].dropna().astype(str)),
            })
            return real(df, xcol, ycol, feature_col, mode=mode, width=width, height=height,
                        value_range=value_range, color_key=color_key, **kwargs)

        monkeypatch.setattr(embedding_plots, "_render_feature_to_image", spy)
        plot_param_screen_features(
            project / "kmerord.sqlite", tmp_path / "plots", mode="categorical"
        )

        taxon_calls = [c for c in captured if c["feature"] == "taxon"]
        assert len(taxon_calls) == 2
        algae_only = next(c for c in taxon_calls if c["values"] == {"algae"})
        both = next(c for c in taxon_calls if c["values"] == {"algae", "bacteria"})
        assert algae_only["color_key"] == both["color_key"]
        assert algae_only["color_key"]["algae"] == both["color_key"]["algae"]
        assert "bacteria" in algae_only["color_key"]

    def test_categorical_mode_skips_continuous_grids(self, tmp_path):
        project, _, _ = _screen_project(tmp_path, [
            "reads_clr_umap_n5_min0.1_2D.tsv",
        ], manifest_rows=[{
            "filename": "reads_clr_umap_n5_min0.1_2D.tsv",
            "axis1_name": "n_neighbors",
            "axis1_value": 5,
            "axis2_name": "min_dist",
            "axis2_value": 0.1,
        }])
        output = tmp_path / "plots"
        plot_param_screen_features(project / "kmerord.sqlite", output, mode="categorical")
        names = [p.name for p in output.rglob("*.png")]
        assert names == ["umap_by_taxon.png"]

    def test_missing_screen_directory_writes_nothing(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        db = project / "kmerord.sqlite"
        _features_db(db, pd.DataFrame({
            "sequence_id": ["read_000"],
            "taxon": ["algae"],
        }))
        output = tmp_path / "plots"
        plot_param_screen_features(db, output, mode="all")
        assert not output.exists()

    def test_high_cardinality_feature_is_skipped(self, tmp_path):
        n = MAX_CATEGORICAL_AGG_BYTES // (220 * 220 * 4) + 2
        ids = [f"read_{i:05d}" for i in range(n)]
        project = tmp_path / "project"
        screen = project / "dr" / "clr" / "umap" / "parameter_screen"
        screen.mkdir(parents=True)
        _embedding_tsv(screen / "reads_clr_umap_n5_min0.1_2D.tsv", "umap", ids[:30], shift=0)
        write_screen_manifest(screen, [{
            "filename": "reads_clr_umap_n5_min0.1_2D.tsv",
            "axis1_name": "n_neighbors",
            "axis1_value": 5,
            "axis2_name": "min_dist",
            "axis2_value": 0.1,
        }])
        _features_db(project / "kmerord.sqlite", pd.DataFrame({
            "sequence_id": ids,
            "accession": [f"acc_{i}" for i in range(n)],
        }))
        output = tmp_path / "plots"
        plot_param_screen_features(project / "kmerord.sqlite", output, mode="categorical")
        assert list(output.rglob("*.png")) == []


def test_density_grid_is_copied_next_to_feature_plots(tmp_path):
    project, screen, _ = _screen_project(tmp_path, [
        "reads_clr_umap_n5_min0.1_2D.tsv",
    ], manifest_rows=[{
        "filename": "reads_clr_umap_n5_min0.1_2D.tsv",
        "axis1_name": "n_neighbors",
        "axis1_value": 5,
        "axis2_name": "min_dist",
        "axis2_value": 0.1,
    }])
    source = screen / "umap_param_screen_density.png"
    source.write_bytes(b"density-grid")
    output = tmp_path / "plots" / "param_screen"

    plot_param_screen_features(project / "kmerord.sqlite", output, mode="all")

    copied = output / "clr" / "umap" / "umap_param_screen_density.png"
    assert copied.read_bytes() == source.read_bytes()
    assert (output / "clr" / "umap" / "umap_by_taxon.png").is_file()
    # The screen directory keeps its own copy.
    assert source.read_bytes() == b"density-grid"


def test_density_mode_copies_the_grid_without_feature_plots(tmp_path):
    project, screen, _ = _screen_project(tmp_path, [
        "reads_clr_umap_n5_min0.1_2D.tsv",
    ], manifest_rows=[{
        "filename": "reads_clr_umap_n5_min0.1_2D.tsv",
        "axis1_name": "n_neighbors",
        "axis1_value": 5,
        "axis2_name": "min_dist",
        "axis2_value": 0.1,
    }])
    (screen / "umap_param_screen_density.png").write_bytes(b"density-grid")
    output = tmp_path / "plots" / "param_screen"

    plot_param_screen_features(project / "kmerord.sqlite", output, mode="density")

    names = sorted(p.name for p in output.rglob("*.png"))
    assert names == ["umap_param_screen_density.png"]
    assert (output / "clr" / "umap" / "umap_param_screen_density.png").read_bytes() == b"density-grid"


@pytest.mark.parametrize("mode", ["density"])
def test_density_mode_does_not_annotate(tmp_path, mode):
    project, _, _ = _screen_project(tmp_path, [
        "reads_clr_umap_n5_min0.1_2D.tsv",
    ], manifest_rows=[{
        "filename": "reads_clr_umap_n5_min0.1_2D.tsv",
        "axis1_name": "n_neighbors",
        "axis1_value": 5,
        "axis2_name": "min_dist",
        "axis2_value": 0.1,
    }])
    output = tmp_path / "plots"
    plot_param_screen_features(project / "kmerord.sqlite", output, mode=mode)
    assert not output.exists()
