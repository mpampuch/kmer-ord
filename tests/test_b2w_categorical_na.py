"""Sparse categorical features keep their missing reads as a grey NA.

rDNA only labels hits, and Tiara's second stage only labels organelles.
Those nulls used to be dropped by datashader's count_cat, so the binner
showed just the called reads.
"""
import numpy as np
import pandas as pd
import bokeh.palettes

from kmer_ord.dash.b2w import (
    _categorical_color_key,
    _categorical_with_missing,
    create_datashader_image,
    create_panel_legend,
)
from kmer_ord.vis.embedding_plots import NA_COLOR, NA_LABEL


def _sparse_labels():
    return pd.Series(["plastid", None, "n/a", "  ", "N/A", "na", "unknown", "NA"])


def test_missing_tokens_become_na_and_called_labels_stay():
    labeled = _categorical_with_missing(_sparse_labels())

    assert labeled.tolist() == [
        "plastid", NA_LABEL, NA_LABEL, NA_LABEL, NA_LABEL, NA_LABEL, "unknown", NA_LABEL,
    ]
    assert set(labeled.cat.categories) == {"plastid", "unknown", NA_LABEL}
    assert "n/a" not in set(labeled.cat.categories)


def test_na_keeps_the_shared_grey_and_does_not_take_a_palette_slot():
    labels = [f"c{i}" for i in range(10)]
    key = _categorical_color_key([NA_LABEL, *labels], "Category10")

    assert key[NA_LABEL] == NA_COLOR
    assert key["c0"] == bokeh.palettes.Category10[10][0]
    assert key["c9"] == bokeh.palettes.Category10[10][9]
    assert NA_COLOR not in {key[label] for label in labels}


def _legend_swatches(component):
    swatches = {}
    children = component.children
    if not isinstance(children, (list, tuple)):
        children = [children]
    for item in children:
        spans = item.children
        if not isinstance(spans, (list, tuple)):
            continue
        swatches[str(spans[1].children)] = spans[0].style["backgroundColor"]
    return swatches


def _sparse_frame():
    """Two separated clouds, plus points past them so neither cloud is the max edge.

    Datashader's bins are half-open, so a point exactly at the maximum is dropped.
    """
    rows = {
        "x": [2.0] * 20 + [8.0] * 20 + [0.0, 10.0],
        "y": [2.0] * 40 + [0.0, 4.0],
        "class_snd_stage": (
            ["plastid"] * 20
            + [None] * 10
            + ["n/a"] * 10
            + ["plastid", None]
        ),
    }
    return pd.DataFrame(rows)


def _visible_rgb(image):
    pixels = np.asarray(image.convert("RGBA"))
    rgb = pixels[..., :3]
    return rgb[pixels[..., 3] > 200]


def _has_color(rgb_rows, hex_color, tolerance=12):
    target = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)])
    if len(rgb_rows) == 0:
        return False
    distance = np.max(np.abs(rgb_rows.astype(int) - target), axis=1)
    return bool(np.any(distance <= tolerance))


def test_missing_reads_are_drawn_and_listed_as_grey_na():
    frame = _sparse_frame()
    image = create_datashader_image(
        frame,
        "x",
        "y",
        "class_snd_stage",
        color_palette="glasbey",
        px_spread=1,
        plot_width=60,
        plot_height=40,
    )
    visible = _visible_rgb(image)
    assert _has_color(visible, NA_COLOR)
    # The plastid calls are a different color, so the cloud is not NA-only.
    grey = np.array([int(NA_COLOR[i:i + 2], 16) for i in (1, 3, 5)])
    assert np.any(np.max(np.abs(visible.astype(int) - grey), axis=1) > 40)

    legend, legend_type = create_panel_legend(
        frame, "class_snd_stage", "viridis", "glasbey",
    )
    assert legend_type == "categorical"
    swatches = _legend_swatches(legend)
    assert swatches[NA_LABEL] == NA_COLOR
    assert swatches["plastid"] != NA_COLOR
    assert "n/a" not in swatches
