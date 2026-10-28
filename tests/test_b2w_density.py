"""Density is a coloring option, not a features-table column.

The binner lists it beside real features. The dropdown value is a sentinel
so the SQL that loads coordinates never asks for a column of that name.
"""
import numpy as np
import pandas as pd

from kmer_ord.dash.b2w import (
    DENSITY_FEATURE,
    _feature_dropdown_options,
    _feature_panel_title,
    _requested_columns,
    create_datashader_image,
    create_panel_legend,
)


def _xy_frame():
    """A dense cloud inside the range, plus one point past it.

    Datashader's bins are half-open, so a point exactly at the maximum is dropped.
    """
    return pd.DataFrame({
        "x": [2.0] * 40 + [9.0],
        "y": [2.0] * 40 + [9.0],
    })


def test_density_is_the_first_feature_option():
    options = _feature_dropdown_options(
        [{"column_name": "gc_content", "type": "continuous"}],
        ["leiden"],
    )

    assert options[0] == {"label": "density", "value": DENSITY_FEATURE}
    assert [option["value"] for option in options] == [
        DENSITY_FEATURE,
        "gc_content",
        "__cluster_sep__",
        "leiden",
    ]


def test_density_sentinel_renders_without_a_feature_column():
    frame = _xy_frame()
    image = create_datashader_image(
        frame,
        "x",
        "y",
        DENSITY_FEATURE,
        px_spread=1,
        plot_width=40,
        plot_height=40,
    )
    implicit = create_datashader_image(
        frame,
        "x",
        "y",
        None,
        px_spread=1,
        plot_width=40,
        plot_height=40,
    )

    assert image.size == (40, 40)
    assert np.array_equal(np.asarray(image), np.asarray(implicit))
    assert np.any(np.asarray(image.convert("RGBA"))[..., 3] > 0)


def test_density_sentinel_is_not_shaded_as_a_numeric_column():
    frame = _xy_frame()
    frame[DENSITY_FEATURE] = [0.0] * 20 + [1.0] * 21
    shaded = create_datashader_image(
        frame,
        "x",
        "y",
        DENSITY_FEATURE,
        color_palette="viridis",
        px_spread=1,
        plot_width=40,
        plot_height=40,
    )
    expected = create_datashader_image(
        frame.drop(columns=[DENSITY_FEATURE]),
        "x",
        "y",
        None,
        px_spread=1,
        plot_width=40,
        plot_height=40,
    )

    assert np.array_equal(np.asarray(shaded), np.asarray(expected))


def test_density_sentinel_is_not_requested_as_a_sql_column():
    columns = _requested_columns(
        ["gc_content", DENSITY_FEATURE, "leiden"],
        filter_values={"length": {"min": 100}},
    )

    assert DENSITY_FEATURE not in columns
    assert columns == {"gc_content", "leiden", "length"}
    assert _requested_columns(DENSITY_FEATURE) == set()


def test_density_panel_legend_and_title_name_density():
    frame = _xy_frame()
    frame["gc_content"] = 0.5
    legend, legend_type = create_panel_legend(
        frame, DENSITY_FEATURE, "viridis", "glasbey",
    )

    assert legend_type == "continuous"
    assert legend.children == "density"
    assert _feature_panel_title(DENSITY_FEATURE) == "density"
    assert _feature_panel_title("gc_content") == "gc_content"
