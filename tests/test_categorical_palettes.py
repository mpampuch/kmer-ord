"""The default categorical palette is glasbey with one light-green swap."""

import pytest
import bokeh.palettes
from colorcet import glasbey

from kmer_ord.vis.categorical_palettes import (
    GLASBEY_LIGHT_GREEN,
    GLASBEY_LIGHT_GREEN_INDEX,
    GLASBEY_ORANGE,
    GLASBEY_ORANGE_INDEX,
    NA_COLOR,
    _DISTINCT_HEX,
    _is_high_luminance_yellow_green,
    categorical_hex_colors,
    glasbey_light_green_replacement,
)


def _rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _min_pairwise_distance(colors):
    best = None
    for index, left in enumerate(colors):
        for right in colors[index + 1:]:
            distance = sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5
            best = distance if best is None else min(best, distance)
    return best


def test_glasbey_indices_match_colorcet():
    assert glasbey[GLASBEY_LIGHT_GREEN_INDEX].lower() == GLASBEY_LIGHT_GREEN
    assert glasbey[GLASBEY_ORANGE_INDEX].lower() == GLASBEY_ORANGE


def test_glasbey_default_swaps_orange_and_light_green_at_indices_four_and_seven():
    assert glasbey_light_green_replacement() == GLASBEY_ORANGE

    pool = categorical_hex_colors(len(glasbey), "glasbey")
    assert pool[GLASBEY_LIGHT_GREEN_INDEX] == GLASBEY_ORANGE
    assert pool[GLASBEY_ORANGE_INDEX] == GLASBEY_LIGHT_GREEN
    assert pool[:4] == list(glasbey[:4])
    assert pool[5:GLASBEY_ORANGE_INDEX] == list(glasbey[5:GLASBEY_ORANGE_INDEX])

    colors = categorical_hex_colors(8, "glasbey")
    assert colors[4] == GLASBEY_ORANGE
    assert colors[7] == GLASBEY_LIGHT_GREEN


def test_first_distinct_colours_are_separated_and_not_pale_yellow():
    colors = categorical_hex_colors(8, "distinct")

    assert colors == list(_DISTINCT_HEX[:8])
    assert _min_pairwise_distance([_rgb(color) for color in colors]) >= 70
    assert all(not _is_high_luminance_yellow_green(color) for color in colors)
    forbidden = {"#DDCC77", "#DDAA33", "#999933", "#97FF00", "#BBBBBB", NA_COLOR.upper()}
    assert forbidden.isdisjoint(color.upper() for color in _DISTINCT_HEX)


def test_yellow_green_filter_keeps_orange_and_true_green():
    assert _is_high_luminance_yellow_green("#97ff00")
    assert _is_high_luminance_yellow_green("#DDCC77")
    assert not _is_high_luminance_yellow_green("#EE7733")
    assert not _is_high_luminance_yellow_green("#117733")
    assert not _is_high_luminance_yellow_green("#0077BB")


def test_distinct_extension_drops_pale_yellow_green():
    colors = categorical_hex_colors(40, "distinct")

    assert len(colors) == 40
    assert len(set(color.upper() for color in colors)) == 40
    assert all(not _is_high_luminance_yellow_green(color) for color in colors)
    assert NA_COLOR.upper() not in {color.upper() for color in colors}


def test_glasbey_and_category10_keep_expected_slices():
    pool = categorical_hex_colors(len(glasbey), "glasbey")
    assert pool[GLASBEY_LIGHT_GREEN_INDEX] == GLASBEY_ORANGE
    assert categorical_hex_colors(5, "glasbey") == pool[:5]
    assert categorical_hex_colors(10, "Category10") == list(bokeh.palettes.Category10[10])
    assert categorical_hex_colors(11, "Category10") == pool[:11]


def test_empty_and_unknown_palette():
    assert categorical_hex_colors(0, "glasbey") == []
    with pytest.raises(ValueError):
        categorical_hex_colors(-1, "glasbey")
    with pytest.raises(ValueError):
        categorical_hex_colors(3, "Set2")
