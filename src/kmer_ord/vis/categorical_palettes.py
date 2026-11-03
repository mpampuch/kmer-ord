"""Categorical colours for embedding plots.

Default is colorcet ``glasbey`` with a fixed swap at indices 4 and 7:
``#97ff00`` (light yellow-green beside dark green) and ``#ffa52f`` (orange)
trade places so early categories get orange instead of light green.
``distinct`` and ``Category10`` remain selectable in Dash for comparison.
"""

from __future__ import annotations

import colorsys

import bokeh.palettes
from colorcet import glasbey

# Missing feature values stay this grey in every categorical embedding plot.
NA_LABEL = "NA"
NA_COLOR = "#BEBEBE"

DEFAULT_CATEGORICAL_PALETTE = "glasbey"

# colorcet glasbey indices — verified against the shipped 256-colour list.
GLASBEY_LIGHT_GREEN = "#97ff00"
GLASBEY_ORANGE = "#ffa52f"
GLASBEY_LIGHT_GREEN_INDEX = 4
GLASBEY_ORANGE_INDEX = 7

# Yellow through yellow-green (~36°–90°). Used when extending ``distinct``.
_YELLOW_GREEN_HUE = (0.10, 0.25)
_YELLOW_GREEN_MIN_LUMINANCE = 0.45
_YELLOW_GREEN_MIN_SATURATION = 0.2
_NA_GREY_MAX_CHANNEL_DISTANCE = 0.08

_DISTINCT_HEX: tuple[str, ...] = (
    "#EE7733",
    "#0077BB",
    "#88CCEE",
    "#882255",
    "#117733",
    "#332288",
    "#EE3377",
    "#44AA99",
    "#33BBEE",
    "#CC3311",
    "#AA4499",
    "#004488",
    "#009988",
    "#CC6677",
    "#BB5566",
)

_glasbey_pool_cache: list[str] | None = None
_distinct_pool_cache: list[str] | None = None


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    text = hex_color.strip().lstrip("#")
    if len(text) == 8:
        text = text[:6]
    if len(text) != 6:
        raise ValueError(f"expected a hex colour, got {hex_color!r}")
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _is_high_luminance_yellow_green(hex_color: str) -> bool:
    red, green, blue = _hex_to_rgb01(hex_color)
    hue, saturation, _value = colorsys.rgb_to_hsv(red, green, blue)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    hue_lo, hue_hi = _YELLOW_GREEN_HUE
    return (
        hue_lo <= hue <= hue_hi
        and luminance >= _YELLOW_GREEN_MIN_LUMINANCE
        and saturation >= _YELLOW_GREEN_MIN_SATURATION
    )


def glasbey_light_green_replacement() -> str:
    """Orange placed at ``GLASBEY_LIGHT_GREEN_INDEX`` in the default glasbey pool."""
    return GLASBEY_ORANGE


def _glasbey_categorical_pool() -> list[str]:
    """glasbey with the light green and orange swatches swapped at indices 4 and 7."""
    global _glasbey_pool_cache
    if _glasbey_pool_cache is not None:
        return _glasbey_pool_cache

    pool = list(glasbey)
    pool[GLASBEY_LIGHT_GREEN_INDEX] = GLASBEY_ORANGE
    pool[GLASBEY_ORANGE_INDEX] = GLASBEY_LIGHT_GREEN
    _glasbey_pool_cache = pool
    return _glasbey_pool_cache


def _near_na_grey(hex_color: str) -> bool:
    red, green, blue = _hex_to_rgb01(hex_color)
    na_red, na_green, na_blue = _hex_to_rgb01(NA_COLOR)
    return max(
        abs(red - na_red),
        abs(green - na_green),
        abs(blue - na_blue),
    ) < _NA_GREY_MAX_CHANNEL_DISTANCE


def _distinct_pool() -> list[str]:
    global _distinct_pool_cache
    if _distinct_pool_cache is not None:
        return _distinct_pool_cache

    used = {color.upper() for color in _DISTINCT_HEX}
    used.add(NA_COLOR.upper())
    extras: list[str] = []
    for color in _glasbey_categorical_pool():
        key = color.upper()
        if key in used:
            continue
        try:
            skip = _is_high_luminance_yellow_green(color) or _near_na_grey(color)
        except ValueError:
            continue
        if skip:
            continue
        extras.append(color)
        used.add(key)
    _distinct_pool_cache = [*_DISTINCT_HEX, *extras]
    return _distinct_pool_cache


def categorical_hex_colors(n: int, palette: str = DEFAULT_CATEGORICAL_PALETTE) -> list[str]:
    """Return ``n`` hex colours for a qualitative palette."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if palette == "distinct":
        pool = _distinct_pool()
    elif palette == "glasbey":
        pool = _glasbey_categorical_pool()
    elif palette == "Category10":
        if n <= 10:
            return list(bokeh.palettes.Category10[10][:n])
        pool = _glasbey_categorical_pool()
    else:
        raise ValueError(
            f"Unknown categorical palette {palette!r}. "
            "Expected 'distinct', 'glasbey', or 'Category10'."
        )
    if n == 0:
        return []
    if n <= len(pool):
        return list(pool[:n])
    return [pool[i % len(pool)] for i in range(n)]
