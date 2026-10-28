# src/kmer_ord/dr/screen_grid.py
import numpy as np

SCREEN_AXES = {
    "umap":     ("n_neighbors", "min_dist"),
    "tsne":     ("perplexity", "learning_rate"),
    "trimap":   ("n_inliers", "weight_temp"),
    "pacmap":   ("n_neighbors", "FP_ratio"),
    "localmap": ("n_neighbors", "FP_ratio"),
}

# Unchanged from the original hardcoded _run_parameter_screen() grids —
# also the fallback range/size source for grid-size-only and range-only
# overrides, and the exact behaviour when no new flags are given at all.
DEFAULT_GRIDS = {
    "umap":     {"n_neighbors": [5, 10, 50, 100, 150], "min_dist": [0, 0.1, 0.25, 0.5, 1.0]},
    "tsne":     {"perplexity": [5, 10, 30, 50, 100], "learning_rate": [10, 100, 200, 500]},
    "trimap":   {"n_inliers": [10, 25, 50, 100, 150], "weight_temp": [0.1, 0.5, 1.0, 2.0, 2.5]},
    "pacmap":   {"n_neighbors": [10, 25, 50, 100, 150], "FP_ratio": [0.1, 0.5, 1.0, 2.0, 5]},
    "localmap": {"n_neighbors": [10, 25, 50, 100, 150], "FP_ratio": [0.1, 0.5, 1.0, 2.0, 5]},
}

# Axes whose sensible range starts at (or spans) zero — additive, not
# multiplicative, so auto-generated grids use linear rather than log spacing.
LINEAR_AXES = {"min_dist"}

# Axes that are inherently integer counts.
INT_AXES = {"n_neighbors", "perplexity", "n_inliers", "learning_rate"}

_NICE_MULTIPLIERS = (1, 2, 5)


def _parse_specs(entries: list[str] | None) -> dict[str, str]:
    """['all=5,320', 'umap=10,300'] -> {'all': '5,320', 'umap': '10,300'}"""
    out = {}
    for entry in entries or []:
        key, _, value = entry.partition("=")
        out[key.strip().lower()] = value.strip()
    return out


def _screen_spec_form(flag: str) -> str:
    if flag.startswith("--screen_range"):
        return "'method=min,max' or 'all=min,max'"
    return "'method=v1,v2,...' or 'all=v1,v2,...'"


def _validate_screen_entry(entry: str, flag: str) -> None:
    """Reject a spec that grid lookup would store and then never apply.

    Resolution only reads keys that are a screenable method or 'all'. A bare
    '0.1,1.0' has no '=', so the whole token becomes the key and the range
    is dropped.
    """
    form = _screen_spec_form(flag)
    if "=" not in entry:
        raise ValueError(f"{flag} {entry!r} must be {form}")

    key, _, raw_value = entry.partition("=")
    method = key.strip().lower()
    value = raw_value.strip()
    if not method or not value:
        raise ValueError(f"{flag} {entry!r} must be {form}")

    allowed = (*SCREEN_AXES, "all")
    if method not in allowed:
        names = ", ".join(allowed)
        raise ValueError(
            f"{flag} method {method!r} is not screenable; use one of {names}. "
            f"Expected {form}."
        )

    parts = [part.strip() for part in value.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(f"{flag} {entry!r} must be {form}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{flag} {entry!r} must be {form}") from exc

    if flag.startswith("--screen_range"):
        if len(numbers) != 2 or not (numbers[0] < numbers[1]):
            raise ValueError(
                f"{flag} {entry!r} needs exactly two numbers with min < max. "
                f"Expected {form}."
            )


def validate_screen_specs(
    values1: list[str] | None = None,
    values2: list[str] | None = None,
    range1: list[str] | None = None,
    range2: list[str] | None = None,
) -> None:
    """Fail on screen specs that would be ignored during grid resolution."""
    for flag, entries in (
        ("--screen_values1", values1),
        ("--screen_values2", values2),
        ("--screen_range1", range1),
        ("--screen_range2", range2),
    ):
        for entry in entries or []:
            _validate_screen_entry(entry, flag)


def _parse_numbers(csv: str) -> list[float]:
    return [float(v) if ("." in v or "e" in v.lower()) else int(v)
            for v in csv.split(",") if v.strip()]


def _parse_grid_spec(spec: str | None) -> tuple[int | None, int | None]:
    if not spec:
        return None, None
    spec = spec.strip().lower()
    if "x" in spec:
        a, b = spec.split("x", 1)
        return int(a), int(b)
    return int(spec), None


def _nice_round(value: float) -> float:
    if value <= 0:
        return value
    exponent = np.floor(np.log10(value))
    log_base = np.log10(value) - exponent  # in [0, 1)
    # candidates include the next decade's leading 1 (log_base == 1.0), since
    # closeness is multiplicative — a value near the top of a decade (e.g.
    # base=8) can be nearer the next decade's "1" than this decade's "5".
    candidates = list(_NICE_MULTIPLIERS) + [10]
    nearest = min(candidates, key=lambda m: abs(np.log10(m) - log_base))
    if nearest == 10:
        return float(10 ** (exponent + 1))
    return float(nearest * (10 ** exponent))


def _auto_grid(lo: float, hi: float, n: int, axis_name: str) -> list[float]:
    if n <= 1:
        raw = [(lo + hi) / 2]
    elif axis_name in LINEAR_AXES:
        raw = list(np.linspace(lo, hi, n))
    else:
        raw = list(np.geomspace(lo, hi, n))

    snapped = sorted({_nice_round(v) for v in raw})
    if axis_name in INT_AXES:
        snapped = sorted({int(round(v)) for v in snapped})
    return snapped


def _resolve_axis(method: str, axis_name: str, values_specs: dict[str, str],
                   range_specs: dict[str, str], grid_size: int | None) -> list[float] | None:
    spec = values_specs.get(method) or values_specs.get("all")
    if spec:
        return _parse_numbers(spec)

    default_vals = DEFAULT_GRIDS[method][axis_name]
    range_str = range_specs.get(method) or range_specs.get("all")
    if range_str:
        lo, hi = _parse_numbers(range_str)
        size = grid_size or len(default_vals)
        return _auto_grid(lo, hi, size, axis_name)

    if grid_size:
        return _auto_grid(min(default_vals), max(default_vals), grid_size, axis_name)

    return None


def resolve_method_grid(
    method: str,
    values1: list[str] | None = None,
    values2: list[str] | None = None,
    range1: list[str] | None = None,
    range2: list[str] | None = None,
    grid_spec: str | None = None,
) -> tuple[list[float], list[float] | None]:
    """
    Resolve the (axis1_values, axis2_values) grid for one screenable method.

    axis2_values is None for a deliberate 1D screen (axis1 only); the caller
    holds axis2 fixed at that method's own scale-preset default in that case.

    No overrides at all -> exact original hardcoded 2D grid, unchanged.
    """
    validate_screen_specs(
        values1=values1,
        values2=values2,
        range1=range1,
        range2=range2,
    )
    axis1_name, axis2_name = SCREEN_AXES[method]

    if not any([values1, values2, range1, range2, grid_spec]):
        return DEFAULT_GRIDS[method][axis1_name], DEFAULT_GRIDS[method][axis2_name]

    v1, v2 = _parse_specs(values1), _parse_specs(values2)
    r1, r2 = _parse_specs(range1), _parse_specs(range2)
    size1, size2 = _parse_grid_spec(grid_spec)

    axis1_vals = _resolve_axis(method, axis1_name, v1, r1, size1) or DEFAULT_GRIDS[method][axis1_name]

    axis2_touched = (method in v2 or "all" in v2 or method in r2 or "all" in r2
                      or size2 is not None)
    if not axis2_touched:
        return axis1_vals, None

    axis2_vals = _resolve_axis(method, axis2_name, v2, r2, size2) or DEFAULT_GRIDS[method][axis2_name]
    return axis1_vals, axis2_vals
