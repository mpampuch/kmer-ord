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


def _expand_semicolon_entries(entries: list[str] | None) -> list[str]:
    """Split 'localmap=0.1,1.0;pacmap=0.5,2.0' into one spec per method.

    Empty pieces are dropped, so a trailing semicolon is harmless. The shell
    still needs quotes around the value: ';' separates commands when bare.
    """
    expanded: list[str] = []
    for entry in entries or []:
        for piece in str(entry).split(";"):
            piece = piece.strip()
            if piece:
                expanded.append(piece)
    return expanded


def _parse_specs(entries: list[str] | None, *, semicolons: bool = False) -> dict[str, str]:
    """['all=5,320', 'umap=10,300'] -> {'all': '5,320', 'umap': '10,300'}.

    With semicolons=True, one flag value may hold several method= specs.
    A later piece, or a later repeated flag, replaces an earlier one.
    """
    source = _expand_semicolon_entries(entries) if semicolons else (entries or [])
    out = {}
    for entry in source:
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


def named_screen_methods(
    values1: list[str] | None = None,
    values2: list[str] | None = None,
    range1: list[str] | None = None,
    range2: list[str] | None = None,
    grid: str | None = None,
) -> list[str]:
    """Methods explicitly named in screen specs, in first-seen order.

    'all' and a bare grid size apply to whatever --dr runs, so they are omitted.
    """
    names: list[str] = []
    seen: set[str] = set()

    def add(key: str) -> None:
        method = key.strip().lower()
        if not method or method == "all" or method in seen:
            return
        seen.add(method)
        names.append(method)

    for entries in (values1, values2, range1, range2):
        for piece in _expand_semicolon_entries(entries):
            add(piece.partition("=")[0])
    if grid is not None and str(grid).strip():
        for piece in _expand_semicolon_entries([grid]):
            if "=" not in piece:
                continue
            add(piece.partition("=")[0])
    return names


def validate_screen_specs(
    values1: list[str] | None = None,
    values2: list[str] | None = None,
    range1: list[str] | None = None,
    range2: list[str] | None = None,
    grid: str | None = None,
) -> None:
    """Fail on screen specs that would be ignored during grid resolution."""
    for flag, entries in (
        ("--screen_values1", values1),
        ("--screen_values2", values2),
        ("--screen_range1", range1),
        ("--screen_range2", range2),
    ):
        for entry in entries or []:
            # One flag value may hold several method= specs joined by ';'.
            pieces = _expand_semicolon_entries([entry])
            if str(entry).strip() and not pieces:
                _validate_screen_entry(str(entry).strip(), flag)
                continue
            for piece in pieces:
                _validate_screen_entry(piece, flag)
    if grid is not None and str(grid).strip():
        _parse_grid_specs(grid)


def _parse_numbers(csv: str) -> list[float]:
    return [float(v) if ("." in v or "e" in v.lower()) else int(v)
            for v in csv.split(",") if v.strip()]


def _parse_grid_size(size: str, shown: str) -> tuple[int, int | None]:
    """'5' -> (5, None). '5x4' -> (5, 4). Counts must be positive integers."""
    text = size.strip().lower()
    form = "'N' or 'NxM'"
    if "x" in text:
        left, right = text.split("x", 1)
        if not left.strip() or not right.strip() or "x" in right:
            raise ValueError(f"--screen_grid {shown!r} size {size!r} must be {form}")
        try:
            n1 = int(left)
            n2 = int(right)
        except ValueError as exc:
            raise ValueError(f"--screen_grid {shown!r} size {size!r} must be {form}") from exc
        if n1 < 1 or n2 < 1:
            raise ValueError(
                f"--screen_grid {shown!r} size {size!r} needs counts greater than zero"
            )
        return n1, n2
    try:
        n1 = int(text)
    except ValueError as exc:
        raise ValueError(f"--screen_grid {shown!r} size {size!r} must be {form}") from exc
    if n1 < 1:
        raise ValueError(
            f"--screen_grid {shown!r} size {size!r} needs a count greater than zero"
        )
    return n1, None


def _parse_grid_specs(spec: str | None) -> dict[str, tuple[int, int | None]]:
    """Resolve --screen_grid into per-method (axis1, axis2) counts.

    '5x4' and 'all=5x4' size every method. 'localmap=5x4;pacmap=3x3' sizes
    only those methods. A bare size mixed into a named list is rejected;
    use 'all=5x4;localmap=3x3' when some methods override a shared size.
    Axis2 is None for a 1D size ('5' or 'method=5').
    """
    if not isinstance(spec, str):
        return {}
    text = spec.strip()
    if not text:
        return {}

    pieces = [piece.strip() for piece in text.split(";") if piece.strip()]
    if not pieces:
        raise ValueError(
            f"--screen_grid {text!r} must be 'N', 'NxM', or 'method=NxM;method=NxM'"
        )

    named = ["=" in piece for piece in pieces]
    if any(named) and not all(named):
        raise ValueError(
            f"--screen_grid {text!r} mixes a bare size with method=size. "
            "Use 'NxM' alone for every method, or name each one "
            "('localmap=5x4;pacmap=3x3'). Use 'all=5x4' to set a default "
            "and override some methods."
        )
    if not any(named):
        if len(pieces) != 1:
            raise ValueError(
                f"--screen_grid {text!r} has more than one bare size. "
                "Use 'NxM' alone for every method, or name each one "
                "('localmap=5x4;pacmap=3x3')."
            )
        # A bare size is the shared default, same as writing all=NxM.
        return {"all": _parse_grid_size(pieces[0], text)}

    allowed = (*SCREEN_AXES, "all")
    out: dict[str, tuple[int, int | None]] = {}
    for piece in pieces:
        key, _, raw_size = piece.partition("=")
        method = key.strip().lower()
        size = raw_size.strip()
        if not method or not size:
            raise ValueError(
                f"--screen_grid {piece!r} must be 'method=N' or 'method=NxM'"
            )
        if method not in allowed:
            names = ", ".join(allowed)
            raise ValueError(
                f"--screen_grid method {method!r} is not screenable; "
                f"use one of {names}."
            )
        # Repeated methods: the rightmost entry wins, same as a later flag.
        out[method] = _parse_grid_size(size, text)
    return out


def _grid_size_for(
    method: str, sizes: dict[str, tuple[int, int | None]]
) -> tuple[int | None, int | None]:
    """Method entry replaces 'all' as a whole, so method=5 does not keep all's axis 2."""
    if method in sizes:
        return sizes[method]
    if "all" in sizes:
        return sizes["all"]
    return None, None


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


def _distinct_readable(value: float, used: set[float]) -> float:
    """A short decimal that is not already in used.

    1-2-5 snapping can map two interior points onto one landmark. The grid
    size is a count of combinations, so the spare point keeps its own value.
    """
    for digits in range(3, 15):
        candidate = float(f"{float(value):.{digits}g}")
        if candidate not in used:
            return candidate
    return float(value)


def _nearest_free_int(seed: int, used: set[int]) -> int:
    delta = 0
    while delta < 100_000:
        candidates = (seed,) if delta == 0 else (seed - delta, seed + delta)
        for candidate in candidates:
            if candidate >= 1 and candidate not in used:
                return candidate
        delta += 1
    raise ValueError(f"cannot place {len(used) + 1} distinct integer screen values")


def _claim_screen_value(preferred, source: float, used: set) -> float:
    """Prefer the nice number. If it is already taken, keep a distinct neighbour."""
    if isinstance(preferred, (int, np.integer)) and not isinstance(preferred, bool):
        number = int(preferred)
        if number not in used:
            used.add(number)
            return number
        number = _nearest_free_int(int(round(source)), used)
        used.add(number)
        return number

    number = float(preferred)
    if number not in used:
        used.add(number)
        return number
    number = _distinct_readable(source, used)
    used.add(number)
    return number


def _values_with_requested_count(raw: list[float], snapped: list) -> list[float]:
    """Return len(raw) screen values. Ends claim their snap before interior points.

    A set of snapped values used to drop collisions, so --screen_grid 5x5 could
    run 5×4 combinations. UMAP min_dist on [0, 1] sends both 0.75 and 1.0 to 1,
    and PaCMAP FP_ratio on [0.1, 1] has only four 1-2-5 landmarks.
    """
    if len(set(snapped)) == len(raw):
        return sorted(set(snapped))

    used: set = set()
    chosen: list[float | None] = [None] * len(raw)
    order = []
    if raw:
        order.append(0)
    if len(raw) > 1:
        order.append(len(raw) - 1)
    order.extend(range(1, len(raw) - 1))
    for index in order:
        chosen[index] = _claim_screen_value(snapped[index], float(raw[index]), used)
    return sorted(chosen)


def _auto_grid(lo: float, hi: float, n: int, axis_name: str) -> list[float]:
    if n <= 1:
        raw = [(lo + hi) / 2]
    elif axis_name in LINEAR_AXES:
        raw = list(np.linspace(lo, hi, n))
    else:
        raw = list(np.geomspace(lo, hi, n))

    snapped = [_nice_round(v) for v in raw]
    if axis_name in INT_AXES:
        snapped = [int(round(v)) for v in snapped]
    return _values_with_requested_count(raw, snapped)


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

    --screen_grid '5x4' (or 'all=5x4') sizes every method.
    'localmap=5x4;pacmap=3x3' sizes only those methods; every other method
    keeps DEFAULT_GRIDS unless its own values or ranges say otherwise.
    A named size replaces 'all' for that method, including a 1D 'method=5'.

    Value and range flags accept the same ';' joining, and still accept a
    repeated flag. A named spec overrides 'all'.
    """
    validate_screen_specs(
        values1=values1,
        values2=values2,
        range1=range1,
        range2=range2,
        grid=grid_spec,
    )
    axis1_name, axis2_name = SCREEN_AXES[method]
    grid_text = grid_spec.strip() if isinstance(grid_spec, str) else ""
    grid_text = grid_text or None

    if not any([values1, values2, range1, range2, grid_text]):
        return DEFAULT_GRIDS[method][axis1_name], DEFAULT_GRIDS[method][axis2_name]

    v1 = _parse_specs(values1, semicolons=True)
    v2 = _parse_specs(values2, semicolons=True)
    r1 = _parse_specs(range1, semicolons=True)
    r2 = _parse_specs(range2, semicolons=True)
    grid_sizes = _parse_grid_specs(grid_text)
    size1, size2 = _grid_size_for(method, grid_sizes)
    # 'all' or this method's own entry. A spec that only names other methods
    # must not resize this one.
    grid_applies = method in grid_sizes or "all" in grid_sizes
    axis1_size = size1 if grid_applies else None

    default_axis1 = DEFAULT_GRIDS[method][axis1_name]
    default_axis2 = DEFAULT_GRIDS[method][axis2_name]
    axis1_vals = _resolve_axis(method, axis1_name, v1, r1, axis1_size) or default_axis1

    axis2_explicit = method in v2 or "all" in v2 or method in r2 or "all" in r2
    # Bare 'N' / 'method=N' is a 1D screen unless this axis has its own values or range.
    if grid_applies and size2 is None and not axis2_explicit:
        return axis1_vals, None

    if axis2_explicit or (grid_applies and size2 is not None):
        axis2_size = size2 if grid_applies else None
        axis2_vals = _resolve_axis(method, axis2_name, v2, r2, axis2_size) or default_axis2
        return axis1_vals, axis2_vals

    # Named grids for other methods used to fall through to the 1D return
    # below, because any grid_spec made size2 look global. Keep the built-in
    # 2D grid instead. An axis-1-only values/range override is still 1D.
    if grid_sizes and not grid_applies:
        axis1_explicit = method in v1 or "all" in v1 or method in r1 or "all" in r1
        if not axis1_explicit:
            return default_axis1, default_axis2

    return axis1_vals, None
