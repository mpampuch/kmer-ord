# How `--screen_params` Works

This document traces the `--screen_params` CLI option through the kmer-ord program: what functions and methods are called, how the parameter grid is resolved, and whether screening runs on all data or a subset.

## Where the Flag Lives

`--screen_params` is a boolean Typer option declared on three CLI commands in `src/kmer_ord/cli/main.py` (lines 136, 236, 624), alongside related grid-override flags:

- `--screen_values1` — explicit axis-1 values per method (`method=v1,v2,...` or `all=v1,v2,...`)
- `--screen_values2` — explicit axis-2 values per method
- `--screen_range1` — axis-1 range for auto-generated grids (`method=min,max`)
- `--screen_range2` — axis-2 range for auto-generated grids
- `--screen_grid` — grid size for auto-generated screens (`N` for 1D, `NxM` for 2D)

Each CLI command forwards these options into the workflow builder unchanged.

## Call Chain

### 1. CLI → Workflow Operation

The CLI passes `screen_params` (and the related override flags) to the `DimensionalityReduction` operation constructor, which stores them on `self`:

- `src/kmer_ord/workflow/operations.py` lines 180–198

### 2. DimensionalityReduction.run()

`DimensionalityReduction.run()`:

1. Loops over every preprocessed matrix (one per normalisation)
2. Loads the full `.npy` matrix into `X`
3. Calls `run_dr_methods(..., screen_params=self.screen_params, ...)`

See `src/kmer_ord/workflow/operations.py` lines 253–270.

### 3. run_dr_methods()

Inside `run_dr_methods` (`src/kmer_ord/dr/methods.py` line 157), for each requested DR method:

```python
if screen_params and method in SCREENABLE_METHODS:
    screen_dir = method_dir / "parameter_screen"
    screen_dir.mkdir(parents=True, exist_ok=True)

    _run_parameter_screen(
        X=X,
        method=method,
        dims=dims,
        seed=seed,
        scale=resolved_scale,
        output_dir=screen_dir,
        normalisation=normalisation,
        input_name=input_name,
        sequence_ids=sequence_ids,
        n_jobs=n_jobs,
        values1=screen_values1,
        values2=screen_values2,
        range1=screen_range1,
        range2=screen_range2,
        grid=screen_grid,
    )
```

See `src/kmer_ord/dr/methods.py` lines 229–249.

### 4. _run_parameter_screen()

`_run_parameter_screen` (`src/kmer_ord/dr/methods.py` line 303):

1. Calls `resolve_method_grid()` from `src/kmer_ord/dr/screen_grid.py` to turn override flags into `(axis1_vals, axis2_vals)`
2. Runs a nested loop over that grid
3. For each combination, calls the method's own estimator directly:
   - `umap.UMAP`
   - `sklearn.manifold.TSNE`
   - `trimap.TRIMAP`
   - `pacmap.PaCMAP`
   - `pacmap.pacmap.LocalMAP`

**Note:** Screening does **not** go through `_run_single_method()`.

### 5. Output and Visualization

For each parameter combination:

- `save_embedding()` writes a TSV with `sequence_id` and embedding columns
- Coordinates are accumulated in `density_combos`

After all combinations for a method finish:

- `render_param_screen_density_grid()` in `src/kmer_ord/vis/embedding_plots.py` draws one density grid per method

See `src/kmer_ord/dr/methods.py` lines 504–517.

## Grid Resolution

Grid resolution is handled by `resolve_method_grid()` in `src/kmer_ord/dr/screen_grid.py`.

### Default behavior (no override flags)

Returns the hardcoded 2D grid from `DEFAULT_GRIDS` for that method. Example for UMAP:

- `n_neighbors`: `[5, 10, 50, 100, 150]`
- `min_dist`: `[0, 0.1, 0.25, 0.5, 1.0]`

That is **5 × 5 = 25 fits** per method per normalisation.

### Override behavior

| Override type | Effect |
|---|---|
| `--screen_values1/2` | Use explicit comma-separated values |
| `--screen_range1/2` | Auto-generate values between min and max |
| `--screen_grid N` | Control number of auto-generated values (1D) |
| `--screen_grid NxM` | Control grid dimensions (2D) |

Auto-generated grids use:

- **Geometric spacing** for most axes
- **Linear spacing** for `min_dist` (listed in `LINEAR_AXES`)
- **Integer casting** for count-like axes (`n_neighbors`, `perplexity`, `n_inliers`, `learning_rate`)
- **Nice rounding** via `_nice_round()` (snaps to 1, 2, 5 multiples of powers of 10)

### 1D vs 2D screening

If axis 2 is never touched by any override flag, `resolve_method_grid()` returns `None` for axis 2. The screen becomes **1D**, and the second parameter is held fixed at that method's `DR_HYPERPARAMS[method]["default"]` value.

## Screenable Methods and Axes

Defined in `src/kmer_ord/dr/screen_grid.py` and `src/kmer_ord/dr/methods.py`:

| Method | Axis 1 | Axis 2 |
|---|---|---|
| `umap` | `n_neighbors` | `min_dist` |
| `tsne` | `perplexity` | `learning_rate` |
| `trimap` | `n_inliers` | `weight_temp` |
| `pacmap` | `n_neighbors` | `FP_ratio` |
| `localmap` | `n_neighbors` | `FP_ratio` |

```python
SCREENABLE_METHODS = {"umap", "tsne", "trimap", "pacmap", "localmap"}
```

Methods like `pca`, `sparse_pca`, `kernel_pca`, and `lle` are **not** screened even when `--screen_params` is set.

## Does It Run on All the Data?

**Yes — the full matrix, no subsampling.**

- `X` is the entire preprocessed matrix loaded in `operations.py` line 219
- It is passed through unchanged to every screening fit
- The only "subsetting" is on the **method axis**: non-screenable methods are skipped

## Important Behavioral Details

### Screening is additive, not a replacement

After parameter screening finishes for a method, the **default embedding still runs** for that method (`methods.py` line 260). So a default UMAP screen means **26 fits** (25 screen + 1 default), not 25.

### Cost scales with normalisations

The loop is multiplicative over normalisations:

```
3 normalisations × 5 screenable methods × 25 combinations = 375 full-dataset fits
```

### Memory check is coarse

`max_memory_gb` is checked once against `X.nbytes * 4` in `operations.py` lines 225–231 before any screening starts. This does not reflect the actual peak memory of UMAP/t-SNE neighbor graphs at large `n_neighbors` or `perplexity`.

### Caching can skip screening entirely

Caching in `operations.py` line 241 keys only on the merged default-embedding TSV existing. If that file is present, the whole DR operation is skipped — **including screening**. Adding `--screen_params` to a previously completed run does nothing unless you pass `--force`.

## Output Locations

For each screenable method and normalisation, outputs go under:

```
dr/<normalisation>/<method>/parameter_screen/
```

Per-combination embedding TSVs are named like:

```
<input_name>_<normalisation>_<method>_<param_str>_<dims>D.tsv
```

Example UMAP param strings:

- 1D: `n50`
- 2D: `n50_min0.1`

A density grid plot is also rendered in the same `parameter_screen/` directory after all combinations for that method complete.

## Quick Reference: Key Files

| File | Role |
|---|---|
| `src/kmer_ord/cli/main.py` | CLI option definitions |
| `src/kmer_ord/workflow/operations.py` | `DimensionalityReduction` operation; loads matrix, calls `run_dr_methods` |
| `src/kmer_ord/dr/methods.py` | `run_dr_methods`, `_run_parameter_screen`, `SCREENABLE_METHODS`, `DR_HYPERPARAMS` |
| `src/kmer_ord/dr/screen_grid.py` | `resolve_method_grid`, `DEFAULT_GRIDS`, `SCREEN_AXES` |
| `src/kmer_ord/vis/embedding_plots.py` | `render_param_screen_density_grid` |
