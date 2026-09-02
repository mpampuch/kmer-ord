# RAM optimization: DR methods vs. parameter screening

Short answer: focus on the DR methods themselves, **but** the screening script has one real accumulation bug you should fix regardless.

## Why screening is mostly fine

The screen is a strictly sequential loop. Each iteration rebinds `model` and `embedding`, so the previous fit's estimator (including UMAP's/PaCMAP's neighbor graphs and index structures, which are the bulk of the memory) loses its last reference and CPython frees it immediately via reference counting — you don't even wait for a generational GC pass. So peak RAM for a 25-combination screen is roughly the peak of **one** fit, not 25. Screening multiplies _time_ by 25, not memory.

That means the ceiling on your memory use is set by the single worst fit, which is a property of the method + hyperparameters, not of the screening machinery. Optimizing the methods (or the input matrix) is where the wins are.

## The exception: `density_combos` grows across the whole screen

```356:367:src/kmer_ord/dr/methods.py
    def save_embedding(embedding: np.ndarray, param_str: str):
        """Helper to save a DataFrame with sequence_id."""
        df = pd.DataFrame(embedding, columns=[f"{method}_{i+1}" for i in range(dims)])
        df.insert(0, "sequence_id", sequence_ids)
        out_file = output_dir / f"{input_name}_{normalisation}_{method}_{param_str}_{dims}D.tsv"
        df.to_csv(out_file, sep="\t", index=False)
        output_paths.append(out_file)
        return df

    def track_density(axis1_value: float, axis2_value: float, df: pd.DataFrame):
        if dims >= 2:
            density_combos.append((axis1_value, axis2_value, df))
```

Every combination's full DataFrame is retained until `render_param_screen_density_grid()` runs at the end. This is a genuine O(n_combos) memory cost, and it's worse than it looks because `df` carries the `sequence_id` column: `insert()` copies the string array into each DataFrame, and pandas stores those as an object column of Python `str` objects (~60-80 bytes each, versus 16 bytes for two float64 coordinates). For 1M sequences x 25 combos that's roughly 0.4 GB of coordinates plus several GB of duplicated ID strings.

Two cheap fixes:

- Append only the coordinate columns, not the ID column: `df[[f"{method}_1", f"{method}_2"]]` — or better, keep the raw `embedding` array (cast to `float32`) since the density plot only needs x/y.
- Or render each panel incrementally and keep only the 2D histogram counts per combination, which is O(bins) instead of O(n_seq).

## Where the real DR-side wins are

- `n_neighbors` **/** `perplexity` **scaling.** The neighbor graph is O(n_seq x k). The `large` presets use `n_neighbors: 200`, and `DEFAULT_GRIDS` screens up to 150. Peak RAM is roughly linear in that value, so the top of the grid is what sets the ceiling — trimming the grid's upper end is the single easiest lever.
- **t-SNE with** `n_jobs > 1`**.** sklearn's Barnes-Hut allocates per-thread working buffers, so threads trade RAM for speed. Same for UMAP's numba parallel paths.
- **The memory guard is misleading.** `operations.py:225-231` estimates peak as `X.nbytes * 4`, which ignores the neighbor graph entirely and is checked once before screening begins. It will happily let a `n_neighbors=150` UMAP OOM you.

## How big is the retained DataFrame, really?

It's the **embedding**, not the k-mer matrix. Each `save_embedding()` call builds an `n_seq x (1 + dims)` table: one `sequence_id` column plus `dims` coordinate columns (`umap_1`, `umap_2`). That's the same thing written to each `*_n5_min0.1_2D.tsv` file. `track_density()` appends that same object to `density_combos`, so at the end of a 25-combination UMAP screen, 25 of these tables are simultaneously alive.

Per-row cost, for `dims=2` with float32 coordinates:

- Numeric part: **8 bytes per row per combination**. Negligible.
- `sequence_id`: the expensive part. `np.load` gives a fixed-width `<U` array, and `insert()` converts it to an object column — a fresh Python `str` per element, per DataFrame. For ~40-character read names that's ~90-100 bytes/row (49-byte `str` header + characters + 8-byte pointer), and the conversion happens independently on every call, so nothing is shared between the 25 copies.

So: ~8 bytes/row of coordinates actually needed, riding on ~95 bytes/row of redundantly duplicated ID strings.

Compared to the k-mer matrix — a canonical 6-mer matrix is 2080 features (4096 non-canonical), and `preprocess_data` already casts to float32 (`preprocess.py:13`), so the matrix is ~8.3 KB/row. For 1M reads:

|                                          | per row | 1M reads |
| ---------------------------------------- | ------- | -------- |
| k-mer matrix (6-mer, canonical, float32) | 8.3 KB  | 8.3 GB   |
| One embedding's coordinates              | 8 B     | 8 MB     |
| One embedding incl. `sequence_id`        | ~95 B   | 95 MB    |
| 25 retained embeddings                   | ~2.4 KB | ~2.4 GB  |

The accumulation is real in absolute terms but it's roughly a quarter to a third of the matrix, not a multiple of it. (The "several GB" figure earlier in this note is right in absolute terms but overstates the severity relative to the matrix.) Worth fixing because it's nearly free, but not the bottleneck.

Which term dominates depends on the regime: the accumulation scales with `n_combos x n_seq` while the matrix scales with `n_seq x 4^k`. At k=6 the matrix dominates; at k=10+ the matrix is ~500 KB/row and the embeddings vanish into the noise. The accumulation only becomes dominant with a very large grid (`--screen_grid 10x10` = 100 combos ~ 9.5 GB at 1M reads) at small k.

## The fix, concretely

The renderer only ever touches the two coordinate columns:

```406:409:src/kmer_ord/vis/embedding_plots.py
    for (v1, v2), df in by_pos.items():
        df_local = df[[xcol, ycol]].dropna()
        if df_local.empty:
            continue
```

So `sequence_id` is retained for every combination purely because `save_embedding` returns the same DataFrame it wrote to disk and `track_density` stores it wholesale. Slice the coordinate columns off before appending:

```python
    def track_density(axis1_value: float, axis2_value: float, df: pd.DataFrame):
        if dims >= 2:
            # keep only the coordinate columns the density renderer reads —
            # retaining the full frame would hold one duplicated object-dtype
            # sequence_id column per grid combination for the whole screen
            coords = df[[f"{method}_1", f"{method}_2"]].copy()
            density_combos.append((axis1_value, axis2_value, coords))
```

Why it works: column selection on a mixed-dtype frame returns a **new** float-only block rather than a view, so the retained object no longer references the `sequence_id` array (the `.copy()` is belt-and-braces). Once the full `df` goes out of scope at the end of each iteration, its ~95 bytes/row of `str` objects is freed immediately. Only ~8 bytes/row survives to the end of the screen — at 1M reads x 25 combinations, ~2.4 GB down to ~8 MB.

Nothing downstream breaks: `render_param_screen_density_grid` is called with `xcol=f"{method}_1"`, `ycol=f"{method}_2"` (`methods.py:513-514`), exactly what's kept. The per-combination TSVs still get `sequence_id`, because `save_embedding` writes to disk before returning — only what's held in RAM changes.

Tidiness caveat: the column names are constructed in three places (`save_embedding`, `track_density`, the `render_...` call site). Worth hoisting into locals near the top of `_run_parameter_screen` so they can't drift apart.

## Conclusion

Fix `density_combos` because it's a small, contained change with a large payoff at scale, then spend optimization effort on the neighbor-count parameters rather than on the screening loop's structure. (Note: there's no dtype win available in preprocessing — `preprocess_data` already casts to float32.) The thing that actually sets peak RAM is neither the matrix nor the embeddings but UMAP's/PaCMAP's internal neighbor graph, which is O(n_seq x n_neighbors) and at `n_neighbors=150` over 1M reads is on the order of the matrix itself, on top of the matrix still being resident.
