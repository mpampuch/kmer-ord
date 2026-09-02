# Deferred DR-stage optimizations (result-changing)

The RAM-optimization pass (see `benchmarks/benchmark_log.tsv` for before/after
numbers) deliberately excluded every change that would alter the scientific
output of the embeddings. This note records those deferred items, from
`dr-stage-RAM-notes.md` and the ppt audit, so they aren't lost. Each one
trades embedding results (usually imperceptibly) for large RAM/time savings
and should be benchmarked with `benchmarks/run_benchmarks.py` plus a visual
comparison of embeddings before adoption.

## 1. Cap the `large` / `medium` scale presets (biggest lever)

`_resolve_scale` maps ≥50k sequences to `large`, which *raises* neighbor
counts far beyond library defaults (`DR_HYPERPARAMS` in `dr/methods.py`):

| method | knob | library default | current `large` |
|---|---|---|---|
| UMAP | `n_neighbors` | 15 | 200 |
| t-SNE | `perplexity` | 30 | 200 |
| TriMAP | `n_inliers` | 10 | 150 |
| PaCMAP | `n_neighbors` / `FP_ratio` | 10 / 2 | 200 / 5 |
| LocalMAP | `n_neighbors` / `FP_ratio` | 10 / 0.5 | 200 / 1 |

Neighbor graphs and pair tables are O(n x k), so the datasets that already
stress RAM get the most expensive k. Recommended caps: neighbors/perplexity
~15–50, PaCMAP `FP_ratio` 2. `n_neighbors` controls locality, not "more
neighbors because n is big".

## 2. UMAP `low_memory=True`

umap-learn defaults this to False; True switches NN-descent to a smaller,
slower mode. Results can differ slightly (approximate kNN search mode).

## 3. `apply_pca=False` for TriMAP/PaCMAP/LocalMAP after `--pca-pre`

These libraries default to an *internal* PCA-to-100 on input >100 dims,
which allocates a float64 copy of X. After `--pca-pre --keep-pcs <=100` the
internal PCA is redundant; disabling it skips that copy. (No-op change when
the input is already <=100-d, result-changing otherwise.)

## 4. UMAP `init="pca"` / `"random"` instead of default `"spectral"`

Spectral init eigen-decomposes the graph Laplacian — a known RAM/time spike
at ~1M points. Changes the layout initialization, hence the final embedding.

## 5. `n_jobs=1` for RAM-limited runs

sklearn t-SNE (Barnes-Hut) and UMAP's numba paths allocate per-thread working
buffers; the CLI default is 4 threads. Same results (up to run-to-run
nondeterminism), trades speed for RAM — a docs/flag-default decision.

## 6. Subprocess per DR method

Numba/sklearn do not return freed RSS to the OS; running several methods in
one process stacks allocator high-water marks. One subprocess per method
actually releases the address space between fits. No result change, but a
structural change to `DimensionalityReduction` — deferred as out of scope.

## 7. Precomputed kNN for screening

UMAP `min_dist` (and PaCMAP `FP_ratio` at fixed `n_neighbors`) don't change
the kNN graph. Computing kNN once per `n_neighbors` value and passing
`precomputed_knn=...` (UMAP) / pair arrays (PaCMAP) would cut screen *time*
dramatically at identical peak RAM. Results identical in principle, but the
plumbing differs per library version — verify before relying on it.
