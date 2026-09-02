# Reducing RAM at the dimensionality-reduction stage

Focus: PCA, t-SNE, UMAP, TriMAP, PaCMAP, and LocalMAP as implemented in `src/kmer_ord/dr/methods.py` and invoked from `DimensionalityReduction` in `src/kmer_ord/workflow/operations.py`.

Peak RAM at this stage is dominated by **keeping the full matrix `X` in RAM** plus **one method’s neighbor/pair structures** — and the `large` presets make those structures as expensive as the matrix itself. Screening is mostly a time multiplier, not a memory one (with one wrapper exception).

## What is actually resident

`DimensionalityReduction.run` loads the whole preprocessed matrix, then never lets go of it:

```219:222:src/kmer_ord/workflow/operations.py
            X = np.load(matrix_path)

            # Load corresponding sequence IDs
            sequence_ids = np.load(seqid_paths[i])
```

Every method then does `fit_transform(X)` on that same object. Methods run **sequentially**, so you pay for `X` plus **one** fit, not all six at once. The memory guard (`X.nbytes * 4`) only looks at `X` and ignores neighbor graphs, pair tables, float64 copies, and `n_jobs` scratch buffers.

For a canonical 6-mer matrix (~2080 features, float32):

| Structure | 1M reads |
|---|---|
| `X` (already in RAM) | ~8.3 GB |
| sklearn float64 copy of `X` (PCA / t-SNE) | another ~8.3 GB |
| UMAP/PaCMAP graph at `n_neighbors=200` | on the order of `X` |

The manifold methods cannot stream `X` the way IncrementalPCA can. They need the rows for kNN. The real levers are: **shrink `X` before those methods see it**, **stop inflating `n_neighbors`/`perplexity` with dataset size**, and **stop libraries from building extra copies**.

---

## Highest-impact lever you already have: `--pca-pre`

`--pca-pre` / `--keep-pcs` runs in `MatrixPreprocessing`, *before* this stage. It is off by default. For UMAP / t-SNE / TriMAP / PaCMAP / LocalMAP this is the single biggest RAM (and time) cut:

- 1M × 2080 float32 ≈ **8.3 GB**
- 1M × 50 PCs float32 ≈ **200 MB**

kNN then runs in 50-d instead of 2080-d. TriMAP and PaCMAP also default to an **internal** PCA-to-100, so you currently risk **your `X` + a second PCA copy** unless you already reduced.

Use something like `--pca-pre --keep-pcs 50` (or 30–100). That is the standard recipe these libraries assume. Do **not** use `--keep-variance` for this: `reduce_dimensions_with_pca` fits a **full** PCA to get the cumulative variance, then fits again — that is the worst PCA path in the repo, and it belongs to preprocessing, not the 2-D visualisation PCA.

---

## The auto scale presets work against RAM

```14:57:src/kmer_ord/dr/methods.py
def _resolve_scale(scale: str, n_seq: int) -> str:
    ...
    if n_seq < 50_000:
        return "medium"
    return "large"
```

`auto` maps ≥50k sequences to `large`, which **raises** neighbor counts:

| | UMAP `n_neighbors` | t-SNE `perplexity` | TriMAP `n_inliers` | PaCMAP `n_neighbors` / `FP_ratio` | LocalMAP `n_neighbors` / `FP_ratio` |
|---|---|---|---|---|---|
| library defaults | 15 | 30 | 10 | 10 / 2 | 10 / 0.5 |
| your `large` | **200** | **200** | **150** | **200 / 5** | **200 / 1** |

Neighbor graphs and pair tables are **O(n × k)**. The datasets that already stress RAM get the most expensive `k`. UMAP’s `n_neighbors` controls *locality*, not “use more neighbors because n is big.” For large n, **15–30 is the usual choice**; 200 is a visualisation preference that roughly 10× the graph.

Order-of-magnitude extras on 1M points, on top of `X`:

- **UMAP** `n_neighbors=200`: sparse fuzzy graph ~`n × k` after symmetrisation — multiple GB, plus NN-Descent scratch.
- **t-SNE** `perplexity=200`: sparse P-matrix ~`n × 3·perplexity` (~600 neighbors/point).
- **PaCMAP** `n_neighbors=200`, `FP_ratio=5`, `MN_ratio=0.5`: ~`n × k × (1 + MN + FP)` ≈ **1.3e9 pairs**. Pair index arrays alone can be **~10 GB**.
- **LocalMAP** `large` is better (`FP_ratio=1`) but still ~4 GB of pairs at k=200.
- **TriMAP** `n_inliers=150` with default `n_outliers=5`, `n_random=5`: ~755 triplets/point → **~7.5e8 triplets** (~9 GB of int32 indices).

Practical fix: cap the `large` (and `medium`) neighbor-like params, or stop using `scale="auto"` on big matrices. `--scale default` is already much closer to library defaults.

---

## Per-method changes

### PCA (the 2-D method in `_run_single_method`, not preprocessing)

```93:96:src/kmer_ord/dr/methods.py
    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=dims, **params)
        embedding = model.fit_transform(X)
```

sklearn `PCA` always wants **float64** and `copy=True` by default. For `dims=2` that copy is the main extra cost; the 2-D embedding itself is tiny.

- Set `svd_solver="randomized"` (or `"covariance_eigh"` on sklearn ≥ 1.4 when `n_samples >> n_features`, which is the 6-mer regime). `"auto"` already tends to pick randomized when `n_components` is tiny, so this is a smaller win than people expect.
- **IncrementalPCA does not help here unless you stop loading all of `X`.** You already have `X` in RAM for the other methods. IncrementalPCA only pays off as a **memmap + batch** path in preprocessing, or if PCA is the *only* method and `X` is `np.load(..., mmap_mode="r")`.
- `copy=False` only avoids a copy if `X` is already C-contiguous float64 — yours is float32, so sklearn will allocate the float64 copy anyway.

If PCA is run *alongside* UMAP/t-SNE, run PCA first (cheap), then the manifold methods on the same `X`; do not keep the PCA estimator.

### t-SNE

```98:101:src/kmer_ord/dr/methods.py
    elif method == "tsne":
        from sklearn.manifold import TSNE
        model = TSNE(n_components=dims, random_state=seed, n_jobs=n_jobs, **params)
```

- Keep Barnes-Hut (`method="barnes_hut"` is the default). Never `method="exact"` — that is N×N.
- **`perplexity` is the RAM knob.** `large` uses 200; 30–50 is the usual range. sklearn uses ~`3 * perplexity` neighbors.
- **`n_jobs > 1` buys speed with per-thread buffers.** CLI default is 4 threads. For RAM, `n_jobs=1`.
- `init="pca"` (your preset) runs an extra PCA on full `X`. Fine after `--pca-pre`; on a raw 6-mer matrix it is another float64 spike. `init="random"` avoids it.
- sklearn TSNE has no `apply_pca` for the *neighbor* space. Neighbors are on whatever you pass. Passing 50 PCs is the real reduction; `init="pca"` only initialises the 2-D layout.

### UMAP

```103:108:src/kmer_ord/dr/methods.py
    elif method == "umap":
        ...
        model = umap.UMAP(n_components=dims, random_state=None, n_jobs=n_jobs, **params)
```

- Set **`low_memory=True`**. umap-learn still defaults this to **False**; it switches NN-Descent to a smaller (slower) mode. It does **not** stream `X` off disk.
- Cut **`n_neighbors`** (200 → 15–50).
- Default **`init="spectral"`** eigen-decomposes the graph Laplacian. At 1M points that is a known RAM/time spike. `init="pca"` or `"random"` avoids it. umap-learn sometimes falls back to random at very large n, but it is better to set it.
- `n_jobs=1` for RAM (numba parallel paths).
- After `fit_transform`, you keep `model.graph_`. That sparse matrix is ~`n × n_neighbors` and is written to disk; then the estimator can die. If downstream clustering does not need it, skip extracting it.

Screening note: `min_dist` does not change the kNN graph. The nested loop refits from scratch for every `(n_neighbors, min_dist)` pair. That does not raise *peak* RAM (one fit at a time), but you can compute kNN once per `n_neighbors` and pass `precomputed_knn=...` for every `min_dist`. Same peak, much less time, and you avoid stacking NN-Descent indexes if the allocator is sticky.

### TriMAP

```110:113:src/kmer_ord/dr/methods.py
    elif method == "trimap":
        from trimap import TRIMAP
        model = TRIMAP(n_dims=dims, **params)
```

- **`n_inliers`** is the RAM knob (your `large` is 150; default is 10).
- Default **`apply_pca=True`** runs sklearn PCA to 100-d *inside* TriMAP. That is a float64 copy of `X` unless you already `--pca-pre`. If you pre-reduce, pass `apply_pca=False`.
- You never set `n_outliers` / `n_random` (defaults 5 / 5). Triplets scale as ~`n × (n_inliers × n_outliers + n_random)`. Raising inliers without cutting outliers is how you get multi-GB triplet arrays.
- `knn_tuple=(indices, distances)` lets you build kNN once (or share with UMAP) and skip the internal search.

### PaCMAP and LocalMAP

```115:125:src/kmer_ord/dr/methods.py
    elif method == "pacmap":
        ...
        model = PaCMAP(n_components=dims, **params)
    elif method == "localmap":
        ...
        model = LocalMAP(n_components=dims, **params)
```

These are the most sensitive to your presets because they **materialise near + mid-near + far pairs**:

`pairs ≈ n × n_neighbors × (1 + MN_ratio + FP_ratio)`

- PaCMAP `large`: `200 × (1 + 0.5 + 5) = 1300` pairs/point.
- Treat **`n_neighbors` and `FP_ratio` as a memory budget**, not quality sliders you max out together. `n_neighbors=10–30`, `FP_ratio=2` is the library default for a reason.
- Same **`apply_pca=True` → 100-d** story as TriMAP. Disable it if `--pca-pre` already ran.
- Both accept precomputed neighbor/pair arrays (`pair_neighbors`, `pair_MN`, `pair_FP`, or a kNN tuple depending on version). Useful for screening `FP_ratio` at fixed `n_neighbors`.

---

## Wrapper issues that still matter at this stage

**1. `density_combos` during `--screen-params`.**
Each combo’s full DataFrame (including a duplicated object-dtype `sequence_id` column) is held until the density grid is drawn. At 1M reads × 25 combos that is ~2 GB of redundant ID strings. Keep only the two float columns (or a float32 `embedding` array). This is documented in `screen-params-RAM-notes.md` and is still unfixed in `methods.py`.

**2. `n_jobs` / threads.**
CLI default is 4. sklearn t-SNE and UMAP/numba allocate per-thread working space. For a RAM-limited run, `--threads 1`.

**3. Numba / sklearn do not return RSS to the OS.**
After a UMAP fit, Python may free the objects while resident set size stays high; the next method then allocates on top. If you run several methods in one process, consider **one subprocess per method** so the address space is actually released. Sequential `del model; gc.collect()` is not enough for RSS.

**4. `np.load` without `mmap_mode`.**
`mmap_mode="r"` only helps IncrementalPCA (or a kNN library that pages rows). UMAP/t-SNE/PaCMAP/TriMAP will fault the whole map in. Not worth it unless PCA is the only method.

**5. The `X.nbytes * 4` guard.**
It will not catch a `n_neighbors=200` PaCMAP OOM. A useful bound is closer to:

- `X + O(n × k × (1 + MN_ratio + FP_ratio))` for PaCMAP
- `X + O(n × 3·perplexity)` for t-SNE
- plus a float64 copy of `X` for sklearn PCA/t-SNE

---

## What to change, in order

1. **`--pca-pre --keep-pcs 50`** (or similar) before any manifold method. Biggest cut, already wired.
2. **Stop auto-scaling neighbor counts up.** Cap `large` at ~15–50 neighbors / perplexity ~30–50; cap PaCMAP `FP_ratio`.
3. **UMAP: `low_memory=True`, `init="pca"` or `"random"`, `n_jobs=1`.**
4. **TriMAP/PaCMAP/LocalMAP: `apply_pca=False` if `--pca-pre` already ran;** otherwise leave it on and accept the PCA spike once.
5. **t-SNE: keep Barnes-Hut, lower perplexity, `n_jobs=1`.**
6. **Fix `density_combos`** so screening does not retain `sequence_id` copies.
7. **PCA as 2-D output:** randomized / `covariance_eigh` is enough; IncrementalPCA only if you memmap and PCA is the sole method.
8. If you still OOM with 1–2 methods: **subprocess per method**, and/or subsample for screening then one full fit.

None of these require dropping PCA, t-SNE, UMAP, TriMAP, PaCMAP, or LocalMAP. The algorithms are already O(n × k), not O(n²). The current wrapper plus the `large` presets are what make them look like they need 4× the matrix.

## Related notes

- `my-notes/screen-params-RAM-notes.md` — screening loop vs. `density_combos` accumulation
- `my-notes/RAM-report.md` — broader method-by-method memory-safety report (includes KernelPCA, SparsePCA, LLE)
- `my-notes/matrix-preprocessing-RAM-notes.md` — preprocessing copies, CLR, IncrementalPCA for `--pca-pre`
