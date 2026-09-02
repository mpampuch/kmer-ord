# Change Log: RAM Optimization with Robust Benchmarking

Date: 2026-09-01

All changes below were verified by the 54-test suite (every optimization reproduces the original numerical output within float tolerance) and logged in `benchmarks/benchmark_log.tsv`. The `matrix-preprocessing-RAM-notes.md` modification predates this session (it was already dirty in git status).

---

## Phase 0a — `src/kmer_ord/utils/benchmark.py` (rewritten)

**Problem.** The old `BenchmarkTimer` recorded `end_rss - start_rss`. That is nearly useless for memory work, for two reasons:

1. It misses the peak. Numeric pipelines allocate large temporaries and free them before the block exits, so a stage that spiked to 20 GB and freed it reports ~0.
2. It can't see child processes. The `ProcessPoolExecutor` workers in k-mer stats and the Rust k-mer counter subprocess allocate in *their own* address space, invisible to the parent's RSS.

**Change.**

- A `_PeakRssSampler` background thread polls `psutil.Process().memory_info().rss` plus all recursive children every 50 ms and keeps the maximum. It samples once at start (so instant blocks still get a value) and once at stop (to catch exit-time state).
- Extra cross-check column from `resource.getrusage(RUSAGE_SELF).ru_maxrss`, normalized to bytes (macOS reports bytes, Linux reports KB) — a kernel-side sanity check on the sampler; it covers process lifetime, so it can legitimately exceed the per-block peak.
- New TSV schema: `timestamp, git_commit, script_name, stage_label, input_file, input_file_size_bytes, input_rows, input_cols, input_args, wall_time_s, cpu_time_s, peak_rss_self_bytes, peak_rss_children_bytes, end_rss_bytes, ru_maxrss_bytes`.
- `git_commit` is resolved relative to the module's own source tree (not the caller's cwd), with a `-dirty` suffix when the working tree is modified — every logged row identifies exactly which code produced it. That's the key that makes commit-to-commit comparison possible.
- `record_input_shape(rows, cols)` lets a stage attach matrix dimensions after loading, so log rows are interpretable ("peak per row" analysis).
- If an existing `benchmark_log.tsv` has the legacy header, it is rotated to `benchmark_log_legacy_<timestamp>.tsv` rather than appended to — mixing schemas in one TSV would corrupt it.
- The context-manager API and constructor signature were preserved, so the six existing `BenchmarkTimer` call sites in `io/kmer_counter.py` work unchanged — and the Rust counter subprocess they wrap now actually shows up in `peak_rss_children_bytes`.

**Tests** (`tests/test_benchmark.py`): allocation freed before block exit must still appear in the peak; a child-process allocation must appear in `peak_rss_children`; schema/row correctness; legacy-log rotation; append behavior.

---

## Phase 0b — `benchmarks/run_benchmarks.py` (new)

**Rationale.** The goal was a *robust, repeatable* way to prove each change reduces RAM. Ad-hoc measurements aren't comparable across commits; this makes the measurement procedure fixed and the inputs deterministic.

**Design decisions.**

- **Per-stage subprocess isolation.** Each stage runs in a fresh Python process. In a shared process, allocator high-water marks and non-returned RSS from earlier stages contaminate later measurements; isolation makes each number attributable to one stage.
- **Tiered datasets** (decided at planning): `--tier small` uses a seeded synthetic Poisson count matrix (default 10,000 x 2,080 — the shape of a canonical 6-mer matrix) cached under `benchmarks/data/`; `--tier full` uses the real 3.9 GB `62_Coelastrummicroporum` matrix. The generator writes in 5,000-row blocks so generating a large synthetic matrix never itself holds the matrix in RAM, and it's byte-deterministic per seed — benchmarks across commits are only comparable if the input is identical (this is unit-tested).
- **Four stages**: `kmer_stats`, `preprocess_clr`, `pca_pre` (CLR + `keep_pcs=50`, mirroring the recommended `--pca-pre --keep-pcs 50` recipe), and `dr_pca` (2-D PCA through the DR dispatch — chosen over UMAP so the small tier stays fast and deterministic).
- **`compare` subcommand** diffs the latest row per stage between two commits (or the two most recent distinct commits in the log) and prints peak-RAM and wall-time deltas as percentages. Intended loop: change -> `run --tier small` -> commit -> `compare`.

---

## Phase 0c — Test suite (new `tests/`, `pyproject.toml`)

**Rationale.** The repo had no tests. Optimizing memory while silently changing scientific output would be worse than the memory problem, so every numerical behavior was frozen *before* being touched.

- `pyproject.toml`: added a `dev` optional-dependency group (`pytest`) and `[tool.pytest.ini_options] testpaths = ["tests"]`.
- `tests/test_preprocess_golden.py`: contains verbatim frozen copies of the *original* implementations (`reference_clr`, `reference_relative`, `reference_zscore`, `reference_variance_pca`, ...) and asserts the package functions match them at `rtol=1e-5, atol=1e-6` (the same tolerances the CLR benchmark script in `my-notes/CLR-optimizations` used). Also pins: index/columns preservation, float32 dtype, input non-mutation (critical because the pipeline reuses the loaded matrix across normalisations), and edge cases (all-zero row in `relative`).
- `tests/test_kmer_stats_golden.py`: a float64 reference of the original metric math, with the corrected column names. Written red-first (TDD) before Phase 1.

---

## Phase 1 — `src/kmer_ord/io/kmer_stats.py` (rewritten)

**Problem** (from the memory audit, confirmed in code): `chunks = list(reader)` materialized every chunk, defeating chunking entirely; `to_numpy(dtype=float)` then converted to float64 (4x the uint16 footprint); with `cpus > 1` all chunks were submitted to the pool at once, adding serialization copies; all results were retained for a final `pd.concat`; and four metric names didn't describe what they measured.

**Changes and rationale.**

| Change | Why |
|---|---|
| Iterate the reader lazily; write each chunk's metrics to the TSV immediately | Peak RAM becomes ~one chunk instead of the whole matrix |
| Parallel mode: bounded deque, at most `cpus` futures in flight | Parallel RAM cost is `cpus` chunk-copies, not the whole matrix + copies |
| Read counts as `uint32`, not `uint16` | The Rust counter emits u32; the u16 downcast could silently overflow at small k on long reads (the audit's exact concern) |
| Compute in `float32` with `float64` accumulators for row reductions (`.sum(axis=1, dtype=np.float64)`) | Output keeps ~3 decimals so float64 element arrays are pure waste; float64 *accumulators* keep integer row sums exact and entropy sums stable at zero extra array cost |
| In-place buffers: counts -> probabilities via `values /= row_sums` on our own array; one reused `plogp` buffer with `where=nonzero_mask`; bits derived as `nats / ln(2)` | Cuts per-chunk temporaries from ~5 chunk-sized arrays to ~2; log(0) never evaluated so no errstate suppression needed; bits/nats are the same quantity in different units so one log pass suffices |
| Renamed metrics: `total_kmer_counts`, `num_nonzero_kmers`, `shannon_entropy_nats`, `shannon_entropy_bits`, new `shannon_evenness` = H/ln(S) | Old names claimed things the values weren't (e.g. `total_nonzero_kmers` was the count *sum*; `shannon_evenness` was raw entropy). New `shannon_evenness` is true Pielou evenness, defined as 1.0 for single-category rows. Nothing downstream hardcoded the old names — `FeatureMerge` passes columns through generically |
| `RunningStats` class (Chan et al. batched Welford) for the dataset summary | The old summary needed the full `combined_metrics` in memory; running accumulators give identical mean/sd(ddof=1)/min/max from the stream (unit-tested against numpy) |
| Returns the output `Path`, not a DataFrame; `output_file` now required; unused `total_rows` param and dead `dropna/fillna` removed | Returning the full table would silently defeat the streaming; uint32 parsing makes NaN handling unreachable code |

**Key test**: `test_streaming_memory_bounded` generates a 50,000 x 200 matrix (~40 MB materialized) and asserts via `tracemalloc` that processing it with 2,000-row chunks peaks under 12 MB — the old implementation cannot pass this.

**Measured**: small tier 1,017 -> 624 MB; full tier (882,730 reads) peaks at **2.9 GB** vs the ~3x matrix size (~22 GB) the notes estimated for the old code.

---

## Phase 2 — `src/kmer_ord/dr/loader.py`, `src/kmer_ord/dr/preprocess.py`, wiring in `operations.py` / `cli/main.py`

### `load_matrix`

- TSV/CSV numeric columns are parsed **directly into float32** via a positional dtype map (`{0: str, rest: float32}`), and the old `df.apply(pd.to_numeric, errors="raise")` validation pass was deleted. Rationale: the old flow parsed to int64/float64 (2x footprint), then `apply` made another full copy, then `preprocess_data` made a float32 copy — the largest object in the pipeline existed in RAM up to three times. Now it exists once, at final dtype, and non-numeric data still fails loudly at parse time (unit-tested).

### `preprocess_data`

- Allocates exactly **one** output-sized float32 buffer (`df.to_numpy(dtype=np.float32, copy=True)` — the copy is mandatory because the input must not be mutated, which is regression-tested) and every method operates on it in place. The returned DataFrame wraps the buffer without copying.
- **CLR** uses the log-difference identity verified in `my-notes/CLR-optimizations/`: `X += 1e-9; np.log(X, out=X); X -= X.mean(axis=1, keepdims=True)`. Mathematically identical to `log(x/gmean(x))` (the benchmark showed max abs diff ~1e-7 and ~50% less peak allocation) with zero full-matrix temporaries, vs. four in the old form.
- `relative` uses in-place broadcast division; `log` uses `np.log1p(X, out=X)`; `zscore` passes `StandardScaler(copy=False)` so sklearn scales our buffer instead of allocating another.

### `reduce_dimensions_with_pca`

- **Variance-threshold bug fix**: the old code called `pca_full.fit_transform(df.values)` and threw the transformed matrix away — it only needed `explained_variance_ratio_`. Now `pca_full.fit(...)`, eliminating a full n x n_components float64 allocation. The second fit at the chosen component count is unchanged, so output is bit-comparable (golden test confirms).
- **New `method="ipca"`** (IncrementalPCA): fits in row batches, then transforms in batches into a preallocated float32 output — no full-matrix float64 workspace ever exists. For `--keep-variance`, it fits a capped component count (<=500, far beyond any realistic cumulative-variance cutoff for k-mer matrices), reads the spectrum, then slices to the threshold. `batch_size` is clamped to >= n_components because sklearn requires it. Tests pin the single-batch case to exact-PCA equality (mathematically the same SVD) and the multi-batch case to determinism + variance capture.

### Wiring

- `MatrixPreprocessing` gained `pca_method` and passes it through; `--pca-pre-method {pca,ipca}` added to all three CLI commands (`project`, `cluster`, `dr`). Default is `pca` so existing invocations behave identically.

**Measured**: `preprocess_clr` 920 -> 470 MB, `pca_pre` 934 -> 557 MB on the small tier; full tier peaks at ~16 GB = matrix (7.3 GB) + exactly one working copy + parse buffers, matching the plan's "one copy" target.

---

## Phase 3 — `src/kmer_ord/dr/methods.py` (`_run_parameter_screen`)

**Problem** (from `screen-params-RAM-notes.md`): `track_density` appended each combination's full DataFrame to `density_combos`, which lives until the density grid renders after the whole screen. Each frame carries an object-dtype `sequence_id` column — a fresh Python `str` per row per combination (~95 bytes/row) — while the renderer only ever reads the two coordinate columns (~8 bytes/row). At 1M reads x 25 combos that's ~2.4 GB retained for no reason.

**Changes.**

- `track_density` now keeps `df[coord_cols[:2]].copy()` — the `.copy()` detaches the float block from the parent frame so the full frame (and its string column) is freed at the end of each loop iteration. Retained cost drops from ~95 to ~8 bytes/row.
- Coordinate column names were constructed independently in three places (`save_embedding`, `track_density`, the render call) and could drift; they're now hoisted into a single `coord_cols` local.
- The per-combination TSVs on disk still contain `sequence_id` — only what's held in RAM changed (regression-tested).

**Test**: runs a real 1x1 t-SNE screen (sklearn-only, no numba warm-up) with the renderer monkeypatched, asserting the retained frames contain only float coordinate columns while the on-disk TSV keeps `sequence_id`. Written red-first against the old code.

---

## Phase 4 — memory guard (`dr/methods.py` + `workflow/operations.py`) and deferred-items note

**Problem.** The guard `est_peak = X.nbytes * 4` only looked at the input matrix. After `--pca-pre` shrinks X to ~200 MB, a `n_neighbors=200` UMAP or `FP_ratio=5` PaCMAP fit on 1M reads allocates multi-GB neighbor/pair structures the guard never sees — it would approve runs that then OOM, which is worse than no guard because it inspires false confidence.

**Changes.**

- New `estimate_peak_memory_gb(n_seq, n_feat, method, scale)` next to `DR_HYPERPARAMS`, encoding the audit's cost models: sparse fuzzy graph O(n x k) x2 for UMAP (NN-descent scratch), float64 copy + n x 3*perplexity P-matrix for t-SNE, n x k x (1 + MN + FP) pair tables for PaCMAP/LocalMAP, n x (n_inliers*5 + 5) triplets for TriMAP, float64 working copy for PCA. Byte constants are deliberately generous — it's a guard, not a profiler. Unknown methods fall back to the old x4 multiplier so the guard never weakens.
- The `DimensionalityReduction` check resolves the scale preset, expands `"all"`, estimates every requested method, and raises `MemoryError` naming the worst method and its estimate. Only active when `--max-memory` is set, exactly as before — results unchanged.
- Tests pin that `large`-preset estimates for all five manifold methods exceed the old heuristic in the exact regime it failed (1M reads x 50 PCs).
- **`my-notes/deferred-dr-preset-optimizations.md`** records the result-changing items deferred by decision at planning, with rationale and recommended values: preset caps (UMAP 200 -> 15-50 neighbors etc.), `low_memory=True`, `apply_pca=False` after `--pca-pre`, non-spectral UMAP init, `n_jobs=1` for RAM-limited runs, subprocess-per-method (numba/sklearn don't return RSS to the OS), and precomputed kNN for screening.

### Also in `operations.py`

- `MatrixPreprocessing.run` and `DimensionalityReduction.run` were the only untimed heavy stages; both now wrap their work in `BenchmarkTimer` (per normalisation for DR) and record matrix shape. The DR cache check was moved *before* the matrix load so a cached run doesn't pay the load or log a misleading benchmark row.

---

## Housekeeping

- `.gitignore`: added `benchmarks/data/` (generated fixtures; the log itself stays tracked so commits can be compared).
- Editable install (`pip install -e . --no-deps`) into `kmerord-env` — the env previously held a static site-packages copy, so source edits wouldn't have been what tests exercised.

## Results summary

| Stage | Small tier before -> after | Full tier (883k reads x 2080) |
|---|---|---|
| kmer_stats | 1017 -> 624 MB | 2.90 GB (vs ~22 GB est. before) |
| preprocess_clr | 920 -> 470 MB | 16.2 GB (matrix + one copy) |
| pca_pre | 934 -> 557 MB | 16.8 GB |
| dr_pca | 1010 -> 620 MB | 16.1 GB |

Wall times also dropped 30-45% across stages. All 54 tests green; every stage's outputs are numerically identical to the pre-optimization implementation.

## Going forward

All of this session's benchmark rows share one dirty commit hash. Once this work is committed, the loop becomes:

1. Make a change
2. `python benchmarks/run_benchmarks.py run --tier small`
3. Commit
4. `python benchmarks/run_benchmarks.py compare` — shows the per-stage peak-RAM/wall-time delta between the two most recent commits
5. Milestone validation: `python benchmarks/run_benchmarks.py run --tier full` (~5 min, peaks ~17 GB)
