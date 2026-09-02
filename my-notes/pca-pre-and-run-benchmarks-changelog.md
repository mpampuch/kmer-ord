# Change Log: PCA-pre Method Polish and Per-run Nested Benchmarks

Date: 2026-09-02

Follow-on to `ram-optimization-changelog.md`. That session shipped IncrementalPCA inside `reduce_dimensions_with_pca` and wired `--pca-pre-method {pca,ipca}` through `project` / `cluster` / `dr`. This session makes that flag a real CLI choice, exposes the IncrementalPCA batch-size knob, and makes **every pipeline run** write a nested (parent + leaf) wall-time / peak-RAM log under `--output`. The standalone harness `benchmarks/run_benchmarks.py` still appends to `benchmarks/benchmark_log.tsv`.

Verified by the test suite (`tests/`) and a live `python benchmarks/run_benchmarks.py run --tier small` that migrated the existing log in place and appended four new rows.

---

## Phase 1 — CLI: `--pca-pre-method` Choice and `--pca-pre-batch-size`

**Problem.** `--pca-pre-method` accepted any string and failed later inside `reduce_dimensions_with_pca`. IncrementalPCA's `batch_size` existed on the Python API (`max(2048, 5 * n_components)` default, clamped to `>= n_components`) but `MatrixPreprocessing` never passed it, so there was no way to tune the RAM/accuracy tradeoff from the CLI.

**Change.**

- `click.Choice(["pca", "ipca"])` on `project`, `cluster`, and `dr` so `--help` lists the values and invalid input (`--pca-pre-method nope`) errors at parse time.
- New `--pca-pre-batch-size` (optional `int`, default `None` = internal default). Wired as `MatrixPreprocessing.pca_batch_size` → `reduce_dimensions_with_pca(..., batch_size=...)`. Ignored when `--pca-pre-method pca` (exact PCA has no batches); does not error.
- Default remains `pca` so existing invocations are unchanged.

**Tests** (`tests/test_pca_pre.py`): CliRunner rejects unknown methods; `--help` lists `pca` / `ipca` and `--pca-pre-batch-size`; `method="pca"` with a `batch_size` still matches exact PCA (the argument is ignored). Existing ipca single-batch / multi-batch / variance-threshold / float32 tests unchanged.

**Recipe.**

```bash
kmer-ord project -i reads.fastq -o out --pca-pre --keep-pcs 50 --pca-pre-method ipca
```

---

## Phase 2 — Per-run log location and `parent_label`

**Problem.** `BenchmarkTimer` defaulted to cwd-relative `benchmarking/benchmark_log.tsv`. A `kmer-ord project -o RESULTS/...` run mixed every launch in that directory into one file, and the RESULTS directory itself had no benchmark artifact. Nested timers (k-mer-counter substeps inside `KmerCount`) also had no way to record which parent stage they belonged to.

**Change.**

- `Context` / `MatrixContext` / `DBContext` share `_HasBenchmarkDir`: `benchmark_dir` is `{output_dir}/benchmarking/`, and `benchmark_timer(...)` injects that path plus `script_name` (`project` / `cluster` / `dr`).
- New TSV column `parent_label`. Nested `BenchmarkTimer` uses a `contextvars.ContextVar` so a timer entered inside another records the outer `stage_label` automatically; a top-level timer writes `N/A`. Call sites do not pass the parent name.
- Pipeline operations use `context.benchmark_timer(...)` instead of a bare `BenchmarkTimer(...)` so they cannot forget the per-run path. `run_kmer_counter` and `run_dr_methods` take `log_dir` / `script_name` so their inner rows land in the same file.

Target path for a projection run:

`{output_dir}/benchmarking/benchmark_log.tsv`

**Tests** (`tests/test_benchmark.py`): nested outer+inner produce two rows with `parent_label` set correctly; `MatrixContext.benchmark_timer` writes under `{output}/benchmarking/`, not cwd.

---

## Phase 3 — Parent timer per stage and leaf timer per inner step

**Problem.** Several heavy stages were untimed (FASTA convert in `Context.__init__`, `FeatureMerge`, `SpatialiteDatabase`, clustering). Dimensionality reduction logged **one combined row** for all `--dr` methods; `run_dr_methods` already printed per-method `perf_counter` times to the console but did not write them to the log. Matrix preprocessing logged CLR and `--pca-pre` as a single block.

**Change.** Both layers are required. Do **not** also wrap at `Runner` — that would add a third copy of the same work.

**Parents** (operation-level, `parent_label=N/A`), including previously untimed work:

- `fasta_convert`, `fastq_to_fasta`, `fasta_stats`, `kmer_count_{k}mer`, `kmer_metrics`, `tiara`, `rdna`, `matrix_preprocessing`, `dimensionality_reduction_{norm}`, `feature-merge`, `spatialite-db`, `clustering`, `add_clustering_to_db`

**Leaves** (nested, `parent_label` = the parent above):

- k-mer counter substeps (already existed): `Kmer_Counter_Run`, `Numpy_Loading`, `Sequence_Headers_Extraction`, `Canonical_Kmers_Generation`, `TSV_Composition`, `Cleanup` — now inherit `kmer_count_{k}mer` as parent
- `preprocess_{norm}` and, if `--pca-pre`, `pca_pre_{norm}`
- `dr_load_{norm}`, then `dr_{norm}_{method}` around `_run_single_method` (replaces console-only `perf_counter`)
- `--screen_params`: each combination is `dr_screen_{norm}_{method}_{param_str}`
- SpatiaLite: `db_fasta`, `db_features`, `db_coordinates`
- clustering: `cluster_{method}_{dr_method}`

For `--dr umap,tsne,trimap,pacmap,localmap,pca` the DR parent covers load + all methods; each method is its own leaf.

**Tests**: `test_run_dr_methods_writes_parent_and_leaf_rows` wraps `run_dr_methods(..., methods=["pca"])` in a parent timer and asserts both `dimensionality_reduction_clr` and `dr_clr_pca` rows, with the leaf pointing at the parent.

---

## Phase 4 — Standalone harness still updates `benchmarks/benchmark_log.tsv`

**Problem.** Adding `parent_label` made the existing `benchmarks/benchmark_log.tsv` header mismatch `LOG_COLUMNS`. The old `_rotate_legacy_log` would have renamed that file to `benchmark_log_legacy_<stamp>.tsv` and started a fresh log — breaking `run_benchmarks.py compare` across the RAM-optimization rows.

The harness itself was already correct: `run_stage_in_this_process` passes `log_dir` explicitly (`DEFAULT_LOG_DIR = benchmarks/`), stages call `reduce_dimensions_with_pca` / `_run_single_method` (not `run_dr_methods`), so they do not spawn pipeline nested timers.

**Change.** `_rotate_legacy_log` became `_prepare_log_file`:

- Header already matches → no-op, append.
- Old header is a **subset** of `LOG_COLUMNS` (this session: missing `parent_label`) → rewrite in place, fill missing fields with `N/A`, then append. History stays in the same filename.
- Unknown / extra columns → still rotate to `benchmark_log_legacy_<stamp>.tsv`.

**Tests.**

- `test_compatible_schema_upgraded_in_place`: previous schema (no `parent_label`) is rewritten in place; old row kept; new row appended; no legacy file.
- `test_old_format_log_rotated_not_corrupted`: a header with `legacy_col` still rotates.
- `test_run_stage_appends_to_explicit_log_dir`: `run_stage_in_this_process("preprocess_clr", ...)` writes `bench_small_preprocess_clr` to the given `--log-dir` with `script_name=run_benchmarks` and `parent_label=N/A`.

**Measured** (live `run --tier small` after the migration, 10k × 2080 synthetic):

| Stage | Peak (self+children) | Wall |
|---|---|---|
| `bench_small_kmer_stats` | 624.4 MB | 1.08 s |
| `bench_small_preprocess_clr` | 459.7 MB | 1.64 s |
| `bench_small_pca_pre` | 543.5 MB | 1.88 s |
| `bench_small_dr_pca` | 636.9 MB | 3.68 s |

Old rows (including `bench_small_kmer_stats_og`) remain in `benchmarks/benchmark_log.tsv` with `parent_label=N/A`. No `benchmark_log_legacy_*.tsv` was created.

---

## Docs

- `README.md`: short usage example with `--pca-pre --keep-pcs 50 --pca-pre-method ipca`, and a note that each run writes `{output}/benchmarking/benchmark_log.tsv`.
- `my-notes/kmer-counting.md`: documents `pca` vs `ipca`, when to pick `ipca`, `--pca-pre-batch-size`, and the per-run nested log.
- `my-notes/ppt_notes_dump.md`: the `--pca-pre-method (pca, ipca)` idea marked as implemented.

---

## Out of scope (unchanged)

- 2-D DR method `pca` in `methods.py` is still exact sklearn PCA, not IncrementalPCA.
- DR preset RAM items in `deferred-dr-preset-optimizations.md` (`apply_pca=False` after `--pca-pre`, neighbor caps, `low_memory=True`, …).
- DRY-refactor of the duplicated PCA flags across the three CLI commands.

---

## Going forward

Pipeline runs: inspect `{output}/benchmarking/benchmark_log.tsv`. Filter `parent_label=N/A` for stage totals; filter `parent_label=<stage>` for the breakdown (e.g. which of umap/tsne/… dominated RAM).

Commit-to-commit RAM loop is unchanged:

1. Make a change
2. `python benchmarks/run_benchmarks.py run --tier small`
3. Commit
4. `python benchmarks/run_benchmarks.py compare`
5. Milestone: `python benchmarks/run_benchmarks.py run --tier full`
