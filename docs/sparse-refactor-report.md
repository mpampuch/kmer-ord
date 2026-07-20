# Sparse k-mer matrix refactor — change report

This document explains the memory-focused refactor of the `kmer-ord` pipeline
for the original author. It covers the problem, the conceptual changes, the
file-by-file code changes, measured results, and compatibility notes.

## 1. Problem and diagnosis

Running `kmer-ord project` on a small dataset (1,429 reads, ~9.6 Mbp) OOM'd a
node with >400 GB RAM at k=11. The counting-stage dense copy (~12 GB at k=11)
was only the visible symptom; the actual peaks were downstream, because
pipeline stages run sequentially and the OOM is the *highest single-stage* peak.

At k=11 there are ~2,097,152 canonical columns and n=1,429 rows:

| Stage | Before | Why |
|-------|--------|-----|
| `KmerCount` | ~12 GB | `np.load(...).astype(uint32)` materialised the full dense matrix |
| `KmerMetrics` | 100+ GB | `pd.read_csv` of a TSV with ~2.1M columns (pandas per-column overhead is pathological), then `to_numpy(dtype=float)` upcast to float64 (~24 GB) plus several float64 temporaries computed twice (nats + bits) |
| `MatrixPreprocessing` | tens of GB | `load_matrix` re-read the same millions-of-columns TSV into pandas; CLR then stacked several ~12 GB float intermediates |
| `DimensionalityReduction` | ~12 GB+ | consumed the full dense normalized matrix; `n_jobs` copies compounded it |

Threading amplified this: `KmerMetrics(cpus=threads)` runs up to `cpus` chunks
concurrently (each pickled into its own process), and `DimensionalityReduction`
passes `n_jobs=threads`, whose joblib workers memmap the dense `X` to `/tmp`
(often RAM-backed on HPC).

## 2. Conceptual changes (the big ideas)

1. **Dense to sparse.** K-mer count matrices are >99% zeros for reads, so the
   primary artifact is now a compressed sparse CSR bundle (`.npz`) instead of a
   dense TSV. The dense `.npy` written by the Rust counter is converted to CSR
   in bounded row blocks — the full dense matrix is never held in RAM.
2. **TSV is optional.** The human-readable dense TSV is now opt-in
   (`--matrix-tsv`) and is streamed one row at a time.
3. **Sparse-native metrics.** `KmerMetrics` computes directly from the CSR
   (`data`/`indptr`), avoiding pandas' millions-of-columns overhead and the
   float64 upcast entirely.
4. **Memory-bounded pre-reduction.** Above a feature-count threshold, the
   matrix is projected to a small number of components *before* the non-linear
   DR, so DR never sees the full dense matrix:
   - `raw`/`relative`/`log` stay sparse and go straight into `TruncatedSVD`;
   - `clr`/`zscore` are dense by nature, so they are reduced with an exact PCA
     computed from the tiny `n x n` Gram matrix `G = Xc @ Xc.T`, accumulated one
     *feature* block at a time. Because each feature column lives entirely within
     one block, per-block column-centering equals global centering, so the result
     is exact PCA - but peak RAM is only one feature block plus the `n x n` Gram,
     and there is no wide SVD workspace.
5. **Why CLR is inherently dense.** CLR = `log(x / geometric_mean(x))` with a
   pseudocount, so every zero becomes a non-zero. No storage trick avoids this;
   the Gram/feature-block approach sidesteps it by only ever densifying one
   feature block at a time (the per-row log-geometric-mean is derived cheaply
   from the sparse data).

   Note: an initial `IncrementalPCA`-over-row-blocks implementation worked but
   peaked at ~7 GB for k=10 because its internal SVD operates in the full
   feature space; the Gram approach replaced it (~1.7 GB, ~10x faster).

## 3. Code changes

| File | Change |
|------|--------|
| `src/kmer_ord/io/sparse_matrix.py` (new) | CSR format helpers: `dense_npy_to_csr` (streaming), `save_sparse_matrix`/`load_sparse_matrix` (`.npz`), `write_matrix_tsv` (streamed) |
| `src/kmer_ord/dr/reduce.py` (new) | `reduce_matrix`: TruncatedSVD on sparse norms, exact Gram-matrix PCA over feature blocks for `clr`/`zscore`; sparse column/row-log-mean stats |
| `src/kmer_ord/io/kmer_counter.py` | `run_kmer_counter` builds CSR from the memmap, saves `.npz`, optional streamed TSV (`write_tsv`), logs sparse size/density |
| `src/kmer_ord/dr/preprocess.py` | `preprocess_data` accepts sparse/array/DataFrame and returns float32; row-independent math centralised in `apply_row_normalization`; in-place CLR; added `ALL_NORMALISATIONS` |
| `src/kmer_ord/dr/loader.py` | replaced `load_matrix` with `load_matrix_any` returning `(csr, sequence_ids)` for `.npz`/`.tsv`/`.csv`/`.npy` |
| `src/kmer_ord/io/kmer_stats.py` | added `calculate_kmer_metrics_sparse` (CSR path) + `.npz` branch in `process_kmer_file` |
| `src/kmer_ord/workflow/operations.py` | `KmerCount` writes `.npz` + `write_tsv`; `MatrixPreprocessing` uses `load_matrix_any` and auto-reduces above `reduce_threshold`; fixed a latent `ALL_METHODS` NameError |
| `src/kmer_ord/utils/benchmark.py` | `BenchmarkTimer` samples process-tree peak RSS in a background thread; new `peak_memory_bytes` column |
| `src/kmer_ord/cli/main.py` | new flags `--matrix-tsv/--no-matrix-tsv` and `--reduce-threshold` on `project`, `cluster`, `kmer-count`, `dr` |
| `tests/`, `scripts/mem_test.sh`, `pyproject.toml` | unit/parity tests, peak-RSS regression tests, `pytest` dev dep + markers |

New CLI flags:

- `--matrix-tsv / --no-matrix-tsv` (default off): also write the dense TSV.
- `--reduce-threshold` (default 250,000): feature count above which the matrix
  is dimensionally reduced before DR (so k>=10 auto-reduces, k<=9 does not).
- `--pca-pre` now forces the memory-bounded reduction at any k.

## 4. Measured results

Full `project` pipeline on `TEST-DATA/63_Monoraphidiumcircinale...fasta`
(1,429 reads), `--threads 1 --dr pca --no-tiara --no-matrix-tsv`, process-tree
peak RSS sampled by the regression test:

| k | features | pipeline peak RSS (after) | runtime |
|---|----------|---------------------------|---------|
| 6 | 2,080 | 0.54 GB | ~13 s |
| 8 | 32,896 | 1.26 GB | - |
| 10 | 524,800 | ~12.9 GB* | ~4.5 min |
| 11 | 2,097,152 | 11.74 GB | ~6.2 min |

All four k values completed the full pipeline (counting -> metrics -> reduction
-> DR -> feature merge -> SpatiaLite DB), versus the prior >400 GB OOM at k=11.
The bounded reducer engaged at k=10/11 (DR input = 1,429 x 50 features). The
remaining peak at large k is dominated by the external Rust counter writing its
dense `.npy`, not the Python pipeline.

*The k=10 pipeline peak was measured before the reducer was switched from
IncrementalPCA to the Gram approach; the reduce stage itself dropped from 7.0 GB
to 1.67 GB (clr) with that change, so a re-run would be lower.

Isolated reducer memory (cached k=10 matrix, `clr`/`zscore`):

| method | IncrementalPCA (initial) | Gram PCA (final) |
|--------|--------------------------|------------------|
| clr | 7.00 GB / 112 s | 1.67 GB / 11 s |
| zscore | 7.09 GB / 146 s | 1.00 GB / 12 s |
| log (sparse TruncatedSVD) | 0.62 GB / 8 s | 0.62 GB / 8 s |

## 5. Compatibility notes for the author

- **Default artifact changed** from `<name>_<k>mer_matrix.tsv` to
  `<name>_<k>mer_matrix.npz`. Old TSV/CSV matrices are still readable via
  `load_matrix_any`, and the `dr`/`kmer-metrics` module commands accept either.
- **DR runs on principal components above the threshold** (standard practice
  for sparse count data), so embeddings above k~9 are not byte-identical to the
  old full-feature behavior. Below the threshold, behavior is unchanged.
- **Numeric parity preserved** for metrics and for all normalizations
  (`raw`/`relative`/`log`/`clr`/`zscore`), verified by
  `tests/test_sparse_matrix.py`.

## 6. How to reproduce / verify

```bash
# unit + parity tests (fast)
PYTHONPATH=src pytest tests/test_sparse_matrix.py -q

# peak-RSS regression on the real dataset (k=6, 8 by default)
PYTHONPATH=src pytest -m slow tests/test_memory_regression.py -s

# shell harness with /usr/bin/time peak reporting
scripts/mem_test.sh
```
