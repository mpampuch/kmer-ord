# What the Code Does

## Overview

**Primary implementation:** `src/kmer_ord/io/kmer_counter.py::run_kmer_counter`

The function runs an external Rust program, `kmer-counter`, via:

```
run_in_env(TOOLS_ENV, cmd)
```

It writes:

- a temporary `.npy` counts array
- a sequence-IDs file

After the Rust binary finishes, the Python wrapper:

1. Loads the `.npy` file using:

   ```Python
   numpy.load(..., mmap_mode='r').astype(np.uint32)
   ```

2. Extracts sequence headers from the FASTA using:

   ```Python
   Bio.SeqIO.parse
   ```

3. Constructs the canonical k-mer key list with `canonical_kmers(k)`, collapsing reverse complements.
4. Writes a TSV where:
   - the first column is `sequence_id`
   - each remaining column corresponds to a canonical k-mer
   - each row contains per-sequence numeric k-mer counts
5. Removes the temporary directory.
6. Returns the TSV path.

## Key Code Locations

- **Runner and orchestration:** `src/kmer_ord/workflow/operations.py::KmerCount`, which calls `run_kmer_counter`
- **Counting wrapper:** `src/kmer_ord/io/kmer_counter.py::run_kmer_counter`
- **Canonical k-mer generation:** `src/kmer_ord/io/kmer_counter.py::canonical_kmers`

---

# Is the K-mer Matrix Counts or Normalized?

The TSV produced by `run_kmer_counter` is a **raw count matrix**.

- Each row corresponds to one input sequence, such as a read or contig.
- Each column corresponds to a canonical k-mer.
- Each value is the integer count of that canonical k-mer observed in that sequence.

### Data Type

The array is loaded as:

```
uint32
```

The Python wrapper writes the array returned by the Rust tool directly to the TSV. **No sequence-length normalization occurs at this stage.**

## Implications for Read Length

Raw k-mer counts scale with sequence length:

- Longer sequences typically have larger counts across many k-mers.
- Shorter sequences typically have smaller total counts.

`kmer-ord` does **not** automatically normalize counts by read length during the k-mer counting step.

Length-related effects are handled later during matrix preprocessing.

---

# Normalization in `MatrixPreprocessing`

Normalization is applied later in:

```
src/kmer_ord/workflow/operations.py::MatrixPreprocessing
```

This component:

1. Loads the TSV via `dr.loader.load_matrix`.
2. Calls `dr.preprocess.preprocess_data`.

Available normalization methods include:

## `raw`

Leaves counts unchanged.

```
Per-sequence raw k-mer counts
```

## `relative`

Divides each row by its row sum:

```
X_ij / sum(X_i)
```

This produces per-sequence k-mer frequencies and helps normalize for:

- read length
- differing total k-mer counts per sequence

## `log`

Applies `log1p` to the counts:

```
log(1 + X)
```

This dampens large values.

## `clr`

Applies the centered log-ratio transform.

## `zscore`

Applies per-feature z-score normalization using `StandardScaler`.

Therefore, if you need a matrix adjusted for read length, use:

```
relative
```

or consider `clr` or `zscore` depending on the downstream analysis.

---

# What the CLR Transform Does

**Implementation:** `src/kmer_ord/dr/preprocess.py::preprocess_data`

The centered log-ratio, or **CLR**, transform is a compositional-data transformation.

Its purpose is to convert data constrained by a constant-sum relationship into real-valued coordinates that are more suitable for methods assuming unconstrained Euclidean geometry.

## Mathematical Operation

For:

```
method == "clr"
```

the implementation performs the following steps.

### 1. Add a Pseudocount

A tiny value is added to every entry to avoid taking the logarithm of zero:

```
X += 1e-9
```

### 2. Compute the Geometric Mean

For each row, compute:

```
g(x) = exp(mean(log(X_row)))
```

### 3. Apply the CLR Transform

For component `i`:

```
CLR_i = log(X_i / g(x))
```

Equivalently:

```
CLR_i = log(X_i) - mean_j(log(X_j))
```

The result is returned as a DataFrame where:

- rows = sequences
- columns = CLR-transformed k-mer features

---

# Practical Consequences of CLR

CLR expresses each component relative to the geometric mean of the sample.

As a result:

- values can be negative or positive
- each row is centered
- the transformed values sum to zero across components within each row

## Scale Invariance

CLR is scale-invariant.

If an entire row is multiplied by a constant, such as due to increased read length:

```
CLR(c × X) = CLR(X)
```

This is because the same multiplicative factor also appears in the row's geometric mean and cancels out.

Therefore, CLR can help mitigate differences driven by read length when counts scale approximately proportionally with sequence length.

## Zero Handling

The implementation adds a pseudocount:

```
1e-9
```

This prevents logarithms of zero.

However, the relative impact of the pseudocount depends on the original counts. For sparse or very short reads, very small counts and zeros may be strongly affected by this transformation, so interpretation requires care.

---

# Practical Guidance

- If you want embeddings or clustering to be less dependent on per-read length, use:
  ```
  --norm relative
  ```
  or:
  ```
  --norm clr
  ```
- `CLR` is often useful for compositional k-mer data.
- If you want to inspect the original length-sensitive count signal, use:
  ```
  --norm raw
  ```
- To reproduce the exact numeric column ordering produced during counting, inspect:
  ```
  canonical_kmers(k)
  ```
  The order generated by this function is used as the TSV header.
- The counting step collapses reverse complements by default using:
  ```
  --collapse 1
  ```
  This produces canonical k-mers.
- If strand-specific counts are required, a different mode of the Rust tool would be needed, if available, or the Python wrapper would need to be modified.

## Data passed into DR

The matrix fed into the dimensionality-reduction (DR) methods is whatever `MatrixPreprocessing` produces—not necessarily raw counts. By default, the CLI uses **CLR (centered log-ratio)** normalization, so DR normally receives **CLR-transformed, scale-invariant data**.

- `--norm raw` → raw counts
- `--norm relative` → per-row frequencies normalized by sequence total
- `--norm clr` → CLR-transformed data (**default**)

---

## Why / where this happens

### k-mer counting

k-mer counting produces a raw per-sequence count TSV via the Rust binary and Python wrapper:

- Wrapper: `src/kmer_ord/io/kmer_counter.py::run_kmer_counter`
- Output: sequence × k-mer integer counts (`uint32`)

### Matrix preprocessing

`MatrixPreprocessing` loads the count TSV and produces the numeric matrix or matrices used by DR:

- Loader: `src/kmer_ord/dr/loader.py::load_matrix`
- Preprocessing: `src/kmer_ord/dr/preprocess.py::preprocess_data`
- Orchestration: `src/kmer_ord/workflow/operations.py::MatrixPreprocessing`

For each requested normalization, `MatrixPreprocessing` saves a numeric `.npy` file:

```
output_dir/matrices/{matrix_stem}_{norm}.npy
```

### Dimensionality reduction

`DimensionalityReduction` reads the `.npy` matrix and passes it to the DR runner:

- `src/kmer_ord/workflow/operations.py::DimensionalityReduction`
- `src/kmer_ord/dr/methods.py::run_dr_methods`

This is where methods such as UMAP and t-SNE receive the preprocessed matrix.

---

## Available normalizations

Implemented in `src/kmer_ord/dr/preprocess.py::preprocess_data`:

- **`raw`** — leaves integer counts unchanged; therefore length-dependent.
- **`relative`** — divides each row by its sum, producing per-sequence k-mer frequencies and removing simple length/total-count differences.
- **`log`** — applies `log1p` to counts. This dampens large counts but does not remove length scaling.
- **`clr`** — centered log-ratio transformation:
  - Adds a small pseudocount (`1e-9`)
  - Computes the row geometric mean $g$
  - Returns $\log(x_i / g)$ for each element
  - Is scale-invariant
- **`zscore`** — applies per-feature `StandardScaler` normalization, centering and scaling features across sequences. This is not inherently length-invariant.

---

## Default CLI behavior

The pipeline default is **CLR**:

```
src/kmer_ord/cli/main.py::run_pipeline
```

The `dr` and `project` commands set:

```Python
normalisation: str = typer.Option("clr", "--norm", ...)
```

Therefore, unless you explicitly override `--norm`, the matrix passed to DR is the **CLR-transformed matrix**, not raw k-mer counts.

---

## Implications for read length

- **Raw counts:** scale approximately with read or contig length. Longer sequences generally have higher total k-mer counts.
- **Relative frequencies:** remove the total-count or length effect by converting each row to proportions.
- **CLR:** removes multiplicative scale. If an entire row is multiplied by a constant, the scaling cancels when each component is expressed relative to the row's geometric mean. It is therefore robust to length-related scaling differences.
- **`log1p` and `zscore`:** do not, by themselves, remove multiplicative length effects. `log1p` primarily reduces skew, while z-scoring standardizes features across sequences.

---

## How to force a particular input to DR

### Use raw counts

```Bash
kmer-ord project ... --norm raw
```

Or, when running DR from an existing matrix:

```Bash
kmer-ord dr -i path/to/matrix.tsv --norm raw
```

### Use relative frequencies

```Bash
--norm relative
```

### Use CLR

```Bash
--norm clr
```

This is also the default.

You can additionally request PCA pre-reduction before DR using:

```
--pca-pre
--keep-pcs
--keep-variance
--pca-pre-method {pca,ipca}
--pca-pre-batch-size N
```

`MatrixPreprocessing` supports these options before the final DR step.

Every `project` / `cluster` / `dr` run also writes `{output}/benchmarking/benchmark_log.tsv`: one parent row per pipeline stage (wall time + peak RSS) plus leaf rows for inner steps (k-mer-counter substeps, each normalisation / PCA-pre, each DR method). Nested rows have `parent_label` set to the outer stage.

---

## Quick code references

- **Loader:** `src/kmer_ord/dr/loader.py::load_matrix`
- **Preprocessing:** `src/kmer_ord/dr/preprocess.py::preprocess_data`
- **Matrix preprocessing orchestration:** `src/kmer_ord/workflow/operations.py::MatrixPreprocessing`
- **DR entry:**  
   `src/kmer_ord/workflow/operations.py::DimensionalityReduction`  
   → `src/kmer_ord/dr/methods.py::run_dr_methods`

## PCA pre-treatment

Passing `--pca-pre` causes the pipeline to run PCA on each preprocessed k-mer matrix and save the PCA scores instead of the full k-mer feature matrix — i.e., the number of feature columns is reduced (rows = sequences stay the same).

reduce_dimensions_with_pca is implemented in `src/kmer_ord/dr/preprocess.py`.
If `keep_pcs` is provided it fits PCA(`n_components=keep_pcs`).
If `keep_variance` is provided it first fits a full PCA to compute cumulative variance, selects the smallest number of PCs reaching the threshold, then fits PCA(`n_components=that_number`).
If neither `keep_pcs` nor `keep_variance` is provided `reduce_dimensions_with_pca` raises `ValueError` (the function requires one of them).

PCA pre-reduction reduces feature dimensionality (columns), which:
speeds up downstream DR (t-SNE / UMAP / PaCMAP) and clustering,
can denoise data and remove extremely high-dimensional sparsity before nonlinear DR.
You must specify how many PCs to keep:
Use --pca-pre plus either --keep-pcs N or --keep-variance 0.90 (or both). If you pass --pca-pre without either, the code will raise an error.

`--pca-pre-method pca` (default) is exact sklearn PCA and uses more RAM. `--pca-pre-method ipca` uses IncrementalPCA: it approximates the same subspace in row batches and is the right choice when the matrix is RAM-limited. Results match exact PCA when the data fits in a single batch. `--pca-pre-batch-size` is optional and only used with `ipca` (default `max(2048, 5 * n_components)`).

Example: kmer-ord project ... --pca-pre --keep-pcs 50
Or: kmer-ord project ... --pca-pre --keep-variance 0.95
Large-matrix / low-RAM recipe: kmer-ord project ... --pca-pre --keep-pcs 50 --pca-pre-method ipca
PCA is applied after the chosen normalisation (raw/relative/log/clr/zscore). So the PCA scores are of the normalised data.
