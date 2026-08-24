# My diagnosis

From my diagnosis, it seems like: `./src/kmer_ord/workflow/operations.py` might be the main memory problem because `matrix = load_matrix(matrix_path)` loads the entire k-mer matrix into memory.

1. Check if this is necessary for all downstream DR technique libraries
2. Check also if RUST program that makes the k-mer count matrix also makes the whole thing in memory first.
3. Check if `pre-pca` reduces the dimension of the k-mer matrix

ChatGPT RAM recommendation

```
How I'd make this substantially more RAM-friendly

The ideal architecture is:

disk
  ↓
load matrix
  ↓
convert to float32
  ↓
preprocess in-place / with minimal copies
  ↓
PCA using a memory-conscious implementation
  ↓
save
  ↓
delete intermediates
  ↓
next normalization

At minimum, I'd explicitly convert to float32 and release intermediates:

matrix = load_matrix(matrix_path)

sequence_ids = matrix.index.to_numpy()

# Extract numeric data once
X = matrix.to_numpy(dtype=np.float32, copy=True)

del matrix

for norm in normalisations:
    X_norm = preprocess_data(X, norm)

    if self.pca_dim_red:
        X_norm = reduce_dimensions_with_pca(
            X_norm,
            keep_pcs=self.keep_pcs,
            keep_variance=self.keep_variance,
        )

    np.save(out_path, X_norm)

    del X_norm

Although even this isn't enough if the matrix is enormous.

If your k-mer matrix is large, the really good solution is to avoid loading the whole pandas DataFrame at once, using chunked/streaming preprocessing or a disk-backed format such as NumPy memmap/Zarr, and use randomized/incremental PCA where appropriate.
```
