# Dimensionality-reduction threading

Every method is launched the same way. `DimensionalityReduction` calls `run_dr_methods` with `isolate=True`, and that starts one fresh worker process per method (`ProcessPoolExecutor`, `max_workers=1`). The worker loads the matrix and fits that method. A crash in one worker skips that method; later methods still run.

Threading inside the worker is not the same for every method.

The CLI `-t` / `--threads` value is stored as `n_jobs` and passed into the constructor only for UMAP and t-SNE.

- UMAP applies `n_jobs` with `numba.set_num_threads`. Its `random_state` is set to `None` so umap-learn does not force `n_jobs` back to 1. The nearest-neighbor step (pynndescent) uses that same Numba thread count. UMAP does not start its own process pool.
- t-SNE passes `n_jobs` through to scikit-learn.

TriMAP, PaCMAP, and LocalMAP have no thread argument. Their heavy loops are Numba `prange` loops, so they use Numba's process-wide thread pool. PaCMAP and LocalMAP also call FAISS for neighbors, and FAISS uses OpenMP.

scikit-learn PCA has no `n_jobs` parameter. Its parallelism is the BLAS library underneath (OpenBLAS or MKL).

`project`, `cluster`, and `dr` call `set_global_threads` before any of this work. That sets `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS`. Each worker inherits those variables. That is how the CLI count reaches TriMAP, PaCMAP, LocalMAP, and PCA.

The `threads` line in the dimensionality-reduction log is that one CLI value for the whole stage, not a separate setting per method.
