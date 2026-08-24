# Dimensionality-Reduction Memory-Safety Report

## Executive summary

I reviewed every dimensionality-reduction method implemented or referenced in the supplied `methods.py`. The most important distinction is between:

1. **Needing the original `X` matrix to be resident/accessible during fitting**, and
2. **Creating an additional dense (N\times N) matrix**, which is usually the much more serious memory problem.

Your current wrapper passes the complete `X` object directly into every estimator via `fit_transform(X)`. For example, PCA, t-SNE, UMAP, TriMap, PaCMAP, LocalMAP, LLE, SparsePCA and KernelPCA are all called this way.

### Bottom line

| Method        | Current implementation   | Dense (N\times N) working matrix? | Can be made substantially more memory-safe?                             | Priority     |
| ------------- | ------------------------ | --------------------------------: | ----------------------------------------------------------------------- | ------------ |
| **PCA**       | `sklearn.PCA`            |                                No | **Yes — IncrementalPCA**                                                | 🔴 High      |
| **SparsePCA** | `sklearn.SparsePCA`      |                                No | **Yes — MiniBatchSparsePCA**                                            | 🔴 High      |
| **KernelPCA** | `sklearn.KernelPCA`      |                 **Yes, normally** | Limited; kernel approximation required                                  | 🔴 Very high |
| **t-SNE**     | Barnes-Hut default       |                    No, by default | **Yes, partly**; sparse/KNN input and out-of-core architecture possible | 🟠 High      |
| **UMAP**      | `umap.UMAP`              |                                No | **Yes**; already has low-memory NN-descent                              | 🟢 Good      |
| **TriMap**    | `TRIMAP`                 |                                No | **Yes**; explicitly designed for large datasets                         | 🟢 Good      |
| **PaCMAP**    | `PaCMAP`                 |                                No | **Yes**, but pair storage can become substantial                        | 🟠 Medium    |
| **LocalMAP**  | `LocalMAP`               |                                No | **Yes**, similar to PaCMAP                                              | 🟠 Medium    |
| **LLE**       | `LocallyLinearEmbedding` |                    Usually sparse | **Yes**, especially with ARPACK; dense solver is dangerous              | 🟠 Medium    |

The key finding is that **KernelPCA is the clearest method in this list that fundamentally constructs an (N\times N) kernel matrix**, while ordinary PCA is the clearest method where you can replace the implementation with a genuinely out-of-core algorithm. `IncrementalPCA` is explicitly designed to operate on batches/memmap data without loading the complete dataset into memory.

---

## 1. PCA — strong candidate for replacement

Your code currently does:

```python
model = PCA(n_components=dims, **params)
embedding = model.fit_transform(X)
```

Standard scikit-learn PCA is **batch-only**. Scikit-learn explicitly states that all data processed by `PCA` must fit in main memory.

### Memory characteristics

PCA does **not** inherently require an (N\times N) matrix, but the input matrix itself must be available to the batch algorithm. Depending on solver/data shape, substantial temporary arrays can also be created.

### Better implementation

`IncrementalPCA` is almost exactly what you want for a memory-safe version. It processes the data in minibatches and has memory complexity approximately proportional to:

[
O(\text{batch size}\times n_\text{features})
]

rather than (O(N\times n\_\text{features})). It can also work with NumPy memory-mapped files.

**Recommendation: HIGH PRIORITY**

Replace:

```python
PCA(...)
```

with an option such as:

```python
IncrementalPCA(
    n_components=dims,
    batch_size=...
)
```

and ideally allow your input loader to supply batches rather than constructing the entire `X` first.

---

## 2. SparsePCA — replaceable with a minibatch implementation

Current code:

```python
model = SparsePCA(
    n_components=dims,
    random_state=seed,
    n_jobs=n_jobs,
    **params
)
embedding = model.fit_transform(X)
```

`SparsePCA` itself works on the complete training matrix. It does not have the same clean out-of-core semantics as IncrementalPCA.

However, scikit-learn provides **`MiniBatchSparsePCA`**, specifically described as the minibatch variant of SparsePCA.

### Recommendation

This is another good candidate for a memory-safe implementation:

```text
SparsePCA
    ↓
MiniBatchSparsePCA
```

The tradeoff is that the minibatch version is an approximation and can be less accurate.

**Recommendation: HIGH PRIORITY if SparsePCA is used on large datasets.**

---

## 3. KernelPCA — biggest fundamental memory problem

Current code:

```python
model = KernelPCA(n_components=dims, n_jobs=n_jobs, **params)
embedding = model.fit_transform(X)
```

This is the most concerning method in your list.

KernelPCA constructs a kernel matrix whose natural shape is:

[
N\times N
]

The current scikit-learn implementation explicitly computes the kernel matrix during `fit`. Its estimator stores eigenvectors indexed by samples, and the dense eigensolver operates on the kernel matrix.

For example:

|       N | float64 (N^2) matrix |
| ------: | -------------------: |
|  10,000 |              ~0.8 GB |
|  25,000 |              ~5.0 GB |
|  50,000 |               ~20 GB |
| 100,000 |               ~80 GB |
| 250,000 |              ~500 GB |

And that is **just one dense matrix**, before eigensolver workspaces and the original `X`.

### Can it be fixed?

Not simply by changing a parameter.

The kernel method itself is the problem. To make this genuinely scalable you would need something such as:

- approximate kernel features,
- Nyström approximation,
- random Fourier features,
- landmark/kernel approximation,
- or a completely different DR method.

So I would classify KernelPCA as:

> **Not memory-safe at large N in its current mathematical formulation.**

**Recommendation: VERY HIGH PRIORITY to disable it automatically above a dataset-size threshold, or replace it with an approximate kernel method.**

---

## 4. t-SNE — depends heavily on the selected algorithm

Your implementation uses:

```python
model = TSNE(
    n_components=dims,
    random_state=seed,
    n_jobs=n_jobs,
    **params
)
embedding = model.fit_transform(X)
```

and your default configuration specifies `init='pca'`.

### Barnes-Hut t-SNE

The modern scikit-learn implementation's default Barnes-Hut algorithm uses approximate nearest neighbors rather than constructing the complete pairwise distance matrix.

So **default Barnes-Hut t-SNE does not inherently require an (N\times N) distance matrix**.

That is good.

### Exact t-SNE

The exact implementation is very different. The source explicitly calculates pairwise distances:

```text
distances = pairwise_distances(X, ...)
```

and subsequently constructs the joint probability representation.

Therefore:

> **Exact t-SNE should be considered an (O(N^2))-memory method.**

Your current code does not explicitly request `method="exact"`, so that is not the current default problem.

### Another issue: PCA initialization

Your configuration specifies:

```python
'init': 'pca'
```

For a very large dataset, I'd consider changing the initialization to either:

- `random`, or
- an incremental/out-of-core PCA embedding.

That avoids sneaking a full-batch PCA into an otherwise scalable t-SNE pipeline.

**Recommendation: MEDIUM/HIGH**

Keep Barnes-Hut, explicitly prohibit `method="exact"` for large datasets, and consider an incremental initialization.

---

## 5. UMAP — already relatively memory-safe

Your implementation:

```python
model = umap.UMAP(
    n_components=dims,
    random_state=None,
    n_jobs=n_jobs,
    **params
)
embedding = model.fit_transform(X)
```

UMAP is one of the better choices in this list for large datasets.

The implementation uses approximate nearest-neighbor search through NN-descent rather than constructing the complete pairwise distance matrix. The current UMAP source exposes a `low_memory=True` option for NN-descent specifically to reduce memory usage.

UMAP also supports sparse input.

### Important distinction

UMAP **still needs access to the complete training dataset** during fitting. `low_memory=True` does not mean "stream the dataset from disk one row at a time."

It means approximately:

> don't unnecessarily construct enormous intermediate neighbor-search structures.

So:

**Good:** no (N^2) distance matrix.

**Not fully out-of-core:** the complete `X` is still supplied to the estimator.

### Recommendation

UMAP should remain one of your preferred large-scale methods.

I would explicitly set:

```python
low_memory=True
```

rather than relying on the library default, because memory behavior is an important property of this pipeline.

**Recommendation: LOW priority for replacement; HIGH priority for exposing/configuring memory controls.**

---

## 6. TriMap — particularly suitable for large datasets

Current implementation:

```python
model = TRIMAP(n_dims=dims, **params)
embedding = model.fit_transform(X)
```

TriMap was specifically designed around sampled triplets rather than a full pairwise distance matrix.

Its implementation supports:

- approximate nearest-neighbor information,
- precomputed k-nearest neighbors,
- and a precomputed distance matrix if explicitly requested.

Importantly, the TriMap authors report that their method scales to millions of points without exhausting memory.

### Memory-safe opportunity

The particularly interesting feature is:

```text
knn_tuple=(knn_nbrs, knn_distances)
```

This means you could potentially separate:

1. neighbor construction,
2. storage of the compact neighbor graph,
3. embedding.

That gives you a route toward a more controlled memory architecture.

**Recommendation: LOW priority for replacing the algorithm.**

TriMap is already one of the better choices for your stated objective.

---

## 7. PaCMAP — scalable, but stores pair information

Current implementation:

```python
model = PaCMAP(n_components=dims, **params)
embedding = model.fit_transform(X)
```

PaCMAP is also designed around sampled pairs rather than a full (N\times N) distance matrix.

However, there is an important memory tradeoff: PaCMAP precomputes and retains its mid-near and further pairs. An analysis of the implementation estimates additional storage around `2.5 * N * n_neighbors` under default settings.

This is much better than (N^2), but it isn't free.

Your configuration actually increases `n_neighbors` with dataset size:

```text
small   = 15
medium  = 100
large   = 200
```

That makes memory use grow substantially on large datasets.

### Recommendation

PaCMAP is fundamentally suitable for large data, but I would make pair counts an explicit memory budget rather than scaling `n_neighbors` aggressively.

**Recommendation: MEDIUM priority.**

---

## 8. LocalMAP — similar story to PaCMAP

Your code:

```python
model = LocalMAP(n_components=dims, **params)
embedding = model.fit_transform(X)
```

LocalMAP is implemented inside the PaCMAP package and is graph/pair based rather than based on an (N\times N) dense distance matrix. The project explicitly supports user-supplied nearest-neighbor information for large-scale datasets.

That is a useful property for your pipeline.

The same caveat applies as PaCMAP: the neighbor/pair graph is smaller than the full data matrix, but it can still become a significant memory allocation.

Your large-scale configuration uses:

```text
n_neighbors = 200
MN_ratio    = 0.7
FP_ratio    = 1.0
```

### Recommendation

Keep it, but expose/monitor the graph and pair allocations.

**Recommendation: MEDIUM priority.**

---

## 9. LLE — deceptively memory-sensitive

Your implementation uses:

```python
LocallyLinearEmbedding(
    n_neighbors=n_neighbors,
    n_components=dims,
    n_jobs=n_jobs,
)
embedding = model.fit_transform(X)
```

LLE is interesting because it does **not necessarily need a dense (N\times N) matrix**.

The current scikit-learn implementation constructs a sparse reconstruction graph when using the sparse/ARPACK path. The documentation states that ARPACK can operate on sparse matrices, whereas the dense eigensolver should be avoided for large problems.

The source confirms this:

- the neighbor graph is constructed sparsely;
- `M = (I-W)'(I-W)` is kept sparse for ARPACK;
- the dense path converts it to a full array.

### This creates an important rule

For large N:

```text
LLE + ARPACK       → potentially reasonable
LLE + dense solver → dangerous
```

The current `eigen_solver="auto"` usually chooses ARPACK for sufficiently large datasets with a small number of requested components, but I would make the choice explicit for a memory-sensitive pipeline.

### Modified LLE caveat

The `modified` implementation allocates:

```text
V = (N, n_neighbors, n_neighbors)
```

and several additional arrays.

That can become surprisingly large.

For example, with 1,000,000 samples and 200 neighbors, that tensor alone would contain:

[
10^6 \times 200 \times 200 = 4\times10^{10}
]

values.

That is completely impractical.

**Recommendation: HIGH caution for LLE**, particularly modified/Hessian/LTSA variants. Standard LLE with sparse ARPACK is much safer.

---

# Overall classification

## Tier 1 — genuinely replaceable with out-of-core implementations

### PCA

**Current:** full-batch PCA.

**Better:** `IncrementalPCA`.

This is the strongest and easiest win. Scikit-learn explicitly supports minibatch/out-of-core PCA and memory-mapped data.

### SparsePCA

**Current:** `SparsePCA`.

**Better:** `MiniBatchSparsePCA`.

This gives you a minibatch alternative already provided by scikit-learn.

---

# Tier 2 — no (N^2) matrix, but still needs the dataset accessible

These are reasonably memory-safe algorithms:

- **UMAP**
- **TriMap**
- **PaCMAP**
- **LocalMAP**
- **Barnes-Hut t-SNE**
- **standard LLE + sparse/ARPACK**

Their major memory structures are neighborhood graphs, sampled pairs, embeddings, etc., rather than a full pairwise matrix.

The distinction is important:

> They are **not streaming algorithms**, but they are much more memory-safe than methods that explicitly materialize (N\times N).

---

# Tier 3 — dangerous at large N

## KernelPCA

This is the biggest problem.

Its fundamental kernel representation is (N\times N).

I would strongly recommend either:

- disabling KernelPCA above a configurable N,
- or replacing it with an approximate kernel feature method.

## Exact t-SNE

Not currently your default, but should be explicitly prevented for large datasets.

The exact algorithm computes pairwise distances and has (O(N^2)) behavior.

## Dense LLE

Also dangerous because the sparse reconstruction matrix is explicitly converted to a dense (N\times N) matrix when `eigen_solver="dense"`.

---

# A bigger problem in your current wrapper

There is another memory issue independent of the individual algorithms.

Your parameter-screening implementation deliberately keeps every embedding in memory:

```python
density_combos: list[tuple[float, float, pd.DataFrame]] = []
```

and then:

```python
density_combos.append((axis1_value, axis2_value, df))
```

That means if you screen, for example, 20 parameter combinations over 1,000,000 sequences, you can retain **20 complete embedding DataFrames simultaneously**.

Even though each embedding is only 2–3 columns, this can become substantial, and it is unnecessary because the embeddings have already been written to disk.

### This is one of the easiest memory fixes in the entire file.

Instead of retaining:

```text
(axis1, axis2, complete DataFrame)
```

retain only what the density plot actually needs, or render panels incrementally.

This is independent of whether UMAP, TriMap, PaCMAP, etc. are memory-efficient.

---

# Recommended architecture

I would change the pipeline to have three memory classes:

### `streamable`

```text
PCA → IncrementalPCA
SparsePCA → MiniBatchSparsePCA
```

These can consume chunks from disk.

### `graph_based`

```text
UMAP
TriMap
PaCMAP
LocalMAP
LLE
Barnes-Hut t-SNE
```

These still need the training data available, but should avoid (N^2) allocations.

For these, a future architecture could use:

```text
disk-backed X
       ↓
batched / approximate KNN
       ↓
compact graph / pair representation
       ↓
embedding
```

rather than treating `X` as one giant in-RAM object.

### `quadratic`

```text
KernelPCA
exact t-SNE
dense LLE
```

These should either be disabled at large N or require an explicit override.

---

# My recommended priority order

**1. Replace PCA with IncrementalPCA.**

This is the cleanest true out-of-core improvement.

**2. Replace SparsePCA with MiniBatchSparsePCA.**

Same reason.

**3. Put a hard memory guard around KernelPCA.**

It is fundamentally an (N^2) method in this implementation.

**4. Put a hard guard around exact t-SNE.**

Keep Barnes-Hut for large datasets.

**5. Force sparse/ARPACK LLE for large datasets.**

Especially avoid modified LLE at very large N.

**6. Explicitly enable `low_memory=True` for UMAP.**

UMAP already has the right architecture; make the memory policy explicit.

**7. Reduce/parameterize PaCMAP/LocalMAP pair counts.**

Your current `large` presets increase neighbor counts substantially.

**8. Fix parameter-screen retention.**

This is a memory problem in _your wrapper itself_, regardless of the DR algorithm.

---

## One important caveat

There is a difference between **"doesn't construct an (N\times N) matrix"** and **"can run without the complete `X` in RAM."**

Most of UMAP, TriMap, PaCMAP, LocalMAP, t-SNE and LLE fall into the first category, not the second. Their APIs still receive the complete training matrix. For example, UMAP's nearest-neighbor routine operates on `X`, although its neighbor representation is only (N\times k).

So if your actual goal is:

> **"My k-mer matrix may be hundreds of GB and I want DR to run without ever loading that entire matrix into RAM."**

then the answer is more restrictive:

**PCA/SparsePCA have straightforward solutions; the graph-based methods need an architectural change around disk-backed/chunked feature access; KernelPCA is the least suitable of the current methods.**

That is the distinction I would use when redesigning this pipeline.
