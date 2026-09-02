**ChatGPT:**

It is **reasonably memory-efficient in the `raw`/`relative`/`log`/`clr` cases, but not truly memory-efficient for large k-mer matrices**. The biggest issue is that your current implementation can hold **several full copies of the matrix simultaneously**, especially during PCA.

### Rough memory picture

Suppose your matrix has:

- `N` samples
- `M` k-mer features
- `float32`

The raw numeric matrix itself costs approximately:

`N × M × 4 bytes`

For example, 100,000 samples × 10,000 k-mers:

- one float32 matrix ≈ **4 GB**

Your code can temporarily require substantially more than that.

### Where memory is being used

#### 1. `matrix = load_matrix(...)`

You first load the entire matrix:

```Python
matrix = load_matrix(matrix_path)
```

If `load_matrix()` returns a pandas DataFrame, you already have the complete matrix in RAM.

Then:

```Python
X = preprocess_data(matrix, norm)
```

and inside:

```Python
X = df.copy().astype(np.float32)
```

This creates another matrix.

So, conceptually:

```
matrix       ~4 GB
X            ~4 GB
----------------
             ~8 GB
```

There can be additional temporary allocations depending on the operation.

---

### 2. `relative`

This is relatively cheap:

```Python
row_sums = X.sum(axis=1)
X = X.div(row_sums, axis=0)
```

But `X.div(...)` creates another DataFrame rather than modifying `X` in place.

So peak memory can be roughly:

```
original matrix
X
result of div
row_sums
```

For a large matrix, this can be significant.

---

### 3. `log`

```Python
X = np.log1p(X)
```

This is somewhat better because NumPy can operate in place if explicitly requested, but your current code doesn't guarantee that.

Then:

```Python
return pd.DataFrame(X, ...)
```

creates/wraps another pandas object.

You could make this substantially more memory efficient.

---

### 4. `clr`

This is probably one of the more memory-intensive preprocessing methods:

```Python
X += 1e-9
geometric_mean = np.exp(np.mean(np.log(X), axis=1))
X = np.log(X.div(geometric_mean, axis=0))
```

There are multiple temporaries here:

```
X
np.log(X)
geometric_mean
X.div(...)
np.log(...)
```

For a multi-GB matrix, peak memory can become **several times the matrix size**.

Also, because you're doing CLR on k-mer counts, I'd consider whether the particular pseudocount and implementation are appropriate for your downstream analysis, but that's separate from memory.

---

### 5. `zscore`

This:

```Python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

can require another full matrix for `X_scaled`.

So again, approximately:

```
X              4 GB
X_scaled       4 GB
other sklearn temporary memory
```

Potentially more depending on sklearn's internals.

---

## The biggest problem: PCA

This is the part I'd be most concerned about.

With variance-based PCA:

```Python
pca_full = PCA()
X_full = pca_full.fit_transform(df.values)
```

You explicitly create the **entire PCA-transformed matrix**.

Then:

```Python
keep_pcs = ...
pca = PCA(n_components=keep_pcs)
X_pca = pca.fit_transform(df.values)
```

So you're effectively doing PCA **twice**.

For example, if your input is:

```
100,000 samples × 10,000 features
```

then:

```
input matrix              ~4 GB float32
PCA output X_full         potentially ~8 GB if float64
second PCA output         potentially another several GB
```

And sklearn's PCA generally performs calculations in floating-point formats that can result in substantially higher memory use than your original float32 DataFrame.

So **PCA is likely to dominate your memory usage**.

---

# There is also a bug/inefficiency in the current PCA design

You currently do:

```Python
pca_full = PCA()
X_full = pca_full.fit_transform(df.values)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

keep_pcs = np.searchsorted(
    cumulative_variance,
    keep_variance
) + 1

pca = PCA(n_components=keep_pcs)

X_pca = pca.fit_transform(df.values)
```

You don't actually need `X_full`.

You only need the explained variance ratios.

But sklearn's normal PCA still computes the transformed representation during `fit_transform`.

You can instead do:

```Python
pca_full = PCA()
pca_full.fit(df.values)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)
```

This avoids retaining `X_full`.

However, it still performs a full PCA decomposition, which itself can be expensive.

---

# Another important issue: `np.save`

This:

```Python
np.save(out_path, X)
```

is good for the **output**, because you're storing a compact binary NumPy array rather than CSV.

But it doesn't make the processing memory-efficient.

`np.save()` basically means:

```
RAM → disk
```

after you've already created the entire result in memory.

If the matrix is huge, a better approach is to use **NumPy memmap** or chunked processing.

---

# What I would change

I'd structure the pipeline around this principle:

```
disk
 ↓
memory-mapped/chunked input
 ↓
preprocess in chunks where possible
 ↓
disk-backed output
 ↓
PCA
 ↓
small final PCA matrix
```

In particular:

### 1. Avoid unnecessary `DataFrame.copy()`

Instead of:

```Python
X = df.copy().astype(np.float32)
```

consider converting once at the pipeline boundary:

```Python
X = df.to_numpy(dtype=np.float32, copy=False)
```

Then do numerical operations on NumPy arrays.

For large matrices, pandas isn't buying you much during numerical preprocessing.

---

### 2. Use in-place operations where safe

For example:

```Python
X = df.to_numpy(dtype=np.float32, copy=True)

if method == "log":
    np.log1p(X, out=X)
    return X
```

That avoids allocating another full matrix for the result.

Similarly, relative normalization can be done with broadcasting:

```Python
row_sums = X.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
X /= row_sums
```

Now you have:

```
X              ~4 GB
row_sums       tiny compared with X
```

rather than another full DataFrame.

---

### 3. Be careful with CLR

You can substantially reduce temporary allocations.

For example, conceptually:

```Python
X += 1e-9
np.log(X, out=X)

row_means = X.mean(axis=1, keepdims=True)

X -= row_means
```

This works because:

![clr-picture](/Users/markpampuch/Dropbox/KAUST/PhD/20260823_kmer-ord-local/kmer-ord/my-notes/clr.png)

So you don't need to calculate:

```Python
geometric_mean
X.div(...)
np.log(...)
```

at all.

That can turn CLR from a multi-copy operation into essentially:

```
X
+
small row-means array
```

---

# PCA is where I'd make the biggest architectural change

If your dataset is genuinely large, I'd consider **IncrementalPCA**:

```Python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(
    n_components=keep_pcs,
    batch_size=2048,
)

for batch in batches:
    ipca.partial_fit(batch)
```

Then transform batches:

```Python
for batch in batches:
    transformed = ipca.transform(batch)
```

This means you never need:

```
entire input
+
entire PCA output
```

in RAM.

Instead:

```
input matrix
+
one batch
+
small PCA model
+
one output batch
```

This is **much more scalable**.

---

## If you want variance-based PCA

You have a slightly harder problem because you don't know `keep_pcs` beforehand.

A reasonable strategy is:

1. Fit IncrementalPCA with a sufficiently large maximum number of components.
2. Examine `explained_variance_ratio_`.
3. Determine the number needed for your threshold.
4. Transform using only those components.

For example:

```
max_pcs = 500

IncrementalPCA(n_components=500)
        ↓
fit batches
        ↓
explained_variance_ratio_
        ↓
find PCs needed for 95%
        ↓
transform batches
```

This avoids creating `X_full`.

---

# One more important consideration: `load_matrix`

Your memory efficiency depends heavily on what this does:

```Python
matrix = load_matrix(matrix_path)
```

If it's something like:

```Python
pd.read_csv(...)
```

then you're inherently loading the entire matrix into RAM.

If your matrices are large, I'd strongly consider changing the storage format to something like:

- `.npy` + `np.memmap`
- Zarr
- HDF5
- Parquet, depending on access patterns

For your particular workflow, **NumPy memmap is probably the simplest** if the matrix is fundamentally a dense numerical array.

Then you could have:

```
kmer_matrix.npy
       ↓
np.memmap
       ↓
batch of rows
       ↓
preprocessing
       ↓
PCA
       ↓
output.npy
```

without loading the entire matrix.

---

# Overall assessment

I'd rank the current implementation roughly like this:

| Part                         | Memory efficiency           |
| ---------------------------- | --------------------------- |
| `raw`                        | 🟢 Good-ish                 |
| `relative`                   | 🟡 Moderate                 |
| `log`                        | 🟡 Moderate                 |
| `clr`                        | 🟠 Poor for large matrices  |
| `zscore`                     | 🟠 Moderate/Poor            |
| fixed-component PCA          | 🔴 Potentially very high    |
| variance-threshold PCA       | 🔴 Worst                    |
| separate `.npy` sequence IDs | 🟢 Excellent                |
| `np.save` output             | 🟢 Good, but not RAM-saving |

The **main thing I'd change is not putting `np.save()` inside `preprocess_data()`**. I'd keep preprocessing responsible for numerical transformation and let the operation handle persistence.

But I **would change the preprocessing functions to operate on NumPy arrays in-place**, and I'd make the `MatrixPreprocessing.run()` pipeline **batch/memmap-aware**.

The most important architectural target is:

```
Current:

DataFrame
   ↓
copy
   ↓
preprocessing temporaries
   ↓
PCA full matrix
   ↓
second PCA
   ↓
np.save


Better:

disk/memmap
   ↓
batch
   ↓
in-place preprocessing
   ↓
IncrementalPCA
   ↓
batch output
   ↓
disk
```

If you tell me approximately **number of samples × number of k-mers** (e.g. `50,000 × 100,000`) and your available RAM, I can estimate the **peak RAM of your current implementation** and show you a memory-efficient rewrite of `MatrixPreprocessing` around that scale.

**ChatGPT:**

Yes. The key distinction is that **“variance PCA” and “IncrementalPCA” are not actually two different PCA objectives**.

- **Standard/variance PCA** usually means the conventional PCA algorithm that finds directions explaining the maximum variance.
- **IncrementalPCA** tries to find essentially the **same kind of PCA directions**, but learns them **batch by batch** instead of requiring the whole matrix in memory.

So the main trade-off is **memory/scalability vs. exactness/speed**, not “variance PCA vs. non-variance PCA.”

## 1. Standard PCA

Suppose you have:

```
N samples × M k-mers
```

Standard PCA conceptually works with the whole dataset at once:

```
X
│
├── center/scale
│
├── compute covariance/SVD
│
└── principal components
```

The first PC is the direction that explains the greatest amount of variance, the second explains the greatest remaining variance, etc.

### Advantages

**1. More accurate/exact**

For a conventional PCA implementation such as `sklearn.decomposition.PCA`, the decomposition is based on the complete dataset.

This is the gold standard if your matrix fits comfortably in memory.

**2. Usually faster for moderate-sized data**

For matrices that aren't enormous, optimized LAPACK/SVD routines can be considerably faster than repeatedly processing batches.

**3. Easier to reason about**

You can do:

```Python
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)
```

and you're done.

**4. Explained variance is straightforward**

You get:

```Python
pca.explained_variance_ratio_
```

and can determine how many PCs are required for, say, 90%, 95%, or 99% variance.

---

### Disadvantages

The big problem for your application is memory.

If you have something like:

```
100,000 samples × 10,000 k-mers
```

then a float32 matrix is already about:

```
4 GB
```

But PCA may require additional memory for:

- centered data
- SVD workspaces
- transformed data
- temporary arrays
- sklearn's internal calculations

And your current code makes things worse by doing:

```Python
pca_full.fit_transform(df.values)
```

and then running PCA again.

So standard PCA can become impractical when the k-mer matrix gets very large.

---

# 2. IncrementalPCA

IncrementalPCA changes **how the PCA is calculated**, not what PCA is trying to accomplish.

Instead of:

```
10 million observations
       ↓
load everything
       ↓
PCA
```

you do:

```
batch 1 ──┐
batch 2 ──┤
batch 3 ──┤
batch 4 ──┤──→ IncrementalPCA
batch 5 ──┤
...       │
batch N ──┘
```

For example:

```Python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(
    n_components=100,
    batch_size=2048
)

for batch in batches:
    ipca.partial_fit(batch)
```

You don't need the entire matrix in RAM.

---

# The biggest advantage: memory

Imagine:

```
Matrix = 100,000 × 10,000 float32
       ≈ 4 GB
```

With ordinary PCA, you might need several GB beyond that.

With IncrementalPCA, you can process:

```
2048 × 10,000
```

at a time.

That batch is only roughly:

```
2048 × 10,000 × 4
≈ 82 MB
```

So your memory profile can look more like:

```
input/storage        disk
                       ↓
                    ~82 MB batch
                       ↓
                 IncrementalPCA
                       ↓
                 small output
```

instead of:

```
4 GB+ matrix
+
PCA temporary memory
+
PCA output
```

This is a **huge advantage for large k-mer datasets**.

---

# But there is an important catch

IncrementalPCA is generally **not mathematically identical to running ordinary PCA on the entire matrix in one shot**.

It processes the data sequentially and updates the estimated principal subspace.

So you should think:

```
Standard PCA
    ↓
PCA of the entire dataset

IncrementalPCA
    ↓
approximation/streaming-style estimation of that PCA
```

The result can be very close, especially with appropriate batch sizes and well-behaved data, but you shouldn't assume the PCs will be bit-for-bit identical.

---

# 3. Accuracy trade-off

This is probably the most important question for your application.

### Standard PCA

You essentially get:

```
global optimum for the PCA objective
```

given the data and numerical algorithm.

### IncrementalPCA

You get an estimate based on sequential batches.

Its result can depend on things such as:

- batch size
- data ordering
- number of samples
- number of components
- preprocessing
- numerical precision

Therefore, if you care about extremely precise PCA directions, standard PCA is preferable when feasible.

But for many downstream analyses, the difference may be negligible compared with the enormous memory savings.

---

# 4. IncrementalPCA is not necessarily faster

It's tempting to think:

> "I'm processing smaller chunks, so it must be faster."

Not necessarily.

For a sufficiently large dataset:

```
Standard PCA
    ↓
potentially very expensive but optimized
```

versus:

```
IncrementalPCA
    ↓
many smaller SVD/update operations
```

IncrementalPCA can actually take **longer**.

You're trading:

**memory → computational efficiency**

rather than magically improving both.

For your problem, though, this may be an excellent trade because:

> A PCA that takes 2 hours and fits in RAM is much better than a PCA that theoretically takes 30 minutes but crashes because it needs 100 GB of RAM.

---

# 5. Batch size matters

With IncrementalPCA:

```Python
IncrementalPCA(
    n_components=100,
    batch_size=2048
)
```

`batch_size` is an important parameter.

### Small batch

```
512 samples
```

Pros:

- lower memory
- can work with constrained RAM

Cons:

- more batches
- potentially slower
- potentially less stable/accurate

### Large batch

```
8192 samples
```

Pros:

- fewer updates
- often faster
- potentially better numerical behavior

Cons:

- more RAM

So you'd generally choose the largest batch that comfortably fits in memory.

---

# 6. One subtle issue with your variance threshold

Your current code wants to do something like:

```Python
keep_variance = 0.95
```

meaning:

> Keep enough PCs to explain 95% of the variance.

With ordinary PCA this is easy:

```Python
pca = PCA()
pca.fit(X)

cumulative = np.cumsum(pca.explained_variance_ratio_)

n_components = np.searchsorted(
    cumulative,
    0.95
) + 1
```

But there's an awkward problem:

**you don't know how many components you need until you've calculated the PCA.**

With IncrementalPCA, you therefore need to decide on a maximum number of components.

For example:

```Python
ipca = IncrementalPCA(
    n_components=500,
    batch_size=2048
)
```

After fitting:

```Python
cumulative = np.cumsum(ipca.explained_variance_ratio_)

n_components = np.searchsorted(
    cumulative,
    0.95
) + 1
```

You might discover:

```
95% variance = 137 PCs
```

Then you only retain those 137 PCs.

---

# 7. A major difference for your particular pipeline

Your current approach is:

```
1. PCA with all components
        ↓
2. calculate explained variance
        ↓
3. determine number of PCs
        ↓
4. PCA again
        ↓
5. save result
```

That's expensive.

With IncrementalPCA you could instead do:

```
                    ┌── batch 1
                    ├── batch 2
matrix on disk ─────┼── batch 3
                    ├── ...
                    └── batch N
                         ↓
                  IncrementalPCA
                         ↓
                  explained variance
                         ↓
                 choose components
                         ↓
               transform batches
                         ↓
                    output.npy
```

This is much more appropriate for a huge matrix.

---

# 8. One thing I would _not_ do

I wouldn't automatically replace:

```Python
PCA()
```

with:

```Python
IncrementalPCA()
```

just because the matrix is large.

I'd first ask:

### Can the matrix comfortably fit in memory?

If yes:

**Use standard PCA.**

It's simpler and generally preferable.

If no:

**Use IncrementalPCA or another out-of-core/dimensionality-reduction strategy.**

---

# 9. Practical comparison

|                             | Standard PCA                         | IncrementalPCA            |
| --------------------------- | ------------------------------------ | ------------------------- |
| PCA objective               | Maximum variance                     | Maximum variance          |
| Processes whole dataset     | Yes                                  | No, batches               |
| RAM requirement             | High                                 | Much lower                |
| Exact same result           | Yes, within numerical implementation | Not guaranteed            |
| Large datasets              | Poor scalability                     | Much better               |
| Speed                       | Usually better when data fits RAM    | Can be slower             |
| Batch processing            | No                                   | Yes                       |
| Works with memmap           | Can, but memory still matters        | Excellent use case        |
| Explained variance          | Easy                                 | Easy after fitting        |
| Variance-threshold workflow | Straightforward                      | Requires choosing max PCs |
| Implementation complexity   | Low                                  | Moderate                  |
| Best use case               | Data fits RAM                        | Data doesn't fit RAM      |

---

# 10. For your k-mer matrix specifically

I'd think about it this way.

### Small/medium dataset

```
50k × 5k
```

If your machine has plenty of RAM:

**Standard PCA**

is probably the better choice.

You get accurate PCA and simpler code.

### Large dataset

```
100k × 10k
```

or especially:

```
100k × 100k
```

I'd strongly consider:

**memmap/chunked storage + IncrementalPCA**

because the raw matrix itself can already be enormous.

### Extremely high-dimensional k-mer data

If you're dealing with hundreds of thousands or millions of k-mer features, I'd also question whether **PCA is the right first dimensionality-reduction step**. Depending on your downstream task, sparse methods, feature filtering, truncated SVD, or other approaches may be considerably more appropriate.

---

## One final distinction that's worth remembering

The terminology can be confusing:

**"Variance PCA" isn't really an alternative to IncrementalPCA.**

Think of it as two separate dimensions:

```
             WHAT are we optimizing?
                       │
                       ▼
              Maximum variance
                       │
             ┌─────────┴─────────┐
             │                   │
       Standard PCA       IncrementalPCA
       whole dataset        batches
```

Both are trying to find directions that capture variance.

The difference is **how they compute those directions**.

For your pipeline, I'd choose **standard PCA if the matrix comfortably fits in RAM; IncrementalPCA if RAM is the limiting factor**. The latter is particularly attractive if you move your k-mer matrix to a memory-mapped/chunked format, because then you can keep the actual dataset on disk and only bring manageable batches into memory.
