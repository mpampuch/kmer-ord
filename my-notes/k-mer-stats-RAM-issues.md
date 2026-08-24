**ChatGPT:**

Yes — with this implementation, **the k-mer matrix is ultimately loaded entirely into RAM**. The `chunksize` does **not** make the overall operation memory-safe.

The critical line is:

```Python
reader = pd.read_csv(..., chunksize=chunksize)
chunks = list(reader)
```

`pd.read_csv(..., chunksize=...)` creates a lazy chunk reader, which is good. But then:

```Python
chunks = list(reader)
```

immediately iterates through **the entire file and stores every chunk in memory**.

So if your matrix is 20 GB, you can end up with roughly the entire parsed matrix in RAM, plus additional copies.

### It actually gets worse

You then do:

```Python
results = [
    calculate_kmer_metrics_chunk(chunk)
    for chunk in chunks
]
```

So you have:

1. **All input chunks** in `chunks`
2. **All metric results** in `results`
3. Eventually another combined dataframe:

```Python
combined_metrics = pd.concat(results)
```

And inside `calculate_kmer_metrics_chunk`, there are substantial temporary arrays:

```Python
values = numeric.to_numpy(dtype=float)
```

This is particularly important.

Your input k-mer counts are `uint16`, which is relatively compact, but this converts the entire numeric portion of **each chunk** to `float64`.

For example, if a chunk has:

- 1,000 rows
- 100,000 k-mer columns

then:

```
uint16 input ≈ 200 MB
float64 values ≈ 800 MB
```

and you additionally create arrays for:

```Python
nonzero_mask
probs
positive
shannon_nats
shannon_bits
```

Some of those are potentially hundreds of MB for a large chunk.

### `cpus > 1` can make memory usage considerably worse

With:

```Python
futures = [
    executor.submit(calculate_kmer_metrics_chunk, chunk)
    for chunk in chunks
]
```

you're submitting **every chunk** to the process pool.

That means you already have all chunks in the parent process, and chunks need to be transferred to worker processes. Depending on the multiprocessing implementation/platform, this can involve additional serialization and copies.

So I would **not** consider the current implementation memory-safe for a large k-mer matrix.

---

## The good news: it's straightforward to fix

You don't need:

```Python
chunks = list(reader)
```

Instead, process one chunk at a time:

```Python
results = []

for chunk in reader:
    results.append(calculate_kmer_metrics_chunk(chunk))

combined_metrics = pd.concat(results)
```

That fixes the biggest problem with the input matrix, but it **still accumulates every result** in `results`.

Since your output is a TSV, you can go one step further and write each result immediately:

```Python
first = True

for chunk in reader:
    metrics = calculate_kmer_metrics_chunk(chunk)

    metrics.to_csv(
        output_file,
        sep="\t",
        mode="w" if first else "a",
        header=first,
    )

    first = False
```

Then you don't need to retain the full `combined_metrics` either.

### One complication

You currently calculate dataset-wide statistics at the end:

```Python
shannon.mean()
shannon.std(ddof=1)
shannon.min()
shannon.max()
```

You don't actually need the whole dataset in memory to calculate these. You can maintain running statistics (or use an online algorithm such as Welford's algorithm) while processing chunks.

So the truly memory-efficient architecture would be:

```
k-mer matrix on disk
        ↓
   read 1 chunk
        ↓
calculate metrics
        ↓
append metrics to output
        ↓
discard chunk
        ↓
   read next chunk
        ↓
       ...
```

RAM then scales approximately with **one chunk**, rather than with the size of the entire k-mer matrix.

### One other thing I noticed

This line:

```Python
"total_nonzero_kmers": row_sums.astype("int64"),
```

isn't counting nonzero k-mers. It's calculating the **sum of k-mer counts**.

You already correctly calculate the number of nonzero/unique k-mers here:

```Python
total_nonzero = nonzero_mask.sum(axis=1)
```

So the name `total_nonzero_kmers` is potentially misleading. If you intended it to mean "total number of k-mer observations/counts", the current code is correct but the name should reflect that. If you intended "number of nonzero k-mers", then it should use `total_nonzero`.

**Bottom line:** `chunksize=1000` currently does **not** protect you from loading the whole matrix. The culprit is specifically `chunks = list(reader)`, and `results`/`pd.concat()` add further memory pressure.

There are **two different quantities** being calculated here, and the variable names make them easy to confuse.

Suppose one row of your k-mer matrix looks like this:

```
kmer_A  kmer_B  kmer_C  kmer_D  kmer_E
   3       0       5       0       2
```

This means, for example:

- kmer_A occurred 3 times
- kmer_B occurred 0 times
- kmer_C occurred 5 times
- kmer_D occurred 0 times
- kmer_E occurred 2 times

There are two reasonable metrics you might want.

### 1. Total number of k-mer observations

You calculate this with:

```Python
row_sums = values.sum(axis=1)
```

For the example:

```
3 + 0 + 5 + 0 + 2 = 10
```

So:

```Python
row_sums
```

gives:

```
10
```

This means **10 total k-mer occurrences/counts** in that sample.

Your output currently calls this:

```Python
"total_nonzero_kmers": row_sums.astype("int64")
```

So the value is `10`.

But the name `total_nonzero_kmers` makes it sound like you're saying:

> "There are 10 different k-mers that have nonzero counts."

That's not true. There are only **3** k-mers with nonzero counts.

---

### 2. Number of k-mers that are actually present

This is what you're calculating here:

```Python
nonzero_mask = values != 0
```

For our example, that produces:

```
True  False  True  False  True
```

Then:

```Python
total_nonzero = nonzero_mask.sum(axis=1)
```

counts the `True` values:

```
3
```

So this metric means:

> **How many distinct k-mer columns have a nonzero abundance in this sample?**

And you currently store it as:

```Python
"num_unique_kmers": total_nonzero
```

That is much closer to the intended meaning.

---

## A concrete example

Imagine you have 100,000 possible k-mers in your matrix.

For one sample:

```
100,000 k-mer columns
       ↓
  1,500 have count > 0
       ↓
  total counts across those k-mers = 25,000
```

Then your two metrics should be:

| Metric                   | Value  | Meaning                            |
| ------------------------ | ------ | ---------------------------------- |
| Total k-mer counts       | 25,000 | Total number of k-mer observations |
| Number of nonzero k-mers | 1,500  | Number of k-mer types detected     |

Your current code produces:

```Python
"total_nonzero_kmers": 25000
"num_unique_kmers": 1500
```

The **values are correct**, but the first name is misleading.

I'd probably rename it to something like:

```Python
metrics_chunk = pd.DataFrame(
    {
        "total_kmer_counts": row_sums.astype("int64"),
        "num_nonzero_kmers": total_nonzero,
        "shannon_evenness": shannon_nats,
        "shannon_diversity": shannon_bits,
    },
    index=kmer_df.index,
)
```

That makes the distinction very clear.

---

## There's another subtle point: "unique k-mers"

I would also consider changing:

```Python
"num_unique_kmers": total_nonzero
```

to:

```Python
"num_nonzero_kmers": total_nonzero
```

Why?

Because **unique k-mers** can mean different things.

For example, suppose:

```
kmer_A = 10
kmer_B = 0
kmer_C = 10
```

There are:

- **2 nonzero k-mer types**: A and C
- **20 total k-mer observations**: 10 + 10
- **3 k-mer columns** in the matrix

So `num_nonzero_kmers` is unambiguous.

If by "unique k-mers" you mean "the number of distinct k-mer sequences detected in the sample," then `num_unique_kmers` is also reasonable—but I'd document that explicitly.

---

## Why this matters for your Shannon calculation

This distinction is actually important because your Shannon calculation uses the **counts**:

```Python
row_sums = values.sum(axis=1)
probs = values / row_sums_safe[:, None]
```

For our example:

```
counts = [3, 0, 5, 0, 2]
total  = 10
```

so:

```
probabilities = [0.3, 0, 0.5, 0, 0.2]
```

Then Shannon diversity is calculated from those probabilities.

So conceptually your four outputs are:

```
total_kmer_counts
    ↓
    10

num_nonzero_kmers
    ↓
    3

shannon_evenness
    ↓
    entropy in nats

shannon_diversity
    ↓
    entropy in bits
```

One more naming issue: **`shannon_evenness` is probably not actually evenness** in the ecological/statistical sense. Your code calculates Shannon entropy in nats:

```Python
shannon_nats = -np.sum(...)
```

True Shannon evenness is typically normalized entropy, e.g. `H / log(S)`, where `S` is the number of observed categories. So that column may also deserve a rename to something like `shannon_entropy_nats`.

Likewise, `shannon_diversity` is actually **Shannon entropy in bits**, not necessarily a "diversity" index in the broader sense.

I would **fix the naming now**, because the underlying calculations are mostly sensible, but the current names could cause confusion downstream.

### What your code is actually calculating

For each sample/row, you have:

1. **Total k-mer abundance**

   ```Python
   row_sums = values.sum(axis=1)
   ```

   → total number of k-mer observations/counts.

2. **Number of observed k-mer types**

   ```Python
   total_nonzero = (values != 0).sum(axis=1)
   ```

   → number of k-mer categories with nonzero abundance.

3. **Shannon entropy in nats**

   ```Python
   shannon_nats = -np.sum(probs * np.log(probs), axis=1)
   ```

4. **Shannon entropy in bits**

   ```Python
   shannon_bits = -np.sum(probs * np.log2(probs), axis=1)
   ```

So I'd rename your output columns to:

```Python
metrics_chunk = pd.DataFrame(
    {
        "total_kmer_counts": row_sums.astype("int64"),
        "num_nonzero_kmers": total_nonzero,
        "shannon_entropy_nats": shannon_nats,
        "shannon_entropy_bits": shannon_bits,
    },
    index=kmer_df.index,
)
```

That is much more scientifically precise.

---

## But what if you actually want "evenness"?

Then you should **calculate it separately**, rather than calling entropy "evenness."

Shannon entropy is:

$$H = -\sum_i p_i \ln(p_i)$$

It depends on both:

- how many k-mer types are present
- how evenly their abundances are distributed

Shannon **evenness** normalizes that entropy by the maximum possible entropy given the number of observed k-mer types:

$$J = \frac{H}{\ln(S)}$$

where:

- $H$ = Shannon entropy in nats
- $S$ = number of nonzero k-mer types

So you could calculate:

```Python
num_nonzero_kmers = (values != 0).sum(axis=1)

shannon_entropy_nats = -np.sum(
    np.where(positive, probs * np.log(probs), 0.0),
    axis=1,
)

shannon_evenness = np.where(
    num_nonzero_kmers > 1,
    shannon_entropy_nats / np.log(num_nonzero_kmers),
    1.0,
)
```

Then your outputs could be:

```
total_kmer_counts
num_nonzero_kmers
shannon_entropy_nats
shannon_entropy_bits
shannon_evenness
```

### I'd recommend this approach

If this pipeline is intended for biological/ecological interpretation, I'd use the five metrics above.

For example:

| Metric                 | Meaning                                          |
| ---------------------- | ------------------------------------------------ |
| `total_kmer_counts`    | Total number of k-mer observations in the sample |
| `num_nonzero_kmers`    | Number of different k-mer types detected         |
| `shannon_entropy_nats` | Shannon entropy using natural logarithm          |
| `shannon_entropy_bits` | Same entropy using log₂                          |
| `shannon_evenness`     | Normalized Shannon entropy, 0–1                  |

One important detail: **`shannon_entropy_nats` and `shannon_entropy_bits` contain the same information**, just expressed in different units. Since

$$H_{\text{bits}} = \frac{H_{\text{nats}}}{\ln 2}$$

you don't necessarily need both unless you specifically want both units.

I'd **drop `shannon_diversity` unless you specifically need entropy in bits**.

One other thing I'd address at the same time: your current code has a potentially significant memory problem (`chunks = list(reader)`), so if you're going to modify `kmer_stats.py`, **I'd fix the streaming/memory behavior at the same time as the metric naming** rather than making two separate changes.
