Current kmer-ord memory usage
Currently kmer-ord uses 9 dimensionality reduction techniques:
PCA
t-SNE
UMAP
TriMAP
PacMAP
LocalMAP
Sparse PCA
Kernel PCA
Local Linear Embedding
Sparse PCA, Kernel PCA,Local Linear Embedding are not as important and may get removed so shouldn’t focus too much on fixing them

Current Potential Memory Issues:
K-MER counting
K-mer counting is done by external Rust tool, saved into .npy compressed file:
Double check if entire Matrix is loaded into RAM while this is being done or if the algorithm is constructing the .npy file in a memory efficient way
The matrix uses u32bit encoding (4 bytes), so the potential memory usage here is approximately:

𝑛×4^𝑘×"4 bytes"

    where 𝑛 is the number of reads, and 𝑘 is the k-mer size.

The downstream output .tsv table will be about:
𝑛×〖(4〗^𝑘/2) ×"4 bytes"
when loaded in RAM, because canonical k-mers will roughly cut this in half
The .tsv file will also have some memory allocated for strings and the palindromic k-mers that don’t have reverse compliments, but these can be ignored

To do: Double check whether this is how much RAM is used at this stage or if it’s more memory efficient.
K-MER counting
After counting the results should be stored into a .npy file, so just ensure that any memory associated with the K-mer counting is cleared

To do: Check whether this is the case
K-MER stats
The whole k-mer matrix is loaded into RAM for stats
Despite having a chunking option in the code, a lazy chunk reader is created but then every file is iterated and every chunk is stored into memory anyways.
This line reads the TSV incrementally:
`reader = pd.read_csv(..., chunksize=chunksize)`
But this line immediately defeats the memory benefit of chunking:
`chunks = list(reader)`
The entire k-mer matrix is loaded into RAM as a list of dataframe chunks.
The matrix itself is not one giant dataframe, but all of its chunks are simultaneously resident in memory.
And right after that,

```
if cpus <= 1:
    results = [calculate_kmer_metrics_chunk(chunk) for chunk in chunks]
else:
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        futures = [executor.submit(calculate_kmer_metrics_chunk, chunk) for chunk in chunks]
        results = [f.result() for f in futures] # collect in submission order
```

Here because you have already loaded the chunks all in the original memory, a copy of each chunk is getting sent to each worker process, and so this is at least doubling the memory when you’re trying to process them concurrently.
K-MER stats
Another side note is this:
The RUST program produces a .npy file that stores everything in u32int format.
Then in kmer_stats.py, all the raw k-mer counts get down-casted to u16int

```
dtypes = {0: "str"} # index column
for col in range(1, num_columns - 1):
dtypes[col] = "uint16"
return dtypes
```

Then immediately after this, in the calculate_kmer_metrics_chunk function, all the data gets converted to float64
`values = numeric.to_numpy(dtype=float)`
That would be  
𝑛×〖(4〗^k/2)×"8 bytes"
And combine this with the broken chunking mentioned in the last slide, the initial matrix (16bit, 2 bytes) gets copied (and casted to float64, 8 bytes), so what your have currently is:

(𝑛×〖(4〗^k/2)×"4) + "(𝑛×〖(4〗^k/2)×" 8) bytes"

Plus some more memory that might get allocated as temporary arrays during the statistics calculations, but the python garbage collector should be able to handle this I don’t this is an issue.
This part of the program is currently very flawed. It isn’t chunking as expected and it’s using WAY TOO MUCH needless RAM to do something like just calculate some statistics.
Even though I think it’s very unlikely that down-casting the raw k-mer counts from u32 to u16 will result in information loss, because it means a single read would need to have a specific kmer at least 65,535 times, it’s still probably an unnecessary operation and it risks breaking the program if someone does try to run it with very small values of k that actually might have values that high on lets say long ONT-sequencing reads
But definitely, converting the data to float64 to do basic stats is super overkill, especially since only up to 3 decimal places are retained in the end.
Just by using float32 embeddings for the k-mer matrix here, you can drop the RAM use by half without having to do anything else
The conversion will still have to allocate a whole temporary array to copy over the numbers (maybe it does this more efficiently than just allocated a whole copy of the matrix block in RAM) but in the worst case it still needs to allocate 2x the size of the matrix in RAM concurrently, whereas the previous implementation would need to do 3x the size of the RAM concurrently (1 for the og matrix, +2 for the double sized 64bit matrix)
K-MER stats
There may be a problem with how the statistics are named, because they may not actually be showing what they claim to be showing:
total_nonzero_kmers
is actually more like total_kmer_counts
num_nonzero_kmers
Is actually more like num_nonzero_kmers
shannon_evenness
Is actually more like shannon_nats
shannon_diversity
Is more like shannon_bits
K-MER stats
To do:
Fix the chunking problem so that the chunks are loaded as needed to the CPUs as intended, and not all at once + copies
Don’t downcast to 16bits and don’t use 32bit floats data
Or maybe in the case that k is below some threshold (e.g. 2 or 3), use 64 bit in case the values of k produce very large counts to retain enough decimal precision, but I don’t know how necessary this is. Could be a good safeguard.
Make sure stats are calculated correctly
K-MER stats
After calculating stats, any RAM that was associated with calculating the stats should be freed.
This likely the case due to the context runner. Once these functions go out of scope the memory should be available.

To do: Just double check this is the case
Matrix Preparation
In the operations.py script, in the MatrixPreprocessing class there is a line that calls `matrix = load_matrix(matrix_path)`, so this gets automatically loaded into RAM as a pandas dataframe
Means you already are using 𝑛×〖(4〗^𝑘/2)×"4 bytes" here

But then inside when you call preprocess_data from preprocess.py, the internal function does this. With the data:
X = df.copy().astype(np.float32) # ensure float32

So you’re potentially using at least 2(𝑛×〖(4〗^𝑘/2)×"4) bytes" at this stage

Instead of doing this, load the kmer-matrix from the k-mer counting step (which should be in tsv), and then do the float32 conversion while parsing, so the whole K-MER matrix is only loaded once in RAM.

```
matrix = pd.read_csv(matrix_path,  sep="\t", index_col=0, dtype=np.float32)
# and remove unneccessarty X = df.copy().astype(np.float32) from preprocess.py
```

This should get the memory usage back down to
𝑛×〖(4〗^𝑘/2)×"4 bytes"
Matrix Preparation - Normalizations
I’m going to ignore analyzing all the other normalization methods because only CLR should be used:

The current CLR calculation potentially produces many temporary allocation arrays because of the 4 operations, X, np.log(X), geometric_mean, X.div(...), np.log(...)
Original in program

- Computes the geometric mean and the division separately
  Standard log-difference version
  Avoids calculating geometric mean and division
  Computes the manthemically simplified form directly

Benchmarks on whole dataset suggests this produces the same output matrix and is at least ~50% more RAM efficient (+ faster)

Matrix Preparation
An idea I have is to also let the user choose the type of PCA used for --pca-pre
--pca-pre-method (pca, ipca)
(Variance PCA or Incremental PCA)

For many datasets containing large amounts of reads, this stage should be enough to reduce the matrix size down to a manageable size and then the rest of the Dimensionality Reduction techniques should probably be able to work with this

.
Screening Parameters:
This just runs iterations of DR methods, and the peak memory here is the memory used for 1 of the DR runs. Peak memory will be the worst fit of the worst permutation of method + hyperparameters.

The PEAK memory at this stage is going to be the k-mer matrix in RAM + memory of the DR run and it’s internal neighbour graph construction. This can be very memory intensive.

There’s a small memory accumulation error I think when collecting the results from the screen. The results contain the x and y embeddings and sequence ID from every screening run. Keeping the screening ID is probably just and this can probably cause a few extra GB of RAM use for no reason. This won’t be the main bottleneck at this stage because the DR and Kmer matrix will be more expensive but is probably an easy bug to fix and can potentially prevent the program from OOM crashing in cases where the DR is teetering on the RAM limit
Only doing the checks for the 6 main methods

But for all of the methods, in the `operations.py` there is a memory check function that eximates the peak memory as `est_peak = X.nbytes \* 4 / (1024 \*\* 3).
But I don’t really know where this formula comes from or if it accurately reflects all the internal neighbour graphs (which can be O(n_reads x n_neighbours) I think for UMAP and PacMAP

Every method then does `fit_transform(X)` on that same object. Methods run **sequentially**, so you pay for `X` plus **one** fit,
RAM-Efficient Workflow

Input
↓
Convert to Rₘ
↓
Rₘ data
↓
K-means counting ✓ RAM efficient
↓
Raw k-means stats ← Needs fixing: Duplicated matrix allocations currently
↓
Matrix preprocessing ← Issue: k-means matrix loaded in RAM twice currently
↓
Normalization ← Fix to use standardized-log data + CLR identity to avoid unnecessary temp arrays
↓
Pre-PCA ← Use incremental PCA to avoid PCA internal float32 copy. Reduce to ≥100 PCA, I think
↓
Hyperparameter + DR methods

Additional notes
For small accumulation error that adds:
n_reads × n_samples × n_features x seq_id_string_bytes

With the changes above, the program should be able to get:
~ n_reads x 4^k × 4 bytes
For the peak RAM cost of the k-means matrix (pre-canonical-correction)
After Pre-PCA, the downstream DR methods should be memory safe on the large datasets.
