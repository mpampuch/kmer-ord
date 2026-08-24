# Memory Usage of `CobiontID/kmer-counter`

Short answer: **it does not store the full `n × 4^k` matrix in RAM** before writing output.

It streams rows to a temporary binary file, then prepends an `.npy` header and copies the temporary file into the final output.

## Key Evidence

In `src/main.rs`:

```rust
let tmp_path = Temp::new_file()?;

{
    let file = File::create(&tmp_path)?;
    let mut file = BufWriter::new(file);
```

The `write_header` function writes the `.npy` header and then streams the temporary data:

```rust
let mut fw = BufWriter::new(File::create(&out)?);
... // writes NPY magic+header
let mut bin_src = File::open(&tmp_path)?;
std::io::copy(&mut bin_src, &mut fw)?;
```

## Memory Implications

- The formula `n × 4^k × 4 bytes` represents the **output file size** (or the equivalent RAM requirement if the entire matrix were materialized), **not peak RAM usage**.
- Peak RAM is roughly dominated by:

  - k-mer key structures (`product`, `keys`, base hashmap), which scale with `4^k`
  - one per-read count map clone (`k_counts_it`), which also scales with `4^k`
  - parser and buffer overhead

Therefore, runtime memory is approximately **O(4^k)** rather than **O(n · 4^k)**, which is consistent with the tool's README claim for v0.1.2.

## Conclusion

**Yes, this tool is memory-efficient with respect to the number of reads (`n`).** It writes the matrix incrementally and does **not** hold the full `n × 4^k` matrix in memory.
