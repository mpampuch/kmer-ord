# src/kmer_ord/io/summary.py
"""Per-sequence FASTA statistics, streamed over byte-range chunks.

Memory design: workers never `read()` a whole chunk. They iterate records in
[start, end), write each TSV row immediately, and return only a compact
lengths array plus Welford moments for GC. Peak RAM scales with one sequence
(plus the lengths vector), not with the FASTA size times the worker count.
"""
from pathlib import Path
import shutil

import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser

from kmer_ord.utils.logging_utils import section, info


def calculate_nx(sequence_lengths, target_percentage=50):
    """Calculate Nx statistic. Returns (Nx value, largest sequence length)."""
    arr = np.asarray(sequence_lengths, dtype=np.int64)
    if arr.size == 0:
        return 0, 0
    sorted_lengths = np.sort(arr)[::-1]
    total_length = int(sorted_lengths.sum())
    target_length = total_length * target_percentage / 100.0
    cumulative = np.cumsum(sorted_lengths, dtype=np.int64)
    idx = int(np.searchsorted(cumulative, target_length, side="left"))
    if idx >= sorted_lengths.size:
        idx = sorted_lengths.size - 1
    return int(sorted_lengths[idx]), int(sorted_lengths[0])


def _find_chunk_starts(fasta_path: Path, n_chunks: int) -> list:
    """
    Return byte offsets of record-boundary starts that divide the file into
    at most n_chunks pieces. Each returned position points to a '>' character.
    Always returns at least [0].
    """
    file_size = fasta_path.stat().st_size
    if file_size == 0 or n_chunks <= 1:
        return [0]

    chunk_size = file_size // n_chunks
    starts = [0]

    with open(fasta_path, "rb") as f:
        for i in range(1, n_chunks):
            target = i * chunk_size
            if target >= file_size:
                break
            f.seek(target)
            f.readline()  # finish the current partial line
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    if pos not in starts:
                        starts.append(pos)
                    break

    return starts


class _ChunkLineHandle:
    """Iterate decoded FASTA lines in the byte range [current, end).

    `_find_chunk_starts` places `end` on a '>' so this range is a complete
    set of records. Stopping at `end` (checked before each readline) prevents
    worker 0 from consuming the rest of the file.
    """

    def __init__(self, binary_file, end):
        self._f = binary_file
        self._end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self._end is not None and self._f.tell() >= self._end:
            raise StopIteration
        line = self._f.readline()
        if not line:
            raise StopIteration
        return line.decode("ascii", errors="replace")


def _process_fasta_chunk(args: tuple) -> tuple:
    """
    Worker: stream records in bytes [start, end), write TSV rows immediately.

    Returns (lengths, n, gc_mean, gc_m2). Defined at module level so it is
    picklable for multiprocessing spawn on macOS.
    """
    fasta_path, start, end, tsv_path = args
    tsv_path = Path(tsv_path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    lengths: list[int] = []
    n = 0
    gc_mean = 0.0
    gc_m2 = 0.0

    with open(fasta_path, "rb") as f, tsv_path.open("w") as out:
        f.seek(start)
        for title, seq in SimpleFastaParser(_ChunkLineHandle(f, end)):
            seq_id = title.split()[0]
            length = len(seq)
            if length == 0:
                gc = at = 0.0
            else:
                s = seq.upper()
                gc = (s.count("G") + s.count("C")) / length * 100
                at = (s.count("A") + s.count("T")) / length * 100
            out.write(f"{seq_id}\t{length}\t{gc}\t{at}\n")
            lengths.append(length)
            n += 1
            delta = gc - gc_mean
            gc_mean += delta / n
            gc_m2 += delta * (gc - gc_mean)

    return np.asarray(lengths, dtype=np.int64), n, gc_mean, gc_m2


def _combine_welford(parts: list[tuple[int, float, float]]) -> tuple[int, float, float]:
    """Merge parallel Welford (n, mean, M2) triples (Chan et al.)."""
    n = 0
    mean = 0.0
    m2 = 0.0
    for n_b, mean_b, m2_b in parts:
        if n_b == 0:
            continue
        delta = mean_b - mean
        n_total = n + n_b
        mean += delta * n_b / n_total
        m2 += m2_b + delta * delta * n * n_b / n_total
        n = n_total
    return n, mean, m2


def _concat_chunk_tsvs(chunk_paths: list[Path], dest: Path) -> None:
    """Write a single headered TSV from headerless chunk files, then delete them."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as out:
        out.write("sequence_id\tLength\tGC_Content\tAT_Content\n")
        for path in chunk_paths:
            with path.open() as inp:
                shutil.copyfileobj(inp, out)
            path.unlink(missing_ok=True)


def calculate_stats(context):
    """
    Calculate per-sequence and overall stats for the canonical FASTA in context.
    Uses parallel chunked processing when context.threads > 1.

    Returns (overall_file, tsv_file). The per-sequence table is on disk, not
    a DataFrame — callers that need rows should read the TSV.
    """
    section("Calculating per-sequence statistics...")
    from concurrent.futures import ProcessPoolExecutor

    fasta_file = context.get("fasta")
    threads = getattr(context, "threads", 1)

    overall_file = context.artifact_path("overall_stats", subdir="summary", suffix=".txt")
    tsv_file = context.artifact_path("stats_per_sequence", subdir="summary", suffix=".tsv")
    tsv_file.parent.mkdir(parents=True, exist_ok=True)

    chunk_starts = _find_chunk_starts(fasta_file, n_chunks=threads)
    chunk_ends = chunk_starts[1:] + [None]
    chunk_paths = [
        tsv_file.parent / f"{tsv_file.stem}.chunk{i}{tsv_file.suffix}"
        for i in range(len(chunk_starts))
    ]
    tasks = [
        (fasta_file, start, end, chunk_path)
        for start, end, chunk_path in zip(chunk_starts, chunk_ends, chunk_paths)
    ]

    if len(tasks) == 1:
        results = [_process_fasta_chunk(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(_process_fasta_chunk, task) for task in tasks]
            # submission order preserves FASTA record order in the concatenated TSV
            results = [future.result() for future in futures]

    length_parts = [r[0] for r in results]
    if all(arr.size == 0 for arr in length_parts):
        for path in chunk_paths:
            path.unlink(missing_ok=True)
        raise RuntimeError(f"No sequences found in {fasta_file}")

    _concat_chunk_tsvs(chunk_paths, tsv_file)

    lengths_arr = np.concatenate(length_parts)
    total_seqs = int(lengths_arr.size)
    total_length = int(lengths_arr.sum())
    n50, _ = calculate_nx(lengths_arr, target_percentage=50)
    n90, _ = calculate_nx(lengths_arr, target_percentage=90)

    n_gc, avg_gc, gc_m2 = _combine_welford([(r[1], r[2], r[3]) for r in results])
    if n_gc > 1:
        std_gc = float(np.sqrt(gc_m2 / (n_gc - 1)))
    else:
        std_gc = 0.0

    with overall_file.open("w") as f:
        f.write(f"Total nr of sequences: {total_seqs}\n")
        f.write(f"Total Length: {total_length} bp\n")
        f.write(f"N50: {n50} bp\n")
        f.write(f"N90: {n90} bp\n")
        f.write(f"Average GC Content: {avg_gc:.2f}%\n")
        f.write(f"GC Content Standard Deviation: {std_gc:.2f}\n")

    w = 20
    info(f"{'sequences':<{w}}  {total_seqs:>14,}")
    info(f"{'total length':<{w}}  {total_length:>11,} bp")
    info(f"{'N50':<{w}}  {n50:>11,} bp")
    info(f"{'N90':<{w}}  {n90:>11,} bp")
    info(f"{'average GC':<{w}}  {avg_gc:>13.2f}%")
    info(f"{'GC std dev':<{w}}  {std_gc:>14.2f}")

    context.register("summary_overall", overall_file)
    context.register("summary_per_sequence", tsv_file)

    return overall_file, tsv_file
