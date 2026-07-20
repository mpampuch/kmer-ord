# src/kmer_ord/io/sparse_matrix.py
"""
Sparse k-mer matrix format and conversion helpers.

K-mer count matrices are extremely sparse for read-level data (a ~7 kb read
touches at most a few thousand of the hundreds of thousands / millions of
canonical k-mer columns), so storing them densely wastes almost all the memory
on zeros. This module keeps the matrix as a scipy CSR matrix and persists it as
a compressed ``.npz`` bundle (counts + sequence ids + k-mer length), converting
the dense ``.npy`` emitted by the external Rust counter without ever holding a
full dense copy in RAM.
"""
import os
from pathlib import Path

import numpy as np
from scipy import sparse

# The Rust counter emits 32-bit counts; we standardise on uint32 everywhere.
COUNT_DTYPE = np.uint32


def dense_npy_to_csr(npy_path, max_block_bytes: int = 128 * 1024 * 1024):
    """
    Convert a dense ``.npy`` count matrix on disk into a CSR matrix, reading it
    in row blocks so peak RAM stays bounded to roughly one block plus the
    growing sparse result (never the full dense matrix).

    Parameters
    ----------
    npy_path : path-like
        Dense matrix written by the Rust k-mer counter (rows = sequences).
    max_block_bytes : int
        Upper bound on the dense bytes materialised at once. The block height is
        derived from the row width so a single dense block stays under this cap.
    """
    memmap = np.load(npy_path, mmap_mode="r")
    n_rows, n_cols = memmap.shape

    bytes_per_row = max(1, n_cols * COUNT_DTYPE(0).nbytes)
    block_rows = max(1, int(max_block_bytes // bytes_per_row))

    blocks = []
    for start in range(0, n_rows, block_rows):
        end = min(start + block_rows, n_rows)
        # Materialise only this block densely, then immediately compress it.
        dense_block = np.asarray(memmap[start:end], dtype=COUNT_DTYPE)
        blocks.append(sparse.csr_matrix(dense_block))

    # Drop the memmap reference so the OS can release the mapping promptly.
    del memmap

    if not blocks:
        return sparse.csr_matrix((0, n_cols), dtype=COUNT_DTYPE)
    return sparse.vstack(blocks, format="csr")


def save_sparse_matrix(path, matrix, sequence_ids, kmer_length: int):
    """
    Persist a CSR matrix plus its row labels and k-mer length as a single
    compressed ``.npz`` bundle. This is the primary k-mer matrix artifact.
    """
    matrix = matrix.tocsr()
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
        sequence_ids=np.asarray(sequence_ids, dtype="U"),
        kmer_length=np.asarray(int(kmer_length), dtype=np.int64),
    )


def load_sparse_matrix(path):
    """
    Load a ``.npz`` bundle written by :func:`save_sparse_matrix`.

    Returns ``(csr_matrix, sequence_ids, kmer_length)``. ``allow_pickle`` is
    disabled so untrusted files cannot execute arbitrary code on load.
    """
    with np.load(path, allow_pickle=False) as npz:
        matrix = sparse.csr_matrix(
            (npz["data"], npz["indices"], npz["indptr"]),
            shape=tuple(int(dim) for dim in npz["shape"]),
        )
        sequence_ids = npz["sequence_ids"]
        kmer_length = int(npz["kmer_length"])
    return matrix, sequence_ids, kmer_length


def write_matrix_tsv(path, matrix, sequence_ids, kmer_length: int):
    """
    Write the human-readable dense TSV (``sequence_id`` + canonical k-mer
    columns), streaming one row at a time so the full dense matrix is never held
    in memory. Kept byte-compatible with the pipeline's original TSV output.
    """
    # Imported lazily to avoid a circular import at module load time.
    from kmer_ord.io.kmer_counter import canonical_kmers

    matrix = matrix.tocsr()
    kmer_keys = canonical_kmers(kmer_length)
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    with open(path, "w") as handle:
        handle.write("sequence_id\t" + "\t".join(kmer_keys) + "\n")
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i).toarray().ravel()
            handle.write(str(sequence_ids[i]) + "\t" + "\t".join(map(str, row)) + "\n")
