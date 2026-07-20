# src/kmer_ord/io/kmer_counter.py
import subprocess
import shutil
import tempfile
from kmer_ord.utils.benchmark import BenchmarkTimer
from pathlib import Path

from kmer_ord.system.env_manager import TOOLS_ENV, run_in_env
from kmer_ord.utils.logging_utils import section, info

def format_size(size_in_bytes):
    return f"{size_in_bytes / (1024*1024):,.2f} MB".replace(",", ".")

def canonical_kmers(k):
    from itertools import product

    acgt = ['A', 'C', 'G', 'T']
    rev_comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    product_kmers = [''.join(p) for p in product(acgt, repeat=k)]
    canon_set = set()
    for kmer in product_kmers:
        rev = ''.join(rev_comp[base] for base in reversed(kmer))
        canon_set.add(min(kmer, rev))
    return sorted(list(canon_set))


def run_kmer_counter(input_file, output_matrix, kmer_length, num_threads,
                     write_tsv=False, script_name="kmer-counter"):
    """
    Run kmer-counter from TOOLS_ENV and save a sparse ``.npz`` k-mer matrix.

    The external Rust counter writes a dense ``.npy``; we convert it to a CSR
    matrix in bounded row blocks (never a full dense copy in RAM) and persist it
    as a compressed ``.npz``. The dense human-readable TSV is only written when
    ``write_tsv=True`` and is streamed one row at a time.

    Parameters
    ----------
    output_matrix : path-like
        Destination for the sparse ``.npz`` matrix artifact.
    write_tsv : bool
        Also emit a dense ``.tsv`` alongside the ``.npz`` (off by default; the
        dense TSV is huge at large k).
    """
    from Bio import SeqIO
    from concurrent.futures import ThreadPoolExecutor

    from kmer_ord.io.sparse_matrix import (
        dense_npy_to_csr,
        save_sparse_matrix,
        write_matrix_tsv,
    )

    section(f"Running k-mer counting (k={kmer_length})...")
    temp_dir = Path(tempfile.mkdtemp(prefix="kmer_counter_temp_"))
    input_args = f"--input {input_file} --kmer {kmer_length} --threads {num_threads}"

    with BenchmarkTimer("Kmer_Counter_Run", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        cmd = ["kmer-counter",
               "--file", str(input_file),
               "--ids", str(temp_dir / "sequence_headers.txt"),
               "--klength", str(kmer_length),
               "--out", str(temp_dir / "kmer_counts.npy"),
               "--collapse", "1"]

        #run inside conda environment
        run_in_env(TOOLS_ENV,
                   cmd,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # Convert the dense .npy to a sparse CSR matrix without a full dense copy.
    npy_file = temp_dir / "kmer_counts.npy"
    with BenchmarkTimer("Sparse_Matrix_Build", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        kmer_matrix = dense_npy_to_csr(npy_file)
        n_rows, n_cols = kmer_matrix.shape
        density = kmer_matrix.nnz / (n_rows * n_cols) if n_rows and n_cols else 0.0
        info(f"sparse matrix: ({n_rows}, {n_cols}) "
             f"nnz={kmer_matrix.nnz:,} density={density:.4%} "
             f"{format_size(_csr_nbytes(kmer_matrix))}")

    info("Extracting sequence headers from fasta...")
    with BenchmarkTimer("Sequence_Headers_Extraction", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            sequence_headers = list(executor.map(lambda r: r.id, SeqIO.parse(input_file, "fasta")))

    info("Saving sparse matrix (.npz)...")
    with BenchmarkTimer("Sparse_Matrix_Save", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        save_sparse_matrix(output_matrix, kmer_matrix, sequence_headers, kmer_length)

    if write_tsv:
        output_tsv = Path(output_matrix).with_suffix(".tsv")
        info("Generating output tsv...")
        with BenchmarkTimer("TSV_Composition", script_name=script_name,
                            input_file=input_file, input_args=input_args):
            write_matrix_tsv(output_tsv, kmer_matrix, sequence_headers, kmer_length)

    with BenchmarkTimer("Cleanup", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        shutil.rmtree(temp_dir)

    info(f"Output saved: {output_matrix}")
    return output_matrix


def _csr_nbytes(matrix) -> int:
    """Total bytes backing a CSR matrix (data + indices + indptr)."""
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
