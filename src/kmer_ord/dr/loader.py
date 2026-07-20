# src/kmer_ord/dr/loader.py
from pathlib import Path


def load_matrix_any(matrix_path):
    """
    Load a k-mer matrix in any supported format, returning
    ``(matrix, sequence_ids)`` where ``matrix`` is a scipy CSR matrix.

    - ``.npz``  -> sparse CSR bundle written by the k-mer counter (preferred).
    - ``.tsv``/``.csv`` -> dense table (first column = sample ids), converted to
      CSR. Kept for backward compatibility with matrices produced before the
      sparse format existed.
    - ``.npy``  -> dense array (no sample ids; positional ids are generated).
    """
    import numpy as np
    from scipy import sparse

    matrix_path = Path(matrix_path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"K-mer matrix not found: {matrix_path}")

    suffix = matrix_path.suffix.lower()

    if suffix == ".npz":
        from kmer_ord.io.sparse_matrix import load_sparse_matrix

        matrix, sequence_ids, _ = load_sparse_matrix(matrix_path)
    elif suffix in (".tsv", ".csv"):
        import pandas as pd

        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(matrix_path, sep=sep, index_col=0)
        df = df.apply(pd.to_numeric, errors="raise")
        matrix = sparse.csr_matrix(df.to_numpy())
        sequence_ids = df.index.to_numpy()
    elif suffix == ".npy":
        arr = np.load(matrix_path)
        matrix = sparse.csr_matrix(arr)
        sequence_ids = np.arange(arr.shape[0])
    else:
        raise ValueError(f"Unsupported matrix format: {suffix}")

    if matrix.shape[0] < 2:
        raise ValueError("Matrix must contain at least 2 samples for DR.")

    return matrix, sequence_ids
