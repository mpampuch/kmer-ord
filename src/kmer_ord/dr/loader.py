# src/kmer_ord/dr/loader.py
from pathlib import Path
import pandas as pd

def load_matrix(matrix_path: Path) -> pd.DataFrame:
    """
    Load k-mer matrix from TSV, CSV, or NPY file.
    First column is sample IDs; rest are numeric features.
    Returns pd.DataFrame of float32 with sample IDs as index.

    Numeric columns are parsed directly into float32 so the matrix exists in
    RAM exactly once at its final datatype.
    """
    import numpy as np
    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(f"K-mer matrix not found: {matrix_path}")

    suffix = matrix_path.suffix.lower()

    if suffix in [".tsv", ".csv"]:
        sep = "\t" if suffix == ".tsv" else ","
        # peek at the header first: positional dtypes need the column count
        # before read_csv can parse straight into float32
        with open(matrix_path) as f:
            num_columns = len(f.readline().rstrip("\n").split(sep))

        # positional dtypes: string index, float32 everywhere else; a
        # non-numeric feature value fails here at parse time (ValueError)
        dtypes: dict[int, type] = {0: str}
        for col in range(1, num_columns):
            dtypes[col] = np.float32

        df = pd.read_csv(matrix_path, sep=sep, index_col=0, dtype=dtypes)
    elif suffix == ".npy":
        arr = np.load(matrix_path)
        df = pd.DataFrame(arr.astype(np.float32, copy=False))
    else:
        raise ValueError(f"Unsupported matrix format: {suffix}")

    if df.shape[0] < 2:
        raise ValueError("Matrix must contain at least 2 samples for DR.")

    return df
