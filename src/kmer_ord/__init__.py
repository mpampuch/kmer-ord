# Filter before any `from Bio import ...`. Importing BiopythonWarning
import warnings

warnings.filterwarnings(
    "ignore",
    message="You may be importing Biopython from inside the source tree.*",
)
