# This file suppresses any "Importing BiopythonWarning" that 
# you get from running this so many conda environments as subprocesses. 
# They were polluting the logs. In this file, they will be filtered out whether you
# run the CLI tool or are running this program from the benchmarking entrypoints.
import warnings

warnings.filterwarnings(
    "ignore",
    message="You may be importing Biopython from inside the source tree.*",
)
