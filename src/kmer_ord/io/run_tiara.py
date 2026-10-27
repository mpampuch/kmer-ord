#src/kmer_ord/io/run_tiara.py
import re
from pathlib import Path
import pandas as pd
from kmer_ord.system.env_manager import TIARA_ENV
from kmer_ord.utils.logging_utils import console, run_logged_command, section

# Tiara always prints the final statistics. "-v" also prints the stage
# sentences; kmer-ord --verbose decides whether the rest of that stream,
# and the model log, are shown.
_CLASS_COUNT = re.compile(r"[^:]+:\s*\d+")


def _tiara_summary_keep():
    """Keep stage sentences, the timing line, and the class-count block.

    Tiara prints a bare "Done" after each stage. Name the stage on that line
    so the summary says whether the first or second stage finished.
    """
    in_stats = False
    stage = None

    def keep(line: str) -> bool:
        nonlocal in_stats, stage
        text = line.strip()
        lowered = text.lower()
        # "Calculating first stage ..." and "Performing second stage ..." both
        # name the stage. Remember the latest one for the following "Done".
        if "first stage" in lowered or "second stage" in lowered:
            stage = "First" if "first stage" in lowered else "Second"
            return True
        if text == "Done":
            if stage:
                console.print(f"{stage} stage done", markup=False, highlight=False)
                return False
            return True
        if "iteration statistics" in lowered:
            in_stats = True
            return True
        if in_stats:
            if _CLASS_COUNT.fullmatch(text):
                return True
            in_stats = False
        if text.startswith("Classification took"):
            return True
        return False

    return keep


def _print_tiara_log(output_file: Path) -> None:
    log_path = output_file.parent / f"log_{output_file.name}"
    if not log_path.is_file():
        return
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        console.print(line, markup=False, highlight=False)


def run_tiara(input_file, output_file, threads=1, script_name="tiara", verbose=False):
    """
    Run Tiara inside its dedicated conda environment.
    """
    section("Running Tiara on reads...")
    input_file = Path(input_file)
    output_file = Path(output_file)

    cmd = ["conda", "run", "--no-capture-output", "-n", TIARA_ENV, "tiara",
           "-i", str(input_file),
           "-o", str(output_file),
           "-t", str(threads),
           "-v"]

    run_logged_command(cmd, check=True, verbose=verbose, keep=_tiara_summary_keep())

    # Tiara writes the full FASTA header (e.g. "read_1 length=... origin=...")
    # into its sequence_id column, but every other kmer-ord table truncates to
    # the first whitespace token (io/summary.py). Left unfixed, the join in
    # FeatureMerge never matches and Tiara columns end up all-null in the
    # database. Normalise here, at the source, so every consumer of this file
    # (FeatureMerge, `run-tiara` standalone output, manual inspection) sees
    # IDs consistent with the rest of the pipeline.
    df = pd.read_csv(output_file, sep="\t")
    df["sequence_id"] = df["sequence_id"].astype(str).str.split().str[0]
    df.to_csv(output_file, sep="\t", index=False)

    if verbose:
        _print_tiara_log(output_file)

    return output_file