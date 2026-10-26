#src/kmer_ord/io/run_rdna.py
from pathlib import Path
import subprocess
import pandas as pd
from kmer_ord.system.env_manager import RDNA_ENV
from kmer_ord.utils.logging_utils import section, warn, info

RDNA_COLUMNS = {
    "domain_clean": "rdna_domain",
    "supergroup": "rdna_supergroup",
    "division": "rdna_division",
    "subdivision": "rdna_subdivision",
    "class": "rdna_class",
    "order": "rdna_order",
    "family": "rdna_family",
    "genus": "rdna_genus",
}

def run_rdna_miner(
    input_file,
    output_dir,
    feature_file,
    threads=1,
    platform="auto",
    db_dir=None,
    rfam_db=None,
    pr2_db=None,
    silva_db=None,
    force=False,
):
    """
    Run rDNA-miner (barrnap -> Flye assembly -> Rfam/cmscan -> DECIPHER ->
    minimap2 back-mapping) inside its dedicated conda environment, then
    condense its per-read taxonomy table into a clean sequence_id-keyed
    feature TSV at `feature_file`.

    rDNA-miner is a multi-tool external pipeline (barrnap, Flye, Infernal,
    R/DECIPHER) with more failure surface than Tiara's single classifier, so
    unlike Tiara this does not raise on failure: a broken run logs a warning
    and returns None, letting the rest of the kmer-ord pipeline continue
    without rDNA features rather than aborting the whole project run.

    Database paths are forwarded only when set. rDNA-miner otherwise uses
    RDNA_MINER_DB, then ~/.rdna-miner/db. `force` is rdna-miner's own
    --force, so a rerun also rebuilds cached steps under the raw directory.
    """
    section("Running rDNA-miner on reads...")
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["conda", "run", "-n", RDNA_ENV, "rdna-miner", "long-read",
           "-i", str(input_file),
           "-o", str(output_dir),
           "-t", str(threads),
           "-p", platform]
    # Omit unset paths so rDNA-miner keeps its own default resolution.
    if db_dir is not None:
        cmd += ["--db-dir", str(db_dir)]
    if rfam_db is not None:
        cmd += ["--rfam-db", str(rfam_db)]
    if pr2_db is not None:
        cmd += ["--pr2-db", str(pr2_db)]
    if silva_db is not None:
        cmd += ["--silva-db", str(silva_db)]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        warn(f"rDNA-miner failed; continuing without rDNA features.\n{result.stderr}")
        return None

    taxonomy_file = output_dir / "taxonomy" / f"{input_file.stem}_taxa_ssu_reads.tsv"
    if not taxonomy_file.exists():
        warn(f"rDNA-miner produced no read-level taxonomy table ({taxonomy_file} missing); "
             "continuing without rDNA features.")
        return None

    df = pd.read_csv(taxonomy_file, sep="\t")
    if df.empty:
        warn("rDNA-miner found no rDNA reads; continuing without rDNA features.")
        return None

    df["sequence_id"] = df["QNAME"].astype(str).str.split().str[0]

    # rDNA-miner's own read->contig mapping keeps supplementary/chimeric SAM
    # records (only unmapped reads are filtered), so a read spanning an
    # ambiguous region can carry two rows with different taxonomy calls.
    # Keep the higher-confidence call per read so sequence_id stays unique,
    # matching what FeatureMerge requires. genus_confidence is only used for
    # this tie-break and is not itself kept as a feature column.
    df = df.sort_values("genus_confidence", ascending=False).drop_duplicates(
        subset="sequence_id", keep="first")

    df = df.rename(columns=RDNA_COLUMNS)
    df = df[["sequence_id"] + list(RDNA_COLUMNS.values())]

    feature_file = Path(feature_file)
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(feature_file, sep="\t", index=False)

    info(f"rDNA-miner: {len(df)} reads assigned taxonomy -> {feature_file}")
    return feature_file
