#src/kmer_ord/io/run_rdna.py
import re
from pathlib import Path
import pandas as pd
from kmer_ord.system.env_manager import RDNA_ENV
from kmer_ord.utils.logging_utils import run_logged_command, section, warn, info

# Stage rules look like "[1/7] Run Barrnap". The other child lines are
# command invocations, path checks, and the artifact dump.
_STAGE_RULE = re.compile(r"\[\d+/\d+\]")
_RDNA_ALERT = ("terminat", "no rdna", "warn", "error", "traceback")


def _rdna_summary_line(line: str) -> bool:
    if _STAGE_RULE.search(line):
        return True
    lowered = line.lower()
    return any(token in lowered for token in _RDNA_ALERT)

# One chosen row comes from either SILVA or PR2. Class through genus already
# share names. PR2 division is the phylum equivalent, so it fills rdna_phylum.
# Supergroup and subdivision exist only in PR2.
FEATURE_COLUMNS = [
    "sequence_id",
    "rdna_domain",
    "rdna_supergroup",
    "rdna_phylum",
    "rdna_subdivision",
    "rdna_class",
    "rdna_order",
    "rdna_family",
    "rdna_genus",
    "rdna_taxonomy_source",
]


def _series_or_na(df: pd.DataFrame, column: str) -> pd.Series:
    """A source rank, or all-missing when that database did not emit the column."""
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def _blank(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.isna() | text.str.strip().eq("")


def unify_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Map SILVA and PR2 rows onto one rank list.

    Use SILVA phylum. Use PR2 division when that database was chosen, or when
    phylum is empty (older PR2-only tables have division and no taxonomy_source).
    """
    phylum = _series_or_na(df, "phylum")
    division = _series_or_na(df, "division")
    source = _series_or_na(df, "taxonomy_source")
    use_division = source.astype("string").str.strip().eq("pr2").fillna(False) | _blank(phylum)
    rdna_phylum = division.where(use_division, phylum)

    unified = pd.DataFrame({
        "sequence_id": df["sequence_id"],
        "rdna_domain": _series_or_na(df, "domain_clean"),
        "rdna_supergroup": _series_or_na(df, "supergroup"),
        "rdna_phylum": rdna_phylum,
        "rdna_subdivision": _series_or_na(df, "subdivision"),
        "rdna_class": _series_or_na(df, "class"),
        "rdna_order": _series_or_na(df, "order"),
        "rdna_family": _series_or_na(df, "family"),
        "rdna_genus": _series_or_na(df, "genus"),
        "rdna_taxonomy_source": source,
    })
    return unified.loc[:, FEATURE_COLUMNS]


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
    verbose=False,
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

    cmd = ["conda", "run", "--no-capture-output", "-n", RDNA_ENV, "rdna-miner", "long-read",
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

    # The summary filter hides command lines. Remember the last child line so
    # a failure that did not match an alert still names what went wrong.
    tail = {"last": None, "showed_alert": False}

    def keep(line: str) -> bool:
        stripped = line.strip()
        if stripped:
            tail["last"] = stripped
        showed = _rdna_summary_line(line)
        if showed and any(token in line.lower() for token in _RDNA_ALERT):
            tail["showed_alert"] = True
        return showed

    code = run_logged_command(cmd, verbose=verbose, keep=keep)
    if code != 0:
        message = "rDNA-miner failed; continuing without rDNA features."
        if not verbose and not tail["showed_alert"] and tail["last"]:
            message = f"{message} {tail['last']}"
        warn(message)
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

    df = unify_taxonomy(df)

    feature_file = Path(feature_file)
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(feature_file, sep="\t", index=False)

    info(f"rDNA-miner: {len(df)} reads assigned taxonomy -> {feature_file}")
    return feature_file
