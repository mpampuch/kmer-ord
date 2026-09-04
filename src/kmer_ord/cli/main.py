# src/kmer_ord/cli/main.py
import typer
import click
from pathlib import Path
from typing import List, Optional
import platform
import datetime

from kmer_ord.workflow.context import Context
from kmer_ord.workflow.runner import Runner
from kmer_ord.utils.logging_utils import section, info, warn, console
from kmer_ord.cli.setup import setup_app
from kmer_ord.utils.threading import set_global_threads

#app = typer.Typer(add_completion=False, rich_markup_mode=None)
app = typer.Typer(add_completion=False,
                  context_settings={"help_option_names": ["-h", "--help"]})

app.add_typer(setup_app)

_PCA_PRE_METHODS = click.Choice(["pca", "ipca"])
_PCA_PRE_METHOD_HELP = (
    "PCA algorithm for --pca-pre: 'pca' (exact, more RAM) or "
    "'ipca' (incremental/batched, low RAM)"
)
_PCA_PRE_BATCH_SIZE_HELP = (
    "IncrementalPCA batch size (used only with --pca-pre-method ipca). "
    "Default: max(2048, 5 * n_components)."
)


#----- header util

def print_header(start_time):
    from importlib.metadata import version
    v = version("kmer-ord")
    console.print()
    console.rule(style='none')
    console.print(f"  kmer-ord  v{v}", style="bold", highlight=False)
    console.print("  Pipeline for projecting kmer profiles in lower dimensional space",highlight=False)
    console.rule(style='none')
    console.print(
        f"  Started  {start_time.isoformat(timespec='seconds')}    "
        f"Python {platform.python_version()}",
        style="dim", highlight=False,
    )
    console.print()


def _print_artifacts(context):
    base = context.output_dir.parent

    def rel(p):
        try:
            return Path(p).relative_to(base)
        except ValueError:
            return Path(p)

    console.rule("output", style="none")

    for name, path in context.artifacts.items():

        # -------------------------
        # list artifacts
        # -------------------------
        if isinstance(path, list):

            # single item -> inline
            if len(path) == 1:
                console.print(
                    f"  {name:<28}  {rel(path[0])}",
                    style="none",
                    markup=False,
                    highlight=False,)

            # multiple items -> multiline
            else:
                console.print(
                    f"  {name}",
                    style="none",
                    markup=False,
                    highlight=False,)

                for p in path:
                    console.print(
                        f"      {rel(p)}",
                        style="none",
                        markup=False,
                        highlight=False,)

 
        # scalar artifacts
        # -------------------------
        else:
            console.print(
                f"  {name:<28}  {rel(path)}",
                style="none",
                markup=False,
                highlight=False,
            )

    console.rule(style="none")


def _print_footer(start_time):
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    total = duration.total_seconds()
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    runtime = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    console.print(
        f"  Finished  {end_time.isoformat(timespec='seconds')}    Runtime  {runtime}",
        style="none", highlight=False,
    )
    console.print()


def format_timedelta(td: "datetime.timedelta") -> str:
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"


# -----------------------------
# Pipeline
# -----------------------------
#@app.command("ordinate", rich_help_panel="Pipeline")
@app.command("project", rich_help_panel="Pipeline")
def run_pipeline(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta/fastq file (can be gzipped)"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    force: bool = typer.Option(False, "-f","--force", help="Force recomputation even if outputs exist"),
    kmer_length: int = typer.Option(6,"-k", "--kmer", help="K-mer length"),
    threads: int = typer.Option(4, "-t","--threads", help="Number of threads"),

    # --- DR options ---
    dr_methods: str = typer.Option("umap","--dr", help="Comma-separated DR methods (default: umap)"),
    scale: str = typer.Option("auto", "-s","--scale", help="Dataset scale presets for DR hyperparameters (auto, small, medium, large, default)"),
    normalisation: str = typer.Option("clr", "--norm", help="Normalization method (raw, relative, log, clr, zscore)"),
    dims: int = typer.Option(2, "-d","--dims", help="Embedding dimensions"),
    pca_pre: bool = typer.Option(False, "--pca-pre", help="Apply PCA before DR"),
    keep_pcs: int = typer.Option(None,"--keep-pcs", help="Number of principal components to retain"),
    keep_variance: float = typer.Option(None,"--keep-variance",help="Variance threshold for PCA (e.g. 0.9)"),
    pca_pre_method: str = typer.Option("pca", "--pca-pre-method", click_type=_PCA_PRE_METHODS, help=_PCA_PRE_METHOD_HELP),
    pca_pre_batch_size: Optional[int] = typer.Option(None, "--pca-pre-batch-size", help=_PCA_PRE_BATCH_SIZE_HELP),
    screen_params: bool = typer.Option(False, "--screen_params", help="Run parameter screening for supported DR methods"),
    screen_values1: List[str] = typer.Option([], "--screen_values1", help="Explicit axis-1 (count-like) values per method: 'method=v1,v2,...' or 'all=v1,v2,...'. Repeatable."),
    screen_values2: List[str] = typer.Option([], "--screen_values2", help="Explicit axis-2 values per method: 'method=v1,v2,...'. Repeatable."),
    screen_range1: List[str] = typer.Option([], "--screen_range1", help="Axis-1 range for auto-generated grids: 'method=min,max' or 'all=min,max'. Repeatable."),
    screen_range2: List[str] = typer.Option([], "--screen_range2", help="Axis-2 range for auto-generated grids: 'method=min,max'. Repeatable."),
    screen_grid: Optional[str] = typer.Option(None, "--screen_grid", help="Grid size for auto-generated screens: 'N' (1D) or 'NxM' (2D). Applies to every screened method."),

    # --- Tiara ---
    run_tiara: bool = typer.Option(False, "--tiara/--no-tiara", help="Run Tiara taxonomic classification and include results in the feature table (requires Tiara environment from kmer-ord setup)."),

    # --- rDNA-miner ---
    run_rdna: bool = typer.Option(False, "--rDNA/--no-rDNA", help="Run rDNA-miner (extract, assemble and classify rDNA reads) and include per-read taxonomy in the feature table (requires rDNA-miner environment from kmer-ord setup)."),
    rdna_platform: str = typer.Option("auto", "--rdna-platform", help="Sequencing platform for rDNA-miner (auto, ont, pacbio)"),
):
    """
    [+] Projection pipeline:
    Convert sequences (FASTQ/FASTA) into k-mer feature space,
    compute sequence-level metrics, and generate a low-dimensional
    (2D/3D) embedding that captures geometric relationships in k-mer space.
    Results are stored in the database for dowstream exploration and annotation.
    | fastq -> fasta -> sequence stats -> kmer-counting -> [tiara] -> [rDNA] -> DR -> database |
    """
    start_time = datetime.datetime.now()
    print_header(start_time)
    set_global_threads(threads)

    import numba
    info(f"Using {threads} threads / {numba.get_num_threads()} numba threads")

    info("loading packages")

    from kmer_ord.io.summary import calculate_stats
    from kmer_ord.workflow.operations import (
        FastqToFasta, FastaStats, KmerCount, KmerMetrics, Tiara, RDNAMiner, MatrixPreprocessing,
        DimensionalityReduction, FeatureMerge, SpatialiteDatabase)

    context = Context(input, output_dir, force=force, threads=threads, script_name="project")

    method_list = [m.strip().lower() for m in dr_methods.split(",")]
    norm_list = [n.strip().lower() for n in normalisation.split(",")]

    operations = [
        FastqToFasta(),
        FastaStats(),
        KmerCount(kmer_length=kmer_length, threads=threads),
        KmerMetrics(chunksize=25000, cpus=threads),
    ]

    if run_tiara:
        info("Tiara classification enabled.")
        operations.append(Tiara(threads=threads))
    else:
        info("Tiara classification skipped (pass --tiara to enable).")

    if run_rdna:
        info("rDNA-miner classification enabled.")
        operations.append(RDNAMiner(threads=threads, platform=rdna_platform))
    else:
        info("rDNA-miner classification skipped (pass --rDNA to enable).")

    operations += [
        MatrixPreprocessing(
            normalisations=norm_list,
            pca_dim_red=pca_pre,
            keep_pcs=keep_pcs,
            keep_variance=keep_variance,
            pca_method=pca_pre_method,
            pca_batch_size=pca_pre_batch_size,
            scale=scale),
        DimensionalityReduction(
            methods=method_list,
            dims=dims,
            scale=scale,
            screen_params=screen_params,
            screen_values1=screen_values1,
            screen_values2=screen_values2,
            screen_range1=screen_range1,
            screen_range2=screen_range2,
            screen_grid=screen_grid,
            threads=threads,),
        FeatureMerge(),
        SpatialiteDatabase()]

    runner = Runner(operations)
    runner.run(context)

    _print_artifacts(context)
    _print_footer(start_time)


@app.command("cluster", rich_help_panel="Pipeline")
def discover_pipeline(
    input: Path = typer.Option(..., "-i", "--input", help="Input fasta/fastq file"),
    output_dir: Path = typer.Option(..., "-o", "--output", help="Output directory"),
    kmer_length: int = typer.Option(6, "-k", "--kmer"),
    dims: int = typer.Option(15, "-d", "--dims", help="High-dimensional embedding size"),
    dr_method: str = typer.Option("umap", "--dr"),
    scale: str = typer.Option("auto", "-s","--scale", help="Dataset scale presets for DR hyperparameters (auto, small, medium, large, default)"),
    normalisation: str = typer.Option("clr", "--norm"),
    pca_pre: bool = typer.Option(False, "--pca-pre", help="Apply PCA before DR"),
    keep_pcs: int = typer.Option(None,"--keep-pcs", help="Number of principal components to retain"),
    keep_variance: float = typer.Option(None,"--keep-variance",help="Variance threshold for PCA (e.g. 0.9)"),
    pca_pre_method: str = typer.Option("pca", "--pca-pre-method", click_type=_PCA_PRE_METHODS, help=_PCA_PRE_METHOD_HELP),
    pca_pre_batch_size: Optional[int] = typer.Option(None, "--pca-pre-batch-size", help=_PCA_PRE_BATCH_SIZE_HELP),
    screen_params: bool = typer.Option(False, "--screen_params", help="Run parameter screening for supported DR methods"),
    screen_values1: List[str] = typer.Option([], "--screen_values1", help="Explicit axis-1 (count-like) values per method: 'method=v1,v2,...' or 'all=v1,v2,...'. Repeatable."),
    screen_values2: List[str] = typer.Option([], "--screen_values2", help="Explicit axis-2 values per method: 'method=v1,v2,...'. Repeatable."),
    screen_range1: List[str] = typer.Option([], "--screen_range1", help="Axis-1 range for auto-generated grids: 'method=min,max' or 'all=min,max'. Repeatable."),
    screen_range2: List[str] = typer.Option([], "--screen_range2", help="Axis-2 range for auto-generated grids: 'method=min,max'. Repeatable."),
    screen_grid: Optional[str] = typer.Option(None, "--screen_grid", help="Grid size for auto-generated screens: 'N' (1D) or 'NxM' (2D). Applies to every screened method."),

    cluster_methods: str = typer.Option("hdbscan", "--cluster", help="Comma-separated clustering methods (leiden,hdbscan,dbscan)"),
    leiden_sweep: bool = typer.Option(False, "--leiden-sweep", help="Run Leiden resolution sweep"),
    hdbscan_sweep: bool = typer.Option(False, "--hdbscan-sweep", help="Run HDBSCAN min_cluster_size sweep"),
    dbscan_sweep: bool = typer.Option(False, "--dbscan-sweep", help="Run DBSCAN eps sweep"),
    threads: int = typer.Option(4, "-t", "--threads"),
    force: bool = typer.Option(False, "-f", "--force"),
    db_path: Path = typer.Option(None, "--db", help="Optional path to existing SQLite/SpatiaLite DB"),):
    """
    [+] Cluster inference pipeline :
    Construct a high-dimensional embedding of k-mer feature space
    and perform unsupervised clustering to infer intrinsic structure 
    among sequences. Embeddings and cluster assignments are 
    integrated into the database for downstream analysis.
    | kmer-profiles -> High-D embedding -> clustering -> database |
    """
    start_time = datetime.datetime.now()
    print_header(start_time)

    section("Starting kmer-ord clustering pipeline...")
    set_global_threads(threads)
    info(f"Using {threads} threads")
    import numba
    info(f"Using {numba.get_num_threads()} Numba threads")
    

    info("loading packages")
    from kmer_ord.workflow.operations import (
        FastqToFasta,
        FastaStats,
        KmerCount,
        KmerMetrics,
        MatrixPreprocessing,
        DimensionalityReduction,
        Clustering,
        AddClusteringToDB)

    context = Context(input, output_dir, force=force, threads=threads, script_name="cluster")

    cluster_list = [c.strip().lower() for c in cluster_methods.split(",")]
    norm_list = [n.strip().lower() for n in normalisation.split(",")]
    method_list = [m.strip().lower() for m in dr_method.split(",")]

    # -----------------------------
    # Core operations
    # -----------------------------
    operations = [
        FastqToFasta(),
        FastaStats(),
        KmerCount(kmer_length=kmer_length, threads=threads),
        KmerMetrics(),
        MatrixPreprocessing(
            normalisations=norm_list,
            pca_dim_red=pca_pre,
            keep_pcs=keep_pcs,
            keep_variance=keep_variance,
            pca_method=pca_pre_method,
            pca_batch_size=pca_pre_batch_size,
            scale=scale),
        DimensionalityReduction(
            methods=method_list,
            dims=dims,
            scale=scale,
            screen_params=screen_params,
            screen_values1=screen_values1,
            screen_values2=screen_values2,
            screen_range1=screen_range1,
            screen_range2=screen_range2,
            screen_grid=screen_grid,
            threads=threads,)]

    # -----------------------------
    # Clustering operations
    # -----------------------------
    for method in cluster_list:
        if method == "leiden":
            operations.append(Clustering(method="leiden",sweep=leiden_sweep))
        elif method == "hdbscan":
            operations.append(Clustering(method="hdbscan",sweep=hdbscan_sweep))
        elif method == "dbscan":
            operations.append(Clustering(method="dbscan",sweep=dbscan_sweep))
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    runner = Runner(operations)
    runner.run(context)

    # Database integration
    if db_path is None:
        db_path = output_dir / "discovery.sqlite"

    add_db_op = AddClusteringToDB(db_path=db_path, force=force)
    add_db_op.run(context)

    info(f"Database saved at: {db_path}")

    _print_artifacts(context)
    _print_footer(start_time)


@app.command("visualise", rich_help_panel="Analysis")
def visualise_db(
    db_path: Path = typer.Option(..., "-d", "--db", help="Path to the SQLite/SpatiaLite database"),
    max_categories: int = typer.Option(10, "--max-categories", help="Max number of categories for categorical feature plots"),
    embeddings: bool = typer.Option(True, "--embeddings/--no-embeddings", help="Generate embedding plots"),
    embedding_mode: str = typer.Option("all", "--embedding-mode", help="Embedding plot mode: density, categorical, continuous, all"),
    features: bool = typer.Option(True, "--features/--no-features", help="Generate feature plots")):
    """
    Visualise database tables.
    - Feature distributions and categorical comparisons
    - Embedding visualisations (UMAP, t-SNE, etc.)
    """
    start_time = datetime.datetime.now()
    print_header(start_time)

    section("Starting kmer-ord visualisation...")
    #set_global_threads(threads)
    #info(f"Using {threads} threads")
    info("loading packages")

    from kmer_ord.workflow.operations import PlotFeatures, PlotEmbeddings
    from kmer_ord.workflow.context import DBContext

    ctx = DBContext(db_path)

    if features:
        section("Generating feature plots")
        plot_features = PlotFeatures(max_categories=max_categories)
        plot_features.run(ctx)

    if embeddings:
        section("Generating embedding plots")
        plot_embeddings = PlotEmbeddings(mode=embedding_mode)
        plot_embeddings.run(ctx)

    info(f"All plots saved to: {ctx.output_dir / 'plots'}")
    _print_footer(start_time)


@app.command("inject", rich_help_panel="Analysis")
def inject_features_cmd(
    db_path: Path = typer.Option(..., "-d", "--db", help="Path to the SQLite/SpatiaLite database"),
    input_file: Path = typer.Option(..., "-i", "--input", help="Tab-separated file (.tsv/.txt) with sequence_id column and new feature columns to add"),
):
    """
    Inject new feature columns from a TSV file into the database features table.

    \b
    Rules:
    - First column must be named 'sequence_id' (matches the pipeline convention)
    - All sequence_ids in the database are preserved (left join)
    - Sequences in the input with no DB match are ignored
    - Sequences in the DB with no input row get NULL for the new columns
    - Columns whose name already exists in the features table are skipped
    - Duplicate sequence_ids in the input file are rejected
    """
    import pandas as pd
    from kmer_ord.io.database import inject_features

    section("Starting feature injection...")

    # ------------------------------------------------------------------
    # Input file checks
    # ------------------------------------------------------------------
    if not input_file.exists():
        warn(f"Input file not found: {input_file}")
        raise typer.Exit(1)

    try:
        input_df = pd.read_csv(input_file, sep="\t")
    except Exception as e:
        warn(f"Could not read input file as tab-separated: {e}")
        raise typer.Exit(1)

    if input_df.empty:
        warn("Input file is empty.")
        raise typer.Exit(1)

    if "sequence_id" not in input_df.columns:
        warn(
            f"Input file must contain a 'sequence_id' column. "
            f"Columns found: {list(input_df.columns)}"
        )
        raise typer.Exit(1)

    if input_df.columns[0] != "sequence_id":
        warn(
            f"Expected 'sequence_id' as the first column but found '{input_df.columns[0]}'. "
            "Proceeding — the join will still work."
        )

    dupes = input_df["sequence_id"][input_df["sequence_id"].duplicated(keep=False)].unique()
    if len(dupes):
        warn(
            f"Duplicate sequence_id values detected in the input file ({len(dupes)} affected IDs). "
            "Resolve duplicates before injecting."
        )
        warn(f"  Examples: {list(dupes[:5])}")
        raise typer.Exit(1)

    data_cols = [c for c in input_df.columns if c != "sequence_id"]
    if not data_cols:
        warn("Input file has no columns beyond 'sequence_id'. Nothing to inject.")
        raise typer.Exit(1)

    info(f"Input: {len(input_df)} rows, {len(data_cols)} data column(s): {data_cols}")

    # ------------------------------------------------------------------
    # Database checks
    # ------------------------------------------------------------------
    if not db_path.exists():
        warn(f"Database file not found: {db_path}")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Inject
    # ------------------------------------------------------------------
    try:
        report = inject_features(db_path, input_df)
    except (RuntimeError, ValueError) as e:
        warn(str(e))
        raise typer.Exit(1)
    except Exception as e:
        warn(f"Unexpected error during injection: {e}")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    info(
        f"Sequence ID coverage: {report['matched']} / {report['total_db']} "
        "database sequences matched."
    )

    if report["unmatched_input"]:
        warn(
            f"{report['unmatched_input']} sequence_id(s) in the input file had no "
            "match in the database and were ignored."
        )

    if report["unmatched_db"]:
        info(
            f"{report['unmatched_db']} database sequence(s) had no entry in the input file "
            "— new columns will be NULL for these."
        )

    if report["skipped_cols"]:
        warn(
            f"Skipped {len(report['skipped_cols'])} column(s) already present "
            "in the features table (existing values kept):"
        )
        for col in report["skipped_cols"]:
            warn(f"    - {col}")

    if report["injected_cols"]:
        section(
            f"Injected {len(report['injected_cols'])} new column(s) into the features table:"
        )
        for col in report["injected_cols"]:
            info(f"  {col}")
    else:
        warn("No new columns were injected (all columns already existed in the features table).")


@app.command("bin", rich_help_panel="Analysis")
def run_binner(
    db_path: Path = typer.Option(..., "-d", "--db", help="Path to SQLite DB"),
    output_dir: Path = typer.Option("bins", "-o", "--output", help="Output dir for bins"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8050, "--port"),
):
    """
    Launch interactive Dash app for binning sequences.
    """
    from kmer_ord.dash.b2w import run_dash_app

    run_dash_app(
        db_path=str(db_path),
        output_dir=str(output_dir),
        host=host,
        port=port,
    )


# -----------------------------
# fastq to fasta
@app.command("fastq-to-fasta", rich_help_panel="Modules")
def fastq_to_fasta_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fastq file (can be gzipped)"),
    output: Path = typer.Option(..., "-o","--output", help="Output fasta file"),
    threads: int = typer.Option(1, "-t", "--threads", help="Threads for seqkit"),
    biopython: bool = typer.Option(False, "--biopython", help="Use BioPython instead of seqkit (legacy)"),
    force: bool = typer.Option(False, "-f","--force", help="Overwrite output if it exists")):
    """
    Convert fastq (or fastq.gz) to fasta. Uses seqkit by default; --biopython for legacy fallback.
    """
    if output.exists() and not force:
        info(f"Skipping conversion, FASTA already exists: {output}")
        return
    if biopython:
        from kmer_ord.io.sequence import fastq_to_fasta
        fastq_to_fasta(input, output)
    else:
        from kmer_ord.io.sequence import fastq_to_fasta_seqkit
        fastq_to_fasta_seqkit(input, output, threads=threads)
    info(f"fastq -> fasta conversion done: {output}")


# -----------------------------
# FASTA stats
@app.command("fasta-stats", rich_help_panel="Modules")
def fasta_stats_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    force: bool = typer.Option(False, "-f","--force", help="Recalculate stats even if outputs exist")):
    """
    Calculate per-sequence and overall statistics from a fasta file.
    """
    from kmer_ord.io.summary import calculate_stats

    context = Context(input, output_dir, force=force)
    overall_file, tsv_file = calculate_stats(context)
    info(f"Stats calculated. Sequence-level tsv: {tsv_file}, Overall: {overall_file}")

# -----------------------------
# K-mer counting
@app.command("kmer-count", rich_help_panel="Modules")
def kmer_count_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    kmer_length: int = typer.Option(6, "-k", "--kmer", help="K-mer length"),
    threads: int = typer.Option(1, "-t", "--threads", help="Number of threads for counting"),
    force: bool = typer.Option(False, "-f","--force", help="Recalculate even if output exists")):
    """
    Count k-mers for a fasta file and save tsv matrix.
    """
    context = Context(input, output_dir, force=force, threads=threads)
    from kmer_ord.workflow.operations import KmerCount

    operation = KmerCount(kmer_length=kmer_length, threads=threads)
    operation.run(context)

    info(f"K-mer counting complete. Matrix saved at: {context.get('kmer_matrix')}")

if __name__ == "__main__":
    app()


# -----------------------------
# kmer-metrics
@app.command("kmer-metrics", rich_help_panel="Modules")
def kmer_metrics_cmd(
    input: Path = typer.Option(..., "-i", "--input",help="Input k-mer matrix TSV"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    chunksize: int = typer.Option(25000, "--chunksize", help="Rows per chunk"),
    cpus: int = typer.Option(1, "--cpus", help="Number of worker processes"),
    force: bool = typer.Option(False, "-f","--force", help="Recompute even if output exists"),):
    """
    Compute per-sequence k-mer metrics (Shannon diversity, unique k-mers, etc.).
    """
    from kmer_ord.workflow.operations import KmerMetrics

    context = Context(input, output_dir, force=force)
    operation = KmerMetrics(chunksize=chunksize, cpus=cpus)
    operation.run(context)
    info(f"K-mer metrics saved at: {context.get('kmer_metrics')}")


# -----------------------------
# DR
@app.command("dr", rich_help_panel="Modules")
def dr_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input k-mer matrix tsv"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    methods: str = typer.Option(..., "-m","--methods", help="Comma-separated DR methods"),
    scale: str = typer.Option("auto", "-s","--scale", help="Dataset scale presets for DR hyperparameters (auto, small, medium, large, default)"),
    normalisation: str = typer.Option("clr", "--norm", help="Normalization method"),
    dims: int = typer.Option(2,"-d", "--dims", help="Embedding dimensions"),
    force: bool = typer.Option(False, "-f","--force", help="Recompute even if output exists"),
    pca_pre: bool = typer.Option(False, "--pca-pre", help="Apply PCA before DR"),
    keep_pcs: int = typer.Option(None, "--keep-pcs"),
    keep_variance: float = typer.Option(None, "--keep-variance"),
    pca_pre_method: str = typer.Option("pca", "--pca-pre-method", click_type=_PCA_PRE_METHODS, help=_PCA_PRE_METHOD_HELP),
    pca_pre_batch_size: Optional[int] = typer.Option(None, "--pca-pre-batch-size", help=_PCA_PRE_BATCH_SIZE_HELP),
    screen_params: bool = typer.Option(False, "--screen_params", help="Run parameter screening for supported DR methods"),
    screen_values1: List[str] = typer.Option([], "--screen_values1", help="Explicit axis-1 (count-like) values per method: 'method=v1,v2,...' or 'all=v1,v2,...'. Repeatable."),
    screen_values2: List[str] = typer.Option([], "--screen_values2", help="Explicit axis-2 values per method: 'method=v1,v2,...'. Repeatable."),
    screen_range1: List[str] = typer.Option([], "--screen_range1", help="Axis-1 range for auto-generated grids: 'method=min,max' or 'all=min,max'. Repeatable."),
    screen_range2: List[str] = typer.Option([], "--screen_range2", help="Axis-2 range for auto-generated grids: 'method=min,max'. Repeatable."),
    screen_grid: Optional[str] = typer.Option(None, "--screen_grid", help="Grid size for auto-generated screens: 'N' (1D) or 'NxM' (2D). Applies to every screened method."),
    threads: int = typer.Option(4, "-t","--threads", help="Number of threads"),
):
    """
    Run dimensionality reduction on an existing k-mer matrix.
    """
    set_global_threads(threads)
    info(f"Using {threads} threads")

    from kmer_ord.workflow.context import MatrixContext
    from kmer_ord.workflow.operations import MatrixPreprocessing, DimensionalityReduction

    context = MatrixContext(input, output_dir, force=force, script_name="dr")

    method_list = [m.strip().lower() for m in methods.split(",")]
    norm_list = [n.strip().lower() for n in normalisation.split(",")]

    operations = [
        MatrixPreprocessing(
            normalisations=norm_list,
            pca_dim_red=pca_pre,
            keep_pcs=keep_pcs,
            keep_variance=keep_variance,
            pca_method=pca_pre_method,
            pca_batch_size=pca_pre_batch_size,
            scale=scale,
        ),
        DimensionalityReduction(
            methods=method_list,
            dims=dims,
            scale=scale,
            screen_params=screen_params,
            screen_values1=screen_values1,
            screen_values2=screen_values2,
            screen_range1=screen_range1,
            screen_range2=screen_range2,
            screen_grid=screen_grid,
            threads=threads,
        ),
    ]

    runner = Runner(operations)
    runner.run(context)

    info(f"DR embeddings saved at: {context.get('dr_embeddings')}")


@app.command("clustering", rich_help_panel="Modules")
def cluster_pipeline(
    input: Path = typer.Option(..., "-i", "--input", help="Input directory containing artifacts"),
    output_dir: Path = typer.Option(..., "-o", "--output"),
    method: str = typer.Option("hdbscan", "--method"),
    force: bool = typer.Option(False, "-f", "--force")):
    """
    Cluster sequences using existing embedding.
    """

    from kmer_ord.workflow.operations import (Clustering, SpatialiteDatabase)

    context = Context(input, output_dir, force=force)

    operations = [Clustering(method=method),
                  SpatialiteDatabase()]

    runner = Runner(operations)
    runner.run(context)

    info("Clustering complete.")
 

@app.command("run-tiara", rich_help_panel="Modules")
def run_tiara_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    threads: int = typer.Option(1, "-t", help="Number of threads"),
    force: bool = typer.Option(False, "-f", "--force", help="Recompute even if output exists"),):
    """
    Run Tiara classification on a fasta file.
    """
    from kmer_ord.workflow.operations import Tiara

    context = Context(input, output_dir, force=force, threads=threads)

    operation = Tiara(threads=threads)
    operation.run(context)

    info(f"Tiara output saved at: {context.get('tiara')}")


@app.command("run-rdna", rich_help_panel="Modules")
def run_rdna_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    threads: int = typer.Option(1, "-t", help="Number of threads"),
    platform: str = typer.Option("auto", "-p", "--platform", help="Sequencing platform (auto, ont, pacbio)"),
    force: bool = typer.Option(False, "-f", "--force", help="Recompute even if output exists"),):
    """
    Run rDNA-miner (extract, assemble and classify rDNA reads) on a fasta file.
    """
    from kmer_ord.workflow.operations import RDNAMiner

    context = Context(input, output_dir, force=force, threads=threads)

    operation = RDNAMiner(threads=threads, platform=platform)
    operation.run(context)

    if context.exists("rdna"):
        info(f"rDNA-miner output saved at: {context.get('rdna')}")
    else:
        info("rDNA-miner produced no output.")


@app.command("build-db", rich_help_panel="Modules")
def build_database(
    input: Path = typer.Option(..., "-i", "--input", help="Input directory containing artifacts"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory for database"),
    force: bool = typer.Option(False, "-f", "--force", help="Recompute even if output exists"),):
    """
    Build Spatialite database from available artifacts.
    """
    from kmer_ord.workflow.operations import (FeatureMerge, SpatialiteDatabase)

    context = Context(input, output_dir, force=force)

    operation = SpatialiteDatabase()
    operation.run(context)

    info(f"Database created at: {context.get('database')}")