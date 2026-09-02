# src/kmer_ord/workflow/operations.py
from pathlib import Path
from kmer_ord.workflow import context
import typer
#import pandas as pd
#import numpy as np
#import importlib.resources as pkg_resources

from kmer_ord.io.summary import calculate_stats
from kmer_ord.io.kmer_counter import run_kmer_counter
from kmer_ord.io.run_tiara import run_tiara
from kmer_ord.io.run_rdna import run_rdna_miner
from .operation import Operation

from kmer_ord.system.env_manager import TOOLS_ENV, run_in_env

class FastqToFasta(Operation):
    name = "fastq_to_fasta"
    produces = ["fasta"]

    def run(self, context):
        with context.benchmark_timer(label=self.name, input_file=context.input_file):
            # No need to convert again — context has already created canonical FASTA.
            # Just register it explicitly.
            fasta = context.fasta
            context.register("fasta", fasta)


class FastaStats(Operation):
    name = "fasta_stats"
    requires = ["fasta"]
    produces = ["summary_overall", "summary_per_sequence"]

    def run(self, context):
        overall_path = context.artifact_path("overall_stats", subdir="summary", suffix=".txt")
        per_seq_path = context.artifact_path("stats_per_sequence", subdir="summary", suffix=".tsv")

        with context.benchmark_timer(label=self.name, input_file=context.fasta):
            if overall_path.exists() and per_seq_path.exists() and not context.force:
                context.logger.info("Skipping FastaStats, summary output files already exist.")
                typer.echo("Skipping FastaStats, summary output files already exist.")
            else:
                df, overall_file, tsv_file = calculate_stats(context)

        # Register artifacts even if skipping
        context.register("summary_overall", overall_path)
        context.register("summary_per_sequence", per_seq_path)


class KmerCount(Operation):
    name = "kmer_count"
    requires = ["fasta"]
    produces = ["kmer_matrix"]

    def __init__(self, kmer_length, threads=1):
        self.kmer_length = kmer_length
        self.threads = threads

    def run(self, context):
        output_path = context.artifact_path(f"{self.kmer_length}mer_matrix",
                                            subdir="kmer", suffix=".tsv")

        with context.benchmark_timer(label=f"{self.name}_{self.kmer_length}mer",
                            input_file=context.get("fasta"),
                            input_args=f"kmer_length={self.kmer_length}, threads={self.threads}"):
            if output_path.exists() and not context.force:
                typer.echo(f"Skipping kmer-counter, matrix already exists: {output_path}")
                context.logger.info(f"Skipping KmerCount, matrix already exists: {output_path}")
            else:
                run_kmer_counter(
                    input_file=context.get("fasta"),
                    output_tsv=output_path,
                    kmer_length=self.kmer_length,
                    num_threads=self.threads,
                    log_dir=str(context.benchmark_dir),
                    script_name=context.script_name)

        context.register("kmer_matrix", output_path)

# -----------------------------
# DR

from kmer_ord.dr.loader import load_matrix
from kmer_ord.dr.preprocess import preprocess_data, reduce_dimensions_with_pca
from kmer_ord.dr.methods import run_dr_methods

## do the numpy np.save etc inside the function preprocess data?
class MatrixPreprocessing(Operation):
    name = "matrix_preprocessing"
    requires = ["kmer_matrix"]
    produces = ["preprocessed_matrices", "preprocessed_sequence_ids"]

    def __init__(
        self,
        normalisations=("clr",),
        pca_dim_red=False,
        keep_pcs=None,
        keep_variance=None,
        pca_method="pca",
        pca_batch_size=None,
        scale="auto",
        max_memory_gb=None,
    ):
        self.normalisations = normalisations
        self.pca_dim_red = pca_dim_red
        self.keep_pcs = keep_pcs
        self.keep_variance = keep_variance
        self.pca_method = pca_method
        self.pca_batch_size = pca_batch_size
        self.scale = scale
        self.max_memory_gb = max_memory_gb

    def run(self, context):
        matrix_path = context.get("kmer_matrix")

        with context.benchmark_timer(
            label=self.name,
            input_file=matrix_path,
            input_args=(
                f"normalisations={','.join(self.normalisations)}, "
                f"pca_dim_red={self.pca_dim_red}, keep_pcs={self.keep_pcs}, "
                f"keep_variance={self.keep_variance}, pca_method={self.pca_method}, "
                f"pca_batch_size={self.pca_batch_size}"
            ),
        ) as bt:
            self._run(context, matrix_path, bt)

    def _run(self, context, matrix_path, bt):
        import numpy as np
        # Load full matrix including sequence_id column
        matrix = load_matrix(matrix_path)
        bt.record_input_shape(*matrix.shape)

        #if "sequence_id" not in matrix.columns:
        #    raise RuntimeError("Input matrix must contain 'sequence_id' column.")

        sequence_ids = matrix.index.to_numpy()

        output_dir = context.output_dir / "matrices"
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = []
        seqid_outputs = []

        normalisations = (
            self.normalisations
            if "all" not in self.normalisations
            else ALL_METHODS
        )

        for norm in normalisations:
            # -----------------------------
            # Save numeric matrix
            # -----------------------------
            out_path = output_dir / f"{matrix_path.stem}_{norm}.npy"
            if out_path.exists() and not context.force:
                outputs.append(out_path)
            else:
                with context.benchmark_timer(
                    label=f"preprocess_{norm}",
                    input_file=matrix_path,
                ):
                    X = preprocess_data(matrix, norm)

                if self.pca_dim_red:
                    with context.benchmark_timer(
                        label=f"pca_pre_{norm}",
                        input_file=matrix_path,
                        input_args=(
                            f"method={self.pca_method}, keep_pcs={self.keep_pcs}, "
                            f"keep_variance={self.keep_variance}, "
                            f"batch_size={self.pca_batch_size}"
                        ),
                    ):
                        X = reduce_dimensions_with_pca(
                            X,
                            keep_pcs=self.keep_pcs,
                            keep_variance=self.keep_variance,
                            method=self.pca_method,
                            batch_size=self.pca_batch_size,
                        )

                np.save(out_path, X)
                outputs.append(out_path)

            # -----------------------------
            # Save sequence IDs separately
            # -----------------------------
            seqid_path = output_dir / f"{matrix_path.stem}_{norm}_sequence_ids.npy"
            if seqid_path.exists() and not context.force:
                seqid_outputs.append(seqid_path)
            else:
                np.save(seqid_path, sequence_ids.astype("U"))
                seqid_outputs.append(seqid_path)

        # Register outputs
        context.register("preprocessed_matrices", outputs)
        context.register("preprocessed_sequence_ids", seqid_outputs)

#GRAPH_METHODS = {"umap", "trimap", "pacmap", "localmap"}  # DR methods that produce a graph

class DimensionalityReduction(Operation):
    name = "dimensionality_reduction"
    requires = ["preprocessed_matrices", "preprocessed_sequence_ids"]
    produces = ["dr_embeddings", "dr_graph"]

    def __init__(
        self,
        methods,
        dims=2,
        scale="auto",
        seed=42,
        screen_params=False,
        screen_values1=None,
        screen_values2=None,
        screen_range1=None,
        screen_range2=None,
        screen_grid=None,
        max_memory_gb=None,
        threads=1,
    ):
        self.methods = methods
        self.dims = dims
        self.scale = scale
        self.seed = seed
        self.screen_params = screen_params
        self.screen_values1 = screen_values1
        self.screen_values2 = screen_values2
        self.screen_range1 = screen_range1
        self.screen_range2 = screen_range2
        self.screen_grid = screen_grid
        self.max_memory_gb = max_memory_gb
        self.threads = threads

    def run(self, context):
        import numpy as np
        matrix_paths = context.get("preprocessed_matrices")
        seqid_paths = context.get("preprocessed_sequence_ids")
        dr_dir = context.output_dir / "dr"
        dr_dir.mkdir(parents=True, exist_ok=True)

        merged_outputs = []
        all_graph_paths = []

        for i, matrix_path in enumerate(matrix_paths):

            norm_label = matrix_path.stem.split("_")[-1]
            input_name = "_".join(matrix_path.stem.split("_")[:-1])

            context.logger.info(f"Running DR for: {matrix_path.name}")

            # Expected merged output location
            merged_output = (
                dr_dir
                / norm_label
                / f"{input_name}_{norm_label}_{self.dims}D_merged_embeddings.tsv"
            )

            # Caching behaviour (checked before loading the matrix so a
            # cached run doesn't pay the load cost or log a benchmark row)
            if merged_output.exists() and not context.force:
                merged_outputs.append(merged_output)

                # collect any existing graphs under this normalisation
                norm_dir = dr_dir / norm_label
                if norm_dir.exists():
                    for graph_file in norm_dir.rglob("*_graph.npz"):
                        all_graph_paths.append(graph_file)

                continue

            with context.benchmark_timer(
                label=f"{self.name}_{norm_label}",
                input_file=matrix_path,
                input_args=(
                    f"methods={','.join(self.methods)}, dims={self.dims}, "
                    f"scale={self.scale}, screen_params={self.screen_params}, "
                    f"threads={self.threads}"
                ),
            ) as bt:
                with context.benchmark_timer(
                    label=f"dr_load_{norm_label}",
                    input_file=matrix_path,
                ) as load_bt:
                    X = np.load(matrix_path)
                    load_bt.record_input_shape(*X.shape)
                    sequence_ids = np.load(seqid_paths[i])
                bt.record_input_shape(*X.shape)

                # Memory check: per-method estimates including neighbor
                # graphs / pair tables (the old X.nbytes*4 heuristic ignored
                # them and let e.g. n_neighbors=200 UMAP fits OOM unchecked)
                if self.max_memory_gb:
                    from kmer_ord.dr.methods import (
                        ALL_METHODS,
                        _resolve_scale,
                        estimate_peak_memory_gb,
                    )
                    n_seq, n_feat = X.shape
                    resolved_scale = _resolve_scale(self.scale, n_seq)
                    methods = (
                        self.methods if "all" not in self.methods else ALL_METHODS
                    )
                    estimates = {
                        m: estimate_peak_memory_gb(n_seq, n_feat, m, resolved_scale)
                        for m in methods
                    }
                    worst_method = max(estimates, key=estimates.get)
                    est_peak = estimates[worst_method]
                    if est_peak > self.max_memory_gb:
                        raise MemoryError(
                            f"Estimated peak {est_peak:.2f} GB for method "
                            f"'{worst_method}' (scale={resolved_scale}) exceeds "
                            f"limit {self.max_memory_gb:.2f} GB"
                        )

                # Run DR with proper sequence IDs
                merged_file, graph_paths = run_dr_methods(
                    X=X,
                    methods=self.methods,
                    dims=self.dims,
                    seed=self.seed,
                    scale=self.scale,
                    screen_params=self.screen_params,
                    screen_values1=self.screen_values1,
                    screen_values2=self.screen_values2,
                    screen_range1=self.screen_range1,
                    screen_range2=self.screen_range2,
                    screen_grid=self.screen_grid,
                    output_dir=dr_dir,
                    normalisation=norm_label,
                    input_name=input_name,
                    sequence_ids=sequence_ids,
                    n_jobs=self.threads,
                    log_dir=str(context.benchmark_dir),
                    script_name=context.script_name,
                )

            merged_outputs.append(merged_file)

            if graph_paths:
                all_graph_paths.extend(graph_paths)

        # Register outputs
        context.register("dr_embeddings", merged_outputs)

        if all_graph_paths:
            context.register("dr_graph", all_graph_paths)


class KmerMetrics(Operation):
    name = "kmer_metrics"
    requires = ["kmer_matrix"]
    produces = ["kmer_metrics"]

    def __init__(self, chunksize=1000, cpus=1):
        self.chunksize = chunksize
        self.cpus = cpus

    def run(self, context):
        from kmer_ord.io.kmer_stats import process_kmer_file
        matrix_path = context.get("kmer_matrix")

        output_file = context.artifact_path(name="kmer_metrics", subdir="kmer", suffix=".tsv")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with context.benchmark_timer(label=self.name, input_file=matrix_path):
            if output_file.exists() and not context.force:
                typer.echo(f"Skipping KmerMetrics, output exists: {output_file}")
                context.logger.info(f"Skipping KmerMetrics, output exists: {output_file}")
            else:
                # metrics are streamed to output_file; nothing is returned
                # into memory here by design (see kmer_stats.process_kmer_file)
                process_kmer_file(
                    input_file=matrix_path,
                    output_file=output_file,
                    chunksize=self.chunksize,
                    cpus=self.cpus)

        context.register("kmer_metrics", output_file)


class Tiara(Operation):
    name = "tiara"
    requires = ["fasta"]
    produces = ["tiara"]

    def __init__(self, threads=1):
        self.threads = threads

    def run(self, context):
        input_fasta = context.get("fasta")

        output_file = context.artifact_path(name="tiara",
                                            subdir="tiara",
                                            suffix=".tsv")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with context.benchmark_timer(
            label=self.name,
            input_file=input_fasta,
            input_args=f"threads={self.threads}"):
            if output_file.exists() and not context.force:
                typer.echo(f"Skipping Tiara, output exists: {output_file}")
                context.logger.info(f"Skipping Tiara, output exists: {output_file}")
            else:
                run_tiara(input_file=input_fasta,
                          output_file=output_file,
                          threads=self.threads)

        context.register("tiara", output_file)


class RDNAMiner(Operation):
    name = "rdna"
    requires = ["fasta"]
    produces = ["rdna"]

    def __init__(self, threads=1, platform="auto"):
        self.threads = threads
        self.platform = platform

    def run(self, context):
        input_fasta = context.get("fasta")

        raw_dir = context.output_dir / "rdna" / "raw"
        feature_file = context.artifact_path(name="rdna",
                                             subdir="rdna",
                                             suffix=".tsv")
        feature_file.parent.mkdir(parents=True, exist_ok=True)

        with context.benchmark_timer(
            label=self.name,
            input_file=input_fasta,
            input_args=f"threads={self.threads} platform={self.platform}"):
            if feature_file.exists() and not context.force:
                typer.echo(f"Skipping rDNA-miner, output exists: {feature_file}")
                context.logger.info(f"Skipping rDNA-miner, output exists: {feature_file}")
            else:
                run_rdna_miner(input_file=input_fasta,
                                output_dir=raw_dir,
                                feature_file=feature_file,
                                threads=self.threads,
                                platform=self.platform)

        if feature_file.exists():
            context.register("rdna", feature_file)


class FeatureMerge(Operation):
    name = "feature-merge"
    requires = ["kmer_metrics", "summary_per_sequence"]
    produces = ["merged_features"]

    def run(self, context):
        import pandas as pd

        with context.benchmark_timer(label=self.name):
            def normalize_id_column(df):
                possible_id_cols = ["sequence_id", "header", "seq_id", "contig", "id"]
                for col in possible_id_cols:
                    if col in df.columns:
                        if col != "sequence_id":
                            df = df.rename(columns={col: "sequence_id"})
                        return df
                raise ValueError(f"No valid sequence ID column found in {df.columns.tolist()}")

            # Required artifacts
            kmer_df = normalize_id_column(pd.read_csv(context.get("kmer_metrics"), sep="\t"))
            summary_df = normalize_id_column(pd.read_csv(context.get("summary_per_sequence"), sep="\t"))
            merged = kmer_df.merge(summary_df, on="sequence_id", how="left")

            #merge tiara based on sequence_id, if exists
            try:
                tiara_path = context.get("tiara")
            except ValueError:
                tiara_path = None

            if tiara_path:
                tiara_df = normalize_id_column(pd.read_csv(tiara_path, sep="\t"))
                merged = merged.merge(tiara_df, on="sequence_id", how="left")

            #merge rdna-miner taxonomy based on sequence_id, if exists
            try:
                rdna_path = context.get("rdna")
            except ValueError:
                rdna_path = None

            if rdna_path:
                rdna_df = normalize_id_column(pd.read_csv(rdna_path, sep="\t"))
                merged = merged.merge(rdna_df, on="sequence_id", how="left")

            if merged["sequence_id"].duplicated().any():
                raise RuntimeError("Duplicate sequence_id detected after merge.")

            output_path = context.output_dir / "features" / "merged_features.tsv"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            merged.to_csv(output_path, sep="\t", index=False)
            context.register("merged_features", output_path)

 
class SpatialiteDatabase(Operation):
    name = "spatialite-db"
    requires = ["fasta", "merged_features", "dr_embeddings"]
    produces = ["database"]

    def __init__(self, db_name="kmerord.sqlite"):
        self.db_name = db_name

    def run(self, context):
        import pandas as pd
        from kmer_ord.io.database import (initialize_spatialite_db,
                                  create_fasta_table,
                                  populate_fasta_table,
                                  create_features_table,
                                  populate_features_table,
                                  create_coordinates_table,
                                  populate_coordinates_table,
                                  inspect_database)
        output_path = context.output_dir / self.db_name

        if output_path.exists() and not context.force:
            context.register("database", output_path)
            return

        with context.benchmark_timer(label=self.name, input_file=output_path):
            if output_path.exists():
                output_path.unlink()

            conn = initialize_spatialite_db(output_path)

            with context.benchmark_timer("db_fasta"):
                fasta_file = context.get("fasta")
                create_fasta_table(conn)
                populate_fasta_table(conn, fasta_file)

            with context.benchmark_timer("db_features"):
                features_path = context.get("merged_features")
                features_df = pd.read_csv(features_path, sep="\t")

                if "sequence_id" not in features_df.columns:
                    raise RuntimeError("merged_features.tsv must contain a 'sequence_id' column.")

                if features_df["sequence_id"].duplicated().any():
                    raise RuntimeError("Duplicate sequence_id detected in merged_features.")

                create_features_table(conn, features_df)
                populate_features_table(conn, features_df)

            with context.benchmark_timer("db_coordinates"):
                embedding_files = context.get("dr_embeddings")
                if not isinstance(embedding_files, list):
                    embedding_files = [embedding_files]

                coords_df = None
                for emb_file in embedding_files:
                    df_emb = pd.read_csv(emb_file, sep="\t")
                    if "sequence_id" not in df_emb.columns:
                        df_emb["sequence_id"] = features_df["sequence_id"].values

                    if coords_df is None:
                        coords_df = df_emb
                    else:
                        new_cols = [c for c in df_emb.columns if c != "sequence_id" and c not in coords_df.columns]
                        coords_df = coords_df.merge(df_emb[["sequence_id"] + new_cols], on="sequence_id", how="inner")

                cols = ["sequence_id"] + [c for c in coords_df.columns if c != "sequence_id"]
                coords_df = coords_df[cols]

                methods = create_coordinates_table(conn, coords_df)
                populate_coordinates_table(conn, coords_df, methods)

            conn.close()
            context.register("database", output_path)


class Clustering(Operation):
    name = "clustering"
    requires = ["dr_embeddings"]
    produces = ["clusters"]

    # Hardcoded sweep grids
    LEIDEN_RESOLUTIONS = [
        0.001, 0.0025, 0.005, 0.0075,
        0.01, 0.025, 0.05, 0.075,
        0.1, 0.25, 0.5, 0.75,
        1.0, 2.5, 5.0
    ]

    DBSCAN_EPS = [
        0.01, 0.025, 0.05, 0.075,
        0.1, 0.25, 0.5, 0.75,
        1.0, 2.5, 5.0
    ]

    HDBSCAN_MCS = [5, 10, 15, 25, 50, 75, 100]

    def __init__(
        self,
        method="leiden",
        sweep=False,
        knn=15,
        seed=42,
        min_samples=5,
        min_cluster_size=10,
        eps=0.5,
    ):
        self.method = method
        self.sweep = sweep
        self.knn = knn
        self.seed = seed
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.eps = eps

    def run(self, context):
        import pandas as pd
        from kmer_ord.cluster.graph import build_knn_graph
        from kmer_ord.cluster.leiden import run_leiden, leiden_resolution_sweep
        from kmer_ord.cluster.hdbscan import run_hdbscan
        from kmer_ord.cluster.dbscan import run_dbscan

        embedding_files = context.get("dr_embeddings")
        if not isinstance(embedding_files, list):
            embedding_files = [embedding_files]

        cluster_outputs = []

        with context.benchmark_timer(
            label=self.name,
            input_args=f"method={self.method}, sweep={self.sweep}",
        ):
            for emb_path in embedding_files:

                emb_path = Path(emb_path)
                embedding_name = emb_path.stem

                output_file = context.artifact_path(
                    name=f"{embedding_name}_{self.method}_clusters",
                    subdir="clusters",
                    suffix=".tsv"
                )

                if output_file.exists() and not context.force:
                    context.logger.info(f"Skipping clustering, exists: {output_file}")
                    cluster_outputs.append(output_file)
                    continue

                df = pd.read_csv(emb_path, sep="\t")
                sequence_ids = df["sequence_id"]

                # Detect each DR method and its dimensionality from column names.
                # Columns follow the pattern {method}_{dim} e.g. umap_1, umap_2, tsne_1.
                dr_methods = {}
                for col in df.columns:
                    if col != "sequence_id" and col.endswith("_1"):
                        base = col[:-2]
                        n = 1
                        while f"{base}_{n + 1}" in df.columns:
                            n += 1
                        dr_methods[base] = n

                cluster_df = pd.DataFrame({"sequence_id": sequence_ids})

                for dr_method, n_dims in dr_methods.items():
                    prefix = f"{dr_method}_{n_dims}"
                    X = df[[f"{dr_method}_{i + 1}" for i in range(n_dims)]].values

                    with context.benchmark_timer(
                        label=f"cluster_{self.method}_{dr_method}",
                        input_file=emb_path,
                    ):
                        # LEIDEN
                        if self.method == "leiden":
                            A = build_knn_graph(X, k=self.knn)
                            if self.sweep:
                                results = leiden_resolution_sweep(
                                    A,
                                    resolutions=self.LEIDEN_RESOLUTIONS,
                                    seed=self.seed
                                )
                                for r, (labels, modularity) in results.items():
                                    cluster_df[f"{prefix}_leiden_r{r:.4f}"] = labels
                            else:
                                labels, modularity = run_leiden(A, resolution=1.0, seed=self.seed)
                                cluster_df[f"{prefix}_leiden"] = labels

                        # HDBSCAN
                        elif self.method == "hdbscan":
                            if self.sweep:
                                for mcs in self.HDBSCAN_MCS:
                                    labels = run_hdbscan(X, min_cluster_size=mcs, min_samples=self.min_samples)
                                    cluster_df[f"{prefix}_hdbscan_mcs{mcs}"] = labels
                            else:
                                labels = run_hdbscan(
                                    X, min_cluster_size=self.min_cluster_size, min_samples=self.min_samples
                                )
                                cluster_df[f"{prefix}_hdbscan"] = labels

                        # DBSCAN
                        elif self.method == "dbscan":
                            if self.sweep:
                                for eps in self.DBSCAN_EPS:
                                    labels = run_dbscan(X, eps=eps, min_samples=self.min_samples)
                                    cluster_df[f"{prefix}_dbscan_eps{eps:.3f}"] = labels
                            else:
                                labels = run_dbscan(X, eps=self.eps, min_samples=self.min_samples)
                                cluster_df[f"{prefix}_dbscan"] = labels

                        else:
                            raise ValueError(f"Unknown clustering method: {self.method}")

                output_file.parent.mkdir(parents=True, exist_ok=True)
                cluster_df.to_csv(output_file, sep="\t", index=False)
                cluster_outputs.append(output_file)

        # SAFE artifact registration (fixes your NameError issue)
        existing = context.artifacts.get("clusters", [])
        if not isinstance(existing, list):
            existing = [existing]

        context.register("clusters", existing + cluster_outputs)


class AddClusteringToDB(Operation):
    name = "add_clustering_to_db"
    requires = ["dr_embeddings", "clusters"]
    produces = ["database"]

    def __init__(self, db_path: Path, force: bool = False):
        self.db_path = db_path
        self.force = force

    def run(self, context):
        from kmer_ord.io.database import (add_dr_and_clusters_to_db, inspect_database)

        embedding_files = context.get("dr_embeddings")
        cluster_files = context.get("clusters")

        if not isinstance(embedding_files, list):
            embedding_files = [embedding_files]
        if not isinstance(cluster_files, list):
            cluster_files = [cluster_files]

        with context.benchmark_timer(label=self.name, input_file=self.db_path):
            add_dr_and_clusters_to_db(
                db_path=self.db_path,
                embedding_files=embedding_files,
                cluster_files=cluster_files,
                force=self.force
            )

        context.register("database", self.db_path)
        inspect_database(self.db_path)


#----------------------------------------------------------#
# PLOTTING

class PlotFeatures(Operation):
    name = "plot_features"
    requires = ["database"]
    produces = ["plots"]

    def __init__(self, max_categories: int = 10):
        # Plots will always go inside a 'plots' folder in the analysis output directory
        self.max_categories = max_categories

    def run(self, context):
        from kmer_ord.vis.feature_plots import plot_numeric_distributions, plot_categorical_vs_numeric

        db_path = context.get("database")
        output_dir = db_path.parent / "plots" / "features"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Currently only the 'features' table, can expand later
        plot_numeric_distributions(db_path, "features", output_dir)
        plot_categorical_vs_numeric(db_path, "features", output_dir, max_categories=self.max_categories)

        context.register("plots", output_dir)


class PlotEmbeddings(Operation):
    name = "plot_embeddings"
    requires = ["database"]
    produces = ["embedding_plots"]

    def __init__(self, mode: str = "all"):
        self.mode = mode  # "density", "categorical", "continuous", "all"

    def run(self, context):
        from kmer_ord.vis.embedding_plots import plot_embeddings_from_db

        db_path = context.get("database")
        output_dir = context.output_dir / "plots" / "embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_embeddings_from_db(db_path=db_path, output_root=output_dir, mode=self.mode,)

        context.register("embedding_plots", output_dir)

class PlotParamScreen(Operation):
    name = "plot_param_screen"
    produces = ["param_screen_plots"]

    def __init__(self, screen_dir: Path, method: str):
        self.screen_dir = screen_dir
        self.method = method

    def run(self, context):
        from kmer_ord.vis.param_screen import plot_param_screen

        output_dir = context.output_dir / "plots" / "param_screen"
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_param_screen(self.screen_dir, self.method, output_dir)

        context.register("param_screen_plots", output_dir)