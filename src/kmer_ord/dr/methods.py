# src/kmer_ord/dr/methods.py
import numpy as np
import pandas as pd
from pathlib import Path
from kmer_ord.utils.logging_utils import section, info, warn, divider, console
from kmer_ord.utils.benchmark import BenchmarkTimer


def _dr_timer(label, log_dir, script_name=None, **kwargs):
    """BenchmarkTimer that writes to the per-run log when log_dir is set."""
    if log_dir is not None:
        kwargs["log_dir"] = log_dir
    if script_name:
        kwargs["script_name"] = script_name
    return BenchmarkTimer(label=label, **kwargs)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def _resolve_scale(scale: str, n_seq: int) -> str:
    """Map 'auto' to a named scale tier based on dataset size."""
    if scale != "auto":
        return scale
    if n_seq < 5_000:
        return "small"
    if n_seq < 50_000:
        return "medium"
    return "large"

# methods supporting parameter screening
SCREENABLE_METHODS = {"umap", "tsne", "trimap", "pacmap", "localmap"}

#current hyperparameters depending on dataset size (scale)
DR_HYPERPARAMS = {
    'umap': {
        'default': {'n_neighbors': 15, 'min_dist': 0.1},
        'small':   {'n_neighbors': 50, 'min_dist': 0.05},
        'medium':  {'n_neighbors': 150, 'min_dist': 0.1},
        'large':   {'n_neighbors': 200, 'min_dist': 0.1}
    },
    'tsne': {
        'default': {'init': 'pca'},
        'small':   {'perplexity': 30,  'init': 'pca', 'learning_rate': 10},
        'medium':  {'perplexity': 100, 'init': 'pca', 'learning_rate': 10},
        'large':   {'perplexity': 200, 'init': 'pca', 'learning_rate': 10}
    },
    'trimap': {
        'default': {'n_inliers': 10, 'weight_temp': 0.5},
        'small':   {'n_inliers': 50, 'weight_temp': 0.3},
        'medium':  {'n_inliers': 100, 'weight_temp': 0.4},
        'large':   {'n_inliers': 150, 'weight_temp': 0.5}
    },
    'pacmap': {
        'default': {'MN_ratio': 0.5, 'FP_ratio': 2},
        'small':   {'n_neighbors': 15, 'MN_ratio': 0.5, 'FP_ratio': 2},
        'medium':  {'n_neighbors': 100, 'MN_ratio': 0.5, 'FP_ratio': 3},
        'large':   {'n_neighbors': 200, 'MN_ratio': 0.5, 'FP_ratio': 5}
    },
    'localmap': {
        'default': {'MN_ratio': 0.5, 'FP_ratio': 0.5},
        'small':   {'n_neighbors': 15, 'MN_ratio': 0.3, 'FP_ratio': 0.5},
        'medium':  {'n_neighbors': 100, 'MN_ratio': 0.5, 'FP_ratio': 1.0},
        'large':   {'n_neighbors': 200, 'MN_ratio': 0.7, 'FP_ratio': 1.0}
    },
    'pca': {'default': {}, 'small': {}, 'medium': {}, 'large': {}},
    'sparse_pca': {'default': {}, 'small': {}, 'medium': {}, 'large': {}},
    'kernel_pca': {'default': {}, 'small': {}, 'medium': {}, 'large': {}},
    'lle': {'default': {}, 'small': {}, 'medium': {}, 'large': {}},
}

ALL_METHODS = ["umap", "tsne", "trimap", "pacmap", "localmap", "pca"]


def estimate_peak_memory_gb(n_seq: int, n_feat: int, method: str, scale: str) -> float:
    """Rough upper bound on peak RAM (GB) for one DR fit on an n_seq x n_feat
    float32 matrix at the given (resolved) scale preset.

    Unlike the old `X.nbytes * 4` heuristic, this accounts for the structures
    that actually dominate at scale (per the memory audit): sparse neighbor
    graphs (UMAP), the sparse P-matrix over ~3*perplexity neighbors (t-SNE),
    near/mid/far pair tables (PaCMAP/LocalMAP), triplet index arrays (TriMAP),
    and sklearn's float64 working copy (PCA/t-SNE). Byte-per-entry constants
    are deliberately generous — this is a guard, not a profiler.
    """
    method = method.lower()
    params = DR_HYPERPARAMS.get(method, {}).get(scale, {})
    x_bytes = n_seq * n_feat * 4  # the float32 matrix held for the whole stage

    if method == "pca":
        # sklearn PCA copies float32 input to a float64 workspace
        extra = n_seq * n_feat * 8
    elif method == "tsne":
        # float64 copy + sparse P-matrix over ~3*perplexity neighbors/point
        # (~16 bytes per stored entry: value + index + sparse overhead)
        perplexity = params.get("perplexity", 30)
        extra = n_seq * n_feat * 8 + n_seq * 3 * perplexity * 16
    elif method == "umap":
        # symmetrised fuzzy graph is O(n x k); factor 2 covers NN-descent
        # scratch structures alongside the final graph
        n_neighbors = params.get("n_neighbors", 15)
        extra = n_seq * n_neighbors * 16 * 2
    elif method in ("pacmap", "localmap"):
        # pairs ~= n x k x (1 + MN_ratio + FP_ratio); ~12 bytes per pair
        # (two int32 endpoints + optimizer state)
        n_neighbors = params.get("n_neighbors", 10)
        mn_ratio = params.get("MN_ratio", 0.5)
        fp_ratio = params.get("FP_ratio", 2)
        extra = n_seq * n_neighbors * (1 + mn_ratio + fp_ratio) * 12
    elif method == "trimap":
        # triplets ~= n x (n_inliers x n_outliers + n_random); library
        # defaults n_outliers=5, n_random=5; ~16 bytes per triplet
        n_inliers = params.get("n_inliers", 10)
        extra = n_seq * (n_inliers * 5 + 5) * 16
    else:
        # unknown method (kernel_pca, sparse_pca, lle, ...): keep the old
        # conservative multiplier so the guard never weakens
        extra = x_bytes * 3

    return (x_bytes + extra) / (1024 ** 3)


def _run_single_method(
    X: np.ndarray,
    method: str,
    dims: int,
    seed: int,
    scale: str = "default",
    n_jobs: int = 1,
):
    import pandas as pd
    import scipy.sparse as sparse
    from sklearn.decomposition import PCA, KernelPCA, SparsePCA
    from sklearn.manifold import TSNE, LocallyLinearEmbedding

    try:
        import umap
    except ImportError:
        umap = None

    #from trimap import TRIMAP
    #from pacmap import PaCMAP
    #from pacmap.pacmap import LocalMAP
    method = method.lower()
    params = DR_HYPERPARAMS.get(method, {}).get(scale, {})
    graph = None

    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=dims, **params)
        embedding = model.fit_transform(X)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        model = TSNE(n_components=dims, random_state=seed, n_jobs=n_jobs, **params)
        embedding = model.fit_transform(X)

    elif method == "umap":
        import umap
        warn("UMAP random seed is disabled to allow parallel execution.")
        model = umap.UMAP(n_components=dims, random_state=None, n_jobs=n_jobs, **params)
        embedding = model.fit_transform(X)
        graph = getattr(model, "graph_", None)

    elif method == "trimap":
        from trimap import TRIMAP
        model = TRIMAP(n_dims=dims, **params)
        embedding = model.fit_transform(X)

    elif method == "pacmap":
        from pacmap import PaCMAP
        model = PaCMAP(n_components=dims, **params)
        embedding = model.fit_transform(X)
        graph = getattr(model, "graph_", None)

    elif method == "localmap":
        from pacmap.pacmap import LocalMAP
        model = LocalMAP(n_components=dims, **params)
        embedding = model.fit_transform(X)
        graph = getattr(model, "graph_", None)

    elif method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        n_neighbors = params.get("n_neighbors", 10)
        model = LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=dims,
            n_jobs=n_jobs,
        )
        embedding = model.fit_transform(X)

    elif method == "sparse_pca":
        from sklearn.decomposition import SparsePCA
        model = SparsePCA(n_components=dims, random_state=seed, n_jobs=n_jobs, **params)
        embedding = model.fit_transform(X)

    elif method == "kernel_pca":
        from sklearn.decomposition import KernelPCA
        model = KernelPCA(n_components=dims, n_jobs=n_jobs, **params)
        embedding = model.fit_transform(X)

    else:
        raise ValueError(f"Unsupported DR method: {method}")

    # ensure graph is sparse if present
    if graph is not None and not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)

    return embedding, graph


def run_dr_methods(
    X: np.ndarray | pd.DataFrame,
    methods: list[str],
    dims: int,
    seed: int,
    scale: str,
    screen_params: bool,
    output_dir: Path,
    normalisation: str,
    input_name: str,
    sequence_ids: list | pd.Index | None = None,
    n_jobs: int = 1,
    screen_values1: list[str] | None = None,
    screen_values2: list[str] | None = None,
    screen_range1: list[str] | None = None,
    screen_range2: list[str] | None = None,
    screen_grid: str | None = None,
    log_dir: str | None = None,
    script_name: str | None = None,
) -> tuple[Path, list[Path]]:
    """
    Run selected DR methods for a single normalisation.

    Adds sequence_id column to all embeddings for downstream merging.
    Saves graph objects if produced by a method.

    Returns
    -------
    merged_file : Path
        Path to merged embeddings file
    graph_paths : list[Path]
        List of graph files saved (may be empty)
    """
    import pandas as pd
    import numpy as np
    import scipy.sparse as sparse
    import time

    # Sequence IDs
    if sequence_ids is None:
        if isinstance(X, pd.DataFrame):
            sequence_ids = X.index
        else:
            sequence_ids = np.arange(X.shape[0])

    if "all" in methods:
        methods = ALL_METHODS

    output_dir.mkdir(parents=True, exist_ok=True)

    n_seq, n_feat = X.shape
    resolved_scale = _resolve_scale(scale, n_seq)
    w = 16
    section(f"dimensionality reduction  ·  {normalisation}")
    info(f"{'input':<{w}}  {n_seq:,} sequences  ×  {n_feat:,} features")
    info(f"{'methods':<{w}}  {', '.join(methods)}")
    scale_label = f"{resolved_scale}  (auto)" if scale == "auto" else resolved_scale
    info(f"{'scale / dims':<{w}}  {scale_label}  /  {dims}D")
    if n_jobs > 1:
        info(f"{'threads':<{w}}  {n_jobs}")

    divider()

    dfs = []
    graph_paths: list[Path] = []

    for method in methods:

        method = method.lower()

        method_dir = output_dir / normalisation / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # Parameter screening
        if screen_params and method in SCREENABLE_METHODS:
            screen_dir = method_dir / "parameter_screen"
            screen_dir.mkdir(parents=True, exist_ok=True)

            _run_parameter_screen(
                X=X,
                method=method,
                dims=dims,
                seed=seed,
                scale=resolved_scale,
                output_dir=screen_dir,
                normalisation=normalisation,
                input_name=input_name,
                sequence_ids=sequence_ids,
                n_jobs=n_jobs,
                values1=screen_values1,
                values2=screen_values2,
                range1=screen_range1,
                range2=screen_range2,
                grid=screen_grid,
                log_dir=log_dir,
                script_name=script_name,
            )

        # Default embedding
        m = 14
        params = DR_HYPERPARAMS.get(method, {}).get(resolved_scale, {})
        info(f"{method:<{m}}  running")
        if params:
            params_str = "  ".join(f"{k}={v}" for k, v in params.items())
            info(f"{'':>{m}}  {params_str}")
        t0 = time.perf_counter()

        with _dr_timer(
            label=f"dr_{normalisation}_{method}",
            log_dir=log_dir,
            script_name=script_name,
        ):
            embedding, graph = _run_single_method(
                X=X,
                method=method,
                dims=dims,
                seed=seed,
                scale=resolved_scale,
                n_jobs=n_jobs,
            )

        elapsed = time.perf_counter() - t0

        # Save embedding
        columns = [f"{method}_{i+1}" for i in range(dims)]
        df_embed = pd.DataFrame(embedding, columns=columns)
        df_embed.insert(0, "sequence_id", sequence_ids)

        out_file = method_dir / f"{input_name}_{normalisation}_{method}_{dims}D.tsv"
        df_embed.to_csv(out_file, sep="\t", index=False)

        # Save graph if available
        graph_note = ""
        if graph is not None:
            graph_file = method_dir / f"{input_name}_{normalisation}_{method}_graph.npz"
            sparse.save_npz(graph_file, graph)
            graph_paths.append(graph_file)
            graph_note = "  (+ graph)"

        info(f"{'':>{m}}  done  {_fmt_time(elapsed)}{graph_note}")
        divider()

        dfs.append(df_embed)

    # Merge embeddings across methods
    merged_df = pd.concat(dfs, axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    merged_file = output_dir / normalisation / f"{input_name}_{normalisation}_{dims}D_merged_embeddings.tsv"
    merged_df.to_csv(merged_file, sep="\t", index=False)
    info(f"merged  →  {normalisation}/{merged_file.name}")

    return merged_file, graph_paths


def _run_parameter_screen(
    X: pd.DataFrame,
    method: str,
    dims: int,
    seed: int,
    scale: str,
    output_dir: Path,
    normalisation: str,
    input_name: str,
    sequence_ids: list | pd.Index,
    n_jobs: int = 1,
    values1: list[str] | None = None,
    values2: list[str] | None = None,
    range1: list[str] | None = None,
    range2: list[str] | None = None,
    grid: str | None = None,
    log_dir: str | None = None,
    script_name: str | None = None,
) -> list[Path]:
    """
    Perform parameter screening for a given DR method.
    Saves individual files for each parameter combination with sequence_id column.
    Returns list of saved paths.

    Grid resolution (values1/values2/range1/range2/grid) is handled by
    dr/screen_grid.py::resolve_method_grid() — see its docstring. No overrides
    given at all reproduces the original hardcoded 2D grid exactly. A 1D grid
    (axis2_vals is None) holds the second parameter fixed at this method's own
    scale-preset default from DR_HYPERPARAMS.
    """

    import pandas as pd
    import numpy as np
    import scipy.sparse as sparse
    from sklearn.decomposition import PCA, KernelPCA, SparsePCA
    from sklearn.manifold import TSNE, LocallyLinearEmbedding

    try:
        import umap
    except ImportError:
        umap = None

    from trimap import TRIMAP
    from pacmap import PaCMAP
    from pacmap.pacmap import LocalMAP

    from kmer_ord.vis.embedding_plots import render_param_screen_density_grid
    from kmer_ord.dr.screen_grid import resolve_method_grid, SCREEN_AXES

    output_paths = []
    # (axis1_value, axis2_value, coords_df) per combination — collected so a
    # single density grid can be rendered once per method after its loop,
    # without re-reading anything back from disk.
    density_combos: list[tuple[float, float, pd.DataFrame]] = []

    # single source of truth for embedding column names: save_embedding,
    # track_density and the render call below must all agree on these
    coord_cols = [f"{method}_{i+1}" for i in range(dims)]

    def save_embedding(embedding: np.ndarray, param_str: str):
        """Helper to save a DataFrame with sequence_id."""
        df = pd.DataFrame(embedding, columns=coord_cols)
        df.insert(0, "sequence_id", sequence_ids)
        out_file = output_dir / f"{input_name}_{normalisation}_{method}_{param_str}_{dims}D.tsv"
        df.to_csv(out_file, sep="\t", index=False)
        output_paths.append(out_file)
        return df

    def track_density(axis1_value: float, axis2_value: float, df: pd.DataFrame):
        if dims >= 2:
            # keep only the two coordinate columns the density renderer reads.
            # Retaining the full frame would hold one duplicated object-dtype
            # sequence_id column (~95 bytes/row) per grid combination for the
            # entire screen — a multi-GB accumulation on large read sets. The
            # .copy() detaches the slice from the full frame so the original
            # (and its sequence_id strings) is freed at end of iteration.
            coords = df[coord_cols[:2]].copy()
            density_combos.append((axis1_value, axis2_value, coords))

    def timed_fit(param_str, model):
        with _dr_timer(
            label=f"dr_screen_{normalisation}_{method}_{param_str}",
            log_dir=log_dir,
            script_name=script_name,
        ):
            return model.fit_transform(X)

    import time as _time

    pw = 36  # fixed width for params column so elapsed times align

    axis1_name, axis2_name = SCREEN_AXES[method] if method in SCREEN_AXES else (None, None)
    axis1_vals, axis2_vals = (
        resolve_method_grid(method, values1, values2, range1, range2, grid)
        if method in SCREEN_AXES else (None, None)
    )
    default_axis2 = DR_HYPERPARAMS.get(method, {}).get("default", {}).get(axis2_name)

    if method == "umap":
        if axis2_vals is None:
            section(f"parameter screen  ·  umap  ({len(axis1_vals)} values, 1D)")
            for n in axis1_vals:
                t0 = _time.perf_counter()
                kwargs = {"n_neighbors": n}
                if default_axis2 is not None:
                    kwargs["min_dist"] = default_axis2
                model = umap.UMAP(n_components=dims, n_jobs=n_jobs, **kwargs)
                embedding = timed_fit(f"n{n}", model)
                df = save_embedding(embedding, param_str=f"n{n}")
                track_density(n, default_axis2 or 0, df)
                info(f"{'n_neighbors=' + str(n):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")
        else:
            section(f"parameter screen  ·  umap  ({len(axis1_vals) * len(axis2_vals)} combinations)")
            for n in axis1_vals:
                for m in axis2_vals:
                    t0 = _time.perf_counter()
                    model = umap.UMAP(n_components=dims, n_neighbors=n, min_dist=m, n_jobs=n_jobs)
                    embedding = timed_fit(f"n{n}_min{m}", model)
                    df = save_embedding(embedding, param_str=f"n{n}_min{m}")
                    track_density(n, m, df)
                    info(f"{'n_neighbors=' + str(n) + '  min_dist=' + str(m):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    elif method == "tsne":
        if axis2_vals is None:
            section(f"parameter screen  ·  tsne  ({len(axis1_vals)} values, 1D)")
            for p in axis1_vals:
                t0 = _time.perf_counter()
                kwargs = {"perplexity": p}
                if default_axis2 is not None:
                    kwargs["learning_rate"] = default_axis2
                model = TSNE(n_components=dims, max_iter=1000, random_state=seed, n_jobs=n_jobs, **kwargs)
                embedding = timed_fit(f"p{p}", model)
                df = save_embedding(embedding, param_str=f"p{p}")
                track_density(p, default_axis2 or 0, df)
                info(f"{'perplexity=' + str(p):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")
        else:
            section(f"parameter screen  ·  tsne  ({len(axis1_vals) * len(axis2_vals)} combinations)")
            for p in axis1_vals:
                for lr in axis2_vals:
                    t0 = _time.perf_counter()
                    model = TSNE(n_components=dims, perplexity=p, learning_rate=lr,
                                 max_iter=1000, random_state=seed, n_jobs=n_jobs)
                    embedding = timed_fit(f"p{p}_lr{lr}", model)
                    df = save_embedding(embedding, param_str=f"p{p}_lr{lr}")
                    track_density(p, lr, df)
                    info(f"{'perplexity=' + str(p) + '  learning_rate=' + str(lr):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    elif method == "trimap":
        if axis2_vals is None:
            section(f"parameter screen  ·  trimap  ({len(axis1_vals)} values, 1D)")
            for n in axis1_vals:
                t0 = _time.perf_counter()
                kwargs = {"n_inliers": n}
                if default_axis2 is not None:
                    kwargs["weight_temp"] = default_axis2
                model = TRIMAP(n_dims=dims, **kwargs)
                embedding = timed_fit(f"inliers{n}", model)
                df = save_embedding(embedding, param_str=f"inliers{n}")
                track_density(n, default_axis2 or 0, df)
                info(f"{'n_inliers=' + str(n):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")
        else:
            section(f"parameter screen  ·  trimap  ({len(axis1_vals) * len(axis2_vals)} combinations)")
            for n in axis1_vals:
                for w in axis2_vals:
                    t0 = _time.perf_counter()
                    model = TRIMAP(n_dims=dims, n_inliers=n, weight_temp=w)
                    embedding = timed_fit(f"inliers{n}_weighttemp{w}", model)
                    df = save_embedding(embedding, param_str=f"inliers{n}_weighttemp{w}")
                    track_density(n, w, df)
                    info(f"{'n_inliers=' + str(n) + '  weight_temp=' + str(w):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    elif method == "pacmap":
        if axis2_vals is None:
            section(f"parameter screen  ·  pacmap  ({len(axis1_vals)} values, 1D)")
            for n in axis1_vals:
                t0 = _time.perf_counter()
                kwargs = {"n_neighbors": n}
                if default_axis2 is not None:
                    kwargs["FP_ratio"] = default_axis2
                model = PaCMAP(n_components=dims, **kwargs)
                embedding = timed_fit(f"n{n}", model)
                df = save_embedding(embedding, param_str=f"n{n}")
                track_density(n, default_axis2 or 0, df)
                info(f"{'n_neighbors=' + str(n):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")
        else:
            section(f"parameter screen  ·  pacmap  ({len(axis1_vals) * len(axis2_vals)} combinations)")
            for n in axis1_vals:
                for fp in axis2_vals:
                    t0 = _time.perf_counter()
                    model = PaCMAP(n_components=dims, n_neighbors=n, FP_ratio=fp)
                    embedding = timed_fit(f"n{n}_FPratio{fp}", model)
                    df = save_embedding(embedding, param_str=f"n{n}_FPratio{fp}")
                    track_density(n, fp, df)
                    info(f"{'n_neighbors=' + str(n) + '  FP_ratio=' + str(fp):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    elif method == "localmap":
        if axis2_vals is None:
            section(f"parameter screen  ·  localmap  ({len(axis1_vals)} values, 1D)")
            for n in axis1_vals:
                t0 = _time.perf_counter()
                kwargs = {"n_neighbors": n}
                if default_axis2 is not None:
                    kwargs["FP_ratio"] = default_axis2
                model = LocalMAP(n_components=dims, **kwargs)
                embedding = timed_fit(f"n{n}", model)
                df = save_embedding(embedding, param_str=f"n{n}")
                track_density(n, default_axis2 or 0, df)
                info(f"{'n_neighbors=' + str(n):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")
        else:
            section(f"parameter screen  ·  localmap  ({len(axis1_vals) * len(axis2_vals)} combinations)")
            for n in axis1_vals:
                for fp in axis2_vals:
                    t0 = _time.perf_counter()
                    model = LocalMAP(n_components=dims, n_neighbors=n, FP_ratio=fp)
                    embedding = timed_fit(f"n{n}_FPratio{fp}", model)
                    df = save_embedding(embedding, param_str=f"n{n}_FPratio{fp}")
                    track_density(n, fp, df)
                    info(f"{'n_neighbors=' + str(n) + '  FP_ratio=' + str(fp):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    else:
        raise ValueError(f"Parameter screening not implemented for method: {method}")

    # Default output: a per-panel density grid, rendered once per method,
    # requiring no feature/label join — see vis/embedding_plots.py's
    # render_param_screen_density_grid() docstring for why panels are scaled
    # independently rather than to a shared axis range.
    if density_combos:
        render_param_screen_density_grid(
            combos=density_combos,
            axis1_name=axis1_name,
            axis2_name=axis2_name,
            xcol=coord_cols[0],
            ycol=coord_cols[1],
            outdir=output_dir,
            method=method,
        )

    return output_paths