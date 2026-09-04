# src/kmer_ord/vis/embedding_plots.py

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
from colorcet import fire, CET_L17, glasbey
from PIL import Image, ImageDraw, ImageFont

from kmer_ord.utils.logging_utils import section, info, warn

# Internal identifier columns (one unique value per sequence) that must never
# be treated as plottable features — colouring an embedding by them is
# meaningless, and datashader's ds.by() allocates a
# (height, width, n_categories) uint32 array, so ~880k sequence_ids at
# 400x400 means a ~526 GiB allocation (MemoryError).
IGNORE_FEATURES = {"sequence_id", "header"}

# Hard cap on the ds.by() aggregation array size for a single categorical
# plot. 1 GiB allows ~1,600 categories at 400x400 — already far beyond the
# 256-colour glasbey palette, so anything skipped here would have been an
# unreadable plot anyway.
MAX_CATEGORICAL_AGG_BYTES = 2**30


def _plottable_feature_columns(features_df):
    """Split a features table into (categorical, continuous) column lists
    suitable for embedding plots, excluding internal identifier columns.
    Returns empty lists when the features table is absent (None)."""
    if features_df is None:
        return [], []
    categorical = [
        col for col in features_df.select_dtypes(include=["object", "category"]).columns
        if col not in IGNORE_FEATURES
    ]
    continuous = [
        col for col in features_df.select_dtypes(include=[np.number]).columns
        if col not in IGNORE_FEATURES
    ]
    return categorical, continuous


def _connect_spatialite(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)

    try:
        conn.execute("SELECT load_extension('mod_spatialite');")
    except sqlite3.OperationalError:
        # On macOS / conda sometimes name differs
        try:
            conn.execute("SELECT load_extension('mod_spatialite.so');")
        except Exception as e:
            raise RuntimeError(
                "SpatiaLite extension could not be loaded. "
                "Ensure mod_spatialite is installed."
            ) from e

    return conn


def plot_embeddings_from_db(db_path: Path, output_root: Path, mode: str = "all"):
    conn = _connect_spatialite(db_path)
    methods = _discover_geometry_methods(conn)
    if not methods:
        return

    features_df = _load_features(conn)
    cluster_df = _load_clusters(conn)
    categorical_cols, continuous_cols = _plottable_feature_columns(features_df)

    # Store precomputed images per method/feature
    per_method_feature = {method: {"categorical": {}, "continuous": {}} for method in methods}

    # Global value range per continuous feature, computed once so every DR
    # method's panel — and the shared legend on a multi-method composite —
    # maps the same value to the same color.
    continuous_ranges = {}
    for col in continuous_cols:
        vals = features_df[col].dropna()
        if not vals.empty:
            continuous_ranges[col] = (float(vals.min()), float(vals.max()))

    for method in methods:
        emb_df = _extract_method_coordinates(conn, method)
        if emb_df.empty:
            continue

        merged = _merge_tables(emb_df, features_df, cluster_df)
        xcol = f"{method}_1"
        ycol = f"{method}_2"
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # Compute and save density plots
        if mode in ("density", "all"):
            _render_density(merged, xcol, ycol, method_dir, method)

        # Loop over categorical features
        if mode in ("categorical", "all"):
            for col in categorical_cols:
                img, color_key = _render_feature_to_image(merged, xcol, ycol, col, mode="categorical")
                if img is not None:
                    per_method_feature[method]["categorical"][col] = (img, color_key)
                    legend_img = _add_categorical_legend(img, color_key)
                    legend_img.save(str(method_dir / f"{method}_by_{col}.png"))

        # Loop over continuous features
        if mode in ("continuous", "all"):
            for col in continuous_cols:
                img, value_range = _render_feature_to_image(
                    merged, xcol, ycol, col, mode="continuous",
                    value_range=continuous_ranges.get(col))
                if img is not None:
                    per_method_feature[method]["continuous"][col] = (img, value_range)
                    legend_img = _add_continuous_legend(img, value_range)
                    legend_img.save(str(method_dir / f"{method}_by_{col}.png"))

    # Build feature composites from cached images
    for col in categorical_cols:
        _render_feature_composite_from_cache(per_method_feature, col, output_root / "feature_composites", mode="categorical")

    for col in continuous_cols:
        _render_feature_composite_from_cache(per_method_feature, col, output_root / "feature_composites", mode="continuous")

    conn.close()




def _discover_geometry_methods(conn):
    query = """
        SELECT f_geometry_column
        FROM geometry_columns
        WHERE f_table_name = 'coordinates';
    """

    df = pd.read_sql_query(query, conn)

    if df.empty:
        return []

    return df["f_geometry_column"].tolist()

def _extract_method_coordinates(conn, method):
    select_sql = f"""
        SELECT
            sequence_id,
            ST_X({method}) AS {method}_1,
            ST_Y({method}) AS {method}_2
        FROM coordinates;
    """
    df = pd.read_sql_query(select_sql, conn)
    return df.dropna(subset=[f"{method}_1", f"{method}_2"])


def _load_features(conn):
    try:
        return pd.read_sql_query("SELECT * FROM features;", conn)
    except Exception:
        return None


def _load_clusters(conn):
    try:
        return pd.read_sql_query("SELECT * FROM clusters;", conn)
    except Exception:
        return None


def _merge_tables(emb_df, features_df, cluster_df=None):
    df = emb_df.copy()
    if features_df is not None:
        df = df.merge(features_df, on="sequence_id", how="left") 
    if cluster_df is not None:
        df = df.merge(cluster_df, on="sequence_id", how="left")
    
    return df


# ============================================================

def _base_canvas(df, xcol, ycol, width=1200, height=1000):
    return ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(df[xcol].min(), df[xcol].max()),
        y_range=(df[ycol].min(), df[ycol].max()),
    )


def _render_density(df, xcol, ycol, outdir, method):
    info(f"plotting datashader density for method: {method}")

    cvs = _base_canvas(df, xcol, ycol)
    agg = cvs.points(df, xcol, ycol)
    img = tf.shade(agg, cmap=fire)
    img = tf.dynspread(img)

    export_image(img, filename=f"{method}_density", export_path=str(outdir))


def _render_feature_to_image(df, xcol, ycol, feature_col, mode="categorical", width=400, height=400, value_range=None):
    """
    Generate a datashader image for a single feature (categorical or continuous) for one DR method.
    Returns (datashader.Image, legend_info): for categorical mode legend_info is a color_key dict
    (category label -> hex color); for continuous mode it's the (vmin, vmax) span actually used.
    None whenever no image is produced.

    value_range : optional (vmin, vmax) to use for continuous shading instead of this panel's own
    local min/max. Pass the feature's global min/max (computed once, before looping over DR
    methods) so every method's panel maps the same value to the same color — a fixed per-panel
    span from local data would otherwise make identical colors mean different things in different
    panels, which is especially misleading once panels share one legend in a composite.
    """
    NA_LABEL = "NA"
    NA_COLOR = "#BEBEBE"

    if mode == "categorical":
        # Keep every point with valid coordinates, regardless of whether it
        # has this feature — sequences lacking the feature (e.g. reads with
        # no rDNA hit) are shown in grey rather than dropped, so the canvas
        # range reflects the full embedding and colored points keep their
        # spatial context instead of being plotted on an otherwise-empty,
        # cropped-to-subset canvas.
        df_local = df[[xcol, ycol, feature_col]].dropna(subset=[xcol, ycol]).copy()
    else:
        df_local = df[[xcol, ycol, feature_col]].dropna().copy()
    if df_local.empty:
        return None, None

    cvs = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(df_local[xcol].min(), df_local[xcol].max()),
        y_range=(df_local[ycol].min(), df_local[ycol].max())
    )

    if mode == "categorical":
        is_na = df_local[feature_col].isna()
        df_local[feature_col] = df_local[feature_col].astype(object)
        df_local.loc[is_na, feature_col] = NA_LABEL
        df_local[feature_col] = df_local[feature_col].astype(str).astype("category")

        # ds.by() allocates a (height, width, n_categories) uint32 array, so
        # a high-cardinality column (e.g. an identifier that slipped past
        # column filtering) would request hundreds of GiB. Skip instead.
        n_categories = len(df_local[feature_col].cat.categories)
        agg_bytes = width * height * n_categories * 4
        if agg_bytes > MAX_CATEGORICAL_AGG_BYTES:
            warn(f"Skipping {feature_col}: {n_categories} unique values would "
                 f"need {agg_bytes / 2**30:.1f} GiB to aggregate")
            return None, None

        agg = cvs.points(df_local, xcol, ycol, ds.by(feature_col, ds.count()))
        categories = list(df_local[feature_col].cat.categories)

        if not categories:
            return None, None

        real_categories = [c for c in categories if c != NA_LABEL]
        if len(real_categories) == 1:
            color_key = {real_categories[0]: "#FF0000"}
        else:
            # glasbey is a qualitative palette built for maximally-distinct
            # adjacent colors; sampling consecutive indices out of a smooth
            # sequential colormap like `fire` here produced near-identical
            # colors for low category counts (e.g. fire[0]/fire[1] are both
            # effectively black), making categories visually indistinguishable.
            palette = [glasbey[i % len(glasbey)] for i in range(len(real_categories))]
            color_key = dict(zip(real_categories, palette))
        if NA_LABEL in categories:
            color_key[NA_LABEL] = NA_COLOR

        # min_alpha=255: with categories this imbalanced (NA usually vastly
        # outnumbers real hits), most cells hold count 1-3 at this
        # resolution. The default alpha scaling renders those near-transparent,
        # making the majority category (here, the grey NA background) barely
        # visible even though every point is being drawn.
        img = tf.shade(agg, color_key=color_key, how="eq_hist", alpha=255, min_alpha=255)

    elif mode == "continuous":
        df_local[feature_col] = df_local[feature_col].astype(float)
        agg = cvs.points(df_local, xcol, ycol, ds.mean(feature_col))
        if value_range is not None:
            vmin, vmax = value_range
        else:
            vmin, vmax = float(df_local[feature_col].min()), float(df_local[feature_col].max())
        # how="linear" + an explicit span, not "eq_hist": eq_hist equalizes
        # against this panel's own histogram, so the same color would mean
        # different values in different panels — fine standalone, but
        # unfaithful once panels are compared side by side or share a legend.
        img = tf.shade(agg, cmap=CET_L17, how="linear", span=(vmin, vmax), alpha=255, min_alpha=255)
        color_key = (vmin, vmax)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    img = tf.dynspread(img)
    return img, color_key


def _draw_legend_onto(pil_img, color_key, swatch=14, row_h=20, padding=8):
    """Return a copy of pil_img with a legend panel (swatch + label per
    category) appended to its right edge."""
    if not color_key:
        return pil_img

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    measurer = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    labels = list(color_key.items())
    max_text_w = max(
        measurer.textbbox((0, 0), str(label), font=font)[2] for label, _ in labels
    )
    legend_w = padding * 3 + swatch + max_text_w
    legend_h = padding * 2 + row_h * len(labels)

    canvas_h = max(pil_img.height, legend_h)
    composite = Image.new("RGBA", (pil_img.width + legend_w, canvas_h), (255, 255, 255, 255))
    composite.paste(pil_img, (0, 0), pil_img)

    draw = ImageDraw.Draw(composite)
    y = padding
    for label, color in labels:
        swatch_box = [pil_img.width + padding, y, pil_img.width + padding + swatch, y + swatch]
        draw.rectangle(swatch_box, fill=color, outline="black")
        draw.text((pil_img.width + padding * 2 + swatch, y - 1), str(label), fill="black", font=font)
        y += row_h

    return composite


def _add_categorical_legend(img, color_key, **kwargs):
    """Datashader-Image entry point for _draw_legend_onto."""
    return _draw_legend_onto(img.to_pil().convert("RGBA"), color_key, **kwargs)


def _draw_colorbar_onto(pil_img, vmin, vmax, cmap, bar_w=18, padding=8):
    """Return a copy of pil_img with a vertical colorbar (top=vmax, bottom=vmin)
    plus endpoint labels appended to its right edge."""
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    label_w = 60
    legend_w = padding * 3 + bar_w + label_w
    bar_h = max(pil_img.height - 2 * padding, 1)

    composite = Image.new("RGBA", (pil_img.width + legend_w, pil_img.height), (255, 255, 255, 255))
    composite.paste(pil_img, (0, 0), pil_img)

    draw = ImageDraw.Draw(composite)
    bar_x0 = pil_img.width + padding
    bar_y0 = padding
    n = len(cmap)
    for i in range(bar_h):
        frac = 1 - (i / max(bar_h - 1, 1))  # top of bar = vmax, bottom = vmin
        color = cmap[int(round(frac * (n - 1)))]
        draw.line([(bar_x0, bar_y0 + i), (bar_x0 + bar_w, bar_y0 + i)], fill=color)
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline="black")

    draw.text((bar_x0 + bar_w + 6, bar_y0 - 2), f"{vmax:.3g}", fill="black", font=font)
    draw.text((bar_x0 + bar_w + 6, bar_y0 + bar_h - 12), f"{vmin:.3g}", fill="black", font=font)

    return composite


def _add_continuous_legend(img, value_range, **kwargs):
    """Datashader-Image entry point for _draw_colorbar_onto."""
    pil_img = img.to_pil().convert("RGBA")
    if value_range is None:
        return pil_img
    vmin, vmax = value_range
    return _draw_colorbar_onto(pil_img, vmin, vmax, CET_L17, **kwargs)


def render_param_screen_density_grid(
    combos: list[tuple[float, float, pd.DataFrame]],
    axis1_name: str,
    axis2_name: str,
    xcol: str,
    ycol: str,
    outdir: Path,
    method: str,
    width: int = 220,
    height: int = 220,
    padding: int = 6,
    border: int = 1,
):
    """
    Render a 2D small-multiple grid of datashader density plots for a
    parameter screen — rows = axis1 values, columns = axis2 values. Each
    panel is scaled to its own data range, deliberately not a shared range:
    hyperparameter settings that collapse structure produce very different
    coordinate spans, and a shared range would shrink a collapsed cluster to
    an unreadable speck instead of showing it fill its own panel.

    No feature/label join needed — this is density-only, so it works on any
    dataset regardless of whether ground-truth or feature columns exist.

    Rendering notes:
    - Each panel gets a real `fire` intensity gradient via `eq_hist`, composited
      onto an explicit BLACK canvas (`tf.set_background(img, "black")`) rather
      than left transparent. `eq_hist` ranks populated cells by their
      *relative* position within that panel's own count histogram, so on a
      transparent/white canvas the majority value can land at either end of
      the colormap depending on incidental point-overlap unique to that
      panel — on one combination the data rendered solid black, on another
      solid (invisible) white, with no single colormap orientation fixing
      both. Forcing a real black background sidesteps the ranking question
      instead of fighting it: `fire`'s darkest in-range color is still never
      pure black, so populated cells stay visible against the background no
      matter which end of the histogram they rank into.
    - Pastes each panel using itself as its own alpha mask
      (`composite.paste(panel, xy, panel)`), not a plain paste. A plain
      `Image.paste()` of an RGBA source copies its alpha channel over the
      destination's too, silently turning most of the opaque white canvas
      background transparent wherever the source panel itself was
      transparent — which, combined with sparse per-panel data, made whole
      panels disappear. Now moot for the panels themselves (they're fully
      opaque after `set_background`) but kept for safety.

    Parameters
    ----------
    combos : list of (axis1_value, axis2_value, df) — df has columns xcol, ycol
    axis1_name, axis2_name : hyperparameter names, used as row/column labels
    xcol, ycol : coordinate column names present in every combo's df
    outdir : directory to save {method}_param_screen_density.png into
    method : DR method name, for the title / filename
    """
    if not combos:
        return

    axis1_vals = sorted({v1 for v1, _, _ in combos})
    axis2_vals = sorted({v2 for _, v2, _ in combos})
    by_pos = {(v1, v2): df for v1, v2, df in combos}

    panels = {}
    for (v1, v2), df in by_pos.items():
        df_local = df[[xcol, ycol]].dropna()
        if df_local.empty:
            continue
        cvs = ds.Canvas(
            plot_width=width, plot_height=height,
            x_range=(df_local[xcol].min(), df_local[xcol].max()),
            y_range=(df_local[ycol].min(), df_local[ycol].max()),
        )
        agg = cvs.points(df_local, xcol, ycol)
        img = tf.shade(agg, cmap=fire, how="eq_hist")
        img = tf.dynspread(img, max_px=3)
        img = tf.set_background(img, "black")
        panels[(v1, v2)] = img.to_pil().resize((width, height)).convert("RGBA")

    if not panels:
        info(f"[!] No panels rendered for {method} parameter screen density grid")
        return

    title_h, col_label_h, row_label_w = 22, 20, 95
    n_rows, n_cols = len(axis1_vals), len(axis2_vals)
    total_width = row_label_w + n_cols * width + (n_cols - 1) * padding
    total_height = title_h + col_label_h + n_rows * height + (n_rows - 1) * padding

    composite = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    draw.text((row_label_w, 3), f"{method.upper()} parameter screen  —  "
                                  f"rows: {axis1_name}   columns: {axis2_name}",
              fill="black", font=font)

    for c, v2 in enumerate(axis2_vals):
        x = row_label_w + c * (width + padding)
        draw.text((x + 4, title_h + 3), f"{axis2_name}={v2}", fill="black", font=font)

    for r, v1 in enumerate(axis1_vals):
        y = title_h + col_label_h + r * (height + padding)
        draw.text((4, y + height // 2 - 6), f"{axis1_name}={v1}", fill="black", font=font)
        for c, v2 in enumerate(axis2_vals):
            x = row_label_w + c * (width + padding)
            panel = panels.get((v1, v2))
            if panel is None:
                continue
            composite.paste(panel, (x, y), panel)  # mask=panel: alpha-composite, don't overwrite
            draw.rectangle([x, y, x + width, y + height], outline="black", width=border)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{method}_param_screen_density.png"
    composite.save(out_path)
    info(f"Saved : {out_path}")


PREFERRED_DR_ORDER = ["pca", "trimap", "pacmap", "localmap", "umap", "tsne"]

def _render_feature_composite_from_cache(per_method_feature, feature_col, outdir, mode, width=400, height=400, padding=10, border=2):
    """
    Combine per-method images into a horizontal composite with borders and DR labels.

    Parameters
    ----------
    per_method_feature : dict
        Dictionary of {method: {"categorical": {}, "continuous": {}}}.
    feature_col : str
        Feature to visualize.
    outdir : Path
        Output directory.
    mode : str
        "categorical" or "continuous".
    width, height : int
        Width/height of each subplot.
    padding : int
        Space between subplots.
    border : int
        Width of the border around each subplot.
    """
    images = []
    shared_legend_info = None
    methods_present = [m for m in PREFERRED_DR_ORDER if m in per_method_feature]

    for method in methods_present:
        cached = per_method_feature[method][mode].get(feature_col)
        if cached is not None:
            img, legend_info = cached
            pil_img = img.to_pil().resize((width, height))
            images.append((method, pil_img))
            if legend_info and shared_legend_info is None:
                shared_legend_info = legend_info

    if not images:
        info(f"[!] No images for feature {feature_col}")
        return

    # Calculate composite size
    total_width = sum(img.width for _, img in images) + padding * (len(images) - 1)
    max_height = max(img.height for _, img in images) + 40  # extra space for labels

    composite = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(composite)

    # Optional: load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    x_offset = 0
    for method, img in images:
        # Draw border
        border_box = [x_offset, 30, x_offset + img.width, 30 + img.height]
        draw.rectangle(border_box, outline="black", width=border)

        # Paste image inside border (mask=img: alpha-composite, don't overwrite)
        composite.paste(img, (x_offset, 30), img)

        # Draw method label
        try:
            bbox = draw.textbbox((0, 0), method, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(method)

        text_x = x_offset + (img.width - text_w) // 2
        draw.text((text_x, 5), method, fill="black", font=font)

        x_offset += img.width + padding

    if shared_legend_info:
        if mode == "categorical":
            composite = _draw_legend_onto(composite, shared_legend_info)
        elif mode == "continuous":
            vmin, vmax = shared_legend_info
            composite = _draw_colorbar_onto(composite, vmin, vmax, CET_L17)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"feature_composite_{feature_col}.png"
    composite.save(out_path)
    info(f"Saved : {out_path}")