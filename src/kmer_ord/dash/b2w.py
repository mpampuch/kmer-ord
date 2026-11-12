import sys
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd
import copy
import os
import sqlite3
import threading
import time
import traceback
import plotly.graph_objects as go
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
import numpy as np
from dash import dash_table
import matplotlib.colors
import matplotlib.pyplot as plt
import io
from PIL import Image
import argparse
from math import ceil
from matplotlib.path import Path

from kmer_ord.vis.categorical_palettes import (
    DEFAULT_CATEGORICAL_PALETTE,
    categorical_hex_colors,
)
from kmer_ord.vis.embedding_plots import NA_COLOR, NA_LABEL
from kmer_ord.utils.logging_utils import info, warn

def run_dash_app(db_path: str, output_dir: str, host="127.0.0.1", port=8050):
    global GLOBAL_DB_PATH, output_dir_default
    GLOBAL_DB_PATH = db_path
    output_dir_default = output_dir

    initialise_app_state()

    print_banner(host=host, port=port)
    app.layout = build_layout()

    print_banner(host=host, port=port)

    # Interval requests must be served while export runs on a background thread.
    # Without threaded=True a one-thread server holds the export POST and the
    # status strip stays on "Export started…" until the file write finishes.
    app.run(host=host, port=port, debug=False, threaded=True)

def print_banner(host="127.0.0.1", port=8050):
    title = "BIN2WIN"
    subtitle = "INTERACTIVE BINNER v1.0"

    padding = 4
    inner_width = len(title) + padding * 2
    border = "#" * (inner_width + 2)  # +2 for the side '#'

    line_title = "#{}#".format(
        (" " * padding + title + " " * padding).center(inner_width)
    )

    lines = []
    lines.append("\n" * 2)
    lines.append(border)
    lines.append(line_title)
    lines.append(border)
    lines.append(subtitle.center(len(border)))
    lines.append("")
    lines.append(f"Please view the GUI at http://{host}:{port}/")
    lines.append("")

    banner = "\n".join(lines) + "\n"

    # Force-write and flush in case stdout is buffered
    sys.stdout.write(banner)
    sys.stdout.flush()


# ---- runtime globals (set at runtime) ----
#GLOBAL_DB_PATH = None
#output_dir_default = None

#feature_info = None
#sidebar = None
#nr_reads = None
#nr_ordinations = None
#count_badges = None

CLUSTER_COLS = []   # column names from the clustering table (populated at init)

# Not a features-table column. The dropdown value is a sentinel so density
# can sit beside real features without being interpolated into SQL.
DENSITY_FEATURE = "__density__"
DENSITY_LABEL = "density"

##############################
# UTILS
##############################
def get_connection(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    try:
        conn.execute("SELECT load_extension('mod_spatialite');")
    except sqlite3.OperationalError:
        pass
    return conn

def get_features_columns(db_path):
    conn = get_connection(db_path)
    try:
        rows = conn.execute("PRAGMA table_info(features);").fetchall()
        columns = [row["name"] for row in rows]
    finally:
        conn.close()

    return columns


def parse_feature_types(db_path):
    conn = get_connection(db_path)
    rows = conn.execute("PRAGMA table_info(features)").fetchall()
    feature_info = []

    for r in rows:
        col = r["name"]
        if col == "sequence_id":
            continue
        t = r["type"].upper()

        if any(x in t for x in ("INT", "REAL", "NUM", "FLOAT", "DOUBLE")):
            col_type = "continuous"
        else:
            col_type = "categorical"

        feature_info.append({"column_name": col,
                             "type": col_type})
    conn.close()
    return feature_info


def get_cluster_columns(db_path):
    """Return column names from the clustering table, or [] if it doesn't exist."""
    conn = get_connection(db_path)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='clustering';"
        ).fetchone()
        if not exists:
            return []
        rows = conn.execute("PRAGMA table_info(clustering);").fetchall()
        return [r["name"] for r in rows if r["name"] != "sequence_id"]
    finally:
        conn.close()


def get_available_coordinate_systems(db_path):
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("PRAGMA table_info(coordinates);")
        coord_systems = [
            row["name"]
            for row in cursor.fetchall()
            if row["name"] not in ("sequence_id", "header", "id")
            and not row["name"].lower().startswith("st_")]
    finally:
        conn.close()
    return coord_systems



def _feature_dropdown_options(feature_info, cluster_cols):
    """Color-by choices. Density is first because it is not a table column."""
    options = [{"label": DENSITY_LABEL, "value": DENSITY_FEATURE}]
    options.extend(
        {"label": feature["column_name"], "value": feature["column_name"]}
        for feature in feature_info
    )
    if cluster_cols:
        options.append({
            "label": "── Clustering ──",
            "value": "__cluster_sep__",
            "disabled": True,
        })
        options.extend({"label": column, "value": column} for column in cluster_cols)
    return options


def _feature_panel_title(feature_name):
    if feature_name == DENSITY_FEATURE:
        return DENSITY_LABEL
    return feature_name


def _requested_columns(feature_cols, filter_values=None):
    """Names requested from the features table.

    Density is a shading mode. Dropping the sentinel here keeps it out of
    the SELECT when Color By or feature comparison includes it.
    """
    cols = set()
    if feature_cols:
        if isinstance(feature_cols, str):
            feature_cols = [feature_cols]
        cols.update(column for column in feature_cols if column)
    if filter_values:
        cols.update(filter_values.keys())
    cols.discard(DENSITY_FEATURE)
    return cols


def load_coordinates_from_db(db_path,
                             coordinate_systems,
                             feature_cols=None, 
                             filter_values=None):

    conn = get_connection(db_path)

    try:
        select = ["c.sequence_id AS header"]
        cols = _requested_columns(feature_cols, filter_values)

        for cs in coordinate_systems:
            select.append(f"ST_X(c.{cs}) AS x_{cs}")
            select.append(f"ST_Y(c.{cs}) AS y_{cs}")

        cluster_col_set = set(CLUSTER_COLS) & cols
        feature_col_set = cols - cluster_col_set

        for c in feature_col_set:
            select.append(f'f."{c}" AS "{c}"')
        # Cluster columns are NOT fetched via SQL JOIN — they are merged in Python
        # after the main query to avoid SQLite type-affinity mismatches on sequence_id.

        query = f"""SELECT {', '.join(select)}
        FROM coordinates AS c
        JOIN features AS f ON c.sequence_id = f.sequence_id
        """

        conds = []
        params = []
        if filter_values:
            for col, b in filter_values.items():
                if col in cluster_col_set:
                    continue  # cluster columns not filterable via SQL

                if b.get("min") is not None:
                    conds.append(f"f.{col} >= ?")
                    params.append(b["min"])

                if b.get("max") is not None:
                    conds.append(f"f.{col} <= ?")
                    params.append(b["max"])

                if b.get("cat"):
                    placeholders = ",".join(["?"] * len(b["cat"]))
                    conds.append(f"f.{col} IN ({placeholders})")
                    params.extend(b["cat"])

        if conds:
            query += " WHERE " + " AND ".join(conds)

        df = pd.read_sql_query(query, conn, params=params)

        for cs in coordinate_systems:
            df[f"x_{cs}"] = df[f"x_{cs}"].astype("float32")
            df[f"y_{cs}"] = df[f"y_{cs}"].astype("float32")

        # Merge cluster columns via Python — both sides cast to str to avoid
        # INTEGER vs TEXT mismatches that silently break SQL JOINs.
        if cluster_col_set:
            col_select = ", ".join(
                ['"sequence_id"'] + [f'"{c}"' for c in sorted(cluster_col_set)]
            )
            cluster_df = pd.read_sql_query(
                f"SELECT {col_select} FROM clustering", conn
            )
            cluster_df["sequence_id"] = cluster_df["sequence_id"].astype(str)
            df = df.merge(
                cluster_df,
                left_on="header",
                right_on="sequence_id",
                how="left",
            )
            df = df.drop(columns=["sequence_id"])

    finally:
        conn.close()
    return df


# Export-optimized version of load_coordinates_from_db. ST_X/ST_Y extraction and Python Path.contains_points() replaced with spatialite ST_Within() to test polygon containment directly in SQL that returns only rows that pass filter AND already inside the bin
def efficient_spatialite_export(db_path,
                                coordinate_systems,
                                polygons,
                                feature_col=None,
                                filter_values=None):
    
    conn = get_connection(db_path)

    try:
        cols = set()
        if feature_col:
            cols.add(feature_col)
        if filter_values:
            cols.update(filter_values.keys())

        select = ["c.sequence_id AS header"]
        for c in sorted(cols):
            select.append(f"f.{c} AS {c}")

        query = f"""
        SELECT {', '.join(select)}
        FROM coordinates AS c
        JOIN features AS f
          ON c.sequence_id = f.sequence_id
        """

        conds = []
        params = []

        if filter_values:
            for col, b in filter_values.items():
                if b.get("min") is not None:
                    conds.append(f"f.{col} >= ?")
                    params.append(b["min"])

                if b.get("max") is not None:
                    conds.append(f"f.{col} <= ?")
                    params.append(b["max"])

                if b.get("cat"):
                    placeholders = ",".join(["?"] * len(b["cat"]))
                    conds.append(f"f.{col} IN ({placeholders})")
                    params.extend(b["cat"])

        # spatial filter using geometry column directly
        cs = coordinate_systems[0]
        pts = polygons[cs]
        
        # ensure polygon ring is closed
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        
        coord_text = ", ".join(f"{float(x)} {float(y)}" for x, y in pts)
        poly_wkt = f"POLYGON(({coord_text}))"
        
        conds.append(
            f"ST_Within(c.{cs}, ST_GeomFromText(?, ST_SRID(c.{cs})))"
        )
        params.append(poly_wkt)
        
        if conds:
            query += " WHERE " + " AND ".join(conds)

        df = pd.read_sql_query(query, conn, params=params)

    finally:
        conn.close()

    return df


def get_number_of_reads(db_path):
    conn = get_connection(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) AS cnt FROM features;").fetchone()["cnt"]
    finally:
        conn.close()
    return count


def get_number_of_ordinations(db_path):
    return len(get_available_coordinate_systems(db_path))

#---- plotting utils ----
_CLUSTER_NOISE_COLOR = "#555555"
_CLUSTER_CATEGORICAL_THRESHOLD = 20


def _cluster_color_key(categories):
    """
    Build a datashader color_key for integer cluster label columns.

    - label -1 (noise/unassigned) → fixed grey
    - ≤20 non-noise labels → glasbey qualitative palette (light green → orange)
    - >20 non-noise labels → Turbo sequential colormap for max perceptual separation

    Returns (color_key_dict, n_clusters, mode)
    where mode is "categorical" or "turbo".
    """
    regular = sorted(c for c in categories if c != -1)
    n = len(regular)

    if n <= _CLUSTER_CATEGORICAL_THRESHOLD:
        colors = categorical_hex_colors(n, DEFAULT_CATEGORICAL_PALETTE)
        mode = "categorical"
    else:
        # plt.get_cmap (here and below) instead of cm.get_cmap, which was
        # removed in Matplotlib 3.9
        turbo = plt.get_cmap("turbo")
        colors = [
            matplotlib.colors.rgb2hex(turbo(i / max(n - 1, 1)))
            for i in range(n)
        ]
        mode = "turbo"

    key = dict(zip(regular, colors))
    if -1 in categories:
        key[-1] = _CLUSTER_NOISE_COLOR
    return key, n, mode


# Tiara writes "n/a" when the second stage made no call. Pandas' default NA
# parsing usually turns that into null on read; a literal that survived is
# the same missing label. "na" is the lowercased form of both "n/a" and "NA".
_NO_CALL_TOKENS = {"n/a", "na"}


def _categorical_with_missing(series):
    """Real labels, with null, blank, and Tiara n/a tokens replaced by NA.

    Datashader's count_cat drops nulls. rDNA only labels reads that hit, and
    Tiara's second stage only labels organelles, so those columns are mostly
    missing. Leaving the nulls out makes the embedding look like it contains
    only the called reads. The NA label is the one the static embedding plots
    already paint grey.
    """
    text = series.astype("string").str.strip()
    missing = text.isna() | text.eq("") | text.str.lower().isin(_NO_CALL_TOKENS)
    return text.mask(missing, NA_LABEL).astype("category")


def _categorical_color_key(categories, color_palette):
    """Palette for the called labels. NA stays grey and does not take a slot.

    A sparse column is almost entirely NA. Counting that label toward the
    Category10 cutoff would push a 10-label feature onto glasbey and hand NA
    a qualitative color of its own.
    """
    real = [category for category in categories if category != NA_LABEL]
    # len(real), not len(categories): NA must not consume a Category10 slot.
    colors = categorical_hex_colors(len(real), color_palette)
    key = dict(zip(real, colors))
    if NA_LABEL in categories:
        key[NA_LABEL] = NA_COLOR
    return key


def _categorical_legend_entries(series, color_palette):
    """(label, hex) pairs in category order, with NA last when it is present."""
    labeled = _categorical_with_missing(series)
    real = [category for category in labeled.cat.categories if category != NA_LABEL]
    categories = [*real, NA_LABEL] if NA_LABEL in labeled.cat.categories else real
    key = _categorical_color_key(categories, color_palette)
    return [(category, key[category]) for category in categories]


def create_datashader_image(df, x_col, y_col,
                            color_col=None, color_palette="viridis",
                            px_spread=3,
                            plot_width=1500, plot_height=1500,
                            cluster_cols=None):
    df_ds = pd.DataFrame({
        'x': df[x_col].to_numpy(dtype='float32'),
        'y': df[y_col].to_numpy(dtype='float32')
    })

    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

    # The density sentinel is a plot mode, not a column, even if a frame
    # happens to contain a series with that name.
    if color_col not in (None, DENSITY_FEATURE) and color_col in df:
        is_cluster = bool(cluster_cols and color_col in cluster_cols)

        if is_cluster:
            labels = df[color_col].fillna(-1).astype(int)
            cat = pd.Categorical(labels)
            df_ds['cat_value'] = cat
            key, _n, _mode = _cluster_color_key(list(cat.categories))
            agg = cvs.points(df_ds, x='x', y='y', agg=ds.count_cat('cat_value'))
            img = tf.shade(agg, color_key=key, how='eq_hist')

        elif pd.api.types.is_numeric_dtype(df[color_col]):
            df_ds['val'] = df[color_col].to_numpy(dtype='float32')
            agg = cvs.points(df_ds, x='x', y='y', agg=ds.mean('val'))
            
            if color_palette in plt.colormaps():
                cmap = plt.get_cmap(color_palette)
            else:
                cmap = plt.get_cmap("viridis")
            cmap_list = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
            
            img = tf.shade(agg, cmap=cmap_list, how='linear')
        
        else:
            # count_cat drops nulls. Assign the labels by position: df_ds has
            # a fresh RangeIndex, and aligning the labeled Series on df's
            # index would turn every label into null again.
            labeled = _categorical_with_missing(df[color_col])
            df_ds['cat_value'] = pd.Categorical(
                labeled.to_numpy(),
                categories=list(labeled.cat.categories),
            )
            key = _categorical_color_key(list(labeled.cat.categories), color_palette)
            agg = cvs.points(df_ds, x='x', y='y', agg=ds.count_cat('cat_value'))
            # min_alpha=255 matches the static embedding plots. NA is usually
            # most of the points, and the default alpha scaling draws that
            # majority nearly transparent on a dark canvas.
            img = tf.shade(
                agg, color_key=key, how='eq_hist', alpha=255, min_alpha=255,
            )
    
    else:
        # No feature, or the density option: local point counts. fire + eq_hist
        # matches the static embedding density plots and the legend under the plot.
        agg = cvs.points(df_ds, x='x', y='y', agg=ds.count())
        img = tf.shade(agg, cmap=fire, how='eq_hist')
    
    img = tf.spread(img, px=max(int(px_spread), 1))
    return img.to_pil()


# eq_hist ranks bins inside one panel, so the bar is relative density, not a count.
DENSITY_LEGEND_CAPTION = (
    "Color map: point density (Datashader 'fire' colormap, eq_hist). "
    "Brighter = higher local density."
)


def _fire_css_gradient(stops=8):
    """Left-to-right CSS gradient sampled from colorcet.fire."""
    last = len(fire) - 1
    colors = [fire[round(i * last / (stops - 1))] for i in range(stops)]
    return "linear-gradient(to right, " + ", ".join(colors) + ")"


def _density_legend(compact=True):
    """Fire scale for density shading.

    compact is one row (labels plus bar) so it fits the 40px feature-comparison
    slot. The shared DR-comparison legend adds the eq_hist caption.
    """
    label_style = {
        "color": "#ccc",
        "fontSize": "0.65rem",
        "lineHeight": "10px",
        "flex": "0 0 auto",
    }
    # fire[0] is black, which disappears on the dark binner background.
    row = html.Div(
        [
            html.Span("low", style=label_style),
            html.Div(
                style={
                    "height": "10px",
                    "width": "160px",
                    "background": _fire_css_gradient(),
                    "border": "1px solid #aaa",
                    "borderRadius": "3px",
                    "flex": "0 0 auto",
                }
            ),
            html.Span("high", style=label_style),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "gap": "6px",
            "width": "100%",
            "maxHeight": "14px",
        },
    )
    if compact:
        return row
    return html.Div(
        [
            row,
            html.Div(
                DENSITY_LEGEND_CAPTION,
                style={
                    "color": "white",
                    "fontSize": "0.75rem",
                    "marginTop": "4px",
                },
            ),
        ],
        style={"color": "white", "textAlign": "center"},
    )


# Small per-panel legend for feature comparison mode.
def create_panel_legend(df, feature_name, continuous_palette, categorical_palette):
    """
    Return:
        (legend_component, legend_type)
    where legend_type is "continuous" or "categorical".
    Must always return exactly 2 values.
    """

    if (
        not feature_name
        or feature_name == DENSITY_FEATURE
        or feature_name not in df.columns
    ):
        return _density_legend(compact=True), "continuous"

    # Cluster column → adaptive legend
    if feature_name in CLUSTER_COLS:
        labels = df[feature_name].fillna(-1).astype(int)
        categories = sorted(labels.unique().tolist())
        key, n_clusters, mode = _cluster_color_key(categories)
        noise_present = -1 in categories

        if mode == "categorical":
            items = []
            for cat_val in categories:
                color = key[cat_val]
                label = "noise" if cat_val == -1 else str(cat_val)
                items.append(
                    html.Div([
                        html.Span(style={
                            "display": "inline-block", "width": "10px", "height": "10px",
                            "marginRight": "5px", "backgroundColor": color,
                            "border": "1px solid #444", "verticalAlign": "middle",
                        }),
                        html.Span(label, style={"verticalAlign": "middle", "fontSize": "0.72rem"})
                    ], style={"display": "inline-block", "marginRight": "10px", "marginBottom": "3px"})
                )
            noise_note = " (grey = noise)" if noise_present else ""
            items.insert(0, html.Div(
                f"{feature_name} — {n_clusters} clusters{noise_note}",
                style={"color": "#aaa", "fontSize": "0.7rem", "marginBottom": "4px", "width": "100%"}
            ))
            return (
                html.Div(items, style={"color": "white", "textAlign": "center",
                                       "fontSize": "0.75rem"}),
                "categorical"
            )
        else:
            noise_note = "  ·  grey = noise/unassigned" if noise_present else ""
            return (
                html.Div([
                    html.Div(
                        f"{feature_name} — {n_clusters} clusters (Turbo scale){noise_note}",
                        style={"color": "#aaa", "fontSize": "0.7rem", "marginBottom": "4px"}
                    ),
                    html.Div(style={
                        "height": "10px", "width": "160px", "margin": "0 auto",
                        "background": "linear-gradient(to right, #30123b, #4777ef, #1ac7c2, "
                                      "#a8fc3d, #fb8022, #7a0403)",
                        "borderRadius": "3px",
                    }),
                    html.Div(
                        f"0 → {n_clusters - 1}",
                        style={"color": "#666", "fontSize": "0.65rem", "marginTop": "2px"}
                    ),
                ], style={"color": "white", "textAlign": "center", "fontSize": "0.75rem"}),
                "categorical"
            )

    # Continuous feature -> small horizontal colorbar
    if pd.api.types.is_numeric_dtype(df[feature_name]):
        if not df[feature_name].notna().any():
            return (
                html.Div(
                    f"{feature_name}: no values",
                    style={"color": "white", "fontSize": "0.75rem", "textAlign": "center"}
                ),
                "continuous"
            )

        vmin = float(df[feature_name].min())
        vmax = float(df[feature_name].max())

        if vmin == vmax:
            return (
                html.Div(
                    f"{feature_name}: constant ({vmin})",
                    style={"color": "white", "fontSize": "0.75rem", "textAlign": "center"}
                ),
                "continuous"
            )

        cmap_name = continuous_palette if continuous_palette in plt.colormaps() else "viridis"
        cmap = plt.get_cmap(cmap_name)

        colorscale = []
        for i in range(256):
            frac = i / 255.0
            r, g, b, _ = cmap(frac)
            colorscale.append([
                frac,
                f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            ])

        legend_trace = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='markers',
            marker=dict(
                size=0.0001,
                color=[vmin, vmax],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(text=feature_name, font=dict(color='white', size=9)),
                    tickfont=dict(color='white', size=8),
                    orientation='h',
                    x=0.5,
                    xanchor='center',
                    thickness=10,
                    len=0.9,
                ),
            ),
            hoverinfo='none',
            showlegend=False
        )

        legend_fig = go.Figure(data=[legend_trace])
        legend_fig.update_xaxes(visible=False)
        legend_fig.update_yaxes(visible=False)
        legend_fig.update_layout(
            margin=dict(l=20, r=20, t=8, b=8),
            height=95,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        return (
            dcc.Graph(
                figure=legend_fig,
                config={"displayModeBar": False},
                style={"height": "95px", "width": "100%"}
            ),
            "continuous"
        )

    # Categorical feature -> compact HTML legend. Same NA mapping as the image,
    # so a missing rDNA or second-stage call is a grey swatch rather than absent.
    entries = _categorical_legend_entries(df[feature_name], categorical_palette)

    if not entries:
        return (
            html.Div(
                f"{feature_name}: no categories",
                style={"color": "white", "fontSize": "0.75rem", "textAlign": "center"}
            ),
            "categorical"
        )

    items = []
    for cat, color in entries:
        items.append(
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "10px",
                            "height": "10px",
                            "marginRight": "5px",
                            "backgroundColor": color,
                            "border": "1px solid #ccc",
                            "verticalAlign": "middle"
                        }
                    ),
                    html.Span(
                        str(cat),
                        style={"verticalAlign": "middle", "fontSize": "0.72rem"}
                    )
                ],
                style={
                    "display": "inline-block",
                    "marginRight": "10px",
                    "marginBottom": "3px"
                }
            )
        )

    return (
        html.Div(
            items,
            style={
                "color": "white",
                "textAlign": "center",
                "paddingTop": "0px",
                "fontSize": "0.75rem"
            }
        ),
        "categorical"
    )

# supporting clean plots button 
def figure_to_serializable(fig):
    return fig.to_dict()


def build_plot_grid_from_serialized(serialized_plots):
    row_comps = []

    max_cols = max(len(row) for row in serialized_plots) if serialized_plots else 1

    for row in serialized_plots:
        cols = []
        col_width = max(1, int(12 / max_cols))

        for item in row:
            cs = item["coordinate_system"]
            panel_id = item["panel_id"]          
            panel_title = item["panel_title"]    
            fig_dict = item["figure"]

            graph_container = html.Div(
                dcc.Graph(
                    id={                         # keep same full id shape so pattern-matching callbacks can see restored plots
                        "type": "scatter-plot",
                        "index": panel_id,
                        "coordinate_system": cs
                    },
                    figure=fig_dict,
                    config={"responsive": True},
                    style={"height": "100%", "width": "100%"}
                ),
                style={
                    "width": "100%",
                    "height": "100%",
                    "minHeight": 0,
                    "overflow": "hidden",
                    "border": "1px solid #444"
                }
            )

            cols.append(
                dbc.Col(
                    [
                        html.H5(
                            # cs,
                            panel_title,   #  panel title = feature name
                            style={
                                "color": "white",
                                "textAlign": "center",
                                "marginBottom": "6px",
                                "flex": "0 0 auto",
                            }
                        ),
                        html.Div(
                            graph_container,
                            style={
                                "flex": "1 1 auto",
                                "minHeight": 0,
                                "display": "flex",
                                "flexDirection": "column",
                            }
                        )
                    ],
                    width=col_width,
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": 0,
                        "height": "100%",
                    }
                )
            )

        row_comps.append(
            dbc.Row(
                cols,
                justify="center",
                #className="mb-2",
                style={
                    "flex": "1 1 0",
                    "minHeight": 0,
                }
            )
        )

    return html.Div(
        row_comps,
        style={
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "gap": "4px",
            "minHeight": 0,
        }
    )

##############################
# BIN LIST COMPONENT
##############################

def _status_bar(msg: str):
    """Return a slim single-line status strip, or None if msg is empty.

    Finished exports stay green. In-progress export lines are blue so a long
    job does not look like an error until it finishes or fails.
    """
    if not msg:
        return None
    msg_lower = msg.lower()
    in_progress = (
        msg_lower.startswith("export started")
        or "spatial query" in msg_lower
        or "wrote csv" in msg_lower
        or "fasta chunk" in msg_lower
        or msg_lower.startswith("bin ")
    )
    if "fail" in msg_lower or "already running" in msg_lower:
        accent, text_col = "#7b3f00", "#ffcc80"   # amber
    elif "successfully" in msg_lower or "exported" in msg_lower:
        accent, text_col = "#2e7d32", "#a5d6a7"   # green
    elif ("found" in msg_lower and "0 points" not in msg_lower) or in_progress:
        accent, text_col = "#1565c0", "#90caf9"   # blue
    else:
        accent, text_col = "#7b3f00", "#ffcc80"   # amber

    return html.Div(
        msg,
        style={
            "borderLeft": f"3px solid {accent}",
            "backgroundColor": "#1e1e1e",
            "color": text_col,
            "fontSize": "0.78rem",
            "padding": "4px 10px",
            "borderRadius": "3px",
            "lineHeight": "1.4",
        }
    )


def _format_bin_filters(fv):
    """Return a compact inline string for all active filters, or empty string."""
    parts = []
    for col, bounds in (fv or {}).items():
        lo   = bounds.get("min")
        hi   = bounds.get("max")
        cats = [v for v in (bounds.get("cat") or []) if v is not None]
        if lo is not None or hi is not None:
            lo_s = f"{lo:g}" if lo is not None else "…"
            hi_s = f"{hi:g}" if hi is not None else "…"
            parts.append(f"{col}: {lo_s}–{hi_s}")
        if cats:
            cat_s = ", ".join(str(v) for v in cats[:2])
            if len(cats) > 2:
                cat_s += f" +{len(cats) - 2}"
            parts.append(f"{col}: {cat_s}")
    return "  ·  ".join(parts)


def _build_bin_list(bins_data):
    """Render a compact single-line list of bins for the main panel."""
    if not bins_data:
        return None

    _strip = {
        "display": "flex",
        "alignItems": "center",
        "backgroundColor": "#252525",
        "border": "1px solid #333",
        "borderRadius": "5px",
        "marginBottom": "4px",
        "padding": "5px 10px",
        "gap": "10px",
        "minHeight": "0",
    }
    _sep = {"color": "#444", "fontSize": "0.75rem", "flexShrink": "0"}

    header = html.Div(
        "Bins",
        style={
            "fontSize": "0.7rem",
            "textTransform": "uppercase",
            "letterSpacing": "1px",
            "color": "#555",
            "marginBottom": "6px",
        }
    )

    rows = [header]
    for idx, b in enumerate(bins_data):
        n_pts       = len(b.get("polygon", []))
        filter_str  = _format_bin_filters(b.get("filters", {}))

        label_parts = [
            html.Span(
                b["bin_name"],
                style={"fontWeight": "600", "fontSize": "0.82rem", "color": "#ddd", "whiteSpace": "nowrap"}
            ),
            html.Span("·", style=_sep),
            html.Span(
                b["coordinate_system"],
                style={"fontSize": "0.78rem", "color": "#888", "whiteSpace": "nowrap"}
            ),
            html.Span("·", style=_sep),
            html.Span(
                f"{n_pts} pts",
                style={"fontSize": "0.78rem", "color": "#666", "whiteSpace": "nowrap"}
            ),
        ]
        if filter_str:
            label_parts += [
                html.Span("·", style=_sep),
                html.Span(
                    filter_str,
                    style={"fontSize": "0.75rem", "color": "#555", "whiteSpace": "nowrap",
                           "overflow": "hidden", "textOverflow": "ellipsis"}
                ),
            ]

        rows.append(
            html.Div(
                [
                    html.Div(label_parts, style={"display": "flex", "alignItems": "center",
                                                 "gap": "8px", "flex": "1", "minWidth": "0",
                                                 "overflow": "hidden"}),
                    html.Button(
                        "×",
                        id={"type": "remove-bin-btn", "index": idx},
                        n_clicks=0,
                        style={
                            "background": "none",
                            "border": "none",
                            "color": "#555",
                            "fontSize": "1rem",
                            "lineHeight": "1",
                            "padding": "0 2px",
                            "cursor": "pointer",
                            "flexShrink": "0",
                        }
                    ),
                ],
                style=_strip,
            )
        )

    return html.Div(rows)


##############################
# BUILD SIDEBAR + LAYOUT
##############################
##############################
# BUILD SIDEBAR
##############################
def build_dynamic_sidebar(feature_info):
    coord_opts = get_available_coordinate_systems(GLOBAL_DB_PATH)

    section_title_style = {
        "fontSize": "0.7rem",
        "textTransform": "uppercase",
        "letterSpacing": "1px",
        "opacity": 0.5,
        "marginBottom": "0.6rem",
    }

    input_style = {
        "backgroundColor": "#252525",
        "background": "#383B3E",
        "border": "1px solid #5f5f5f",
        "color": "#e5e5e5",
        "fontSize": "0.8rem",
    }

    button_style = {
        "backgroundColor": "#222",
        "border": "1px solid #333",
        "color": "#ddd",
        "fontSize": "0.75rem",
        "padding": "6px 10px",
    }

    children = []

    # PLOT CONTROLS
    children += [
        html.Div("Plot Controls", style=section_title_style),

        #  mode toggle
        dbc.Label("Plot Mode", className="small"),
        dcc.RadioItems(
            id="plot-mode-toggle",
            options=[
                {"label": "DR comparison", "value": "dr-comparison"},
                {"label": "Feature comparison", "value": "feature-comparison"},
            ],
            value="dr-comparison",
            labelStyle={"display": "block", "marginBottom": "4px"},
            inputStyle={"marginRight": "6px"},
            style={"fontSize": "0.8rem", "marginBottom": "0.75rem"},
        ),

        dbc.Label("Coordinate Systems", className="small"),
        dcc.Dropdown(
            id="coordinate-systems-checklist",
            options=[{"label": cs, "value": cs} for cs in coord_opts],
            multi=True,
            searchable=False,
            placeholder="Select DR methods",
        ),

        html.Div([   # wrap so we can hide/show by mode
            dbc.Label("Color By", className="small mt-3"),
            dcc.Dropdown(
                id="color-feature-dropdown",
                options=_feature_dropdown_options(feature_info, CLUSTER_COLS),
                placeholder="Select feature",
                searchable=False,
            ),
        ], id="dr-color-feature-wrapper"),

        # feature comparison selector
        html.Div([
            dbc.Label("Features to Compare", className="small mt-3"),
            dcc.Dropdown(
                id="feature-comparison-dropdown",
                options=_feature_dropdown_options(feature_info, CLUSTER_COLS),
                multi=True,
                placeholder="Select features",
                searchable=False,
            ),
        ], id="feature-comparison-wrapper", style={"display": "none"}),


        # separate palette selectors for numeric vs categorical features
        dbc.Label("Continuous Palette", className="small mt-3"),
        dcc.Dropdown(
            id="continuous-palette-dropdown",
            options=[
                {"label": "viridis", "value": "viridis"},
                {"label": "plasma", "value": "plasma"},
                {"label": "inferno", "value": "inferno"},
            ],
            value="viridis",
            searchable=False,
        ),
        
        dbc.Label("Categorical Palette", className="small mt-3"),
        dcc.Dropdown(
            id="categorical-palette-dropdown",
            options=[
                {"label": "glasbey", "value": "glasbey"},
                {"label": "distinct", "value": "distinct"},
                {"label": "Category10", "value": "Category10"},
            ],
            value=DEFAULT_CATEGORICAL_PALETTE,
            searchable=False,
        ),

        dbc.Label("Pixel Spread", className="small mt-3"),
        dcc.Input(
            id="px-spread-input",
            type="number",
            min=1,
            step=1,
            value=3,
            style={**input_style, "width": "100%"},
        ),

        dbc.Button(
            "Update Plots",
            id="update-plots-button",
            className="w-100 mt-3",
            style=button_style,
        ),
    ]

    # ACTIONS
    children += [
        html.Div("Actions", style=section_title_style),

        dbc.Button(
            "Export Bins",
            id="export-bins-button",
            className="w-100 mb-2",
            style=button_style,
        ),
    ]

    filter_children = []

    for feat in feature_info:
        nm, tp = feat['column_name'], feat['type']

        if tp == "continuous":
            filter_children.append(
                html.Div([
                    dbc.Label(nm, className="small"),
                    dbc.Row([
                        dbc.Col(
                            dcc.Input(
                                id={"type": "continuous-filter-min", "column_name": nm},
                                type="number",
                                placeholder="Min",
                                style={**input_style, "width": "100%"},
                            ),
                            width=6
                        ),
                        dbc.Col(
                            dcc.Input(
                                id={"type": "continuous-filter-max", "column_name": nm},
                                type="number",
                                placeholder="Max",
                                style={**input_style, "width": "100%"},
                            ),
                            width=6
                        ),
                    ], className="g-2"),
                ], className="mb-3")
            )

        else:
            filter_children.append(
                html.Div([
                    dbc.Label(nm, className="small"),
                    dcc.Dropdown(
                        id={"type": "cat-checklist", "column_name": nm},
                        multi=True,
                        placeholder="Select values",
                        style=input_style,
                    ),
                ], className="mb-3")
            )

    children += [
        html.Div("Filters (Advanced)", style=section_title_style),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    filter_children,
                    title="Show Filters"
                )
            ],
            start_collapsed=True,
            flush=True,
        ),
    ]

    sidebar_style = {
        "width": "18rem",
        "padding": "1.5rem 1rem",
        "overflowY": "auto",
        "backgroundColor": "#32383E",
        "borderRight": "1px solid #181818"}

    return html.Div(children, style=sidebar_style, className="sidebar")

##############################
# APP INITIALIZATION
##############################

external_stylesheets = [
    dbc.themes.SLATE,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def initialise_app_state():
    global feature_info, sidebar, nr_reads, nr_ordinations, count_badges, CLUSTER_COLS

    feature_info = parse_feature_types(GLOBAL_DB_PATH)
    CLUSTER_COLS = get_cluster_columns(GLOBAL_DB_PATH)
    sidebar = build_dynamic_sidebar(feature_info)
    nr_reads = get_number_of_reads(GLOBAL_DB_PATH)
    nr_ordinations = get_number_of_ordinations(GLOBAL_DB_PATH)
    count_badges = make_count_badges(GLOBAL_DB_PATH)


store_bins = dcc.Store(id='bins-store', data=[])
store_overlay = dcc.Store(id='overlay-store', data=[])
store_base_plots = dcc.Store(id='base-plots-store', data=None)   

##############################
# LAYOUT
##############################


def make_count_badges(db_path):
    conn = sqlite3.connect(db_path)
    try:
        nr_reads = conn.execute("SELECT COUNT(*) FROM features;").fetchone()[0]
        all_coords = conn.execute("PRAGMA table_info(coordinates);").fetchall()
        nr_ordinations = sum(
            1 for c in all_coords
            if c[1] not in ("sequence_id", "header")
            and not c[1].lower().startswith("st_"))

        # Feature types
        df_sample = pd.read_sql_query("SELECT * FROM features LIMIT 5;", conn)

        cat_count = 0
        num_count = 0

        for col in df_sample.columns:
            if col == "sequence_id":
                continue
            try:
                pd.to_numeric(df_sample[col].dropna().astype(str))
                num_count += 1
            except ValueError:
                cat_count += 1

    finally:
        conn.close()

    badge_style = {"background": "#32383E", "border": "1px solid #2f2f2f",
                   "color": "#D2D2D2", "fontSize": "0.75rem", "padding": "6px 10px", "borderRadius": "14px"}

    return html.Div(
        [
            dbc.Badge(f"Reads: {nr_reads:,}",color=None, style=badge_style, className="me-2"),
            dbc.Badge(f"Ordinations: {nr_ordinations}",color=None, style=badge_style, className="me-2"),
            dbc.Badge(f"Numerical: {num_count}",color=None, style=badge_style, className="me-2"),
            dbc.Badge(f"Categorical: {cat_count}",color=None, style=badge_style,),
        ],
        style={
            #"marginBottom": "1rem",
            "paddingTop": "1rem",
            "paddingLeft": "1rem",
            #"marginLeft": "18rem",  # align with main content
        }
    )



##############################
# APP LAYOUT
##############################
def build_layout():
    BANNER_HEIGHT = "90px"
    SIDEBAR_WIDTH = "18rem"

    # -----------------------
    # Banner (fixed top)
    # -----------------------
    banner = html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div([
                                    html.H2(
                                        "b2w",
                                        style={"margin": 0, "fontWeight": "600", "letterSpacing": "1px"},
                                    ),
                                    html.Div(
                                        "Interactive binning",
                                        style={"fontSize": "0.85rem", "opacity": 0.7},
                                    ),
                                ])
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div(
                                            "kmer-ord",
                                            style={
                                                "textAlign": "right",
                                                "fontSize": "0.8rem",
                                                "opacity": 0.6,
                                            },
                                        ),
                                        html.Div(
                                            [
                                                dbc.Badge(
                                                    [html.I(className="bi bi-github me-1"),
                                                    "FDBoever/kmer-ord"],
                                                    href="https://github.com/FDBoever/kmer-ord",
                                                    target="_blank",
                                                    #color="secondary",
                                                    #className="me-2",
                                                    style={"fontSize": "0.7rem",
                                                        "background": "rgb(69, 95, 177)",
                                                        "color": "#fff",
                                                        "padding": "6px 7px",
                                                        "borderRadius": "4px"},
                                                ),
                                            ],
                                            style={"textAlign": "right", "marginTop": "6px"},
                                        ),
                                    ]
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                        style={"height": BANNER_HEIGHT},
                    )
                ],
                fluid=True
            )
        ],
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "right": 0,
            "height": BANNER_HEIGHT,
            "background": "linear-gradient(90deg, #111111, #1c1c1c)",
            "zIndex": 1000,
            "borderBottom": "1px solid #2a2a2a",
        },
    )

    sidebar_style = {
        "flex": f"0 0 {SIDEBAR_WIDTH}",
        "maxWidth": SIDEBAR_WIDTH,
        "overflowY": "auto",
        "backgroundColor": "#1c1c1c"
    }

    main_content_style = {
        "flex": "1 1 auto",
        "overflow": "hidden",
        "display": "flex",
        "paddingRight": "2rem",
        "flexDirection": "column",
    }

    main_content = html.Div(
        [
            count_badges,
            dbc.Card(
                [
                    dbc.CardHeader("Visualise and create Bins"),
                    dbc.CardBody(
                        [   
                            # --- Resizable plots container: plots shrink/grow to fit box height ---
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                id='coordinates-plots-container',
                                                style={
                                                    "flex": "1 1 auto",
                                                    "minHeight": 0,
                                                    "display": "flex",
                                                    "flexDirection": "column",
                                                }
                                            ),
                                            html.Div(
                                                id='color-legend-container',
                                                style={
                                                    "textAlign": "center",
                                                    "marginTop": "10px",
                                                    "marginBottom": "20px",
                                                    "flex": "0 0 auto",
                                                }
                                            ),
                                        ],
                                        style={
                                            "height": "100%",
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "minHeight": 0,
                                        }
                                    )
                                ],
                                style={
                                    "height": "70vh",
                                    "minHeight": "450px",
                                    "resize": "vertical",
                                    "overflow": "hidden",
                                    "border": "1px solid #333",
                                    "padding": "8px",
                                    "flex": "0 0 auto",
                                }
                            ),
                            html.Div(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(
                                                "Bin name",
                                                style={
                                                    "backgroundColor": "#1e1e1e",
                                                    "border": "1px solid #3a3a3a",
                                                    "color": "#777",
                                                    "fontSize": "0.75rem",
                                                    "padding": "4px 10px",
                                                }
                                            ),
                                            dbc.Input(
                                                id='bin-name-input',
                                                type='text',
                                                placeholder='Enter bin name',
                                                style={
                                                    "backgroundColor": "#252525",
                                                    "border": "1px solid #3a3a3a",
                                                    "borderLeft": "none",
                                                    "color": "#ddd",
                                                    "fontSize": "0.78rem",
                                                }
                                            ),
                                        ],
                                        size="sm",
                                        style={"width": "260px", "flexShrink": "0"},
                                    ),
                                    html.Div(style={"width": "1px", "backgroundColor": "#2e2e2e", "alignSelf": "stretch"}),
                                    html.Div(
                                        [
                                            *[
                                                dbc.Button(
                                                    label,
                                                    id=btn_id,
                                                    size="sm",
                                                    style={
                                                        "backgroundColor": "#1e1e1e",
                                                        "border": "1px solid #3a3a3a",
                                                        "color": "#888",
                                                        "fontSize": "0.75rem",
                                                        "padding": "4px 14px",
                                                        "whiteSpace": "nowrap",
                                                    }
                                                )
                                                for label, btn_id in [
                                                    ("Create Bin",     "create-bin-button"),
                                                    ("Inspect Bin",    "inspect-bin-button"),
                                                    ("Overlay Points", "overlay-points-button"),
                                                    ("Clear Plots",    "clear-plots-button"),
                                                ]
                                            ],
                                        ],
                                        style={
                                            "display": "flex",
                                            "gap": "6px",
                                            "alignItems": "center",
                                            "flexWrap": "wrap",
                                        }
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "14px",
                                    "marginBottom": "10px",
                                    "flexWrap": "wrap",
                                }
                            ),
                            html.Div(id='error-message', className="mb-2"),
                            html.Div(id='bin-list-container', className="mb-3"),
                            html.Div(id='bin-table-container', className="mt-3")
                        ]
                    )
                ],
                className="m-3 w-100"
            )
        ],
        style=main_content_style
    )


    footer = html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col("2026 Green team / SAMS", width=6,
                            style={"fontSize": "0.8rem", "opacity": 0.6}),
                    dbc.Col("Version 1.0", width=6,
                            style={"textAlign": "right", "fontSize": "0.8rem", "opacity": 0.6}),
                ]
            ),
            fluid=True
        ),
        style={
            "padding": "10px 10px",
            "borderTop": "1px solid #2a2a2a",
            "backgroundColor": "#111111",
            "width": "100%",
        }
    )

    # -----------------------
    layout = html.Div(
        [
            banner,

            html.Div(
                [
                    # Sidebar + main content row
                    html.Div([html.Div(sidebar, style=sidebar_style),
                            html.Div(main_content, style=main_content_style)],
                        style={"display": "flex",
                            "marginTop": BANNER_HEIGHT,
                            "minHeight": "calc(100vh - " + BANNER_HEIGHT + ")"}
                    ),
                    footer,
                ],
                className="dbc",
                style={"display": "flex", "flexDirection": "column"}
            ),
            store_bins,
            store_overlay,
            store_base_plots, #NEW
            # Disabled until Export is clicked. Polls the in-process progress
            # dict so the status strip can move while the export thread runs.
            dcc.Interval(
                id="export-progress-interval",
                interval=1000,
                disabled=True,
            ),
        ]
    )
    return layout



##############################
# CATEGORICAL OPTIONS CALLBACK
##############################
@app.callback(
    Output({'type':'cat-checklist','column_name':MATCH}, 'options'),
    Input({'type':'cat-checklist','column_name':MATCH}, 'id')
)
def populate_categorical_options(matching_id):
    col = matching_id["column_name"]
    conn = sqlite3.connect(GLOBAL_DB_PATH)
    try:
        res = conn.execute(f"SELECT DISTINCT {col} FROM features "
                           f"ORDER BY {col} LIMIT 200;").fetchall()
        unique_vals = [r[0] for r in res if r[0] is not None]
    finally:
        conn.close()
    return [{"label":str(v),"value":v} for v in unique_vals]


##############################
# MAIN PLOT CALLBACK (RESPONSIVE SIZING)
##############################
@app.callback(
    Output('coordinates-plots-container', 'children'),
    Output('color-legend-container', 'children'),
    Output('base-plots-store', 'data'),
    Input('update-plots-button', 'n_clicks'),
    Input('overlay-store', 'data'),
    State('plot-mode-toggle', 'value'),
    State('coordinate-systems-checklist', 'value'),
    State('color-feature-dropdown', 'value'),
    State('feature-comparison-dropdown', 'value'),
    State('continuous-palette-dropdown', 'value'),
    State('categorical-palette-dropdown', 'value'),
    State({'type': 'continuous-filter-min', 'column_name': ALL}, 'value'),
    State({'type': 'continuous-filter-min', 'column_name': ALL}, 'id'),
    State({'type': 'continuous-filter-max', 'column_name': ALL}, 'value'),
    State({'type': 'continuous-filter-max', 'column_name': ALL}, 'id'),
    State({'type': 'cat-checklist', 'column_name': ALL}, 'value'),
    State({'type': 'cat-checklist', 'column_name': ALL}, 'id'),
    State('px-spread-input', 'value'),
    prevent_initial_call=True
)
def update_multiple_coord_plots(
    _btn, overlay_headers,
    plot_mode,
    selected_coords, selected_feature, selected_features,
    continuous_palette, categorical_palette,
    all_min_vals, all_min_ids,
    all_max_vals, all_max_ids,
    all_cat_vals, all_cat_ids,
    px_spread
):
    if not selected_coords:
        return [], "", None

    triggered = ctx.triggered_id
    cache_base = (triggered == 'update-plots-button')

    # single-select DR in feature mode arrives as string
    if isinstance(selected_coords, str):
        selected_coords = [selected_coords]

    # guard against accidental string splitting
    if isinstance(selected_features, str):
        selected_features = [selected_features]

    fv = {}
    for v, i in zip(all_min_vals, all_min_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["min"] = v
    for v, i in zip(all_max_vals, all_max_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["max"] = v
    for vals, i in zip(all_cat_vals, all_cat_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["cat"] = vals

    # unified panel specs
    panel_specs = []

    if plot_mode == "feature-comparison":
        if not selected_coords or len(selected_coords) != 1 or not selected_features:
            return [], "Select exactly one coordinate system and one or more features.", None

        cs = selected_coords[0]
        for feat in selected_features:
            panel_specs.append({
                "panel_id": f"{cs}__{feat}",
                "coordinate_system": cs,
                "feature": feat,
                "title": _feature_panel_title(feat),
            })

        feature_cols_to_load = list(selected_features)
        coord_systems_to_load = [cs]

    else:
        if not selected_coords:
            return [], "", None

        for cs in selected_coords:
            panel_specs.append({
                "panel_id": cs,
                "coordinate_system": cs,
                "feature": selected_feature,
                "title": cs,
            })

        feature_cols_to_load = [selected_feature] if selected_feature else None
        coord_systems_to_load = list(selected_coords)

    df = load_coordinates_from_db(
        db_path=GLOBAL_DB_PATH,
        coordinate_systems=coord_systems_to_load,
        feature_cols=feature_cols_to_load,
        filter_values=fv
    )

    n_plots = len(panel_specs)

    if n_plots == 1:
        plots_per_row = 1
    elif n_plots == 2:
        plots_per_row = 2
    else:
        plots_per_row = 3

    rows = ceil(n_plots / plots_per_row)
    row_comps = []
    serialized_rows = []

    PANEL_LEGEND_SLOT_HEIGHT = "40px"   #  fixed equal legend area for all feature-comparison panels
    PANEL_LEGEND_TOP_PAD = "10px"        #  same distance from plot to legend for all legend types

    for r in range(rows):
        slice_panels = panel_specs[r * plots_per_row:(r + 1) * plots_per_row]
        cols = []
        serialized_row = []
        col_width = max(1, int(12 / plots_per_row))

        for panel in slice_panels:
            cs = panel["coordinate_system"]
            panel_feature = panel["feature"]
            panel_title = panel["title"]
            panel_id = panel["panel_id"]

            xcol, ycol = f"x_{cs}", f"y_{cs}"

            fig = go.Figure()
            fig.update_layout(template="plotly_dark")

            # choose numeric vs categorical palette automatically
            if panel_feature is not None and panel_feature in df.columns:
                if pd.api.types.is_numeric_dtype(df[panel_feature]):
                    panel_palette = continuous_palette
                else:
                    panel_palette = categorical_palette
            else:
                panel_palette = continuous_palette

            base_img = create_datashader_image(
                df, xcol, ycol, panel_feature, panel_palette,
                px_spread=px_spread,
                cluster_cols=CLUSTER_COLS,
            )
            fig.add_layout_image({
                "source": base_img,
                "xref": "x", "yref": "y",
                "x": df[xcol].min(), "y": df[ycol].max(),
                "sizex": df[xcol].max() - df[xcol].min(),
                "sizey": df[ycol].max() - df[ycol].min(),
                "sizing": "stretch",
                "opacity": 1.0,
                "layer": "below"
            })

            if overlay_headers:
                df['overlay_flag'] = df['header'].isin(overlay_headers).astype('category')
                cvs = ds.Canvas(plot_width=500, plot_height=500)
                agg = cvs.points(df, xcol, ycol, ds.count_cat('overlay_flag'))
                color_key = {False: 'red', True: 'green'}
                overlay_img = tf.shade(agg, color_key=color_key, how='eq_hist')

                px_spread_use = 1 if px_spread is None or px_spread <= 0 else int(px_spread)

                overlay_img = tf.spread(overlay_img, px=px_spread_use)
                overlay_img = overlay_img.to_pil().convert("RGBA")
                fig.add_layout_image({
                    "source": overlay_img,
                    "xref": "x", "yref": "y",
                    "x": df[xcol].min(), "y": df[ycol].max(),
                    "sizex": df[xcol].max() - df[xcol].min(),
                    "sizey": df[ycol].max() - df[ycol].min(),
                    "sizing": "stretch",
                    "opacity": 1.0,
                    "layer": "above"
                })

            fig.update_layout(
                dragmode='lasso',
                yaxis_scaleanchor="x",
                xaxis=dict(
                    visible=False,
                    range=[df[xcol].min(), df[xcol].max()],
                    autorange=False,
                    constrain='domain'
                ),
                yaxis=dict(
                    visible=False,
                    range=[df[ycol].min(), df[ycol].max()],
                    autorange=False,
                    constrain='domain'
                ),
                margin=dict(l=10, r=10, t=10, b=10)
            )

            graph_container = html.Div(
                dcc.Graph(
                    id={
                        "type": "scatter-plot",
                        "index": panel_id,
                        "coordinate_system": cs
                    },
                    figure=fig,
                    config={"responsive": True},
                    style={"height": "100%", "width": "100%"}
                ),
                style={
                    "width": "100%",
                    "height": "100%",
                    "minHeight": 0,
                    "overflow": "hidden",
                    "border": "1px solid #444"
                }
            )

            if plot_mode == "feature-comparison":
                # assumes create_panel_legend always returns (component, type)
                panel_legend, panel_legend_type = create_panel_legend(
                    df=df,
                    feature_name=panel_feature,
                    continuous_palette=continuous_palette,
                    categorical_palette=categorical_palette
                )
            else:
                panel_legend, panel_legend_type = None, None

            if cache_base:
                serialized_row.append({
                    "coordinate_system": cs,
                    "panel_id": panel_id,
                    "panel_title": panel_title,
                    "figure": figure_to_serializable(fig)
                })

            cols.append(
                dbc.Col(
                    [
                        html.H5(
                            panel_title,
                            style={
                                "color": "white",
                                "textAlign": "center",
                                "marginBottom": "6px",
                                "flex": "0 0 auto",
                            }
                        ),
                        html.Div(
                            graph_container,
                            style={
                                "flex": "1 1 auto",
                                "minHeight": 0,
                                "display": "flex",
                                "flexDirection": "column",
                            }
                        ),

                        # fixed legend slot + same padding/margins for continuous and categorical
                        html.Div(
                            panel_legend,
                            style={
                                "flex": "0 0 auto",
                                "height": PANEL_LEGEND_SLOT_HEIGHT,
                                "minHeight": PANEL_LEGEND_SLOT_HEIGHT,
                                "maxHeight": PANEL_LEGEND_SLOT_HEIGHT,
                                "marginTop": "0px",
                                "paddingTop": PANEL_LEGEND_TOP_PAD,
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "flex-start",
                                "overflow": "hidden",
                            }
                        ) if panel_legend is not None else html.Div()
                    ],
                    width=col_width,
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": 0,
                    }
                )
            )

        if cache_base:
            serialized_rows.append(serialized_row)

        row_comps.append(
            dbc.Row(
                cols,
                justify="center",
                #className="mb-2",
                style={
                    "flex": "1 1 0",
                    "minHeight": 0,
                }
            )
        )

    # Shared legend only for DR comparison mode
    legend_component = ""
    if plot_mode == "dr-comparison":
        if (
            selected_feature
            and selected_feature != DENSITY_FEATURE
            and selected_feature in df.columns
        ):
            if pd.api.types.is_numeric_dtype(df[selected_feature]):
                if df[selected_feature].notna().any():
                    vmin = float(df[selected_feature].min())
                    vmax = float(df[selected_feature].max())

                    if vmin != vmax:
                        cmap_name = continuous_palette if continuous_palette in plt.colormaps() else "viridis"
                        cmap = plt.get_cmap(cmap_name)

                        colorscale = []
                        for i in range(256):
                            frac = i / 255.0
                            r, g, b, _ = cmap(frac)
                            colorscale.append([
                                frac,
                                f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
                            ])

                        legend_trace = go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode='markers',
                            marker=dict(
                                size=0.0001,
                                color=[vmin, vmax],
                                colorscale=colorscale,
                                showscale=True,
                                colorbar=dict(
                                    title=dict(text=selected_feature, font=dict(color='white')),
                                    tickfont=dict(color='white'),
                                    orientation='h',
                                    x=0.5,
                                    xanchor='center',
                                    thickness=15,
                                    len=0.8,
                                ),
                            ),
                            hoverinfo='none',
                            showlegend=False
                        )

                        legend_fig = go.Figure(data=[legend_trace])
                        legend_fig.update_xaxes(visible=False)
                        legend_fig.update_yaxes(visible=False)
                        legend_fig.update_layout(
                            margin=dict(l=40, r=40, t=10, b=20),
                            height=120,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )

                        legend_component = dcc.Graph(
                            figure=legend_fig,
                            style={"height": "120px"}
                        )
                    else:
                        legend_component = html.Div(
                            f"Legend: {selected_feature} has a constant value ({vmin}).",
                            style={"color": "white"}
                        )
            else:
                entries = _categorical_legend_entries(df[selected_feature], categorical_palette)
                if entries:
                    items = []
                    for cat, color in entries:
                        items.append(
                            html.Div(
                                [
                                    html.Span(
                                        style={
                                            "display": "inline-block",
                                            "width": "12px",
                                            "height": "12px",
                                            "marginRight": "6px",
                                            "backgroundColor": color,
                                            "border": "1px solid #ccc",
                                            "verticalAlign": "middle"
                                        }
                                    ),
                                    html.Span(str(cat), style={"verticalAlign": "middle"})
                                ],
                                style={
                                    "display": "inline-block",
                                    "marginRight": "12px",
                                    "marginBottom": "4px"
                                }
                            )
                        )

                    legend_component = html.Div(
                        [
                            html.Div(
                                f"Legend: {selected_feature}",
                                style={"marginBottom": "4px"}
                            ),
                            html.Div(items)
                        ],
                        style={"color": "white"}
                    )
        else:
            legend_component = _density_legend(compact=False)
    else:
        legend_component = html.Div()

    plots_grid = html.Div(
        row_comps,
        style={
            "height": "110%",
            "display": "flex",
            "flexDirection": "column",
            "gap": "8px",
            "minHeight": 0,
        }
    )

    if cache_base:
        return plots_grid, legend_component, serialized_rows
    else:
        return plots_grid, legend_component, dash.no_update
        
##############################
# PLOT MODE UI CALLBACK
##############################
@app.callback(
    Output("dr-color-feature-wrapper", "style"),
    Output("feature-comparison-wrapper", "style"),
    Output("coordinate-systems-checklist", "multi"),
    Input("plot-mode-toggle", "value"),
)
def toggle_plot_mode_ui(plot_mode):
    if plot_mode == "feature-comparison":
        return {"display": "none"}, {"display": "block"}, False   # single DR in feature mode
    return {"display": "block"}, {"display": "none"}, True        # multi DR in DR mode

##############################
# CLEAN PLOTS CALLBACK  - triggering button returns all plots to the state they were last time Update Plots was used.
##############################
@app.callback(
    Output('coordinates-plots-container', 'children', allow_duplicate=True),
    Output('color-legend-container', 'children', allow_duplicate=True),
    Output('overlay-store', 'data', allow_duplicate=True),
    Input('clear-plots-button', 'n_clicks'),
    State('base-plots-store', 'data'),
    State('color-legend-container', 'children'),
    prevent_initial_call=True
)
def clear_plots_to_base(_n, base_plots_data, current_legend):
    if not base_plots_data:
        return dash.no_update, dash.no_update, []

    # Restore last cached base figures without re-running Datashader
    row_comps = build_plot_grid_from_serialized(base_plots_data)

    # Clear overlay-store so future plot actions start clean
    return row_comps, current_legend, []

##############################
# OVERLAY POINTS CALLBACK
##############################
@app.callback(
    Output('overlay-store','data'),
    Input('overlay-points-button','n_clicks'),
    State({'type':'scatter-plot','index':ALL,'coordinate_system':ALL}, 'selectedData'),  # match the full graph id shape
    State({'type':'scatter-plot','index':ALL,'coordinate_system':ALL}, 'id'),   
    State('coordinate-systems-checklist','value'),
    State({'type':'continuous-filter-min','column_name':ALL}, 'value'),
    State({'type':'continuous-filter-min','column_name':ALL}, 'id'),
    State({'type':'continuous-filter-max','column_name':ALL}, 'value'),
    State({'type':'continuous-filter-max','column_name':ALL}, 'id'),
    State({'type':'cat-checklist','column_name':ALL}, 'value'),
    State({'type':'cat-checklist','column_name':ALL}, 'id'),
    prevent_initial_call=True
)
def overlay_points(
    n, all_sel, all_plot_ids, selected_coords,  
    all_min_vals, all_min_ids,
    all_max_vals, all_max_ids,
    all_cat_vals, all_cat_ids
):
    fv = {}
    for v, i in zip(all_min_vals, all_min_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["min"] = v
    for v, i in zip(all_max_vals, all_max_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["max"] = v
    for vals, i in zip(all_cat_vals, all_cat_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["cat"] = vals

    if isinstance(selected_coords, str):   
        selected_coords = [selected_coords]

    df = load_coordinates_from_db(
        db_path=GLOBAL_DB_PATH,
        coordinate_systems=selected_coords,
        feature_cols=None,   
        filter_values=fv
    )

    headers = set()
    # for sel, cs in zip(all_sel, selected_coords):
    for sel, graph_id in zip(all_sel, all_plot_ids):  
        if sel and 'lassoPoints' in sel:
            cs = graph_id["coordinate_system"]        
            xcol, ycol = f"x_{cs}", f"y_{cs}"
            pts = list(zip(sel['lassoPoints']['x'], sel['lassoPoints']['y']))
            path = Path(pts)
            mask = path.contains_points(df[[xcol, ycol]].values)
            headers.update(df.loc[mask, 'header'])

    return list(headers)


def _has_quality_string(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _bin_record_header(header, full_header, has_full_header: bool) -> str:
    if has_full_header and not pd.isna(full_header):
        return str(full_header)
    return str(header)


def write_bin_sequences(records: pd.DataFrame, out_dir: str, bin_name: str) -> str:
    """Write one bin as FASTQ only when every row has a quality string.

    A file named .fastq that also contains '>' records is not valid FASTQ,
    so a partial quality column falls back to FASTA for the whole bin.

    iterrows() builds a Series per read and dominated export time after the
    CSV had already been written. Zip the columns instead. The format rule
    is unchanged.
    """
    has_full_header = "full_header" in records.columns
    write_fastq = (
        not records.empty
        and "qualities" in records.columns
        and bool(records["qualities"].map(_has_quality_string).all())
    )

    # An empty frame has nothing to zip. Skip the column reads so a header-less
    # empty frame still produces an empty FASTA, matching the old iterrows path.
    if records.empty:
        headers = []
        sequences = []
        full_headers = None
        qualities = None
    else:
        headers = records["header"].tolist()
        sequences = records["sequence"].tolist()
        full_headers = records["full_header"].tolist() if has_full_header else None
        qualities = records["qualities"].tolist() if write_fastq else None

    if write_fastq:
        out_path = os.path.join(out_dir, f"{bin_name}.fastq")
        with open(out_path, "w") as handle:
            for index, header in enumerate(headers):
                full_header = full_headers[index] if full_headers is not None else None
                resolved = _bin_record_header(header, full_header, has_full_header)
                handle.write(
                    f"@{resolved}\n{sequences[index]}\n+\n{qualities[index]}\n"
                )
        return out_path

    out_path = os.path.join(out_dir, f"{bin_name}.fasta")
    with open(out_path, "w") as handle:
        for index, header in enumerate(headers):
            full_header = full_headers[index] if full_headers is not None else None
            resolved = _bin_record_header(header, full_header, has_full_header)
            handle.write(f">{resolved}\n{sequences[index]}\n")
    return out_path


def bin_export_format_note(paths) -> str:
    """Name the sequence format actually written for the status line."""
    suffixes = {os.path.splitext(path)[1] for path in paths}
    if suffixes == {".fastq"}:
        return "FASTQ"
    if suffixes == {".fasta"}:
        return "FASTA"
    if not suffixes:
        return "no sequence files"
    return "FASTQ and FASTA"


# SQLite binds at most 999 variables. 900 leaves room for the rest of the query.
FASTA_EXPORT_CHUNK_SIZE = 900
# A multi-million-read bin would otherwise write one log line per chunk.
FASTA_CHUNK_LOG_EVERY = 10
EXPORT_STARTED = "Export started…"
EXPORT_ALREADY_RUNNING = "Export already running."

_export_lock = threading.Lock()
_export_thread = None


def _fresh_export_state():
    return {
        "running": False,
        "message": "",
        "stage": "idle",
        "bin_index": None,
        "bin_total": None,
        "error": None,
        "history": [],
    }


_export_state = _fresh_export_state()


def reset_export_progress() -> None:
    """Clear the in-process export status. Tests call this between cases."""
    global _export_state, _export_thread
    with _export_lock:
        _export_state = _fresh_export_state()
        _export_thread = None


def export_progress_snapshot() -> dict:
    with _export_lock:
        snapshot = {key: value for key, value in _export_state.items() if key != "history"}
        snapshot["history"] = list(_export_state["history"])
        return snapshot


def export_progress_history() -> list:
    with _export_lock:
        return list(_export_state["history"])


def wait_for_export(timeout: float = 5.0) -> None:
    """Block until the current export thread finishes. Used by tests."""
    with _export_lock:
        thread = _export_thread
    if thread is None:
        return
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("bin export did not finish")


def _should_log_fasta_chunk(index: int, n_chunks: int, every: int = FASTA_CHUNK_LOG_EVERY) -> bool:
    """Log every chunk when there are few; otherwise the first, last, and every Nth.

    ``index`` is 1-based.
    """
    if n_chunks <= every:
        return True
    return index == 1 or index == n_chunks or index % every == 0


def _note_export_progress(message, *, stage, bin_index=None, bin_total=None, log=True):
    """Record the latest stage. ``log=False`` updates the GUI line without a log row."""
    if log:
        info(message)
    with _export_lock:
        _export_state["message"] = message
        _export_state["stage"] = stage
        if bin_index is not None:
            _export_state["bin_index"] = bin_index
        if bin_total is not None:
            _export_state["bin_total"] = bin_total
        if log:
            _export_state["history"].append(message)


def _finish_export(message, error=None):
    with _export_lock:
        _export_state["running"] = False
        _export_state["message"] = message
        _export_state["stage"] = "error" if error else "complete"
        _export_state["error"] = error
        _export_state["history"].append(message)


def _export_finished_message(output_dir, sequence_paths) -> str:
    format_note = bin_export_format_note(sequence_paths)
    return (
        f"Bins exported to {output_dir} "
        f"({format_note}; using full headers when available)."
    )


def _load_bin_fasta(db_path, headers, bin_name, chunk_size, bin_index, bin_total):
    """Read fasta rows in SQL batches. The chunk line is logged before each query."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(fasta);")
        cols = [row[1] for row in cursor.fetchall()]
        select_cols = ["header", "sequence"]
        if "full_header" in cols:
            select_cols.insert(1, "full_header")
        if "qualities" in cols:
            select_cols.append("qualities")
        select_sql = ", ".join(select_cols)

        n_chunks = ceil(len(headers) / chunk_size)
        frames = []
        for chunk_index, start in enumerate(range(0, len(headers), chunk_size), start=1):
            batch = headers[start:start + chunk_size]
            _note_export_progress(
                f"Bin '{bin_name}': fasta chunk {chunk_index}/{n_chunks}",
                stage="fasta",
                bin_index=bin_index,
                bin_total=bin_total,
                log=_should_log_fasta_chunk(chunk_index, n_chunks),
            )
            placeholders = ", ".join("?" * len(batch))
            query = (
                f"SELECT {select_sql} FROM fasta "
                f"WHERE header IN ({placeholders});"
            )
            frame = pd.read_sql_query(query, conn, params=batch)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    finally:
        conn.close()


def run_bin_export(bins_data, db_path, output_dir, chunk_size=FASTA_EXPORT_CHUNK_SIZE) -> str:
    """Export every bin. Each stage is logged before the next one starts."""
    os.makedirs(output_dir, exist_ok=True)
    n_bins = len(bins_data)
    sequence_paths = []
    _note_export_progress(
        f"Export started: {n_bins} bin(s) -> {output_dir}",
        stage="start",
        bin_total=n_bins,
    )

    for bin_index, bin_entry in enumerate(bins_data, start=1):
        name = bin_entry["bin_name"]
        coordinate_system = bin_entry.get("coordinate_system")
        polygon = bin_entry.get("polygon") or []
        filters = _format_bin_filters(bin_entry.get("filters")) or "none"
        _note_export_progress(
            (
                f"Bin {bin_index}/{n_bins} '{name}': {coordinate_system}, "
                f"{len(polygon)} polygon points, filters: {filters}"
            ),
            stage="bin",
            bin_index=bin_index,
            bin_total=n_bins,
        )

        started = time.perf_counter()
        df_filt = efficient_spatialite_export(
            db_path=db_path,
            coordinate_systems=[coordinate_system],
            polygons={coordinate_system: polygon},
            feature_col=None,
            filter_values=bin_entry.get("filters"),
        )
        _note_export_progress(
            (
                f"Bin '{name}': spatial query finished in "
                f"{time.perf_counter() - started:.1f}s, {len(df_filt)} rows"
            ),
            stage="spatial",
            bin_index=bin_index,
            bin_total=n_bins,
        )

        csv_path = os.path.join(output_dir, f"{name}.csv")
        df_filt.to_csv(csv_path, index=False)
        _note_export_progress(
            f"Bin '{name}': wrote CSV {csv_path} ({os.path.getsize(csv_path)} bytes)",
            stage="csv",
            bin_index=bin_index,
            bin_total=n_bins,
        )

        if "header" not in df_filt.columns:
            _note_export_progress(
                f"Bin '{name}': spatial result has no header column; skipped sequences",
                stage="sequences",
                bin_index=bin_index,
                bin_total=n_bins,
            )
            continue

        headers = df_filt["header"].unique().tolist()
        if not headers:
            _note_export_progress(
                f"Bin '{name}': spatial query returned no sequences",
                stage="sequences",
                bin_index=bin_index,
                bin_total=n_bins,
            )
            continue

        fasta_df = _load_bin_fasta(
            db_path,
            headers,
            name,
            chunk_size,
            bin_index=bin_index,
            bin_total=n_bins,
        )
        if fasta_df.empty:
            _note_export_progress(
                f"Bin '{name}': no sequences found for {len(headers)} headers",
                stage="sequences",
                bin_index=bin_index,
                bin_total=n_bins,
            )
            continue

        write_started = time.perf_counter()
        out_path = write_bin_sequences(fasta_df, output_dir, name)
        sequence_paths.append(out_path)
        suffix = os.path.splitext(out_path)[1].lstrip(".").upper()
        _note_export_progress(
            (
                f"Bin '{name}': wrote {out_path} "
                f"({suffix}, {len(fasta_df)} rows, {time.perf_counter() - write_started:.1f}s)"
            ),
            stage="sequences",
            bin_index=bin_index,
            bin_total=n_bins,
        )

    message = _export_finished_message(output_dir, sequence_paths)
    _note_export_progress(f"Export complete: {message}", stage="complete")
    return message


def _export_worker(bins_data, db_path, output_dir, chunk_size):
    try:
        message = run_bin_export(
            bins_data, db_path, output_dir, chunk_size=chunk_size
        )
        _finish_export(message)
    except Exception as exc:
        details = traceback.format_exc()
        warn(f"Export failed: {exc}")
        info(details)
        _finish_export(f"Export failed: {exc}", error=details)


def start_bin_export(bins_data, db_path, output_dir, chunk_size=FASTA_EXPORT_CHUNK_SIZE) -> str:
    """Start one export thread. A second call while it runs does not start another.

    The click handler returns immediately. Dash cannot change the status strip
    until that handler returns, and a single-thread server would not serve the
    progress interval until the POST finished.
    """
    global _export_thread
    payload = copy.deepcopy(list(bins_data))
    with _export_lock:
        if _export_state["running"]:
            already_running = True
        else:
            already_running = False
            _export_state["running"] = True
            _export_state["message"] = EXPORT_STARTED
            _export_state["stage"] = "start"
            _export_state["bin_index"] = None
            _export_state["bin_total"] = len(payload)
            _export_state["error"] = None
            _export_state["history"] = [EXPORT_STARTED]
            _export_thread = None
    if already_running:
        warn(EXPORT_ALREADY_RUNNING)
        return EXPORT_ALREADY_RUNNING

    # Log before start() so this line cannot land after the worker's first stage.
    info(EXPORT_STARTED)
    thread = threading.Thread(
        target=_export_worker,
        args=(payload, db_path, output_dir, chunk_size),
        daemon=True,
        name="bin-export",
    )
    with _export_lock:
        _export_thread = thread
    thread.start()
    return EXPORT_STARTED


##############################
# BINS + INSPECT + EXPORT (FASTQ OR FASTA, CHUNKED, FULL HEADER)
##############################
@app.callback(
    [
        Output('bins-store','data'),
        Output('error-message','children'),
        Output('bin-table-container','children'),
        Output('bin-list-container','children'),
        Output('export-progress-interval', 'disabled'),
        Output('bin-name-input', 'value'),
    ],
    [
        Input('create-bin-button','n_clicks'),
        Input('inspect-bin-button','n_clicks'),
        Input('export-bins-button','n_clicks'),
    ],
    [
        State({'type':'scatter-plot','index':ALL,'coordinate_system':ALL}, 'selectedData'),  # match the full graph id shape
        State({'type':'scatter-plot','index':ALL,'coordinate_system':ALL}, 'id'),
        State('bin-name-input','value'),
        State('bins-store','data'),
        State({'type':'continuous-filter-min','column_name':ALL}, 'value'),
        State({'type':'continuous-filter-min','column_name':ALL}, 'id'),
        State({'type':'continuous-filter-max','column_name':ALL}, 'value'),
        State({'type':'continuous-filter-max','column_name':ALL}, 'id'),
        State({'type':'cat-checklist','column_name':ALL}, 'value'),
        State({'type':'cat-checklist','column_name':ALL}, 'id'),
        State('color-feature-dropdown','value'),
        State('coordinate-systems-checklist','value'),
    ],
    prevent_initial_call=True
)

def handle_bin_operations(
    create_clicks, inspect_clicks, export_clicks,
    all_sel, all_plot_ids, bin_name, bins_data,   
    all_min_vals, all_min_ids,
    all_max_vals, all_max_ids,
    all_cat_vals, all_cat_ids,
    color_feature, selected_coords
):
    ctx_trig = ctx.triggered_id
    error_message = ""
    table_content = None
    interval_disabled = dash.no_update
    bin_name_value = dash.no_update
    if not bins_data:
        bins_data = []

    fv = {}
    for v, i in zip(all_min_vals, all_min_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["min"] = v
    for v, i in zip(all_max_vals, all_max_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["max"] = v
    for vals, i in zip(all_cat_vals, all_cat_ids):
        fv.setdefault(i["column_name"], {"min": None, "max": None, "cat": []})["cat"] = vals

    if isinstance(selected_coords, str):  
        selected_coords = [selected_coords]

    if ctx_trig == 'create-bin-button':
        # active_lassos = [
        #     (cs, sel)
        #     for sel, cs in zip(all_sel, selected_coords)
        #     if sel and 'lassoPoints' in sel
        # ]
        active_lassos = [   #  derive cs from plot ids
            (graph_id["coordinate_system"], sel)
            for sel, graph_id in zip(all_sel, all_plot_ids)
            if sel and 'lassoPoints' in sel
        ]

        existing_names = {b["bin_name"] for b in bins_data}
        if not bin_name or bin_name.strip() == "":
            error_message = "Please provide a bin name."
        elif bin_name.strip() in existing_names:
            error_message = f"A bin named '{bin_name.strip()}' already exists. Choose a different name."
        elif len(active_lassos) == 0:
            error_message = "Please create a lasso selection before creating a bin."
        elif len(active_lassos) > 1:
            error_message = "Only one lasso may be active. Clear the old lasso before creating a new bin."
        else:
            active_cs, active_sel = active_lassos[0]
            polygon = list(zip(
                active_sel['lassoPoints']['x'],
                active_sel['lassoPoints']['y']
            ))

            bins_data.append({
                "bin_name": bin_name,
                "filters": fv,
                "coordinate_system": active_cs,
                "polygon": polygon
            })
            error_message = f"Bin '{bin_name}' created successfully."
            bin_name_value = ""

    elif ctx_trig == 'inspect-bin-button':
        active_lassos = [
            (graph_id["coordinate_system"], sel)
            for sel, graph_id in zip(all_sel, all_plot_ids)
            if sel and 'lassoPoints' in sel
        ]

        if len(active_lassos) == 0:
            error_message = "No lasso selection to inspect."
        elif len(active_lassos) > 1:
            error_message = "Only one lasso may be active for inspection."
        else:
            active_cs, active_sel = active_lassos[0]

            df_master = load_coordinates_from_db(
                db_path=GLOBAL_DB_PATH,
                coordinate_systems=[active_cs],
                feature_cols=[color_feature] if color_feature else None,  
                filter_values=fv
            )

            xcol, ycol = f"x_{active_cs}", f"y_{active_cs}"
            pts = list(zip(
                active_sel['lassoPoints']['x'],
                active_sel['lassoPoints']['y']
            ))
            path = Path(pts)
            mask = path.contains_points(df_master[[xcol, ycol]].values)

            if mask.any():
                sub_df = df_master.loc[mask].copy()
                table_content = dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in sub_df.columns],
                    data=sub_df.to_dict("records"),
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_as_list_view=True,
                )
                error_message = f"Inspect Bin: {len(sub_df)} points found."
            else:
                error_message = "Inspect Bin: 0 points found."

    elif ctx_trig == 'export-bins-button':
        # Return as soon as the thread is running. The interval callback below
        # is what moves the status strip; this POST cannot do that itself.
        if not bins_data:
            error_message = "No bins to export."
        else:
            error_message = start_bin_export(
                bins_data, GLOBAL_DB_PATH, output_dir_default
            )
            if error_message == EXPORT_STARTED:
                interval_disabled = False

    return (
        bins_data,
        _status_bar(error_message),
        table_content,
        _build_bin_list(bins_data),
        interval_disabled,
        bin_name_value,
    )


@app.callback(
    Output('error-message', 'children', allow_duplicate=True),
    Output('export-progress-interval', 'disabled', allow_duplicate=True),
    Input('export-progress-interval', 'n_intervals'),
    prevent_initial_call=True,
)
def poll_export_progress(_n_intervals):
    """Push the latest export line, then stop polling once the thread finishes."""
    snapshot = export_progress_snapshot()
    if snapshot["running"]:
        return _status_bar(snapshot["message"]), False
    return _status_bar(snapshot["message"]), True


@app.callback(
    Output('bins-store', 'data', allow_duplicate=True),
    Output('bin-list-container', 'children', allow_duplicate=True),
    Input({'type': 'remove-bin-btn', 'index': ALL}, 'n_clicks'),
    State('bins-store', 'data'),
    prevent_initial_call=True,
)
def remove_bin(n_clicks_list, bins_data):
    if not any(n_clicks_list):
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    bins_data = bins_data or []
    triggered = ctx.triggered_id
    if triggered is None:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    idx = triggered["index"]
    if 0 <= idx < len(bins_data):
        bins_data = [b for i, b in enumerate(bins_data) if i != idx]
    return bins_data, _build_bin_list(bins_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="INTERACTIVE KMER-BASED READ ORDINATION INTERFACE - BIN2WIN"
    )
    parser.add_argument("-d", "--database", required=True)
    parser.add_argument("-o", "--output_dir", default="bins")

    args = parser.parse_args()

    run_dash_app(
        db_path=args.database,
        output_dir=args.output_dir
    )