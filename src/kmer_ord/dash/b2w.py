import sys
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd
import os
import sqlite3
import plotly.graph_objects as go
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
import numpy as np
from dash import dash_table
import matplotlib.colors
import matplotlib.pyplot as plt
import bokeh.palettes
from colorcet import glasbey
import io
from PIL import Image
import argparse
from math import ceil
from matplotlib.path import Path

def run_dash_app(db_path: str, output_dir: str, host="127.0.0.1", port=8050):
    global GLOBAL_DB_PATH, output_dir_default
    GLOBAL_DB_PATH = db_path
    output_dir_default = output_dir

    initialise_app_state()

    print_banner(host=host, port=port)
    app.layout = build_layout()

    print_banner(host=host, port=port)

    app.run(host=host, port=port, debug=False)

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



def load_coordinates_from_db(db_path,
                             coordinate_systems,
                             feature_cols=None, 
                             filter_values=None):

    conn = get_connection(db_path)

    try:
        select = ["c.sequence_id AS header"]
        cols = set()
        if feature_cols:   
            if isinstance(feature_cols, str):
                feature_cols = [feature_cols]
            cols.update(feature_cols)

        if filter_values:
            cols.update(filter_values.keys())

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
    - ≤20 non-noise labels → glasbey qualitative palette
    - >20 non-noise labels → Turbo sequential colormap for max perceptual separation

    Returns (color_key_dict, n_clusters, mode)
    where mode is "categorical" or "turbo".
    """
    regular = sorted(c for c in categories if c != -1)
    n = len(regular)

    if n <= _CLUSTER_CATEGORICAL_THRESHOLD:
        colors = glasbey[:n]
        mode = "categorical"
    else:
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

    if color_col is not None and color_col in df:
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
            #  Datashader count_cat() requires a categorical column
            cat = df[color_col].astype("category")
            df_ds['cat_value'] = pd.Categorical(cat)
            categories = list(cat.cat.categories)
        
            # use categorical palette selector only
            if color_palette == "Category10":
                if len(categories) <= 10:
                    palette = bokeh.palettes.Category10[10][:len(categories)]
                else:
                    palette = glasbey[:len(categories)]
            elif color_palette == "glasbey":
                palette = glasbey[:len(categories)]
            else:
                palette = glasbey[:len(categories)]
        
            key = dict(zip(categories, palette))
            agg = cvs.points(df_ds, x='x', y='y', agg=ds.count_cat('cat_value'))
            img = tf.shade(agg, color_key=key, how='eq_hist')
    
    else:
        agg = cvs.points(df_ds, x='x', y='y', agg=ds.count())
        cmap = plt.get_cmap("plasma")
        img = tf.shade(agg, cmap=[matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)], how='eq_hist')
        #img = tf.shade(agg, cmap="fire", how='eq_hist')
    
    img = tf.spread(img, px=max(int(px_spread), 1))
    return img.to_pil()
    
# Small per-panel legend for feature comparison mode.
def create_panel_legend(df, feature_name, continuous_palette, categorical_palette):
    """
    Return:
        (legend_component, legend_type)
    where legend_type is "continuous" or "categorical".
    Must always return exactly 2 values.
    """

    if not feature_name or feature_name not in df.columns:
        return (
            html.Div(
                "Density map",
                style={"color": "white", "fontSize": "0.75rem", "textAlign": "center"}
            ),
            "continuous"
        )

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

    # Categorical feature -> compact HTML legend
    cats = df[feature_name].astype("category").cat.categories.tolist()

    if not cats:
        return (
            html.Div(
                f"{feature_name}: no categories",
                style={"color": "white", "fontSize": "0.75rem", "textAlign": "center"}
            ),
            "categorical"
        )

    if categorical_palette == "Category10" and len(cats) <= 10:
        palette = bokeh.palettes.Category10[10][:len(cats)]
    else:
        palette = glasbey[:len(cats)]

    items = []
    for cat, color in zip(cats, palette):
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
    """Return a slim single-line status strip, or None if msg is empty."""
    if not msg:
        return None
    msg_lower = msg.lower()
    if "successfully" in msg_lower or "exported" in msg_lower:
        accent, text_col = "#2e7d32", "#a5d6a7"   # green
    elif "found" in msg_lower and "0 points" not in msg_lower:
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
                options=(
                    [{"label": f['column_name'], "value": f['column_name']}
                     for f in feature_info]
                    + ([{"label": "── Clustering ──", "value": "__cluster_sep__", "disabled": True}]
                       + [{"label": c, "value": c} for c in CLUSTER_COLS]
                       if CLUSTER_COLS else [])
                ),
                placeholder="Select feature",
                searchable=False,
            ),
        ], id="dr-color-feature-wrapper"),

        # feature comparison selector
        html.Div([
            dbc.Label("Features to Compare", className="small mt-3"),
            dcc.Dropdown(
                id="feature-comparison-dropdown",
                options=(
                    [{"label": f['column_name'], "value": f['column_name']}
                     for f in feature_info]
                    + ([{"label": "── Clustering ──", "value": "__cluster_sep__", "disabled": True}]
                       + [{"label": c, "value": c} for c in CLUSTER_COLS]
                       if CLUSTER_COLS else [])
                ),
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
                {"label": "Category10", "value": "Category10"},
                {"label": "glasbey", "value": "glasbey"},
            ],
            value="glasbey",
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
                "title": feat,
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
        if selected_feature and (selected_feature in df.columns):
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
                cats = df[selected_feature].astype("category").cat.categories.tolist()
                if cats:
                    if categorical_palette == "Category10" and len(cats) <= 10:
                        palette = bokeh.palettes.Category10[10][:len(cats)]
                    else:
                        palette = glasbey[:len(cats)]

                    items = []
                    for cat, color in zip(cats, palette):
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
            legend_component = html.Div(
                "Color map: point density (Datashader 'fire' colormap, eq_hist). "
                "Brighter = higher local density.",
                style={"color": "white"}
            )
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


##############################
# BINS + INSPECT + EXPORT (FASTQ/FASTA, CHUNKED, FULL HEADER)
##############################
@app.callback(
    [
        Output('bins-store','data'),
        Output('error-message','children'),
        Output('bin-table-container','children'),
        Output('bin-list-container','children')
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
        if not bins_data:
            error_message = "No bins to export."
        else:
            os.makedirs(output_dir_default, exist_ok=True)

            for bin_entry in bins_data:
                df_filt = efficient_spatialite_export(
                    db_path=GLOBAL_DB_PATH,
                    coordinate_systems=[bin_entry["coordinate_system"]],
                    polygons={bin_entry["coordinate_system"]: bin_entry["polygon"]},
                    feature_col=None,
                    filter_values=bin_entry["filters"]
                )

                csv_path = os.path.join(
                    output_dir_default,
                    f"{bin_entry['bin_name']}.csv"
                )
                df_filt.to_csv(csv_path, index=False)

                if 'header' in df_filt:
                    headers = df_filt['header'].unique().tolist()
                    if headers:
                        conn = get_connection(GLOBAL_DB_PATH)
                        try:
                            cursor = conn.cursor()
                            cursor.execute("PRAGMA table_info(fasta);")
                            cols = [r[1] for r in cursor.fetchall()]
                            has_qualities_col = 'qualities' in cols
                            has_full_header_col = 'full_header' in cols

                            select_cols_list = ["header", "sequence"]
                            if has_full_header_col:
                                select_cols_list.insert(1, "full_header")
                            if has_qualities_col:
                                select_cols_list.append("qualities")
                            select_cols = ", ".join(select_cols_list)

                            chunk_size = 900
                            dfs = []
                            for i in range(0, len(headers), chunk_size):
                                batch = headers[i:i + chunk_size]
                                placeholders = ", ".join("?" * len(batch))
                                fasta_q = (
                                    f"SELECT {select_cols} FROM fasta "
                                    f"WHERE header IN ({placeholders});"
                                )
                                df_chunk = pd.read_sql_query(
                                    fasta_q, conn, params=batch
                                )
                                if not df_chunk.empty:
                                    dfs.append(df_chunk)

                            if not dfs:
                                continue

                            fasta_df = pd.concat(dfs, ignore_index=True)

                            write_fastq = (
                                has_qualities_col
                                and 'qualities' in fasta_df.columns
                                and fasta_df['qualities'].notna().any()
                            )

                            def get_output_header(row):
                                if (
                                    has_full_header_col
                                    and 'full_header' in row
                                    and not pd.isna(row['full_header'])
                                ):
                                    return row['full_header']
                                return row['header']

                            if write_fastq:
                                out_path = os.path.join(
                                    output_dir_default,
                                    f"{bin_entry['bin_name']}.fastq"
                                )
                                with open(out_path, 'w') as fh:
                                    for _, row in fasta_df.iterrows():
                                        q = row.get('qualities')
                                        header_out = get_output_header(row)
                                        if pd.isna(q):
                                            fh.write(
                                                f">{header_out}\n{row['sequence']}\n"
                                            )
                                        else:
                                            fh.write(
                                                f"@{header_out}\n"
                                                f"{row['sequence']}\n"
                                                f"+\n"
                                                f"{q}\n"
                                            )
                            else:
                                out_path = os.path.join(
                                    output_dir_default,
                                    f"{bin_entry['bin_name']}.fasta"
                                )
                                with open(out_path, 'w') as fh:
                                    for _, row in fasta_df.iterrows():
                                        header_out = get_output_header(row)
                                        fh.write(
                                            f">{header_out}\n{row['sequence']}\n"
                                        )
                        finally:
                            conn.close()

            error_message = (
                f"Bins exported to {output_dir_default} "
                f"(FASTQ if qualities present, otherwise FASTA; using full headers when available)."
            )

    return bins_data, _status_bar(error_message), table_content, _build_bin_list(bins_data)


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