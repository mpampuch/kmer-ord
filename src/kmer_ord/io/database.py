from os import name
from pathlib import Path
import pandas as pd
import sqlite3
from kmer_ord.utils.logging_utils import info, console

# ---------------------------------------------------------
def initialize_spatialite_db(db_file: Path):
    import sqlite3
    from sqlite3 import OperationalError

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    try:
        conn.enable_load_extension(True)
        cursor.execute("SELECT load_extension('mod_spatialite');")
        cursor.execute("SELECT InitSpatialMetaData(1);")
    except OperationalError as e:
        conn.close()
        raise RuntimeError(f"Error initializing SpatiaLite: {e}")

    # Speed PRAGMAs
    cursor.execute("PRAGMA journal_mode = MEMORY;")
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA temp_store = MEMORY;")
    cursor.execute("PRAGMA cache_size = -500000;")

    conn.commit()
    return conn

# ---------------------------------------------------------

def create_fasta_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fasta (
            header TEXT PRIMARY KEY,
            full_header TEXT,
            sequence TEXT,
            qualities TEXT);
        """)
    conn.commit()


def sequence_source_for_database(input_file: Path, canonical_fasta: Path, preserve_fastq: bool) -> Path:
    """Choose the file stored in the fasta table.

    A FASTQ input is stored directly when preserve_fastq is true, so quality
    strings survive. Otherwise the canonical FASTA is stored and qualities
    stay empty. FASTA inputs have no quality strings either way.
    """
    from kmer_ord.io.sequence import sequence_format

    input_file = Path(input_file)
    canonical_fasta = Path(canonical_fasta)
    if preserve_fastq and sequence_format(input_file) == "fastq":
        return input_file
    return canonical_fasta


def populate_fasta_table(conn, fasta_file: Path, batch_size=50000):
    from Bio.SeqIO.FastaIO import SimpleFastaParser
    from Bio.SeqIO.QualityIO import FastqGeneralIterator
    from kmer_ord.io.sequence import open_sequence_text, sequence_format

    fasta_file = Path(fasta_file)
    kind = sequence_format(fasta_file)
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO fasta (header, full_header, sequence, qualities)
        VALUES (?, ?, ?, ?);
    """

    conn.execute("BEGIN TRANSACTION;")
    batch = []

    def flush():
        if batch:
            cursor.executemany(insert_sql, batch)
            batch.clear()

    with open_sequence_text(fasta_file) as handle:
        if kind == "fastq":
            for title, seq, qual in FastqGeneralIterator(handle):
                short_id = title.split()[0]
                # A FASTQ record is invalid when the quality line is a different
                # length from the sequence, so refuse to store it.
                if len(qual) != len(seq):
                    raise ValueError(
                        f"Quality length {len(qual)} does not match sequence "
                        f"length {len(seq)} for read '{short_id}'."
                    )
                batch.append((short_id, title, seq, qual))
                if len(batch) >= batch_size:
                    flush()
        else:
            for title, seq in SimpleFastaParser(handle):
                short_id = title.split()[0]
                batch.append((short_id, title, seq, None))
                if len(batch) >= batch_size:
                    flush()

    flush()
    conn.commit()


# ---------------------------------------------------------
# FEATURES TABLE (auto-merged artifacts)
# ---------------------------------------------------------

def create_features_table(conn, df: pd.DataFrame):
    import pandas as pd
    cursor = conn.cursor()

    create_sql = "CREATE TABLE features (sequence_id TEXT PRIMARY KEY"

    for col in df.columns[1:]:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            sqltype = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            sqltype = "REAL"
        else:
            sqltype = "TEXT"
        create_sql += f", {col} {sqltype}"

    create_sql += ");"
    cursor.execute(create_sql)
    conn.commit()


def populate_features_table(conn, df: pd.DataFrame):
    import pandas as pd
    cursor = conn.cursor()

    columns = ", ".join(df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    insert_sql = f"INSERT INTO features ({columns}) VALUES ({placeholders});"

    conn.execute("BEGIN TRANSACTION;")
    cursor.executemany(insert_sql, df.to_numpy().tolist())
    conn.commit()


# ---------------------------------------------------------
# COORDINATES TABLE (from DR embeddings)
# ---------------------------------------------------------

def create_coordinates_table(conn, df: pd.DataFrame):
    """
    Create the coordinates table with proper sequence_id and geometry columns.
    """
    import pandas as pd
    cursor = conn.cursor()

    # Use sequence_id as primary key
    cursor.execute("""
        CREATE TABLE coordinates (
            sequence_id TEXT PRIMARY KEY,
            FOREIGN KEY (sequence_id) REFERENCES features (sequence_id)
        );
    """)

    methods = set()

    # detect methods dynamically from df columns (skip the first column which is sequence_id)
    for col in df.columns[1:]:
        base = col.rsplit("_", 1)[0]
        methods.add(base)

    for method in methods:
        # determine number of dimensions
        dims = 0
        i = 1
        while f"{method}_{i}" in df.columns:
            dims += 1
            i += 1

        if dims not in (2, 3):
            raise ValueError(f"{method} must have 2 or 3 dimensions for spatial storage.")

        geom_type = "POINT" if dims == 2 else "POINTZ"
        coord_dim = "XY" if dims == 2 else "XYZ"

        cursor.execute(
            f"""
            SELECT AddGeometryColumn(
                'coordinates',
                '{method}',
                0,
                '{geom_type}',
                '{coord_dim}'
            );
            """
        )
    
    # Create spatial indices for all geometry columns
    for method in methods:
        cursor.execute(f"SELECT CreateSpatialIndex('coordinates', '{method}');")

    conn.commit()
    return methods




def populate_coordinates_table(conn, df: pd.DataFrame, methods):
    """
    Populate the coordinates table using batch inserts for performance.
    """
    cursor = conn.cursor()

    # Sort methods for stable column ordering
    methods = sorted(methods)

    # Determine dimensionality per method
    method_dims = {}
    for method in methods:
        i = 1
        while f"{method}_{i}" in df.columns:
            i += 1
        method_dims[method] = i - 1

    # Validate dims and build vectorized WKT series
    wkt_series = {}
    for method in methods:
        n = method_dims[method]
        if n == 2:
            wkt_series[method] = (
                "POINT(" + df[f"{method}_1"].astype(str) + " " + df[f"{method}_2"].astype(str) + ")"
            )
        elif n == 3:
            wkt_series[method] = (
                "POINT Z(" + df[f"{method}_1"].astype(str) + " "
                + df[f"{method}_2"].astype(str) + " "
                + df[f"{method}_3"].astype(str) + ")"
            )
        else:
            raise ValueError(f"{method} must have 2 or 3 dimensions.")

    # Build SQL once
    columns = ["sequence_id"] + methods
    placeholders = ["?"] + [f"ST_GeomFromText(?, 0)" for _ in methods]
    insert_sql = f"INSERT INTO coordinates ({','.join(columns)}) VALUES ({','.join(placeholders)});"

    # Build all rows as list of tuples
    sequence_ids = df.iloc[:, 0].tolist()
    wkt_arrays = [wkt_series[m].tolist() for m in methods]
    rows = list(zip(sequence_ids, *wkt_arrays))

    conn.execute("BEGIN TRANSACTION;")
    cursor.executemany(insert_sql, rows)
    conn.commit()


# -----------------------------
# DR EMBEDDINGS TABLES
# -----------------------------
def save_dr_embeddings(conn, df: pd.DataFrame, method_name: str, force: bool = False):
    import pandas as pd
    table_name = f"embedding_{method_name.lower()}"

    # Check if table exists
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
    )
    exists = cursor.fetchone() is not None

    if exists and not force:
        info(f"Skipping DR embeddings for '{method_name}', table already exists")
        return table_name

    if exists and force:
        info(f"Overwriting existing embeddings table '{table_name}'")
        conn.execute(f"DROP TABLE IF EXISTS {table_name};")

    df.to_sql(table_name, conn, index=False)
    return table_name

# -----------------------------
# CLUSTERING TABLE
# -----------------------------
def save_clustering_results(conn: sqlite3.Connection, cluster_files: list[Path], force: bool = False):
    """
    Merge multiple clustering result files into a single flat table and save as 'Clustering'.
    """
    import sqlite3
    import pandas as pd
    dfs = [pd.read_csv(f, sep="\t") for f in cluster_files]
    merged_df = pd.concat(dfs, axis=1)

    # Keep only one sequence_id column
    if "sequence_id" in merged_df.columns:
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Ensure sequence_id is stored as TEXT so it matches the coordinates table
    merged_df["sequence_id"] = merged_df["sequence_id"].astype(str)

    table_name = "clustering"

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    exists = cursor.fetchone() is not None

    if exists and not force:
        info(f"Skipping clustering table, already exists (use --force to overwrite)")
        return table_name

    if exists and force:
        info(f"Overwriting existing clustering table")
        conn.execute(f"DROP TABLE IF EXISTS {table_name};")

    merged_df.to_sql(table_name, conn, index=False)
    return table_name

# -----------------------------
# WRAPPER TO ADD DR + CLUSTERING TO DB
# -----------------------------
def add_dr_and_clusters_to_db(db_path: Path, embedding_files: list[Path], cluster_files: list[Path], force: bool = False):
    """
    Opens (or creates) DB, adds DR embedding tables and a single Clustering table.
    Returns the sqlite3 connection path.
    """
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(str(db_path))

    # Save embeddings
    for emb_file in embedding_files:
        df = pd.read_csv(emb_file, sep="\t")
        # method name extraction from filename
        method_name = Path(emb_file).stem.split("_")[-3]
        save_dr_embeddings(conn, df, method_name, force=force)

    # Save clustering
    save_clustering_results(conn, cluster_files, force=force)

    conn.commit()
    conn.close()
    return db_path

# ---------------------------------------------------------
# FEATURE INJECTION
# ---------------------------------------------------------

def inject_features(db_path: Path, input_df: "pd.DataFrame") -> dict:
    """
    Merge new columns from input_df into the features table, keyed on sequence_id.

    Rules
    -----
    - Left join: every sequence already in the DB is preserved.
    - Columns already present in the features table are skipped (existing values win).
    - Input rows whose sequence_id has no match in the DB are silently dropped.
    - DB sequences with no matching row in the input get NULL for all new columns.
    - Column names that contain characters outside [A-Za-z0-9_\-.] are rejected to
      prevent SQL identifier injection.

    Returns a report dict consumed by the CLI for on-screen output.
    """
    import sqlite3
    import re
    import pandas as pd

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Verify features table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='features';"
        )
        if not cursor.fetchone():
            raise RuntimeError(
                "No 'features' table found in the database. "
                "Run the full projection pipeline first."
            )

        # Current features columns
        cursor.execute("PRAGMA table_info(features);")
        existing_cols = {row[1] for row in cursor.fetchall()}

        # Sequence IDs currently in DB
        db_seqids = {
            row[0]
            for row in cursor.execute("SELECT sequence_id FROM features;").fetchall()
        }

        input_seqids = set(input_df["sequence_id"])
        matched      = input_seqids & db_seqids
        unmatched_input = input_seqids - db_seqids
        unmatched_db    = db_seqids   - input_seqids

        if not matched:
            raise RuntimeError(
                f"None of the {len(input_seqids)} sequence_id value(s) in the input file "
                f"match the {len(db_seqids)} sequence_id value(s) in the database. "
                "Verify that your input file was produced with the same dataset."
            )

        # Split columns into new vs already-present
        data_cols    = [c for c in input_df.columns if c != "sequence_id"]
        skipped_cols = [c for c in data_cols if c in existing_cols]
        new_cols     = [c for c in data_cols if c not in existing_cols]

        # Guard against unsafe identifier characters
        safe_re = re.compile(r'^[A-Za-z0-9_\-\.]+$')
        unsafe_cols = [c for c in new_cols if not safe_re.match(c)]
        if unsafe_cols:
            raise ValueError(
                f"Column name(s) contain characters not allowed in SQL identifiers "
                f"and cannot be injected: {unsafe_cols}"
            )

        # Only update rows whose sequence_id exists in the DB
        matched_df = input_df[input_df["sequence_id"].isin(db_seqids)].copy()

        conn.execute("BEGIN TRANSACTION;")

        for col in new_cols:
            dtype = input_df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sqltype = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                sqltype = "REAL"
            else:
                sqltype = "TEXT"

            cursor.execute(f'ALTER TABLE features ADD COLUMN "{col}" {sqltype};')

            seqids = matched_df["sequence_id"].tolist()
            vals   = [None if pd.isna(v) else v for v in matched_df[col].tolist()]

            cursor.executemany(
                f'UPDATE features SET "{col}" = ? WHERE sequence_id = ?;',
                list(zip(vals, seqids)),
            )

        conn.commit()

    finally:
        conn.close()

    return {
        "injected_cols":   new_cols,
        "skipped_cols":    skipped_cols,
        "matched":         len(matched),
        "unmatched_input": len(unmatched_input),
        "unmatched_db":    len(unmatched_db),
        "total_db":        len(db_seqids),
    }


# ---------------------------------------------------------
# DATABASE INSPECTION (DEBUGGING / QA)
# ---------------------------------------------------------

def inspect_database(db_file: Path, limit: int = 5):
    """
    Inspect a SpatiaLite/SQLite database.
    Dynamically lists all tables, including:
    - fasta
    - features
    - coordinates
    - embedding_<method>
    - Clustering
    Shows the first `limit` rows for each table.
    """
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # Try enabling SpatiaLite extension
    try:
        conn.enable_load_extension(True)
        cursor.execute("SELECT load_extension('mod_spatialite');")
    except sqlite3.OperationalError:
        pass

    # Query all user tables
    cursor.execute("""SELECT name FROM sqlite_master WHERE type='table';""")
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        info("No tables found in database.")
        conn.close()
        return
    
    SYSTEM_PREFIXES = ("sqlite_",
                       "idx_",
                       "geometry_",
                       "spatial_ref_",
                       "virts_",
                       "views_")

    SYSTEM_TABLES = {"data_licenses",
                     "sql_statements_log",
                     "SpatialIndex",
                     "spatial_ref_sys",
                     "KNN2",
                     "geometry_columns",
                     "spatialite_history",
                     "ElementaryGeometries"}

    tables = [t for t in tables
              if not t.startswith(SYSTEM_PREFIXES)
              and t not in SYSTEM_TABLES]

    from rich.table import Table
    from rich import box as rich_box

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        schema = cursor.fetchall()

        console.print()
        console.rule(table, style="dim")

        if not schema:
            info("Table not found or empty.")
            continue

        try:
            if table == "coordinates":
                geom_query = """SELECT f_geometry_column FROM geometry_columns WHERE f_table_name = 'coordinates';"""
                geom_cols = pd.read_sql_query(geom_query, conn)

                if geom_cols.empty:
                    df = pd.read_sql_query(f"SELECT * FROM coordinates LIMIT {limit};", conn)
                else:
                    geom_names = geom_cols["f_geometry_column"].tolist()
                    select_parts = ["CAST(sequence_id AS TEXT) AS sequence_id"]
                    for col in geom_names:
                        select_parts.append(f"ST_X({col}) AS {col}_1")
                        select_parts.append(f"ST_Y({col}) AS {col}_2")
                        select_parts.append(f"ST_Z({col}) AS {col}_3")
                    select_sql = f"SELECT {', '.join(select_parts)} FROM coordinates LIMIT {limit};"
                    df = pd.read_sql_query(select_sql, conn)
                    df = df.dropna(axis=1, how="all")
            else:
                df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {limit};", conn)

            rich_tbl = Table(box=rich_box.SIMPLE, show_header=True,
                             header_style="dim", style="dim",
                             show_edge=False, pad_edge=True)
            for col in df.columns:
                rich_tbl.add_column(col, no_wrap=False, max_width=40)
            for _, row in df.iterrows():
                rich_tbl.add_row(*[str(v) if v is not None else "" for v in row])
            console.print(rich_tbl)

        except Exception as e:
            info(f"Error reading table {table}: {e}")

    conn.close()