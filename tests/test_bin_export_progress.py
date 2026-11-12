"""Bin export writes every row, logs stages as it goes, and rejects overlap."""
import threading
from pathlib import Path

import pandas as pd

from kmer_ord.dash import b2w
from kmer_ord.dash.b2w import (
    EXPORT_ALREADY_RUNNING,
    EXPORT_STARTED,
    _should_log_fasta_chunk,
    export_progress_history,
    export_progress_snapshot,
    reset_export_progress,
    start_bin_export,
    wait_for_export,
    write_bin_sequences,
)
from kmer_ord.utils.logging_utils import close_run_log, enable_run_log


def _bin_entry():
    return {
        "bin_name": "binA",
        "coordinate_system": "umap",
        "polygon": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        "filters": {"gc": {"min": 0.2, "max": None, "cat": []}},
    }


class _Cursor:
    def execute(self, *args, **kwargs):
        return self

    def fetchall(self):
        return [(0, "header"), (1, "sequence")]


class _Connection:
    def cursor(self):
        return _Cursor()

    def close(self):
        return None


def _install_fake_reads(monkeypatch):
    def fake_spatial(db_path, coordinate_systems, polygons, feature_col=None, filter_values=None):
        return pd.DataFrame({"header": ["r1", "r2"]})

    def fake_read_sql(query, conn, params=None):
        batch = list(params or [])
        return pd.DataFrame({
            "header": batch,
            "sequence": ["ACGT"] * len(batch),
        })

    monkeypatch.setattr(b2w, "efficient_spatialite_export", fake_spatial)
    monkeypatch.setattr(b2w, "get_connection", lambda db_path: _Connection())
    monkeypatch.setattr(b2w.pd, "read_sql_query", fake_read_sql)


def test_write_bin_sequences_writes_every_row(tmp_path):
    n = 1000
    records = pd.DataFrame({
        "header": [f"r{i}" for i in range(n)],
        "sequence": ["ACGT"] * n,
    })
    out = Path(write_bin_sequences(records, str(tmp_path), "bulk"))
    text = out.read_text()
    assert text.count(">") == n
    assert text.startswith(">r0\nACGT\n")
    assert text.endswith(f">r{n - 1}\nACGT\n")


def test_fasta_chunk_logs_are_sampled_when_there_are_many_chunks():
    assert _should_log_fasta_chunk(1, 3)
    assert _should_log_fasta_chunk(2, 3)
    assert _should_log_fasta_chunk(3, 3)
    assert _should_log_fasta_chunk(1, 25)
    assert not _should_log_fasta_chunk(2, 25)
    assert _should_log_fasta_chunk(10, 25)
    assert not _should_log_fasta_chunk(11, 25)
    assert _should_log_fasta_chunk(20, 25)
    assert _should_log_fasta_chunk(25, 25)


def test_export_logs_stages_and_records_progress(tmp_path, monkeypatch):
    _install_fake_reads(monkeypatch)
    reset_export_progress()
    log_path = tmp_path / "kmer-ord.log"
    out_dir = tmp_path / "bins"
    enable_run_log(log_path)
    try:
        message = start_bin_export([_bin_entry()], "unused.db", str(out_dir), chunk_size=1)
        assert message == EXPORT_STARTED
        wait_for_export()

        snapshot = export_progress_snapshot()
        history = "\n".join(export_progress_history())
        log_text = log_path.read_text(encoding="utf-8")
    finally:
        close_run_log()
        reset_export_progress()

    assert snapshot["running"] is False
    assert snapshot["stage"] == "complete"
    assert snapshot["bin_index"] == 1
    assert snapshot["bin_total"] == 1
    assert "exported" in snapshot["message"].lower()

    for blob in (history, log_text):
        assert blob.index("Export started") < blob.index("polygon points")
        assert blob.index("polygon points") < blob.index("spatial query finished")
        assert blob.index("spatial query finished") < blob.index("wrote CSV")
        assert blob.index("wrote CSV") < blob.index("fasta chunk 1/2")
        assert blob.index("fasta chunk 1/2") < blob.index("fasta chunk 2/2")
        assert blob.index("fasta chunk 2/2") < blob.index("Export complete:")

    fasta = (out_dir / "binA.fasta").read_text()
    assert ">r1\nACGT\n" in fasta
    assert ">r2\nACGT\n" in fasta
    csv_text = (out_dir / "binA.csv").read_text()
    assert "r1" in csv_text and "r2" in csv_text


def test_overlapping_export_is_rejected(monkeypatch):
    reset_export_progress()
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def blocked(bins_data, db_path, output_dir, chunk_size=900):
        calls["n"] += 1
        started.set()
        assert release.wait(5)
        return "Bins exported to out (no sequence files; using full headers when available)."

    monkeypatch.setattr(b2w, "run_bin_export", blocked)
    try:
        first = start_bin_export([_bin_entry()], "unused.db", "out")
        assert first == EXPORT_STARTED
        assert started.wait(2)
        assert export_progress_snapshot()["running"] is True

        second = start_bin_export([_bin_entry()], "unused.db", "out")
        assert second == EXPORT_ALREADY_RUNNING
        assert calls["n"] == 1
        assert export_progress_snapshot()["running"] is True
    finally:
        release.set()
        wait_for_export()
        reset_export_progress()


def test_export_failure_is_logged_and_clears_running(tmp_path, monkeypatch):
    def boom(db_path, coordinate_systems, polygons, feature_col=None, filter_values=None):
        raise RuntimeError("spatialite down")

    monkeypatch.setattr(b2w, "efficient_spatialite_export", boom)
    reset_export_progress()
    log_path = tmp_path / "kmer-ord.log"
    enable_run_log(log_path)
    try:
        message = start_bin_export([_bin_entry()], "unused.db", str(tmp_path / "bins"))
        assert message == EXPORT_STARTED
        wait_for_export()
        snapshot = export_progress_snapshot()
        log_text = log_path.read_text(encoding="utf-8")
    finally:
        close_run_log()
        reset_export_progress()

    assert snapshot["running"] is False
    assert snapshot["stage"] == "error"
    assert "spatialite down" in snapshot["message"]
    assert snapshot["error"] and "Traceback" in snapshot["error"]
    assert "Export failed: spatialite down" in log_text
    assert "Traceback" in log_text
