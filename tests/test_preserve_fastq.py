"""FASTQ quality scores can be stored in SQLite and exported as FASTQ."""
import gzip
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from kmer_ord.cli.main import _preserve_fastq_enabled, app
from kmer_ord.dash.b2w import bin_export_format_note, write_bin_sequences
from kmer_ord.io.database import (
    create_fasta_table,
    populate_fasta_table,
    sequence_source_for_database,
)
from kmer_ord.io.sequence import load_fasta_or_convert, sequence_format

_runner = CliRunner()

FASTQ = """@read_a length=4
ACGT
+
IIII
@read_b extra
TTTT
+
####
"""

FASTA = """>read_a length=4
ACGT
>read_b extra
TTTT
"""


def _populate(path, batch_size=50000):
    conn = sqlite3.connect(":memory:")
    create_fasta_table(conn)
    populate_fasta_table(conn, path, batch_size=batch_size)
    rows = conn.execute(
        "SELECT header, full_header, sequence, qualities FROM fasta ORDER BY header"
    ).fetchall()
    conn.close()
    return rows


def test_preserve_fastq_help_is_true_or_false():
    for command in ("project", "build-db", "clustering"):
        result = _runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--preserve-fastq" in result.output
        assert "--no-preserve-fastq" not in result.output
        assert "[default: true]" in result.output
        assert "false" in result.output


def test_preserve_fastq_rejects_other_values():
    result = _runner.invoke(
        app,
        ["project", "-i", "x.fastq", "-o", "out", "--preserve-fastq", "maybe"],
    )
    assert result.exit_code != 0
    message = (result.output or str(result.exception)).lower()
    assert "maybe" in message
    assert "--no-preserve-fastq" not in (result.output or "")


def test_preserve_fastq_defaults_to_true_and_accepts_false():
    click_app = get_command(app)
    for command in ("project", "build-db", "clustering"):
        default = click_app.commands[command].make_context(
            command, ["-i", "x.fastq", "-o", "out"]
        )
        turned_off = click_app.commands[command].make_context(
            command, ["-i", "x.fastq", "-o", "out", "--preserve-fastq", "FALSE"]
        )
        assert default.params["preserve_fastq"].lower() == "true"
        assert turned_off.params["preserve_fastq"].lower() == "false"
        assert _preserve_fastq_enabled(default.params["preserve_fastq"]) is True
        assert _preserve_fastq_enabled(turned_off.params["preserve_fastq"]) is False


def test_sequence_format_includes_gzip_suffixes():
    assert sequence_format("reads.fastq") == "fastq"
    assert sequence_format("reads.fq.gz") == "fastq"
    assert sequence_format("reads.fasta.gz") == "fasta"
    assert sequence_format("reads.fa") == "fasta"
    with pytest.raises(ValueError, match="Unsupported file type"):
        sequence_format("reads.txt")


def test_plain_fastq_stores_qualities_and_short_ids(tmp_path):
    path = tmp_path / "reads.fastq"
    path.write_text(FASTQ)
    assert _populate(path, batch_size=1) == [
        ("read_a", "read_a length=4", "ACGT", "IIII"),
        ("read_b", "read_b extra", "TTTT", "####"),
    ]


def test_gzipped_fastq_stores_qualities(tmp_path):
    path = tmp_path / "reads.fastq.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(FASTQ)
    rows = _populate(path)
    assert rows[0] == ("read_a", "read_a length=4", "ACGT", "IIII")
    assert rows[1][3] == "####"


def test_fasta_stores_null_qualities(tmp_path):
    path = tmp_path / "reads.fasta"
    path.write_text(FASTA)
    rows = _populate(path)
    assert [row[3] for row in rows] == [None, None]
    assert rows[0][:3] == ("read_a", "read_a length=4", "ACGT")


def test_preserve_false_stores_null_qualities_for_the_same_fastq(tmp_path):
    fastq = tmp_path / "reads.fastq"
    fasta = tmp_path / "reads.fastq.fasta"
    fastq.write_text(FASTQ)
    fasta.write_text(FASTA)

    dropped = sequence_source_for_database(fastq, fasta, preserve_fastq=False)
    assert dropped == fasta
    assert all(row[3] is None for row in _populate(dropped))

    kept = sequence_source_for_database(fastq, fasta, preserve_fastq=True)
    assert kept == fastq
    assert _populate(kept)[0][3] == "IIII"


def test_fasta_input_has_no_qualities_either_way(tmp_path):
    fasta = tmp_path / "reads.fasta"
    canonical = tmp_path / "canonical.fasta"
    fasta.write_text(FASTA)
    canonical.write_text(FASTA)
    assert sequence_source_for_database(fasta, canonical, True) == canonical
    assert sequence_source_for_database(fasta, canonical, False) == canonical
    assert all(row[3] is None for row in _populate(canonical))


def test_mismatched_quality_length_raises(tmp_path):
    path = tmp_path / "bad.fastq"
    path.write_text("@bad\nACGT\n+\nII\n")
    with pytest.raises(ValueError, match="bad"):
        _populate(path)


def test_export_fastq_uses_the_full_header(tmp_path):
    records = pd.DataFrame(
        {
            "header": ["read_a"],
            "full_header": ["read_a length=4"],
            "sequence": ["ACGT"],
            "qualities": ["IIII"],
        }
    )
    out = Path(write_bin_sequences(records, str(tmp_path), "binA"))
    assert out.name == "binA.fastq"
    assert not (tmp_path / "binA.fasta").exists()
    assert out.read_text() == "@read_a length=4\nACGT\n+\nIIII\n"
    assert bin_export_format_note([str(out)]) == "FASTQ"


def test_export_fasta_uses_header_when_full_header_is_absent(tmp_path):
    records = pd.DataFrame({"header": ["read_a"], "sequence": ["ACGT"]})
    out = Path(write_bin_sequences(records, str(tmp_path), "binA"))
    assert out.name == "binA.fasta"
    assert not (tmp_path / "binA.fastq").exists()
    assert out.read_text() == ">read_a\nACGT\n"
    assert bin_export_format_note([str(out)]) == "FASTA"


def test_export_does_not_mix_formats_when_a_quality_is_missing(tmp_path):
    records = pd.DataFrame(
        {
            "header": ["read_a", "read_b"],
            "full_header": ["read_a length=4", None],
            "sequence": ["ACGT", "TTTT"],
            "qualities": ["IIII", None],
        }
    )
    out = Path(write_bin_sequences(records, str(tmp_path), "binA"))
    assert out.name == "binA.fasta"
    assert not (tmp_path / "binA.fastq").exists()
    assert out.read_text() == ">read_a length=4\nACGT\n>read_b\nTTTT\n"


def test_canonical_fasta_from_gzipped_fastq_drops_qualities(tmp_path):
    source = tmp_path / "reads.fastq.gz"
    with gzip.open(source, "wt") as handle:
        handle.write(FASTQ)
    fasta = load_fasta_or_convert(source, tmp_path / "work", use_seqkit=False)
    text = fasta.read_text()
    assert ">read_a" in text
    assert "IIII" not in text
