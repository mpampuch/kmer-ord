"""The run log is a flushed plain-text copy of the shared console."""
from typer.testing import CliRunner

from kmer_ord.cli.main import app
from kmer_ord.utils.logging_utils import (
    close_run_log,
    console,
    divider,
    enable_run_log,
    info,
    section,
    warn,
)

_runner = CliRunner()


def test_helpers_and_direct_calls_are_flushed_to_the_file(tmp_path):
    path = tmp_path / "out" / "kmer-ord.log"
    enable_run_log(path)

    info("alpha-info")
    warn("alpha-warn")
    section("alpha-section")
    divider()
    console.print("alpha-print", markup=False, highlight=False)
    console.rule("alpha-rule", style="none")

    # Read while the handle is still open: the line must already be on disk.
    text = path.read_text(encoding="utf-8")
    assert "alpha-info" in text
    assert "WARNING  alpha-warn" in text
    assert text.count("alpha-section") == 1
    assert "─" * 20 in text
    assert "alpha-print" in text
    assert text.count("alpha-rule") == 1


def test_second_write_appends(tmp_path):
    path = tmp_path / "kmer-ord.log"
    enable_run_log(path)
    info("first-line")
    info("second-line")
    close_run_log()

    enable_run_log(path)
    info("third-line")

    text = path.read_text(encoding="utf-8")
    assert text.index("first-line") < text.index("second-line") < text.index("third-line")


def test_capture_still_returns_section_text(tmp_path):
    with console.capture() as capture:
        section("matrix preprocessing")
    assert "matrix preprocessing" in capture.get()

    enable_run_log(tmp_path / "kmer-ord.log")
    with console.capture() as capture:
        section("matrix preprocessing")
    assert "matrix preprocessing" in capture.get()
    assert "matrix preprocessing" in (tmp_path / "kmer-ord.log").read_text(encoding="utf-8")


def test_close_run_log_stops_further_writes(tmp_path):
    path = tmp_path / "kmer-ord.log"
    enable_run_log(path)
    info("kept-line")
    close_run_log()
    info("dropped-line")

    text = path.read_text(encoding="utf-8")
    assert "kept-line" in text
    assert "dropped-line" not in text


def test_fasta_stats_writes_the_same_message_to_terminal_and_log(tmp_path):
    fasta = tmp_path / "reads.fa"
    fasta.write_text(">r1\nATGC\n")
    out = tmp_path / "out"

    result = _runner.invoke(app, ["fasta-stats", "-i", str(fasta), "-o", str(out)])

    assert result.exit_code == 0, result.output
    log = (out / "kmer-ord.log").read_text(encoding="utf-8")
    assert "Stats calculated" in result.output
    assert "Stats calculated" in log
