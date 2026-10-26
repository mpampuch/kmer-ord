"""rDNA-miner database flags on project/run-rdna, and the rdna-db commands."""

import ast
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kmer_ord.cli.main import app
from kmer_ord.io.run_rdna import run_rdna_miner
from kmer_ord.system.env_manager import RDNA_ENV, pin_barrnap

_runner = CliRunner()

_DB_FLAGS = ("--rdna-db-dir", "--rdna-rfam-db", "--rdna-pr2-db", "--rdna-silva-db")


def _message(result) -> str:
    return (result.output or "") + str(result.exception or "")


def test_pin_barrnap_replaces_unpinned_dependency():
    pinned = pin_barrnap("dependencies:\n  - barrnap\n  - flye\n")
    assert "  - barrnap=0.9\n" in pinned
    assert "  - flye\n" in pinned


def test_pin_barrnap_raises_when_dependency_is_missing():
    with pytest.raises(RuntimeError, match="barrnap"):
        pin_barrnap("dependencies:\n  - flye\n")


def test_rdna_db_cli_does_not_import_env_manager_at_startup():
    """Importing the CLI must not require a symbol from env_manager.

    The container build copied an older env_manager.py over the repo file.
    That copy has no RDNA_DB_CHOICES, so a module-level import crashed
    `kmer-ord setup` before any command ran.
    """
    source = Path(__file__).resolve().parents[1].joinpath("src/kmer_ord/cli/rdna_db.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "kmer_ord.system.env_manager":
            raise AssertionError("rdna_db.py imports env_manager at module level")
    from kmer_ord.cli.main import app
    assert app is not None


def test_help_lists_rdna_db_flags():
    for command in ("project", "run-rdna"):
        result = _runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, _message(result)
        for flag in _DB_FLAGS:
            assert flag in result.output


def test_rdna_db_help_lists_install_and_status():
    result = _runner.invoke(app, ["rdna-db", "--help"])
    assert result.exit_code == 0, _message(result)
    assert "install" in result.output
    assert "status" in result.output


def test_setup_help_lists_rdna_db_dir():
    result = _runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0, _message(result)
    assert "--rdna-db-dir" in result.output


def test_rdna_db_install_rejects_unknown_database():
    result = _runner.invoke(app, ["rdna-db", "install", "nope"])
    assert result.exit_code != 0
    assert "nope" in _message(result).lower()


def test_rdna_db_install_command(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(list(cmd))

    monkeypatch.setattr("kmer_ord.system.env_manager.subprocess.run", fake_run)
    db_dir = tmp_path / "dbs"
    result = _runner.invoke(
        app,
        ["rdna-db", "install", "pr2", "--db-dir", str(db_dir), "--force"],
    )
    assert result.exit_code == 0, _message(result)
    assert calls == [[
        "conda", "run", "--no-capture-output", "-n", RDNA_ENV,
        "rdna-miner", "db", "install", "pr2",
        "--db-dir", str(db_dir),
        "--force",
    ]]


def test_rdna_db_install_failure_exits_nonzero(monkeypatch):
    def fake_run(cmd, check=False, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr("kmer_ord.system.env_manager.subprocess.run", fake_run)
    result = _runner.invoke(app, ["rdna-db", "install", "all"])
    assert result.exit_code != 0


def test_rdna_db_status_command(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(list(cmd))

    monkeypatch.setattr("kmer_ord.system.env_manager.subprocess.run", fake_run)
    db_dir = tmp_path / "dbs"
    result = _runner.invoke(app, ["rdna-db", "status", "--db-dir", str(db_dir)])
    assert result.exit_code == 0, _message(result)
    assert calls == [[
        "conda", "run", "--no-capture-output", "-n", RDNA_ENV,
        "rdna-miner", "db", "status",
        "--db-dir", str(db_dir),
    ]]


def test_project_rejects_missing_rdna_db_dir(tmp_path):
    result = _runner.invoke(app, [
        "project",
        "-i", str(tmp_path / "in.fa"),
        "-o", str(tmp_path / "out"),
        "--rDNA",
        "--rdna-db-dir", str(tmp_path / "missing-db"),
    ])
    assert result.exit_code != 0
    assert "does not exist" in _message(result).lower()


def test_project_warns_when_db_flags_set_without_rdna(monkeypatch, tmp_path):
    # Context conversion needs a real FASTA. Stop before the rest of the pipeline.
    monkeypatch.setattr("kmer_ord.cli.main.Runner.run", lambda self, context: None)
    db = tmp_path / "dbs"
    db.mkdir()
    fasta = tmp_path / "reads.fasta"
    fasta.write_text(">read1\nACGT\n")
    result = _runner.invoke(app, [
        "project",
        "-i", str(fasta),
        "-o", str(tmp_path / "out"),
        "--rdna-db-dir", str(db),
    ])
    assert result.exit_code == 0, _message(result)
    # Rich wraps the warning, so compare with collapsed whitespace.
    text = " ".join(result.output.split())
    assert "rDNA-miner classification skipped" in text
    assert "apply only when --rDNA is enabled" in text
    assert "rDNA-miner will not run" in text


def _capture_rdna_cmd(monkeypatch):
    calls = []

    def fake_run(cmd, capture_output=True, text=True, **kwargs):
        calls.append(list(cmd))

        class Result:
            returncode = 1
            stderr = "stopped in test"

        return Result()

    monkeypatch.setattr("kmer_ord.io.run_rdna.subprocess.run", fake_run)
    return calls


def test_run_rdna_miner_forwards_db_flags_and_force(monkeypatch, tmp_path):
    calls = _capture_rdna_cmd(monkeypatch)
    db = tmp_path / "dbs"
    rfam = tmp_path / "rfam"
    pr2 = tmp_path / "pr2"
    silva = tmp_path / "silva"
    run_rdna_miner(
        input_file=tmp_path / "reads.fasta",
        output_dir=tmp_path / "raw",
        feature_file=tmp_path / "features.tsv",
        threads=4,
        platform="ont",
        db_dir=db,
        rfam_db=rfam,
        pr2_db=pr2,
        silva_db=silva,
        force=True,
    )
    cmd = calls[0]
    assert cmd[cmd.index("--db-dir") + 1] == str(db)
    assert cmd[cmd.index("--rfam-db") + 1] == str(rfam)
    assert cmd[cmd.index("--pr2-db") + 1] == str(pr2)
    assert cmd[cmd.index("--silva-db") + 1] == str(silva)
    assert "--force" in cmd
    assert "long-read" in cmd


def test_run_rdna_miner_omits_unset_db_flags(monkeypatch, tmp_path):
    calls = _capture_rdna_cmd(monkeypatch)
    run_rdna_miner(
        input_file=tmp_path / "reads.fasta",
        output_dir=tmp_path / "raw",
        feature_file=tmp_path / "features.tsv",
    )
    cmd = calls[0]
    for flag in ("--db-dir", "--rfam-db", "--pr2-db", "--silva-db", "--force"):
        assert flag not in cmd


def test_rdna_operation_passes_force_and_db_paths(monkeypatch, tmp_path):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("kmer_ord.workflow.operations.run_rdna_miner", fake_run)

    fasta = tmp_path / "reads.fasta"
    fasta.write_text(">read1\nACGT\n")
    db = tmp_path / "dbs"
    db.mkdir()

    from kmer_ord.workflow.context import Context
    from kmer_ord.workflow.operations import RDNAMiner

    context = Context(fasta, tmp_path / "out", force=True)
    RDNAMiner(threads=3, platform="ont", db_dir=db, rfam_db=db).run(context)

    assert captured["force"] is True
    assert captured["db_dir"] == db
    assert captured["rfam_db"] == db
    assert captured["pr2_db"] is None
    assert captured["silva_db"] is None
    assert captured["threads"] == 3
    assert captured["platform"] == "ont"
