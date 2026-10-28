"""rDNA-miner and Tiara output is summarized unless --verbose is set."""
import subprocess

import pytest
from typer.testing import CliRunner

from kmer_ord.cli.main import app
from kmer_ord.io.run_rdna import run_rdna_miner
from kmer_ord.io.run_tiara import run_tiara
from kmer_ord.utils.logging_utils import (
    close_run_log,
    console,
    enable_run_log,
    run_logged_command,
)

_runner = CliRunner()

_STAGES = [
    "[1/7] Run Barrnap",
    "[2/7] Filter putative rDNA reads",
    "[3/7] Assemble rDNA reads with Flye",
    "[4/7] Annotate rDNA operons with Rfam/cmscan",
    "[5/7] Assign taxonomy with DECIPHER",
    "[6/7] Map reads to rDNA assembly",
    "[7/7] Complete read-level taxonomy assignment",
]


def _patch_popen(monkeypatch, chunks, code=0):
    def fake_popen(cmd, stdout=None, stderr=None, text=None, bufsize=None, env=None):
        assert env["PYTHONUNBUFFERED"] == "1"
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.STDOUT

        class Proc:
            def __init__(self):
                self.stdout = iter(chunks)

            def wait(self):
                return code

        return Proc()

    monkeypatch.setattr("kmer_ord.utils.logging_utils.subprocess.Popen", fake_popen)


def _rdna_chunks():
    chunks = [f"{stage} ────────\n" for stage in _STAGES]
    chunks.append("    Running: barrnap --kingdom bac --threads 4 reads.fasta\n")
    chunks.append("    Input FASTA exists: True, path: /tmp/reads.fasta\n")
    chunks.append("\r 10%|█         | 1/10\r 20%|██        | 2/10")
    return chunks


def _write_taxonomy(output_dir):
    path = output_dir / "taxonomy" / "reads_taxa_ssu_reads.tsv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "QNAME\tgenus_confidence\tdomain_clean\tsupergroup\tdivision\t"
        "subdivision\tclass\torder\tfamily\tgenus\n"
        "read_1 extra\t0.9\tEukaryota\tSAR\tAlveolata\tCiliophora\t"
        "Spirotrichea\tSporadotrichida\tOxytrichidae\tSterkiella\n",
        encoding="utf-8",
    )


def _run_rdna(tmp_path, verbose=False):
    output_dir = tmp_path / "raw"
    _write_taxonomy(output_dir)
    return run_rdna_miner(
        input_file=tmp_path / "reads.fasta",
        output_dir=output_dir,
        feature_file=tmp_path / "features.tsv",
        verbose=verbose,
    )


def test_progress_redraw_is_not_logged(monkeypatch, tmp_path):
    _patch_popen(monkeypatch, [
        "kept line\r\n",
        "\r 10%|█         | 1/10\r 20%|██        | 2/10",
    ])
    path = tmp_path / "kmer-ord.log"
    enable_run_log(path)
    try:
        with console.capture() as capture:
            code = run_logged_command(["echo"], verbose=True)
        text = capture.get()
        logged = path.read_text(encoding="utf-8")
    finally:
        close_run_log()

    assert code == 0
    assert "kept line" in text
    assert "kept line" in logged
    assert "20%" not in text
    assert "20%" not in logged


def test_rdna_summary_keeps_stage_lines_only(monkeypatch, tmp_path):
    _patch_popen(monkeypatch, _rdna_chunks())
    path = tmp_path / "kmer-ord.log"
    enable_run_log(path)
    try:
        with console.capture() as capture:
            result = _run_rdna(tmp_path)
        text = capture.get()
        logged = path.read_text(encoding="utf-8")
    finally:
        close_run_log()

    assert result == tmp_path / "features.tsv"
    for stage in _STAGES:
        assert text.count(stage) == 1
        assert logged.count(stage) == 1
    assert "Running: barrnap" not in text
    assert "Running: barrnap" not in logged
    assert "Input FASTA exists" not in logged
    assert "20%" not in logged


def test_rdna_verbose_prints_the_command_line(monkeypatch, tmp_path):
    _patch_popen(monkeypatch, _rdna_chunks())
    path = tmp_path / "kmer-ord.log"
    enable_run_log(path)
    try:
        with console.capture() as capture:
            _run_rdna(tmp_path, verbose=True)
        text = capture.get()
        logged = path.read_text(encoding="utf-8")
    finally:
        close_run_log()

    assert "Running: barrnap --kingdom bac" in text
    assert "Running: barrnap --kingdom bac" in logged
    assert "20%" not in logged


def test_rdna_failure_reports_the_last_line_and_skips_taxonomy(monkeypatch, tmp_path):
    _patch_popen(monkeypatch, ["flye failed to assemble\n"], code=1)

    def fail_if_taxonomy_is_read(*args, **kwargs):
        raise AssertionError("taxonomy table was read")

    monkeypatch.setattr("kmer_ord.io.run_rdna.pd.read_csv", fail_if_taxonomy_is_read)
    with console.capture() as capture:
        result = run_rdna_miner(
            input_file=tmp_path / "reads.fasta",
            output_dir=tmp_path / "raw",
            feature_file=tmp_path / "features.tsv",
        )

    assert result is None
    text = " ".join(capture.get().split())
    assert "rDNA-miner failed; continuing without rDNA features. flye failed to assemble" in text


def test_rdna_failure_does_not_repeat_a_shown_alert(monkeypatch, tmp_path):
    _patch_popen(
        monkeypatch,
        ["Traceback (most recent call last):\n", "RuntimeError: barrnap missing\n"],
        code=1,
    )
    with console.capture() as capture:
        result = run_rdna_miner(
            input_file=tmp_path / "reads.fasta",
            output_dir=tmp_path / "raw",
            feature_file=tmp_path / "features.tsv",
        )

    assert result is None
    text = capture.get()
    assert text.count("RuntimeError: barrnap missing") == 1
    assert "features. RuntimeError" not in text


def _tiara_chunks():
    return [
        "Calculating first stage sequence representations took: 0.400000 seconds\n",
        "Performing first stage of classification.\n",
        "Done\n",
        "Calculating second stage sequence representations took: 0.200000 seconds\n",
        "Performing second stage of classification.\n",
        "Done\n",
        "Classification took 1.5 seconds.\n",
        "12.0 sequences per second.\n",
        "Classification done.\n",
        "First iteration statistics:\n",
        "\tbacteria: 1\n",
        "Second iteration statistics:\n",
        "\tmitochondrion: 1\n",
        "Output saved to somewhere.\n",
        "\r 50%|█████     | 1/2\r",
    ]


def _write_tiara_output(output):
    output.parent.mkdir(parents=True)
    output.write_text(
        "sequence_id\tclass_fst_stage\tclass_snd_stage\n"
        "read_1 length=10\tbacteria\tn/a\n",
        encoding="utf-8",
    )
    log = output.parent / f"log_{output.name}"
    log.write_text(
        "First iteration statistics:\n\tbacteria: 1\nModels used:\n\tfirst_nnet_kmer_6.pkl\n",
        encoding="utf-8",
    )
    return log


def test_tiara_summary_shows_statistics_and_rewrites_ids(monkeypatch, tmp_path):
    output = tmp_path / "tiara" / "reads.tsv"
    _write_tiara_output(output)
    _patch_popen(monkeypatch, _tiara_chunks())
    path = tmp_path / "kmer-ord.log"
    enable_run_log(path)
    try:
        with console.capture() as capture:
            run_tiara(tmp_path / "reads.fasta", output)
        text = capture.get()
        logged = path.read_text(encoding="utf-8")
    finally:
        close_run_log()

    assert "Calculating first stage sequence representations took" in text
    assert "Performing first stage of classification." in text
    assert "First stage done" in text
    assert "Calculating second stage sequence representations took" in logged
    assert "Performing second stage of classification." in logged
    assert "Second stage done" in logged
    assert "Classification took 1.5 seconds." in text
    assert "bacteria: 1" in logged
    assert "mitochondrion: 1" in logged
    assert "sequences per second" not in text
    assert "Output saved" not in text
    assert "Models used" not in text
    assert "50%" not in logged
    rewritten = output.read_text(encoding="utf-8")
    assert "read_1\t" in rewritten
    assert "length=" not in rewritten
    header, row = (line.split("\t") for line in rewritten.splitlines())
    assert row[0] == "read_1"
    assert row[header.index("class_fst_stage")] == "bacteria"
    assert row[header.index("class_combined")] == "bacteria"


def test_tiara_verbose_prints_the_log_file(monkeypatch, tmp_path):
    output = tmp_path / "tiara" / "reads.tsv"
    _write_tiara_output(output)
    _patch_popen(monkeypatch, _tiara_chunks())
    with console.capture() as capture:
        run_tiara(tmp_path / "reads.fasta", output, verbose=True)

    text = capture.get()
    assert "sequences per second" in text
    assert "Models used" in text
    assert "first_nnet_kmer_6.pkl" in text
    assert "50%" not in text


def test_tiara_failure_raises_after_streaming(monkeypatch, tmp_path):
    _patch_popen(monkeypatch, ["Classification took 1.5 seconds.\n"], code=1)
    output = tmp_path / "tiara" / "reads.tsv"
    with console.capture() as capture:
        with pytest.raises(subprocess.CalledProcessError):
            run_tiara(tmp_path / "reads.fasta", output)
    assert "Classification took 1.5 seconds." in capture.get()


def test_help_lists_verbose():
    for command in ("project", "run-tiara", "run-rdna"):
        result = _runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--verbose" in result.output
