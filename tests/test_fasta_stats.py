"""Golden-output and streaming-memory tests for per-sequence FASTA stats.

The old implementation slurped each byte-range chunk (`f.read(end-start)` +
decode + parse) and pickled every (id, length, gc, at) tuple back to the
parent. Peak RAM scaled with the FASTA size times the number of workers,
which OOM-killed 58 Gbp jobs on Ibex.
"""
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from Bio.SeqIO.FastaIO import SimpleFastaParser

from kmer_ord.io.summary import calculate_nx, calculate_stats

RTOL = 1e-9
ATOL = 1e-9


def _stats_context(fasta: Path, output_dir: Path, threads: int = 1):
    """Minimal context: no FASTQ conversion, just what calculate_stats reads."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(name, subdir=None, suffix=None):
        suffix = suffix or ".dat"
        path = output_dir
        if subdir:
            path = path / subdir
            path.mkdir(parents=True, exist_ok=True)
        return path / f"{fasta.stem}_{name}{suffix}"

    artifacts = {"fasta": fasta}
    return SimpleNamespace(
        fasta=fasta,
        output_dir=output_dir,
        threads=threads,
        artifacts=artifacts,
        get=lambda name: artifacts[name],
        register=lambda name, path: artifacts.__setitem__(name, Path(path)),
        artifact_path=artifact_path,
    )


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for seq_id, seq in records:
            f.write(f">{seq_id}\n")
            if seq:
                for i in range(0, len(seq), 60):
                    f.write(seq[i : i + 60] + "\n")
            else:
                f.write("\n")


def _reference_from_fasta(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as handle:
        for title, seq in SimpleFastaParser(handle):
            seq_id = title.split()[0]
            length = len(seq)
            if length == 0:
                gc = at = 0.0
            else:
                s = seq.upper()
                gc = (s.count("G") + s.count("C")) / length * 100
                at = (s.count("A") + s.count("T")) / length * 100
            rows.append((seq_id, length, gc, at))
    return pd.DataFrame(
        rows, columns=["sequence_id", "Length", "GC_Content", "AT_Content"]
    )


def test_golden_tsv_and_overall_match_biopython(tmp_path):
    records = [
        ("read_a extra comment", "ATGCATGC"),
        ("read_b", "GGGGCCCC"),  # 100% GC
        ("empty", ""),
        ("multiline", "AA" + "T" * 70 + "GC"),  # wraps at 60 chars
        ("with_n", "ATNNNNGC"),
    ]
    fasta = tmp_path / "tiny.fasta"
    _write_fasta(fasta, records)

    ctx = _stats_context(fasta, tmp_path / "out", threads=1)
    overall_file, tsv_file = calculate_stats(ctx)

    got = pd.read_csv(tsv_file, sep="\t")
    expected = _reference_from_fasta(fasta)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        rtol=RTOL,
        atol=ATOL,
    )

    lengths = expected["Length"].to_numpy(dtype=np.int64)
    n50, _ = calculate_nx(lengths, 50)
    n90, _ = calculate_nx(lengths, 90)
    text = overall_file.read_text()
    assert f"Total nr of sequences: {len(expected)}" in text
    assert f"Total Length: {int(lengths.sum())} bp" in text
    assert f"N50: {n50} bp" in text
    assert f"N90: {n90} bp" in text
    avg_gc = float(expected["GC_Content"].mean())
    std_gc = float(expected["GC_Content"].std(ddof=1))
    assert f"Average GC Content: {avg_gc:.2f}%" in text
    assert f"GC Content Standard Deviation: {std_gc:.2f}" in text


def test_parallel_matches_serial_tsv(tmp_path):
    rng = np.random.default_rng(0)
    records = [
        (f"seq_{i}", "".join(rng.choice(list("ATGC"), size=80)))
        for i in range(80)
    ]
    fasta = tmp_path / "many.fasta"
    _write_fasta(fasta, records)

    serial = _stats_context(fasta, tmp_path / "serial", threads=1)
    parallel = _stats_context(fasta, tmp_path / "parallel", threads=4)
    _, serial_tsv = calculate_stats(serial)
    _, parallel_tsv = calculate_stats(parallel)

    assert serial_tsv.read_bytes() == parallel_tsv.read_bytes()


def test_streaming_memory_bounded(tmp_path):
    """Peak allocations must scale with one record, not the whole FASTA.

    ~20,000 x 1 kb sequences is ~20 MB on disk. The old worker did
    `data = f.read(end-start); data.decode(...)` — that alone is two full
    copies of the file (~40 MB) and cannot pass an 8 MB tracemalloc cap.
    """
    fasta = tmp_path / "big.fasta"
    seq = "ACGT" * 250  # 1000 bp
    n_records = 20_000
    with fasta.open("w") as f:
        for i in range(n_records):
            f.write(f">seq_{i}\n{seq}\n")

    ctx = _stats_context(fasta, tmp_path / "out", threads=1)
    tracemalloc.start()
    tracemalloc.reset_peak()
    overall_file, tsv_file = calculate_stats(ctx)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 8 * 1024 * 1024, (
        f"peak traced allocations {peak / 1e6:.1f} MB suggest the FASTA is "
        "being slurped rather than streamed record-by-record"
    )

    with tsv_file.open() as f:
        n_rows = sum(1 for _ in f) - 1
    assert n_rows == n_records
    assert overall_file.exists()


def test_calculate_nx_matches_sorted_scan():
    lengths = [10, 30, 20, 40]
    n50, longest = calculate_nx(lengths, 50)
    assert longest == 40
    # total=100, target=50: 40 + 30 = 70 >= 50 → N50 is 30
    assert n50 == 30
    n90, _ = calculate_nx(lengths, 90)
    # 40+30+20=90 >= 90 → N90 is 20
    assert n90 == 20


def test_empty_fasta_raises(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    ctx = _stats_context(fasta, tmp_path / "out", threads=1)
    with pytest.raises(RuntimeError, match="No sequences"):
        calculate_stats(ctx)
