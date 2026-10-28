"""class_combined keeps Tiara's first stage and replaces organelle with the second."""
from contextlib import nullcontext

import pandas as pd

from kmer_ord.io.run_tiara import add_class_combined
from kmer_ord.workflow.operations import FeatureMerge


def _classes(rows):
    return pd.DataFrame(rows, columns=["class_fst_stage", "class_snd_stage"])


def test_non_organelle_labels_are_kept_when_second_stage_is_null():
    result = add_class_combined(_classes([
        ["eukarya", None],
        ["bacteria", None],
        ["archaea", None],
    ]))

    assert result["class_combined"].tolist() == ["eukarya", "bacteria", "archaea"]


def test_organelle_is_replaced_by_a_real_second_stage_call():
    result = add_class_combined(_classes([
        ["organelle", "plastid"],
        ["organelle", "mitochondrion"],
        ["Organelle", "plastid"],
    ]))

    assert result["class_combined"].tolist() == ["plastid", "mitochondrion", "plastid"]


def test_organelle_without_a_second_stage_call_stays_organelle():
    result = add_class_combined(_classes([
        ["organelle", None],
        ["organelle", ""],
        ["organelle", "   "],
        ["organelle", "n/a"],
        ["organelle", "N/A"],
    ]))

    assert result["class_combined"].tolist() == ["organelle"] * 5


def test_unknown_second_stage_replaces_organelle():
    result = add_class_combined(_classes([
        ["organelle", "unknown"],
        ["unknown", None],
    ]))

    assert result["class_combined"].tolist() == ["unknown", "unknown"]


def test_non_organelle_rows_ignore_a_second_stage_value():
    result = add_class_combined(_classes([["bacteria", "plastid"]]))

    assert result["class_combined"].tolist() == ["bacteria"]


def test_existing_class_combined_is_recomputed_from_the_two_stages():
    frame = _classes([["organelle", "mitochondrion"]])
    frame["class_combined"] = ["stale"]

    result = add_class_combined(frame)

    assert result["class_combined"].tolist() == ["mitochondrion"]
    assert frame["class_combined"].tolist() == ["stale"]


def test_missing_stage_column_does_not_add_class_combined():
    fst_only = pd.DataFrame({"class_fst_stage": ["eukarya"]})
    snd_only = pd.DataFrame({"class_snd_stage": ["plastid"]})

    assert "class_combined" not in add_class_combined(fst_only).columns
    assert "class_combined" not in add_class_combined(snd_only).columns
    assert "class_combined" not in fst_only.columns
    assert "class_combined" not in snd_only.columns


def test_source_stage_columns_are_left_unchanged():
    frame = _classes([["organelle", "plastid"], ["eukarya", "n/a"]])

    result = add_class_combined(frame)

    assert result["class_fst_stage"].tolist() == ["organelle", "eukarya"]
    assert result["class_snd_stage"].tolist() == ["plastid", "n/a"]


class _StubContext:
    """Just enough of Context for FeatureMerge: paths, a timer, and register."""

    def __init__(self, output_dir, artifacts):
        self.output_dir = output_dir
        self.artifacts = artifacts
        self.registered = {}

    def benchmark_timer(self, label, **kwargs):
        return nullcontext()

    def get(self, name):
        if name not in self.artifacts:
            raise ValueError(f"Artifact '{name}' not found in context.")
        return self.artifacts[name]

    def register(self, name, path):
        self.registered[name] = path


def test_feature_merge_adds_class_combined_from_the_two_tiara_columns(tmp_path):
    kmer = tmp_path / "kmer.tsv"
    summary = tmp_path / "summary.tsv"
    tiara = tmp_path / "tiara.tsv"
    kmer.write_text("sequence_id\tkmer_count\nr1\t1\nr2\t2\nr3\t3\n", encoding="utf-8")
    summary.write_text("sequence_id\tlength\nr1\t10\nr2\t20\nr3\t30\n", encoding="utf-8")
    # Cached Tiara output: both stages, a stale combined column, and literal n/a.
    tiara.write_text(
        "sequence_id\tclass_fst_stage\tclass_snd_stage\tclass_combined\n"
        "r1\teukarya\tn/a\tstale\n"
        "r2\torganelle\tplastid\tstale\n"
        "r3\torganelle\tn/a\tstale\n",
        encoding="utf-8",
    )
    context = _StubContext(tmp_path, {
        "kmer_metrics": kmer,
        "summary_per_sequence": summary,
        "tiara": tiara,
    })

    FeatureMerge().run(context)

    merged = pd.read_csv(tmp_path / "features" / "merged_features.tsv", sep="\t")
    by_id = merged.set_index("sequence_id")
    assert by_id.loc["r1", "class_combined"] == "eukarya"
    assert by_id.loc["r2", "class_combined"] == "plastid"
    assert by_id.loc["r2", "class_fst_stage"] == "organelle"
    assert by_id.loc["r3", "class_combined"] == "organelle"
    assert context.registered["merged_features"] == tmp_path / "features" / "merged_features.tsv"
