"""Screen spec strings must name a method. A bare range is otherwise ignored."""
import pytest

from kmer_ord.dr.screen_grid import (
    named_screen_methods,
    resolve_method_grid,
    validate_screen_specs,
)


def test_bare_range_raises_and_names_flag():
    with pytest.raises(ValueError, match=r"--screen_range2"):
        validate_screen_specs(range2=["0.1,1.0"])


def test_valid_localmap_and_all_ranges():
    validate_screen_specs(range2=["localmap=0.1,1.0"])
    validate_screen_specs(range2=["all=0.1,1.0"])
    validate_screen_specs(range2=["localmap=0.1,0.5", "all=0.2,1.0"])
    validate_screen_specs(values1=["localmap=10,25,50"])


def test_unknown_method_empty_value_and_unordered_range():
    with pytest.raises(ValueError, match="pca"):
        validate_screen_specs(range2=["pca=0.1,1.0"])
    with pytest.raises(ValueError, match=r"--screen_range1"):
        validate_screen_specs(range1=["umap="])
    with pytest.raises(ValueError, match="min < max"):
        validate_screen_specs(range1=["umap=10,5"])


def test_resolve_method_grid_rejects_bare_range():
    with pytest.raises(ValueError, match=r"method=min,max"):
        resolve_method_grid("localmap", range2=["0.1,1.0"], grid_spec="5x4")


def test_valid_range_still_resolves():
    _axis1, axis2 = resolve_method_grid(
        "localmap", range2=["localmap=0.1,1.0"], grid_spec="5x4"
    )
    assert axis2 == [0.1, 0.2, 0.5, 1.0]


def test_grid_size_survives_nice_rounding_collisions():
    """5x5 must run 25 combinations. 1-2-5 snapping used to collapse an axis to 4.

    UMAP min_dist on [0, 1] maps both 0.75 and 1 onto 1. PaCMAP FP_ratio on
    [0.1, 1] has only four 1-2-5 landmarks. localmap=3x3 stays 9.
    """
    grid = "all=5x5;localmap=3x3"
    ranges = ["localmap=0.1,1.0;pacmap=0.1,1.0"]
    for method in ("umap", "tsne", "trimap", "pacmap"):
        axis1, axis2 = resolve_method_grid(method, range2=ranges, grid_spec=grid)
        assert len(axis1) == 5, method
        assert len(axis2) == 5, method
        assert len(set(axis1)) == 5, method
        assert len(set(axis2)) == 5, method

    axis1, axis2 = resolve_method_grid("localmap", range2=ranges, grid_spec=grid)
    assert len(axis1) == 3
    assert len(axis2) == 3
    assert len(set(axis1)) == 3
    assert len(set(axis2)) == 3


@pytest.mark.parametrize(
    "command, base_args",
    [
        ("project", ["-i", "x.fa", "-o", "out"]),
        ("cluster", ["-i", "x.fa", "-o", "out"]),
        ("dr", ["-i", "x.tsv", "-o", "out", "-m", "umap"]),
    ],
)
def test_cli_bare_screen_range2_fails_before_work(command, base_args):
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    result = CliRunner().invoke(
        app, [command, *base_args, "--screen_range2", "0.1,1.0"]
    )
    assert result.exit_code != 0
    message = result.output or str(result.exception)
    assert "--screen_range2" in message
    assert "method=min,max" in message


def test_semicolon_ranges_match_separate_flags():
    combined_local = resolve_method_grid(
        "localmap",
        range1=["localmap=10,100;pacmap=20,80"],
        range2=["localmap=0.1,1.0;pacmap=0.5,2.0"],
        grid_spec="5x4",
    )
    separate_local = resolve_method_grid(
        "localmap",
        range1=["localmap=10,100"],
        range2=["localmap=0.1,1.0"],
        grid_spec="5x4",
    )
    combined_pac = resolve_method_grid(
        "pacmap",
        range1=["localmap=10,100;pacmap=20,80"],
        range2=["localmap=0.1,1.0; pacmap=0.5,2.0"],
        grid_spec="5x4",
    )
    separate_pac = resolve_method_grid(
        "pacmap",
        range1=["pacmap=20,80"],
        range2=["pacmap=0.5,2.0"],
        grid_spec="5x4",
    )
    assert combined_local == separate_local
    assert combined_pac == separate_pac


def test_semicolon_range_named_entry_overrides_all_and_later_flag_wins():
    overridden = resolve_method_grid(
        "localmap",
        range2=["all=0.1,1.0;localmap=0.5,2.0"],
        grid_spec="5x4",
    )
    named = resolve_method_grid(
        "localmap", range2=["localmap=0.5,2.0"], grid_spec="5x4"
    )
    shared = resolve_method_grid(
        "pacmap",
        range2=["all=0.1,1.0;localmap=0.5,2.0"],
        grid_spec="5x4",
    )
    shared_only = resolve_method_grid(
        "pacmap", range2=["all=0.1,1.0"], grid_spec="5x4"
    )
    later_flag = resolve_method_grid(
        "pacmap",
        range2=["localmap=0.1,1.0;pacmap=0.5,2.0", "pacmap=0.1,1.0"],
        grid_spec="5x4",
    )
    assert overridden == named
    assert shared == shared_only
    assert later_flag == shared_only


def test_semicolon_range_rejects_bad_piece():
    with pytest.raises(ValueError, match="min < max"):
        validate_screen_specs(range2=["localmap=0.1,1.0;pacmap=1.0,0.1"])
    with pytest.raises(ValueError, match="pca"):
        validate_screen_specs(range1=["localmap=10,100;pca=10,50"])


def test_semicolon_values_match_separate_flags():
    combined_local = resolve_method_grid(
        "localmap",
        values1=["localmap=10,25,50;pacmap=10,50,100"],
        values2=["localmap=0.1,0.5,1.0;pacmap=0.5,1.0,2.0"],
    )
    separate_local = resolve_method_grid(
        "localmap",
        values1=["localmap=10,25,50"],
        values2=["localmap=0.1,0.5,1.0"],
    )
    combined_pac = resolve_method_grid(
        "pacmap",
        values1=["localmap=10,25,50;pacmap=10,50,100"],
        values2=["localmap=0.1,0.5,1.0; pacmap=0.5,1.0,2.0"],
    )
    separate_pac = resolve_method_grid(
        "pacmap",
        values1=["pacmap=10,50,100"],
        values2=["pacmap=0.5,1.0,2.0"],
    )
    assert combined_local == separate_local
    assert combined_pac == separate_pac


def test_semicolon_values_named_entry_overrides_all_and_later_flag_wins():
    overridden = resolve_method_grid(
        "localmap",
        values2=["all=0.1,0.5,1.0;localmap=0.5,1.0,2.0"],
    )
    named = resolve_method_grid("localmap", values2=["localmap=0.5,1.0,2.0"])
    shared = resolve_method_grid(
        "pacmap",
        values2=["all=0.1,0.5,1.0;localmap=0.5,1.0,2.0"],
    )
    shared_only = resolve_method_grid("pacmap", values2=["all=0.1,0.5,1.0"])
    later_flag = resolve_method_grid(
        "pacmap",
        values2=["localmap=0.1,1.0;pacmap=0.5,2.0", "pacmap=0.1,0.5,1.0"],
    )
    assert overridden == named
    assert shared == shared_only
    assert later_flag == shared_only


def test_semicolon_values_reject_bad_piece():
    with pytest.raises(ValueError, match="pca"):
        validate_screen_specs(values1=["umap=10,20;pca=5,10"])
    with pytest.raises(ValueError, match=r"--screen_values2"):
        validate_screen_specs(values2=["localmap=0.1,1.0;0.5,2.0"])


def test_bare_grid_applies_to_every_method_and_equals_all():
    for method in ("localmap", "pacmap", "umap"):
        assert resolve_method_grid(method, grid_spec="5x4") == resolve_method_grid(
            method, grid_spec="all=5x4"
        )
        assert resolve_method_grid(method, grid_spec="5") == resolve_method_grid(
            method, grid_spec="all=5"
        )
        _, axis2 = resolve_method_grid(method, grid_spec="5")
        assert axis2 is None


def test_named_grid_sizes_only_those_methods():
    spec = "localmap=5x4;pacmap=3x3"
    assert resolve_method_grid("localmap", grid_spec=spec) == resolve_method_grid(
        "localmap", grid_spec="5x4"
    )
    assert resolve_method_grid("pacmap", grid_spec=spec) == resolve_method_grid(
        "pacmap", grid_spec="3x3"
    )
    assert resolve_method_grid("umap", grid_spec=spec) == resolve_method_grid("umap")
    assert resolve_method_grid(
        "localmap", grid_spec="LocalMAP=5x4; PaCMAP=3x3"
    ) == resolve_method_grid("localmap", grid_spec="5x4")


def test_named_grid_overrides_all_including_one_dimensional():
    spec = "all=5x4;localmap=5"
    local_axis1, local_axis2 = resolve_method_grid("localmap", grid_spec=spec)
    bare_axis1, bare_axis2 = resolve_method_grid("localmap", grid_spec="5")
    assert local_axis2 is None
    assert bare_axis2 is None
    assert local_axis1 == bare_axis1
    assert resolve_method_grid("pacmap", grid_spec=spec) == resolve_method_grid(
        "pacmap", grid_spec="5x4"
    )
    assert resolve_method_grid("umap", grid_spec=spec) == resolve_method_grid(
        "umap", grid_spec="5x4"
    )


def test_named_grid_does_not_hide_a_range_on_another_method():
    spec = "localmap=5x4"
    pac = resolve_method_grid("pacmap", range2=["pacmap=0.5,2.0"], grid_spec=spec)
    ranged = resolve_method_grid("pacmap", range2=["pacmap=0.5,2.0"])
    assert pac == ranged
    assert resolve_method_grid("umap", grid_spec=spec) == resolve_method_grid("umap")


def test_grid_rejects_mixed_bare_named_unknown_and_bad_size():
    with pytest.raises(ValueError, match="bare size"):
        validate_screen_specs(grid="5x4;localmap=3x3")
    with pytest.raises(ValueError, match="bare size"):
        resolve_method_grid("localmap", grid_spec="5x4;3x3")
    with pytest.raises(ValueError, match="pca"):
        validate_screen_specs(grid="pca=5x4")
    with pytest.raises(ValueError, match="greater than zero"):
        validate_screen_specs(grid="localmap=0x4")
    with pytest.raises(ValueError, match="NxM"):
        validate_screen_specs(grid="localmap=5x4x3")


@pytest.mark.parametrize(
    "command, base_args",
    [
        ("project", ["-i", "x.fa", "-o", "out"]),
        ("cluster", ["-i", "x.fa", "-o", "out"]),
        ("dr", ["-i", "x.tsv", "-o", "out", "-m", "umap"]),
    ],
)
def test_cli_mixed_grid_fails_before_work(command, base_args):
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    result = CliRunner().invoke(
        app, [command, *base_args, "--screen_grid", "5x4;localmap=3x3"]
    )
    assert result.exit_code != 0
    message = result.output or str(result.exception)
    assert "--screen_grid" in message
    assert "bare size" in message


@pytest.mark.parametrize(
    "command, base_args",
    [
        ("project", ["-i", "x.fa", "-o", "out"]),
        ("cluster", ["-i", "x.fa", "-o", "out"]),
        ("dr", ["-i", "x.tsv", "-o", "out", "-m", "umap"]),
    ],
)
def test_cli_semicolon_values_pass_validation(command, base_args, tmp_path):
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    args = [str(tmp_path) if arg == "out" else arg for arg in base_args]
    result = CliRunner().invoke(
        app,
        [
            command,
            *args,
            "--screen_params",
            "--screen_values1",
            "localmap=10,25,50;pacmap=10,50,100",
            "--screen_values2",
            "all=0.1,0.5,1.0;localmap=0.1,1.0,2.0",
        ],
    )
    message = result.output or str(result.exception)
    assert "require --screen_params" not in message
    assert "method=v1,v2" not in message
    assert "not screenable" not in message


@pytest.mark.parametrize(
    "command, base_args",
    [
        ("project", ["-i", "x.fa", "-o", "out"]),
        ("cluster", ["-i", "x.fa", "-o", "out"]),
        ("dr", ["-i", "x.tsv", "-o", "out", "-m", "umap"]),
    ],
)
def test_cli_semicolon_range_and_named_grid_pass_validation(command, base_args, tmp_path):
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    # A spec that passes validation continues into the command, so keep its
    # output directory out of the repo.
    args = [str(tmp_path) if arg == "out" else arg for arg in base_args]
    result = CliRunner().invoke(
        app,
        [
            command,
            *args,
            "--screen_params",
            "--screen_range1",
            "localmap=10,100;pacmap=20,80",
            "--screen_range2",
            "localmap=0.1,1.0;pacmap=0.1,1.0",
            "--screen_grid",
            "localmap=5x4;pacmap=3x3",
        ],
    )
    message = result.output or str(result.exception)
    assert "require --screen_params" not in message
    assert "method=min,max" not in message
    assert "bare size" not in message
    assert "not screenable" not in message


def test_named_screen_methods_skip_all_and_bare_grid():
    assert named_screen_methods(
        values1=["localmap=10,25;pacmap=10,50"],
        range2=["all=0.1,1.0;localmap=0.2,1.0"],
        grid="5x4",
    ) == ["localmap", "pacmap"]
    assert named_screen_methods(grid="umap=5;localmap=5x4") == ["umap", "localmap"]
    assert named_screen_methods(grid="all=5x4") == []


@pytest.mark.parametrize(
    "command, base_args",
    [
        ("project", ["-i", "x.fa", "-o", "out"]),
        ("cluster", ["-i", "x.fa", "-o", "out"]),
        ("dr", ["-i", "x.tsv", "-o", "out", "-m", "umap"]),
    ],
)
def test_cli_screen_settings_require_screen_params(command, base_args):
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    result = CliRunner().invoke(
        app, [command, *base_args, "--screen_range2", "localmap=0.1,1.0", "--screen_grid", "5x4"]
    )
    assert result.exit_code != 0
    message = result.output or str(result.exception)
    assert "--screen_range2" in message
    assert "--screen_grid" in message
    assert "require --screen_params" in message


@pytest.mark.parametrize(
    "command, method_args, method_flag",
    [
        ("project", ["--dr", "umap"], "--dr"),
        ("cluster", ["--dr", "umap"], "--dr"),
        ("dr", ["-m", "umap"], "--methods"),
    ],
)
def test_cli_warns_when_screen_method_is_not_run(
    command, method_args, method_flag, tmp_path, monkeypatch
):
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    monkeypatch.setattr("kmer_ord.cli.main.Runner.run", lambda self, context: None)
    fasta = tmp_path / "reads.fasta"
    fasta.write_text(">read1\nACGT\n")
    result = CliRunner().invoke(
        app,
        [
            command,
            "-i",
            str(fasta),
            "-o",
            str(tmp_path / command),
            *method_args,
            "--screen_params",
            "--screen_range2",
            "localmap=0.1,1.0;pacmap=0.5,2.0",
            "--screen_grid",
            "umap=5x4",
        ],
    )
    text = " ".join((result.output or "").split())
    assert "Screen settings for localmap, pacmap will not be used" in text
    assert f"{method_flag} is umap" in text


@pytest.mark.parametrize(
    "command, method_args",
    [
        ("project", ["--dr", "localmap"]),
        ("project", ["--dr", "all"]),
        ("dr", ["-m", "localmap,pacmap"]),
    ],
)
def test_cli_does_not_warn_when_configured_method_runs(
    command, method_args, tmp_path, monkeypatch
):
    from typer.testing import CliRunner
    from kmer_ord.cli.main import app

    monkeypatch.setattr("kmer_ord.cli.main.Runner.run", lambda self, context: None)
    fasta = tmp_path / "reads.fasta"
    fasta.write_text(">read1\nACGT\n")
    result = CliRunner().invoke(
        app,
        [
            command,
            "-i",
            str(fasta),
            "-o",
            str(tmp_path / command),
            *method_args,
            "--screen_params",
            "--screen_range2",
            "localmap=0.1,1.0",
            "--screen_grid",
            "5x4",
        ],
    )
    text = " ".join((result.output or "").split())
    assert "will not be used" not in text
