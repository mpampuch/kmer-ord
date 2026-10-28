"""Screen spec strings must name a method. A bare range is otherwise ignored."""
import pytest

from kmer_ord.dr.screen_grid import resolve_method_grid, validate_screen_specs


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
