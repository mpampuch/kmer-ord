"""--dr and dr --methods accept a comma-separated list of method names."""

import pytest
from typer.testing import CliRunner

from kmer_ord.cli.main import app

_FULL_LIST = "umap,tsne,trimap,pacmap,localmap,pca"
_runner = CliRunner()


def _message(result) -> str:
    return (result.output or "") + str(result.exception or "")


@pytest.mark.parametrize(
    "command, flag",
    [
        ("project", "--dr"),
        ("cluster", "--dr"),
        ("dr", "-m"),
    ],
)
@pytest.mark.parametrize("methods", [_FULL_LIST, "umap", "all", "umap, pca"])
def test_comma_separated_dr_passes_choice_parsing(tmp_path, command, flag, methods):
    """Choice parsing must accept one method, 'all', or a comma-separated list.

    The command may still fail because the input file is missing. That failure
    must not be Click rejecting the method list.
    """
    args = [
        command,
        "-i",
        str(tmp_path / "missing.fa"),
        "-o",
        str(tmp_path / "out"),
        flag,
        methods,
    ]
    result = _runner.invoke(app, args)
    message = _message(result)
    assert "is not one of" not in message
    assert "must be one or more of" not in message


@pytest.mark.parametrize(
    "command, flag",
    [
        ("project", "--dr"),
        ("cluster", "--dr"),
        ("dr", "-m"),
    ],
)
def test_unknown_dr_method_is_rejected(command, flag):
    result = _runner.invoke(
        app,
        [command, "-i", "x.fa", "-o", "out", flag, "umap,nope"],
    )
    assert result.exit_code != 0
    message = _message(result).lower()
    assert "nope" in message
    assert "is not one of" in message


def test_project_help_shows_comma_separated_dr_metavar():
    # Rich truncates the metavar in the help table, so check the parameter
    # itself for the full token and the rendered help for the visible prefix.
    import typer

    project = typer.main.get_command(app).commands["project"]
    dr = next(param for param in project.params if "--dr" in param.opts)
    assert dr.metavar == "METHOD[,METHOD,...]"

    result = _runner.invoke(app, ["project", "--help"])
    assert result.exit_code == 0
    assert "METHOD[,METHOD" in result.output
