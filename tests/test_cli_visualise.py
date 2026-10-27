"""`visualize` is an alias of the `visualise` command."""

import pytest
from typer.testing import CliRunner

from kmer_ord.cli.main import app

_runner = CliRunner()


def test_help_lists_only_visualise():
    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Match command rows, not the word "visualize" inside the visualise docstring.
    names = []
    for line in result.output.splitlines():
        stripped = line.strip().lstrip("│").strip()
        token = stripped.split(maxsplit=1)[0] if stripped else ""
        if token in {"visualise", "visualize"}:
            names.append(token)
    assert names == ["visualise"]
    assert "`visualize` is an accepted alias." in result.output


@pytest.mark.parametrize("command", ["visualise", "visualize"])
def test_help_documents_db_option(command):
    result = _runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "--db" in result.output


@pytest.mark.parametrize("command", ["visualise", "visualize"])
def test_missing_db_fails_the_same_way(command):
    """Either spelling must reject a missing --db before any plotting runs."""
    result = _runner.invoke(app, [command])
    assert result.exit_code != 0
    message = (result.output or "") + str(result.exception or "")
    assert "--db" in message
