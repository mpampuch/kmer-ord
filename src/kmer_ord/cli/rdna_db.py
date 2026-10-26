# src/kmer_ord/cli/rdna_db.py
from pathlib import Path
from typing import Optional

import click
import typer

from kmer_ord.system.env_manager import RDNA_DB_CHOICES

rdna_db_app = typer.Typer(help="Install and inspect rDNA-miner databases.")


@rdna_db_app.command("install")
def rdna_db_install(
    db: str = typer.Argument(
        ...,
        click_type=click.Choice(list(RDNA_DB_CHOICES), case_sensitive=False),
        help="Database to install: rfam, silva, pr2, or all.",
    ),
    db_dir: Optional[Path] = typer.Option(
        None,
        "--db-dir",
        "-d",
        file_okay=False,
        dir_okay=True,
        help="Directory to install databases into. Created if it does not exist.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if the database already exists.",
    ),
):
    """Install rDNA-miner databases into the rDNA-miner environment."""
    from kmer_ord.system.env_manager import install_rdna_miner_db
    install_rdna_miner_db(db=db, db_dir=db_dir, force=force)


@rdna_db_app.command("status")
def rdna_db_status(
    db_dir: Optional[Path] = typer.Option(
        None,
        "--db-dir",
        "-d",
        file_okay=False,
        dir_okay=True,
        help="Database directory to inspect. Defaults to RDNA_MINER_DB, then ~/.rdna-miner/db.",
    ),
):
    """Show which rDNA-miner databases are installed."""
    from kmer_ord.system.env_manager import rdna_miner_db_status
    rdna_miner_db_status(db_dir=db_dir)
