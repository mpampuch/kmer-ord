# src/kmer_ord/system/rdna_miner_db.py
"""Install and inspect rDNA-miner databases.

Kept out of env_manager.py on purpose. The container build copies an older
env_manager.py over the repo file to pin barrnap. That copy does not include
these commands, and `kmer-ord rdna-db` would fail if it imported them from there.
"""
import subprocess
from pathlib import Path

from kmer_ord.rdna_databases import RDNA_DB_CHOICES
from kmer_ord.system.env_manager import RDNA_ENV
from kmer_ord.utils.logging_utils import info, warn


def _rdna_miner_cmd(env: str, args: list[str], db_dir: Path | None = None, force: bool = False) -> list[str]:
    """Build a `conda run rdna-miner ...` command.

    --no-capture-output lets download progress reach the terminal. conda
    buffers output until the command exits otherwise.
    """
    cmd = ["conda", "run", "--no-capture-output", "-n", env, "rdna-miner", *args]
    if db_dir is not None:
        cmd.extend(["--db-dir", str(db_dir)])
    if force:
        cmd.append("--force")
    return cmd


def install_rdna_miner_db(
    env: str = RDNA_ENV,
    db: str = "all",
    db_dir: Path | None = None,
    force: bool = False,
):
    """
    Install rDNA-miner databases inside its environment.
    Equivalent to: `rdna-miner db install <db> [--db-dir PATH] [--force]`

    Raises CalledProcessError when the install fails so callers exit non-zero.
    """
    if db not in RDNA_DB_CHOICES:
        raise ValueError(
            f"Unknown rDNA-miner database: {db!r}. "
            f"Choose from {', '.join(RDNA_DB_CHOICES)}."
        )
    info(f"Installing rDNA-miner database '{db}'...")
    cmd = _rdna_miner_cmd(env, ["db", "install", db], db_dir=db_dir, force=force)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        warn(f"Failed to install rDNA-miner databases: {e}")
        raise
    info("rDNA-miner databases installed successfully.")


def rdna_miner_db_status(env: str = RDNA_ENV, db_dir: Path | None = None):
    """
    Show which rDNA-miner databases are installed.
    Equivalent to: `rdna-miner db status [--db-dir PATH]`
    """
    cmd = _rdna_miner_cmd(env, ["db", "status"], db_dir=db_dir)
    subprocess.run(cmd, check=True)
