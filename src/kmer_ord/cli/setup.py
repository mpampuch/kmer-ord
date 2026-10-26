# src/kmer_ord/cli/setup.py
from pathlib import Path
from typing import Optional

from kmer_ord.utils.logging_utils import section, info, warn
import typer


setup_app = typer.Typer(help="Setup external dependencies for kmerord.")


@setup_app.command("setup", rich_help_panel="Installation")
def setup(
    force: bool = typer.Option(False, "-f", "--force", help="Recreate environments even if they already exist."),
    with_tiara: bool = typer.Option(True, "--tiara/--no-tiara", help="Install the Tiara taxonomic classification environment (default: on)."),
    rdna_db_dir: Optional[Path] = typer.Option(
        None,
        "--rdna-db-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory for rDNA-miner databases. Used only when --force reinstalls them.",
    ),):
    """
    Setup all required external dependencies:
    - tools environment (kmer-counter, rust)
    - tiara environment (optional, pass --no-tiara to skip)
    - kmer-counter Rust installation from GitHub
    - rDNA-miner environment
    """
    from kmer_ord.system.env_manager import (TOOLS_ENV, TIARA_ENV, RDNA_ENV,
                                             env_exists, create_tools_env, create_rdna_env,
                                             create_tiara_env, install_rust_tool, install_rdna_miner,
                                             check_tool)
    from kmer_ord.system.rdna_miner_db import install_rdna_miner_db

    print("-" * 70)
    section("Starting kmerord setup...")

    # Tools environment
    if env_exists(TOOLS_ENV):
        if force:
            info(f"{TOOLS_ENV} exists. Recreating (--force enabled)...")
            create_tools_env(TOOLS_ENV, recreate=True)
        else:
            info(f"{TOOLS_ENV} already exists. Skipping.")
    else:
        info(f"Creating {TOOLS_ENV}...")
        create_tools_env(TOOLS_ENV)
        info(f"{TOOLS_ENV} created.")

    # Tiara environment (optional)
    if with_tiara:
        if env_exists(TIARA_ENV):
            if force:
                info(f"{TIARA_ENV} exists. Recreating (--force enabled)...")
                create_tiara_env(TIARA_ENV, recreate=True)
            else:
                info(f"{TIARA_ENV} already exists. Skipping.")
        else:
            info(f"Creating {TIARA_ENV}...")
            create_tiara_env(TIARA_ENV)
            info(f"{TIARA_ENV} created.")
    else:
        info("Skipping Tiara environment (--no-tiara).")

    # Rust GitHub tool
    section("Installing Rust-based tool...")
    install_rust_tool()
    info("Rust tool installation complete.")

    # rDNA-miner environment. A fresh setup creates the env only.
    # Databases are downloaded on --force, or via `kmer-ord rdna-db install`.
    section("Setting up rDNA-miner...")

    installed_rdna_db = False
    if env_exists(RDNA_ENV):
        if force:
            info(f"{RDNA_ENV} exists. Recreating (--force enabled)...")
            repo_path = create_rdna_env(RDNA_ENV, recreate=True)
            install_rdna_miner(RDNA_ENV, repo_path)
            install_rdna_miner_db(RDNA_ENV, db_dir=rdna_db_dir)
            installed_rdna_db = True
        else:
            info(f"{RDNA_ENV} already exists. Skipping.")
    else:
        info(f"Creating {RDNA_ENV}...")
        repo_path = create_rdna_env(RDNA_ENV)
        install_rdna_miner(RDNA_ENV, repo_path)
        info(f"{RDNA_ENV} created.")

    if rdna_db_dir is not None and not installed_rdna_db:
        warn(
            "--rdna-db-dir applies when --force reinstalls rDNA-miner databases. "
            "Install them with `kmer-ord rdna-db install` or `kmer-ord setup --force`."
        )

    # Verify installed tools
    section("Verifying external tools...")

    tools_to_check = [
        ("kmer-counter", TOOLS_ENV, ["--help"]),
    ]
    if with_tiara:
        tools_to_check.append(("tiara", TIARA_ENV, ["--help"]))
    tools_to_check += [
        ("barrnap", RDNA_ENV, ["--help"]),
        ("flye", RDNA_ENV, ["--help"]),
        ("minimap2", RDNA_ENV, ["--help"]),
        ("samtools", RDNA_ENV, ["--help"]),
        ("Rscript", RDNA_ENV, ["--version"]),
    ]

    success = True
    for tool, env, args in tools_to_check:
        if not check_tool(tool, env=env, args=args):
            success = False

    if success:
        print("-" * 70)
        info("\nAll tools verified successfully")
        info("\nSetup complete. kmerord is ready to use.")
    else:
        print("-" * 70)
        warn("Some tools failed verification. Please check installation or PATH.")
