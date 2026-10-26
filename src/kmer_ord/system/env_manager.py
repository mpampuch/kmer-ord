# src/kmer_ord/system/env_manager.py
import subprocess
import shutil
#import pkg_resources
from pathlib import Path
from importlib.resources import files
import tempfile

from kmer_ord.utils.logging_utils import section, info, warn

# Environment constants
ENV_VERSION = "0.1"
TOOLS_ENV = f"kmerord-dependencies-{ENV_VERSION}"
TIARA_ENV = f"kmerord-tiara-{ENV_VERSION}"

TOOLS_YAML = f"envs/kmerord-dependencies-{ENV_VERSION}.yml"
TIARA_YAML = f"envs/kmerord-tiara-{ENV_VERSION}.yml"

RDNA_ENV = f"kmerord-rdna-{ENV_VERSION}"
RDNA_YAML = f"envs/kmerord-rdna-{ENV_VERSION}.yml"
RDNA_REPO = "https://github.com/FDBoever/rDNA-miner"

# Utility functions
def conda_exists():
    """Check if conda is available in PATH."""
    return shutil.which("conda") is not None

def env_exists(name: str) -> bool:
    """Check if a conda environment exists."""
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True
    )
    return name in result.stdout


#def _get_yaml_path(yaml_file: str) -> str:
#    return pkg_resources.resource_filename("kmer_ord", yaml_file)

def _get_yaml_path(yaml_file: str) -> str:
    return str(files("kmer_ord").joinpath(yaml_file))

def _create_env_from_yaml(env_name: str, yaml_file: str, recreate: bool = False):
    """
    Create a conda environment from a YAML file.
    """
    if recreate and env_exists(env_name):
        info(f"{env_name} exists. Removing (--force enabled)...")
        subprocess.run(["conda", "remove", "-y", "-n", env_name, "--all"], check=False)

    yaml_path = _get_yaml_path(yaml_file)
    if not Path(yaml_path).exists():
        raise FileNotFoundError(f"Environment YAML not found: {yaml_path}")

    subprocess.run([
        "conda", "env", "create",
        "-f", yaml_path,
        "-n", env_name
    ], check=True)


# functions
def create_tools_env(name: str = TOOLS_ENV, recreate: bool = False):
    """Create the tools environment (barrnap, infernal, rust) from YAML."""
    _create_env_from_yaml(name, TOOLS_YAML, recreate=recreate)

def create_tiara_env(name: str = TIARA_ENV, recreate: bool = False):
    """Create the Tiara environment from YAML."""
    _create_env_from_yaml(name, TIARA_YAML, recreate=recreate)

def run_in_env(env: str, cmd: list[str], **kwargs):
    """Run a command inside a conda environment."""
    full_cmd = ["conda", "run", "-n", env] + cmd
    subprocess.run(full_cmd, **kwargs)

def install_rdna_miner(env: str, repo_path: Path):
    """
    Install rDNA-miner into its environment.
    """
    info("Installing rDNA-miner...")
    subprocess.run(
        ["conda", "run", "-n", env, "pip", "install", "."],
        cwd=str(repo_path),
        check=True
    )
    info("rDNA-miner installation complete.")

# Rust tool installation 
RUST_TOOL_REPO = "https://github.com/CobiontID/kmer-counter"
RUST_TOOL_NAME = "kmer-counter"

def install_rust_tool(env: str = TOOLS_ENV, force: bool = False):
    """
    Install the Rust tool into the conda environment bin directory.
    """
    # Determine conda env path
    conda_env_path_result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True, text=True, check=True
    )
    env_path = None
    for line in conda_env_path_result.stdout.splitlines():
        if line.startswith(env):
            env_path = line.split()[-1]
            break
    if env_path is None:
        raise RuntimeError(f"Conda environment {env} not found")

    bin_dir = Path(env_path) / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    kmer_counter_bin = bin_dir / RUST_TOOL_NAME

    # If binary exists and force=False, skip
    if kmer_counter_bin.exists() and not force:
        info(f"{RUST_TOOL_NAME} already installed in {env}. Skipping.")
        return

    # Clean temp dir
    tmp_dir = Path("/tmp/kmerord_rust_tool_build")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # Clone repo
    info(f"Cloning {RUST_TOOL_REPO}...")
    subprocess.run(["git", "clone", RUST_TOOL_REPO, str(tmp_dir)], check=True)

    # Build in release mode
    info("Building Rust tool...")
    subprocess.run(
        ["conda", "run", "-n", env, "cargo", "build", "--release"],
        cwd=str(tmp_dir),
        check=True
    )

    # Copy binary from target/release to env/bin
    target_bin = tmp_dir / "target" / "release" / RUST_TOOL_NAME
    if not target_bin.exists():
        raise RuntimeError(f"Build succeeded but {target_bin} not found")

    shutil.copy(target_bin, kmer_counter_bin)
    info(f"{RUST_TOOL_NAME} installed in {bin_dir}")



def pin_barrnap(yaml_text: str, version: str = "0.9") -> str:
    """Pin barrnap in an rDNA-miner environment.yml.

    rDNA-miner calls barrnap flags that changed after 0.9, so the conda env
    is created from a patched copy of upstream's YAML.
    """
    patched = []
    found = False
    for line in yaml_text.splitlines():
        if line.strip().startswith("- barrnap"):
            indent = line[:len(line) - len(line.lstrip())]
            patched.append(f"{indent}- barrnap={version}")
            found = True
        else:
            patched.append(line)
    if not found:
        raise RuntimeError("Could not find a 'barrnap' dependency in rDNA-miner environment.yml")
    return "\n".join(patched) + "\n"


def create_rdna_env(name: str = RDNA_ENV, recreate: bool = False):
    """
    Clone rDNA-miner and create its conda environment from its own YAML.
    Pins barrnap to 0.9 before creating the environment.
    """
    if recreate and env_exists(name):
        info(f"{name} exists. Removing (--force enabled)...")
        subprocess.run(["conda", "remove", "-y", "-n", name, "--all"], check=False)

    tmp_dir = Path(tempfile.mkdtemp(prefix="kmerord_rdna_"))

    info(f"Cloning {RDNA_REPO}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", RDNA_REPO, str(tmp_dir)],
        check=True
    )

    yaml_path = tmp_dir / "environment.yml"
    if not yaml_path.exists():
        raise FileNotFoundError("rDNA-miner environment.yml not found")

    info("Pinning barrnap=0.9 in rDNA-miner environment.yml")
    yaml_path.write_text(pin_barrnap(yaml_path.read_text()))

    info(f"Creating {name} from rDNA-miner YAML...")
    subprocess.run([
        "conda", "env", "create",
        "-f", str(yaml_path),
        "-n", name
    ], check=True)

    return tmp_dir  # important for next step


#------------------------------
def check_tool(tool_name: str, env: str = None, args: list[str] = None):
    """
    Check if a tool is available. If `env` is provided, run inside conda env.
    """
    if args is None:
        args = ["--help"]

    cmd = [tool_name] + args
    if env:
        cmd = ["conda", "run", "-n", env] + cmd

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        info(f"{tool_name} detected and working")
        return True
    except subprocess.CalledProcessError:
        warn(f"{tool_name} failed verification")
        return False