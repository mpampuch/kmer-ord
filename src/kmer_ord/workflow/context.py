# src/kmer_ord/workflow/context.py
from pathlib import Path
from typing import Dict
import logging


class _HasBenchmarkDir:
    """Shared per-run benchmark log location: {output_dir}/benchmarking/."""

    script_name = None

    @property
    def benchmark_dir(self) -> Path:
        d = Path(self.output_dir) / "benchmarking"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def benchmark_timer(self, label, **kwargs):
        from kmer_ord.utils.benchmark import BenchmarkTimer
        kwargs.setdefault("log_dir", str(self.benchmark_dir))
        if self.script_name and "script_name" not in kwargs:
            kwargs["script_name"] = self.script_name
        return BenchmarkTimer(label=label, **kwargs)


class Context(_HasBenchmarkDir):
    """
    Context holds input/output paths and artifacts for the pipeline.

    Responsibilities:
    1. Manage canonical FASTA generation from any supported input (FASTA, FASTQ, gzipped or plain).
    2. Register and retrieve named artifacts.
    3. Provide consistent paths for output files with subdirectory structure.
    4. Provide existence checks and skip logic for pipeline steps.
    """

    # Standard subdirectories for artifact types
    SUBDIRS = {
        "summary": "summary",
        "kmer": "kmer",
        "projections": "projections",
        "figures": "figures"
    }

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        force: bool = False,
        threads: int = 1,
        script_name: str | None = None,
    ):
        from kmer_ord.io.sequence import load_fasta_or_convert
        self.input_file: Path = Path(input_file)
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.force: bool = force  # force recalculation even if files exist
        self.threads: int = threads
        self.script_name = script_name
        self.artifacts: Dict[str, Path] = {}
        self.logger = logging.getLogger("kmer-ord")

        # Automatically create canonical FASTA
        with self.benchmark_timer("fasta_convert", input_file=self.input_file):
            self.fasta: Path = load_fasta_or_convert(
                self.input_file, self.output_dir, force=self.force, threads=self.threads
            )
        self.register("fasta", self.fasta)

    # -------------------------
    # Artifact management
    # -------------------------
    def register(self, name: str, path):
        """
        Register a file or list of files as artifacts in the context.
        """
        if isinstance(path, (list, tuple)):
            self.artifacts[name] = [Path(p) for p in path]
        else:
            self.artifacts[name] = Path(path)

    def get(self, name: str) -> Path:
        """Retrieve a registered artifact by name."""
        if name not in self.artifacts:
            raise ValueError(f"Artifact '{name}' not found in context.")
        return self.artifacts[name]

    # -------------------------
    # Canonical artifact path generation
    # -------------------------
    def artifact_path(self, name: str, subdir: str = None, suffix: str = None) -> Path:
        """
        Generate a canonical path for an artifact.

        Parameters
        ----------
        name : str
            Logical name of the artifact (e.g., "summary_stats_tsv").
        subdir : str, optional
            Subdirectory under output_dir (e.g., "summary", "kmer", "figures").
            If None, placed directly in output_dir.
        suffix : str, optional
            File extension (e.g., ".tsv", ".txt", ".fasta").
            If None, defaults to ".dat".
        """
        base = self.fasta.stem  # canonical FASTA basename
        suffix = suffix or ".dat"
        path = self.output_dir

        if subdir:
            path = path / subdir
            path.mkdir(parents=True, exist_ok=True)

        return path / f"{base}_{name}{suffix}"

    # -------------------------
    # Convenience methods for skipping / force
    # -------------------------
    def exists(self, name: str) -> bool:
        """
        Check if a registered artifact exists on disk.
        Supports both single Path and list of Paths.
        Returns False if force=True.
        """
        if name not in self.artifacts:
            return False

        artifact = self.artifacts[name]

        if self.force:
            return False

        if isinstance(artifact, list):
            return all(p.exists() for p in artifact)
        return artifact.exists()

    def artifact_exists_or_skip(self, name: str) -> bool:
        """
        Returns True if the artifact exists and should skip this step.
        """
        if name not in self.artifacts:
            return False
        exists = self.exists(name)
        if exists:
            self.logger.info(f"Skipping '{name}', artifact already exists at {self.artifacts[name]}")
        return exists
    
class DBContext(_HasBenchmarkDir):
    def __init__(self, db_path: Path, script_name: str | None = None):
        self.output_dir = db_path.parent
        self.script_name = script_name
        self.artifacts = {"database": db_path}

    def register(self, name: str, path):
        self.artifacts[name] = path

    def get(self, name: str):
        return self.artifacts[name]


class MatrixContext(_HasBenchmarkDir):
    """Lightweight context for matrix-based workflows that skip FASTA input."""

    def __init__(
        self,
        matrix_path: Path,
        output_dir: Path,
        force: bool = False,
        script_name: str | None = None,
    ):
        self.input_file = Path(matrix_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.script_name = script_name
        self.artifacts: Dict[str, Path] = {}
        self.logger = logging.getLogger("kmer-ord")
        self.register("kmer_matrix", matrix_path)

    def register(self, name: str, path):
        if isinstance(path, (list, tuple)):
            self.artifacts[name] = [Path(p) for p in path]
        else:
            self.artifacts[name] = Path(path)

    def get(self, name: str):
        if name not in self.artifacts:
            raise ValueError(f"Artifact '{name}' not found in context.")
        return self.artifacts[name]

    def artifact_path(self, name: str, subdir: str = None, suffix: str = None) -> Path:
        base = self.input_file.stem
        suffix = suffix or ".dat"
        path = self.output_dir
        if subdir:
            path = path / subdir
            path.mkdir(parents=True, exist_ok=True)
        return path / f"{base}_{name}{suffix}"

    def exists(self, name: str) -> bool:
        if name not in self.artifacts:
            return False
        if self.force:
            return False
        artifact = self.artifacts[name]
        if isinstance(artifact, list):
            return all(p.exists() for p in artifact)
        return artifact.exists()