"""
Output Manager

Creates and manages the run directory structure, saves artifacts and metadata.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import logging
import platform
import subprocess
import sys

import pandas as pd

from src.config.schema import PipelineConfig


logger = logging.getLogger(__name__)

# Step directory names in pipeline order
STEP_DIRS = [
    "01_constant",
    "02_missing",
    "03_iv",
    "04_psi",
    "05_correlation",
    "06_selection",
    "07_evaluation",
]


def _get_package_version(package: str) -> str:
    """Get the version string of an installed package.

    Args:
        package: Package name.

    Returns:
        Version string, or 'not installed' if unavailable.
    """
    try:
        import importlib.metadata
        return importlib.metadata.version(package)
    except Exception:
        return "not installed"


def _get_git_hash() -> str:
    """Get the current git commit hash.

    Returns:
        Short commit hash, 'uncommitted' if dirty, or 'no-git' if not a repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return "no-git"
        commit = result.stdout.strip()

        # Check for uncommitted changes
        dirty_check = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=5,
        )
        if dirty_check.returncode != 0:
            return f"{commit}-dirty"
        return commit
    except Exception:
        return "no-git"


def _compute_input_hash(input_path: str, n_rows: int = 1000) -> str:
    """Compute MD5 hash of the first n rows of an input file for quick comparison.

    Args:
        input_path: Path to the input file.
        n_rows: Number of rows to hash.

    Returns:
        MD5 hex digest, or 'unknown' on failure.
    """
    try:
        p = Path(input_path)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p).head(n_rows)
        elif p.suffix == ".csv":
            df = pd.read_csv(p, nrows=n_rows)
        else:
            return "unknown-format"
        content = df.to_csv(index=False).encode("utf-8")
        return hashlib.md5(content).hexdigest()
    except Exception:
        return "unknown"


class OutputManager:
    """Manages the output directory structure and artifact saving for a pipeline run.

    Creates a unique run directory under the configured base_dir with the structure:
        {base_dir}/{run_id}/
            config/
            data/
            steps/01_constant/ ... steps/07_evaluation/
            reports/
            logs/

    The run_id format is {YYYYMMDD}_{HHMMSS}_{short_hash} where short_hash
    is derived from the config for uniqueness.

    Args:
        config: The pipeline configuration.
        run_start: Optional datetime for the run start. Defaults to now.
    """

    def __init__(self, config: PipelineConfig, run_start: Optional[datetime] = None):
        self._config = config
        self._run_start = run_start or datetime.now()
        self._run_end: Optional[datetime] = None
        self._status = "running"

        # Generate run_id
        config_json = config.model_dump_json()
        short_hash = hashlib.md5(config_json.encode()).hexdigest()[:6]
        timestamp = self._run_start.strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{timestamp}_{short_hash}"

        # Create directory structure
        self._base_dir = Path(config.output.base_dir)
        self._run_dir = self._base_dir / self._run_id
        self._create_directories()

        logger.info("Output directory: %s", self._run_dir)

    @property
    def run_id(self) -> str:
        """The unique identifier for this run."""
        return self._run_id

    @property
    def run_dir(self) -> Path:
        """Root directory for this run."""
        return self._run_dir

    def _create_directories(self) -> None:
        """Create the full run directory structure."""
        dirs = [
            self._run_dir / "config",
            self._run_dir / "data",
            self._run_dir / "reports",
            self._run_dir / "logs",
        ]
        for step_dir in STEP_DIRS:
            dirs.append(self._run_dir / "steps" / step_dir)

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def save_config_snapshot(self, config: PipelineConfig) -> Path:
        """Save the frozen config to the run directory.

        Args:
            config: The pipeline configuration to snapshot.

        Returns:
            Path to the saved config file.
        """
        config_path = self._run_dir / "config" / "pipeline_config.yaml"
        config_dict = config.model_dump()

        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.debug("Config snapshot saved to %s", config_path)
        return config_path

    def save_step_results(self, step_name: str, results_dict: Dict[str, Any]) -> None:
        """Save step results as parquet files in the step directory.

        Args:
            step_name: Name of the step (e.g., '01_constant').
            results_dict: Dict of name -> DataFrame or other serializable data.
        """
        step_dir = self._run_dir / "steps" / step_name
        step_dir.mkdir(parents=True, exist_ok=True)

        for name, obj in results_dict.items():
            if isinstance(obj, pd.DataFrame):
                out_path = step_dir / f"{name}.parquet"
                obj.to_parquet(out_path, index=False)
                logger.debug("Saved %s to %s", name, out_path)
            elif isinstance(obj, (dict, list)):
                out_path = step_dir / f"{name}.json"
                with open(out_path, "w") as f:
                    json.dump(obj, f, indent=2, default=str)
            else:
                out_path = step_dir / f"{name}.txt"
                with open(out_path, "w") as f:
                    f.write(str(obj))

    def save_artifact(
        self,
        name: str,
        obj: Any,
        fmt: str = "parquet",
        subdir: str = "data",
    ) -> Path:
        """Save a generic artifact to the run directory.

        Args:
            name: Artifact name (without extension).
            obj: The object to save.
            fmt: Format - 'parquet', 'csv', 'json', 'joblib'.
            subdir: Subdirectory within the run dir.

        Returns:
            Path to the saved artifact.
        """
        target_dir = self._run_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet" and isinstance(obj, pd.DataFrame):
            path = target_dir / f"{name}.parquet"
            obj.to_parquet(path, index=False)
        elif fmt == "csv" and isinstance(obj, pd.DataFrame):
            path = target_dir / f"{name}.csv"
            obj.to_csv(path, index=False)
        elif fmt == "json":
            path = target_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(obj, f, indent=2, default=str)
        elif fmt == "joblib":
            path = target_dir / f"{name}.joblib"
            import joblib
            joblib.dump(obj, path)
        else:
            path = target_dir / f"{name}.{fmt}"
            with open(path, "w") as f:
                f.write(str(obj))

        logger.debug("Artifact saved: %s", path)
        return path

    def save_run_metadata(self) -> Path:
        """Collect and save run metadata to run_metadata.json.

        Includes git info, package versions, OS info, timing, and input hash.

        Returns:
            Path to the metadata file.
        """
        self._run_end = self._run_end or datetime.now()
        duration = (self._run_end - self._run_start).total_seconds()

        metadata = {
            "run_id": self._run_id,
            "git_commit": _get_git_hash(),
            "python_version": sys.version,
            "package_versions": {
                "pandas": _get_package_version("pandas"),
                "xgboost": _get_package_version("xgboost"),
                "scikit-learn": _get_package_version("scikit-learn"),
                "numpy": _get_package_version("numpy"),
                "pydantic": _get_package_version("pydantic"),
            },
            "os_info": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "run_start": self._run_start.isoformat(),
            "run_end": self._run_end.isoformat(),
            "duration_seconds": round(duration, 2),
            "status": self._status,
            "input_file_hash": _compute_input_hash(self._config.data.input_path),
        }

        path = self._run_dir / "run_metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Run metadata saved to %s", path)
        return path

    def get_step_dir(self, step_name: str) -> Path:
        """Get the directory path for a specific step.

        Args:
            step_name: Step directory name (e.g., '01_constant').

        Returns:
            Path to the step directory.
        """
        step_dir = self._run_dir / "steps" / step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def get_log_path(self) -> Path:
        """Get the path for the run log file.

        Returns:
            Path to the log file.
        """
        return self._run_dir / "logs" / "pipeline.log"

    def mark_complete(self, status: str = "success") -> None:
        """Mark the run as complete.

        Args:
            status: Final status ('success' or 'failed').
        """
        self._status = status
        self._run_end = datetime.now()

    def mark_failed(self) -> None:
        """Mark the run as failed."""
        self.mark_complete(status="failed")
