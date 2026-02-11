"""
Experiment Tracker

Lightweight CSV-based experiment tracking for comparing runs across sessions.
Optionally logs to MLflow when available.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import logging

import pandas as pd

from src.config.schema import PipelineConfig

logger = logging.getLogger(__name__)

# Optional MLflow support
try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def _config_hash(config: PipelineConfig) -> str:
    """Compute a short hash of the config for deduplication.

    Args:
        config: Pipeline configuration.

    Returns:
        6-character hex digest.
    """
    return hashlib.md5(config.model_dump_json().encode()).hexdigest()[:6]


class ExperimentTracker:
    """Track experiments to local CSV for easy comparison across runs.

    Each row in the CSV log represents one pipeline run with its config
    fingerprint, key metrics, duration, and status.

    Args:
        log_path: Path to the experiment log CSV file.
        mlflow_enabled: Whether to also log to MLflow (requires mlflow installed).
        mlflow_experiment_name: MLflow experiment name.
    """

    COLUMNS = [
        "run_id",
        "timestamp",
        "config_hash",
        "input_file",
        "train_end_date",
        "n_features_selected",
        "train_auc",
        "test_auc",
        "oot_mean_auc",
        "duration_seconds",
        "status",
        "notes",
    ]

    def __init__(
        self,
        log_path: str = "outputs/experiment_log.csv",
        mlflow_enabled: bool = False,
        mlflow_experiment_name: str = "credit-scoring-model-dev",
    ):
        self.log_path = Path(log_path)
        self.mlflow_enabled = mlflow_enabled and _HAS_MLFLOW
        self.mlflow_experiment_name = mlflow_experiment_name

        if self.mlflow_enabled:
            mlflow.set_experiment(self.mlflow_experiment_name)
            logger.info("MLflow tracking enabled (experiment: %s)", self.mlflow_experiment_name)

    def log_run(
        self,
        run_id: str,
        config: PipelineConfig,
        metrics: Dict[str, Any],
        duration: float,
        status: str = "success",
        notes: str = "",
    ) -> None:
        """Append a run entry to the experiment log.

        Args:
            run_id: Unique identifier for this run.
            config: The pipeline configuration used.
            metrics: Dict with keys like 'train_auc', 'test_auc', 'oot_mean_auc',
                'n_features_selected'.
            duration: Total run duration in seconds.
            status: Run status ('success' or 'failed').
            notes: Free-text notes about the run.
        """
        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_hash": _config_hash(config),
            "input_file": config.data.input_path,
            "train_end_date": config.splitting.train_end_date,
            "n_features_selected": metrics.get("n_features_selected", 0),
            "train_auc": metrics.get("train_auc", None),
            "test_auc": metrics.get("test_auc", None),
            "oot_mean_auc": metrics.get("oot_mean_auc", None),
            "duration_seconds": round(duration, 1),
            "status": status,
            "notes": notes,
        }

        # Append to CSV
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new_row_df = pd.DataFrame([row])

        if self.log_path.exists():
            existing = pd.read_csv(self.log_path)
            combined = pd.concat([existing, new_row_df], ignore_index=True)
        else:
            combined = new_row_df

        combined.to_csv(self.log_path, index=False)
        logger.info("Logged run %s to %s", run_id, self.log_path)

        # Optional MLflow logging
        if self.mlflow_enabled:
            self._log_to_mlflow(run_id, config, metrics, duration, status, notes)

    def get_history(self) -> pd.DataFrame:
        """Load the full experiment history.

        Returns:
            DataFrame with all logged runs, or empty DataFrame if no log exists.
        """
        if not self.log_path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.read_csv(self.log_path)

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare configs and metrics side by side for given run IDs.

        Args:
            run_ids: List of run_id strings to compare.

        Returns:
            DataFrame with one row per run, filtered to the requested IDs.

        Raises:
            ValueError: If no matching runs found.
        """
        history = self.get_history()
        if history.empty:
            raise ValueError("No experiment history found")

        matched = history[history["run_id"].isin(run_ids)]
        if matched.empty:
            raise ValueError(f"No runs found matching IDs: {run_ids}")

        return matched.set_index("run_id")

    def get_best_run(self, metric: str = "test_auc", ascending: bool = False) -> pd.Series:
        """Get the best run by a specific metric.

        Args:
            metric: Column name to sort by.
            ascending: Sort order (False = highest first).

        Returns:
            Series with the best run's data.

        Raises:
            ValueError: If no history or metric column not found.
        """
        history = self.get_history()
        if history.empty:
            raise ValueError("No experiment history found")
        if metric not in history.columns:
            raise ValueError(f"Metric '{metric}' not found in columns: {list(history.columns)}")

        sorted_df = history.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
        return sorted_df.iloc[0]

    def _log_to_mlflow(
        self,
        run_id: str,
        config: PipelineConfig,
        metrics: Dict[str, Any],
        duration: float,
        status: str,
        notes: str,
    ) -> None:
        """Log run details to MLflow.

        Args:
            run_id: Unique run identifier.
            config: Pipeline configuration.
            metrics: Metric values.
            duration: Run duration in seconds.
            status: Run status.
            notes: Free-text notes.
        """
        with mlflow.start_run(run_name=run_id):
            # Log params
            mlflow.log_param("train_end_date", config.splitting.train_end_date)
            mlflow.log_param("input_file", config.data.input_path)
            mlflow.log_param("config_hash", _config_hash(config))
            mlflow.log_param("iv_min", config.steps.iv.min_iv)
            mlflow.log_param("iv_max", config.steps.iv.max_iv)
            mlflow.log_param("psi_threshold", config.steps.psi.threshold)
            mlflow.log_param("correlation_threshold", config.steps.correlation.threshold)
            mlflow.log_param("auc_threshold", config.steps.selection.auc_threshold)

            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    mlflow.log_metric(key, value)
            mlflow.log_metric("duration_seconds", duration)

            # Log tags
            mlflow.set_tag("status", status)
            if notes:
                mlflow.set_tag("notes", notes)
