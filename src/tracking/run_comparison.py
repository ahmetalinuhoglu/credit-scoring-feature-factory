"""
Run Comparison

Compare two or more pipeline runs by loading their saved configs, metadata,
and metrics from the output directory structure.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class RunComparison:
    """Compare two or more pipeline runs from their output directories.

    Each run directory is expected to have:
        config/pipeline_config.yaml
        run_metadata.json
        steps/07_evaluation/ (with results)

    Args:
        base_dir: Root output directory containing run subdirectories.
    """

    def __init__(self, base_dir: str = "outputs/model_development"):
        self.base_dir = Path(base_dir)

    def list_runs(self) -> List[str]:
        """List all available run IDs in the base directory.

        Returns:
            Sorted list of run directory names.
        """
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load config, metadata, and available metrics from a run directory.

        Args:
            run_id: The run directory name.

        Returns:
            Dict with keys 'run_id', 'config', 'metadata', 'metrics'.

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        result: Dict[str, Any] = {"run_id": run_id}

        # Load config
        config_path = run_dir / "config" / "pipeline_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                result["config"] = yaml.safe_load(f) or {}
        else:
            result["config"] = {}

        # Load metadata
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result["metadata"] = json.load(f)
        else:
            result["metadata"] = {}

        # Extract key metrics from evaluation step results
        result["metrics"] = self._extract_metrics(run_dir)

        return result

    def compare(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare configs and metrics across multiple runs.

        Args:
            run_ids: List of run IDs to compare.

        Returns:
            DataFrame with run_id as index and config/metric columns.
        """
        rows = []
        for run_id in run_ids:
            try:
                run_data = self.load_run(run_id)
                row = self._flatten_run(run_data)
                rows.append(row)
            except FileNotFoundError:
                logger.warning("Run %s not found, skipping", run_id)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("run_id")
        return df

    def diff_configs(self, run_id_a: str, run_id_b: str) -> pd.DataFrame:
        """Highlight config differences between two runs.

        Args:
            run_id_a: First run ID.
            run_id_b: Second run ID.

        Returns:
            DataFrame with columns [parameter, run_a, run_b] showing only
            parameters that differ between the two runs.
        """
        run_a = self.load_run(run_id_a)
        run_b = self.load_run(run_id_b)

        flat_a = self._flatten_config(run_a.get("config", {}))
        flat_b = self._flatten_config(run_b.get("config", {}))

        all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))

        diffs = []
        for key in all_keys:
            val_a = flat_a.get(key)
            val_b = flat_b.get(key)
            if val_a != val_b:
                diffs.append({
                    "parameter": key,
                    run_id_a: val_a,
                    run_id_b: val_b,
                })

        return pd.DataFrame(diffs)

    def _extract_metrics(self, run_dir: Path) -> Dict[str, Any]:
        """Extract metrics from evaluation step results.

        Args:
            run_dir: Path to the run directory.

        Returns:
            Dict with metric names and values.
        """
        metrics: Dict[str, Any] = {}

        # Try to load evaluation results
        eval_dir = run_dir / "steps" / "07_evaluation"
        if eval_dir.exists():
            results_path = eval_dir / "results.parquet"
            if results_path.exists():
                try:
                    df = pd.read_parquet(results_path)
                    # Extract AUC values by period if available
                    if "period" in df.columns and "auc" in df.columns:
                        for _, row in df.iterrows():
                            metrics[f"auc_{row['period']}"] = row["auc"]
                except Exception as e:
                    logger.debug("Could not load evaluation results: %s", e)

            metadata_path = eval_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        meta = json.load(f)
                    metrics.update(meta)
                except Exception as e:
                    logger.debug("Could not load evaluation metadata: %s", e)

        # Also check selection step for feature count
        sel_dir = run_dir / "steps" / "06_selection"
        if sel_dir.exists():
            output_features_path = sel_dir / "output_features.json"
            if output_features_path.exists():
                try:
                    with open(output_features_path) as f:
                        features = json.load(f)
                    metrics["n_features_selected"] = len(features)
                except Exception:
                    pass

        # Load from run metadata
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                metrics["duration_seconds"] = meta.get("duration_seconds")
                metrics["status"] = meta.get("status")
                metrics["git_commit"] = meta.get("git_commit")
            except Exception:
                pass

        return metrics

    def _flatten_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a run's data into a single row dict.

        Args:
            run_data: Dict from load_run().

        Returns:
            Flat dict suitable for DataFrame row.
        """
        row: Dict[str, Any] = {"run_id": run_data["run_id"]}

        # Flatten key config params
        config = run_data.get("config", {})
        splitting = config.get("splitting", {})
        row["train_end_date"] = splitting.get("train_end_date")

        steps = config.get("steps", {})
        row["iv_min"] = steps.get("iv", {}).get("min_iv")
        row["iv_max"] = steps.get("iv", {}).get("max_iv")
        row["psi_threshold"] = steps.get("psi", {}).get("threshold")
        row["correlation_threshold"] = steps.get("correlation", {}).get("threshold")
        row["auc_threshold"] = steps.get("selection", {}).get("auc_threshold")

        # Flatten metadata
        meta = run_data.get("metadata", {})
        row["duration_seconds"] = meta.get("duration_seconds")
        row["status"] = meta.get("status")
        row["git_commit"] = meta.get("git_commit")

        # Flatten metrics
        metrics = run_data.get("metrics", {})
        for key, value in metrics.items():
            if key not in row:
                row[key] = value

        return row

    def _flatten_config(
        self, config: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """Recursively flatten a nested config dict to dot-notation keys.

        Args:
            config: Nested config dict.
            prefix: Current key prefix.

        Returns:
            Flat dict with dot-notation keys.
        """
        flat: Dict[str, Any] = {}
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            elif isinstance(value, list):
                flat[full_key] = json.dumps(value)
            else:
                flat[full_key] = value
        return flat
