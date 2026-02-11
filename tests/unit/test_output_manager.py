"""
Unit Tests for Output Manager

Tests directory creation, run_id format, config snapshots,
artifact save/load, and metadata generation.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from src.config.schema import PipelineConfig, OutputConfig
from src.io.output_manager import OutputManager, STEP_DIRS


# ===================================================================
# Directory Structure
# ===================================================================

class TestDirectoryStructure:
    """Test that OutputManager creates the correct directory tree."""

    def test_run_dir_created(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        assert om.run_dir.exists()

    def test_config_subdir_created(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        assert (om.run_dir / "config").is_dir()

    def test_data_subdir_created(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        assert (om.run_dir / "data").is_dir()

    def test_reports_subdir_created(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        assert (om.run_dir / "reports").is_dir()

    def test_logs_subdir_created(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        assert (om.run_dir / "logs").is_dir()

    def test_all_step_dirs_created(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        for step_dir in STEP_DIRS:
            assert (om.run_dir / "steps" / step_dir).is_dir(), (
                f"Missing step dir: {step_dir}"
            )

    def test_seven_step_dirs_exist(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        step_dirs = list((om.run_dir / "steps").iterdir())
        assert len(step_dirs) == 7


# ===================================================================
# Run ID Format
# ===================================================================

class TestRunId:
    """Test run_id format: YYYYMMDD_HHMMSS_XXXXXX."""

    def test_run_id_format(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        pattern = r"^\d{8}_\d{6}_[a-f0-9]{6}$"
        assert re.match(pattern, om.run_id), (
            f"run_id '{om.run_id}' does not match YYYYMMDD_HHMMSS_XXXXXX format"
        )

    def test_run_id_contains_timestamp(self, tmp_path):
        fixed_time = datetime(2024, 3, 15, 10, 30, 45)
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config, run_start=fixed_time)
        assert om.run_id.startswith("20240315_103045_")

    def test_run_id_uniqueness_from_config(self, tmp_path):
        config1 = PipelineConfig(
            output={"base_dir": str(tmp_path / "outputs")},
            reproducibility={"global_seed": 42},
        )
        config2 = PipelineConfig(
            output={"base_dir": str(tmp_path / "outputs")},
            reproducibility={"global_seed": 99},
        )
        fixed_time = datetime(2024, 1, 1, 0, 0, 0)
        om1 = OutputManager(config1, run_start=fixed_time)
        om2 = OutputManager(config2, run_start=fixed_time)
        # Different configs at same time should have different hashes
        assert om1.run_id != om2.run_id

    def test_run_dir_matches_run_id(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        assert om.run_dir.name == om.run_id


# ===================================================================
# Config Snapshot
# ===================================================================

class TestConfigSnapshot:
    """Test saving config snapshot to run directory."""

    def test_save_config_snapshot_creates_file(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_config_snapshot(config)
        assert path.exists()
        assert path.name == "pipeline_config.yaml"

    def test_save_config_snapshot_in_config_dir(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_config_snapshot(config)
        assert path.parent.name == "config"

    def test_config_snapshot_content_is_valid_yaml(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_config_snapshot(config)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "data" in data
        assert "steps" in data

    def test_config_snapshot_preserves_values(self, tmp_path):
        config = PipelineConfig(
            output={"base_dir": str(tmp_path / "outputs")},
            steps={"iv": {"min_iv": 0.05, "max_iv": 0.40}},
        )
        om = OutputManager(config)
        path = om.save_config_snapshot(config)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["steps"]["iv"]["min_iv"] == 0.05
        assert data["steps"]["iv"]["max_iv"] == 0.40


# ===================================================================
# Step Results
# ===================================================================

class TestStepResults:
    """Test saving step results."""

    def test_save_step_results_dataframe(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        df = pd.DataFrame({"feature": ["a", "b"], "iv": [0.1, 0.2]})
        om.save_step_results("01_constant", {"results": df})
        out_path = om.run_dir / "steps" / "01_constant" / "results.parquet"
        assert out_path.exists()

    def test_save_step_results_dict(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        om.save_step_results("01_constant", {"metadata": {"key": "value"}})
        out_path = om.run_dir / "steps" / "01_constant" / "metadata.json"
        assert out_path.exists()

    def test_save_step_results_string(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        om.save_step_results("01_constant", {"summary": "some text"})
        out_path = om.run_dir / "steps" / "01_constant" / "summary.txt"
        assert out_path.exists()

    def test_save_step_results_creates_step_dir(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        om.save_step_results("custom_step", {"data": {"key": "val"}})
        assert (om.run_dir / "steps" / "custom_step").is_dir()

    def test_saved_dataframe_round_trip(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        original_df = pd.DataFrame({"feature": ["a", "b", "c"], "score": [0.1, 0.5, 0.9]})
        om.save_step_results("03_iv", {"results": original_df})
        loaded = pd.read_parquet(om.run_dir / "steps" / "03_iv" / "results.parquet")
        pd.testing.assert_frame_equal(original_df, loaded)


# ===================================================================
# Artifact Saving
# ===================================================================

class TestArtifactSaving:
    """Test saving generic artifacts."""

    def test_save_artifact_parquet(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = om.save_artifact("test_data", df, fmt="parquet")
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_save_artifact_csv(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = om.save_artifact("test_data", df, fmt="csv")
        assert path.exists()
        assert path.suffix == ".csv"

    def test_save_artifact_json(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_artifact("metadata", {"key": "value"}, fmt="json")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["key"] == "value"

    def test_save_artifact_custom_subdir(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        df = pd.DataFrame({"a": [1]})
        path = om.save_artifact("model", df, fmt="parquet", subdir="models")
        assert "models" in str(path)
        assert path.exists()

    def test_save_artifact_returns_correct_path(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        df = pd.DataFrame({"a": [1]})
        path = om.save_artifact("test_file", df, fmt="csv", subdir="data")
        assert path.name == "test_file.csv"


# ===================================================================
# Run Metadata
# ===================================================================

class TestRunMetadata:
    """Test run metadata generation and saving."""

    def test_metadata_file_created(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_run_metadata()
        assert path.exists()
        assert path.name == "run_metadata.json"

    def test_metadata_contains_required_fields(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_run_metadata()
        with open(path) as f:
            meta = json.load(f)

        required_fields = [
            "run_id", "git_commit", "python_version",
            "package_versions", "os_info", "run_start",
            "run_end", "duration_seconds", "status",
            "input_file_hash",
        ]
        for field in required_fields:
            assert field in meta, f"Missing required field: {field}"

    def test_metadata_run_id_matches(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_run_metadata()
        with open(path) as f:
            meta = json.load(f)
        assert meta["run_id"] == om.run_id

    def test_metadata_package_versions(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_run_metadata()
        with open(path) as f:
            meta = json.load(f)
        assert "pandas" in meta["package_versions"]
        assert "xgboost" in meta["package_versions"]
        assert "numpy" in meta["package_versions"]

    def test_metadata_duration_is_numeric(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_run_metadata()
        with open(path) as f:
            meta = json.load(f)
        assert isinstance(meta["duration_seconds"], (int, float))
        assert meta["duration_seconds"] >= 0

    def test_metadata_status_default_running(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        path = om.save_run_metadata()
        with open(path) as f:
            meta = json.load(f)
        assert meta["status"] == "running"


# ===================================================================
# Mark Complete / Failed
# ===================================================================

class TestRunStatus:
    """Test run status management."""

    def test_mark_complete_success(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        om.mark_complete("success")
        path = om.save_run_metadata()
        with open(path) as f:
            meta = json.load(f)
        assert meta["status"] == "success"

    def test_mark_failed(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        om.mark_failed()
        path = om.save_run_metadata()
        with open(path) as f:
            meta = json.load(f)
        assert meta["status"] == "failed"


# ===================================================================
# Utility Methods
# ===================================================================

class TestUtilityMethods:
    """Test helper/utility methods."""

    def test_get_step_dir_returns_path(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        step_dir = om.get_step_dir("03_iv")
        assert step_dir.is_dir()
        assert step_dir.name == "03_iv"

    def test_get_step_dir_creates_if_missing(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        step_dir = om.get_step_dir("99_custom_step")
        assert step_dir.exists()

    def test_get_log_path(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        log_path = om.get_log_path()
        assert log_path.name == "pipeline.log"
        assert "logs" in str(log_path)
