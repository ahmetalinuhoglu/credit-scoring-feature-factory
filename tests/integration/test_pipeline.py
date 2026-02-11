"""
Integration Tests for End-to-End Pipeline

Tests that run the full or partial pipeline on synthetic data,
verifying output directory structure, file generation, and result consistency.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config.schema import PipelineConfig
from src.io.output_manager import OutputManager, STEP_DIRS


@pytest.mark.integration
class TestPipelineOutputStructure:
    """Test that the pipeline output directory structure is correct."""

    def test_output_manager_creates_full_tree(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)

        expected_subdirs = ["config", "data", "reports", "logs"]
        for subdir in expected_subdirs:
            assert (om.run_dir / subdir).is_dir()

        for step_dir in STEP_DIRS:
            assert (om.run_dir / "steps" / step_dir).is_dir()

    def test_config_snapshot_in_output(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        om.save_config_snapshot(config)

        config_file = om.run_dir / "config" / "pipeline_config.yaml"
        assert config_file.exists()

    def test_metadata_in_output(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)
        om.save_run_metadata()

        metadata_file = om.run_dir / "run_metadata.json"
        assert metadata_file.exists()


@pytest.mark.integration
class TestDataSplitterIntegration:
    """Integration test for data splitting with real-like data."""

    def test_full_split_pipeline(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        # Verify no data is lost
        total_rows = len(result.train) + len(result.test)
        for qdf in result.oot_quarters.values():
            total_rows += len(qdf)
        assert total_rows == len(large_sample_data)

    def test_split_indices_cover_all_data(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        assert len(result.split_indices_df) == len(large_sample_data)
        sets = result.split_indices_df["set_name"].unique()
        assert "train" in sets
        assert "test" in sets


@pytest.mark.integration
class TestConstantFilterIntegration:
    """Integration test for constant filter on larger dataset."""

    def test_constant_filter_on_large_data(self, large_sample_data, feature_columns):
        from src.components.constant_filter import ConstantFilter
        from src.config.schema import ConstantConfig

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)

        X = large_sample_data[feature_columns]
        y = large_sample_data["target"]
        result = filt.fit(X, y)

        # Should still find the 2 constant features
        assert "const_feat_1" in result.eliminated_features
        assert "const_feat_2" in result.eliminated_features
        assert result.n_eliminated == 2


@pytest.mark.integration
class TestOutputManagerIntegration:
    """Integration test for saving multiple artifacts."""

    def test_save_multiple_step_results(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)

        # Simulate saving results for multiple steps
        for step in ["01_constant", "02_missing", "03_iv"]:
            df = pd.DataFrame({
                "Feature": [f"feat_{i}" for i in range(10)],
                "Status": ["Kept"] * 8 + ["Eliminated"] * 2,
            })
            om.save_step_results(step, {"results": df})

        # Verify all files exist
        for step in ["01_constant", "02_missing", "03_iv"]:
            assert (om.run_dir / "steps" / step / "results.parquet").exists()

    def test_save_config_and_metadata(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)

        om.save_config_snapshot(config)
        om.mark_complete("success")
        om.save_run_metadata()

        assert (om.run_dir / "config" / "pipeline_config.yaml").exists()
        assert (om.run_dir / "run_metadata.json").exists()

        with open(om.run_dir / "run_metadata.json") as f:
            meta = json.load(f)
        assert meta["status"] == "success"

    def test_artifact_formats(self, tmp_path):
        config = PipelineConfig(output={"base_dir": str(tmp_path / "outputs")})
        om = OutputManager(config)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        om.save_artifact("data_parquet", df, fmt="parquet")
        om.save_artifact("data_csv", df, fmt="csv")
        om.save_artifact("meta_json", {"key": "val"}, fmt="json")

        assert (om.run_dir / "data" / "data_parquet.parquet").exists()
        assert (om.run_dir / "data" / "data_csv.csv").exists()
        assert (om.run_dir / "data" / "meta_json.json").exists()


@pytest.mark.integration
class TestPipelineDeterminism:
    """Test that pipeline components produce deterministic results."""

    def test_constant_filter_deterministic(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter
        from src.config.schema import ConstantConfig

        config = ConstantConfig(min_unique_values=2)

        filt1 = ConstantFilter(config)
        result1 = filt1.fit(sample_X, sample_y)

        filt2 = ConstantFilter(config)
        result2 = filt2.fit(sample_X, sample_y)

        assert result1.eliminated_features == result2.eliminated_features
        assert result1.output_features == result2.output_features
        pd.testing.assert_frame_equal(
            result1.results_df.reset_index(drop=True),
            result2.results_df.reset_index(drop=True),
        )

    def test_data_splitter_deterministic(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")

        splitter1 = DataSplitter(data_cfg, split_cfg, seed=42)
        result1 = splitter1.split(large_sample_data)

        splitter2 = DataSplitter(data_cfg, split_cfg, seed=42)
        result2 = splitter2.split(large_sample_data)

        pd.testing.assert_frame_equal(result1.train, result2.train)
        pd.testing.assert_frame_equal(result1.test, result2.test)
        assert result1.feature_columns == result2.feature_columns
