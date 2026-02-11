"""
Integration Tests for Notebook/Script Parity

Tests that the pipeline produces consistent results when invoked
programmatically (simulating notebook usage) vs. through the standard API.
"""

import pytest
import numpy as np
import pandas as pd

from src.config.schema import PipelineConfig
from src.io.output_manager import OutputManager


@pytest.mark.integration
class TestProgrammaticAPIUsage:
    """Test pipeline usage through Python API (notebook-style)."""

    def test_config_creation_programmatic(self):
        """Test creating config purely from code without YAML."""
        config = PipelineConfig(
            data={"target_column": "target", "date_column": "date"},
            steps={"iv": {"min_iv": 0.05, "max_iv": 0.40}},
            reproducibility={"global_seed": 123},
        )
        assert config.data.target_column == "target"
        assert config.steps.iv.min_iv == 0.05
        assert config.reproducibility.global_seed == 123

    def test_output_manager_programmatic(self, tmp_path):
        """Test using OutputManager from code."""
        config = PipelineConfig(
            output={"base_dir": str(tmp_path / "nb_outputs")},
        )
        om = OutputManager(config)

        # Save artifacts like a notebook would
        df = pd.DataFrame({"feature": ["a", "b"], "importance": [0.5, 0.3]})
        path = om.save_artifact("feature_importance", df, fmt="csv")
        assert path.exists()

        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_data_splitter_programmatic(self, large_sample_data):
        """Test data splitting from code matches expected contract."""
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig(
            target_column="target",
            date_column="date",
            id_columns=["application_id", "customer_id"],
            exclude_columns=["applicant_type"],
        )
        split_cfg = SplittingConfig(train_end_date="2022-07-01", test_size=0.25)
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        assert len(result.train) > 0
        assert len(result.test) > 0
        assert isinstance(result.feature_columns, list)
        assert len(result.feature_columns) > 0

    def test_constant_filter_programmatic(self, sample_X, sample_y):
        """Test constant filter from code."""
        from src.components.constant_filter import ConstantFilter
        from src.config.schema import ConstantConfig

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        X_clean, result = filt.fit_transform(sample_X, sample_y)

        assert "const_feat_1" not in X_clean.columns
        assert "const_feat_2" not in X_clean.columns
        assert result.n_eliminated == 2

    def test_config_from_yaml_and_programmatic_match(self, tmp_config_yaml):
        """Test that loading from YAML and creating programmatically give same result."""
        from src.config.loader import load_config

        yaml_config = load_config(str(tmp_config_yaml))

        prog_config = PipelineConfig(
            data={
                "input_path": "data/sample/sample_features.parquet",
                "target_column": "target",
                "date_column": "date",
                "id_columns": ["application_id", "customer_id"],
                "exclude_columns": ["applicant_type"],
            },
            splitting={"train_end_date": "2024-06-30", "test_size": 0.20, "stratify": True},
            steps={
                "constant": {"enabled": True, "min_unique_values": 2},
                "missing": {"enabled": True, "threshold": 0.70},
                "iv": {"enabled": True, "min_iv": 0.02, "max_iv": 0.50, "n_bins": 10, "min_samples_per_bin": 50},
                "psi": {
                    "enabled": True,
                    "threshold": 0.25,
                    "n_bins": 10,
                    "checks": [{"type": "quarterly"}, {"type": "yearly"}, {"type": "consecutive"}],
                },
                "correlation": {"enabled": True, "threshold": 0.90, "method": "pearson"},
                "selection": {"enabled": True, "method": "forward", "auc_threshold": 0.0001, "max_features": None},
            },
            model={
                "algorithm": "xgboost",
                "params": {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "n_estimators": 50,
                    "early_stopping_rounds": 10,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
            },
            evaluation={
                "metrics": ["auc", "gini", "ks"],
                "precision_at_k": [5, 10, 20],
                "n_deciles": 10,
                "calculate_score_psi": True,
            },
            validation={
                "enabled": True,
                "checks": {
                    "min_auc": 0.65,
                    "max_overfit_gap": 0.05,
                    "max_oot_degradation": 0.08,
                    "max_score_psi": 0.25,
                    "max_feature_concentration": 0.50,
                    "min_oot_samples": 30,
                    "check_monotonicity": True,
                },
            },
            reproducibility={"global_seed": 42, "save_config": True, "save_metadata": True, "log_level": "DEBUG"},
        )

        # Compare key fields - ignore paths which may differ due to resolution
        assert yaml_config.data.target_column == prog_config.data.target_column
        assert yaml_config.steps.iv.min_iv == prog_config.steps.iv.min_iv
        assert yaml_config.steps.correlation.threshold == prog_config.steps.correlation.threshold
        assert yaml_config.reproducibility.global_seed == prog_config.reproducibility.global_seed


@pytest.mark.integration
class TestMultiStepPipeline:
    """Test running multiple pipeline steps in sequence."""

    def test_split_then_constant_filter(self, large_sample_data):
        """Test the first two pipeline stages work together."""
        from src.components.data_splitter import DataSplitter
        from src.components.constant_filter import ConstantFilter
        from src.config.schema import DataConfig, SplittingConfig, ConstantConfig

        # Step 0: Split
        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        split_result = splitter.split(large_sample_data)

        # Step 1: Constant filter on training features
        X_train = split_result.train[split_result.feature_columns]
        y_train = split_result.train["target"]

        const_config = ConstantConfig(min_unique_values=2)
        const_filter = ConstantFilter(const_config)
        X_clean, const_result = const_filter.fit_transform(X_train, y_train)

        # Verify chain
        assert const_result.n_input == len(split_result.feature_columns)
        assert const_result.n_eliminated >= 2  # const_feat_1, const_feat_2
        assert "const_feat_1" not in X_clean.columns

        # Apply same transform to test set
        X_test = split_result.test[split_result.feature_columns]
        X_test_clean = const_filter.transform(X_test)
        assert X_test_clean.columns.tolist() == X_clean.columns.tolist()
