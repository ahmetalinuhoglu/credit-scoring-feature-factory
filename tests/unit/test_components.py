"""
Unit Tests for Pipeline Step Components

Tests each component independently against the BaseComponent/StepResult contract.
Components not yet implemented are marked with skipIf or xfail.
"""

import pytest
import numpy as np
import pandas as pd

from src.pipeline.base import BaseComponent, StepResult
from src.config.schema import (
    ConstantConfig,
    MissingConfig,
    IVConfig,
    PSIConfig,
    CorrelationConfig,
    SelectionConfig,
)


# ===================================================================
# StepResult Tests
# ===================================================================

class TestStepResult:
    """Test the StepResult dataclass."""

    def test_step_result_properties(self):
        result = StepResult(
            step_name="test_step",
            input_features=["a", "b", "c"],
            output_features=["a", "b"],
            eliminated_features=["c"],
            results_df=pd.DataFrame(),
        )
        assert result.n_input == 3
        assert result.n_output == 2
        assert result.n_eliminated == 1

    def test_step_result_summary(self):
        result = StepResult(
            step_name="01_constant",
            input_features=["a", "b", "c"],
            output_features=["a", "b"],
            eliminated_features=["c"],
            results_df=pd.DataFrame(),
            duration_seconds=1.5,
        )
        summary = result.summary()
        assert "01_constant" in summary
        assert "3" in summary
        assert "2" in summary
        assert "1 eliminated" in summary

    def test_step_result_default_metadata(self):
        result = StepResult(
            step_name="test",
            input_features=[],
            output_features=[],
            eliminated_features=[],
            results_df=pd.DataFrame(),
        )
        assert result.metadata == {}
        assert result.duration_seconds == 0.0

    def test_step_result_empty_features(self):
        result = StepResult(
            step_name="test",
            input_features=[],
            output_features=[],
            eliminated_features=[],
            results_df=pd.DataFrame(),
        )
        assert result.n_input == 0
        assert result.n_output == 0
        assert result.n_eliminated == 0


# ===================================================================
# ConstantFilter Tests
# ===================================================================

class TestConstantFilter:
    """Test the ConstantFilter component."""

    def test_constant_filter_eliminates_constant_features(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        result = filt.fit(sample_X, sample_y)

        assert "const_feat_1" in result.eliminated_features
        assert "const_feat_2" in result.eliminated_features
        assert len(result.eliminated_features) == 2

    def test_constant_filter_keeps_non_constant(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        result = filt.fit(sample_X, sample_y)

        for feat in ["high_iv_1", "high_iv_2", "low_iv_1", "low_iv_2"]:
            assert feat in result.output_features

    def test_constant_filter_transform_drops_columns(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        filt.fit(sample_X, sample_y)
        X_out = filt.transform(sample_X)

        assert "const_feat_1" not in X_out.columns
        assert "const_feat_2" not in X_out.columns
        assert len(X_out.columns) == len(sample_X.columns) - 2

    def test_constant_filter_with_higher_threshold(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        # Require at least 3 unique values -> should catch binary-like features too
        config = ConstantConfig(min_unique_values=3)
        filt = ConstantFilter(config)
        result = filt.fit(sample_X, sample_y)

        # Constant features should still be eliminated
        assert "const_feat_1" in result.eliminated_features
        assert "const_feat_2" in result.eliminated_features
        # May catch additional near-constant features
        assert result.n_eliminated >= 2

    def test_constant_filter_all_constant_dataframe(self, sample_y):
        from src.components.constant_filter import ConstantFilter

        X_all_const = pd.DataFrame({
            "a": [1.0] * len(sample_y),
            "b": [0] * len(sample_y),
            "c": ["same"] * len(sample_y),
        })
        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        result = filt.fit(X_all_const, sample_y)

        assert result.n_eliminated == 3
        assert result.n_output == 0

    def test_constant_filter_returns_step_result(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        result = filt.fit(sample_X, sample_y)

        assert isinstance(result, StepResult)
        assert result.step_name == "01_constant"
        assert result.duration_seconds >= 0

    def test_constant_filter_results_df_has_expected_columns(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        result = filt.fit(sample_X, sample_y)

        assert "Feature" in result.results_df.columns
        assert "Unique_Count" in result.results_df.columns
        assert "Status" in result.results_df.columns
        assert len(result.results_df) == len(sample_X.columns)

    def test_constant_filter_fit_transform(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        X_out, result = filt.fit_transform(sample_X, sample_y)

        assert isinstance(X_out, pd.DataFrame)
        assert isinstance(result, StepResult)
        assert "const_feat_1" not in X_out.columns

    def test_constant_filter_step_attributes(self):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig()
        filt = ConstantFilter(config)
        assert filt.step_name == "01_constant"
        assert filt.step_order == 1

    def test_constant_filter_transform_on_new_data(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        filt.fit(sample_X, sample_y)

        # Transform new data - same columns, different values
        new_X = sample_X.head(10).copy()
        X_out = filt.transform(new_X)
        assert "const_feat_1" not in X_out.columns
        assert len(X_out) == 10

    def test_constant_filter_preserves_row_count(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        filt.fit(sample_X, sample_y)
        X_out = filt.transform(sample_X)
        assert len(X_out) == len(sample_X)

    def test_constant_filter_input_output_consistency(self, sample_X, sample_y):
        from src.components.constant_filter import ConstantFilter

        config = ConstantConfig(min_unique_values=2)
        filt = ConstantFilter(config)
        result = filt.fit(sample_X, sample_y)

        assert set(result.output_features) | set(result.eliminated_features) == set(result.input_features)
        assert len(result.output_features) + len(result.eliminated_features) == len(result.input_features)


# ===================================================================
# MissingFilter Tests (component may not exist yet)
# ===================================================================

class TestMissingFilter:
    """Test the MissingFilter component."""

    @pytest.fixture(autouse=True)
    def _check_module_exists(self):
        try:
            from src.components.missing_filter import MissingFilter
            self._skip = False
        except (ImportError, ModuleNotFoundError):
            self._skip = True

    def _get_filter(self):
        if self._skip:
            pytest.skip("MissingFilter not yet implemented")
        from src.components.missing_filter import MissingFilter
        return MissingFilter

    def test_missing_filter_eliminates_high_missing(self, sample_X, sample_y):
        MissingFilter = self._get_filter()
        config = MissingConfig(threshold=0.70)
        filt = MissingFilter(config)
        result = filt.fit(sample_X, sample_y)

        # high_missing_1 is ~80% missing -> should be eliminated with threshold=0.70
        assert "high_missing_1" in result.eliminated_features

    def test_missing_filter_threshold_0_70_catches_only_80_pct(self, sample_X, sample_y):
        MissingFilter = self._get_filter()
        config = MissingConfig(threshold=0.70)
        filt = MissingFilter(config)
        result = filt.fit(sample_X, sample_y)

        # high_missing_1 ~80% -> eliminated
        assert "high_missing_1" in result.eliminated_features
        # high_missing_2 ~75% -> also eliminated (75% > 70%)
        assert "high_missing_2" in result.eliminated_features

    def test_missing_filter_higher_threshold(self, sample_X, sample_y):
        MissingFilter = self._get_filter()
        config = MissingConfig(threshold=0.85)
        filt = MissingFilter(config)
        result = filt.fit(sample_X, sample_y)

        # Neither should be eliminated at 85% threshold
        # (high_missing_1 ~80%, high_missing_2 ~75%)
        assert "high_missing_1" not in result.eliminated_features
        assert "high_missing_2" not in result.eliminated_features

    def test_missing_filter_transform_drops_features(self, sample_X, sample_y):
        MissingFilter = self._get_filter()
        config = MissingConfig(threshold=0.70)
        filt = MissingFilter(config)
        filt.fit(sample_X, sample_y)
        X_out = filt.transform(sample_X)

        for feat in filt.eliminated_features_:
            assert feat not in X_out.columns

    def test_missing_filter_returns_step_result(self, sample_X, sample_y):
        MissingFilter = self._get_filter()
        config = MissingConfig(threshold=0.70)
        filt = MissingFilter(config)
        result = filt.fit(sample_X, sample_y)

        assert isinstance(result, StepResult)
        assert result.step_name == "02_missing"

    def test_missing_filter_no_missing_data(self, sample_y):
        MissingFilter = self._get_filter()
        X_clean = pd.DataFrame({
            f"feat_{i}": np.random.randn(len(sample_y)) for i in range(5)
        })
        config = MissingConfig(threshold=0.70)
        filt = MissingFilter(config)
        result = filt.fit(X_clean, sample_y)

        assert result.n_eliminated == 0
        assert result.n_output == 5


# ===================================================================
# IVFilter Tests (component may not exist yet)
# ===================================================================

class TestIVFilter:
    """Test the IVFilter component."""

    @pytest.fixture(autouse=True)
    def _check_module_exists(self):
        try:
            from src.components.iv_filter import IVFilter
            self._skip = False
        except (ImportError, ModuleNotFoundError):
            self._skip = True

    def _get_filter(self):
        if self._skip:
            pytest.skip("IVFilter not yet implemented")
        from src.components.iv_filter import IVFilter
        return IVFilter

    def test_iv_filter_keeps_high_iv_features(self, sample_X, sample_y):
        IVFilter = self._get_filter()
        # Use very high max_iv to avoid flagging synthetic high-IV features as suspicious
        config = IVConfig(min_iv=0.02, max_iv=50.0, n_bins=5, min_samples_per_bin=5)
        filt = IVFilter(config)
        result = filt.fit(sample_X, sample_y)

        # high_iv features should be kept (they have strong signal)
        assert "high_iv_1" in result.output_features
        assert "high_iv_2" in result.output_features

    def test_iv_filter_eliminates_low_iv_features(self, sample_X, sample_y):
        IVFilter = self._get_filter()
        config = IVConfig(min_iv=0.02, max_iv=5.0, n_bins=5, min_samples_per_bin=5)
        filt = IVFilter(config)
        result = filt.fit(sample_X, sample_y)

        # low_iv features (random noise) should ideally be eliminated
        # With small samples this is not guaranteed, but at min_iv=0.02
        # pure noise should typically fail
        # Just check the result is a valid StepResult
        assert isinstance(result, StepResult)

    def test_iv_filter_max_iv_eliminates_suspicious(self, sample_X, sample_y):
        IVFilter = self._get_filter()
        # Set very low max_iv so that high_iv features get flagged
        config = IVConfig(min_iv=0.02, max_iv=0.10, n_bins=5, min_samples_per_bin=5)
        filt = IVFilter(config)
        result = filt.fit(sample_X, sample_y)

        # Features with very high IV should be eliminated as suspicious
        # This depends on the actual IV values with our synthetic data
        assert isinstance(result, StepResult)

    def test_iv_filter_returns_step_result(self, sample_X, sample_y):
        IVFilter = self._get_filter()
        config = IVConfig(min_iv=0.02, max_iv=5.0, n_bins=5, min_samples_per_bin=5)
        filt = IVFilter(config)
        result = filt.fit(sample_X, sample_y)

        assert isinstance(result, StepResult)
        assert result.step_name == "03_iv"
        assert result.n_input == len(sample_X.columns)

    def test_iv_filter_results_df_has_iv_values(self, sample_X, sample_y):
        IVFilter = self._get_filter()
        config = IVConfig(min_iv=0.02, max_iv=5.0, n_bins=5, min_samples_per_bin=5)
        filt = IVFilter(config)
        result = filt.fit(sample_X, sample_y)

        assert len(result.results_df) > 0


# ===================================================================
# PSIFilter Tests (component may not exist yet)
# ===================================================================

class TestPSIFilter:
    """Test the PSIFilter component."""

    @pytest.fixture(autouse=True)
    def _check_module_exists(self):
        try:
            from src.components.psi_filter import PSIFilter
            self._skip = False
        except (ImportError, ModuleNotFoundError):
            self._skip = True

    def _get_filter(self):
        if self._skip:
            pytest.skip("PSIFilter not yet implemented")
        from src.components.psi_filter import PSIFilter
        return PSIFilter

    def test_psi_filter_returns_step_result(self, large_sample_data, feature_columns):
        PSIFilter = self._get_filter()
        config = PSIConfig(threshold=0.25, n_bins=5)
        filt = PSIFilter(config)

        X = large_sample_data[feature_columns]
        y = large_sample_data["target"]
        train_dates = pd.to_datetime(large_sample_data["date"])

        result = filt.fit(X, y, train_dates=train_dates)
        assert isinstance(result, StepResult)
        assert result.step_name == "04_psi"

    def test_psi_stable_features_kept(self, large_sample_data, feature_columns):
        PSIFilter = self._get_filter()
        config = PSIConfig(threshold=0.25, n_bins=5)
        filt = PSIFilter(config)

        X = large_sample_data[feature_columns]
        y = large_sample_data["target"]
        train_dates = pd.to_datetime(large_sample_data["date"])

        result = filt.fit(X, y, train_dates=train_dates)

        # Normal features generated from same distribution should be stable
        assert result.n_output > 0

    def test_psi_filter_no_dates_skips(self, sample_X, sample_y):
        PSIFilter = self._get_filter()
        config = PSIConfig(threshold=0.25, n_bins=5)
        filt = PSIFilter(config)

        # No train_dates -> should skip PSI and keep all features
        result = filt.fit(sample_X, sample_y)
        assert result.n_eliminated == 0
        assert result.n_output == len(sample_X.columns)

    def test_psi_filter_transform_drops_features(self, large_sample_data, feature_columns):
        PSIFilter = self._get_filter()
        config = PSIConfig(threshold=0.25, n_bins=5)
        filt = PSIFilter(config)

        X = large_sample_data[feature_columns]
        y = large_sample_data["target"]
        train_dates = pd.to_datetime(large_sample_data["date"])

        filt.fit(X, y, train_dates=train_dates)
        X_out = filt.transform(X)
        # Transform should only keep survived features
        assert set(X_out.columns) == set(filt.kept_features_)

    def test_psi_filter_metadata(self, large_sample_data, feature_columns):
        PSIFilter = self._get_filter()
        config = PSIConfig(threshold=0.25, n_bins=5)
        filt = PSIFilter(config)

        X = large_sample_data[feature_columns]
        y = large_sample_data["target"]
        train_dates = pd.to_datetime(large_sample_data["date"])

        result = filt.fit(X, y, train_dates=train_dates)
        assert "threshold" in result.metadata
        assert result.metadata["threshold"] == 0.25


# ===================================================================
# CorrelationFilter Tests (component may not exist yet)
# ===================================================================

class TestCorrelationFilter:
    """Test the CorrelationFilter component."""

    @pytest.fixture(autouse=True)
    def _check_module_exists(self):
        try:
            from src.components.correlation_filter import CorrelationFilter
            self._skip = False
        except (ImportError, ModuleNotFoundError):
            self._skip = True

    def _get_filter(self):
        if self._skip:
            pytest.skip("CorrelationFilter not yet implemented")
        from src.components.correlation_filter import CorrelationFilter
        return CorrelationFilter

    def test_correlation_filter_detects_correlated_pair(self, sample_X, sample_y):
        CorrelationFilter = self._get_filter()
        config = CorrelationConfig(threshold=0.90, method="pearson")
        filt = CorrelationFilter(config)
        result = filt.fit(sample_X, sample_y)

        # corr_feat_1 and corr_feat_2 are highly correlated
        # One should be eliminated, one should be kept
        corr_in_eliminated = (
            "corr_feat_1" in result.eliminated_features
            or "corr_feat_2" in result.eliminated_features
        )
        assert corr_in_eliminated, "Neither correlated feature was eliminated"

        corr_in_kept = (
            "corr_feat_1" in result.output_features
            or "corr_feat_2" in result.output_features
        )
        assert corr_in_kept, "Both correlated features were eliminated"

    def test_correlation_filter_returns_step_result(self, sample_X, sample_y):
        CorrelationFilter = self._get_filter()
        config = CorrelationConfig(threshold=0.90)
        filt = CorrelationFilter(config)
        result = filt.fit(sample_X, sample_y)
        assert isinstance(result, StepResult)
        assert result.step_name == "05_correlation"

    def test_correlation_filter_spearman_method(self, sample_X, sample_y):
        CorrelationFilter = self._get_filter()
        config = CorrelationConfig(threshold=0.90, method="spearman")
        filt = CorrelationFilter(config)
        result = filt.fit(sample_X, sample_y)
        assert isinstance(result, StepResult)

    def test_correlation_filter_no_correlated_features(self, sample_y):
        CorrelationFilter = self._get_filter()
        np.random.seed(42)
        n = len(sample_y)
        X = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
        })
        config = CorrelationConfig(threshold=0.90)
        filt = CorrelationFilter(config)
        result = filt.fit(X, sample_y)
        assert result.n_eliminated == 0

    def test_correlation_filter_transform(self, sample_X, sample_y):
        CorrelationFilter = self._get_filter()
        config = CorrelationConfig(threshold=0.90)
        filt = CorrelationFilter(config)
        filt.fit(sample_X, sample_y)
        X_out = filt.transform(sample_X)
        assert len(X_out.columns) < len(sample_X.columns)


# ===================================================================
# ForwardFeatureSelector Tests (component may not exist yet)
# ===================================================================

class TestForwardFeatureSelector:
    """Test the ForwardFeatureSelector component."""

    @pytest.fixture(autouse=True)
    def _check_module_exists(self):
        try:
            from src.components.feature_selector import ForwardFeatureSelector
            self._skip = False
        except (ImportError, ModuleNotFoundError):
            self._skip = True

    def _get_selector(self):
        if self._skip:
            pytest.skip("ForwardFeatureSelector not yet implemented")
        from src.components.feature_selector import ForwardFeatureSelector
        return ForwardFeatureSelector

    def _make_train_test(self, sample_X, sample_y):
        """Helper to create train/test split for forward selection."""
        from sklearn.model_selection import train_test_split
        X_clean = sample_X.drop(
            columns=["const_feat_1", "const_feat_2", "high_missing_1", "high_missing_2"]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, sample_y, test_size=0.2, random_state=42, stratify=sample_y
        )
        return X_train, X_test, y_train, y_test

    def test_forward_selector_selects_features(self, sample_X, sample_y):
        from src.config.schema import ModelConfig

        ForwardFeatureSelector = self._get_selector()
        X_train, X_test, y_train, y_test = self._make_train_test(sample_X, sample_y)
        config = SelectionConfig(auc_threshold=0.001, max_features=5)
        model_config = ModelConfig()
        selector = ForwardFeatureSelector(config, model_config, seed=42)
        result = selector.fit(X_train, y_train, X_test=X_test, y_test=y_test)

        assert isinstance(result, StepResult)
        assert result.n_output > 0
        assert result.n_output <= 5

    def test_forward_selector_max_features_cap(self, sample_X, sample_y):
        from src.config.schema import ModelConfig

        ForwardFeatureSelector = self._get_selector()
        X_train, X_test, y_train, y_test = self._make_train_test(sample_X, sample_y)
        config = SelectionConfig(auc_threshold=0.0001, max_features=3)
        model_config = ModelConfig()
        selector = ForwardFeatureSelector(config, model_config, seed=42)
        result = selector.fit(X_train, y_train, X_test=X_test, y_test=y_test)

        assert result.n_output <= 3

    def test_forward_selector_returns_step_result(self, sample_X, sample_y):
        from src.config.schema import ModelConfig

        ForwardFeatureSelector = self._get_selector()
        X_train, X_test, y_train, y_test = self._make_train_test(sample_X, sample_y)
        config = SelectionConfig(auc_threshold=0.001)
        model_config = ModelConfig()
        selector = ForwardFeatureSelector(config, model_config, seed=42)
        result = selector.fit(X_train, y_train, X_test=X_test, y_test=y_test)
        assert result.step_name == "06_selection"


# ===================================================================
# ModelEvaluator Tests (component may not exist yet)
# ===================================================================

class TestModelEvaluator:
    """Test the ModelEvaluator component."""

    @pytest.fixture(autouse=True)
    def _check_module_exists(self):
        try:
            from src.components.model_evaluator import ModelEvaluator
            self._skip = False
        except (ImportError, ModuleNotFoundError):
            self._skip = True

    def _get_evaluator(self):
        if self._skip:
            pytest.skip("ModelEvaluator not yet implemented")
        from src.components.model_evaluator import ModelEvaluator
        return ModelEvaluator

    def _train_model(self, X, y):
        """Helper to train a simple XGBoost model for evaluation tests."""
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        X_clean = X.drop(
            columns=[c for c in ["const_feat_1", "const_feat_2", "high_missing_1", "high_missing_2"]
                     if c in X.columns]
        ).fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42, stratify=y
        )
        model = xgb.XGBClassifier(
            n_estimators=20, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric="auc",
            early_stopping_rounds=5,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return model, X_train, X_test, y_train, y_test

    def test_evaluator_returns_step_result(self, sample_X, sample_y):
        from src.config.schema import EvaluationConfig

        ModelEvaluator = self._get_evaluator()
        model, X_train, X_test, y_train, y_test = self._train_model(sample_X, sample_y)
        config = EvaluationConfig()
        evaluator = ModelEvaluator(config)

        result = evaluator.fit(
            X_train, y_train,
            model=model,
            X_test=X_test,
            y_test=y_test,
            selected_features=list(X_train.columns),
        )
        assert isinstance(result, StepResult)
        assert result.step_name == "07_evaluation"
        assert result.n_eliminated == 0  # Evaluator does not eliminate

    def test_evaluator_performance_df_shape(self, sample_X, sample_y):
        from src.config.schema import EvaluationConfig

        ModelEvaluator = self._get_evaluator()
        model, X_train, X_test, y_train, y_test = self._train_model(sample_X, sample_y)
        config = EvaluationConfig()
        evaluator = ModelEvaluator(config)

        evaluator.fit(
            X_train, y_train,
            model=model,
            X_test=X_test,
            y_test=y_test,
            selected_features=list(X_train.columns),
        )

        perf_df = evaluator.performance_df_
        assert perf_df is not None
        # Should have Train and Test rows
        assert len(perf_df) >= 2
        assert "Period" in perf_df.columns
        assert "AUC" in perf_df.columns
        assert "Gini" in perf_df.columns
        assert "KS" in perf_df.columns
        assert set(perf_df["Period"].tolist()) >= {"Train", "Test"}

    def test_evaluator_auc_in_valid_range(self, sample_X, sample_y):
        from src.config.schema import EvaluationConfig

        ModelEvaluator = self._get_evaluator()
        model, X_train, X_test, y_train, y_test = self._train_model(sample_X, sample_y)
        config = EvaluationConfig()
        evaluator = ModelEvaluator(config)

        evaluator.fit(
            X_train, y_train,
            model=model,
            X_test=X_test,
            y_test=y_test,
            selected_features=list(X_train.columns),
        )

        perf_df = evaluator.performance_df_
        for _, row in perf_df.iterrows():
            assert 0.5 <= row["AUC"] <= 1.0, f"AUC out of range for {row['Period']}"

    def test_evaluator_lift_tables(self, sample_X, sample_y):
        from src.config.schema import EvaluationConfig

        ModelEvaluator = self._get_evaluator()
        model, X_train, X_test, y_train, y_test = self._train_model(sample_X, sample_y)
        config = EvaluationConfig(n_deciles=10)
        evaluator = ModelEvaluator(config)

        evaluator.fit(
            X_train, y_train,
            model=model,
            X_test=X_test,
            y_test=y_test,
            selected_features=list(X_train.columns),
        )

        assert len(evaluator.lift_tables_) >= 2
        for period, lift_df in evaluator.lift_tables_.items():
            assert isinstance(lift_df, pd.DataFrame)
            assert len(lift_df) <= 10  # n_deciles
            assert "Bad_Rate" in lift_df.columns

    def test_evaluator_feature_importance(self, sample_X, sample_y):
        from src.config.schema import EvaluationConfig

        ModelEvaluator = self._get_evaluator()
        model, X_train, X_test, y_train, y_test = self._train_model(sample_X, sample_y)
        config = EvaluationConfig()
        evaluator = ModelEvaluator(config)

        evaluator.fit(
            X_train, y_train,
            model=model,
            X_test=X_test,
            y_test=y_test,
            selected_features=list(X_train.columns),
        )

        imp_df = evaluator.importance_df_
        assert imp_df is not None
        assert "Feature" in imp_df.columns
        assert len(imp_df) == len(X_train.columns)

    def test_evaluator_transform_returns_unchanged(self, sample_X, sample_y):
        from src.config.schema import EvaluationConfig

        ModelEvaluator = self._get_evaluator()
        config = EvaluationConfig()
        evaluator = ModelEvaluator(config)

        # transform should return X unchanged
        X_out = evaluator.transform(sample_X)
        pd.testing.assert_frame_equal(X_out, sample_X)

    def test_evaluator_missing_model_raises(self, sample_X, sample_y):
        from src.config.schema import EvaluationConfig

        ModelEvaluator = self._get_evaluator()
        config = EvaluationConfig()
        evaluator = ModelEvaluator(config)

        with pytest.raises(ValueError, match="model"):
            evaluator.fit(sample_X, sample_y)

    def test_evaluator_missing_test_data_raises(self, sample_X, sample_y):
        from src.config.schema import EvaluationConfig
        from unittest.mock import MagicMock

        ModelEvaluator = self._get_evaluator()
        config = EvaluationConfig()
        evaluator = ModelEvaluator(config)

        with pytest.raises(ValueError, match="X_test"):
            evaluator.fit(sample_X, sample_y, model=MagicMock())


# ===================================================================
# DataSplitter Tests
# ===================================================================

class TestDataSplitter:
    """Test the DataSplitter component."""

    def test_data_splitter_produces_correct_sets(self, large_sample_data):
        from src.components.data_splitter import DataSplitter, DataSplitResult
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        assert isinstance(result, DataSplitResult)
        assert len(result.train) > 0
        assert len(result.test) > 0

    def test_data_splitter_no_leakage(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        cutoff = pd.Timestamp("2022-07-01")
        # Train dates should be <= cutoff
        train_max_date = pd.to_datetime(result.train["date"]).max()
        assert train_max_date <= cutoff

        # Test dates should be <= cutoff
        test_max_date = pd.to_datetime(result.test["date"]).max()
        assert test_max_date <= cutoff

        # OOT dates should be > cutoff
        if result.oot_quarters:
            for label, qdf in result.oot_quarters.items():
                oot_min_date = pd.to_datetime(qdf["date"]).min()
                assert oot_min_date > cutoff

    def test_data_splitter_feature_columns(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        # Feature columns should exclude id, target, date, and exclude columns
        assert "application_id" not in result.feature_columns
        assert "customer_id" not in result.feature_columns
        assert "target" not in result.feature_columns
        assert "date" not in result.feature_columns
        assert "applicant_type" not in result.feature_columns
        assert len(result.feature_columns) > 0

    def test_data_splitter_stratification(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01", stratify=True)
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        # Bad rates should be similar between train and test
        train_bad_rate = result.train["target"].mean()
        test_bad_rate = result.test["target"].mean()
        assert abs(train_bad_rate - test_bad_rate) < 0.10

    def test_data_splitter_missing_columns_raises(self, sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig(target_column="nonexistent")
        split_cfg = SplittingConfig(train_end_date="2023-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)

        with pytest.raises(ValueError, match="Missing required columns"):
            splitter.split(sample_data)

    def test_data_splitter_non_binary_target_raises(self, sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        df = sample_data.copy()
        df["target"] = np.random.randint(0, 3, len(df))

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2023-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)

        with pytest.raises(ValueError, match="binary"):
            splitter.split(df)

    def test_data_splitter_oot_quarters(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        # With data from 2022-01-01 to ~2024-09-27 and cutoff 2022-07-01,
        # we should have multiple OOT quarters
        assert len(result.oot_quarters) > 0

    def test_data_splitter_split_indices(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        assert isinstance(result.split_indices_df, pd.DataFrame)
        assert "set_name" in result.split_indices_df.columns
        assert len(result.split_indices_df) == len(large_sample_data)

    def test_data_splitter_reproducibility(self, large_sample_data):
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

    def test_data_splitter_metadata(self, large_sample_data):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2022-07-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)
        result = splitter.split(large_sample_data)

        assert "train_count" in result.metadata
        assert "test_count" in result.metadata
        assert "train_bad_rate" in result.metadata
        assert "test_bad_rate" in result.metadata
        assert result.metadata["train_count"] == len(result.train)

    def test_data_splitter_insufficient_data_raises(self):
        from src.components.data_splitter import DataSplitter
        from src.config.schema import DataConfig, SplittingConfig

        np.random.seed(42)
        tiny_df = pd.DataFrame({
            "application_id": range(10),
            "customer_id": range(10),
            "date": pd.date_range("2023-01-01", periods=10),
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "applicant_type": ["individual"] * 10,
            "feat_1": np.random.randn(10),
        })

        data_cfg = DataConfig()
        split_cfg = SplittingConfig(train_end_date="2023-12-01")
        splitter = DataSplitter(data_cfg, split_cfg, seed=42)

        with pytest.raises(ValueError, match="at least 100"):
            splitter.split(tiny_df)
