"""
Tests for src.model_development.evaluator

Covers: calculate_metrics, evaluate_model_quarterly, compute_quarterly_trend,
compute_confusion_metrics, bootstrap_auc_ci.
"""

import pytest
import numpy as np
import pandas as pd

from src.model_development.evaluator import (
    calculate_metrics,
    evaluate_model_quarterly,
    evaluate_model_summary,
    evaluate_quarterly_chronological,
    compute_variable_quarterly_auc,
    compute_quarterly_trend,
    compute_confusion_metrics,
    bootstrap_auc_ci,
)


# ===================================================================
# calculate_metrics
# ===================================================================

class TestCalculateMetrics:
    def test_returns_auc_gini_ks(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.choice([0, 1], n, p=[0.8, 0.2])
        y_score = np.where(
            y_true == 1,
            rng.beta(5, 2, n),
            rng.beta(2, 5, n),
        )
        metrics = calculate_metrics(y_true, y_score)
        assert "auc" in metrics
        assert "gini" in metrics
        assert "ks" in metrics

    def test_auc_between_0_and_1(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.choice([0, 1], n, p=[0.8, 0.2])
        y_score = rng.rand(n)
        metrics = calculate_metrics(y_true, y_score)
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_gini_is_2auc_minus_1(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.choice([0, 1], n, p=[0.8, 0.2])
        y_score = rng.rand(n)
        metrics = calculate_metrics(y_true, y_score)
        expected_gini = 2 * metrics["auc"] - 1
        assert abs(metrics["gini"] - expected_gini) < 1e-3

    def test_ks_between_0_and_1(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.choice([0, 1], n, p=[0.8, 0.2])
        y_score = rng.rand(n)
        metrics = calculate_metrics(y_true, y_score)
        assert 0.0 <= metrics["ks"] <= 1.0

    def test_perfect_model_high_auc(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        metrics = calculate_metrics(y_true, y_score)
        assert metrics["auc"] == 1.0

    def test_metrics_are_rounded(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.choice([0, 1], n, p=[0.8, 0.2])
        y_score = rng.rand(n)
        metrics = calculate_metrics(y_true, y_score)
        # Values should be rounded to 4 decimal places
        for key in ["auc", "gini", "ks"]:
            str_val = str(metrics[key])
            if "." in str_val:
                decimals = len(str_val.split(".")[1])
                assert decimals <= 4


# ===================================================================
# evaluate_model_quarterly
# ===================================================================

class TestEvaluateModelQuarterly:
    def test_returns_three_outputs(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        perf_df, lift_tables, imp_df = evaluate_model_quarterly(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            train_df=ds.train,
            test_df=ds.test,
            oot_quarters=ds.oot_quarters,
            target_column=ds.target_column,
        )
        assert isinstance(perf_df, pd.DataFrame)
        assert isinstance(lift_tables, dict)
        assert isinstance(imp_df, pd.DataFrame)

    def test_performance_df_has_train_test(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        perf_df, _, _ = evaluate_model_quarterly(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            train_df=ds.train,
            test_df=ds.test,
            oot_quarters=ds.oot_quarters,
            target_column=ds.target_column,
        )
        periods = perf_df["Period"].tolist()
        assert "Train" in periods
        assert "Test" in periods

    def test_performance_df_columns(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        perf_df, _, _ = evaluate_model_quarterly(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            train_df=ds.train,
            test_df=ds.test,
            oot_quarters=ds.oot_quarters,
            target_column=ds.target_column,
        )
        for col in ["Period", "AUC", "Gini", "KS", "N_Samples"]:
            assert col in perf_df.columns

    def test_importance_df_has_rank(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        _, _, imp_df = evaluate_model_quarterly(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            train_df=ds.train,
            test_df=ds.test,
            oot_quarters=ds.oot_quarters,
            target_column=ds.target_column,
        )
        assert "Rank" in imp_df.columns
        assert "Feature" in imp_df.columns


# ===================================================================
# compute_quarterly_trend
# ===================================================================

class TestComputeQuarterlyTrend:
    def test_returns_trend_df(self, performance_df_fixture):
        trend_df = compute_quarterly_trend(performance_df_fixture)
        assert isinstance(trend_df, pd.DataFrame)
        assert len(trend_df) > 0

    def test_trend_columns(self, performance_df_fixture):
        trend_df = compute_quarterly_trend(performance_df_fixture)
        for col in ["Period", "AUC", "Delta_AUC", "AUC_vs_Train", "Trend"]:
            assert col in trend_df.columns

    def test_trend_labels(self, performance_df_fixture):
        trend_df = compute_quarterly_trend(performance_df_fixture)
        valid_trends = {"Improving", "Degrading", "Stable"}
        for t in trend_df["Trend"]:
            assert t in valid_trends

    def test_empty_when_no_oot(self):
        df = pd.DataFrame({
            "Period": ["Train", "Test"],
            "AUC": [0.85, 0.82],
            "Gini": [0.70, 0.64],
            "KS": [0.55, 0.52],
        })
        trend_df = compute_quarterly_trend(df)
        assert len(trend_df) == 0


# ===================================================================
# compute_confusion_metrics
# ===================================================================

class TestComputeConfusionMetrics:
    def test_returns_dataframe(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 100, p=[0.8, 0.2])
        y_prob = rng.rand(100)
        cm_df = compute_confusion_metrics(y_true, y_prob)
        assert isinstance(cm_df, pd.DataFrame)

    def test_default_thresholds(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 100, p=[0.8, 0.2])
        y_prob = rng.rand(100)
        cm_df = compute_confusion_metrics(y_true, y_prob)
        assert len(cm_df) == 5  # Default 5 thresholds

    def test_custom_thresholds(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 100, p=[0.8, 0.2])
        y_prob = rng.rand(100)
        cm_df = compute_confusion_metrics(y_true, y_prob, thresholds=[0.3, 0.5])
        assert len(cm_df) == 2

    def test_columns_present(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 100, p=[0.8, 0.2])
        y_prob = rng.rand(100)
        cm_df = compute_confusion_metrics(y_true, y_prob)
        for col in ["Threshold", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1", "Accuracy"]:
            assert col in cm_df.columns

    def test_tp_fp_tn_fn_sum_to_n(self):
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.choice([0, 1], n, p=[0.8, 0.2])
        y_prob = rng.rand(n)
        cm_df = compute_confusion_metrics(y_true, y_prob, thresholds=[0.5])
        row = cm_df.iloc[0]
        assert row["TP"] + row["FP"] + row["TN"] + row["FN"] == n


# ===================================================================
# bootstrap_auc_ci
# ===================================================================

class TestBootstrapAUCCI:
    def test_returns_dataframe(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        datasets_list = [("Train", ds.train), ("Test", ds.test)]
        ci_df = bootstrap_auc_ci(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            datasets=datasets_list,
            target_column=ds.target_column,
            n_iterations=20,
            n_jobs=1,
        )
        assert isinstance(ci_df, pd.DataFrame)

    def test_columns_present(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        datasets_list = [("Train", ds.train)]
        ci_df = bootstrap_auc_ci(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            datasets=datasets_list,
            target_column=ds.target_column,
            n_iterations=20,
            n_jobs=1,
        )
        for col in ["Period", "AUC", "CI_Lower", "CI_Upper"]:
            assert col in ci_df.columns

    def test_ci_lower_leq_upper(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        datasets_list = [("Train", ds.train)]
        ci_df = bootstrap_auc_ci(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            datasets=datasets_list,
            target_column=ds.target_column,
            n_iterations=50,
            n_jobs=1,
        )
        for _, row in ci_df.iterrows():
            if row["CI_Lower"] is not None and row["CI_Upper"] is not None:
                assert row["CI_Lower"] <= row["CI_Upper"]


# ===================================================================
# evaluate_model_summary
# ===================================================================

class TestEvaluateModelSummary:
    def test_returns_three_rows(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        perf_df, lift_tables, imp_df = evaluate_model_summary(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            train_df=ds.train,
            test_df=ds.test,
            oot_quarters=ds.oot_quarters,
            target_column=ds.target_column,
        )
        periods = perf_df["Period"].tolist()
        assert "Train" in periods
        assert "Test" in periods
        assert "OOT" in periods
        # No OOT_Q3 etc. — only combined OOT
        assert not any(p.startswith("OOT_") for p in periods)

    def test_columns_match_quarterly(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        perf_df, _, _ = evaluate_model_summary(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            train_df=ds.train,
            test_df=ds.test,
            oot_quarters=ds.oot_quarters,
            target_column=ds.target_column,
        )
        for col in ["Period", "AUC", "Gini", "KS", "N_Samples"]:
            assert col in perf_df.columns


# ===================================================================
# evaluate_quarterly_chronological
# ===================================================================

class TestEvaluateQuarterlyChronological:
    def test_returns_dataframe(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = evaluate_quarterly_chronological(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            datasets=ds,
            target_column=ds.target_column,
            date_column=ds.date_column,
        )
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = evaluate_quarterly_chronological(
            model=trained_xgb_model,
            selected_features=ds.feature_columns,
            datasets=ds,
            target_column=ds.target_column,
            date_column=ds.date_column,
        )
        for col in ["Quarter", "N_Samples", "N_Bads", "Bad_Rate", "AUC", "Gini", "KS"]:
            assert col in result.columns


# ===================================================================
# compute_variable_quarterly_auc
# ===================================================================

class TestComputeVariableQuarterlyAUC:
    def test_returns_dataframe(self, datasets_fixture):
        ds = datasets_fixture
        result = compute_variable_quarterly_auc(
            features=ds.feature_columns,
            datasets=ds,
            target_column=ds.target_column,
            date_column=ds.date_column,
            n_jobs=1,
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_summary_columns(self, datasets_fixture):
        ds = datasets_fixture
        result = compute_variable_quarterly_auc(
            features=ds.feature_columns,
            datasets=ds,
            target_column=ds.target_column,
            date_column=ds.date_column,
            n_jobs=1,
        )
        assert "Feature" in result.columns
        assert "Avg_AUC" in result.columns
        assert "Min_AUC" in result.columns
        assert "Trend_Slope" in result.columns

    def test_all_features_present(self, datasets_fixture):
        ds = datasets_fixture
        result = compute_variable_quarterly_auc(
            features=ds.feature_columns,
            datasets=ds,
            target_column=ds.target_column,
            date_column=ds.date_column,
            n_jobs=1,
        )
        assert set(result["Feature"].tolist()) == set(ds.feature_columns)
