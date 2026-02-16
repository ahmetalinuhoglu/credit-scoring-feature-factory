"""
Tests for src.model_development.subsegment_evaluator

Covers: evaluate_by_subsegment, compute_confusion_by_subsegment.
"""

import pytest
import numpy as np
import pandas as pd

from src.model_development.subsegment_evaluator import (
    evaluate_by_subsegment,
    compute_confusion_by_subsegment,
)


# ===================================================================
# Tests for evaluate_by_subsegment
# ===================================================================

class TestEvaluateBySubsegment:
    def test_returns_dict_of_dataframes(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = evaluate_by_subsegment(
            model=trained_xgb_model,
            datasets=ds,
            selected_features=ds.feature_columns,
            subsegment_columns=["applicant_type"],
            target_column=ds.target_column,
        )
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(val, pd.DataFrame)

    def test_single_subsegment_column(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = evaluate_by_subsegment(
            model=trained_xgb_model,
            datasets=ds,
            selected_features=ds.feature_columns,
            subsegment_columns=["applicant_type"],
            target_column=ds.target_column,
        )
        assert "applicant_type" in result

    def test_result_has_expected_columns(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = evaluate_by_subsegment(
            model=trained_xgb_model,
            datasets=ds,
            selected_features=ds.feature_columns,
            subsegment_columns=["applicant_type"],
            target_column=ds.target_column,
        )
        if "applicant_type" in result:
            df = result["applicant_type"]
            for col in ["Subsegment_Value", "Period", "N_Samples", "AUC"]:
                assert col in df.columns

    def test_subsegment_values_present(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = evaluate_by_subsegment(
            model=trained_xgb_model,
            datasets=ds,
            selected_features=ds.feature_columns,
            subsegment_columns=["applicant_type"],
            target_column=ds.target_column,
        )
        if "applicant_type" in result:
            values = result["applicant_type"]["Subsegment_Value"].unique()
            # Should have at least one of 'new' or 'existing'
            assert len(values) >= 1

    def test_multiple_periods_evaluated(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = evaluate_by_subsegment(
            model=trained_xgb_model,
            datasets=ds,
            selected_features=ds.feature_columns,
            subsegment_columns=["applicant_type"],
            target_column=ds.target_column,
        )
        if "applicant_type" in result:
            periods = result["applicant_type"]["Period"].unique()
            # Should have Train and Test at minimum
            assert len(periods) >= 2

    def test_missing_subsegment_column_handled(self, trained_xgb_model, datasets_fixture):
        """Non-existent column should return empty or skip gracefully."""
        ds = datasets_fixture
        result = evaluate_by_subsegment(
            model=trained_xgb_model,
            datasets=ds,
            selected_features=ds.feature_columns,
            subsegment_columns=["nonexistent_column"],
            target_column=ds.target_column,
        )
        # Should either not have the key or have an empty DataFrame
        if "nonexistent_column" in result:
            assert len(result["nonexistent_column"]) == 0


# ===================================================================
# Tests for compute_confusion_by_subsegment
# ===================================================================

class TestComputeConfusionBySubsegment:
    def test_returns_dict(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = compute_confusion_by_subsegment(
            model=trained_xgb_model,
            datasets=ds,
            selected_features=ds.feature_columns,
            subsegment_columns=["applicant_type"],
            target_column=ds.target_column,
        )
        assert isinstance(result, dict)

    def test_correct_columns(self, trained_xgb_model, datasets_fixture):
        ds = datasets_fixture
        result = compute_confusion_by_subsegment(
            model=trained_xgb_model,
            datasets=ds,
            selected_features=ds.feature_columns,
            subsegment_columns=["applicant_type"],
            target_column=ds.target_column,
            thresholds=[0.3, 0.5],
        )
        if "applicant_type" in result:
            df = result["applicant_type"]
            for col in ["Subsegment_Value", "Period", "Threshold", "TP", "FP", "TN", "FN"]:
                assert col in df.columns
