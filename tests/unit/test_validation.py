"""
Unit Tests for Validation Checks

Tests data quality validation (DataValidator), model quality validation
(ModelValidator), and the ValidationReport / CheckResult containers.
"""

import pytest
import numpy as np
import pandas as pd

from src.validation.data_checks import (
    DataValidator,
    ValidationReport,
    CheckResult,
    Severity,
    Status,
)
from src.validation.model_checks import ModelValidator
from src.config.schema import PipelineConfig


# ===================================================================
# CheckResult Tests
# ===================================================================

class TestCheckResult:
    """Test the CheckResult data structure."""

    def test_check_result_creation(self):
        result = CheckResult(
            check_name="test_check",
            status=Status.PASS,
            message="All good",
            severity=Severity.INFO,
        )
        assert result.check_name == "test_check"
        assert result.status == Status.PASS

    def test_check_result_failed(self):
        result = CheckResult(
            check_name="bad_check",
            status=Status.FAIL,
            message="Failed badly",
            severity=Severity.CRITICAL,
        )
        assert result.status == Status.FAIL
        assert result.severity == Severity.CRITICAL

    def test_check_result_with_details(self):
        result = CheckResult(
            check_name="detail_check",
            status=Status.WARNING,
            message="Warning",
            severity=Severity.WARNING,
            details={"key": "value"},
            recommendation="Fix it",
        )
        assert result.details["key"] == "value"
        assert result.recommendation == "Fix it"


# ===================================================================
# ValidationReport Tests
# ===================================================================

class TestValidationReport:
    """Test the ValidationReport container."""

    def test_report_creation(self):
        report = ValidationReport()
        assert isinstance(report, ValidationReport)
        assert len(report.checks) == 0

    def test_report_has_critical_failures_false(self):
        report = ValidationReport()
        report.add(CheckResult(
            check_name="passing_check",
            status=Status.PASS,
            message="OK",
            severity=Severity.CRITICAL,
        ))
        assert report.has_critical_failures is False

    def test_report_has_critical_failures_true(self):
        report = ValidationReport()
        report.add(CheckResult(
            check_name="critical_fail",
            status=Status.FAIL,
            message="Bad",
            severity=Severity.CRITICAL,
        ))
        assert report.has_critical_failures is True

    def test_report_mixed_results(self):
        report = ValidationReport()
        report.add(CheckResult("pass1", Status.PASS, "OK", severity=Severity.INFO))
        report.add(CheckResult("warning1", Status.WARNING, "Warn", severity=Severity.WARNING))
        report.add(CheckResult("critical1", Status.FAIL, "Fail", severity=Severity.CRITICAL))
        assert report.has_critical_failures is True
        assert report.pass_count == 1
        assert report.warning_count == 1
        assert report.fail_count == 1

    def test_report_no_critical_with_warning_failures(self):
        report = ValidationReport()
        report.add(CheckResult("warn_fail", Status.FAIL, "Warning level", severity=Severity.WARNING))
        assert report.has_critical_failures is False

    def test_report_summary(self):
        report = ValidationReport()
        report.add(CheckResult("check1", Status.PASS, "OK"))
        summary = report.summary()
        assert "1 PASS" in summary
        assert "check1" in summary

    def test_report_to_dataframe(self):
        report = ValidationReport()
        report.add(CheckResult("check1", Status.PASS, "OK", severity=Severity.INFO))
        report.add(CheckResult("check2", Status.FAIL, "Fail", severity=Severity.CRITICAL))
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Check_Name" in df.columns


# ===================================================================
# DataValidator Tests
# ===================================================================

class TestDataValidator:
    """Test pre-pipeline data quality checks."""

    @pytest.fixture
    def validator(self):
        config = PipelineConfig()
        return DataValidator(config)

    def test_non_binary_target_detected(self, validator):
        df = pd.DataFrame({
            "application_id": range(5),
            "customer_id": range(5),
            "date": pd.date_range("2023-01-01", periods=5),
            "target": [0, 1, 2, 3, 4],
            "feat": [1, 2, 3, 4, 5],
        })
        report = validator.validate(df)
        target_checks = [c for c in report.checks if "binary" in c.check_name.lower()]
        assert any(c.status == Status.FAIL for c in target_checks)

    def test_null_targets_detected(self, validator):
        df = pd.DataFrame({
            "application_id": range(5),
            "customer_id": range(5),
            "date": pd.date_range("2023-01-01", periods=5),
            "target": [0, 1, np.nan, 0, 1],
            "feat": [1, 2, 3, 4, 5],
        })
        report = validator.validate(df)
        null_checks = [c for c in report.checks if "null" in c.check_name.lower()]
        assert any(c.status == Status.FAIL for c in null_checks)

    def test_duplicate_ids_detected(self, validator):
        df = pd.DataFrame({
            "application_id": [1, 1, 2, 3, 3],
            "customer_id": range(5),
            "date": pd.date_range("2023-01-01", periods=5),
            "target": [0, 1, 0, 1, 0],
            "feat": [1, 2, 3, 4, 5],
        })
        report = validator.validate(df)
        dup_checks = [c for c in report.checks if "duplicate" in c.check_name.lower()]
        assert any(c.status in (Status.FAIL, Status.WARNING) for c in dup_checks)

    def test_valid_data_passes(self, validator):
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "application_id": range(n),
            "customer_id": range(n),
            "date": pd.date_range("2022-01-01", periods=n, freq="1D"),
            "target": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
        })
        report = validator.validate(df)
        assert report.has_critical_failures is False

    def test_empty_dataframe_fails(self, validator):
        df = pd.DataFrame(columns=["application_id", "customer_id", "date", "target"])
        report = validator.validate(df)
        assert report.has_critical_failures is True

    def test_leakage_detection(self, validator):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n, p=[0.8, 0.2])
        df = pd.DataFrame({
            "application_id": range(n),
            "customer_id": range(n),
            "date": pd.date_range("2022-01-01", periods=n),
            "target": target,
            # Feature that is essentially the target -> leakage
            "leaky_feat": target * 10 + np.random.randn(n) * 0.01,
            "normal_feat": np.random.randn(n),
        })
        report = validator.validate(df)
        leakage_checks = [c for c in report.checks if "leakage" in c.check_name.lower()]
        assert len(leakage_checks) > 0
        # The leaky feature should trigger a warning
        leakage_results = [c for c in leakage_checks if c.status in (Status.FAIL, Status.WARNING)]
        assert len(leakage_results) > 0


# ===================================================================
# ModelValidator Tests
# ===================================================================

class TestModelValidator:
    """Test post-pipeline model quality checks."""

    @pytest.fixture
    def validator(self):
        config = PipelineConfig()
        return ModelValidator(config)

    @pytest.fixture
    def good_performance_df(self):
        return pd.DataFrame({
            "Period": ["Train", "Test", "OOT_2024Q3", "OOT_2024Q4"],
            "AUC": [0.85, 0.83, 0.81, 0.80],
            "Gini": [0.70, 0.66, 0.62, 0.60],
            "KS": [0.50, 0.48, 0.45, 0.44],
            "N": [5000, 1250, 800, 750],
            "N_Bad": [850, 213, 136, 128],
        })

    @pytest.fixture
    def importance_df(self):
        return pd.DataFrame({
            "Feature": ["feat_a", "feat_b", "feat_c"],
            "Importance": [0.3, 0.3, 0.4],
        })

    def test_good_model_passes(self, validator, good_performance_df, importance_df):
        report = validator.validate(good_performance_df, importance_df)
        assert report.has_critical_failures is False

    def test_overfit_detection(self, validator, importance_df):
        perf = pd.DataFrame({
            "Period": ["Train", "Test"],
            "AUC": [0.95, 0.80],
            "Gini": [0.90, 0.60],
            "KS": [0.60, 0.40],
            "N": [5000, 1250],
            "N_Bad": [850, 213],
        })
        report = validator.validate(perf, importance_df)
        overfit_checks = [c for c in report.checks if "overfit" in c.check_name.lower()]
        assert any(c.status == Status.WARNING for c in overfit_checks)

    def test_no_overfit_passes(self, validator, importance_df):
        perf = pd.DataFrame({
            "Period": ["Train", "Test"],
            "AUC": [0.85, 0.83],
            "Gini": [0.70, 0.66],
            "KS": [0.50, 0.48],
            "N": [5000, 1250],
            "N_Bad": [850, 213],
        })
        report = validator.validate(perf, importance_df)
        overfit_checks = [c for c in report.checks if "overfit" in c.check_name.lower()]
        assert all(c.status != Status.FAIL for c in overfit_checks)

    def test_oot_degradation_detection(self, validator, importance_df):
        perf = pd.DataFrame({
            "Period": ["Train", "Test", "OOT_2024Q3"],
            "AUC": [0.85, 0.83, 0.70],
            "Gini": [0.70, 0.66, 0.40],
            "KS": [0.50, 0.48, 0.30],
            "N": [5000, 1250, 800],
            "N_Bad": [850, 213, 136],
        })
        report = validator.validate(perf, importance_df)
        oot_checks = [c for c in report.checks if "oot" in c.check_name.lower() and "stability" in c.check_name.lower()]
        assert any(c.status == Status.WARNING for c in oot_checks)

    def test_min_auc_check(self, validator, importance_df):
        perf = pd.DataFrame({
            "Period": ["Train", "Test"],
            "AUC": [0.60, 0.58],
            "Gini": [0.20, 0.16],
            "KS": [0.15, 0.12],
            "N": [5000, 1250],
            "N_Bad": [850, 213],
        })
        report = validator.validate(perf, importance_df)
        auc_checks = [c for c in report.checks if "discrimination" in c.check_name.lower() and "auc" in c.check_name.lower()]
        assert any(c.status == Status.FAIL for c in auc_checks)

    def test_feature_concentration_warning(self, validator, good_performance_df):
        # One feature dominates
        concentrated_importance = pd.DataFrame({
            "Feature": ["feat_a", "feat_b", "feat_c"],
            "Importance": [0.9, 0.05, 0.05],
        })
        report = validator.validate(good_performance_df, concentrated_importance)
        conc_checks = [c for c in report.checks if "concentration" in c.check_name.lower()]
        assert any(c.status == Status.WARNING for c in conc_checks)

    def test_empty_performance_data(self, validator, importance_df):
        perf = pd.DataFrame()
        report = validator.validate(perf, importance_df)
        assert report.has_critical_failures is True

    def test_score_psi_check(self, validator, good_performance_df, importance_df):
        psi_df = pd.DataFrame({
            "Period": ["OOT_2024Q3", "OOT_2024Q4"],
            "PSI": [0.05, 0.10],
        })
        report = validator.validate(good_performance_df, importance_df, score_psi_df=psi_df)
        psi_checks = [c for c in report.checks if "psi" in c.check_name.lower()]
        assert any(c.status == Status.PASS for c in psi_checks)

    def test_high_score_psi_warning(self, validator, good_performance_df, importance_df):
        psi_df = pd.DataFrame({
            "Period": ["OOT_2024Q3"],
            "PSI": [0.50],
        })
        report = validator.validate(good_performance_df, importance_df, score_psi_df=psi_df)
        psi_checks = [c for c in report.checks if "psi" in c.check_name.lower()]
        assert any(c.status == Status.WARNING for c in psi_checks)
