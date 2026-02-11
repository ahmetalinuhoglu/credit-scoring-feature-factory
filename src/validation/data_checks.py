"""
Pre-Pipeline Data Quality Checks

Validates the input DataFrame before the model development pipeline starts.
Blocks execution on critical failures (e.g. target not binary, no data).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config.schema import PipelineConfig

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class CheckResult:
    """Result of a single validation check."""

    check_name: str
    status: Status
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.INFO
    recommendation: str = ""


@dataclass
class ValidationReport:
    """Collection of check results with convenience methods."""

    checks: List[CheckResult] = field(default_factory=list)

    @property
    def has_critical_failures(self) -> bool:
        return any(
            c.status == Status.FAIL and c.severity == Severity.CRITICAL
            for c in self.checks
        )

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.status == Status.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == Status.FAIL)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.status == Status.WARNING)

    def summary(self) -> str:
        lines = [
            f"Validation Report: {self.pass_count} PASS, "
            f"{self.warning_count} WARNING, {self.fail_count} FAIL"
        ]
        for c in self.checks:
            marker = {"PASS": "+", "FAIL": "X", "WARNING": "!"}[c.status.value]
            lines.append(f"  [{marker}] {c.check_name}: {c.message}")
        return "\n".join(lines)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for Excel reporting."""
        rows = []
        for c in self.checks:
            rows.append({
                "Check_Name": c.check_name,
                "Status": c.status.value,
                "Severity": c.severity.value,
                "Message": c.message,
                "Recommendation": c.recommendation,
            })
        return pd.DataFrame(rows)


class DataValidator:
    """Runs pre-pipeline data quality checks.

    Designed to catch data issues that would cause silent failures or
    misleading results downstream. Critical checks block pipeline execution.
    """

    def __init__(self, config: PipelineConfig):
        self.target_column = config.data.target_column
        self.date_column = config.data.date_column
        self.id_columns = list(config.data.id_columns)

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all data quality checks.

        Args:
            df: The full input DataFrame (before splitting).

        Returns:
            ValidationReport with all check results.
        """
        report = ValidationReport()

        report.add(self._check_not_empty(df))
        report.add(self._check_target_exists(df))

        if self.target_column in df.columns:
            report.add(self._check_target_binary(df))
            report.add(self._check_target_no_nulls(df))
            report.add(self._check_bad_rate_range(df))

        report.add(self._check_date_column(df))
        if self.date_column in df.columns:
            report.add(self._check_date_range(df))

        report.add(self._check_feature_types(df))
        report.add(self._check_duplicate_ids(df))
        report.add(self._check_sample_size(df))

        if self.target_column in df.columns:
            report.add(self._check_leakage(df))

        for c in report.checks:
            level = {
                Status.PASS: logging.INFO,
                Status.WARNING: logging.WARNING,
                Status.FAIL: logging.WARNING,
            }[c.status]
            logger.log(level, "DATA_CHECK | %s | %s | %s", c.status.value, c.check_name, c.message)

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_not_empty(self, df: pd.DataFrame) -> CheckResult:
        if len(df) == 0:
            return CheckResult(
                check_name="Non-empty dataset",
                status=Status.FAIL,
                message="DataFrame is empty (0 rows).",
                severity=Severity.CRITICAL,
                recommendation="Provide a DataFrame with at least one row.",
            )
        return CheckResult(
            check_name="Non-empty dataset",
            status=Status.PASS,
            message=f"Dataset has {len(df):,} rows.",
            severity=Severity.CRITICAL,
        )

    def _check_target_exists(self, df: pd.DataFrame) -> CheckResult:
        if self.target_column not in df.columns:
            return CheckResult(
                check_name="Target column exists",
                status=Status.FAIL,
                message=f"Target column '{self.target_column}' not found.",
                severity=Severity.CRITICAL,
                recommendation=f"Ensure the target column '{self.target_column}' is present.",
            )
        return CheckResult(
            check_name="Target column exists",
            status=Status.PASS,
            message=f"Target column '{self.target_column}' present.",
            severity=Severity.CRITICAL,
        )

    def _check_target_binary(self, df: pd.DataFrame) -> CheckResult:
        unique_values = df[self.target_column].dropna().unique()
        is_binary = set(unique_values).issubset({0, 1, 0.0, 1.0})
        if not is_binary:
            return CheckResult(
                check_name="Target is binary",
                status=Status.FAIL,
                message=f"Target has non-binary values: {sorted(unique_values)[:10]}.",
                severity=Severity.CRITICAL,
                details={"unique_values": sorted(unique_values)[:20]},
                recommendation="Target must contain only 0 and 1.",
            )
        return CheckResult(
            check_name="Target is binary",
            status=Status.PASS,
            message="Target is binary (0/1).",
            severity=Severity.CRITICAL,
        )

    def _check_target_no_nulls(self, df: pd.DataFrame) -> CheckResult:
        null_count = df[self.target_column].isna().sum()
        if null_count > 0:
            return CheckResult(
                check_name="Target has no nulls",
                status=Status.FAIL,
                message=f"Target has {null_count:,} null values ({null_count/len(df):.2%}).",
                severity=Severity.CRITICAL,
                details={"null_count": int(null_count)},
                recommendation="Remove or impute null target values before running the pipeline.",
            )
        return CheckResult(
            check_name="Target has no nulls",
            status=Status.PASS,
            message="No null values in target.",
            severity=Severity.CRITICAL,
        )

    def _check_bad_rate_range(self, df: pd.DataFrame) -> CheckResult:
        bad_rate = df[self.target_column].mean()
        if bad_rate < 0.01:
            return CheckResult(
                check_name="Bad rate within range",
                status=Status.FAIL,
                message=f"Bad rate {bad_rate:.4%} is below 1%. Model may not have enough signal.",
                severity=Severity.CRITICAL,
                details={"bad_rate": float(bad_rate)},
                recommendation="Increase observation window or adjust default definition.",
            )
        if bad_rate > 0.50:
            return CheckResult(
                check_name="Bad rate within range",
                status=Status.FAIL,
                message=f"Bad rate {bad_rate:.4%} exceeds 50%. Check target definition.",
                severity=Severity.CRITICAL,
                details={"bad_rate": float(bad_rate)},
                recommendation="Verify the target definition is correct.",
            )
        if bad_rate < 0.03 or bad_rate > 0.40:
            return CheckResult(
                check_name="Bad rate within range",
                status=Status.WARNING,
                message=f"Bad rate {bad_rate:.4%} is unusual. Typical range is 3%-40%.",
                severity=Severity.WARNING,
                details={"bad_rate": float(bad_rate)},
                recommendation="Review target definition and data completeness.",
            )
        return CheckResult(
            check_name="Bad rate within range",
            status=Status.PASS,
            message=f"Bad rate {bad_rate:.4%} is within acceptable range.",
            details={"bad_rate": float(bad_rate)},
            severity=Severity.WARNING,
        )

    def _check_date_column(self, df: pd.DataFrame) -> CheckResult:
        if self.date_column not in df.columns:
            return CheckResult(
                check_name="Date column exists",
                status=Status.FAIL,
                message=f"Date column '{self.date_column}' not found.",
                severity=Severity.CRITICAL,
                recommendation=f"Ensure '{self.date_column}' is present for train/OOT splitting.",
            )
        try:
            pd.to_datetime(df[self.date_column])
        except Exception:
            return CheckResult(
                check_name="Date column exists",
                status=Status.FAIL,
                message=f"Date column '{self.date_column}' cannot be parsed as datetime.",
                severity=Severity.CRITICAL,
                recommendation="Convert to a standard date format (YYYY-MM-DD).",
            )
        return CheckResult(
            check_name="Date column exists",
            status=Status.PASS,
            message=f"Date column '{self.date_column}' present and parseable.",
            severity=Severity.CRITICAL,
        )

    def _check_date_range(self, df: pd.DataFrame) -> CheckResult:
        dates = pd.to_datetime(df[self.date_column], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates) == 0:
            return CheckResult(
                check_name="Date range coverage",
                status=Status.FAIL,
                message="No valid dates found.",
                severity=Severity.CRITICAL,
                recommendation="Check date column format.",
            )
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        n_quarters = ((max_date.year - min_date.year) * 4
                      + (max_date.quarter - min_date.quarter) + 1)
        if n_quarters < 2:
            return CheckResult(
                check_name="Date range coverage",
                status=Status.FAIL,
                message=f"Date range covers only {n_quarters} quarter(s). Need at least 2.",
                severity=Severity.CRITICAL,
                details={"min_date": str(min_date.date()), "max_date": str(max_date.date()),
                         "quarters": int(n_quarters)},
                recommendation="Provide data spanning at least 2 quarters for OOT evaluation.",
            )
        return CheckResult(
            check_name="Date range coverage",
            status=Status.PASS,
            message=f"Date range covers {n_quarters} quarters ({min_date.date()} to {max_date.date()}).",
            details={"min_date": str(min_date.date()), "max_date": str(max_date.date()),
                     "quarters": int(n_quarters)},
            severity=Severity.WARNING,
        )

    def _check_feature_types(self, df: pd.DataFrame) -> CheckResult:
        exclude = set(self.id_columns + [self.target_column, self.date_column])
        feature_cols = [c for c in df.columns if c not in exclude]
        non_numeric = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            return CheckResult(
                check_name="Features are numeric",
                status=Status.WARNING,
                message=f"{len(non_numeric)} non-numeric feature(s) found.",
                severity=Severity.WARNING,
                details={"non_numeric_features": non_numeric[:20]},
                recommendation="Encode or remove non-numeric features before pipeline.",
            )
        return CheckResult(
            check_name="Features are numeric",
            status=Status.PASS,
            message=f"All {len(feature_cols)} features are numeric.",
            severity=Severity.WARNING,
        )

    def _check_duplicate_ids(self, df: pd.DataFrame) -> CheckResult:
        id_col = self.id_columns[0] if self.id_columns else None
        if id_col is None or id_col not in df.columns:
            return CheckResult(
                check_name="No duplicate IDs",
                status=Status.WARNING,
                message="Primary ID column not found; skipping duplicate check.",
                severity=Severity.WARNING,
            )
        dup_count = df[id_col].duplicated().sum()
        if dup_count > 0:
            return CheckResult(
                check_name="No duplicate IDs",
                status=Status.WARNING,
                message=f"{dup_count:,} duplicate {id_col} values found.",
                severity=Severity.WARNING,
                details={"duplicate_count": int(dup_count)},
                recommendation="Deduplicate or verify that duplicates are intentional.",
            )
        return CheckResult(
            check_name="No duplicate IDs",
            status=Status.PASS,
            message=f"No duplicate {id_col} values.",
            severity=Severity.INFO,
        )

    def _check_sample_size(self, df: pd.DataFrame) -> CheckResult:
        n = len(df)
        if n < 500:
            return CheckResult(
                check_name="Sufficient sample size",
                status=Status.WARNING,
                message=f"Only {n:,} rows. Recommend >= 500 for reliable modeling.",
                severity=Severity.WARNING,
                details={"row_count": n},
                recommendation="Collect more data or widen observation window.",
            )
        return CheckResult(
            check_name="Sufficient sample size",
            status=Status.PASS,
            message=f"{n:,} rows available.",
            details={"row_count": n},
            severity=Severity.INFO,
        )

    def _check_leakage(self, df: pd.DataFrame) -> CheckResult:
        """Flag features with suspiciously high single-feature AUC (>0.95)."""
        exclude = set(self.id_columns + [self.target_column, self.date_column])
        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not feature_cols:
            return CheckResult(
                check_name="Leakage detection",
                status=Status.PASS,
                message="No numeric features to check.",
                severity=Severity.WARNING,
            )

        y = df[self.target_column].values
        if len(np.unique(y[~np.isnan(y)])) < 2:
            return CheckResult(
                check_name="Leakage detection",
                status=Status.WARNING,
                message="Cannot check leakage: target has fewer than 2 classes.",
                severity=Severity.WARNING,
            )

        # Sample top-50 features by variance for speed
        variances = df[feature_cols].var().sort_values(ascending=False)
        candidates = variances.head(50).index.tolist()

        suspicious: List[str] = []
        for col in candidates:
            vals = df[col].values
            mask = ~(np.isnan(vals) | np.isnan(y))
            if mask.sum() < 50:
                continue
            try:
                auc = roc_auc_score(y[mask], vals[mask])
                auc = max(auc, 1 - auc)  # handle inverted
                if auc > 0.95:
                    suspicious.append(col)
            except Exception:
                continue

        if suspicious:
            return CheckResult(
                check_name="Leakage detection",
                status=Status.WARNING,
                message=f"{len(suspicious)} feature(s) with AUC > 0.95: possible leakage.",
                severity=Severity.WARNING,
                details={"suspicious_features": suspicious},
                recommendation="Investigate these features for target leakage or data errors.",
            )
        return CheckResult(
            check_name="Leakage detection",
            status=Status.PASS,
            message="No features with suspiciously high AUC detected.",
            severity=Severity.WARNING,
        )
