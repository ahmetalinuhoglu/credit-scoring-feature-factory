"""
Post-Pipeline Model Quality Checks

Validates model performance, stability, and risk metrics after evaluation.
Produces a ValidationReport with pass/fail/warning per check.
"""

from typing import Any, Dict, List, Optional

import logging

import numpy as np
import pandas as pd

from src.config.schema import PipelineConfig, ValidationChecksConfig
from src.validation.data_checks import (
    CheckResult,
    Severity,
    Status,
    ValidationReport,
)

logger = logging.getLogger(__name__)


class ModelValidator:
    """Runs post-pipeline model quality checks.

    Evaluates discrimination, stability, overfitting, concentration,
    monotonicity, and sample adequacy against configurable thresholds.
    """

    def __init__(self, config: PipelineConfig):
        checks_cfg: ValidationChecksConfig = config.validation.checks
        self.min_auc = checks_cfg.min_auc
        self.max_overfit_gap = checks_cfg.max_overfit_gap
        self.max_oot_degradation = checks_cfg.max_oot_degradation
        self.max_score_psi = checks_cfg.max_score_psi
        self.max_feature_concentration = checks_cfg.max_feature_concentration
        self.min_oot_samples = checks_cfg.min_oot_samples
        self.check_monotonicity = checks_cfg.check_monotonicity

    def validate(
        self,
        performance_df: pd.DataFrame,
        importance_df: pd.DataFrame,
        woe_bins: Optional[Dict[str, pd.DataFrame]] = None,
        score_psi_df: Optional[pd.DataFrame] = None,
        oot_details: Optional[pd.DataFrame] = None,
    ) -> ValidationReport:
        """Run all model quality checks.

        Args:
            performance_df: Per-period metrics with columns Period, AUC, Gini,
                KS, N, N_Bad at minimum.
            importance_df: Feature importance with columns Feature, Importance.
            woe_bins: Optional dict of feature -> WoE bin DataFrame with
                columns Bin, WoE (for monotonicity check).
            score_psi_df: Optional DataFrame with columns Period, PSI.
            oot_details: Optional DataFrame with OOT quarter details
                (Period, N, N_Bad).

        Returns:
            ValidationReport with all check results.
        """
        report = ValidationReport()

        if performance_df is None or performance_df.empty:
            report.add(CheckResult(
                check_name="Performance data",
                status=Status.FAIL,
                message="No performance data provided.",
                severity=Severity.CRITICAL,
                recommendation="Run the evaluation step before validation.",
            ))
            return report

        report.add(self._check_discrimination(performance_df))
        report.add(self._check_gini(performance_df))
        report.add(self._check_overfit(performance_df))
        report.add(self._check_oot_stability(performance_df))

        if score_psi_df is not None and not score_psi_df.empty:
            report.add(self._check_score_psi(score_psi_df))

        if importance_df is not None and not importance_df.empty:
            report.add(self._check_concentration(importance_df))

        if self.check_monotonicity and woe_bins:
            report.add(self._check_monotonicity(woe_bins))

        oot_src = oot_details if oot_details is not None else performance_df
        report.add(self._check_oot_sample_size(oot_src))

        for c in report.checks:
            level = {
                Status.PASS: logging.INFO,
                Status.WARNING: logging.WARNING,
                Status.FAIL: logging.WARNING,
            }[c.status]
            logger.log(level, "MODEL_CHECK | %s | %s | %s",
                       c.status.value, c.check_name, c.message)

        return report

    # ------------------------------------------------------------------
    # Discrimination
    # ------------------------------------------------------------------

    def _check_discrimination(self, perf: pd.DataFrame) -> CheckResult:
        """AUC >= min_auc for every period."""
        if "AUC" not in perf.columns:
            return CheckResult(
                check_name="Discrimination (AUC)",
                status=Status.FAIL,
                message="AUC column missing from performance data.",
                severity=Severity.CRITICAL,
            )

        failing_periods = []
        for _, row in perf.iterrows():
            auc = row["AUC"]
            if pd.notna(auc) and auc < self.min_auc:
                failing_periods.append(
                    {"period": str(row.get("Period", "?")), "auc": round(float(auc), 4)}
                )

        if failing_periods:
            return CheckResult(
                check_name="Discrimination (AUC)",
                status=Status.FAIL,
                message=(
                    f"{len(failing_periods)} period(s) with AUC < {self.min_auc}: "
                    + ", ".join(f"{p['period']}={p['auc']}" for p in failing_periods)
                ),
                severity=Severity.CRITICAL,
                details={"failing_periods": failing_periods},
                recommendation=(
                    "Review feature selection; consider relaxing elimination thresholds "
                    "or adding more features."
                ),
            )
        min_auc = perf["AUC"].min()
        return CheckResult(
            check_name="Discrimination (AUC)",
            status=Status.PASS,
            message=f"All periods have AUC >= {self.min_auc} (min observed: {min_auc:.4f}).",
            details={"min_auc": round(float(min_auc), 4)},
            severity=Severity.CRITICAL,
        )

    def _check_gini(self, perf: pd.DataFrame) -> CheckResult:
        """Gini >= 2*min_auc - 1 for every period (derived from AUC threshold)."""
        gini_col = "Gini" if "Gini" in perf.columns else None
        if gini_col is None:
            # Compute from AUC if available
            if "AUC" in perf.columns:
                gini_vals = 2 * perf["AUC"] - 1
            else:
                return CheckResult(
                    check_name="Discrimination (Gini)",
                    status=Status.WARNING,
                    message="Neither Gini nor AUC column found.",
                    severity=Severity.WARNING,
                )
        else:
            gini_vals = perf[gini_col]

        min_gini_threshold = 2 * self.min_auc - 1
        min_gini = gini_vals.min()
        if min_gini < min_gini_threshold:
            return CheckResult(
                check_name="Discrimination (Gini)",
                status=Status.WARNING,
                message=f"Minimum Gini {min_gini:.4f} below implied threshold {min_gini_threshold:.4f}.",
                severity=Severity.WARNING,
                details={"min_gini": round(float(min_gini), 4)},
                recommendation="Review model discrimination across periods.",
            )
        return CheckResult(
            check_name="Discrimination (Gini)",
            status=Status.PASS,
            message=f"All Gini values >= {min_gini_threshold:.4f} (min: {min_gini:.4f}).",
            details={"min_gini": round(float(min_gini), 4)},
            severity=Severity.WARNING,
        )

    # ------------------------------------------------------------------
    # Overfitting
    # ------------------------------------------------------------------

    def _check_overfit(self, perf: pd.DataFrame) -> CheckResult:
        """Train AUC - Test AUC > max_overfit_gap triggers WARNING."""
        if "AUC" not in perf.columns or "Period" not in perf.columns:
            return CheckResult(
                check_name="Overfitting",
                status=Status.WARNING,
                message="Cannot check overfitting: missing AUC or Period columns.",
                severity=Severity.WARNING,
            )

        train_row = perf[perf["Period"].str.lower().str.contains("train")]
        test_row = perf[perf["Period"].str.lower().str.contains("test")]

        if train_row.empty or test_row.empty:
            return CheckResult(
                check_name="Overfitting",
                status=Status.WARNING,
                message="Cannot identify Train/Test rows in performance data.",
                severity=Severity.WARNING,
            )

        train_auc = float(train_row["AUC"].iloc[0])
        test_auc = float(test_row["AUC"].iloc[0])
        gap = train_auc - test_auc

        if gap > self.max_overfit_gap:
            return CheckResult(
                check_name="Overfitting",
                status=Status.WARNING,
                message=(
                    f"AUC gap Train-Test = {gap:.4f} exceeds {self.max_overfit_gap:.4f}. "
                    f"(Train={train_auc:.4f}, Test={test_auc:.4f})"
                ),
                severity=Severity.WARNING,
                details={"train_auc": train_auc, "test_auc": test_auc, "gap": round(gap, 4)},
                recommendation=(
                    "Consider reducing model complexity (max_depth, n_estimators) "
                    "or increasing regularization."
                ),
            )
        return CheckResult(
            check_name="Overfitting",
            status=Status.PASS,
            message=f"AUC gap Train-Test = {gap:.4f} within threshold {self.max_overfit_gap:.4f}.",
            details={"train_auc": train_auc, "test_auc": test_auc, "gap": round(gap, 4)},
            severity=Severity.WARNING,
        )

    # ------------------------------------------------------------------
    # OOT stability
    # ------------------------------------------------------------------

    def _check_oot_stability(self, perf: pd.DataFrame) -> CheckResult:
        """OOT AUC should not degrade more than max_oot_degradation from test."""
        if "AUC" not in perf.columns or "Period" not in perf.columns:
            return CheckResult(
                check_name="OOT stability",
                status=Status.WARNING,
                message="Cannot check OOT stability: missing columns.",
                severity=Severity.WARNING,
            )

        test_row = perf[perf["Period"].str.lower().str.contains("test")]
        oot_rows = perf[perf["Period"].str.lower().str.startswith("oot")]

        if test_row.empty or oot_rows.empty:
            return CheckResult(
                check_name="OOT stability",
                status=Status.WARNING,
                message="Cannot identify Test/OOT rows for stability check.",
                severity=Severity.WARNING,
            )

        test_auc = float(test_row["AUC"].iloc[0])
        degraded = []
        for _, row in oot_rows.iterrows():
            oot_auc = float(row["AUC"])
            drop = test_auc - oot_auc
            if drop > self.max_oot_degradation:
                degraded.append({
                    "period": str(row["Period"]),
                    "oot_auc": round(oot_auc, 4),
                    "drop": round(drop, 4),
                })

        if degraded:
            return CheckResult(
                check_name="OOT stability",
                status=Status.WARNING,
                message=(
                    f"{len(degraded)} OOT period(s) degrade > {self.max_oot_degradation:.4f} from test: "
                    + ", ".join(f"{d['period']}(drop={d['drop']})" for d in degraded)
                ),
                severity=Severity.WARNING,
                details={"degraded_periods": degraded, "test_auc": test_auc},
                recommendation="Investigate time drift. Consider retraining with more recent data.",
            )
        oot_min = oot_rows["AUC"].min()
        return CheckResult(
            check_name="OOT stability",
            status=Status.PASS,
            message=(
                f"All OOT periods within {self.max_oot_degradation:.4f} of test AUC "
                f"(min OOT AUC: {oot_min:.4f})."
            ),
            details={"test_auc": test_auc, "min_oot_auc": round(float(oot_min), 4)},
            severity=Severity.WARNING,
        )

    # ------------------------------------------------------------------
    # Score PSI
    # ------------------------------------------------------------------

    def _check_score_psi(self, psi_df: pd.DataFrame) -> CheckResult:
        """Score PSI between train and OOT < max_score_psi."""
        psi_col = "PSI" if "PSI" in psi_df.columns else "psi"
        if psi_col not in psi_df.columns:
            return CheckResult(
                check_name="Score PSI",
                status=Status.WARNING,
                message="PSI column not found in score_psi_df.",
                severity=Severity.WARNING,
            )

        max_psi = float(psi_df[psi_col].max())
        if max_psi > self.max_score_psi:
            return CheckResult(
                check_name="Score PSI",
                status=Status.WARNING,
                message=f"Maximum score PSI {max_psi:.4f} exceeds threshold {self.max_score_psi:.4f}.",
                severity=Severity.WARNING,
                details={"max_psi": round(max_psi, 4)},
                recommendation="Score distribution has shifted. Investigate population changes.",
            )
        return CheckResult(
            check_name="Score PSI",
            status=Status.PASS,
            message=f"Score PSI {max_psi:.4f} within threshold {self.max_score_psi:.4f}.",
            details={"max_psi": round(max_psi, 4)},
            severity=Severity.WARNING,
        )

    # ------------------------------------------------------------------
    # Concentration
    # ------------------------------------------------------------------

    def _check_concentration(self, importance_df: pd.DataFrame) -> CheckResult:
        """No single feature contributes > max_feature_concentration of total importance."""
        imp_col = None
        for candidate in ("Importance", "importance", "Gain", "gain"):
            if candidate in importance_df.columns:
                imp_col = candidate
                break
        if imp_col is None:
            return CheckResult(
                check_name="Feature concentration",
                status=Status.WARNING,
                message="No importance column found.",
                severity=Severity.WARNING,
            )

        total = importance_df[imp_col].sum()
        if total == 0:
            return CheckResult(
                check_name="Feature concentration",
                status=Status.WARNING,
                message="Total importance is zero.",
                severity=Severity.WARNING,
            )

        max_share = float(importance_df[imp_col].max() / total)
        if max_share > self.max_feature_concentration:
            top_feat = importance_df.loc[importance_df[imp_col].idxmax()]
            feat_name = top_feat.get("Feature", top_feat.get("feature", "?"))
            return CheckResult(
                check_name="Feature concentration",
                status=Status.WARNING,
                message=(
                    f"Feature '{feat_name}' contributes {max_share:.2%} of total importance "
                    f"(threshold: {self.max_feature_concentration:.0%})."
                ),
                severity=Severity.WARNING,
                details={"feature": str(feat_name), "share": round(max_share, 4)},
                recommendation="Consider removing dominant feature and retraining for a more robust model.",
            )
        return CheckResult(
            check_name="Feature concentration",
            status=Status.PASS,
            message=f"Max feature share {max_share:.2%} within {self.max_feature_concentration:.0%} threshold.",
            details={"max_share": round(max_share, 4)},
            severity=Severity.WARNING,
        )

    # ------------------------------------------------------------------
    # Monotonicity
    # ------------------------------------------------------------------

    def _check_monotonicity(self, woe_bins: Dict[str, pd.DataFrame]) -> CheckResult:
        """Check if WoE trend is monotonic for each selected feature."""
        non_monotonic: List[str] = []
        for feature, bins_df in woe_bins.items():
            woe_col = None
            for candidate in ("WoE", "woe", "WOE"):
                if candidate in bins_df.columns:
                    woe_col = candidate
                    break
            if woe_col is None:
                continue
            woe_values = bins_df[woe_col].dropna().values
            if len(woe_values) < 2:
                continue
            diffs = np.diff(woe_values)
            is_mono = np.all(diffs >= 0) or np.all(diffs <= 0)
            if not is_mono:
                non_monotonic.append(feature)

        if non_monotonic:
            return CheckResult(
                check_name="WoE monotonicity",
                status=Status.WARNING,
                message=f"{len(non_monotonic)} feature(s) have non-monotonic WoE: {non_monotonic[:10]}.",
                severity=Severity.INFO,
                details={"non_monotonic_features": non_monotonic},
                recommendation=(
                    "Non-monotonic WoE is not always problematic but may indicate "
                    "noisy bins or complex relationships. Review binning."
                ),
            )
        return CheckResult(
            check_name="WoE monotonicity",
            status=Status.PASS,
            message="All selected features have monotonic WoE trends.",
            severity=Severity.INFO,
        )

    # ------------------------------------------------------------------
    # OOT sample size
    # ------------------------------------------------------------------

    def _check_oot_sample_size(self, perf: pd.DataFrame) -> CheckResult:
        """Each OOT quarter has >= min_oot_samples bads."""
        if "Period" not in perf.columns:
            return CheckResult(
                check_name="OOT sample size",
                status=Status.WARNING,
                message="Period column missing.",
                severity=Severity.WARNING,
            )

        oot_rows = perf[perf["Period"].str.lower().str.startswith("oot")]
        if oot_rows.empty:
            return CheckResult(
                check_name="OOT sample size",
                status=Status.WARNING,
                message="No OOT periods found.",
                severity=Severity.WARNING,
            )

        bad_col = None
        for candidate in ("N_Bad", "n_bad", "Bad", "bad", "bads"):
            if candidate in oot_rows.columns:
                bad_col = candidate
                break

        if bad_col is None:
            return CheckResult(
                check_name="OOT sample size",
                status=Status.WARNING,
                message="No bad-count column found in performance data.",
                severity=Severity.WARNING,
                recommendation="Add N_Bad column to performance output.",
            )

        small_periods = []
        for _, row in oot_rows.iterrows():
            n_bad = int(row[bad_col])
            if n_bad < self.min_oot_samples:
                small_periods.append({
                    "period": str(row["Period"]),
                    "n_bad": n_bad,
                })

        if small_periods:
            return CheckResult(
                check_name="OOT sample size",
                status=Status.WARNING,
                message=(
                    f"{len(small_periods)} OOT period(s) with < {self.min_oot_samples} bads: "
                    + ", ".join(f"{p['period']}({p['n_bad']})" for p in small_periods)
                ),
                severity=Severity.WARNING,
                details={"small_periods": small_periods},
                recommendation="Results for these periods may not be statistically reliable.",
            )
        min_bad = int(oot_rows[bad_col].min())
        return CheckResult(
            check_name="OOT sample size",
            status=Status.PASS,
            message=f"All OOT periods have >= {self.min_oot_samples} bads (min: {min_bad}).",
            details={"min_bad": min_bad},
            severity=Severity.INFO,
        )
