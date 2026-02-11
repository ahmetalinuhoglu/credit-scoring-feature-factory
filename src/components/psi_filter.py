"""
PSI (Population Stability Index) Filter Component

Eliminates features whose distribution is unstable within the training data
as measured by PSI across configurable comparison strategies.
"""

from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging
import time

import numpy as np
import pandas as pd

from src.pipeline.base import BaseComponent, StepResult
from src.config.schema import PSIConfig, PSICheckConfig

logger = logging.getLogger(__name__)

STEP_NAME = "04_psi"


# ──────────────────────────────────────────────────────────────
# PSI Check Strategies
# ──────────────────────────────────────────────────────────────

class PSICheck(ABC):
    """Base class for PSI comparison strategies.

    Each check defines how to split training data into (baseline, comparison)
    pairs. PSI is computed for each pair per feature.
    """

    @abstractmethod
    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Return list of (label, baseline_mask, comparison_mask).

        Args:
            dates: Series of datetime values aligned with the training data.

        Returns:
            List of tuples, each containing a label and two boolean mask arrays.
        """
        pass


class QuarterlyPSICheck(PSICheck):
    """Compare each quarter's distribution vs the overall training distribution."""

    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        quarters = dates.dt.to_period("Q")
        all_mask = np.ones(len(dates), dtype=bool)
        splits = []
        for q in sorted(quarters.unique()):
            q_mask = (quarters == q).values
            if q_mask.sum() >= 10:
                splits.append((f"Q_{q}_vs_All", all_mask, q_mask))
        return splits


class YearlyPSICheck(PSICheck):
    """Compare each year's distribution vs the overall training distribution."""

    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        years = dates.dt.year
        all_mask = np.ones(len(dates), dtype=bool)
        splits = []
        for year in sorted(years.unique()):
            y_mask = (years == year).values
            if y_mask.sum() >= 10:
                splits.append((f"Y_{year}_vs_All", all_mask, y_mask))
        return splits


class ConsecutiveQuarterPSICheck(PSICheck):
    """Compare consecutive quarters: Q1 vs Q2, Q2 vs Q3, etc."""

    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        quarters = dates.dt.to_period("Q")
        sorted_qs = sorted(quarters.unique())
        splits = []
        for prev_q, next_q in zip(sorted_qs[:-1], sorted_qs[1:]):
            prev_mask = (quarters == prev_q).values
            next_mask = (quarters == next_q).values
            if prev_mask.sum() >= 10 and next_mask.sum() >= 10:
                splits.append((f"{prev_q}_vs_{next_q}", prev_mask, next_mask))
        return splits


class HalfSplitPSICheck(PSICheck):
    """Compare first half vs second half of training data by date."""

    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        median_date = dates.median()
        first_half = (dates <= median_date).values
        second_half = (dates > median_date).values
        if first_half.sum() < 10 or second_half.sum() < 10:
            return []
        return [("first_half_vs_second_half", first_half, second_half)]


class DateSplitPSICheck(PSICheck):
    """Compare distribution before vs after a specific date.

    Args:
        split_date: Date string to split on.
        label: Optional descriptive label for this check.
    """

    def __init__(self, split_date: str, label: Optional[str] = None):
        self.split_date = pd.Timestamp(split_date)
        self.label = label or f"before_vs_after_{split_date}"

    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        before = (dates <= self.split_date).values
        after = (dates > self.split_date).values
        if before.sum() < 10 or after.sum() < 10:
            logger.warning(
                f"{STEP_NAME} | DateSplit '{self.label}': insufficient data "
                f"(before={before.sum()}, after={after.sum()}), skipping"
            )
            return []
        return [(self.label, before, after)]


# ──────────────────────────────────────────────────────────────
# Factory to build PSICheck instances from config
# ──────────────────────────────────────────────────────────────

_CHECK_REGISTRY = {
    "quarterly": QuarterlyPSICheck,
    "yearly": YearlyPSICheck,
    "consecutive": ConsecutiveQuarterPSICheck,
    "halfsplit": HalfSplitPSICheck,
}


def build_psi_checks(check_configs: List[PSICheckConfig]) -> List[PSICheck]:
    """Build PSICheck instances from a list of PSICheckConfig.

    Args:
        check_configs: List of PSICheckConfig from the pipeline config.

    Returns:
        List of PSICheck instances.

    Raises:
        ValueError: If an unknown check type is encountered.
    """
    checks: List[PSICheck] = []
    for cfg in check_configs:
        if cfg.type == "date_split":
            if cfg.date is None:
                raise ValueError("date_split PSI check requires a 'date' field")
            checks.append(DateSplitPSICheck(split_date=cfg.date, label=cfg.label))
        elif cfg.type in _CHECK_REGISTRY:
            checks.append(_CHECK_REGISTRY[cfg.type]())
        else:
            raise ValueError(
                f"Unknown PSI check type: '{cfg.type}'. "
                f"Valid types: {list(_CHECK_REGISTRY.keys()) + ['date_split']}"
            )
    return checks


# ──────────────────────────────────────────────────────────────
# PSI Filter Component
# ──────────────────────────────────────────────────────────────

class PSIFilter(BaseComponent):
    """Remove features with unstable distributions within training data.

    Uses pluggable PSI checks to define how training data is split into
    baseline vs comparison groups. No OOT data is used -- keeping OOT
    purely for final evaluation.

    Args:
        config: PSIConfig with threshold, n_bins, and checks list.
    """

    step_name = STEP_NAME
    step_order = 4

    def __init__(self, config: PSIConfig):
        self.threshold = config.threshold
        self.n_bins = config.n_bins
        self.checks = build_psi_checks(config.checks)
        self.kept_features_: List[str] = []
        self.eliminated_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> StepResult:
        """Calculate PSI for each feature across all configured checks.

        Args:
            X: Training feature DataFrame.
            y: Training target Series (unused).
            **kwargs:
                train_dates: pd.Series of datetime values aligned with X rows.
                    Required for PSI computation.

        Returns:
            StepResult with per-feature PSI for each check, max/mean PSI, and status.
        """
        t0 = time.time()
        features = list(X.columns)
        train_dates: Optional[pd.Series] = kwargs.get("train_dates")

        if train_dates is None:
            logger.warning(
                f"{STEP_NAME} | No train_dates provided, skipping PSI filtering"
            )
            self.kept_features_ = list(features)
            self.eliminated_features_ = []
            results_df = pd.DataFrame({
                "Feature": features,
                "Max_PSI": 0.0,
                "Status": "Kept",
                "Reason": "No dates provided",
            })
            return StepResult(
                step_name=self.step_name,
                input_features=features,
                output_features=list(features),
                eliminated_features=[],
                results_df=results_df,
                duration_seconds=round(time.time() - t0, 1),
            )

        # Collect all splits from all checks
        all_splits: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for check in self.checks:
            splits = check.get_splits(train_dates)
            all_splits.extend(splits)
            logger.info(
                f"{STEP_NAME} | Check {check.__class__.__name__}: "
                f"{len(splits)} comparison(s)"
            )

        if not all_splits:
            logger.warning(
                f"{STEP_NAME} | No valid splits found, skipping PSI filtering"
            )
            self.kept_features_ = list(features)
            self.eliminated_features_ = []
            results_df = pd.DataFrame({
                "Feature": features,
                "Max_PSI": 0.0,
                "Status": "Kept",
                "Reason": "No valid splits",
            })
            return StepResult(
                step_name=self.step_name,
                input_features=features,
                output_features=list(features),
                eliminated_features=[],
                results_df=results_df,
                duration_seconds=round(time.time() - t0, 1),
            )

        split_labels = [label for label, _, _ in all_splits]
        logger.info(
            f"{STEP_NAME} | Checking {len(features)} features across "
            f"{len(all_splits)} comparisons"
        )

        rows = []
        kept, eliminated = [], []

        for feat in features:
            values = X[feat].values

            psi_results: Dict[str, Optional[float]] = {}
            for label, base_mask, comp_mask in all_splits:
                base_vals = values[base_mask]
                comp_vals = values[comp_mask]
                # Drop NaN
                base_vals = base_vals[~pd.isna(base_vals)]
                comp_vals = comp_vals[~pd.isna(comp_vals)]
                psi_results[label] = self._calculate_psi(base_vals, comp_vals)

            valid_psis = [v for v in psi_results.values() if v is not None]
            max_psi = max(valid_psis) if valid_psis else 0.0
            mean_psi = float(np.mean(valid_psis)) if valid_psis else 0.0

            if max_psi >= self.threshold:
                eliminated.append(feat)
                status = "Eliminated"
                reason = f"Max PSI {max_psi:.4f} >= {self.threshold}"
            else:
                kept.append(feat)
                status = "Kept"
                reason = ""

            row: Dict[str, Any] = {"Feature": feat}
            for label in split_labels:
                v = psi_results.get(label)
                row[f"PSI_{label}"] = round(v, 4) if v is not None else None
            row["Max_PSI"] = round(max_psi, 4)
            row["Mean_PSI"] = round(mean_psi, 4)
            row["Status"] = status
            row["Reason"] = reason
            rows.append(row)

        self.kept_features_ = kept
        self.eliminated_features_ = eliminated

        results_df = pd.DataFrame(rows).sort_values("Max_PSI", ascending=False)
        duration = time.time() - t0

        logger.info(
            f"{STEP_NAME} | Eliminated {len(eliminated)} features "
            f"({len(kept)} remaining) in {duration:.1f}s"
        )

        return StepResult(
            step_name=self.step_name,
            input_features=features,
            output_features=kept,
            eliminated_features=eliminated,
            results_df=results_df,
            metadata={
                "threshold": self.threshold,
                "n_bins": self.n_bins,
                "n_checks": len(self.checks),
                "n_comparisons": len(all_splits),
            },
            duration_seconds=round(duration, 1),
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop unstable features from the DataFrame.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with only kept features (columns present in X).
        """
        cols = [c for c in self.kept_features_ if c in X.columns]
        return X[cols]

    def _calculate_psi(
        self, expected: np.ndarray, actual: np.ndarray
    ) -> Optional[float]:
        """Calculate PSI between two distributions.

        Args:
            expected: Baseline distribution values.
            actual: Comparison distribution values.

        Returns:
            PSI value, or None if calculation is not possible.
        """
        try:
            if len(expected) < 10 or len(actual) < 10:
                return None

            # Create bins from expected distribution
            try:
                _, bins = pd.qcut(
                    expected, q=self.n_bins, retbins=True, duplicates="drop"
                )
            except ValueError:
                n = min(5, len(np.unique(expected)))
                if n < 2:
                    return None
                _, bins = pd.qcut(expected, q=n, retbins=True, duplicates="drop")

            bins[0] = -np.inf
            bins[-1] = np.inf

            expected_binned = pd.cut(expected, bins=bins)
            actual_binned = pd.cut(actual, bins=bins)

            expected_pct = (
                pd.Series(expected_binned).value_counts(normalize=True).sort_index()
            )
            actual_pct = (
                pd.Series(actual_binned).value_counts(normalize=True).sort_index()
            )

            all_bins = expected_pct.index.union(actual_pct.index)
            expected_pct = expected_pct.reindex(all_bins, fill_value=0.0001)
            actual_pct = actual_pct.reindex(all_bins, fill_value=0.0001)

            # Clip to avoid log(0) and division by zero
            epsilon = 1e-4
            expected_pct = expected_pct.clip(lower=epsilon)
            actual_pct = actual_pct.clip(lower=epsilon)

            psi = float(
                ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
            )
            return psi if np.isfinite(psi) else None
        except Exception:
            return None
