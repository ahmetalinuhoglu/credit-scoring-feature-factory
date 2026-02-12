"""
Feature Eliminators

Five-step variable elimination for credit scoring model development:
1. Constant features
2. High missing rate
3. Low IV (Information Value)
4. Unstable PSI (Population Stability Index)
5. High correlation (greedy, IV-ordered)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, roc_curve


logger = logging.getLogger(__name__)


@dataclass
class EliminationResult:
    """Result of a feature elimination step."""
    step_name: str
    kept_features: List[str]
    eliminated_features: List[str]
    details_df: pd.DataFrame

    @property
    def n_kept(self) -> int:
        return len(self.kept_features)

    @property
    def n_eliminated(self) -> int:
        return len(self.eliminated_features)


class BaseEliminator(ABC):
    """Base class for feature eliminators."""

    step_name: str = ""

    @abstractmethod
    def eliminate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        features: List[str],
        **kwargs,
    ) -> EliminationResult:
        pass


class ConstantEliminator(BaseEliminator):
    """Remove features with zero or near-zero variance."""

    step_name = "01_Constant"

    def __init__(self, min_unique: int = 2, min_variance: float = 0.0):
        self.min_unique = min_unique
        self.min_variance = min_variance

    def eliminate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        features: List[str],
        **kwargs,
    ) -> EliminationResult:
        rows = []
        kept, eliminated = [], []

        for feat in features:
            series = X_train[feat]
            n_unique = series.nunique()
            variance = series.var() if pd.api.types.is_numeric_dtype(series) else 0.0

            is_constant = n_unique < self.min_unique
            is_near_zero = (
                self.min_variance > 0
                and pd.api.types.is_numeric_dtype(series)
                and variance <= self.min_variance
            )
            eliminate = is_constant or is_near_zero

            if eliminate:
                eliminated.append(feat)
                status = "Eliminated"
            else:
                kept.append(feat)
                status = "Kept"

            rows.append({
                'Feature': feat,
                'Unique_Count': n_unique,
                'Variance': round(variance, 6) if variance is not None else None,
                'Status': status,
            })

        details_df = pd.DataFrame(rows).sort_values('Unique_Count')
        logger.info(
            f"CONSTANT | Eliminated {len(eliminated)} features "
            f"({len(kept)} remaining)"
        )
        return EliminationResult(self.step_name, kept, eliminated, details_df)


class MissingEliminator(BaseEliminator):
    """Remove features with high missing rate."""

    step_name = "02_Missing"

    def __init__(self, max_missing_rate: float = 0.70):
        self.max_missing_rate = max_missing_rate

    def eliminate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        features: List[str],
        **kwargs,
    ) -> EliminationResult:
        n_rows = len(X_train)
        rows = []
        kept, eliminated = [], []

        for feat in features:
            missing_count = X_train[feat].isna().sum()
            missing_rate = missing_count / n_rows if n_rows > 0 else 0

            if missing_rate > self.max_missing_rate:
                eliminated.append(feat)
                status = "Eliminated"
            else:
                kept.append(feat)
                status = "Kept"

            rows.append({
                'Feature': feat,
                'Missing_Count': int(missing_count),
                'Missing_Rate': round(missing_rate, 4),
                'Total_Rows': n_rows,
                'Status': status,
            })

        details_df = pd.DataFrame(rows).sort_values(
            'Missing_Rate', ascending=False
        )
        logger.info(
            f"MISSING | Eliminated {len(eliminated)} features "
            f"({len(kept)} remaining)"
        )
        return EliminationResult(self.step_name, kept, eliminated, details_df)


def _calculate_iv_static(
    feat_values: np.ndarray, target_values: np.ndarray, n_bins: int
) -> Optional[float]:
    """Calculate IV for a single feature (module-level, picklable for joblib)."""
    try:
        data = pd.DataFrame({'feature': feat_values, 'target': target_values}).dropna()
        if len(data) < 50:
            return None

        total_goods = (data['target'] == 0).sum()
        total_bads = (data['target'] == 1).sum()
        if total_goods == 0 or total_bads == 0:
            return None

        try:
            data['bin'] = pd.qcut(
                data['feature'], q=n_bins,
                labels=False, duplicates='drop',
            )
        except (ValueError, TypeError):
            return None

        iv = 0.0
        epsilon = 1e-6
        for _, group in data.groupby('bin'):
            goods = (group['target'] == 0).sum()
            bads = (group['target'] == 1).sum()
            pct_goods = max(goods / total_goods, epsilon)
            pct_bads = max(bads / total_bads, epsilon)
            woe = np.log(pct_goods / pct_bads)
            iv += (pct_goods - pct_bads) * woe

        return float(iv)
    except Exception:
        return None


def _calculate_univariate_metrics_static(
    feat_values: np.ndarray, target_values: np.ndarray
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate univariate AUC, Gini, KS for a single feature (module-level, picklable)."""
    try:
        data = pd.DataFrame({'feature': feat_values, 'target': target_values}).dropna()
        if len(data) < 50 or data['target'].nunique() < 2:
            return None, None, None

        auc = roc_auc_score(data['target'], data['feature'])
        gini = 2 * auc - 1
        fpr, tpr, _ = roc_curve(data['target'], data['feature'])
        ks = float(max(tpr - fpr))
        return float(auc), float(gini), ks
    except Exception:
        return None, None, None


def _iv_category_label(iv: Optional[float]) -> str:
    """Return IV strength category label."""
    if iv is None:
        return "unknown"
    if iv < 0.02:
        return "useless"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspicious"


def _compute_iv_feature_metrics(
    feat_name: str,
    feat_values: np.ndarray,
    target_values: np.ndarray,
    n_bins: int,
    min_iv: float,
    max_iv: float,
) -> Dict:
    """Compute IV + univariate metrics for one feature (picklable, for joblib)."""
    iv = _calculate_iv_static(feat_values, target_values, n_bins)
    iv_category = _iv_category_label(iv)
    uni_auc, uni_gini, uni_ks = _calculate_univariate_metrics_static(
        feat_values, target_values
    )

    reason = ""
    if iv is None:
        status = "Eliminated"
        reason = "Could not calculate IV"
    elif iv < min_iv:
        status = "Eliminated"
        reason = f"IV {iv:.4f} < {min_iv} (useless)"
    elif iv > max_iv:
        status = "Eliminated"
        reason = f"IV {iv:.4f} > {max_iv} (suspicious)"
    else:
        status = "Kept"

    return {
        'Feature': feat_name,
        'IV_Score': round(iv, 4) if iv is not None else None,
        'IV_Category': iv_category,
        'Univariate_AUC': round(uni_auc, 4) if uni_auc else None,
        'Univariate_Gini': round(uni_gini, 4) if uni_gini else None,
        'Univariate_KS': round(uni_ks, 4) if uni_ks else None,
        'Status': status,
        'Reason': reason,
    }


class IVEliminator(BaseEliminator):
    """Remove features with low or suspicious Information Value."""

    step_name = "03_IV_Analysis"

    def __init__(
        self,
        min_iv: float = 0.02,
        max_iv: float = 0.50,
        n_bins: int = 10,
        n_jobs: int = 1,
    ):
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def eliminate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        features: List[str],
        **kwargs,
    ) -> EliminationResult:
        target_values = y_train.values

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_iv_feature_metrics)(
                feat, X_train[feat].values, target_values,
                self.n_bins, self.min_iv, self.max_iv,
            )
            for feat in features
        )

        rows = []
        kept, eliminated = [], []
        for row in results:
            rows.append(row)
            if row['Status'] == 'Kept':
                kept.append(row['Feature'])
            else:
                eliminated.append(row['Feature'])

        details_df = pd.DataFrame(rows).sort_values(
            'IV_Score', ascending=False, na_position='last'
        )
        logger.info(
            f"IV | Eliminated {len(eliminated)} features "
            f"({len(kept)} remaining)"
        )
        return EliminationResult(self.step_name, kept, eliminated, details_df)

    def _calculate_iv(
        self, series: pd.Series, target: pd.Series
    ) -> Optional[float]:
        """Calculate Information Value for a single feature."""
        try:
            data = pd.DataFrame({'feature': series, 'target': target}).dropna()
            if len(data) < 50:
                return None

            total_goods = (data['target'] == 0).sum()
            total_bads = (data['target'] == 1).sum()
            if total_goods == 0 or total_bads == 0:
                return None

            try:
                data['bin'] = pd.qcut(
                    data['feature'], q=self.n_bins,
                    labels=False, duplicates='drop',
                )
            except (ValueError, TypeError):
                return None

            iv = 0.0
            epsilon = 1e-6
            for _, group in data.groupby('bin'):
                goods = (group['target'] == 0).sum()
                bads = (group['target'] == 1).sum()
                pct_goods = max(goods / total_goods, epsilon)
                pct_bads = max(bads / total_bads, epsilon)
                woe = np.log(pct_goods / pct_bads)
                iv += (pct_goods - pct_bads) * woe

            return float(iv)
        except Exception:
            return None

    @staticmethod
    def _iv_category(iv: Optional[float]) -> str:
        if iv is None:
            return "unknown"
        if iv < 0.02:
            return "useless"
        if iv < 0.10:
            return "weak"
        if iv < 0.30:
            return "medium"
        if iv < 0.50:
            return "strong"
        return "suspicious"

    @staticmethod
    def _calculate_univariate_metrics(
        series: pd.Series, target: pd.Series
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate univariate AUC, Gini, KS for a single feature."""
        try:
            data = pd.DataFrame({'feature': series, 'target': target}).dropna()
            if len(data) < 50 or data['target'].nunique() < 2:
                return None, None, None

            auc = roc_auc_score(data['target'], data['feature'])
            gini = 2 * auc - 1
            fpr, tpr, _ = roc_curve(data['target'], data['feature'])
            ks = float(max(tpr - fpr))
            return float(auc), float(gini), ks
        except Exception:
            return None, None, None


# ──────────────────────────────────────────────────────────────
# PSI Check Strategies
# ──────────────────────────────────────────────────────────────

class PSICheck(ABC):
    """
    Base class for PSI comparison strategies.

    Each check defines how to split training data into (baseline, comparison)
    pairs. PSI is computed for each pair per feature.

    Usage:
        checks = [
            QuarterlyPSICheck(),
            DateSplitPSICheck('2024-04-01', label='Pre/Post Apr 2024'),
            YearlyPSICheck(),
        ]
        psi_elim = PSIEliminator(checks=checks)
    """

    @abstractmethod
    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Return list of (label, baseline_mask, comparison_mask).

        Masks are boolean arrays aligned with the dates Series.
        """
        pass


class QuarterlyPSICheck(PSICheck):
    """Compare each quarter's distribution vs the overall training distribution."""

    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        quarters = dates.dt.to_period('Q')
        all_mask = np.ones(len(dates), dtype=bool)
        splits = []
        for q in sorted(quarters.unique()):
            q_mask = (quarters == q).values
            if q_mask.sum() >= 10:
                splits.append((f"Q_{q}_vs_All", all_mask, q_mask))
        return splits


class DateSplitPSICheck(PSICheck):
    """Compare distribution before vs after a specific date."""

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
                f"PSI | DateSplit '{self.label}': insufficient data on one side "
                f"(before={before.sum()}, after={after.sum()}), skipping"
            )
            return []
        return [(self.label, before, after)]


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
    """Compare consecutive quarters: Q1 vs Q2, Q2 vs Q3, Q3 vs Q4, etc."""

    def get_splits(
        self, dates: pd.Series
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        quarters = dates.dt.to_period('Q')
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


# ──────────────────────────────────────────────────────────────
# PSI helpers (module-level, picklable for joblib)
# ──────────────────────────────────────────────────────────────

def _calculate_psi_static(
    expected: np.ndarray, actual: np.ndarray, n_bins: int
) -> Optional[float]:
    """Calculate PSI between two distributions (module-level, picklable)."""
    try:
        if len(expected) < 10 or len(actual) < 10:
            return None

        try:
            _, bins = pd.qcut(
                expected, q=n_bins, retbins=True, duplicates='drop'
            )
        except ValueError:
            n = min(5, len(np.unique(expected)))
            if n < 2:
                return None
            _, bins = pd.qcut(expected, q=n, retbins=True, duplicates='drop')

        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_bins = pd.cut(expected, bins=bins)
        actual_bins = pd.cut(actual, bins=bins)

        expected_pct = (
            pd.Series(expected_bins).value_counts(normalize=True).sort_index()
        )
        actual_pct = (
            pd.Series(actual_bins).value_counts(normalize=True).sort_index()
        )

        all_bins_idx = expected_pct.index.union(actual_pct.index)
        expected_pct = expected_pct.reindex(all_bins_idx, fill_value=0.0001)
        actual_pct = actual_pct.reindex(all_bins_idx, fill_value=0.0001)

        epsilon = 1e-4
        expected_pct = expected_pct.clip(lower=epsilon)
        actual_pct = actual_pct.clip(lower=epsilon)

        psi = float(
            ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
        )
        return psi if np.isfinite(psi) else None
    except Exception:
        return None


def _compute_psi_for_feature(
    feat_name: str,
    feat_values: np.ndarray,
    all_splits: List[Tuple[str, np.ndarray, np.ndarray]],
    n_bins: int,
    critical_threshold: float,
) -> Dict:
    """Compute PSI across all splits for one feature (picklable, for joblib)."""
    psi_results = {}
    for label, base_mask, comp_mask in all_splits:
        base_vals = feat_values[base_mask]
        comp_vals = feat_values[comp_mask]
        base_vals = base_vals[~pd.isna(base_vals)]
        comp_vals = comp_vals[~pd.isna(comp_vals)]
        psi_results[label] = _calculate_psi_static(base_vals, comp_vals, n_bins)

    valid_psis = [v for v in psi_results.values() if v is not None]
    max_psi = max(valid_psis) if valid_psis else 0.0
    mean_psi = float(np.mean(valid_psis)) if valid_psis else 0.0

    if max_psi >= critical_threshold:
        status = "Eliminated"
        reason = f"Max PSI {max_psi:.4f} >= {critical_threshold}"
    else:
        status = "Kept"
        reason = ""

    row = {'Feature': feat_name}
    for label, _, _ in all_splits:
        v = psi_results.get(label)
        row[f'PSI_{label}'] = round(v, 4) if v is not None else None
    row['Max_PSI'] = round(max_psi, 4)
    row['Mean_PSI'] = round(mean_psi, 4)
    row['Status'] = status
    row['Reason'] = reason
    return row


# ──────────────────────────────────────────────────────────────
# PSI Eliminator
# ──────────────────────────────────────────────────────────────

class PSIEliminator(BaseEliminator):
    """
    Remove features with unstable distributions within training data.

    Uses pluggable PSI checks to define how training data is split into
    baseline vs comparison groups. No OOT data is used — keeping OOT
    purely for final evaluation.

    Default: QuarterlyPSICheck (each quarter vs overall training).
    """

    step_name = "04_PSI_Stability"

    def __init__(
        self,
        critical_threshold: float = 0.25,
        n_bins: int = 10,
        checks: Optional[List[PSICheck]] = None,
        n_jobs: int = 1,
    ):
        self.critical_threshold = critical_threshold
        self.n_bins = n_bins
        self.checks = checks if checks is not None else [QuarterlyPSICheck()]
        self.n_jobs = n_jobs

    def eliminate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        features: List[str],
        train_dates: Optional[pd.Series] = None,
        **kwargs,
    ) -> EliminationResult:
        if train_dates is None:
            logger.warning("PSI | No dates provided, skipping PSI elimination")
            details_df = pd.DataFrame({
                'Feature': features, 'Max_PSI': 0.0,
                'Status': 'Kept', 'Reason': 'No dates provided',
            })
            return EliminationResult(self.step_name, list(features), [], details_df)

        # Precompute all splits from all checks
        all_splits = []
        for check in self.checks:
            splits = check.get_splits(train_dates)
            all_splits.extend(splits)
            logger.info(
                f"PSI | Check {check.__class__.__name__}: "
                f"{len(splits)} comparison(s)"
            )

        if not all_splits:
            logger.warning("PSI | No valid splits found, skipping PSI elimination")
            details_df = pd.DataFrame({
                'Feature': features, 'Max_PSI': 0.0,
                'Status': 'Kept', 'Reason': 'No valid splits',
            })
            return EliminationResult(self.step_name, list(features), [], details_df)

        split_labels = [label for label, _, _ in all_splits]
        logger.info(
            f"PSI | Checking {len(features)} features across "
            f"{len(all_splits)} comparisons"
        )

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_psi_for_feature)(
                feat, X_train[feat].values, all_splits,
                self.n_bins, self.critical_threshold,
            )
            for feat in features
        )

        rows = []
        kept, eliminated = [], []
        for row in results:
            rows.append(row)
            if row['Status'] == 'Kept':
                kept.append(row['Feature'])
            else:
                eliminated.append(row['Feature'])

        details_df = pd.DataFrame(rows).sort_values(
            'Max_PSI', ascending=False
        )
        logger.info(
            f"PSI | Eliminated {len(eliminated)} features "
            f"({len(kept)} remaining)"
        )
        return EliminationResult(self.step_name, kept, eliminated, details_df)

    def _calculate_psi(
        self, expected: np.ndarray, actual: np.ndarray
    ) -> Optional[float]:
        """Calculate PSI between two distributions."""
        return _calculate_psi_static(expected, actual, self.n_bins)


class CorrelationEliminator(BaseEliminator):
    """
    Remove correlated features using greedy IV-ordered approach.

    Rule: Features are sorted by IV descending. The best feature eliminates
    all features correlated above threshold. An eliminated feature cannot
    eliminate others.
    """

    step_name = "05_Correlation"

    def __init__(
        self,
        max_correlation: float = 0.90,
        method: str = 'pearson',
    ):
        self.max_correlation = max_correlation
        self.method = method

    def eliminate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        features: List[str],
        iv_scores: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> EliminationResult:
        if iv_scores is None:
            iv_scores = {}

        # Sort features by IV descending
        sorted_features = sorted(
            features,
            key=lambda f: iv_scores.get(f, 0) or 0,
            reverse=True,
        )

        # Calculate correlation matrix
        logger.info(
            f"CORRELATION | Computing {self.method} correlation matrix "
            f"for {len(sorted_features)} features"
        )
        corr_matrix = X_train[sorted_features].corr(method=self.method)

        # Greedy elimination
        eliminated = set()
        elimination_log = []  # (eliminated_feat, by_feat, correlation)

        for feat_a in sorted_features:
            if feat_a in eliminated:
                continue
            for feat_b in sorted_features:
                if feat_b == feat_a or feat_b in eliminated:
                    continue
                corr_val = corr_matrix.loc[feat_a, feat_b]
                if abs(corr_val) > self.max_correlation:
                    eliminated.add(feat_b)
                    elimination_log.append((feat_b, feat_a, corr_val))

        kept = [f for f in sorted_features if f not in eliminated]
        eliminated_list = [f for f in sorted_features if f in eliminated]

        # Build details DataFrame
        rows = []
        for elim_feat, by_feat, corr_val in elimination_log:
            rows.append({
                'Eliminated_Feature': elim_feat,
                'Eliminated_By': by_feat,
                'Correlation': round(corr_val, 4),
                'Eliminated_IV': round(iv_scores.get(elim_feat, 0) or 0, 4),
                'Kept_IV': round(iv_scores.get(by_feat, 0) or 0, 4),
            })

        details_df = pd.DataFrame(rows)

        # Also build a correlation pairs sheet for the full picture
        corr_pairs_rows = []
        seen = set()
        for i, fa in enumerate(sorted_features):
            for fb in sorted_features[i + 1:]:
                corr_val = corr_matrix.loc[fa, fb]
                if abs(corr_val) > self.max_correlation:
                    pair_key = tuple(sorted([fa, fb]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        if fa in eliminated:
                            decision = f"{fa} eliminated by {fb}" if fb not in eliminated else "both eliminated"
                        elif fb in eliminated:
                            decision = f"{fb} eliminated by {fa}"
                        else:
                            decision = "both kept"
                        corr_pairs_rows.append({
                            'Feature_A': fa,
                            'Feature_B': fb,
                            'Correlation': round(corr_val, 4),
                            'IV_A': round(iv_scores.get(fa, 0) or 0, 4),
                            'IV_B': round(iv_scores.get(fb, 0) or 0, 4),
                            'Decision': decision,
                        })

        self.corr_pairs_df = pd.DataFrame(corr_pairs_rows)

        logger.info(
            f"CORRELATION | Eliminated {len(eliminated_list)} features "
            f"({len(kept)} remaining)"
        )
        return EliminationResult(
            self.step_name, kept, eliminated_list, details_df
        )


class VIFEliminator(BaseEliminator):
    """
    Remove multicollinear features using Variance Inflation Factor.

    Iteratively removes features with VIF above threshold until all VIF <= threshold.
    When iv_aware=True, among features exceeding the threshold, the one with lowest IV
    is dropped first (preserving more predictive features).
    """

    step_name = "07_VIF"

    def __init__(self, threshold: float = 5.0, iv_aware: bool = True):
        self.threshold = threshold
        self.iv_aware = iv_aware

    def eliminate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        features: List[str],
        iv_scores: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> EliminationResult:
        current_features = list(features)

        logger.info(
            f"VIF | Starting VIF check with {len(current_features)} features, "
            f"threshold={self.threshold}"
        )

        # Edge case: 0, 1, or 2 features — return immediately without VIF check
        if len(current_features) <= 2:
            logger.info(
                f"VIF | Only {len(current_features)} feature(s), "
                f"skipping VIF check"
            )
            details_df = pd.DataFrame({
                'Feature': current_features,
                'VIF_Initial': np.nan,
                'VIF_Final': np.nan,
                'IV_Score': [
                    round((iv_scores or {}).get(f, 0) or 0, 4)
                    for f in current_features
                ],
                'Status': 'Kept',
                'Elimination_Round': np.nan,
            })
            return EliminationResult(
                self.step_name, list(current_features), [], details_df
            )

        # Track initial VIF values and elimination rounds
        vif_initial: Dict[str, float] = {}
        elimination_log: List[Dict[str, Any]] = []
        iteration = 0

        while len(current_features) > 1:
            iteration += 1

            # Compute VIF using correlation matrix inverse
            corr = X_train[current_features].corr().values
            try:
                inv = np.linalg.inv(corr)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(corr)
            vif_values = pd.Series(np.diag(inv), index=current_features)

            # Record initial VIF on first iteration
            if iteration == 1:
                for feat in current_features:
                    vif_initial[feat] = round(float(vif_values[feat]), 4)

            max_vif = vif_values.max()
            max_vif_feat = vif_values.idxmax()

            if max_vif <= self.threshold:
                logger.info(
                    f"VIF | Iteration {iteration}: max VIF={max_vif:.4f}, "
                    f"all features below threshold"
                )
                break

            # Determine which feature to drop
            high_vif_features = vif_values[vif_values > self.threshold]

            if self.iv_aware and iv_scores:
                # Among features with VIF > threshold, drop the one with lowest IV
                iv_of_high_vif = {
                    f: (iv_scores.get(f, 0) or 0) for f in high_vif_features.index
                }
                drop_feat = min(iv_of_high_vif, key=iv_of_high_vif.get)
                drop_vif = float(vif_values[drop_feat])
                drop_iv = iv_of_high_vif[drop_feat]
                logger.info(
                    f"VIF | Iteration {iteration}: max VIF={max_vif:.4f} "
                    f"({max_vif_feat}), dropping {drop_feat} "
                    f"(VIF={drop_vif:.4f}, IV={drop_iv:.4f}, "
                    f"lowest IV among high-VIF)"
                )
            else:
                # Drop the feature with the highest VIF
                drop_feat = max_vif_feat
                drop_vif = float(max_vif)
                drop_iv = (iv_scores or {}).get(drop_feat, 0) or 0
                logger.info(
                    f"VIF | Iteration {iteration}: Dropped {drop_feat} "
                    f"(VIF={drop_vif:.4f}, IV={drop_iv:.4f})"
                )

            elimination_log.append({
                'Feature': drop_feat,
                'VIF_at_Drop': round(drop_vif, 4),
                'IV_Score': round(drop_iv, 4),
                'Elimination_Round': iteration,
            })

            # Record initial VIF for features that didn't exist on iteration 1
            # (shouldn't happen, but safe)
            if drop_feat not in vif_initial:
                vif_initial[drop_feat] = round(drop_vif, 4)

            current_features.remove(drop_feat)

        # Compute final VIF for kept features
        vif_final: Dict[str, float] = {}
        if len(current_features) > 2:
            corr = X_train[current_features].corr().values
            try:
                inv = np.linalg.inv(corr)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(corr)
            final_vif_values = pd.Series(np.diag(inv), index=current_features)
            for feat in current_features:
                vif_final[feat] = round(float(final_vif_values[feat]), 4)
        elif len(current_features) > 0:
            # 1-2 features remaining, VIF is not meaningful
            for feat in current_features:
                vif_final[feat] = np.nan

        # Build eliminated features list
        eliminated_features = [entry['Feature'] for entry in elimination_log]

        # Build details DataFrame
        rows = []
        for feat in features:
            if feat in current_features:
                rows.append({
                    'Feature': feat,
                    'VIF_Initial': vif_initial.get(feat, np.nan),
                    'VIF_Final': vif_final.get(feat, np.nan),
                    'IV_Score': round((iv_scores or {}).get(feat, 0) or 0, 4),
                    'Status': 'Kept',
                    'Elimination_Round': np.nan,
                })
            else:
                entry = next(
                    (e for e in elimination_log if e['Feature'] == feat), None
                )
                rows.append({
                    'Feature': feat,
                    'VIF_Initial': vif_initial.get(feat, np.nan),
                    'VIF_Final': np.nan,
                    'IV_Score': round((iv_scores or {}).get(feat, 0) or 0, 4),
                    'Status': 'Eliminated',
                    'Elimination_Round': (
                        entry['Elimination_Round'] if entry else np.nan
                    ),
                })

        details_df = pd.DataFrame(rows)

        logger.info(
            f"VIF | Eliminated {len(eliminated_features)} features "
            f"({len(current_features)} remaining)"
        )
        return EliminationResult(
            self.step_name, list(current_features), eliminated_features,
            details_df
        )
