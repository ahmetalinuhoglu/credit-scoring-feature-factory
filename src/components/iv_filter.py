"""
Information Value (IV) Filter Component

Eliminates features whose IV falls outside the [min_iv, max_iv] range.
Computes WoE binning boundaries, univariate AUC/Gini/KS, and handles edge
cases with Laplace smoothing and min_samples_per_bin enforcement.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.pipeline.base import BaseComponent, StepResult
from src.config.schema import IVConfig

logger = logging.getLogger(__name__)

STEP_NAME = "03_iv"


@dataclass
class WoEBin:
    """Single WoE bin definition."""

    bin_idx: int
    lower: float
    upper: float
    count: int
    goods: int
    bads: int
    woe: float
    iv_contribution: float


class IVFilter(BaseComponent):
    """Remove features with IV outside the acceptable range.

    Computes IV with WoE binning, stores bin boundaries for deployment,
    and calculates univariate AUC/Gini/KS for each feature.

    Args:
        config: IVConfig with min_iv, max_iv, n_bins, min_samples_per_bin.
    """

    step_name = STEP_NAME
    step_order = 3

    def __init__(self, config: IVConfig):
        self.min_iv = config.min_iv
        self.max_iv = config.max_iv
        self.n_bins = config.n_bins
        self.min_samples_per_bin = config.min_samples_per_bin
        self.kept_features_: List[str] = []
        self.eliminated_features_: List[str] = []
        self.iv_scores_: Dict[str, float] = {}
        self.woe_bins_: Dict[str, List[WoEBin]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> StepResult:
        """Calculate IV and univariate metrics for each feature.

        Features with IV < min_iv or IV > max_iv are eliminated. WoE binning
        boundaries are saved for downstream deployment/scorecard use.

        Args:
            X: Training feature DataFrame.
            y: Training target Series (binary 0/1).

        Returns:
            StepResult with IV scores, categories, univariate metrics,
            WoE tables, and elimination reasons.
        """
        t0 = time.time()
        features = list(X.columns)
        rows = []
        kept, eliminated = [], []
        woe_table_rows = []

        for feat in features:
            iv, bins = self._calculate_iv_with_bins(X[feat], y)
            iv_category = self._iv_category(iv)
            uni_auc, uni_gini, uni_ks = self._calculate_univariate_metrics(
                X[feat], y
            )

            reason = ""
            if iv is None:
                eliminated.append(feat)
                reason = "Could not calculate IV"
                status = "Eliminated"
            elif iv < self.min_iv:
                eliminated.append(feat)
                reason = f"IV {iv:.4f} < {self.min_iv} (useless)"
                status = "Eliminated"
            elif iv > self.max_iv:
                eliminated.append(feat)
                reason = f"IV {iv:.4f} > {self.max_iv} (suspicious)"
                status = "Eliminated"
            else:
                kept.append(feat)
                status = "Kept"

            if iv is not None:
                self.iv_scores_[feat] = iv
            if bins:
                self.woe_bins_[feat] = bins
                for b in bins:
                    woe_table_rows.append({
                        "Feature": feat,
                        "Bin": b.bin_idx,
                        "Lower": b.lower,
                        "Upper": b.upper,
                        "Count": b.count,
                        "Goods": b.goods,
                        "Bads": b.bads,
                        "WoE": round(b.woe, 4),
                        "IV_Contribution": round(b.iv_contribution, 6),
                    })

            rows.append({
                "Feature": feat,
                "IV_Score": round(iv, 4) if iv is not None else None,
                "IV_Category": iv_category,
                "Univariate_AUC": round(uni_auc, 4) if uni_auc is not None else None,
                "Univariate_Gini": round(uni_gini, 4) if uni_gini is not None else None,
                "Univariate_KS": round(uni_ks, 4) if uni_ks is not None else None,
                "Status": status,
                "Reason": reason,
            })

        self.kept_features_ = kept
        self.eliminated_features_ = eliminated
        self.woe_table_df_ = pd.DataFrame(woe_table_rows) if woe_table_rows else pd.DataFrame()

        results_df = pd.DataFrame(rows).sort_values(
            "IV_Score", ascending=False, na_position="last"
        )
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
                "min_iv": self.min_iv,
                "max_iv": self.max_iv,
                "n_bins": self.n_bins,
                "n_features_with_woe": len(self.woe_bins_),
            },
            duration_seconds=round(duration, 1),
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop features outside the IV range.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with only kept features (columns present in X).
        """
        cols = [c for c in self.kept_features_ if c in X.columns]
        return X[cols]

    def _calculate_iv_with_bins(
        self, series: pd.Series, target: pd.Series
    ) -> Tuple[Optional[float], List[WoEBin]]:
        """Calculate IV and return WoE bin details for a single feature.

        Uses Laplace smoothing (epsilon) when a bin has zero events in either
        class. Enforces min_samples_per_bin by reducing the number of bins.

        Args:
            series: Feature values.
            target: Binary target values.

        Returns:
            Tuple of (iv_score, list_of_woe_bins). iv_score is None on failure.
        """
        try:
            data = pd.DataFrame({"feature": series, "target": target}).dropna()
            if len(data) < self.min_samples_per_bin * 2:
                return None, []

            total_goods = int((data["target"] == 0).sum())
            total_bads = int((data["target"] == 1).sum())
            if total_goods == 0 or total_bads == 0:
                return None, []

            # Determine bin count respecting min_samples_per_bin
            max_bins = max(2, len(data) // self.min_samples_per_bin)
            n_bins = min(self.n_bins, max_bins)

            try:
                data["bin"], bin_edges = pd.qcut(
                    data["feature"],
                    q=n_bins,
                    labels=False,
                    duplicates="drop",
                    retbins=True,
                )
            except (ValueError, TypeError):
                return None, []

            if data["bin"].nunique() < 2:
                return None, []

            epsilon = 1e-6
            iv = 0.0
            bins: List[WoEBin] = []

            for bin_idx, group in data.groupby("bin"):
                goods = int((group["target"] == 0).sum())
                bads = int((group["target"] == 1).sum())
                count = len(group)

                # Laplace smoothing for zero-event bins
                pct_goods = max(goods / total_goods, epsilon)
                pct_bads = max(bads / total_bads, epsilon)

                woe = float(np.log(pct_goods / pct_bads))
                iv_contrib = float((pct_goods - pct_bads) * woe)
                iv += iv_contrib

                # Determine bin boundaries from edges
                idx = int(bin_idx)
                lower = float(bin_edges[idx]) if idx < len(bin_edges) else float("-inf")
                upper = (
                    float(bin_edges[idx + 1])
                    if idx + 1 < len(bin_edges)
                    else float("inf")
                )

                bins.append(WoEBin(
                    bin_idx=idx,
                    lower=lower,
                    upper=upper,
                    count=count,
                    goods=goods,
                    bads=bads,
                    woe=woe,
                    iv_contribution=iv_contrib,
                ))

            return float(iv), bins
        except Exception:
            return None, []

    @staticmethod
    def _iv_category(iv: Optional[float]) -> str:
        """Classify IV into standard categories."""
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
        """Calculate univariate AUC, Gini, and KS for a single feature.

        Args:
            series: Feature values.
            target: Binary target values.

        Returns:
            Tuple of (AUC, Gini, KS). All None if calculation fails.
        """
        try:
            data = pd.DataFrame({"feature": series, "target": target}).dropna()
            if len(data) < 50 or data["target"].nunique() < 2:
                return None, None, None

            auc = roc_auc_score(data["target"], data["feature"])
            gini = 2 * auc - 1
            fpr, tpr, _ = roc_curve(data["target"], data["feature"])
            ks = float(max(tpr - fpr))
            return float(auc), float(gini), ks
        except Exception:
            return None, None, None
