"""
Correlation Filter Component

Eliminates highly correlated features using a greedy IV-ordered approach.
The higher-IV feature in each correlated pair is kept. Optionally computes
VIF (Variance Inflation Factor) for surviving features.
"""

from typing import Any, Dict, List, Optional
import logging
import time

import numpy as np
import pandas as pd

from src.pipeline.base import BaseComponent, StepResult
from src.config.schema import CorrelationConfig

logger = logging.getLogger(__name__)

STEP_NAME = "05_correlation"


class CorrelationFilter(BaseComponent):
    """Remove correlated features using greedy IV-ordered elimination.

    Features are sorted by IV descending. The highest-IV feature eliminates
    all features correlated above the threshold. An eliminated feature cannot
    eliminate others.

    Args:
        config: CorrelationConfig with threshold and method (pearson/spearman/kendall).
    """

    step_name = STEP_NAME
    step_order = 5

    def __init__(self, config: CorrelationConfig):
        self.threshold = config.threshold
        self.method = config.method
        self.kept_features_: List[str] = []
        self.eliminated_features_: List[str] = []
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.corr_pairs_df_: Optional[pd.DataFrame] = None
        self.vif_df_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> StepResult:
        """Run greedy correlation elimination keeping higher-IV features.

        Args:
            X: Training feature DataFrame.
            y: Training target Series (unused directly).
            **kwargs:
                iv_scores: Dict[str, float] mapping feature names to IV scores.
                    Required for IV-ordered elimination.

        Returns:
            StepResult with correlation pairs, decisions, and VIF values.
        """
        t0 = time.time()
        features = list(X.columns)
        iv_scores: Dict[str, float] = kwargs.get("iv_scores", {})

        # Sort features by IV descending
        sorted_features = sorted(
            features,
            key=lambda f: iv_scores.get(f, 0) or 0,
            reverse=True,
        )

        # Compute correlation matrix
        logger.info(
            f"{STEP_NAME} | Computing {self.method} correlation matrix "
            f"for {len(sorted_features)} features"
        )
        corr_matrix = X[sorted_features].corr(method=self.method)
        self.correlation_matrix_ = corr_matrix

        # Greedy elimination
        eliminated = set()
        elimination_log: List[Dict[str, Any]] = []

        for feat_a in sorted_features:
            if feat_a in eliminated:
                continue
            for feat_b in sorted_features:
                if feat_b == feat_a or feat_b in eliminated:
                    continue
                corr_val = corr_matrix.loc[feat_a, feat_b]
                if abs(corr_val) > self.threshold:
                    eliminated.add(feat_b)
                    elimination_log.append({
                        "eliminated": feat_b,
                        "by": feat_a,
                        "correlation": corr_val,
                    })

        kept = [f for f in sorted_features if f not in eliminated]
        eliminated_list = [f for f in sorted_features if f in eliminated]

        self.kept_features_ = kept
        self.eliminated_features_ = eliminated_list

        # Build elimination details DataFrame
        elim_rows = []
        for entry in elimination_log:
            elim_rows.append({
                "Eliminated_Feature": entry["eliminated"],
                "Eliminated_By": entry["by"],
                "Correlation": round(entry["correlation"], 4),
                "Eliminated_IV": round(iv_scores.get(entry["eliminated"], 0) or 0, 4),
                "Kept_IV": round(iv_scores.get(entry["by"], 0) or 0, 4),
            })
        results_df = pd.DataFrame(elim_rows)

        # Build full correlation pairs sheet
        corr_pairs_rows = []
        seen = set()
        for i, fa in enumerate(sorted_features):
            for fb in sorted_features[i + 1:]:
                corr_val = corr_matrix.loc[fa, fb]
                if abs(corr_val) > self.threshold:
                    pair_key = tuple(sorted([fa, fb]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        if fa in eliminated:
                            decision = (
                                f"{fa} eliminated by {fb}"
                                if fb not in eliminated
                                else "both eliminated"
                            )
                        elif fb in eliminated:
                            decision = f"{fb} eliminated by {fa}"
                        else:
                            decision = "both kept"
                        corr_pairs_rows.append({
                            "Feature_A": fa,
                            "Feature_B": fb,
                            "Correlation": round(corr_val, 4),
                            "IV_A": round(iv_scores.get(fa, 0) or 0, 4),
                            "IV_B": round(iv_scores.get(fb, 0) or 0, 4),
                            "Decision": decision,
                        })
        self.corr_pairs_df_ = pd.DataFrame(corr_pairs_rows)

        # Compute VIF for surviving features
        self.vif_df_ = self._compute_vif(X, kept)

        duration = time.time() - t0

        logger.info(
            f"{STEP_NAME} | Eliminated {len(eliminated_list)} features "
            f"({len(kept)} remaining) in {duration:.1f}s"
        )

        return StepResult(
            step_name=self.step_name,
            input_features=features,
            output_features=kept,
            eliminated_features=eliminated_list,
            results_df=results_df,
            metadata={
                "threshold": self.threshold,
                "method": self.method,
                "n_corr_pairs": len(corr_pairs_rows),
            },
            duration_seconds=round(duration, 1),
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop correlated features from the DataFrame.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with only kept features (columns present in X).
        """
        cols = [c for c in self.kept_features_ if c in X.columns]
        return X[cols]

    @staticmethod
    def _compute_vif(
        X: pd.DataFrame, features: List[str]
    ) -> Optional[pd.DataFrame]:
        """Compute Variance Inflation Factor for a set of features.

        VIF measures multicollinearity. VIF > 5 indicates moderate correlation,
        VIF > 10 indicates high correlation.

        Args:
            X: Training DataFrame.
            features: List of feature names to compute VIF for.

        Returns:
            DataFrame with Feature and VIF columns, or None on failure.
        """
        if len(features) < 2:
            return None

        try:
            from numpy.linalg import LinAlgError

            subset = X[features].dropna()
            if len(subset) < len(features) + 1:
                return None

            # Standardize to avoid numerical issues
            std = subset.std()
            nonzero = std > 0
            if nonzero.sum() < 2:
                return None

            cols = [f for f, nz in zip(features, nonzero) if nz]
            subset = subset[cols]
            subset = (subset - subset.mean()) / subset.std()

            corr = subset.corr().values
            try:
                inv_corr = np.linalg.inv(corr)
                vif_values = np.diag(inv_corr)
            except (LinAlgError, np.linalg.LinAlgError):
                # Fallback: use pseudo-inverse
                inv_corr = np.linalg.pinv(corr)
                vif_values = np.diag(inv_corr)

            return pd.DataFrame({
                "Feature": cols,
                "VIF": [round(float(v), 2) for v in vif_values],
            }).sort_values("VIF", ascending=False)
        except Exception:
            logger.warning(f"{STEP_NAME} | VIF computation failed, skipping")
            return None
