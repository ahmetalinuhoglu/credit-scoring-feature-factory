"""
Constant Feature Filter Component

Eliminates features with zero or near-zero variance (fewer than min_unique_values
unique values) on training data.
"""

from typing import Any, Dict, List
import logging
import time

import pandas as pd

from src.pipeline.base import BaseComponent, StepResult
from src.config.schema import ConstantConfig

logger = logging.getLogger(__name__)

STEP_NAME = "01_constant"


class ConstantFilter(BaseComponent):
    """Remove features with constant or near-constant values.

    Args:
        config: ConstantConfig with min_unique_values threshold.
    """

    step_name = STEP_NAME
    step_order = 1

    def __init__(self, config: ConstantConfig):
        self.min_unique_values = config.min_unique_values
        self.kept_features_: List[str] = []
        self.eliminated_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> StepResult:
        """Identify features with fewer than min_unique_values unique values.

        Args:
            X: Training feature DataFrame.
            y: Training target Series (unused, kept for interface consistency).

        Returns:
            StepResult with per-feature unique count, variance, and status.
        """
        t0 = time.time()
        features = list(X.columns)
        rows = []
        kept, eliminated = [], []

        for feat in features:
            series = X[feat]
            n_unique = series.nunique()
            variance = (
                float(series.var()) if pd.api.types.is_numeric_dtype(series) else 0.0
            )

            if n_unique < self.min_unique_values:
                eliminated.append(feat)
                status = "Eliminated"
            else:
                kept.append(feat)
                status = "Kept"

            rows.append({
                "Feature": feat,
                "Unique_Count": n_unique,
                "Variance": round(variance, 6) if variance is not None else None,
                "Status": status,
            })

        self.kept_features_ = kept
        self.eliminated_features_ = eliminated

        results_df = pd.DataFrame(rows).sort_values("Unique_Count")
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
            metadata={"min_unique_values": self.min_unique_values},
            duration_seconds=round(duration, 1),
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop eliminated features from the DataFrame.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with only kept features (columns present in X).
        """
        cols = [c for c in self.kept_features_ if c in X.columns]
        return X[cols]
