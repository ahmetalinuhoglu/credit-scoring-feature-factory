"""
Missing Value Filter Component

Eliminates features whose missing rate on training data exceeds the configured
threshold. Reports missing rates across train/test/OOT for comparison.
"""

from typing import Any, Dict, List, Optional
import logging
import time

import pandas as pd

from src.pipeline.base import BaseComponent, StepResult
from src.config.schema import MissingConfig

logger = logging.getLogger(__name__)

STEP_NAME = "02_missing"


class MissingFilter(BaseComponent):
    """Remove features with high missing rate on training data.

    Args:
        config: MissingConfig with threshold (max allowed missing fraction).
    """

    step_name = STEP_NAME
    step_order = 2

    def __init__(self, config: MissingConfig):
        self.threshold = config.threshold
        self.kept_features_: List[str] = []
        self.eliminated_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> StepResult:
        """Identify features with missing rate exceeding threshold.

        Decision is based on training data only, but missing rates for test
        and OOT sets are reported for comparison if provided via kwargs.

        Args:
            X: Training feature DataFrame.
            y: Training target Series (unused).
            **kwargs:
                X_test: Optional test feature DataFrame for comparison.
                X_oot: Optional dict of {label: DataFrame} for OOT comparison.

        Returns:
            StepResult with per-feature missing rates on train (+ test/OOT if available).
        """
        t0 = time.time()
        features = list(X.columns)
        X_test: Optional[pd.DataFrame] = kwargs.get("X_test")
        X_oot: Optional[Dict[str, pd.DataFrame]] = kwargs.get("X_oot")

        n_train = len(X)
        rows = []
        kept, eliminated = [], []

        for feat in features:
            train_missing = int(X[feat].isna().sum())
            train_rate = train_missing / n_train if n_train > 0 else 0.0

            if train_rate > self.threshold:
                eliminated.append(feat)
                status = "Eliminated"
            else:
                kept.append(feat)
                status = "Kept"

            row: Dict[str, Any] = {
                "Feature": feat,
                "Train_Missing_Count": train_missing,
                "Train_Missing_Rate": round(train_rate, 4),
                "Train_Total_Rows": n_train,
                "Status": status,
            }

            # Test missing rate for comparison
            if X_test is not None and feat in X_test.columns:
                n_test = len(X_test)
                test_missing = int(X_test[feat].isna().sum())
                row["Test_Missing_Count"] = test_missing
                row["Test_Missing_Rate"] = round(
                    test_missing / n_test if n_test > 0 else 0.0, 4
                )

            # OOT missing rates for comparison
            if X_oot is not None:
                for label in sorted(X_oot.keys()):
                    oot_df = X_oot[label]
                    if feat in oot_df.columns:
                        n_oot = len(oot_df)
                        oot_missing = int(oot_df[feat].isna().sum())
                        row[f"OOT_{label}_Missing_Rate"] = round(
                            oot_missing / n_oot if n_oot > 0 else 0.0, 4
                        )

            rows.append(row)

        self.kept_features_ = kept
        self.eliminated_features_ = eliminated

        results_df = pd.DataFrame(rows).sort_values(
            "Train_Missing_Rate", ascending=False
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
            metadata={"threshold": self.threshold},
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
