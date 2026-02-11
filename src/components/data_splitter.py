"""
Data Splitter Component

Splits raw data into Train/Test/OOT by application date with stratification.
Not a BaseComponent (does not filter features), but produces structured DataSplitResult.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.schema import DataConfig, SplittingConfig

logger = logging.getLogger(__name__)

STEP_NAME = "00_data_split"


@dataclass
class DataSplitResult:
    """Result of data splitting."""

    train: pd.DataFrame
    test: pd.DataFrame
    oot_quarters: Dict[str, pd.DataFrame]
    feature_columns: List[str]
    split_indices_df: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def oot_labels(self) -> List[str]:
        """Sorted list of OOT quarter labels."""
        return sorted(self.oot_quarters.keys())

    @property
    def all_oot(self) -> pd.DataFrame:
        """Combine all OOT quarters into a single DataFrame."""
        if not self.oot_quarters:
            return pd.DataFrame()
        return pd.concat(self.oot_quarters.values(), ignore_index=True)


class DataSplitter:
    """Splits data into train/test/OOT sets by date with stratified sampling.

    Args:
        data_config: DataConfig with column names and paths.
        splitting_config: SplittingConfig with train_end_date, test_size, stratify.
        seed: Global random seed for reproducibility.
    """

    def __init__(
        self,
        data_config: DataConfig,
        splitting_config: SplittingConfig,
        seed: int = 42,
    ):
        self.target_column = data_config.target_column
        self.date_column = data_config.date_column
        self.id_columns = list(data_config.id_columns)
        self.exclude_columns = list(data_config.exclude_columns)
        self.train_end_date = splitting_config.train_end_date
        self.test_size = splitting_config.test_size
        self.stratify = splitting_config.stratify
        self.seed = seed

    def split(self, df: pd.DataFrame) -> DataSplitResult:
        """Split a DataFrame into train, test, and OOT quarter sets.

        Args:
            df: Full DataFrame with features, target, date, and id columns.

        Returns:
            DataSplitResult with train, test, OOT quarters, feature columns,
            split indices, and metadata.

        Raises:
            ValueError: If required columns are missing, target is not binary,
                or splits have insufficient samples.
        """
        t0 = time.time()

        self._validate_input(df)

        # Ensure date column is datetime
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        cutoff = pd.Timestamp(self.train_end_date)
        logger.info(f"{STEP_NAME} | Train end date: {cutoff.strftime('%Y-%m-%d')}")

        # Split into training period and OOT period
        train_period = df[df[self.date_column] <= cutoff].copy()
        oot_period = df[df[self.date_column] > cutoff].copy()

        logger.info(
            f"{STEP_NAME} | Training period: {len(train_period):,} rows "
            f"({train_period[self.date_column].min():%Y-%m-%d} to "
            f"{train_period[self.date_column].max():%Y-%m-%d})"
        )

        if len(train_period) < 100:
            raise ValueError(
                f"Training period has only {len(train_period)} rows. "
                f"Need at least 100 for a meaningful split."
            )

        # Identify feature columns
        exclude_set = set(
            self.id_columns + self.exclude_columns
            + [self.target_column, self.date_column]
        )
        feature_columns = [c for c in df.columns if c not in exclude_set]
        logger.info(f"{STEP_NAME} | Feature columns: {len(feature_columns)}")

        # Stratified train/test split
        stratify_col = train_period[self.target_column] if self.stratify else None
        train_df, test_df = train_test_split(
            train_period,
            test_size=self.test_size,
            stratify=stratify_col,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        logger.info(
            f"{STEP_NAME} | Train: {len(train_df):,} rows "
            f"(bad rate: {train_df[self.target_column].mean():.2%})"
        )
        logger.info(
            f"{STEP_NAME} | Test: {len(test_df):,} rows "
            f"(bad rate: {test_df[self.target_column].mean():.2%})"
        )

        # Split OOT into quarters
        oot_quarters = self._split_into_quarters(oot_period)

        for label in sorted(oot_quarters.keys()):
            qdf = oot_quarters[label]
            bad_rate = qdf[self.target_column].mean() if len(qdf) > 0 else 0
            logger.info(
                f"{STEP_NAME} | OOT {label}: {len(qdf):,} rows "
                f"(bad rate: {bad_rate:.2%})"
            )

        if not oot_quarters:
            logger.warning(f"{STEP_NAME} | No OOT data found after {self.train_end_date}")

        # Build split indices for reproducibility
        split_indices_df = self._build_split_indices(train_df, test_df, oot_quarters)

        # Build metadata
        metadata = self._build_metadata(train_df, test_df, oot_quarters)
        metadata["duration_seconds"] = round(time.time() - t0, 1)

        logger.info(
            f"{STEP_NAME} | Split completed in {metadata['duration_seconds']:.1f}s"
        )

        return DataSplitResult(
            train=train_df,
            test=test_df,
            oot_quarters=oot_quarters,
            feature_columns=feature_columns,
            split_indices_df=split_indices_df,
            metadata=metadata,
        )

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate the input DataFrame has required columns and valid target."""
        missing_cols = []
        for col in [self.target_column, self.date_column]:
            if col not in df.columns:
                missing_cols.append(col)
        for col in self.id_columns:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate target is binary
        unique_targets = df[self.target_column].dropna().unique()
        if not set(unique_targets).issubset({0, 1}):
            raise ValueError(
                f"Target column '{self.target_column}' must be binary (0/1). "
                f"Found values: {sorted(unique_targets)}"
            )

        if len(unique_targets) < 2:
            raise ValueError(
                f"Target column '{self.target_column}' has only one class: {unique_targets}"
            )

    def _split_into_quarters(
        self, oot_period: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Split OOT data into quarterly buckets."""
        if len(oot_period) == 0:
            return {}

        quarters = {}
        oot_period = oot_period.copy()
        oot_period["_quarter"] = oot_period[self.date_column].dt.to_period("Q")

        for period, group in oot_period.groupby("_quarter"):
            label = str(period)
            quarters[label] = group.drop(columns=["_quarter"]).reset_index(drop=True)

        return quarters

    def _build_split_indices(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        oot_quarters: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Build a DataFrame mapping each row to its split set for reproducibility."""
        id_col = self.id_columns[0] if self.id_columns else None
        rows = []

        def _add_rows(df: pd.DataFrame, set_name: str) -> None:
            for idx in range(len(df)):
                row = {"set_name": set_name}
                if id_col and id_col in df.columns:
                    row[id_col] = df[id_col].iloc[idx]
                rows.append(row)

        _add_rows(train_df, "train")
        _add_rows(test_df, "test")
        for label in sorted(oot_quarters.keys()):
            _add_rows(oot_quarters[label], f"oot_{label}")

        return pd.DataFrame(rows)

    def _build_metadata(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        oot_quarters: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Build metadata dictionary with counts and bad rates per set."""
        meta: Dict[str, Any] = {
            "train_end_date": self.train_end_date,
            "test_size": self.test_size,
            "seed": self.seed,
            "train_count": len(train_df),
            "train_bad_rate": round(float(train_df[self.target_column].mean()), 4),
            "test_count": len(test_df),
            "test_bad_rate": round(float(test_df[self.target_column].mean()), 4),
        }

        for label in sorted(oot_quarters.keys()):
            qdf = oot_quarters[label]
            meta[f"oot_{label}_count"] = len(qdf)
            if len(qdf) > 0:
                meta[f"oot_{label}_bad_rate"] = round(
                    float(qdf[self.target_column].mean()), 4
                )

        return meta
