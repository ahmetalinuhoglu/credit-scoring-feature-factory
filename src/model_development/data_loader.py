"""
Data Loader

Loads feature parquet and splits into Train/Test/OOT by application_date.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class DataSets:
    """Container for train/test/OOT data splits."""
    train: pd.DataFrame
    test: pd.DataFrame
    oot_quarters: Dict[str, pd.DataFrame] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    meta_columns: List[str] = field(default_factory=list)
    target_column: str = 'target'
    date_column: str = 'application_date'

    @property
    def all_oot(self) -> pd.DataFrame:
        """Combine all OOT quarters into single DataFrame."""
        if not self.oot_quarters:
            return pd.DataFrame()
        return pd.concat(self.oot_quarters.values(), ignore_index=True)

    @property
    def oot_labels(self) -> List[str]:
        """Sorted list of OOT quarter labels."""
        return sorted(self.oot_quarters.keys())


def load_and_split(
    input_path: str,
    train_end_date: str,
    target_column: str = 'target',
    date_column: str = 'application_date',
    id_columns: Optional[List[str]] = None,
    meta_columns: Optional[List[str]] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DataSets:
    """
    Load parquet and split into Train/Test/OOT.

    Args:
        input_path: Path to features parquet file.
        train_end_date: Cutoff date (YYYY-MM-DD). Data up to this date is train+test,
                        data after is OOT (split into quarters automatically).
        target_column: Name of binary target column.
        date_column: Name of application date column.
        id_columns: Identifier columns to exclude from features.
        meta_columns: Metadata columns to exclude from features.
        test_size: Fraction of training period data to use as test set.
        random_state: Random seed for reproducible train/test split.

    Returns:
        DataSets with train, test, and OOT quarter DataFrames.
    """
    if id_columns is None:
        id_columns = ['application_id', 'customer_id']
    if meta_columns is None:
        meta_columns = ['applicant_type']

    # Load parquet
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns):,} columns")

    # Ensure date column is datetime
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    else:
        raise ValueError(f"Date column '{date_column}' not found in data")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    # Parse cutoff date
    cutoff = pd.Timestamp(train_end_date)
    logger.info(f"Train end date: {cutoff.strftime('%Y-%m-%d')}")

    # Split into training period and OOT period
    train_period = df[df[date_column] <= cutoff].copy()
    oot_period = df[df[date_column] > cutoff].copy()

    logger.info(
        f"Training period: {len(train_period):,} rows "
        f"({train_period[date_column].min().strftime('%Y-%m-%d')} to "
        f"{train_period[date_column].max().strftime('%Y-%m-%d')})"
    )

    # Identify feature columns (exclude id, meta, target, date)
    exclude_cols = set(id_columns + meta_columns + [target_column, date_column])
    feature_columns = [c for c in df.columns if c not in exclude_cols]
    logger.info(f"Feature columns: {len(feature_columns)}")

    # Stratified train/test split within training period
    train_df, test_df = _stratified_split(
        train_period, target_column, test_size, random_state
    )

    logger.info(
        f"Train: {len(train_df):,} rows "
        f"(bad rate: {train_df[target_column].mean():.2%})"
    )
    logger.info(
        f"Test: {len(test_df):,} rows "
        f"(bad rate: {test_df[target_column].mean():.2%})"
    )

    # Split OOT into quarters
    oot_quarters = _split_into_quarters(oot_period, date_column)

    for label, qdf in sorted(oot_quarters.items()):
        bad_rate = qdf[target_column].mean() if len(qdf) > 0 else 0
        logger.info(
            f"OOT {label}: {len(qdf):,} rows (bad rate: {bad_rate:.2%})"
        )

    if not oot_quarters:
        logger.warning("No OOT data found after train_end_date")

    return DataSets(
        train=train_df,
        test=test_df,
        oot_quarters=oot_quarters,
        feature_columns=feature_columns,
        id_columns=id_columns,
        meta_columns=meta_columns,
        target_column=target_column,
        date_column=date_column,
    )


def _stratified_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split maintaining target distribution."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _split_into_quarters(
    df: pd.DataFrame,
    date_column: str,
) -> Dict[str, pd.DataFrame]:
    """Split DataFrame into quarterly buckets based on date column."""
    if len(df) == 0:
        return {}

    quarters = {}
    df = df.copy()
    df['_quarter'] = df[date_column].dt.to_period('Q')

    for period, group in df.groupby('_quarter'):
        label = str(period)  # e.g. "2023Q3"
        quarters[label] = group.drop(columns=['_quarter']).reset_index(drop=True)

    return quarters
