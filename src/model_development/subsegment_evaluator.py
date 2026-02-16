"""
Subsegment Model Evaluator

Evaluates model performance broken down by subsegment columns (e.g., applicant_type).
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd

from src.model_development.evaluator import calculate_metrics


logger = logging.getLogger(__name__)


def evaluate_by_subsegment(
    model,
    datasets,  # DataSets object
    selected_features: List[str],
    subsegment_columns: List[str],
    target_column: str = 'target',
) -> Dict[str, pd.DataFrame]:
    """Evaluate model performance broken down by subsegment columns.

    For each subsegment column, for each unique value, for each period
    (Train, Test, OOT_*), compute: N_Samples, N_Bads, Bad_Rate, AUC, Gini, KS.

    Args:
        model: Trained model with predict_proba(X)[:, 1] interface
        datasets: DataSets object with train/test/oot_quarters
        selected_features: Features used in the model
        subsegment_columns: List of column names to break down by
        target_column: Name of binary target column

    Returns:
        Dict mapping column_name -> DataFrame with columns:
            Subsegment_Column, Subsegment_Value, Period, N_Samples, N_Bads,
            Bad_Rate, AUC, Gini, KS
    """
    # Build periods list: [('Train', df), ('Test', df), ('OOT_Q1', df), ...]
    periods = [('Train', datasets.train), ('Test', datasets.test)]
    for label in sorted(datasets.oot_quarters.keys()):
        periods.append((f'OOT_{label}', datasets.oot_quarters[label]))

    results = {}

    for col in subsegment_columns:
        rows = []

        for period_name, df in periods:
            if col not in df.columns:
                logger.warning(f"Subsegment column '{col}' not in {period_name}, skipping")
                continue

            for seg_value in sorted(df[col].unique()):
                seg_mask = df[col] == seg_value
                seg_df = df[seg_mask]

                if len(seg_df) < 10:
                    continue

                y = seg_df[target_column].values
                if len(np.unique(y)) < 2:
                    # Only one class - can't compute AUC
                    rows.append({
                        'Subsegment_Column': col,
                        'Subsegment_Value': str(seg_value),
                        'Period': period_name,
                        'N_Samples': len(y),
                        'N_Bads': int(y.sum()),
                        'Bad_Rate': round(float(y.mean()), 4),
                        'AUC': None,
                        'Gini': None,
                        'KS': None,
                    })
                    continue

                X = seg_df[selected_features]
                y_prob = model.predict_proba(X)[:, 1]
                metrics = calculate_metrics(y, y_prob)

                rows.append({
                    'Subsegment_Column': col,
                    'Subsegment_Value': str(seg_value),
                    'Period': period_name,
                    'N_Samples': len(y),
                    'N_Bads': int(y.sum()),
                    'Bad_Rate': round(float(y.mean()), 4),
                    'AUC': metrics['auc'],
                    'Gini': metrics['gini'],
                    'KS': metrics['ks'],
                })

        if rows:
            results[col] = pd.DataFrame(rows)
            logger.info(f"SUBSEGMENT | Column '{col}': {len(rows)} segment-period combinations evaluated")

    return results


def compute_confusion_by_subsegment(
    model,
    datasets,  # DataSets object
    selected_features: List[str],
    subsegment_columns: List[str],
    target_column: str = 'target',
    thresholds: List[float] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute confusion matrix metrics broken down by subsegment columns.

    Args:
        model: Trained model
        datasets: DataSets object
        selected_features: Features used in model
        subsegment_columns: Columns to break down by
        target_column: Target column name
        thresholds: Probability thresholds

    Returns:
        Dict mapping column_name -> DataFrame with columns:
            Subsegment_Value, Period, Threshold, TP, FP, TN, FN,
            Precision, Recall, F1, Accuracy
    """
    from src.model_development.evaluator import compute_confusion_metrics

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    periods = [('Train', datasets.train), ('Test', datasets.test)]
    for label in sorted(datasets.oot_quarters.keys()):
        periods.append((f'OOT_{label}', datasets.oot_quarters[label]))

    results = {}

    for col in subsegment_columns:
        all_rows = []

        for period_name, df in periods:
            if col not in df.columns:
                continue

            for seg_value in sorted(df[col].unique()):
                seg_df = df[df[col] == seg_value]

                if len(seg_df) < 10:
                    continue

                y = seg_df[target_column].values
                if len(np.unique(y)) < 2:
                    continue

                X = seg_df[selected_features]
                y_prob = model.predict_proba(X)[:, 1]

                confusion_df = compute_confusion_metrics(y, y_prob, thresholds)
                confusion_df.insert(0, 'Period', period_name)
                confusion_df.insert(0, 'Subsegment_Value', str(seg_value))
                all_rows.append(confusion_df)

        if all_rows:
            results[col] = pd.concat(all_rows, ignore_index=True)
            logger.info(f"SUBSEGMENT | Confusion metrics for '{col}': {len(all_rows)} segments")

    return results
