"""
Quarterly Model Evaluator

Evaluates the final model on Train, Test, and each OOT quarter.
Produces performance tables, lift tables, and feature importance.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_score
import xgboost as xgb


logger = logging.getLogger(__name__)


def evaluate_model_quarterly(
    model: xgb.XGBClassifier,
    selected_features: List[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oot_quarters: Dict[str, pd.DataFrame],
    target_column: str = 'target',
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Evaluate model across train, test, and each OOT quarter.

    Args:
        model: Trained XGBoost model.
        selected_features: List of features used in the model.
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        oot_quarters: Dict mapping quarter labels to DataFrames.
        target_column: Name of the target column.

    Returns:
        Tuple of:
        - performance_df: Summary table with metrics per period
        - lift_tables: Dict mapping period to its lift table
        - importance_df: Feature importance DataFrame
    """
    periods = [('Train', train_df), ('Test', test_df)]
    for label in sorted(oot_quarters.keys()):
        periods.append((f'OOT_{label}', oot_quarters[label]))

    perf_rows = []
    lift_tables = {}

    for period_name, df in periods:
        if len(df) == 0:
            continue

        X = df[selected_features]
        y = df[target_column].values

        if len(np.unique(y)) < 2:
            logger.warning(
                f"OOT | {period_name}: only one class present, skipping"
            )
            continue

        y_prob = model.predict_proba(X)[:, 1]
        metrics = _calculate_metrics(y, y_prob)
        lift_table = _create_lift_table(y, y_prob)
        lift_tables[period_name] = lift_table

        # Precision and lift at top 10%
        p10, lift10 = _precision_lift_at_k(y, y_prob, k=0.10)

        perf_rows.append({
            'Period': period_name,
            'N_Samples': len(y),
            'N_Bads': int(y.sum()),
            'Bad_Rate': round(y.mean(), 4),
            'AUC': metrics['auc'],
            'Gini': metrics['gini'],
            'KS': metrics['ks'],
            'Precision_at_10pct': round(p10, 4) if p10 else None,
            'Lift_at_10pct': round(lift10, 2) if lift10 else None,
        })

        logger.info(
            f"OOT | {period_name}: AUC={metrics['auc']:.4f}, "
            f"Gini={metrics['gini']:.4f}, KS={metrics['ks']:.4f}"
        )

    performance_df = pd.DataFrame(perf_rows)

    # Feature importance
    importance_df = _feature_importance(model, selected_features)

    return performance_df, lift_tables, importance_df


def _calculate_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    """Calculate AUC, Gini, KS."""
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1

    # KS
    df = pd.DataFrame({'score': y_score, 'target': y_true})
    df = df.sort_values('score', ascending=False)
    total_bads = df['target'].sum()
    total_goods = len(df) - total_bads
    df['cum_bads'] = df['target'].cumsum() / total_bads
    df['cum_goods'] = (1 - df['target']).cumsum() / total_goods
    ks = float(abs(df['cum_bads'] - df['cum_goods']).max())

    return {
        'auc': round(auc, 4),
        'gini': round(gini, 4),
        'ks': round(ks, 4),
    }


def _create_lift_table(
    y_true: np.ndarray, y_score: np.ndarray, n_deciles: int = 10
) -> pd.DataFrame:
    """Create decile-based lift table."""
    df = pd.DataFrame({'score': y_score, 'target': y_true})

    try:
        df['decile'] = pd.qcut(
            df['score'], q=n_deciles,
            labels=list(range(n_deciles, 0, -1)),
            duplicates='drop',
        )
    except ValueError:
        df['decile'] = pd.qcut(
            df['score'].rank(method='first'), q=n_deciles,
            labels=list(range(n_deciles, 0, -1)),
            duplicates='drop',
        )

    lift = df.groupby('decile', observed=False).agg({
        'score': ['min', 'max', 'mean'],
        'target': ['count', 'sum', 'mean'],
    }).round(4)

    lift.columns = [
        'Score_Min', 'Score_Max', 'Score_Mean',
        'Count', 'Bads', 'Bad_Rate',
    ]

    overall_bad_rate = df['target'].mean()
    lift['Lift'] = lift['Bad_Rate'] / overall_bad_rate

    lift = lift.sort_index(ascending=False)
    lift['Cum_Count'] = lift['Count'].cumsum()
    lift['Cum_Bads'] = lift['Bads'].cumsum()
    lift['Cum_Bad_Rate'] = lift['Cum_Bads'] / lift['Cum_Count']
    lift['Cum_Lift'] = lift['Cum_Bad_Rate'] / overall_bad_rate

    total_bads = df['target'].sum()
    lift['Capture_Rate'] = lift['Cum_Bads'] / total_bads

    return lift.reset_index()


def _precision_lift_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate precision and lift at top k%."""
    try:
        n = len(y_true)
        top_n = max(1, int(n * k))
        order = np.argsort(-y_score)
        top_actuals = y_true[order[:top_n]]
        precision = float(top_actuals.mean())
        overall_rate = float(y_true.mean())
        lift = precision / overall_rate if overall_rate > 0 else None
        return precision, lift
    except Exception:
        return None, None


def _feature_importance(
    model: xgb.XGBClassifier, features: List[str]
) -> pd.DataFrame:
    """Extract and sort feature importances."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
    }).sort_values('Importance', ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    df['Cumulative_Importance'] = df['Importance'].cumsum()
    return df.reset_index(drop=True)
