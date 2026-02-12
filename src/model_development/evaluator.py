"""
Quarterly Model Evaluator

Evaluates the final model on Train, Test, and each OOT quarter.
Produces performance tables, lift tables, and feature importance.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
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
    importance_type: str = 'gain',
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
    importance_df = _feature_importance(model, selected_features, importance_type=importance_type)

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


def _bootstrap_auc_batch(
    y: np.ndarray, y_prob: np.ndarray, n: int, seeds: List[int]
) -> List[Optional[float]]:
    """Compute a batch of bootstrap AUC samples (picklable, for joblib)."""
    aucs = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=n)
        y_boot = y[idx]
        prob_boot = y_prob[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        aucs.append(roc_auc_score(y_boot, prob_boot))
    return aucs


def bootstrap_auc_ci(
    model: xgb.XGBClassifier,
    selected_features: List[str],
    datasets: List[Tuple[str, pd.DataFrame]],
    target_column: str = 'target',
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for AUC across periods.

    Args:
        model: Trained XGBoost model.
        selected_features: List of features used in the model.
        datasets: List of (period_name, DataFrame) tuples.
        target_column: Name of the target column.
        n_iterations: Number of bootstrap iterations.
        confidence_level: Confidence level for the interval.

    Returns:
        DataFrame with columns: Period, AUC, CI_Lower, CI_Upper, N_Bootstrap.
    """
    rng = np.random.RandomState(42)
    alpha = (1 - confidence_level) / 2
    rows = []

    for period_name, df in datasets:
        if len(df) == 0:
            continue

        X = df[selected_features]
        y = df[target_column].values

        if len(np.unique(y)) < 2:
            logger.warning(
                f"Bootstrap | {period_name}: only one class present, skipping"
            )
            continue

        y_prob = model.predict_proba(X)[:, 1]
        point_auc = roc_auc_score(y, y_prob)

        n = len(y)
        if n < 10:
            rows.append({
                'Period': period_name,
                'AUC': round(point_auc, 4),
                'CI_Lower': None,
                'CI_Upper': None,
                'N_Bootstrap': 0,
            })
            continue

        # Pre-generate all seeds deterministically from main RNG
        all_seeds = rng.randint(0, 2**31, size=n_iterations).tolist()

        # Split seeds into chunks for parallel execution
        actual_n_jobs = effective_n_jobs(n_jobs)
        chunk_size = max(1, n_iterations // actual_n_jobs)
        seed_chunks = [
            all_seeds[i:i + chunk_size]
            for i in range(0, n_iterations, chunk_size)
        ]

        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_auc_batch)(y, y_prob, n, chunk)
            for chunk in seed_chunks
        )
        boot_aucs = []
        for batch in batch_results:
            boot_aucs.extend(batch)

        if len(boot_aucs) == 0:
            rows.append({
                'Period': period_name,
                'AUC': round(point_auc, 4),
                'CI_Lower': None,
                'CI_Upper': None,
                'N_Bootstrap': 0,
            })
            continue

        boot_aucs = np.array(boot_aucs)
        ci_lower = float(np.percentile(boot_aucs, 100 * alpha))
        ci_upper = float(np.percentile(boot_aucs, 100 * (1 - alpha)))

        rows.append({
            'Period': period_name,
            'AUC': round(point_auc, 4),
            'CI_Lower': round(ci_lower, 4),
            'CI_Upper': round(ci_upper, 4),
            'N_Bootstrap': len(boot_aucs),
        })

        logger.info(
            f"Bootstrap | {period_name}: AUC={point_auc:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}] ({len(boot_aucs)} iterations)"
        )

    return pd.DataFrame(rows)


def compute_score_psi(
    train_scores: np.ndarray,
    oot_scores_dict: Dict[str, np.ndarray],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI of predicted scores between train and each OOT period.

    Args:
        train_scores: Array of predicted probabilities for training set.
        oot_scores_dict: Dict mapping period_name -> scores array.
        n_bins: Number of bins for PSI calculation.

    Returns:
        DataFrame with columns: Period_1, Period_2, PSI, Status.
    """
    rows = []

    for period_name, oot_scores in oot_scores_dict.items():
        psi = _calculate_score_psi(train_scores, oot_scores, n_bins)
        if psi is None:
            status = 'N/A'
        elif psi < 0.1:
            status = 'Stable'
        elif psi < 0.25:
            status = 'Moderate'
        else:
            status = 'Significant'

        rows.append({
            'Period_1': 'Train',
            'Period_2': period_name,
            'PSI': round(psi, 4) if psi is not None else None,
            'Status': status,
        })

    return pd.DataFrame(rows)


def _calculate_score_psi(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> Optional[float]:
    """Calculate PSI between two score distributions."""
    try:
        if len(expected) < 10 or len(actual) < 10:
            return None

        try:
            _, bins = pd.qcut(expected, q=n_bins, retbins=True, duplicates='drop')
        except ValueError:
            n = min(5, len(np.unique(expected)))
            if n < 2:
                return None
            _, bins = pd.qcut(expected, q=n, retbins=True, duplicates='drop')

        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_bins = pd.cut(expected, bins=bins)
        actual_bins = pd.cut(actual, bins=bins)

        expected_pct = pd.Series(expected_bins).value_counts(normalize=True).sort_index()
        actual_pct = pd.Series(actual_bins).value_counts(normalize=True).sort_index()

        all_bins = expected_pct.index.union(actual_pct.index)
        expected_pct = expected_pct.reindex(all_bins, fill_value=0.0001)
        actual_pct = actual_pct.reindex(all_bins, fill_value=0.0001)

        epsilon = 1e-4
        expected_pct = expected_pct.clip(lower=epsilon)
        actual_pct = actual_pct.clip(lower=epsilon)

        psi = float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())
        return psi if np.isfinite(psi) else None
    except Exception:
        return None


def _feature_importance(
    model: xgb.XGBClassifier, features: List[str],
    importance_type: str = 'gain',
) -> pd.DataFrame:
    """Extract and sort feature importances."""
    try:
        booster_scores = model.get_booster().get_score(importance_type=importance_type)
        importances = np.array([booster_scores.get(f, 0.0) for f in features])
        total = importances.sum()
        if total > 0:
            importances = importances / total
    except Exception:
        importances = model.feature_importances_
    df = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
    }).sort_values('Importance', ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    df['Cumulative_Importance'] = df['Importance'].cumsum()
    return df.reset_index(drop=True)
