"""
Sequential Feature Selector

Forward stepwise feature selection using XGBoost with AUC significance threshold.
"""

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb


logger = logging.getLogger(__name__)


def forward_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    iv_scores: Optional[Dict[str, float]] = None,
    auc_threshold: float = 0.0001,
    xgb_params: Optional[Dict] = None,
) -> Tuple[List[str], pd.DataFrame, 'xgb.XGBClassifier']:
    """
    Forward stepwise feature selection using XGBoost.

    Features are tried in IV-descending order. A feature is added to the
    selected set if it improves test AUC by at least `auc_threshold`.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.
        X_test: Test features DataFrame.
        y_test: Test target Series.
        features: Candidate feature names (already filtered by previous steps).
        iv_scores: IV scores per feature for ordering candidates.
        auc_threshold: Minimum AUC improvement to include a feature.
        xgb_params: XGBoost parameters. Uses sensible defaults if None.

    Returns:
        Tuple of (selected_features, selection_details_df, final_model).
    """
    if iv_scores is None:
        iv_scores = {}

    if xgb_params is None:
        xgb_params = _default_xgb_params()

    # Sort candidates by IV descending
    sorted_candidates = sorted(
        features,
        key=lambda f: iv_scores.get(f, 0) or 0,
        reverse=True,
    )

    # Auto-balance classes
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    params = xgb_params.copy()
    if params.pop('scale_pos_weight', None) == 'auto':
        params['scale_pos_weight'] = neg_count / pos_count

    # early_stopping_rounds goes to constructor in xgboost >= 2.0
    early_stopping_rounds = params.pop('early_stopping_rounds', 30)
    params['early_stopping_rounds'] = early_stopping_rounds

    selected: List[str] = []
    baseline_auc = 0.5
    step_rows = []

    logger.info(
        f"SELECTION | Starting forward selection with {len(sorted_candidates)} "
        f"candidates, AUC threshold={auc_threshold}"
    )

    for i, candidate in enumerate(sorted_candidates):
        trial_features = selected + [candidate]

        trial_auc = _train_and_evaluate(
            X_train[trial_features], y_train,
            X_test[trial_features], y_test,
            params,
        )

        improvement = trial_auc - baseline_auc

        if improvement >= auc_threshold:
            selected.append(candidate)
            baseline_auc = trial_auc
            status = "Added"
            logger.info(
                f"SELECTION | Step {len(selected):3d}: ADDED {candidate}, "
                f"AUC={trial_auc:.6f}, +{improvement:.6f}"
            )
        else:
            status = "Skipped"
            logger.info(
                f"SELECTION | Step {i + 1:3d}: SKIP  {candidate}, "
                f"AUC={trial_auc:.6f}, +{improvement:.6f} < {auc_threshold}"
            )

        step_rows.append({
            'Step': i + 1,
            'Feature': candidate,
            'Feature_IV': round(iv_scores.get(candidate, 0) or 0, 4),
            'AUC_After': round(trial_auc, 6),
            'AUC_Improvement': round(improvement, 6),
            'Status': status,
            'Cumulative_Features': len(selected),
        })

    details_df = pd.DataFrame(step_rows)

    # Train final model on selected features
    logger.info(
        f"SELECTION | Final model: {len(selected)} features, "
        f"AUC={baseline_auc:.6f}"
    )

    final_model = _train_final_model(
        X_train[selected], y_train,
        X_test[selected], y_test,
        params,
    )

    return selected, details_df, final_model


def _train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,
) -> float:
    """Train XGBoost and return test AUC."""
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, y_prob))


def _train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,
) -> xgb.XGBClassifier:
    """Train and return the final XGBoost model."""
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    return model


def _default_xgb_params() -> Dict:
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 30,
        'scale_pos_weight': 'auto',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }
