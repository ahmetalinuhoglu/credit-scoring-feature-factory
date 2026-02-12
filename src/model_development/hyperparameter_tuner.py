"""
Hyperparameter Tuner

Optuna-based hyperparameter optimization for XGBoost credit scoring models.
Uses TPE sampler with stability-aware objective: each trial is evaluated on
train, test, and OOT periods. The objective penalises deviation across periods
so that only models that generalise well are selected.

Fallback: when no OOT data is available, uses stratified k-fold CV on
training data (original behaviour).
"""

from typing import Dict, List, Optional, Tuple
import logging
import threading

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import optuna


logger = logging.getLogger(__name__)


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    n_trials: int = 100,
    timeout: Optional[int] = None,
    cv: int = 5,
    scoring: str = 'roc_auc',
    random_state: int = 42,
    fixed_params: Optional[Dict] = None,
    n_jobs: int = 1,
    oot_quarters: Optional[Dict[str, pd.DataFrame]] = None,
    target_column: str = 'target',
    stability_weight: float = 1.0,
) -> Tuple[Dict, pd.DataFrame, xgb.XGBClassifier]:
    """
    Tune XGBoost hyperparameters using Optuna.

    When OOT quarters are provided the objective is *stability-aware*:
    each trial trains on the full training set (early-stopped on test)
    and is evaluated on train, test, and every OOT quarter.  The score
    is ``mean(all_aucs) - stability_weight * std(all_aucs)`` so that
    models with unstable cross-period performance are penalised even if
    their average AUC is high.

    When no OOT quarters are provided, falls back to stratified k-fold
    CV on training data (original behaviour).

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data (used for early stopping + evaluation)
    features : feature column names
    n_trials : number of Optuna trials
    timeout : max seconds for search
    cv : CV folds (used only in fallback mode)
    random_state : seed
    fixed_params : extra fixed XGBoost params
    n_jobs : parallel workers (-1 = all cores)
    oot_quarters : dict mapping quarter label -> DataFrame with target
    target_column : name of binary target column in OOT DataFrames
    stability_weight : penalty multiplier for std across periods
        0 = optimise raw mean AUC; 1 = equal weight on stability

    Returns
    -------
    best_params, trial_history_df, tuned_model
    """
    # Build fixed params
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())

    fixed = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': random_state,
        'early_stopping_rounds': 30,
        'scale_pos_weight': neg_count / pos_count,
    }
    if fixed_params:
        fixed.update(fixed_params)

    # Nested parallelism: when running Optuna trials in parallel,
    # force XGBoost to single-threaded per trial.
    optuna_n_jobs = 1
    if n_jobs != 1:
        optuna_n_jobs = n_jobs
        fixed['n_jobs'] = 1

    # Decide objective mode
    use_stability = bool(oot_quarters)
    if use_stability:
        # Pre-extract OOT arrays once (avoid repeated DataFrame lookups)
        oot_arrays = []
        for label in sorted(oot_quarters.keys()):
            oot_df = oot_quarters[label]
            y_oot = oot_df[target_column].values
            if len(np.unique(y_oot)) >= 2:
                oot_arrays.append((label, oot_df[features], y_oot))

    mode_str = "stability-aware" if use_stability else f"CV (k={cv})"
    logger.info(
        f"TUNING | Starting Optuna tuning — {mode_str}, "
        f"{n_trials} trials, n_jobs={n_jobs}"
        + (f", stability_weight={stability_weight}" if use_stability else "")
    )
    logger.info(
        f"TUNING | scale_pos_weight={fixed['scale_pos_weight']:.4f} "
        f"(neg={neg_count}, pos={pos_count})"
    )

    best_score = -np.inf
    _best_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Stability-aware objective (train + test + OOT)
    # ------------------------------------------------------------------
    if use_stability:
        def objective(trial: optuna.Trial) -> float:
            nonlocal best_score

            params = _suggest_params(trial)
            all_params = {**params, **fixed}

            model = xgb.XGBClassifier(**all_params)
            model.fit(
                X_train[features], y_train,
                eval_set=[(X_test[features], y_test)],
                verbose=False,
            )

            # Evaluate on every period
            period_aucs = {}
            train_prob = model.predict_proba(X_train[features])[:, 1]
            period_aucs['Train'] = float(roc_auc_score(y_train, train_prob))

            test_prob = model.predict_proba(X_test[features])[:, 1]
            period_aucs['Test'] = float(roc_auc_score(y_test, test_prob))

            for label, X_oot, y_oot in oot_arrays:
                oot_prob = model.predict_proba(X_oot)[:, 1]
                period_aucs[f'OOT_{label}'] = float(
                    roc_auc_score(y_oot, oot_prob)
                )

            all_aucs = list(period_aucs.values())
            mean_auc = float(np.mean(all_aucs))
            std_auc = float(np.std(all_aucs))
            score = mean_auc - stability_weight * std_auc

            # Store per-period AUCs for trial history
            for k, v in period_aucs.items():
                trial.set_user_attr(f'AUC_{k}', round(v, 4))
            trial.set_user_attr('AUC_Mean', round(mean_auc, 4))
            trial.set_user_attr('AUC_Std', round(std_auc, 4))

            # Thread-safe best tracking & logging
            with _best_lock:
                is_new_best = score > best_score
                if is_new_best:
                    best_score = score

            if trial.number % 10 == 0 or is_new_best:
                auc_parts = ", ".join(
                    f"{k}={v:.4f}" for k, v in period_aucs.items()
                )
                msg = (
                    f"TUNING | Trial {trial.number}: "
                    f"Score={score:.4f} "
                    f"(Mean={mean_auc:.4f}, Std={std_auc:.4f}) "
                    f"[{auc_parts}]"
                )
                if is_new_best:
                    msg += " (new best)"
                logger.info(msg)

            return score

    # ------------------------------------------------------------------
    # Fallback: CV-only objective (no OOT data available)
    # ------------------------------------------------------------------
    else:
        def objective(trial: optuna.Trial) -> float:
            nonlocal best_score

            params = _suggest_params(trial)
            all_params = {**params, **fixed}

            skf = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
            cv_aucs = []

            for train_idx, val_idx in skf.split(X_train[features], y_train):
                X_fold_tr = X_train[features].iloc[train_idx]
                y_fold_tr = y_train.iloc[train_idx]
                X_fold_val = X_train[features].iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                model = xgb.XGBClassifier(**all_params)
                model.fit(
                    X_fold_tr, y_fold_tr,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False,
                )
                y_prob = model.predict_proba(X_fold_val)[:, 1]
                cv_aucs.append(roc_auc_score(y_fold_val, y_prob))

            mean_auc = float(np.mean(cv_aucs))
            std_auc = float(np.std(cv_aucs))
            trial.set_user_attr('cv_std', std_auc)

            with _best_lock:
                is_new_best = mean_auc > best_score
                if is_new_best:
                    best_score = mean_auc

            if trial.number % 10 == 0 or is_new_best:
                msg = (
                    f"TUNING | Trial {trial.number}: "
                    f"CV AUC = {mean_auc:.4f} ± {std_auc:.4f}"
                )
                if is_new_best:
                    msg += " (new best)"
                logger.info(msg)

            return mean_auc

    # Run optimisation
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, n_jobs=optuna_n_jobs,
    )

    # ── Best trial summary ────────────────────────────────────────────
    best_trial = study.best_trial
    logger.info(
        f"TUNING | Best trial: #{best_trial.number}, "
        f"Score = {best_trial.value:.4f}"
    )
    best_params_str = ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in best_trial.params.items()
    )
    logger.info(f"TUNING | Best params: {best_params_str}")
    if use_stability:
        auc_parts = ", ".join(
            f"{k}={v}"
            for k, v in best_trial.user_attrs.items()
            if k.startswith('AUC_')
        )
        logger.info(f"TUNING | Best trial AUCs: {auc_parts}")

    # ── Trial history DataFrame ───────────────────────────────────────
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {'Trial': trial.number}
        row.update(trial.params)
        row['n_estimators'] = 1000
        if use_stability:
            for k, v in trial.user_attrs.items():
                if k.startswith('AUC_'):
                    row[k] = v
            row['Score'] = round(trial.value, 4)
        else:
            row['CV_AUC_Mean'] = trial.value
            row['CV_AUC_Std'] = trial.user_attrs.get('cv_std', None)
        row['Duration_Sec'] = (
            round(trial.duration.total_seconds(), 1)
            if trial.duration else None
        )
        rows.append(row)

    trial_history_df = pd.DataFrame(rows)

    # Ensure column order
    param_cols = [
        'Trial', 'max_depth', 'learning_rate', 'n_estimators',
        'subsample', 'colsample_bytree', 'min_child_weight',
        'gamma', 'reg_alpha', 'reg_lambda',
    ]
    remaining_cols = [
        c for c in trial_history_df.columns
        if c not in param_cols
    ]
    ordered_cols = [
        c for c in param_cols if c in trial_history_df.columns
    ] + remaining_cols
    trial_history_df = trial_history_df[ordered_cols]

    # ── Train final model with best params ────────────────────────────
    logger.info("TUNING | Training final model with best params...")

    best_params = study.best_params.copy()
    best_params.update(fixed)
    best_params['n_estimators'] = 1000
    best_params['n_jobs'] = -1  # restore full parallelism for final model

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        X_train[features], y_train,
        eval_set=[(X_test[features], y_test)],
        verbose=False,
    )

    y_test_pred = final_model.predict_proba(X_test[features])[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)
    logger.info(f"TUNING | Final model test AUC: {test_auc:.4f}")

    return best_params, trial_history_df, final_model


# ------------------------------------------------------------------
# Shared hyperparameter search space
# ------------------------------------------------------------------

def _suggest_params(trial: optuna.Trial) -> Dict:
    """Suggest XGBoost hyperparameters for an Optuna trial."""
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.005, 0.1, log=True
        ),
        'n_estimators': 1000,
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree', 0.5, 1.0
        ),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float(
            'reg_alpha', 1e-4, 10.0, log=True
        ),
        'reg_lambda': trial.suggest_float(
            'reg_lambda', 1e-3, 25.0, log=True
        ),
    }
