"""
Sequential Feature Selection (Forward & Backward)

Proper SFS/SBS implementation using cross-validated AUC with XGBoost.
Includes elbow detection (1-SE rule) and performance chart generation.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default XGBoost parameters
# ---------------------------------------------------------------------------

def _default_xgb_params() -> Dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,
        "scale_pos_weight": "auto",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }


# ---------------------------------------------------------------------------
# CV evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_feature_set_cv(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    xgb_params: Dict,
    cv: int,
    random_state: int,
) -> Tuple[float, float]:
    """Return (mean_auc, std_auc) from StratifiedKFold CV."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    aucs: List[float] = []

    for train_idx, val_idx in skf.split(X[features], y):
        X_tr = X[features].iloc[train_idx]
        X_val = X[features].iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_prob))

    return float(np.mean(aucs)), float(np.std(aucs))


# ---------------------------------------------------------------------------
# Elbow detection (1-SE rule)
# ---------------------------------------------------------------------------

def _find_elbow_1se(mean_aucs: List[float], std_aucs: List[float]) -> int:
    """
    1-SE rule: find the best mean CV AUC, then return the index of the
    smallest k where mean_auc >= best_auc - 1 * std_auc_at_best.

    Returns the 0-based index into the lists.
    """
    best_idx = int(np.argmax(mean_aucs))
    threshold = mean_aucs[best_idx] - 1.0 * std_aucs[best_idx]

    for i in range(len(mean_aucs)):
        if mean_aucs[i] >= threshold:
            return i

    # Fallback: return best index
    return best_idx


# ---------------------------------------------------------------------------
# Performance chart
# ---------------------------------------------------------------------------

def _save_performance_chart(
    step_details_df: pd.DataFrame,
    direction: str,
    output_dir: Path,
) -> str:
    """
    Save a performance chart showing CV AUC vs number of features.

    Returns the path to the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = str(output_dir / f"selection_chart_{timestamp}.png")

    n_features = step_details_df["N_Features"].values
    mean_aucs = step_details_df["Mean_CV_AUC"].values
    std_aucs = step_details_df["Std_CV_AUC"].values

    # Find the optimal row
    optimal_rows = step_details_df[step_details_df["Is_Optimal"]]
    if len(optimal_rows) > 0:
        optimal_k = optimal_rows.iloc[0]["N_Features"]
    else:
        optimal_k = None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Mean AUC line
    ax.plot(n_features, mean_aucs, "o-", color="#2563eb", linewidth=2, markersize=5,
            label="Mean CV AUC")

    # Shaded band: +/- 1 std dev
    ax.fill_between(
        n_features,
        mean_aucs - std_aucs,
        mean_aucs + std_aucs,
        alpha=0.2,
        color="#2563eb",
        label="\u00b11 Std Dev",
    )

    # Vertical dashed line at elbow
    if optimal_k is not None:
        ax.axvline(
            x=optimal_k,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label=f"Optimal k={optimal_k} (1-SE rule)",
        )

    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("Mean CV AUC", fontsize=12)
    ax.set_title("Sequential Feature Selection Performance", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Force integer x-ticks when range is small
    if len(n_features) <= 30:
        ax.set_xticks(n_features)

    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"SELECTION | Performance chart saved to {chart_path}")
    return chart_path


# ---------------------------------------------------------------------------
# Prepare XGBoost params (shared logic)
# ---------------------------------------------------------------------------

def _prepare_xgb_params(
    xgb_params: Optional[Dict],
    y_train: pd.Series,
    random_state: int,
) -> Dict:
    """
    Prepare XGBoost params: apply auto scale_pos_weight, move
    early_stopping_rounds to constructor kwarg (xgboost >= 2.0 compat).
    """
    params = xgb_params.copy() if xgb_params else _default_xgb_params()

    # Auto-balance classes
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    if params.pop("scale_pos_weight", None) == "auto":
        params["scale_pos_weight"] = neg_count / pos_count

    # early_stopping_rounds goes to constructor in xgboost >= 2.0
    early_stopping_rounds = params.pop("early_stopping_rounds", 30)
    params["early_stopping_rounds"] = early_stopping_rounds

    # Ensure random_state is set
    params.setdefault("random_state", random_state)

    return params


# ---------------------------------------------------------------------------
# Forward selection
# ---------------------------------------------------------------------------

def _forward_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: List[str],
    iv_scores: Optional[Dict[str, float]],
    xgb_params: Dict,
    cv: int,
    min_features: int,
    max_features: int,
    tolerance: float,
    patience: int,
    random_state: int,
    n_jobs: int = 1,
) -> List[Dict]:
    """
    Proper sequential forward selection (SFS).

    At each step, evaluate ALL remaining candidates and pick the one that
    maximises mean CV AUC.
    """
    # If iv_scores provided, sort candidates by IV descending as a speedup
    # hint for the first step (all candidates are still evaluated)
    if iv_scores:
        remaining = sorted(
            candidates,
            key=lambda f: iv_scores.get(f, 0) or 0,
            reverse=True,
        )
    else:
        remaining = list(candidates)

    selected: List[str] = []
    step_rows: List[Dict] = []
    best_auc_so_far = -np.inf
    no_improve_count = 0

    for step in range(1, max_features + 1):
        if not remaining:
            logger.info("SELECTION | No more candidates remaining.")
            break

        logger.info(
            f"SELECTION | Forward step {step}/{max_features}: "
            f"evaluating {len(remaining)} candidates..."
        )

        # Evaluate every remaining candidate (parallel across candidates)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_feature_set_cv)(
                X_train, y_train, selected + [c], xgb_params, cv, random_state
            )
            for c in remaining
        )

        best_candidate = None
        best_mean_auc = -np.inf
        best_std_auc = 0.0
        for i, (mean_auc, std_auc) in enumerate(results):
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_std_auc = std_auc
                best_candidate = remaining[i]

        # Add the best candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)

        feature_iv = round((iv_scores or {}).get(best_candidate, 0) or 0, 4)

        logger.info(
            f"SELECTION | Step {step}: ADDED {best_candidate} "
            f"(IV={feature_iv}), CV AUC={best_mean_auc:.4f} "
            f"\u00b1 {best_std_auc:.4f}"
        )

        step_rows.append({
            "Step": step,
            "N_Features": len(selected),
            "Added_Feature": best_candidate,
            "Feature_IV": feature_iv,
            "Mean_CV_AUC": round(best_mean_auc, 6),
            "Std_CV_AUC": round(best_std_auc, 6),
            "Is_Optimal": False,  # will be set later by elbow detection
        })

        # Early stopping check (only after min_features reached)
        improvement = best_mean_auc - best_auc_so_far
        if best_mean_auc > best_auc_so_far:
            best_auc_so_far = best_mean_auc

        if len(selected) >= min_features:
            if improvement < tolerance:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(
                        f"SELECTION | Early stopping at step {step} "
                        f"(no improvement for {patience} steps)"
                    )
                    break
            else:
                no_improve_count = 0

    return step_rows


# ---------------------------------------------------------------------------
# Backward selection
# ---------------------------------------------------------------------------

def _backward_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: List[str],
    iv_scores: Optional[Dict[str, float]],
    xgb_params: Dict,
    cv: int,
    min_features: int,
    max_features: int,
    tolerance: float,
    patience: int,
    random_state: int,
    n_jobs: int = 1,
) -> List[Dict]:
    """
    Proper sequential backward selection (SBS).

    Start with all features, at each step remove the one whose removal
    causes the least AUC degradation (or even improves it).
    """
    current_features = list(candidates)
    step_rows: List[Dict] = []
    no_degrade_count = 0

    # Evaluate baseline with all features
    baseline_auc, baseline_std = _evaluate_feature_set_cv(
        X_train, y_train, current_features, xgb_params, cv, random_state
    )
    logger.info(
        f"SELECTION | Backward baseline: {len(current_features)} features, "
        f"CV AUC={baseline_auc:.4f} \u00b1 {baseline_std:.4f}"
    )

    step_num = 0

    while len(current_features) > min_features:
        step_num += 1

        logger.info(
            f"SELECTION | Backward step {step_num}: evaluating removal of "
            f"{len(current_features)} features..."
        )

        # Evaluate removing each feature (parallel across candidates)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_feature_set_cv)(
                X_train, y_train,
                [f for f in current_features if f != feature],
                xgb_params, cv, random_state,
            )
            for feature in current_features
        )

        best_removal = None
        best_mean_auc = -np.inf
        best_std_auc = 0.0
        for i, (mean_auc, std_auc) in enumerate(results):
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_std_auc = std_auc
                best_removal = current_features[i]

        # Remove the feature whose removal causes least degradation
        current_features.remove(best_removal)

        feature_iv = round((iv_scores or {}).get(best_removal, 0) or 0, 4)

        logger.info(
            f"SELECTION | Step {step_num}: REMOVED {best_removal} "
            f"(IV={feature_iv}), CV AUC={best_mean_auc:.4f} "
            f"\u00b1 {best_std_auc:.4f}"
        )

        step_rows.append({
            "Step": step_num,
            "N_Features": len(current_features),
            "Removed_Feature": best_removal,
            "Feature_IV": feature_iv,
            "Mean_CV_AUC": round(best_mean_auc, 6),
            "Std_CV_AUC": round(best_std_auc, 6),
            "Is_Optimal": False,  # will be set later by elbow detection
        })

        # Early stopping: if removing any feature degrades AUC by more than
        # tolerance for `patience` consecutive steps, stop
        degradation = baseline_auc - best_mean_auc
        if degradation > tolerance:
            no_degrade_count += 1
            if no_degrade_count >= patience:
                logger.info(
                    f"SELECTION | Early stopping at step {step_num} "
                    f"(degradation > {tolerance} for {patience} steps)"
                )
                break
        else:
            no_degrade_count = 0

        # Update baseline
        baseline_auc = best_mean_auc

    return step_rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sequential_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    direction: str = "forward",
    scoring: str = "roc_auc",
    cv: int = 5,
    min_features: int = 1,
    max_features: int = 20,
    tolerance: float = 0.001,
    patience: int = 3,
    iv_scores: Optional[Dict[str, float]] = None,
    xgb_params: Optional[Dict] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    output_dir: Optional[str] = None,
) -> Tuple[List[str], pd.DataFrame, str]:
    """
    Sequential feature selection (forward or backward) with cross-validated
    AUC, elbow detection (1-SE rule), and performance chart.

    Args:
        X_train:        Training features DataFrame.
        y_train:        Training target Series.
        X_test:         Test features DataFrame (unused in CV but kept for
                        API compatibility and potential future use).
        y_test:         Test target Series (unused in CV but kept for API
                        compatibility).
        features:       Candidate feature names.
        direction:      'forward' or 'backward'.
        scoring:        Scoring metric (currently only 'roc_auc' supported).
        cv:             Number of cross-validation folds.
        min_features:   Minimum number of features to select.
        max_features:   Maximum number of features to select.
        tolerance:      Minimum AUC improvement to avoid early stopping.
        patience:       Number of consecutive non-improving steps before
                        early stopping triggers.
        iv_scores:      IV scores per feature (speedup hint for ordering).
        xgb_params:     XGBoost parameters. Uses sensible defaults if None.
        random_state:   Random seed for reproducibility.
        n_jobs:         Number of parallel jobs (passed into XGBoost params).
        output_dir:     Directory to save the performance chart. Defaults to 'outputs'.

    Returns:
        Tuple of:
            - selected_features: list of feature names at the optimal step
            - step_details_df: DataFrame with per-step metrics
            - chart_path: path to the saved performance chart PNG
    """
    if direction not in ("forward", "backward"):
        raise ValueError(
            f"direction must be 'forward' or 'backward', got '{direction}'"
        )

    # Cap max_features to available candidates
    max_features = min(max_features, len(features))

    # Prepare XGBoost params
    params = _prepare_xgb_params(xgb_params, y_train, random_state)

    # Nested parallelism handling: when outer n_jobs > 1, force XGBoost
    # to single-threaded to avoid CPU oversubscription. When n_jobs == 1,
    # let XGBoost use all cores itself.
    outer_n_jobs = 1
    if n_jobs != 1:
        outer_n_jobs = n_jobs
        params["n_jobs"] = 1
    # else: keep XGBoost n_jobs as-is (default -1)

    logger.info(
        f"SELECTION | Starting {direction} selection with {len(features)} "
        f"candidates, cv={cv}, max_features={max_features}, "
        f"tolerance={tolerance}, patience={patience}, n_jobs={n_jobs}"
    )

    # Run selection
    if direction == "forward":
        step_rows = _forward_selection(
            X_train=X_train,
            y_train=y_train,
            candidates=list(features),
            iv_scores=iv_scores,
            xgb_params=params,
            cv=cv,
            min_features=min_features,
            max_features=max_features,
            tolerance=tolerance,
            patience=patience,
            random_state=random_state,
            n_jobs=outer_n_jobs,
        )
    else:
        step_rows = _backward_selection(
            X_train=X_train,
            y_train=y_train,
            candidates=list(features),
            iv_scores=iv_scores,
            xgb_params=params,
            cv=cv,
            min_features=min_features,
            max_features=max_features,
            tolerance=tolerance,
            patience=patience,
            random_state=random_state,
            n_jobs=outer_n_jobs,
        )

    # Build step details DataFrame
    step_details_df = pd.DataFrame(step_rows)

    if step_details_df.empty:
        logger.warning("SELECTION | No selection steps were recorded.")
        return [], pd.DataFrame(), ""

    # --- Elbow detection (1-SE rule) ---
    mean_aucs = step_details_df["Mean_CV_AUC"].tolist()
    std_aucs = step_details_df["Std_CV_AUC"].tolist()
    optimal_idx = _find_elbow_1se(mean_aucs, std_aucs)
    step_details_df.loc[optimal_idx, "Is_Optimal"] = True

    optimal_k = int(step_details_df.loc[optimal_idx, "N_Features"])
    logger.info(
        f"SELECTION | Optimal feature count: {optimal_k} (1-SE rule)"
    )

    # Determine the selected features at the optimal step
    if direction == "forward":
        # In forward mode, features are accumulated step by step
        selected_features = step_details_df.loc[
            : optimal_idx, "Added_Feature"
        ].tolist()
    else:
        # In backward mode, we started with all features and removed one
        # per step. The optimal step tells us which features remain.
        removed_up_to_optimal = step_details_df.loc[
            : optimal_idx, "Removed_Feature"
        ].tolist()
        selected_features = [
            f for f in features if f not in removed_up_to_optimal
        ]

    logger.info(
        f"SELECTION | Selected {len(selected_features)} features: "
        f"{selected_features}"
    )

    # --- Performance chart ---
    chart_output_dir = Path(output_dir) if output_dir else Path("outputs")
    chart_path = _save_performance_chart(step_details_df, direction, chart_output_dir)

    return selected_features, step_details_df, chart_path
