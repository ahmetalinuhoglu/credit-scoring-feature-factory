"""
Forward Feature Selector Component

Stepwise forward selection using XGBoost. Features are tried in IV-descending
order. A feature is added if it improves test AUC by at least auc_threshold.
"""

from typing import Any, Dict, List, Optional
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from src.pipeline.base import BaseComponent, StepResult
from src.config.schema import SelectionConfig, ModelConfig

logger = logging.getLogger(__name__)

STEP_NAME = "06_selection"


class ForwardFeatureSelector(BaseComponent):
    """Forward stepwise feature selection using XGBoost.

    Features are tried in IV-descending order. A feature is added to the
    selected set if it improves test AUC by at least auc_threshold.

    Args:
        config: SelectionConfig with method, auc_threshold, max_features.
        model_config: ModelConfig with XGBoost params.
        seed: Global random seed (used as XGBoost random_state).
    """

    step_name = STEP_NAME
    step_order = 6

    def __init__(
        self,
        config: SelectionConfig,
        model_config: ModelConfig,
        seed: int = 42,
    ):
        self.auc_threshold = config.auc_threshold
        self.max_features = config.max_features
        self.xgb_params = dict(model_config.params)
        self.seed = seed
        self.kept_features_: List[str] = []
        self.eliminated_features_: List[str] = []
        self.model_: Optional[xgb.XGBClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> StepResult:
        """Run forward stepwise selection on candidate features.

        Args:
            X: Training feature DataFrame.
            y: Training target Series.
            **kwargs:
                X_test: pd.DataFrame of test features (required).
                y_test: pd.Series of test target (required).
                iv_scores: Dict[str, float] for ordering candidates.

        Returns:
            StepResult with selection steps, per-step AUC, features added/skipped.

        Raises:
            ValueError: If X_test or y_test are not provided.
        """
        t0 = time.time()
        features = list(X.columns)
        X_test: Optional[pd.DataFrame] = kwargs.get("X_test")
        y_test: Optional[pd.Series] = kwargs.get("y_test")
        iv_scores: Dict[str, float] = kwargs.get("iv_scores", {})

        if X_test is None or y_test is None:
            raise ValueError(
                f"{STEP_NAME} requires X_test and y_test in kwargs"
            )

        # Sort candidates by IV descending
        sorted_candidates = sorted(
            features,
            key=lambda f: iv_scores.get(f, 0) or 0,
            reverse=True,
        )

        # Apply max_features cap to candidates
        if self.max_features is not None and self.max_features < len(sorted_candidates):
            sorted_candidates = sorted_candidates[:self.max_features]
            logger.info(
                f"{STEP_NAME} | Capped candidates to top {self.max_features} by IV"
            )

        # Prepare XGBoost params
        params = self._prepare_params(y)

        selected: List[str] = []
        baseline_auc = 0.5
        step_rows = []

        logger.info(
            f"{STEP_NAME} | Starting forward selection with "
            f"{len(sorted_candidates)} candidates, "
            f"AUC threshold={self.auc_threshold}"
        )

        for i, candidate in enumerate(sorted_candidates):
            trial_features = selected + [candidate]

            trial_auc = self._train_and_evaluate(
                X[trial_features], y,
                X_test[trial_features], y_test,
                params,
            )

            improvement = trial_auc - baseline_auc

            if improvement >= self.auc_threshold:
                selected.append(candidate)
                baseline_auc = trial_auc
                status = "Added"
                logger.info(
                    f"{STEP_NAME} | Step {len(selected):3d}: ADDED {candidate}, "
                    f"AUC={trial_auc:.6f}, +{improvement:.6f}"
                )
            else:
                status = "Skipped"
                logger.info(
                    f"{STEP_NAME} | Step {i + 1:3d}: SKIP  {candidate}, "
                    f"AUC={trial_auc:.6f}, +{improvement:.6f} < {self.auc_threshold}"
                )

            step_rows.append({
                "Step": i + 1,
                "Feature": candidate,
                "Feature_IV": round(iv_scores.get(candidate, 0) or 0, 4),
                "AUC_After": round(trial_auc, 6),
                "AUC_Improvement": round(improvement, 6),
                "Status": status,
                "Cumulative_Features": len(selected),
            })

        self.kept_features_ = selected
        self.eliminated_features_ = [f for f in features if f not in selected]

        # Train final model on selected features
        if selected:
            logger.info(
                f"{STEP_NAME} | Training final model on {len(selected)} features"
            )
            self.model_ = self._train_final_model(
                X[selected], y, X_test[selected], y_test, params
            )
            final_auc = float(roc_auc_score(
                y_test, self.model_.predict_proba(X_test[selected])[:, 1]
            ))
            logger.info(f"{STEP_NAME} | Final model AUC: {final_auc:.6f}")
        else:
            logger.warning(f"{STEP_NAME} | No features selected")
            self.model_ = None

        results_df = pd.DataFrame(step_rows)
        duration = time.time() - t0

        logger.info(
            f"{STEP_NAME} | Selected {len(selected)} features from "
            f"{len(features)} candidates in {duration:.1f}s"
        )

        return StepResult(
            step_name=self.step_name,
            input_features=features,
            output_features=selected,
            eliminated_features=self.eliminated_features_,
            results_df=results_df,
            metadata={
                "auc_threshold": self.auc_threshold,
                "max_features": self.max_features,
                "final_auc": baseline_auc,
                "n_candidates_tried": len(sorted_candidates),
            },
            duration_seconds=round(duration, 1),
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only the chosen features from the DataFrame.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with only selected features (columns present in X).
        """
        cols = [c for c in self.kept_features_ if c in X.columns]
        return X[cols]

    def _prepare_params(self, y_train: pd.Series) -> Dict[str, Any]:
        """Prepare XGBoost parameters with auto scale_pos_weight and seed.

        Args:
            y_train: Training target for computing class balance.

        Returns:
            Dict of XGBoost constructor parameters.
        """
        params = self.xgb_params.copy()
        params["random_state"] = self.seed
        params["verbosity"] = 0
        params["n_jobs"] = params.get("n_jobs", -1)

        # Auto-balance classes
        if params.get("scale_pos_weight") == "auto":
            neg_count = int((y_train == 0).sum())
            pos_count = int((y_train == 1).sum())
            params["scale_pos_weight"] = neg_count / pos_count if pos_count > 0 else 1.0

        # XGBoost >= 2.0: early_stopping_rounds must be in constructor
        if "early_stopping_rounds" not in params:
            params["early_stopping_rounds"] = 30

        return params

    @staticmethod
    def _train_and_evaluate(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Dict[str, Any],
    ) -> float:
        """Train XGBoost and return test AUC.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.
            params: XGBoost constructor parameters.

        Returns:
            Test AUC score.
        """
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, y_prob))

    @staticmethod
    def _train_final_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Dict[str, Any],
    ) -> xgb.XGBClassifier:
        """Train and return the final XGBoost model.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features for early stopping.
            y_test: Test target for early stopping.
            params: XGBoost constructor parameters.

        Returns:
            Trained XGBClassifier.
        """
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        return model
