"""
Tests for src.model_development.hyperparameter_tuner

Covers: tune_hyperparameters with n_trials=3 (quick), CV fallback, stability mode.
"""

import pytest
import numpy as np
import pandas as pd
import xgboost as xgb

from src.model_development.hyperparameter_tuner import tune_hyperparameters


# ===================================================================
# Helper
# ===================================================================

def _make_tuning_data(n=200, seed=42):
    """Build small data for quick tuning tests."""
    rng = np.random.RandomState(seed)
    target = np.zeros(n, dtype=int)
    target[: int(n * 0.20)] = 1
    rng.shuffle(target)

    data = {}
    for i in range(5):
        data[f"feat_{i}"] = target * (0.5 + i * 0.2) + rng.randn(n)

    df = pd.DataFrame(data)
    features = list(df.columns)
    y = pd.Series(target, name="target")

    split = int(n * 0.8)
    X_train = df.iloc[:split].reset_index(drop=True)
    X_test = df.iloc[split:].reset_index(drop=True)
    y_train = y.iloc[:split].reset_index(drop=True)
    y_test = y.iloc[split:].reset_index(drop=True)

    return X_train, y_train, X_test, y_test, features


# ===================================================================
# Tests
# ===================================================================

class TestTuneHyperparameters:
    def test_returns_correct_types(self):
        X_train, y_train, X_test, y_test, features = _make_tuning_data()
        best_params, trial_df, model = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            n_trials=3,
            cv=2,
            n_jobs=1,
        )
        assert isinstance(best_params, dict)
        assert isinstance(trial_df, pd.DataFrame)
        assert isinstance(model, xgb.XGBClassifier)

    def test_cv_fallback_mode(self):
        """Without oot_quarters, uses CV fallback."""
        X_train, y_train, X_test, y_test, features = _make_tuning_data()
        best_params, trial_df, model = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            n_trials=3,
            cv=2,
            oot_quarters=None,
            n_jobs=1,
        )
        assert len(trial_df) == 3
        # CV fallback should have CV_AUC_Mean in trial_df
        assert "CV_AUC_Mean" in trial_df.columns

    def test_stability_mode(self):
        """With oot_quarters, uses stability-aware objective."""
        X_train, y_train, X_test, y_test, features = _make_tuning_data(n=300)
        rng = np.random.RandomState(42)

        # Build a small OOT quarter
        n_oot = 50
        oot_target = np.zeros(n_oot, dtype=int)
        oot_target[:10] = 1
        rng.shuffle(oot_target)
        oot_data = {f: rng.randn(n_oot) for f in features}
        oot_data["target"] = oot_target
        oot_df = pd.DataFrame(oot_data)

        oot_quarters = {"2024Q4": oot_df}

        best_params, trial_df, model = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            n_trials=3,
            oot_quarters=oot_quarters,
            target_column="target",
            n_jobs=1,
        )
        assert len(trial_df) == 3
        # Stability mode should have Score column
        assert "Score" in trial_df.columns

    def test_best_params_has_expected_keys(self):
        X_train, y_train, X_test, y_test, features = _make_tuning_data()
        best_params, _, _ = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            n_trials=3,
            cv=2,
            n_jobs=1,
        )
        assert "max_depth" in best_params
        assert "learning_rate" in best_params

    def test_tuned_model_can_predict(self):
        X_train, y_train, X_test, y_test, features = _make_tuning_data()
        _, _, model = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            n_trials=3,
            cv=2,
            n_jobs=1,
        )
        preds = model.predict_proba(X_test[features])[:, 1]
        assert len(preds) == len(X_test)
        assert all(0.0 <= p <= 1.0 for p in preds)
