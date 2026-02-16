"""
Tests for src.model_development.feature_selector

Covers: sequential_feature_selection (forward), patience-based early stopping,
min_features, max_features constraints.
"""

import pytest
import numpy as np
import pandas as pd

from src.model_development.feature_selector import sequential_feature_selection


# ===================================================================
# Helper
# ===================================================================

def _make_selection_data(n=200, n_features=6, seed=42):
    """Build small data suitable for quick forward selection."""
    rng = np.random.RandomState(seed)
    target = np.zeros(n, dtype=int)
    target[: int(n * 0.20)] = 1
    rng.shuffle(target)
    y = pd.Series(target, name="target")

    data = {}
    for i in range(n_features):
        strength = 0.5 + i * 0.3
        data[f"feat_{i}"] = target * strength + rng.randn(n)

    X = pd.DataFrame(data)
    features = list(X.columns)

    # Split into train/test
    split = int(n * 0.8)
    X_train = X.iloc[:split].reset_index(drop=True)
    X_test = X.iloc[split:].reset_index(drop=True)
    y_train = y.iloc[:split].reset_index(drop=True)
    y_test = y.iloc[split:].reset_index(drop=True)

    return X_train, y_train, X_test, y_test, features


# ===================================================================
# Tests
# ===================================================================

class TestForwardSelection:
    def test_returns_features_and_df(self, tmp_path):
        X_train, y_train, X_test, y_test, features = _make_selection_data()
        selected, step_df, chart_path = sequential_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            direction="forward",
            cv=2,
            max_features=3,
            min_features=1,
            patience=2,
            tolerance=0.001,
            n_jobs=1,
            output_dir=str(tmp_path),
        )
        assert isinstance(selected, list)
        assert len(selected) >= 1
        assert isinstance(step_df, pd.DataFrame)
        assert len(step_df) >= 1

    def test_max_features_constraint(self, tmp_path):
        X_train, y_train, X_test, y_test, features = _make_selection_data()
        selected, step_df, _ = sequential_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            direction="forward",
            cv=2,
            max_features=2,
            min_features=1,
            patience=10,
            tolerance=0.0,
            n_jobs=1,
            output_dir=str(tmp_path),
        )
        # Should not exceed max_features
        assert len(selected) <= 2

    def test_min_features_respected(self, tmp_path):
        X_train, y_train, X_test, y_test, features = _make_selection_data()
        selected, step_df, _ = sequential_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            direction="forward",
            cv=2,
            max_features=5,
            min_features=2,
            patience=1,
            tolerance=0.5,  # Very high tolerance so early stopping triggers early
            n_jobs=1,
            output_dir=str(tmp_path),
        )
        # Even with early stopping, should have at least min_features steps recorded
        # (patience kicks in only after min_features reached)
        assert len(step_df) >= 2

    def test_patience_early_stopping(self, tmp_path):
        """With very high tolerance, early stopping should trigger quickly."""
        X_train, y_train, X_test, y_test, features = _make_selection_data(n_features=6)
        selected, step_df, _ = sequential_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            direction="forward",
            cv=2,
            max_features=6,
            min_features=1,
            patience=1,
            tolerance=0.5,  # Extremely high => first step OK, then "no improvement"
            n_jobs=1,
            output_dir=str(tmp_path),
        )
        # Should stop well before max_features
        assert len(step_df) < 6

    def test_step_df_has_expected_columns(self, tmp_path):
        X_train, y_train, X_test, y_test, features = _make_selection_data()
        _, step_df, _ = sequential_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            direction="forward",
            cv=2,
            max_features=2,
            min_features=1,
            patience=3,
            n_jobs=1,
            output_dir=str(tmp_path),
        )
        assert "Mean_CV_AUC" in step_df.columns
        assert "Std_CV_AUC" in step_df.columns
        assert "Added_Feature" in step_df.columns
        assert "Is_Optimal" in step_df.columns

    def test_chart_path_saved(self, tmp_path):
        X_train, y_train, X_test, y_test, features = _make_selection_data()
        _, _, chart_path = sequential_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            direction="forward",
            cv=2,
            max_features=2,
            min_features=1,
            patience=3,
            n_jobs=1,
            output_dir=str(tmp_path),
        )
        assert chart_path
        import os
        assert os.path.exists(chart_path)

    def test_invalid_direction_raises(self, tmp_path):
        X_train, y_train, X_test, y_test, features = _make_selection_data()
        with pytest.raises(ValueError, match="direction must be"):
            sequential_feature_selection(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                features=features,
                direction="sideways",
                cv=2,
                max_features=2,
                n_jobs=1,
                output_dir=str(tmp_path),
            )

    def test_iv_scores_hint(self, tmp_path):
        """Forward selection should still work when iv_scores are provided."""
        X_train, y_train, X_test, y_test, features = _make_selection_data()
        iv_scores = {f: 0.1 * i for i, f in enumerate(features)}
        selected, step_df, _ = sequential_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            direction="forward",
            cv=2,
            max_features=2,
            min_features=1,
            patience=3,
            iv_scores=iv_scores,
            n_jobs=1,
            output_dir=str(tmp_path),
        )
        assert len(selected) >= 1

    def test_one_se_optimal_marked(self, tmp_path):
        """The step_df should have exactly one row with Is_Optimal=True."""
        X_train, y_train, X_test, y_test, features = _make_selection_data()
        _, step_df, _ = sequential_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features=features,
            direction="forward",
            cv=2,
            max_features=3,
            min_features=1,
            patience=3,
            n_jobs=1,
            output_dir=str(tmp_path),
        )
        assert step_df["Is_Optimal"].sum() == 1
