"""
Model Evaluator Component

Evaluates the final model across Train, Test, and each OOT quarter.
Produces performance metrics, decile lift tables, feature importance,
score distributions, and score PSI for model stability assessment.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.pipeline.base import BaseComponent, StepResult
from src.config.schema import EvaluationConfig

logger = logging.getLogger(__name__)

STEP_NAME = "07_evaluation"


class ModelEvaluator(BaseComponent):
    """Evaluate the final model on train, test, and OOT quarters.

    Computes AUC, Gini, KS, Precision@k, Lift@k, decile lift tables,
    feature importance, score distributions, and score PSI.

    This component does not eliminate features -- transform() returns X unchanged.

    Args:
        config: EvaluationConfig with metrics, precision_at_k, n_deciles,
            calculate_score_psi.
    """

    step_name = STEP_NAME
    step_order = 7

    def __init__(self, config: EvaluationConfig):
        self.metrics = config.metrics
        self.precision_at_k = config.precision_at_k
        self.n_deciles = config.n_deciles
        self.calculate_score_psi = config.calculate_score_psi
        self.kept_features_: List[str] = []
        self.eliminated_features_: List[str] = []
        self.performance_df_: Optional[pd.DataFrame] = None
        self.lift_tables_: Dict[str, pd.DataFrame] = {}
        self.importance_df_: Optional[pd.DataFrame] = None
        self.score_distributions_: Dict[str, pd.DataFrame] = {}
        self.score_psi_df_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> StepResult:
        """Evaluate the final model across all data periods.

        Args:
            X: Training feature DataFrame (used for train-set evaluation).
            y: Training target Series.
            **kwargs:
                model: Trained XGBClassifier (required).
                X_test: pd.DataFrame of test features (required).
                y_test: pd.Series of test target (required).
                oot_data: Dict[str, pd.DataFrame] with OOT quarter DataFrames
                    containing both features and target column.
                selected_features: List[str] of feature names used in model.
                target_column: str, name of target column in OOT DataFrames.
                    Defaults to 'target'.

        Returns:
            StepResult with performance_df, lift_tables, importance_df,
            score distributions, and score PSI.

        Raises:
            ValueError: If required kwargs are missing.
        """
        t0 = time.time()
        features = list(X.columns)
        self.kept_features_ = list(features)
        self.eliminated_features_ = []

        model = kwargs.get("model")
        X_test = kwargs.get("X_test")
        y_test = kwargs.get("y_test")
        oot_data: Dict[str, pd.DataFrame] = kwargs.get("oot_data", {})
        selected_features: List[str] = kwargs.get("selected_features", features)
        target_column: str = kwargs.get("target_column", "target")

        if model is None:
            raise ValueError(f"{STEP_NAME} requires 'model' in kwargs")
        if X_test is None or y_test is None:
            raise ValueError(f"{STEP_NAME} requires 'X_test' and 'y_test' in kwargs")

        # Build evaluation periods
        periods: List[Tuple[str, np.ndarray, np.ndarray]] = []
        train_proba = model.predict_proba(X[selected_features])[:, 1]
        periods.append(("Train", y.values, train_proba))

        test_proba = model.predict_proba(X_test[selected_features])[:, 1]
        periods.append(("Test", y_test.values, test_proba))

        for label in sorted(oot_data.keys()):
            oot_df = oot_data[label]
            if len(oot_df) == 0:
                continue
            oot_y = oot_df[target_column].values
            if len(np.unique(oot_y)) < 2:
                logger.warning(
                    f"{STEP_NAME} | OOT_{label}: only one class present, skipping"
                )
                continue
            oot_proba = model.predict_proba(oot_df[selected_features])[:, 1]
            periods.append((f"OOT_{label}", oot_y, oot_proba))

        # Evaluate each period
        perf_rows = []
        all_score_arrays: Dict[str, np.ndarray] = {}

        for period_name, y_true, y_prob in periods:
            metrics = self._calculate_metrics(y_true, y_prob)
            lift_table = self._create_lift_table(y_true, y_prob, self.n_deciles)
            self.lift_tables_[period_name] = lift_table

            row: Dict[str, Any] = {
                "Period": period_name,
                "N_Samples": len(y_true),
                "N_Bads": int(y_true.sum()),
                "Bad_Rate": round(float(y_true.mean()), 4),
                "AUC": metrics["auc"],
                "Gini": metrics["gini"],
                "KS": metrics["ks"],
            }

            # Precision and Lift at configured k values
            for k in self.precision_at_k:
                prec, lift = self._precision_lift_at_k(y_true, y_prob, k=k / 100.0)
                row[f"Precision_at_{k}pct"] = round(prec, 4) if prec is not None else None
                row[f"Lift_at_{k}pct"] = round(lift, 2) if lift is not None else None

            perf_rows.append(row)
            all_score_arrays[period_name] = y_prob

            # Score distribution
            self.score_distributions_[period_name] = self._score_distribution(
                y_prob, period_name
            )

            logger.info(
                f"{STEP_NAME} | {period_name}: AUC={metrics['auc']:.4f}, "
                f"Gini={metrics['gini']:.4f}, KS={metrics['ks']:.4f}"
            )

        self.performance_df_ = pd.DataFrame(perf_rows)

        # Feature importance
        self.importance_df_ = self._feature_importance(model, selected_features)

        # Score PSI between train and each other period
        if self.calculate_score_psi and "Train" in all_score_arrays:
            self.score_psi_df_ = self._calculate_score_psi(all_score_arrays)

        # Build results_df as the performance summary
        results_df = self.performance_df_.copy()
        duration = time.time() - t0

        logger.info(f"{STEP_NAME} | Evaluation completed in {duration:.1f}s")

        return StepResult(
            step_name=self.step_name,
            input_features=features,
            output_features=features,
            eliminated_features=[],
            results_df=results_df,
            metadata={
                "n_periods": len(periods),
                "n_oot_quarters": len(oot_data),
                "n_selected_features": len(selected_features),
                "score_psi_calculated": self.calculate_score_psi,
            },
            duration_seconds=round(duration, 1),
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X unchanged (evaluation does not eliminate features).

        Args:
            X: DataFrame to transform.

        Returns:
            The same DataFrame, unchanged.
        """
        return X

    @staticmethod
    def _calculate_metrics(
        y_true: np.ndarray, y_score: np.ndarray
    ) -> Dict[str, float]:
        """Calculate AUC, Gini, and KS.

        Args:
            y_true: True binary labels.
            y_score: Predicted probabilities.

        Returns:
            Dict with 'auc', 'gini', 'ks' keys.
        """
        auc = roc_auc_score(y_true, y_score)
        gini = 2 * auc - 1

        # KS statistic
        df = pd.DataFrame({"score": y_score, "target": y_true})
        df = df.sort_values("score", ascending=False)
        total_bads = df["target"].sum()
        total_goods = len(df) - total_bads
        if total_bads == 0 or total_goods == 0:
            ks = 0.0
        else:
            df["cum_bads"] = df["target"].cumsum() / total_bads
            df["cum_goods"] = (1 - df["target"]).cumsum() / total_goods
            ks = float(abs(df["cum_bads"] - df["cum_goods"]).max())

        return {
            "auc": round(auc, 4),
            "gini": round(gini, 4),
            "ks": round(ks, 4),
        }

    @staticmethod
    def _create_lift_table(
        y_true: np.ndarray, y_score: np.ndarray, n_deciles: int = 10
    ) -> pd.DataFrame:
        """Create decile-based lift table.

        Args:
            y_true: True binary labels.
            y_score: Predicted probabilities.
            n_deciles: Number of decile bins.

        Returns:
            DataFrame with decile-level statistics.
        """
        df = pd.DataFrame({"score": y_score, "target": y_true})

        try:
            df["decile"] = pd.qcut(
                df["score"],
                q=n_deciles,
                labels=list(range(n_deciles, 0, -1)),
                duplicates="drop",
            )
        except ValueError:
            df["decile"] = pd.qcut(
                df["score"].rank(method="first"),
                q=n_deciles,
                labels=list(range(n_deciles, 0, -1)),
                duplicates="drop",
            )

        lift = df.groupby("decile", observed=False).agg({
            "score": ["min", "max", "mean"],
            "target": ["count", "sum", "mean"],
        }).round(4)

        lift.columns = [
            "Score_Min", "Score_Max", "Score_Mean",
            "Count", "Bads", "Bad_Rate",
        ]

        overall_bad_rate = df["target"].mean()
        lift["Lift"] = (
            lift["Bad_Rate"] / overall_bad_rate if overall_bad_rate > 0 else 0
        )

        lift = lift.sort_index(ascending=False)
        lift["Cum_Count"] = lift["Count"].cumsum()
        lift["Cum_Bads"] = lift["Bads"].cumsum()
        lift["Cum_Bad_Rate"] = lift["Cum_Bads"] / lift["Cum_Count"]
        lift["Cum_Lift"] = (
            lift["Cum_Bad_Rate"] / overall_bad_rate if overall_bad_rate > 0 else 0
        )

        total_bads = df["target"].sum()
        lift["Capture_Rate"] = (
            lift["Cum_Bads"] / total_bads if total_bads > 0 else 0
        )

        return lift.reset_index()

    @staticmethod
    def _precision_lift_at_k(
        y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate precision and lift at top k%.

        Args:
            y_true: True binary labels.
            y_score: Predicted probabilities.
            k: Fraction of top-scored samples to consider.

        Returns:
            Tuple of (precision, lift). Both None on failure.
        """
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

    @staticmethod
    def _feature_importance(model: Any, features: List[str]) -> pd.DataFrame:
        """Extract gain, weight, and cover importance from XGBoost model.

        Args:
            model: Trained XGBClassifier.
            features: List of feature names.

        Returns:
            DataFrame with Feature, Gain, Weight, Cover, and Rank columns.
        """
        try:
            booster = model.get_booster()
            gain = booster.get_score(importance_type="gain")
            weight = booster.get_score(importance_type="weight")
            cover = booster.get_score(importance_type="cover")

            rows = []
            for feat in features:
                # XGBoost uses feature indices (f0, f1, ...) as keys
                # but sklearn wrapper maps to feature names
                rows.append({
                    "Feature": feat,
                    "Gain": round(gain.get(feat, 0), 4),
                    "Weight": int(weight.get(feat, 0)),
                    "Cover": round(cover.get(feat, 0), 4),
                })

            df = pd.DataFrame(rows).sort_values("Gain", ascending=False)
            df["Rank"] = range(1, len(df) + 1)

            # Normalize gain for cumulative importance
            total_gain = df["Gain"].sum()
            if total_gain > 0:
                df["Gain_Normalized"] = round(df["Gain"] / total_gain, 4)
                df["Cumulative_Importance"] = df["Gain_Normalized"].cumsum()
            else:
                df["Gain_Normalized"] = 0.0
                df["Cumulative_Importance"] = 0.0

            return df.reset_index(drop=True)
        except Exception:
            # Fallback to sklearn feature_importances_
            importances = model.feature_importances_
            df = pd.DataFrame({
                "Feature": features,
                "Gain": [round(float(v), 4) for v in importances],
            }).sort_values("Gain", ascending=False)
            df["Rank"] = range(1, len(df) + 1)
            df["Cumulative_Importance"] = (
                df["Gain"] / df["Gain"].sum()
            ).cumsum() if df["Gain"].sum() > 0 else 0.0
            return df.reset_index(drop=True)

    @staticmethod
    def _score_distribution(
        y_prob: np.ndarray, period_name: str, n_bins: int = 50
    ) -> pd.DataFrame:
        """Compute score distribution histogram data.

        Args:
            y_prob: Predicted probabilities.
            period_name: Name of the evaluation period.
            n_bins: Number of histogram bins.

        Returns:
            DataFrame with bin edges, counts, and density.
        """
        counts, bin_edges = np.histogram(y_prob, bins=n_bins)
        total = len(y_prob)
        rows = []
        for i in range(len(counts)):
            rows.append({
                "Period": period_name,
                "Bin_Lower": round(float(bin_edges[i]), 4),
                "Bin_Upper": round(float(bin_edges[i + 1]), 4),
                "Bin_Center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 4),
                "Count": int(counts[i]),
                "Density": round(counts[i] / total, 4) if total > 0 else 0,
            })
        return pd.DataFrame(rows)

    def _calculate_score_psi(
        self, score_arrays: Dict[str, np.ndarray], n_bins: int = 10
    ) -> pd.DataFrame:
        """Calculate PSI of predicted scores between Train and each other period.

        Args:
            score_arrays: Dict mapping period name to array of predicted probabilities.
            n_bins: Number of bins for PSI calculation.

        Returns:
            DataFrame with Period and Score_PSI columns.
        """
        train_scores = score_arrays.get("Train")
        if train_scores is None or len(train_scores) < 10:
            return pd.DataFrame()

        rows = []
        for period_name, scores in score_arrays.items():
            if period_name == "Train":
                continue
            psi = self._psi_between(train_scores, scores, n_bins)
            rows.append({
                "Period": period_name,
                "Score_PSI": round(psi, 4) if psi is not None else None,
            })
            if psi is not None:
                logger.info(f"{STEP_NAME} | Score PSI {period_name}: {psi:.4f}")

        return pd.DataFrame(rows)

    @staticmethod
    def _psi_between(
        expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
    ) -> Optional[float]:
        """Calculate PSI between two score distributions.

        Args:
            expected: Baseline (train) scores.
            actual: Comparison scores.
            n_bins: Number of quantile bins.

        Returns:
            PSI value, or None on failure.
        """
        try:
            if len(expected) < 10 or len(actual) < 10:
                return None

            try:
                _, bins = pd.qcut(
                    expected, q=n_bins, retbins=True, duplicates="drop"
                )
            except ValueError:
                n = min(5, len(np.unique(expected)))
                if n < 2:
                    return None
                _, bins = pd.qcut(expected, q=n, retbins=True, duplicates="drop")

            bins[0] = -np.inf
            bins[-1] = np.inf

            expected_binned = pd.cut(expected, bins=bins)
            actual_binned = pd.cut(actual, bins=bins)

            expected_pct = (
                pd.Series(expected_binned).value_counts(normalize=True).sort_index()
            )
            actual_pct = (
                pd.Series(actual_binned).value_counts(normalize=True).sort_index()
            )

            all_bins = expected_pct.index.union(actual_pct.index)
            expected_pct = expected_pct.reindex(all_bins, fill_value=0.0001)
            actual_pct = actual_pct.reindex(all_bins, fill_value=0.0001)

            epsilon = 1e-4
            expected_pct = expected_pct.clip(lower=epsilon)
            actual_pct = actual_pct.clip(lower=epsilon)

            psi = float(
                ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
            )
            return psi if np.isfinite(psi) else None
        except Exception:
            return None
