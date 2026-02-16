"""
Tests for src.model_development.eliminators

Covers: ConstantEliminator, MissingEliminator, IVEliminator,
PSIEliminator, CorrelationEliminator, VIFEliminator,
TemporalPerformanceEliminator.
"""

import pytest
import numpy as np
import pandas as pd

from src.model_development.eliminators import (
    EliminationResult,
    ConstantEliminator,
    MissingEliminator,
    IVEliminator,
    PSIEliminator,
    QuarterlyPSICheck,
    CorrelationEliminator,
    VIFEliminator,
    TemporalPerformanceEliminator,
)


# ===================================================================
# Helper to build synthetic data
# ===================================================================

def _make_data(n=200, seed=42):
    """Build synthetic X_train, y_train with various feature types."""
    rng = np.random.RandomState(seed)
    target = np.zeros(n, dtype=int)
    target[: int(n * 0.20)] = 1
    rng.shuffle(target)
    y_train = pd.Series(target, name="target")

    data = {}
    # Constant feature
    data["const_feat"] = np.ones(n)
    # Near-constant (1 unique value besides NaN)
    data["near_const"] = np.full(n, 5.0)
    # Good feature with 2+ unique values
    data["good_feat"] = rng.randn(n)
    # High missing feature
    vals = rng.randn(n)
    vals[rng.rand(n) < 0.80] = np.nan
    data["high_missing"] = vals
    # Low missing feature
    vals2 = rng.randn(n)
    vals2[rng.rand(n) < 0.05] = np.nan
    data["low_missing"] = vals2
    # Feature correlated with target (medium IV)
    data["medium_iv"] = target * 1.5 + rng.randn(n) * 0.8
    # Feature with no correlation (low IV / noise)
    data["noise_feat"] = rng.randn(n)
    # High-IV feature (very strongly correlated)
    data["high_iv"] = target * 3.0 + rng.randn(n) * 0.3
    # Correlated pair
    base = rng.randn(n)
    data["corr_a"] = base + rng.randn(n) * 0.01
    data["corr_b"] = base + rng.randn(n) * 0.01
    # Multicollinear features for VIF
    data["multi_a"] = rng.randn(n)
    data["multi_b"] = data["multi_a"] * 1.01 + rng.randn(n) * 0.01
    data["multi_c"] = rng.randn(n)

    X_train = pd.DataFrame(data)
    return X_train, y_train


# ===================================================================
# EliminationResult
# ===================================================================

class TestEliminationResult:
    def test_n_kept(self, elimination_result_fixture):
        assert elimination_result_fixture.n_kept == 5

    def test_n_eliminated(self, elimination_result_fixture):
        assert elimination_result_fixture.n_eliminated == 5

    def test_step_name(self, elimination_result_fixture):
        assert elimination_result_fixture.step_name == "03_IV_Analysis"


# ===================================================================
# ConstantEliminator
# ===================================================================

class TestConstantEliminator:
    def test_eliminates_single_unique(self):
        X, y = _make_data()
        elim = ConstantEliminator(min_unique=2)
        result = elim.eliminate(X, y, ["const_feat", "good_feat"])
        assert "const_feat" in result.eliminated_features
        assert "good_feat" in result.kept_features

    def test_keeps_features_with_two_plus_unique(self):
        X, y = _make_data()
        elim = ConstantEliminator(min_unique=2)
        result = elim.eliminate(X, y, ["good_feat", "medium_iv"])
        assert "good_feat" in result.kept_features
        assert "medium_iv" in result.kept_features
        assert result.n_eliminated == 0

    def test_near_zero_variance_elimination(self):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "tiny_var": np.full(n, 1.0) + rng.randn(n) * 1e-10,
            "normal_var": rng.randn(n),
        })
        y = pd.Series(rng.choice([0, 1], n))
        elim = ConstantEliminator(min_unique=2, min_variance=1e-5)
        result = elim.eliminate(X, y, ["tiny_var", "normal_var"])
        assert "tiny_var" in result.eliminated_features
        assert "normal_var" in result.kept_features

    def test_details_df_has_expected_columns(self):
        X, y = _make_data()
        elim = ConstantEliminator()
        result = elim.eliminate(X, y, ["const_feat", "good_feat"])
        assert "Feature" in result.details_df.columns
        assert "Unique_Count" in result.details_df.columns
        assert "Status" in result.details_df.columns

    def test_step_name(self):
        assert ConstantEliminator.step_name == "01_Constant"


# ===================================================================
# MissingEliminator
# ===================================================================

class TestMissingEliminator:
    def test_eliminates_high_missing(self):
        X, y = _make_data()
        elim = MissingEliminator(max_missing_rate=0.70)
        result = elim.eliminate(X, y, ["high_missing", "low_missing"])
        assert "high_missing" in result.eliminated_features

    def test_keeps_low_missing(self):
        X, y = _make_data()
        elim = MissingEliminator(max_missing_rate=0.70)
        result = elim.eliminate(X, y, ["high_missing", "low_missing"])
        assert "low_missing" in result.kept_features

    def test_no_missing_all_kept(self):
        X, y = _make_data()
        elim = MissingEliminator(max_missing_rate=0.70)
        result = elim.eliminate(X, y, ["good_feat", "medium_iv"])
        assert result.n_eliminated == 0

    def test_details_df_has_missing_rate(self):
        X, y = _make_data()
        elim = MissingEliminator()
        result = elim.eliminate(X, y, ["high_missing", "low_missing"])
        assert "Missing_Rate" in result.details_df.columns
        assert "Missing_Count" in result.details_df.columns

    def test_step_name(self):
        assert MissingEliminator.step_name == "02_Missing"

    def test_threshold_boundary(self):
        """Feature with exactly the threshold missing rate should be kept (>= vs >)."""
        rng = np.random.RandomState(42)
        n = 100
        vals = rng.randn(n)
        # Set exactly 70 to NaN
        vals[:70] = np.nan
        X = pd.DataFrame({"feat": vals})
        y = pd.Series(rng.choice([0, 1], n))
        elim = MissingEliminator(max_missing_rate=0.70)
        result = elim.eliminate(X, y, ["feat"])
        # 70/100 = 0.70 -- the check is > 0.70, so it should be kept
        assert "feat" in result.kept_features


# ===================================================================
# IVEliminator
# ===================================================================

class TestIVEliminator:
    def test_eliminates_low_iv(self):
        """A truly random feature should be eliminated with a high min_iv threshold."""
        rng = np.random.RandomState(42)
        n = 500
        target = np.zeros(n, dtype=int)
        target[:100] = 1
        rng.shuffle(target)
        y = pd.Series(target)
        # Pure noise -- no relationship to target
        X = pd.DataFrame({"pure_noise": rng.randn(n)})
        # Use a high min_iv threshold to ensure elimination
        elim = IVEliminator(min_iv=0.30, max_iv=5.0, n_bins=10)
        result = elim.eliminate(X, y, ["pure_noise"])
        assert "pure_noise" in result.eliminated_features

    def test_keeps_medium_iv(self):
        """A feature strongly correlated with target should be kept with wide IV bounds."""
        X, y = _make_data(n=500)
        # medium_iv has high actual IV; use a very wide range to keep it
        elim = IVEliminator(min_iv=0.02, max_iv=50.0, n_bins=10)
        result = elim.eliminate(X, y, ["medium_iv"])
        assert "medium_iv" in result.kept_features

    def test_eliminates_suspicious_iv(self):
        """Very high IV (>0.50) considered suspicious."""
        X, y = _make_data(n=500)
        elim = IVEliminator(min_iv=0.02, max_iv=0.50, n_bins=10)
        result = elim.eliminate(X, y, ["high_iv"])
        # high_iv is perfectly correlated, should have IV > 0.50
        assert "high_iv" in result.eliminated_features

    def test_details_df_has_iv_columns(self):
        X, y = _make_data(n=500)
        elim = IVEliminator(min_iv=0.02, max_iv=0.50, n_bins=10)
        result = elim.eliminate(X, y, ["medium_iv", "noise_feat"])
        assert "IV_Score" in result.details_df.columns
        assert "IV_Category" in result.details_df.columns

    def test_step_name(self):
        assert IVEliminator.step_name == "03_IV_Analysis"

    def test_with_n_jobs(self):
        X, y = _make_data(n=500)
        elim = IVEliminator(min_iv=0.02, max_iv=5.0, n_bins=10, n_jobs=1)
        result = elim.eliminate(X, y, ["medium_iv", "noise_feat"])
        assert isinstance(result, EliminationResult)

    def test_univariate_metrics_present(self):
        X, y = _make_data(n=500)
        elim = IVEliminator(min_iv=0.02, max_iv=5.0, n_bins=10)
        result = elim.eliminate(X, y, ["medium_iv"])
        assert "Univariate_AUC" in result.details_df.columns


# ===================================================================
# PSIEliminator
# ===================================================================

class TestPSIEliminator:
    def test_with_quarterly_check(self):
        """Quarterly PSI check runs and returns a valid EliminationResult."""
        rng = np.random.RandomState(42)
        # Use larger n with dates spanning a single year to ensure
        # each quarter has roughly equal distribution of the same feature
        n = 1000
        dates = pd.date_range("2023-01-01", periods=n, freq="6h")
        target = rng.choice([0, 1], n, p=[0.8, 0.2])
        y = pd.Series(target)
        # Use a uniform random feature -- consistent distribution across quarters
        X = pd.DataFrame({"stable_feat": rng.randn(n)})
        train_dates = pd.Series(dates)

        elim = PSIEliminator(
            critical_threshold=0.25,
            checks=[QuarterlyPSICheck()],
        )
        result = elim.eliminate(X, y, ["stable_feat"], train_dates=train_dates)
        assert isinstance(result, EliminationResult)
        # With 1000 samples from the same distribution, PSI should be low
        assert "stable_feat" in result.kept_features

    def test_no_dates_skips_psi(self):
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({"feat": rng.randn(n)})
        y = pd.Series(rng.choice([0, 1], n))
        elim = PSIEliminator()
        result = elim.eliminate(X, y, ["feat"], train_dates=None)
        assert "feat" in result.kept_features
        assert result.n_eliminated == 0

    def test_unstable_feature_eliminated(self):
        """Feature with drastically different distribution across quarters."""
        rng = np.random.RandomState(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="1D")
        target = rng.choice([0, 1], n, p=[0.8, 0.2])
        y = pd.Series(target)

        # Create a feature that shifts its distribution by quarter
        quarters = pd.Series(dates).dt.to_period("Q")
        unique_quarters = sorted(quarters.unique())
        feat_vals = np.zeros(n)
        for i, q in enumerate(unique_quarters):
            mask = (quarters == q).values
            feat_vals[mask] = rng.randn(mask.sum()) + i * 10  # massive shift

        X = pd.DataFrame({"shifting_feat": feat_vals})
        train_dates = pd.Series(dates)

        elim = PSIEliminator(
            critical_threshold=0.25,
            checks=[QuarterlyPSICheck()],
        )
        result = elim.eliminate(X, y, ["shifting_feat"], train_dates=train_dates)
        assert "shifting_feat" in result.eliminated_features

    def test_details_df_has_max_psi(self):
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="2D")
        X = pd.DataFrame({"feat": rng.randn(n)})
        y = pd.Series(rng.choice([0, 1], n))

        elim = PSIEliminator(checks=[QuarterlyPSICheck()])
        result = elim.eliminate(X, y, ["feat"], train_dates=pd.Series(dates))
        assert "Max_PSI" in result.details_df.columns

    def test_step_name(self):
        assert PSIEliminator.step_name == "04_PSI_Stability"


# ===================================================================
# CorrelationEliminator
# ===================================================================

class TestCorrelationEliminator:
    def test_eliminates_correlated_features(self):
        X, y = _make_data()
        iv_scores = {"corr_a": 0.20, "corr_b": 0.10}
        elim = CorrelationEliminator(max_correlation=0.90)
        result = elim.eliminate(X, y, ["corr_a", "corr_b"], iv_scores=iv_scores)
        # corr_a has higher IV so corr_b should be eliminated
        assert "corr_a" in result.kept_features
        assert "corr_b" in result.eliminated_features

    def test_iv_ordering_preserved(self):
        """Higher IV feature kept over lower IV feature in correlated pair."""
        X, y = _make_data()
        iv_scores = {"corr_a": 0.05, "corr_b": 0.30}
        elim = CorrelationEliminator(max_correlation=0.90)
        result = elim.eliminate(X, y, ["corr_a", "corr_b"], iv_scores=iv_scores)
        # corr_b has higher IV, so corr_a should be eliminated
        assert "corr_b" in result.kept_features
        assert "corr_a" in result.eliminated_features

    def test_uncorrelated_features_kept(self):
        X, y = _make_data()
        elim = CorrelationEliminator(max_correlation=0.90)
        result = elim.eliminate(X, y, ["good_feat", "medium_iv"])
        assert "good_feat" in result.kept_features
        assert "medium_iv" in result.kept_features

    def test_corr_pairs_df_attribute(self):
        X, y = _make_data()
        iv_scores = {"corr_a": 0.20, "corr_b": 0.10}
        elim = CorrelationEliminator(max_correlation=0.90)
        elim.eliminate(X, y, ["corr_a", "corr_b"], iv_scores=iv_scores)
        assert hasattr(elim, "corr_pairs_df")
        assert isinstance(elim.corr_pairs_df, pd.DataFrame)

    def test_details_df_has_correlation_column(self):
        X, y = _make_data()
        iv_scores = {"corr_a": 0.20, "corr_b": 0.10}
        elim = CorrelationEliminator(max_correlation=0.90)
        result = elim.eliminate(X, y, ["corr_a", "corr_b"], iv_scores=iv_scores)
        if len(result.details_df) > 0:
            assert "Correlation" in result.details_df.columns

    def test_step_name(self):
        assert CorrelationEliminator.step_name == "05_Correlation"

    def test_no_iv_scores_uses_zero(self):
        X, y = _make_data()
        elim = CorrelationEliminator(max_correlation=0.90)
        # Should work without iv_scores (defaults to 0 for all)
        result = elim.eliminate(X, y, ["corr_a", "corr_b"])
        # One should be eliminated
        assert result.n_eliminated == 1


# ===================================================================
# VIFEliminator
# ===================================================================

class TestVIFEliminator:
    def test_eliminates_multicollinear(self):
        X, y = _make_data()
        elim = VIFEliminator(threshold=5.0, iv_aware=False)
        result = elim.eliminate(
            X, y, ["multi_a", "multi_b", "multi_c"]
        )
        # multi_a and multi_b are nearly identical -> one should be removed
        assert result.n_eliminated >= 1

    def test_iv_aware_mode(self):
        X, y = _make_data()
        iv_scores = {"multi_a": 0.30, "multi_b": 0.05, "multi_c": 0.10}
        elim = VIFEliminator(threshold=5.0, iv_aware=True)
        result = elim.eliminate(
            X, y, ["multi_a", "multi_b", "multi_c"],
            iv_scores=iv_scores,
        )
        # multi_b has the lowest IV among high-VIF features, so it should be eliminated first
        if result.n_eliminated > 0:
            assert "multi_b" in result.eliminated_features

    def test_few_features_skips(self):
        X, y = _make_data()
        elim = VIFEliminator(threshold=5.0)
        result = elim.eliminate(X, y, ["good_feat"])
        # Only 1 feature, VIF check is skipped
        assert result.n_eliminated == 0
        assert "good_feat" in result.kept_features

    def test_two_features_skips(self):
        X, y = _make_data()
        elim = VIFEliminator(threshold=5.0)
        result = elim.eliminate(X, y, ["good_feat", "medium_iv"])
        assert result.n_eliminated == 0

    def test_details_df_columns(self):
        X, y = _make_data()
        elim = VIFEliminator(threshold=5.0)
        result = elim.eliminate(X, y, ["multi_a", "multi_b", "multi_c"])
        assert "VIF_Initial" in result.details_df.columns
        assert "Status" in result.details_df.columns

    def test_step_name(self):
        assert VIFEliminator.step_name == "09_VIF"


# ===================================================================
# TemporalPerformanceEliminator
# ===================================================================

class TestTemporalPerformanceEliminator:
    def test_basic_flow(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.RandomState(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="1D")
        target = np.zeros(n, dtype=int)
        target[:60] = 1
        rng.shuffle(target)
        y = pd.Series(target)

        feat = target * 1.5 + rng.randn(n) * 0.8
        X = pd.DataFrame({"feat_good": feat, "feat_noise": rng.randn(n)})
        train_dates = pd.Series(dates)

        elim = TemporalPerformanceEliminator(
            min_trend_slope=-0.02,
            max_auc_degradation=0.05,
        )
        result = elim.eliminate(
            X, y, ["feat_good", "feat_noise"],
            train_dates=train_dates,
        )
        assert isinstance(result, EliminationResult)

    def test_no_dates_skips(self):
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({"feat": rng.randn(n)})
        y = pd.Series(rng.choice([0, 1], n))
        elim = TemporalPerformanceEliminator()
        result = elim.eliminate(X, y, ["feat"], train_dates=None)
        assert "feat" in result.kept_features

    def test_step_name(self):
        assert TemporalPerformanceEliminator.step_name == "07_Temporal_Filter"
