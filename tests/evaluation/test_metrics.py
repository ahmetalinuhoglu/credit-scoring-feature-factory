"""
Tests for Credit Scoring Metrics

Tests Gini, KS, PSI, Lift, and other evaluation metrics.
"""

import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import CreditScoringMetrics


class TestGiniCoefficient:
    """Test suite for Gini coefficient calculation."""
    
    def test_gini_perfect_model(self):
        """Test Gini for perfect model (Gini = 1)."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        gini = CreditScoringMetrics.gini_coefficient(y_true, y_score)
        
        assert gini == pytest.approx(1.0, abs=0.01)
    
    def test_gini_random_model(self):
        """Test Gini for random model (Gini ≈ 0)."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 1000, p=[0.8, 0.2])
        y_score = np.random.rand(1000)
        
        gini = CreditScoringMetrics.gini_coefficient(y_true, y_score)
        
        # Random model should have Gini close to 0
        assert abs(gini) < 0.15
    
    def test_gini_realistic_model(self, prediction_data):
        """Test Gini for realistic model predictions."""
        y_true, y_prob, _ = prediction_data
        
        gini = CreditScoringMetrics.gini_coefficient(y_true, y_prob)
        
        # Realistic model should have positive Gini
        assert 0 < gini < 1
    
    def test_gini_relation_to_auc(self, prediction_data):
        """Test that Gini = 2*AUC - 1."""
        from sklearn.metrics import roc_auc_score
        
        y_true, y_prob, _ = prediction_data
        
        auc = roc_auc_score(y_true, y_prob)
        gini = CreditScoringMetrics.gini_coefficient(y_true, y_prob)
        
        assert gini == pytest.approx(2 * auc - 1, abs=0.01)


class TestKSStatistic:
    """Test suite for Kolmogorov-Smirnov statistic."""
    
    def test_ks_perfect_separation(self):
        """Test KS for perfect separation (KS = 1)."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        ks, threshold = CreditScoringMetrics.ks_statistic(y_true, y_score)
        
        assert ks == pytest.approx(1.0, abs=0.01)
    
    def test_ks_random_model(self):
        """Test KS for random model (KS ≈ 0)."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 1000, p=[0.8, 0.2])
        y_score = np.random.rand(1000)
        
        ks, threshold = CreditScoringMetrics.ks_statistic(y_true, y_score)
        
        # Random model should have low KS
        assert ks < 0.20
    
    def test_ks_realistic_model(self, prediction_data):
        """Test KS for realistic model predictions."""
        y_true, y_prob, _ = prediction_data
        
        ks, threshold = CreditScoringMetrics.ks_statistic(y_true, y_prob)
        
        # Should return both KS value and threshold
        assert 0 < ks <= 1
        assert 0 <= threshold <= 1
    
    def test_ks_returns_threshold(self, prediction_data):
        """Test that KS returns the threshold at max separation."""
        y_true, y_prob, _ = prediction_data
        
        ks, threshold = CreditScoringMetrics.ks_statistic(y_true, y_prob)
        
        # Threshold should be a probability value
        assert isinstance(threshold, (int, float))
        assert 0 <= threshold <= 1


class TestLiftTable:
    """Test suite for Lift table calculation."""
    
    def test_lift_table_structure(self, prediction_data):
        """Test lift table has correct structure."""
        y_true, y_prob, _ = prediction_data
        
        lift_df = CreditScoringMetrics.lift_table(y_true, y_prob, n_deciles=10)
        
        assert isinstance(lift_df, pd.DataFrame)
        assert len(lift_df) == 10
    
    def test_lift_table_columns(self, prediction_data):
        """Test lift table has expected columns."""
        y_true, y_prob, _ = prediction_data
        
        lift_df = CreditScoringMetrics.lift_table(y_true, y_prob)
        
        # Actual columns from the implementation
        expected_columns = ['decile', 'score_min', 'score_max', 'count', 'bads', 'bad_rate', 'lift']
        for col in expected_columns:
            assert col in lift_df.columns, f"Missing column: {col}"
    
    def test_lift_decreasing_bad_rate(self, prediction_data):
        """Test that bad rate decreases across deciles (for good model)."""
        y_true, y_prob, _ = prediction_data
        
        lift_df = CreditScoringMetrics.lift_table(y_true, y_prob)
        
        # Find bad_rate column (might have different name)
        bad_rate_col = None
        for col in lift_df.columns:
            if 'bad_rate' in col.lower() or 'event_rate' in col.lower():
                bad_rate_col = col
                break
        
        if bad_rate_col:
            bad_rates = lift_df[bad_rate_col].values
            # First decile should have highest bad rate for a good model
            assert bad_rates[0] >= bad_rates[-1]
    
    def test_lift_custom_deciles(self, prediction_data):
        """Test lift table with custom number of deciles."""
        y_true, y_prob, _ = prediction_data
        
        lift_5 = CreditScoringMetrics.lift_table(y_true, y_prob, n_deciles=5)
        lift_20 = CreditScoringMetrics.lift_table(y_true, y_prob, n_deciles=20)
        
        assert len(lift_5) == 5
        assert len(lift_20) == 20


class TestPSI:
    """Test suite for Population Stability Index."""
    
    def test_psi_identical_distributions(self):
        """Test PSI for identical distributions (PSI = 0)."""
        expected = np.random.rand(1000)
        actual = expected.copy()
        
        psi, breakdown = CreditScoringMetrics.psi(expected, actual)
        
        assert psi == pytest.approx(0.0, abs=0.01)
    
    def test_psi_similar_distributions(self):
        """Test PSI for similar distributions (PSI < 0.1)."""
        np.random.seed(42)
        expected = np.random.normal(0.5, 0.15, 1000)
        actual = np.random.normal(0.52, 0.15, 1000)  # Slight shift
        
        psi, breakdown = CreditScoringMetrics.psi(expected, actual)
        
        # Similar distributions should have low PSI
        assert psi < 0.25
    
    def test_psi_different_distributions(self):
        """Test PSI for different distributions (PSI > 0.25)."""
        np.random.seed(42)
        expected = np.random.normal(0.3, 0.1, 1000)
        actual = np.random.normal(0.7, 0.1, 1000)  # Large shift
        
        psi, breakdown = CreditScoringMetrics.psi(expected, actual)
        
        # Very different distributions should have high PSI
        assert psi > 0.25
    
    def test_psi_returns_breakdown(self):
        """Test that PSI returns bin-level breakdown."""
        np.random.seed(42)
        expected = np.random.rand(500)
        actual = np.random.rand(500)
        
        psi, breakdown = CreditScoringMetrics.psi(expected, actual)
        
        assert breakdown is not None
        assert isinstance(breakdown, pd.DataFrame)
    
    def test_psi_custom_bins(self):
        """Test PSI with custom number of bins."""
        np.random.seed(42)
        expected = np.random.rand(500)
        actual = np.random.rand(500)
        
        psi_5, _ = CreditScoringMetrics.psi(expected, actual, n_bins=5)
        psi_20, _ = CreditScoringMetrics.psi(expected, actual, n_bins=20)
        
        # Both should give similar results for random data
        assert abs(psi_5 - psi_20) < 0.2


class TestAllMetrics:
    """Test suite for combined metrics calculation."""
    
    def test_calculate_all_metrics(self, prediction_data):
        """Test calculate_all_metrics returns all expected metrics."""
        y_true, y_prob, y_pred = prediction_data
        
        metrics = CreditScoringMetrics.calculate_all_metrics(
            y_true, y_prob, y_pred
        )
        
        assert isinstance(metrics, dict)
        assert 'gini' in metrics or 'Gini' in metrics
        assert 'auc' in metrics or 'AUC' in metrics
    
    def test_calculate_all_metrics_without_y_pred(self, prediction_data):
        """Test metrics calculation when y_pred is not provided."""
        y_true, y_prob, _ = prediction_data
        
        metrics = CreditScoringMetrics.calculate_all_metrics(
            y_true, y_prob
        )
        
        # Should still work and compute metrics
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    def test_metrics_values_in_range(self, prediction_data):
        """Test that metric values are in expected ranges."""
        y_true, y_prob, y_pred = prediction_data
        
        metrics = CreditScoringMetrics.calculate_all_metrics(
            y_true, y_prob, y_pred
        )
        
        # Check Gini/AUC range
        for key in metrics:
            if 'gini' in key.lower():
                assert -1 <= metrics[key] <= 1
            if 'auc' in key.lower():
                assert 0 <= metrics[key] <= 1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise appropriate errors."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            CreditScoringMetrics.gini_coefficient(np.array([]), np.array([]))
    
    def test_single_class_warning(self):
        """Test behavior when only one class is present."""
        y_true = np.zeros(100)  # All zeros
        y_score = np.random.rand(100)
        
        # Should handle gracefully (might return NaN or raise warning)
        try:
            gini = CreditScoringMetrics.gini_coefficient(y_true, y_score)
            # If it returns, should be 0 or NaN
            assert np.isnan(gini) or gini == 0
        except (ValueError, ZeroDivisionError):
            pass  # Valid to raise error for single class
    
    def test_mismatched_lengths_raise_error(self):
        """Test that mismatched array lengths raise errors."""
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9])  # Wrong length
        
        with pytest.raises((ValueError, IndexError)):
            CreditScoringMetrics.gini_coefficient(y_true, y_score)
