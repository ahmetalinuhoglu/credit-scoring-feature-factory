"""
Tests for Feature Quality Checker

Tests FeatureQualityChecker functionality for feature analysis.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.quality.feature_quality import FeatureQualityChecker, FeatureQualityResult


class TestFeatureQualityResult:
    """Test suite for FeatureQualityResult dataclass."""
    
    def test_result_creation(self):
        """Test FeatureQualityResult creation."""
        result = FeatureQualityResult(
            feature_name='test_feature',
            null_ratio=0.05,
            unique_count=100
        )
        
        assert result.feature_name == 'test_feature'
        assert result.null_ratio == 0.05
        assert result.unique_count == 100
    
    def test_result_with_iv(self):
        """Test FeatureQualityResult with IV score."""
        result = FeatureQualityResult(
            feature_name='predictive_feature',
            null_ratio=0.01,
            unique_count=50,
            iv_score=0.35,
            iv_category='strong'
        )
        
        assert result.iv_score == 0.35
        assert result.iv_category == 'strong'
    
    def test_result_to_dict(self):
        """Test FeatureQualityResult to_dict method."""
        result = FeatureQualityResult(
            feature_name='feature1',
            null_ratio=0.1,
            unique_count=10,
            iv_score=0.2,
            iv_category='medium'
        )
        
        d = result.to_dict()
        
        assert d['feature_name'] == 'feature1'
        assert d['iv_score'] == 0.2


class TestFeatureQualityCheckerInit:
    """Test suite for FeatureQualityChecker initialization."""
    
    def test_init_basic(self, base_config, mock_spark_session):
        """Test FeatureQualityChecker initialization."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker is not None
        assert checker.name == "FeatureQualityChecker"
    
    def test_init_with_custom_name(self, base_config, mock_spark_session):
        """Test FeatureQualityChecker with custom name."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session,
            name="CustomChecker"
        )
        
        assert checker.name == "CustomChecker"
    
    def test_iv_thresholds_available(self, base_config, mock_spark_session):
        """Test IV thresholds are available."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert hasattr(checker, 'IV_THRESHOLDS')
        assert 'useless' in checker.IV_THRESHOLDS
        assert 'strong' in checker.IV_THRESHOLDS
    
    def test_validate(self, base_config, mock_spark_session):
        """Test validate method."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker.validate() is True


class TestFeatureQualityCheckerIVCategory:
    """Test suite for IV categorization."""
    
    def test_get_iv_category_useless(self, base_config, mock_spark_session):
        """Test IV category for useless features."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker._get_iv_category(0.01) == 'useless'
    
    def test_get_iv_category_weak(self, base_config, mock_spark_session):
        """Test IV category for weak features."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker._get_iv_category(0.05) == 'weak'
    
    def test_get_iv_category_medium(self, base_config, mock_spark_session):
        """Test IV category for medium features."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker._get_iv_category(0.15) == 'medium'
    
    def test_get_iv_category_strong(self, base_config, mock_spark_session):
        """Test IV category for strong features."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker._get_iv_category(0.35) == 'strong'
    
    def test_get_iv_category_suspicious(self, base_config, mock_spark_session):
        """Test IV category for suspicious features."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker._get_iv_category(0.6) == 'suspicious'


class TestFeatureQualityCheckerIVSummary:
    """Test suite for IV summary methods."""
    
    def test_get_iv_summary(self, base_config, mock_spark_session):
        """Test get_iv_summary method."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        # Create mock results
        results = {
            'f1': FeatureQualityResult('f1', 0.01, 10, iv_score=0.01, iv_category='useless'),
            'f2': FeatureQualityResult('f2', 0.02, 20, iv_score=0.15, iv_category='medium'),
            'f3': FeatureQualityResult('f3', 0.01, 30, iv_score=0.35, iv_category='strong')
        }
        
        summary = checker.get_iv_summary(results)
        
        assert 'useless' in summary
        assert 'medium' in summary
        assert 'strong' in summary
        assert 'f1' in summary['useless']
        assert 'f2' in summary['medium']
        assert 'f3' in summary['strong']


class TestFeatureQualityCheckerHighCorrelations:
    """Test suite for correlation analysis."""
    
    def test_find_high_correlations(self, base_config, mock_spark_session):
        """Test finding high correlations."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        correlation_matrix = {
            'f1': {'f1': 1.0, 'f2': 0.95, 'f3': 0.3},
            'f2': {'f1': 0.95, 'f2': 1.0, 'f3': 0.2},
            'f3': {'f1': 0.3, 'f2': 0.2, 'f3': 1.0}
        }
        
        high_corr = checker.find_high_correlations(correlation_matrix, threshold=0.9)
        
        assert len(high_corr) == 1
        assert high_corr[0][2] == 0.95
    
    def test_find_high_correlations_no_results(self, base_config, mock_spark_session):
        """Test finding high correlations with no results."""
        checker = FeatureQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        correlation_matrix = {
            'f1': {'f1': 1.0, 'f2': 0.3},
            'f2': {'f1': 0.3, 'f2': 1.0}
        }
        
        high_corr = checker.find_high_correlations(correlation_matrix, threshold=0.9)
        
        assert len(high_corr) == 0
