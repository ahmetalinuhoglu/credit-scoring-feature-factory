"""
Tests for Feature Extractor

Tests FeatureExtractor functionality for credit scoring feature generation.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.features.feature_extractor import FeatureExtractor
from src.core.exceptions import FeatureEngineeringError


class TestFeatureExtractorInit:
    """Test suite for FeatureExtractor initialization."""
    
    def test_init_basic(self, base_config, mock_spark_session):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert extractor is not None
        assert extractor.name == "FeatureExtractor"
    
    def test_init_with_custom_name(self, base_config, mock_spark_session):
        """Test FeatureExtractor with custom name."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session,
            name="CustomExtractor"
        )
        
        assert extractor.name == "CustomExtractor"
    
    def test_validate(self, base_config, mock_spark_session):
        """Test validate method."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert extractor.validate() is True
    
    def test_feature_list_initialized(self, base_config, mock_spark_session):
        """Test that generated features list is initialized."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert hasattr(extractor, '_generated_features')
        assert extractor._generated_features == []


class TestFeatureExtractorExtract:
    """Test suite for extract_features method."""
    
    def test_run_calls_extract_features(self, base_config, mock_spark_session):
        """Test run method calls extract_features."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        mock_apps = MagicMock()
        mock_bureau = MagicMock()
        
        with patch.object(extractor, 'extract_features', return_value=MagicMock()) as mock_extract:
            extractor.run(mock_apps, mock_bureau)
            
            mock_extract.assert_called_once()


class TestFeatureExtractorRegistration:
    """Test suite for feature registration."""
    
    def test_register_features(self, base_config, mock_spark_session):
        """Test registering generated features."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        features = ['feature_1', 'feature_2', 'feature_3']
        extractor._register_features(features)
        
        # _generated_features is a list of dicts
        feature_names = [f['name'] for f in extractor._generated_features]
        assert 'feature_1' in feature_names
        assert 'feature_2' in feature_names
        assert 'feature_3' in feature_names
    
    def test_generated_features_property(self, base_config, mock_spark_session):
        """Test generated_features property."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        # Use _register_features to set up properly
        extractor._register_features(['f1', 'f2'])
        
        result = extractor.generated_features
        
        assert len(result) == 2
        assert result[0]['name'] == 'f1'
        assert result[1]['name'] == 'f2'
    
    def test_get_feature_names(self, base_config, mock_spark_session):
        """Test get_feature_names method."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        # Use _register_features to set up properly
        extractor._register_features(['feature_a', 'feature_b'])
        
        result = extractor.get_feature_names()
        
        assert 'feature_a' in result
        assert 'feature_b' in result


class TestFeatureExtractorCreditProducts:
    """Test suite for credit product constants."""
    
    def test_credit_products_defined(self, base_config, mock_spark_session):
        """Test that credit product types are defined."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert hasattr(extractor, 'CREDIT_PRODUCTS')
        assert len(extractor.CREDIT_PRODUCTS) > 0
    
    def test_non_credit_products_defined(self, base_config, mock_spark_session):
        """Test that non-credit product types are defined."""
        extractor = FeatureExtractor(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert hasattr(extractor, 'NON_CREDIT_PRODUCTS')
        assert len(extractor.NON_CREDIT_PRODUCTS) > 0
