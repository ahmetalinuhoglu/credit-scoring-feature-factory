"""
Tests for Data Dictionary Generator

Tests DataDictionaryGenerator functionality for automatic documentation.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path
import json

from src.features.data_dictionary import DataDictionaryGenerator


class TestDataDictionaryGeneratorInit:
    """Test suite for DataDictionaryGenerator initialization."""
    
    def test_init_basic(self, base_config, mock_spark_session):
        """Test DataDictionaryGenerator initialization."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator is not None
        assert generator.name == "DataDictionaryGenerator"
    
    def test_init_with_custom_name(self, base_config, mock_spark_session):
        """Test DataDictionaryGenerator with custom name."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session,
            name="CustomGenerator"
        )
        
        assert generator.name == "CustomGenerator"
    
    def test_init_with_output_dir(self, base_config, mock_spark_session):
        """Test DataDictionaryGenerator with output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = DataDictionaryGenerator(
                config=base_config,
                spark_session=mock_spark_session,
                output_dir=temp_dir
            )
            
            assert generator.output_dir == Path(temp_dir)
    
    def test_validate(self, base_config, mock_spark_session):
        """Test validate method."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator.validate() is True


class TestDataDictionaryGeneratorGenerate:
    """Test suite for generate method."""
    
    def test_run_calls_generate(self, base_config, mock_spark_session):
        """Test run method calls generate."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        mock_df = MagicMock()
        
        with patch.object(generator, 'generate', return_value={}) as mock_gen:
            generator.run(mock_df, target_column='target')
            
            mock_gen.assert_called_once()


class TestDataDictionaryGeneratorIV:
    """Test suite for IV categorization."""
    
    def test_categorize_iv_useless(self, base_config, mock_spark_session):
        """Test IV categorization for useless features."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._categorize_iv(0.01) == 'useless'
    
    def test_categorize_iv_weak(self, base_config, mock_spark_session):
        """Test IV categorization for weak features."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._categorize_iv(0.05) == 'weak'
    
    def test_categorize_iv_medium(self, base_config, mock_spark_session):
        """Test IV categorization for medium features."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._categorize_iv(0.15) == 'medium'
    
    def test_categorize_iv_strong(self, base_config, mock_spark_session):
        """Test IV categorization for strong features."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._categorize_iv(0.35) == 'strong'
    
    def test_categorize_iv_suspicious(self, base_config, mock_spark_session):
        """Test IV categorization for suspicious features."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._categorize_iv(0.6) == 'suspicious'
    
    def test_categorize_iv_unknown(self, base_config, mock_spark_session):
        """Test IV categorization for None."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._categorize_iv(None) == 'unknown'


class TestDataDictionaryGeneratorHelpers:
    """Test suite for helper methods."""
    
    def test_get_description_generated(self, base_config, mock_spark_session):
        """Test generating description from feature name."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        desc = generator._get_description('num_credit_accounts_12m')
        
        assert desc is not None
        assert len(desc) > 0
    
    def test_get_category_amount(self, base_config, mock_spark_session):
        """Test feature categorization for amount features."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._get_category('total_amount') == 'amount'
    
    def test_get_category_count(self, base_config, mock_spark_session):
        """Test feature categorization for count features."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._get_category('num_accounts') == 'count'
    
    def test_simplify_dtype_float(self, base_config, mock_spark_session):
        """Test dtype simplification for float."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._simplify_dtype('DoubleType()') == 'float'
    
    def test_simplify_dtype_integer(self, base_config, mock_spark_session):
        """Test dtype simplification for integer."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._simplify_dtype('IntegerType()') == 'integer'
    
    def test_simplify_dtype_string(self, base_config, mock_spark_session):
        """Test dtype simplification for string."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._simplify_dtype('StringType()') == 'string'
    
    def test_safe_float(self, base_config, mock_spark_session):
        """Test safe float conversion."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert generator._safe_float(10) == 10.0
        assert generator._safe_float('5.5') == 5.5
        assert generator._safe_float(None) is None
        assert generator._safe_float('invalid') is None


class TestDataDictionaryGeneratorExport:
    """Test suite for export functionality."""
    
    def test_export_yaml(self, base_config, mock_spark_session, temp_output_dir):
        """Test YAML export."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session,
            output_dir=str(temp_output_dir)
        )
        
        dictionary = {
            'metadata': {
                'generated_at': '2024-01-01',
                'total_features': 1,
                'total_rows': 1000
            },
            'features': [
                {'name': 'feature_1', 'data_type': 'float', 'description': 'Test feature'}
            ]
        }
        
        result = generator.export(dictionary, formats=['yaml'])
        
        assert 'yaml' in result
        assert Path(result['yaml']).exists()
    
    def test_export_json(self, base_config, mock_spark_session, temp_output_dir):
        """Test JSON export."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session,
            output_dir=str(temp_output_dir)
        )
        
        dictionary = {
            'metadata': {
                'generated_at': '2024-01-01',
                'total_features': 1,
                'total_rows': 500
            },
            'features': [{'name': 'f1', 'data_type': 'string'}]
        }
        
        result = generator.export(dictionary, formats=['json'])
        
        assert 'json' in result
        assert Path(result['json']).exists()
    
    def test_export_html(self, base_config, mock_spark_session, temp_output_dir):
        """Test HTML export."""
        generator = DataDictionaryGenerator(
            config=base_config,
            spark_session=mock_spark_session,
            output_dir=str(temp_output_dir)
        )
        
        dictionary = {
            'metadata': {
                'generated_at': '2024-01-01',
                'total_features': 1,
                'total_rows': 1000
            },
            'features': [
                {
                    'name': 'feature_1',
                    'data_type': 'float',
                    'iv_score': 0.35,
                    'iv_category': 'strong',
                    'description': 'Test',
                    'category': 'amount',
                    'null_ratio': 0.01,
                    'statistics': {'mean': 100, 'std': 15}
                }
            ]
        }
        
        result = generator.export(dictionary, formats=['html'])
        
        assert 'html' in result
        content = Path(result['html']).read_text()
        assert '<html>' in content.lower()
