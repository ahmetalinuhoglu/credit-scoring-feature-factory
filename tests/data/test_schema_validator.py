"""
Tests for Schema Validator

Tests SchemaValidationResult and SchemaValidator functionality.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from src.data.schema_validator import SchemaValidationResult, SchemaValidator
from src.core.exceptions import SchemaValidationError


class TestSchemaValidationResult:
    """Test suite for SchemaValidationResult dataclass."""
    
    def test_result_creation(self):
        """Test SchemaValidationResult creation."""
        result = SchemaValidationResult(
            is_valid=True,
            missing_columns=[],
            extra_columns=[],
            type_mismatches=[],
            errors=[]
        )
        
        assert result.is_valid is True
        assert len(result.missing_columns) == 0
    
    def test_result_with_errors(self):
        """Test SchemaValidationResult with validation errors."""
        result = SchemaValidationResult(
            is_valid=False,
            missing_columns=['col1', 'col2'],
            extra_columns=['extra_col'],
            type_mismatches=[{'column': 'col3', 'expected': 'string', 'actual': 'int'}],
            errors=['Missing columns: col1, col2']
        )
        
        assert result.is_valid is False
        assert len(result.missing_columns) == 2
        assert len(result.type_mismatches) == 1
    
    def test_to_dict(self):
        """Test to_dict serialization."""
        result = SchemaValidationResult(
            is_valid=True,
            missing_columns=[],
            extra_columns=['extra'],
            type_mismatches=[],
            errors=[]
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['is_valid'] is True
        assert result_dict['extra_columns'] == ['extra']


class TestSchemaValidatorInit:
    """Test suite for SchemaValidator initialization."""
    
    def test_init_basic(self, data_config, mock_spark_session):
        """Test SchemaValidator initialization."""
        validator = SchemaValidator(
            config=data_config,
            spark_session=mock_spark_session
        )
        
        assert validator is not None
        assert validator.name == "SchemaValidator"
    
    def test_init_with_custom_name(self, data_config, mock_spark_session):
        """Test SchemaValidator with custom name."""
        validator = SchemaValidator(
            config=data_config,
            spark_session=mock_spark_session,
            name="CustomValidator"
        )
        
        assert validator.name == "CustomValidator"
    
    def test_schemas_loaded_from_config(self, mock_spark_session):
        """Test schemas are loaded from config."""
        config = {
            'data': {
                'schemas': {
                    'test_schema': {
                        'columns': [
                            {'name': 'id', 'type': 'string'},
                            {'name': 'value', 'type': 'integer'}
                        ]
                    }
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        assert 'test_schema' in validator.schemas


class TestSchemaValidatorValidation:
    """Test suite for SchemaValidator validate method."""
    
    def test_validate_with_schemas(self, mock_spark_session):
        """Test validate returns True when schemas configured."""
        config = {
            'data': {
                'schemas': {
                    'test': {'columns': []}
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        assert validator.validate() is True
    
    def test_validate_without_schemas(self, data_config, mock_spark_session):
        """Test validate handles missing schemas gracefully."""
        config = {'data': {}}
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        # Should still return True, just logs warning
        assert validator.validate() is True


class TestSchemaValidatorValidateSchema:
    """Test suite for validate_schema method."""
    
    def test_validate_schema_success(self, mock_spark_session):
        """Test validate_schema with matching schema."""
        config = {
            'data': {
                'schemas': {
                    'test_table': {
                        'columns': [
                            {'name': 'id', 'type': 'string'},
                            {'name': 'value', 'type': 'integer'}
                        ]
                    }
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        # Create mock DataFrame with matching schema
        mock_df = MagicMock()
        mock_field1 = MagicMock()
        mock_field1.name = 'id'
        mock_field1.dataType = MagicMock(__str__=lambda x: 'StringType()')
        
        mock_field2 = MagicMock()
        mock_field2.name = 'value'
        mock_field2.dataType = MagicMock(__str__=lambda x: 'IntegerType()')
        
        mock_df.schema.fields = [mock_field1, mock_field2]
        
        result = validator.validate_schema(mock_df, 'test_table')
        
        assert result.is_valid is True
    
    def test_validate_schema_missing_columns(self, mock_spark_session):
        """Test validate_schema detects missing columns."""
        config = {
            'data': {
                'schemas': {
                    'test_table': {
                        'columns': [
                            {'name': 'id', 'type': 'string'},
                            {'name': 'missing_col', 'type': 'string'}
                        ]
                    }
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        mock_df = MagicMock()
        mock_field = MagicMock()
        mock_field.name = 'id'
        mock_field.dataType = MagicMock(__str__=lambda x: 'StringType()')
        mock_df.schema.fields = [mock_field]
        
        result = validator.validate_schema(mock_df, 'test_table')
        
        assert result.is_valid is False
        assert 'missing_col' in result.missing_columns
    
    def test_validate_schema_extra_columns_strict(self, mock_spark_session):
        """Test validate_schema detects extra columns in strict mode."""
        config = {
            'data': {
                'schemas': {
                    'test_table': {
                        'columns': [
                            {'name': 'id', 'type': 'string'}
                        ]
                    }
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        mock_df = MagicMock()
        mock_field1 = MagicMock()
        mock_field1.name = 'id'
        mock_field1.dataType = MagicMock(__str__=lambda x: 'StringType()')
        
        mock_field2 = MagicMock()
        mock_field2.name = 'extra_col'
        mock_field2.dataType = MagicMock(__str__=lambda x: 'StringType()')
        
        mock_df.schema.fields = [mock_field1, mock_field2]
        
        result = validator.validate_schema(mock_df, 'test_table', strict=True)
        
        assert 'extra_col' in result.extra_columns
    
    def test_validate_schema_unknown_name_raises(self, data_config, mock_spark_session):
        """Test validate_schema raises for unknown schema name."""
        validator = SchemaValidator(config=data_config, spark_session=mock_spark_session)
        
        mock_df = MagicMock()
        
        with pytest.raises(SchemaValidationError):
            validator.validate_schema(mock_df, 'nonexistent_schema')
    
    def test_run_calls_validate_schema(self, mock_spark_session):
        """Test run method calls validate_schema."""
        config = {
            'data': {
                'schemas': {
                    'test': {'columns': []}
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        mock_df = MagicMock()
        mock_df.schema.fields = []
        
        result = validator.run(mock_df, 'test')
        
        assert isinstance(result, SchemaValidationResult)


class TestSchemaValidatorTypeMaching:
    """Test suite for type matching logic."""
    
    def test_type_matches_string(self, mock_spark_session):
        """Test type matching for string type."""
        validator = SchemaValidator(
            config={'data': {'schemas': {}}},
            spark_session=mock_spark_session
        )
        
        assert validator._type_matches('string', 'StringType()') is True
        assert validator._type_matches('string', 'IntegerType()') is False
    
    def test_type_matches_integer(self, mock_spark_session):
        """Test type matching for integer types."""
        validator = SchemaValidator(
            config={'data': {'schemas': {}}},
            spark_session=mock_spark_session
        )
        
        assert validator._type_matches('integer', 'IntegerType()') is True
        assert validator._type_matches('integer', 'LongType()') is True
    
    def test_type_matches_float(self, mock_spark_session):
        """Test type matching for float types."""
        validator = SchemaValidator(
            config={'data': {'schemas': {}}},
            spark_session=mock_spark_session
        )
        
        assert validator._type_matches('float', 'FloatType()') is True
        assert validator._type_matches('float', 'DoubleType()') is True


class TestSchemaValidatorHelpers:
    """Test suite for helper methods."""
    
    def test_validate_and_raise_success(self, mock_spark_session):
        """Test validate_and_raise passes for valid schema."""
        config = {
            'data': {
                'schemas': {
                    'test': {'columns': []}
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        mock_df = MagicMock()
        mock_df.schema.fields = []
        
        # Should not raise
        validator.validate_and_raise(mock_df, 'test')
    
    def test_validate_and_raise_failure(self, mock_spark_session):
        """Test validate_and_raise raises for invalid schema."""
        config = {
            'data': {
                'schemas': {
                    'test': {
                        'columns': [{'name': 'required_col', 'type': 'string'}]
                    }
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        mock_df = MagicMock()
        mock_df.schema.fields = []
        
        with pytest.raises(SchemaValidationError):
            validator.validate_and_raise(mock_df, 'test')
    
    def test_get_schema_info(self, mock_spark_session):
        """Test get_schema_info returns schema details."""
        config = {
            'data': {
                'schemas': {
                    'my_schema': {'columns': [{'name': 'col1'}]}
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        info = validator.get_schema_info('my_schema')
        
        assert 'columns' in info
    
    def test_list_schemas(self, mock_spark_session):
        """Test list_schemas returns schema names."""
        config = {
            'data': {
                'schemas': {
                    'schema1': {},
                    'schema2': {}
                }
            }
        }
        
        validator = SchemaValidator(config=config, spark_session=mock_spark_session)
        
        schemas = validator.list_schemas()
        
        assert 'schema1' in schemas
        assert 'schema2' in schemas
