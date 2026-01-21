"""
Tests for BigQuery Reader

Tests BigQueryReader with mocked BigQuery/Spark interactions.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.data.bigquery_reader import BigQueryReader
from src.core.exceptions import DataReaderError


class TestBigQueryReaderInit:
    """Test suite for BigQueryReader initialization."""
    
    def test_init_basic(self, data_config, mock_spark_session):
        """Test BigQueryReader initialization."""
        reader = BigQueryReader(
            config=data_config,
            spark_session=mock_spark_session
        )
        
        assert reader is not None
    
    def test_init_with_project_id(self, data_config, mock_spark_session):
        """Test BigQueryReader with explicit project ID."""
        reader = BigQueryReader(
            config=data_config,
            spark_session=mock_spark_session,
            project_id='my-project'
        )
        
        assert reader.project_id == 'my-project'
    
    def test_init_with_dataset(self, data_config, mock_spark_session):
        """Test BigQueryReader with explicit dataset."""
        reader = BigQueryReader(
            config=data_config,
            spark_session=mock_spark_session,
            dataset='my_dataset'
        )
        
        assert reader.dataset == 'my_dataset'
    
    def test_init_loads_from_config(self, mock_spark_session):
        """Test BigQueryReader loads settings from config."""
        # Use correct config paths matching implementation
        config = {
            'gcp': {
                'project_id': 'config-project',
                'bigquery': {
                    'dataset': 'config_dataset'
                }
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        
        assert reader.project_id == 'config-project'
        assert reader.dataset == 'config_dataset'


class TestBigQueryReaderValidate:
    """Test suite for BigQueryReader validation."""
    
    def test_validate_success(self, mock_spark_session):
        """Test validation with valid config."""
        config = {
            'gcp': {
                'project_id': 'valid-project',
                'bigquery': {
                    'dataset': 'valid_dataset'
                }
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        
        assert reader.validate() is True
    
    def test_validate_missing_project(self, mock_spark_session):
        """Test validation fails without project."""
        config = {'gcp': {'bigquery': {'dataset': 'test'}}}
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        
        assert reader.validate() is False
    
    def test_validate_missing_dataset(self, mock_spark_session):
        """Test validation fails without dataset."""
        config = {'gcp': {'project_id': 'my-project'}}
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        
        assert reader.validate() is False


def create_mock_spark_read(mock_spark_session):
    """Helper to set up mock Spark read chain."""
    mock_df = MagicMock()
    mock_df.count.return_value = 1000
    mock_df.select.return_value = mock_df
    mock_df.filter.return_value = mock_df
    mock_df.sample.return_value = mock_df
    
    mock_reader = MagicMock()
    mock_reader.option.return_value = mock_reader
    mock_reader.load.return_value = mock_df
    
    mock_spark_session.read.format.return_value = mock_reader
    
    return mock_df


class TestBigQueryReaderRead:
    """Test suite for BigQueryReader read operations."""
    
    def test_read_basic(self, mock_spark_session):
        """Test basic read operation."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        result = reader.read('my_table')
        
        assert result is not None
        mock_spark_session.read.format.assert_called_with("bigquery")
    
    def test_read_with_columns(self, mock_spark_session):
        """Test read with column selection."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        result = reader.read('my_table', columns=['col1', 'col2'])
        
        assert result is not None
    
    def test_read_with_filter(self, mock_spark_session):
        """Test read with filter expression."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        result = reader.read('my_table', filter_expr="status = 'active'")
        
        assert result is not None
    
    def test_read_with_sample(self, mock_spark_session):
        """Test read with sampling."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        result = reader.read('my_table', sample_fraction=0.1)
        
        assert result is not None
        mock_df.sample.assert_called()


class TestBigQueryReaderReadWithFilter:
    """Test suite for read_with_filter method."""
    
    def test_read_with_filter_values(self, mock_spark_session):
        """Test read_with_filter with value filtering."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        result = reader.read_with_filter(
            source='my_table',
            filter_column='id',
            filter_values=['id1', 'id2', 'id3']
        )
        
        assert result is not None
    
    def test_read_with_filter_large_value_list(self, mock_spark_session):
        """Test read_with_filter handles large filter lists."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        # Large list of values
        filter_values = [f'id_{i}' for i in range(100)]
        
        result = reader.read_with_filter(
            source='my_table',
            filter_column='id',
            filter_values=filter_values
        )
        
        assert result is not None


class TestBigQueryReaderReadWithDateRange:
    """Test suite for read_with_date_range method."""
    
    def test_read_with_date_range(self, mock_spark_session):
        """Test read_with_date_range method."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        result = reader.read_with_date_range(
            source='my_table',
            date_column='created_at',
            start_date=start_date,
            end_date=end_date
        )
        
        assert result is not None


class TestBigQueryReaderConvenienceMethods:
    """Test suite for convenience methods."""
    
    def test_read_applications(self, mock_spark_session):
        """Test read_applications convenience method."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            },
            'data': {
                'data_sources': {
                    'applications': {'table': 'applications_table'}
                }
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        result = reader.read_applications()
        
        assert result is not None
    
    def test_read_credit_bureau(self, mock_spark_session):
        """Test read_credit_bureau convenience method."""
        config = {
            'gcp': {
                'project_id': 'test-project',
                'bigquery': {'dataset': 'test_dataset'}
            },
            'data': {
                'data_sources': {
                    'credit_bureau': {'table': 'credit_bureau_table'}
                }
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        mock_df = create_mock_spark_read(mock_spark_session)
        
        result = reader.read_credit_bureau()
        
        assert result is not None


class TestBigQueryReaderFullTableName:
    """Test suite for table name handling."""
    
    def test_get_full_table_name(self, mock_spark_session):
        """Test _get_full_table_name method."""
        config = {
            'gcp': {
                'project_id': 'my-project',
                'bigquery': {'dataset': 'my_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        
        full_name = reader._get_full_table_name('my_table')
        
        assert 'my-project' in full_name
        assert 'my_dataset' in full_name
        assert 'my_table' in full_name
    
    def test_get_full_table_name_already_qualified(self, mock_spark_session):
        """Test _get_full_table_name with already qualified name."""
        config = {
            'gcp': {
                'project_id': 'my-project',
                'bigquery': {'dataset': 'my_dataset'}
            }
        }
        
        reader = BigQueryReader(config=config, spark_session=mock_spark_session)
        
        # Already qualified name should be returned as-is
        full_name = reader._get_full_table_name('other-project.other_dataset.table')
        
        assert full_name == 'other-project.other_dataset.table'
