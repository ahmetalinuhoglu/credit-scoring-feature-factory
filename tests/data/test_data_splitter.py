"""
Tests for Data Splitter

Tests DataSplitter functionality for train/test/validation splitting.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.data.data_splitter import DataSplitter, DataSplit


class TestDataSplit:
    """Test suite for DataSplit dataclass."""
    
    def test_datasplit_creation(self):
        """Test DataSplit creation."""
        mock_train = MagicMock()
        mock_test = MagicMock()
        
        split = DataSplit(train=mock_train, test=mock_test)
        
        assert split.train is mock_train
        assert split.test is mock_test
    
    def test_datasplit_with_validation(self):
        """Test DataSplit with validation set."""
        mock_train = MagicMock()
        mock_test = MagicMock()
        mock_val = MagicMock()
        
        split = DataSplit(train=mock_train, test=mock_test, validation=mock_val)
        
        assert split.validation is mock_val
    
    def test_has_validation_true(self):
        """Test has_validation property when validation exists."""
        split = DataSplit(
            train=MagicMock(),
            test=MagicMock(),
            validation=MagicMock()
        )
        
        assert split.has_validation is True
    
    def test_has_validation_false(self):
        """Test has_validation property when no validation."""
        split = DataSplit(
            train=MagicMock(),
            test=MagicMock()
        )
        
        assert split.has_validation is False


class TestDataSplitterInit:
    """Test suite for DataSplitter initialization."""
    
    def test_init_with_defaults(self, base_config, mock_spark_session):
        """Test DataSplitter initialization with defaults."""
        splitter = DataSplitter(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert splitter is not None
        assert splitter.name == "DataSplitter"
    
    def test_init_with_custom_name(self, base_config, mock_spark_session):
        """Test DataSplitter with custom name."""
        splitter = DataSplitter(
            config=base_config,
            spark_session=mock_spark_session,
            name="CustomSplitter"
        )
        
        assert splitter.name == "CustomSplitter"
    
    def test_splits_from_config(self, mock_spark_session):
        """Test that split sizes are loaded from config."""
        config = {
            'model': {
                'training': {
                    'test_size': 0.25,
                    'validation_size': 0.15,
                    'stratify': True,
                    'random_state': 123
                }
            }
        }
        
        splitter = DataSplitter(config=config, spark_session=mock_spark_session)
        
        assert splitter.test_size == 0.25
        assert splitter.validation_size == 0.15
        assert splitter.stratify is True
        assert splitter.random_state == 123


class TestDataSplitterValidation:
    """Test suite for DataSplitter validation."""
    
    def test_validate_success(self, mock_spark_session):
        """Test validation with valid config."""
        config = {
            'model': {
                'training': {
                    'test_size': 0.2,
                    'validation_size': 0.1
                }
            }
        }
        
        splitter = DataSplitter(config=config, spark_session=mock_spark_session)
        
        assert splitter.validate() is True
    
    def test_validate_invalid_sizes(self, mock_spark_session):
        """Test validation fails with too large splits."""
        config = {
            'model': {
                'training': {
                    'test_size': 0.6,
                    'validation_size': 0.5  # Total exceeds 1.0
                }
            }
        }
        
        splitter = DataSplitter(config=config, spark_session=mock_spark_session)
        
        assert splitter.validate() is False


class TestDataSplitterSplit:
    """Test suite for split method."""
    
    def test_run_calls_split(self, base_config, mock_spark_session):
        """Test run method calls split."""
        splitter = DataSplitter(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        mock_df = MagicMock()
        
        with patch.object(splitter, 'split') as mock_split:
            mock_split.return_value = DataSplit(MagicMock(), MagicMock())
            
            splitter.run(mock_df, target_column='target')
            
            mock_split.assert_called_once()
