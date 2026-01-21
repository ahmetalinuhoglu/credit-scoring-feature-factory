"""
Tests for Hyperparameter Tuner

Tests HyperparameterTuner with Optuna and Grid Search.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from src.models.hyperparameter_tuner import HyperparameterTuner
from src.core.exceptions import HyperparameterTuningError


class TestHyperparameterTunerInit:
    """Test suite for HyperparameterTuner initialization."""
    
    def test_init_optuna_method(self, model_config):
        """Test HyperparameterTuner with Optuna method."""
        tuner = HyperparameterTuner(
            config=model_config,
            method='optuna'
        )
        
        assert tuner is not None
        assert tuner.method == 'optuna'
    
    def test_init_grid_search_method(self, model_config):
        """Test HyperparameterTuner with Grid Search method."""
        tuner = HyperparameterTuner(
            config=model_config,
            method='grid_search'
        )
        
        assert tuner.method == 'grid_search'
    
    def test_init_with_custom_name(self, model_config):
        """Test HyperparameterTuner with custom name."""
        tuner = HyperparameterTuner(
            config=model_config,
            method='optuna',
            name="CustomTuner"
        )
        
        assert tuner.name == "CustomTuner"
    
    def test_init_best_params_none(self, model_config):
        """Test best_params is None initially."""
        tuner = HyperparameterTuner(config=model_config)
        
        assert tuner.best_params_ is None
        assert tuner.best_score_ is None


class TestHyperparameterTunerValidation:
    """Test suite for validation."""
    
    def test_validate_optuna(self, model_config):
        """Test validation with Optuna method."""
        tuner = HyperparameterTuner(config=model_config, method='optuna')
        
        assert tuner.validate() is True
    
    def test_validate_grid_search(self, model_config):
        """Test validation with Grid Search method."""
        tuner = HyperparameterTuner(config=model_config, method='grid_search')
        
        assert tuner.validate() is True
    
    def test_validate_invalid_method(self, model_config):
        """Test validation fails with invalid method."""
        tuner = HyperparameterTuner(config=model_config, method='invalid')
        
        assert tuner.validate() is False


class TestHyperparameterTunerOptuna:
    """Test suite for Optuna tuning."""
    
    def test_tune_optuna_basic(self, model_config, train_test_data):
        """Test basic Optuna tuning."""
        X_train, X_test, y_train, y_test = train_test_data
        
        tuner = HyperparameterTuner(config=model_config, method='optuna')
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.get_tuning_param_space.return_value = {
            'param1': {'type': 'float', 'low': 0.01, 'high': 1.0},
            'param2': {'type': 'int', 'low': 1, 'high': 10}
        }
        mock_model.model = MagicMock()
        mock_model.model.fit = MagicMock()
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = MagicMock()
            mock_study.best_params = {'param1': 0.5, 'param2': 5}
            mock_study.best_value = 0.85
            mock_study.trials = []
            mock_create_study.return_value = mock_study
            
            result = tuner.tune(
                mock_model, X_train, y_train,
                n_trials=5, timeout=10
            )
            
            mock_create_study.assert_called()
    
    def test_tune_optuna_with_categorical(self, model_config, train_test_data):
        """Test Optuna tuning with categorical parameters."""
        X_train, X_test, y_train, y_test = train_test_data
        
        tuner = HyperparameterTuner(config=model_config, method='optuna')
        
        mock_model = MagicMock()
        mock_model.get_tuning_param_space.return_value = {
            'solver': {'type': 'categorical', 'choices': ['lbfgs', 'saga']}
        }
        mock_model.model = MagicMock()
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = MagicMock()
            mock_study.best_params = {'solver': 'lbfgs'}
            mock_study.best_value = 0.80
            mock_study.trials = []
            mock_create_study.return_value = mock_study
            
            result = tuner.tune(mock_model, X_train, y_train, n_trials=3)
            
            assert result is not None


class TestHyperparameterTunerGridSearch:
    """Test suite for Grid Search tuning."""
    
    def test_tune_grid_search_basic(self, model_config, train_test_data):
        """Test basic Grid Search tuning."""
        X_train, X_test, y_train, y_test = train_test_data
        
        tuner = HyperparameterTuner(config=model_config, method='grid_search')
        
        mock_model = MagicMock()
        mock_model.get_tuning_param_grid.return_value = {
            'param1': [0.1, 0.5, 1.0],
            'param2': [1, 5, 10]
        }
        mock_model.model = MagicMock()
        
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid:
            mock_grid_instance = MagicMock()
            mock_grid_instance.best_params_ = {'param1': 0.5, 'param2': 5}
            mock_grid_instance.best_score_ = 0.82
            mock_grid_instance.cv_results_ = {'mean_test_score': [0.8, 0.82, 0.81]}
            mock_grid.return_value = mock_grid_instance
            
            result = tuner.tune(mock_model, X_train, y_train)
            
            mock_grid.assert_called()


class TestHyperparameterTunerResults:
    """Test suite for tuning results."""
    
    def test_best_params_stored(self, model_config, train_test_data):
        """Test best params are stored after tuning."""
        X_train, X_test, y_train, y_test = train_test_data
        
        tuner = HyperparameterTuner(config=model_config, method='optuna')
        
        mock_model = MagicMock()
        mock_model.get_tuning_param_space.return_value = {
            'param1': {'type': 'float', 'low': 0.01, 'high': 1.0}
        }
        mock_model.model = MagicMock()
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = MagicMock()
            mock_study.best_params = {'param1': 0.5}
            mock_study.best_value = 0.90
            mock_study.trials = []
            mock_create_study.return_value = mock_study
            
            tuner.tune(mock_model, X_train, y_train, n_trials=5)
            
            assert tuner.best_params_ == {'param1': 0.5}
            assert tuner.best_score_ == 0.90
    
    def test_get_tuning_history(self, model_config, train_test_data):
        """Test get_tuning_history method."""
        tuner = HyperparameterTuner(config=model_config, method='optuna')
        
        # Initially should be None
        assert tuner.get_tuning_history() is None
        
        # After setting results
        tuner.study_results_ = [
            {'trial': 0, 'params': {'p': 1}, 'value': 0.8},
            {'trial': 1, 'params': {'p': 2}, 'value': 0.85}
        ]
        
        history = tuner.get_tuning_history()
        
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 2


class TestHyperparameterTunerRun:
    """Test suite for run method."""
    
    def test_run_calls_tune(self, model_config, train_test_data):
        """Test run method calls tune."""
        X_train, X_test, y_train, y_test = train_test_data
        
        tuner = HyperparameterTuner(config=model_config, method='optuna')
        
        mock_model = MagicMock()
        
        with patch.object(tuner, 'tune', return_value={'best_params': {}}) as mock_tune:
            tuner.run(mock_model, X_train, y_train)
            
            mock_tune.assert_called_once()


class TestHyperparameterTunerErrorHandling:
    """Test suite for error handling."""
    
    def test_tune_handles_errors(self, model_config, train_test_data):
        """Test tuning handles errors gracefully."""
        X_train, X_test, y_train, y_test = train_test_data
        
        tuner = HyperparameterTuner(config=model_config, method='optuna')
        
        mock_model = MagicMock()
        mock_model.get_tuning_param_space.side_effect = Exception("Test error")
        
        with pytest.raises(HyperparameterTuningError):
            tuner.tune(mock_model, X_train, y_train)
