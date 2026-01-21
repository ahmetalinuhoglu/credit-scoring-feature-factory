"""
Tests for XGBoost Model

Tests XGBoost classifier functionality including fit, predict, save/load.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.models.xgboost_model import XGBoostModel
from src.core.exceptions import ModelTrainingError


class TestXGBoostModelInit:
    """Test suite for XGBoostModel initialization."""
    
    def test_init_with_default_params(self, model_config):
        """Test model initialization with default parameters."""
        model = XGBoostModel(config=model_config)
        
        assert model is not None
        assert model.name == "XGBoostModel"
        assert model.is_fitted is False
    
    def test_init_with_custom_name(self, model_config):
        """Test model initialization with custom name."""
        model = XGBoostModel(config=model_config, name="CustomXGB")
        
        assert model.name == "CustomXGB"
    
    def test_default_params_loaded(self, model_config):
        """Test that default parameters are loaded from config."""
        model = XGBoostModel(config=model_config)
        
        assert model.default_params is not None
        assert 'n_estimators' in model.default_params or 'max_depth' in model.default_params


class TestXGBoostModelFit:
    """Test suite for XGBoost model fitting."""
    
    def test_fit_basic(self, model_config, train_test_data):
        """Test basic model fitting."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        assert model.is_fitted is True
    
    def test_fit_with_validation(self, model_config, train_test_data):
        """Test model fitting with validation data."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        
        assert model.is_fitted is True
    
    def test_fit_returns_self(self, model_config, train_test_data):
        """Test that fit returns the model instance."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        result = model.fit(X_train, y_train)
        
        assert result is model
    
    def test_fit_stores_feature_names(self, model_config, train_test_data):
        """Test that feature names are stored after fitting."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_names')
        assert model.feature_names == list(X_train.columns)
    
    def test_fit_with_empty_data_raises(self, model_config):
        """Test that fitting with empty data raises error."""
        model = XGBoostModel(config=model_config)
        
        with pytest.raises((ValueError, ModelTrainingError)):
            model.fit(pd.DataFrame(), pd.Series(dtype=int))


class TestXGBoostModelPredict:
    """Test suite for XGBoost predictions."""
    
    def test_predict_returns_array(self, model_config, train_test_data):
        """Test that predict returns numpy array."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
    
    def test_predict_binary_classes(self, model_config, train_test_data):
        """Test that predictions are binary (0 or 1)."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        unique_values = set(predictions)
        assert unique_values.issubset({0, 1})
    
    def test_predict_proba_returns_probabilities(self, model_config, train_test_data):
        """Test that predict_proba returns probabilities."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        probabilities = model.predict_proba(X_test)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(X_test)
        # All probabilities should be between 0 and 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_predict_without_fit_raises(self, model_config, train_test_data):
        """Test that predicting without fitting raises error."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        
        with pytest.raises(ModelTrainingError):
            model.predict(X_test)


class TestXGBoostFeatureImportance:
    """Test suite for feature importance extraction."""
    
    def test_feature_importance_available(self, model_config, train_test_data):
        """Test that feature importances are available after fitting."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')
        assert model.feature_importances_ is not None
    
    def test_feature_importance_dict(self, model_config, train_test_data):
        """Test feature importance as dictionary."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        
        assert isinstance(importances, dict)
        assert set(importances.keys()) == set(X_train.columns)
    
    def test_feature_importance_values_positive(self, model_config, train_test_data):
        """Test that feature importance values are non-negative."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        for value in model.feature_importances_.values():
            assert value >= 0


class TestXGBoostSaveLoad:
    """Test suite for model persistence."""
    
    def test_save_model(self, model_config, train_test_data):
        """Test saving model to file."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model.pkl"
            model.save(str(save_path))
            
            assert save_path.exists()
    
    def test_load_model(self, model_config, train_test_data):
        """Test loading model from file."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        original_predictions = model.predict(X_test)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model.pkl"
            model.save(str(save_path))
            
            # Load the model
            loaded_model = XGBoostModel(config=model_config)
            loaded_model.load(str(save_path))
            loaded_predictions = loaded_model.predict(X_test)
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_loaded_model_has_attributes(self, model_config, train_test_data):
        """Test that loaded model has all necessary attributes."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model.pkl"
            model.save(str(save_path))
            
            loaded_model = XGBoostModel(config=model_config)
            loaded_model.load(str(save_path))
            
            assert loaded_model.is_fitted is True
            assert hasattr(loaded_model, 'feature_names')


class TestXGBoostParameters:
    """Test suite for hyperparameter handling."""
    
    def test_get_params(self, model_config, train_test_data):
        """Test getting model parameters."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(config=model_config)
        model.fit(X_train, y_train)
        
        params = model.get_params()
        
        assert isinstance(params, dict)
    
    def test_get_tuning_param_space(self, model_config):
        """Test getting parameter space for tuning."""
        model = XGBoostModel(config=model_config)
        
        param_space = model.get_tuning_param_space()
        
        assert isinstance(param_space, dict)
        assert len(param_space) > 0
