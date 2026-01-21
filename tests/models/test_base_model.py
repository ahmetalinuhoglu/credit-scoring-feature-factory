"""
Tests for Base Model

Tests BaseModel abstract class functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.models.base_model import BaseModel


class DummySklearnModel:
    """Simple sklearn-like model for testing."""
    
    def __init__(self):
        self.is_fitted_ = False
        
    def fit(self, X, y):
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        return np.zeros(len(X))
    
    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
    
    def get_params(self, deep=True):
        return {}
    
    def set_params(self, **params):
        return self


class ConcreteModel(BaseModel):
    """Concrete implementation for testing abstract BaseModel."""
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the model."""
        X, y = self._validate_input(X, y)
        self.model = DummySklearnModel()
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importances_ = {col: 0.1 for col in X.columns}
        return self
    
    def predict(self, X):
        """Generate predictions."""
        if not self.is_fitted:
            from src.core.exceptions import ModelTrainingError
            raise ModelTrainingError("Model not fitted")
        X, _ = self._validate_input(X)
        return np.zeros(len(X))
    
    def predict_proba(self, X):
        """Generate probability predictions."""
        if not self.is_fitted:
            from src.core.exceptions import ModelTrainingError
            raise ModelTrainingError("Model not fitted")
        X, _ = self._validate_input(X)
        return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])


class TestBaseModelInit:
    """Test suite for BaseModel initialization."""
    
    def test_init_basic(self, model_config):
        """Test BaseModel initialization."""
        model = ConcreteModel(config=model_config)
        
        assert model is not None
        assert model.is_fitted is False
    
    def test_init_with_name(self, model_config):
        """Test BaseModel with custom name."""
        model = ConcreteModel(config=model_config, name="TestModel")
        
        assert model.name == "TestModel"
    
    def test_model_none_initially(self, model_config):
        """Test that model is None before fitting."""
        model = ConcreteModel(config=model_config)
        
        assert model.model is None
    
    def test_feature_names_empty(self, model_config):
        """Test feature names are empty initially."""
        model = ConcreteModel(config=model_config)
        
        assert model.feature_names == []
    
    def test_validate(self, model_config):
        """Test validate method."""
        model = ConcreteModel(config=model_config)
        
        assert model.validate() is True


class TestBaseModelFit:
    """Test suite for BaseModel fit functionality."""
    
    def test_fit_sets_is_fitted(self, model_config, train_test_data):
        """Test that fit sets is_fitted flag."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        assert model.is_fitted is True
    
    def test_fit_stores_feature_names(self, model_config, train_test_data):
        """Test that fit stores feature names."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        assert model.feature_names == list(X_train.columns)
    
    def test_fit_returns_self(self, model_config, train_test_data):
        """Test that fit returns self."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        result = model.fit(X_train, y_train)
        
        assert result is model
    
    def test_run_calls_fit(self, model_config, train_test_data):
        """Test that run calls fit."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        result = model.run(X_train, y_train)
        
        assert model.is_fitted is True


class TestBaseModelPredict:
    """Test suite for BaseModel predict functionality."""
    
    def test_predict_after_fit(self, model_config, train_test_data):
        """Test predict works after fit."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
    
    def test_predict_proba_after_fit(self, model_config, train_test_data):
        """Test predict_proba works after fit."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        probabilities = model.predict_proba(X_test)
        
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2


class TestBaseModelFeatureImportance:
    """Test suite for feature importance."""
    
    def test_feature_importance_available(self, model_config, train_test_data):
        """Test feature importance after fit."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        assert model.feature_importances_ is not None
    
    def test_get_feature_importance(self, model_config, train_test_data):
        """Test get_feature_importance method."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        importances = model.get_feature_importance()
        
        assert isinstance(importances, dict)
    
    def test_get_feature_importance_top_n(self, model_config, train_test_data):
        """Test get_feature_importance with top_n."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        importances = model.get_feature_importance(top_n=2)
        
        assert len(importances) <= 2
    
    def test_get_feature_importance_before_fit(self, model_config):
        """Test get_feature_importance before fit returns empty."""
        model = ConcreteModel(config=model_config)
        
        importances = model.get_feature_importance()
        
        assert importances == {}


class TestBaseModelParams:
    """Test suite for parameter handling."""
    
    def test_get_params(self, model_config, train_test_data):
        """Test get_params method."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        params = model.get_params()
        
        assert isinstance(params, dict)
    
    def test_set_params(self, model_config, train_test_data):
        """Test set_params method."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        result = model.set_params(test_param='value')
        
        assert result is model
    
    def test_best_params_property(self, model_config):
        """Test best_params property."""
        model = ConcreteModel(config=model_config)
        
        assert model.best_params is None
        
        model.best_params = {'param': 'value'}
        
        assert model.best_params == {'param': 'value'}


class TestBaseModelSaveLoad:
    """Test suite for model persistence."""
    
    def test_save_model(self, model_config, train_test_data):
        """Test saving model."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model.pkl"
            model.save(str(save_path))
            
            assert save_path.exists()
    
    def test_load_model(self, model_config, train_test_data):
        """Test loading model."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model.pkl"
            model.save(str(save_path))
            
            loaded_model = ConcreteModel(config=model_config)
            loaded_model.load(str(save_path))
            
            assert loaded_model.is_fitted is True


class TestBaseModelValidation:
    """Test suite for input validation."""
    
    def test_validate_input_dataframe(self, model_config):
        """Test validate_input with DataFrame."""
        model = ConcreteModel(config=model_config)
        
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        
        X_val, y_val = model._validate_input(X, y)
        
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(y_val, pd.Series)
    
    def test_validate_input_length_mismatch(self, model_config):
        """Test validate_input raises on length mismatch."""
        model = ConcreteModel(config=model_config)
        model.is_fitted = False
        model.feature_names = []
        
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0, 1])  # Wrong length
        
        with pytest.raises(ValueError):
            model._validate_input(X, y)
    
    def test_validate_input_missing_features(self, model_config, train_test_data):
        """Test validate_input raises on missing features."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = ConcreteModel(config=model_config)
        model.fit(X_train, y_train)
        
        # Create test data with missing features
        X_missing = X_test.drop(columns=[X_test.columns[0]])
        
        with pytest.raises(ValueError):
            model._validate_input(X_missing)
