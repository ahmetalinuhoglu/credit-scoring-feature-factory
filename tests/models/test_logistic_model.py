"""
Tests for Logistic Regression Model

Tests Logistic Regression classifier functionality including fit, predict, coefficients.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.models.logistic_model import LogisticRegressionModel
from src.core.exceptions import ModelTrainingError


class TestLogisticModelInit:
    """Test suite for LogisticRegressionModel initialization."""
    
    def test_init_with_default_params(self, model_config):
        """Test model initialization with default parameters."""
        model = LogisticRegressionModel(config=model_config)
        
        assert model is not None
        assert model.name == "LogisticRegressionModel"
        assert model.is_fitted is False
    
    def test_init_with_custom_name(self, model_config):
        """Test model initialization with custom name."""
        model = LogisticRegressionModel(config=model_config, name="CustomLR")
        
        assert model.name == "CustomLR"
    
    def test_scaler_initialized(self, model_config):
        """Test that StandardScaler is initialized."""
        model = LogisticRegressionModel(config=model_config)
        
        assert hasattr(model, 'scaler')
        assert model.scaler is not None


class TestLogisticModelFit:
    """Test suite for Logistic Regression model fitting."""
    
    def test_fit_basic(self, model_config, train_test_data):
        """Test basic model fitting."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        assert model.is_fitted is True
    
    def test_fit_returns_self(self, model_config, train_test_data):
        """Test that fit returns the model instance."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        result = model.fit(X_train, y_train)
        
        assert result is model
    
    def test_fit_stores_feature_names(self, model_config, train_test_data):
        """Test that feature names are stored after fitting."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_names')
        assert model.feature_names == list(X_train.columns)
    
    def test_scaler_fitted(self, model_config, train_test_data):
        """Test that scaler is fitted during model fitting."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        # Scaler should have mean_ and scale_ after fitting
        assert hasattr(model.scaler, 'mean_')
        assert hasattr(model.scaler, 'scale_')


class TestLogisticModelPredict:
    """Test suite for Logistic Regression predictions."""
    
    def test_predict_returns_array(self, model_config, train_test_data):
        """Test that predict returns numpy array."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
    
    def test_predict_binary_classes(self, model_config, train_test_data):
        """Test that predictions are binary (0 or 1)."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        unique_values = set(predictions)
        assert unique_values.issubset({0, 1})
    
    def test_predict_proba_returns_probabilities(self, model_config, train_test_data):
        """Test that predict_proba returns probabilities."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        probabilities = model.predict_proba(X_test)
        
        assert isinstance(probabilities, np.ndarray)
        # All probabilities should be between 0 and 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_predict_without_fit_raises(self, model_config, train_test_data):
        """Test that predicting without fitting raises error."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        
        with pytest.raises(ModelTrainingError):
            model.predict(X_test)
    
    def test_scaling_applied_in_predict(self, model_config, train_test_data):
        """Test that scaling is applied during prediction."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        # This should not raise (scaling should be applied automatically)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)


class TestLogisticCoefficients:
    """Test suite for coefficient extraction."""
    
    def test_coefficients_available(self, model_config, train_test_data):
        """Test that coefficients are available after fitting."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        coefficients = model.get_coefficients()
        
        assert coefficients is not None
        assert isinstance(coefficients, dict)
    
    def test_coefficients_match_features(self, model_config, train_test_data):
        """Test that coefficient keys match feature names."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        coefficients = model.get_coefficients()
        
        assert set(coefficients.keys()) == set(X_train.columns)
    
    def test_feature_importances_absolute(self, model_config, train_test_data):
        """Test that feature importances are absolute values of coefficients."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        # Feature importances should be non-negative
        importances = model.feature_importances_
        
        for value in importances.values():
            assert value >= 0


class TestSolverPenaltyCompatibility:
    """Test solver-penalty compatibility handling."""
    
    def test_lbfgs_with_l2(self, model_config, train_test_data):
        """Test lbfgs solver works with L2 penalty."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        # Should not raise
        model.fit(X_train, y_train)
        
        assert model.is_fitted
    
    def test_validate_params_fixes_incompatibility(self, model_config):
        """Test that _validate_params fixes incompatible solver-penalty combinations."""
        model = LogisticRegressionModel(config=model_config)
        
        # lbfgs doesn't support l1
        params = {'solver': 'lbfgs', 'penalty': 'l1'}
        fixed_params = model._validate_params(params)
        
        # Should fix the penalty
        assert fixed_params['penalty'] != 'l1' or fixed_params['solver'] != 'lbfgs'


class TestLogisticTuning:
    """Test suite for hyperparameter tuning support."""
    
    def test_get_tuning_param_grid(self, model_config):
        """Test getting parameter grid for tuning."""
        model = LogisticRegressionModel(config=model_config)
        
        param_grid = model.get_tuning_param_grid()
        
        assert isinstance(param_grid, dict)
        assert 'C' in param_grid
        assert len(param_grid['C']) > 0
    
    def test_get_params(self, model_config, train_test_data):
        """Test getting current model parameters."""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(config=model_config)
        model.fit(X_train, y_train)
        
        params = model.get_params()
        
        assert isinstance(params, dict)
        assert 'C' in params or 'max_iter' in params
