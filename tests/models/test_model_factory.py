"""
Tests for Model Factory

Tests ModelFactory pattern for creating model instances.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models.model_factory import ModelFactory
from src.models.base_model import BaseModel
from src.models.xgboost_model import XGBoostModel
from src.models.logistic_model import LogisticRegressionModel


class TestModelFactoryCreate:
    """Test suite for ModelFactory.create method."""
    
    def test_create_xgboost(self, model_config):
        """Test creating XGBoost model."""
        model = ModelFactory.create('xgboost', model_config)
        
        assert isinstance(model, XGBoostModel)
    
    def test_create_logistic_regression(self, model_config):
        """Test creating Logistic Regression model."""
        model = ModelFactory.create('logistic_regression', model_config)
        
        assert isinstance(model, LogisticRegressionModel)
    
    def test_create_with_custom_name(self, model_config):
        """Test creating model with custom name."""
        model = ModelFactory.create('xgboost', model_config, name='MyXGB')
        
        assert model.name == 'MyXGB'
    
    def test_create_case_insensitive(self, model_config):
        """Test model type is case insensitive."""
        model1 = ModelFactory.create('XGBoost', model_config)
        model2 = ModelFactory.create('XGBOOST', model_config)
        model3 = ModelFactory.create('xgboost', model_config)
        
        assert isinstance(model1, XGBoostModel)
        assert isinstance(model2, XGBoostModel)
        assert isinstance(model3, XGBoostModel)
    
    def test_create_unknown_type_raises(self, model_config):
        """Test creating unknown model type raises error."""
        with pytest.raises(ValueError) as excinfo:
            ModelFactory.create('unknown_model', model_config)
        
        assert 'Unknown model type' in str(excinfo.value)


class TestModelFactoryRegister:
    """Test suite for ModelFactory.register method."""
    
    def test_register_new_model(self, model_config):
        """Test registering a new model type."""
        
        class CustomModel(BaseModel):
            def fit(self, X, y, X_val=None, y_val=None):
                return self
            
            def predict(self, X):
                return None
            
            def predict_proba(self, X):
                return None
        
        # Register the model
        ModelFactory.register('custom_model', CustomModel)
        
        # Verify it can be created
        model = ModelFactory.create('custom_model', model_config)
        
        assert isinstance(model, CustomModel)
        
        # Cleanup: remove from registry
        if 'custom_model' in ModelFactory._models:
            del ModelFactory._models['custom_model']
    
    def test_register_non_basemodel_raises(self, model_config):
        """Test registering non-BaseModel class raises error."""
        
        class NotAModel:
            pass
        
        with pytest.raises(TypeError):
            ModelFactory.register('not_a_model', NotAModel)


class TestModelFactoryCreateFromConfig:
    """Test suite for ModelFactory.create_from_config method."""
    
    def test_create_from_config_enabled_models(self):
        """Test creating models from config with enabled flag."""
        config = {
            'model': {
                'models': {
                    'xgboost': {
                        'enabled': True,
                        'default_params': {'n_estimators': 100}
                    },
                    'logistic_regression': {
                        'enabled': False,
                        'default_params': {}
                    }
                }
            }
        }
        
        models = ModelFactory.create_from_config(config)
        
        assert 'xgboost' in models
        assert 'logistic_regression' not in models
    
    def test_create_from_config_multiple_models(self):
        """Test creating multiple models from config."""
        config = {
            'model': {
                'models': {
                    'xgboost': {
                        'enabled': True,
                        'default_params': {}
                    },
                    'logistic_regression': {
                        'enabled': True,
                        'default_params': {}
                    }
                }
            }
        }
        
        models = ModelFactory.create_from_config(config)
        
        assert len(models) == 2
        assert 'xgboost' in models
        assert 'logistic_regression' in models
    
    def test_create_from_config_empty(self):
        """Test creating from config with no models."""
        config = {
            'model': {
                'models': {}
            }
        }
        
        models = ModelFactory.create_from_config(config)
        
        assert len(models) == 0
    
    def test_create_from_config_no_model_section(self):
        """Test creating from config without model section."""
        config = {}
        
        models = ModelFactory.create_from_config(config)
        
        assert len(models) == 0


class TestModelFactoryListModels:
    """Test suite for ModelFactory.list_models method."""
    
    def test_list_models(self):
        """Test listing available models."""
        models = ModelFactory.list_models()
        
        assert isinstance(models, list)
        assert 'xgboost' in models
        assert 'logistic_regression' in models


class TestModelFactoryGetModelClass:
    """Test suite for ModelFactory.get_model_class method."""
    
    def test_get_model_class_xgboost(self):
        """Test getting XGBoost model class."""
        model_class = ModelFactory.get_model_class('xgboost')
        
        assert model_class is XGBoostModel
    
    def test_get_model_class_logistic(self):
        """Test getting Logistic Regression model class."""
        model_class = ModelFactory.get_model_class('logistic_regression')
        
        assert model_class is LogisticRegressionModel
    
    def test_get_model_class_unknown(self):
        """Test getting unknown model class returns None."""
        model_class = ModelFactory.get_model_class('unknown')
        
        assert model_class is None
    
    def test_get_model_class_case_insensitive(self):
        """Test get_model_class is case insensitive."""
        model_class1 = ModelFactory.get_model_class('XGBoost')
        model_class2 = ModelFactory.get_model_class('xgboost')
        
        assert model_class1 is model_class2
