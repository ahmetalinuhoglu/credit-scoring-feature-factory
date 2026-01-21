"""
Model Factory

Factory pattern for creating model instances.
"""

from typing import Any, Dict, Optional, Type

from src.models.base_model import BaseModel
from src.models.xgboost_model import XGBoostModel
from src.models.logistic_model import LogisticRegressionModel


class ModelFactory:
    """
    Factory for creating model instances.
    
    Supports dynamic model registration and creation.
    """
    
    # Registry of available models
    _models: Dict[str, Type[BaseModel]] = {
        'xgboost': XGBoostModel,
        'logistic_regression': LogisticRegressionModel
    }
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type.
        
        Args:
            name: Model name for lookup
            model_class: Model class
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must be a subclass of BaseModel")
        cls._models[name] = model_class
    
    @classmethod
    def create(
        cls,
        model_type: str,
        config: Dict[str, Any],
        name: Optional[str] = None
    ) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model ('xgboost', 'logistic_regression')
            config: Model configuration
            name: Optional instance name
            
        Returns:
            Model instance
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._models.keys())}"
            )
        
        model_class = cls._models[model_type]
        return model_class(config, name)
    
    @classmethod
    def create_from_config(
        cls,
        full_config: Dict[str, Any]
    ) -> Dict[str, BaseModel]:
        """
        Create all enabled models from configuration.
        
        Args:
            full_config: Full configuration dictionary
            
        Returns:
            Dictionary of model name to model instance
        """
        models = {}
        models_config = full_config.get('model', {}).get('models', {})
        
        for model_type, model_config in models_config.items():
            if not model_config.get('enabled', False):
                continue
            
            model = cls.create(model_type, model_config)
            models[model_type] = model
        
        return models
    
    @classmethod
    def list_models(cls) -> list:
        """List all available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Optional[Type[BaseModel]]:
        """Get model class by type."""
        return cls._models.get(model_type.lower())
