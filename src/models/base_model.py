"""
Base Model

Abstract base class for all machine learning models.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.base import PandasComponent


class BaseModel(PandasComponent):
    """
    Abstract base class for ML models.
    
    All models (XGBoost, Logistic Regression, etc.) inherit from this class.
    Provides consistent interface for training, prediction, and evaluation.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None
    ):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
            name: Optional model name
        """
        super().__init__(config, name)
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[Dict[str, float]] = None
        self._best_params: Optional[Dict[str, Any]] = None
        
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted classes
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        pass
    
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> 'BaseModel':
        """Run is implemented as fit."""
        return self.fit(X, y, **kwargs)
    
    def validate(self) -> bool:
        """Validate model configuration."""
        return True
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return self.get_config('default_params', {})
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self
    
    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters from tuning."""
        return self._best_params
    
    @best_params.setter
    def best_params(self, params: Dict[str, Any]) -> None:
        """Set best parameters."""
        self._best_params = params
    
    def get_feature_importance(
        self,
        top_n: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            top_n: Return only top N features
            
        Returns:
            Dictionary of feature name to importance score
        """
        if self.feature_importances_ is None:
            return {}
        
        sorted_features = dict(
            sorted(
                self.feature_importances_.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        )
        
        if top_n:
            return dict(list(sorted_features.items())[:top_n])
        
        return sorted_features
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        import joblib
        
        artifact = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_,
            'best_params': self._best_params,
            'config': self.config
        }
        
        joblib.dump(artifact, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            Self
        """
        import joblib
        
        artifact = joblib.load(path)
        
        self.model = artifact['model']
        self.feature_names = artifact['feature_names']
        self.feature_importances_ = artifact['feature_importances']
        self._best_params = artifact['best_params']
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from {path}")
        
        return self
    
    def _validate_input(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Validate and prepare input data.
        
        Args:
            X: Features
            y: Target (optional)
            
        Returns:
            Validated (X, y) tuple
        """
        # Ensure X is DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Store feature names on first fit
        if not self.feature_names:
            self.feature_names = list(X.columns)
        
        # Validate columns match
        if self.is_fitted:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")
            
            # Ensure column order
            X = X[self.feature_names]
        
        # Validate y if provided
        if y is not None:
            if not isinstance(y, pd.Series):
                y = pd.Series(y)
            
            if len(X) != len(y):
                raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        return X, y
