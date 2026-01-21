"""
Logistic Regression Model

Logistic Regression classifier wrapper with Grid Search tuning support.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel
from src.core.exceptions import ModelTrainingError


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier for credit scoring.
    
    Features:
    - Sklearn LogisticRegression with regularization
    - Automatic feature scaling
    - Class weight balancing
    - Grid Search hyperparameter tuning
    - Coefficient-based feature importance
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None
    ):
        """
        Initialize Logistic Regression model.
        
        Args:
            config: Model configuration
            name: Optional model name
        """
        super().__init__(config, name or "LogisticRegressionModel")
        
        # Get default parameters
        self.default_params = self.get_config('default_params', {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'penalty': 'l2',
            'C': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        })
        
        self.model = None
        self.scaler = StandardScaler()
        self._use_scaling = True
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'LogisticRegressionModel':
        """
        Fit Logistic Regression model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (not used, for interface consistency)
            y_val: Validation target (not used)
            
        Returns:
            Self
        """
        self._start_execution()
        
        try:
            X, y = self._validate_input(X, y)
            
            # Prepare parameters
            params = self.default_params.copy()
            
            # Apply best params from tuning if available
            if self._best_params:
                params.update(self._best_params)
            
            # Handle solver/penalty compatibility
            params = self._validate_params(params)
            
            # Scale features
            if self._use_scaling:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X.values
            
            # Create and fit model
            self.model = LogisticRegression(**params)
            self.model.fit(X_scaled, y)
            
            # Extract feature importances (coefficients)
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
                self.feature_importances_ = dict(zip(
                    self.feature_names,
                    np.abs(coefficients)  # Absolute values for importance
                ))
                
                # Also store raw coefficients
                self._coefficients = dict(zip(
                    self.feature_names,
                    coefficients
                ))
            
            self.is_fitted = True
            self.logger.info(f"Model fitted on {len(X)} samples")
            
            self._end_execution()
            return self
            
        except Exception as e:
            self._end_execution()
            raise ModelTrainingError(
                f"Logistic Regression training failed: {e}",
                model_name=self.name,
                cause=e
            )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted classes
        """
        if not self.is_fitted:
            raise ModelTrainingError("Model not fitted", model_name=self.name)
        
        X, _ = self._validate_input(X)
        
        if self._use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions for the positive class.
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities for the positive class (1D array)
        """
        if not self.is_fitted:
            raise ModelTrainingError("Model not fitted", model_name=self.name)
        
        X, _ = self._validate_input(X)
        
        if self._use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Return only the positive class probability (column 1)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix solver/penalty compatibility.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Fixed parameters
        """
        solver = params.get('solver', 'lbfgs')
        penalty = params.get('penalty', 'l2')
        
        # Solver-penalty compatibility rules
        compatibility = {
            'lbfgs': ['l2', None],
            'liblinear': ['l1', 'l2'],
            'saga': ['l1', 'l2', 'elasticnet', None],
            'newton-cg': ['l2', None],
            'sag': ['l2', None]
        }
        
        if solver in compatibility:
            valid_penalties = compatibility[solver]
            if penalty not in valid_penalties:
                # Fall back to valid penalty
                new_penalty = valid_penalties[0]
                self.logger.warning(
                    f"Penalty '{penalty}' not compatible with solver '{solver}'. "
                    f"Using '{new_penalty}'"
                )
                params['penalty'] = new_penalty
        
        # Remove l1_ratio if not using elasticnet
        if params.get('penalty') != 'elasticnet' and 'l1_ratio' in params:
            del params['l1_ratio']
        
        return params
    
    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return self.default_params
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Get model coefficients (signed).
        
        Returns:
            Dictionary of feature name to coefficient
        """
        if hasattr(self, '_coefficients'):
            return self._coefficients
        return {}
    
    def get_tuning_param_grid(self) -> Dict[str, List[Any]]:
        """
        Get parameter grid for Grid Search tuning.
        
        Returns:
            Parameter grid dictionary
        """
        return self.get_config('tuning.param_grid', {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        })
