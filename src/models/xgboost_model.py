"""
XGBoost Model

XGBoost classifier wrapper with hyperparameter tuning support.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

import xgboost as xgb

from src.models.base_model import BaseModel
from src.core.exceptions import ModelTrainingError


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for credit scoring.
    
    Features:
    - Native XGBoost with early stopping
    - Automatic class weight balancing
    - Hyperparameter tuning via Optuna
    - Feature importance extraction
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None
    ):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration
            name: Optional model name
        """
        super().__init__(config, name or "XGBoostModel")
        
        # Get default parameters
        self.default_params = self.get_config('default_params', {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        })
        
        # Handle scale_pos_weight
        self.auto_balance = self.default_params.pop('scale_pos_weight', None) == 'auto'
        
        self.model = None
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'XGBoostModel':
        """
        Fit XGBoost model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target
            
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
            
            # Auto-balance classes
            if self.auto_balance:
                neg_count = (y == 0).sum()
                pos_count = (y == 1).sum()
                params['scale_pos_weight'] = neg_count / pos_count
                self.logger.info(f"Auto scale_pos_weight: {params['scale_pos_weight']:.2f}")
            
            # Extract early stopping params
            early_stopping_rounds = params.pop('early_stopping_rounds', None)
            
            # Create model
            self.model = xgb.XGBClassifier(**params)
            
            # Fit with or without validation
            if X_val is not None and y_val is not None and early_stopping_rounds:
                X_val, y_val = self._validate_input(X_val, y_val)
                
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                self.logger.info(f"Best iteration: {self.model.best_iteration}")
            else:
                self.model.fit(X, y)
            
            # Extract feature importances
            self.feature_importances_ = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            self.is_fitted = True
            self.logger.info(f"Model fitted on {len(X)} samples")
            
            self._end_execution()
            return self
            
        except Exception as e:
            self._end_execution()
            raise ModelTrainingError(
                f"XGBoost training failed: {e}",
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
        return self.model.predict(X)
    
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
        # Return only the positive class probability (column 1)
        return self.model.predict_proba(X)[:, 1]
    
    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return self.default_params
    
    def get_tuning_param_space(self) -> Dict[str, Any]:
        """
        Get parameter space for Optuna tuning.
        
        Returns:
            Parameter space dictionary
        """
        return self.get_config('tuning.param_space', {
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'step': 50},
            'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
            'gamma': {'type': 'float', 'low': 0, 'high': 5},
            'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 10.0, 'log': True},
            'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 10.0, 'log': True}
        })
