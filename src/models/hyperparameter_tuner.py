"""
Hyperparameter Tuner

Implements Optuna and Grid Search tuning for models.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.core.base import PandasComponent
from src.core.exceptions import HyperparameterTuningError
from src.models.base_model import BaseModel


class HyperparameterTuner(PandasComponent):
    """
    Hyperparameter tuning using Optuna or Grid Search.
    
    Supports:
    - Optuna Bayesian optimization
    - Grid Search
    - Cross-validation
    - Custom objective functions
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        method: str = 'optuna',
        name: Optional[str] = None
    ):
        """
        Initialize the tuner.
        
        Args:
            config: Tuning configuration
            method: Tuning method ('optuna' or 'grid_search')
            name: Optional tuner name
        """
        super().__init__(config, name or "HyperparameterTuner")
        
        self.method = method.lower()
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.study_results_: Optional[List[Dict]] = None
        
    def validate(self) -> bool:
        if self.method not in ['optuna', 'grid_search']:
            self.logger.error(f"Invalid method: {self.method}")
            return False
        return True
    
    def run(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> Dict[str, Any]:
        """Run tuning."""
        return self.tune(model, X, y, **kwargs)
    
    def tune(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'roc_auc',
        n_trials: int = 100,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for a model.
        
        Args:
            model: Model to tune
            X: Training features
            y: Training target
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_trials: Number of Optuna trials (ignored for grid search)
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best params and score
        """
        self._start_execution()
        
        self.logger.info(f"Starting {self.method} tuning with {cv}-fold CV")
        
        try:
            if self.method == 'optuna':
                result = self._tune_optuna(model, X, y, cv, scoring, n_trials, timeout)
            else:
                result = self._tune_grid_search(model, X, y, cv, scoring)
            
            self.best_params_ = result['best_params']
            self.best_score_ = result['best_score']
            
            self.logger.info(f"Best score: {self.best_score_:.4f}")
            self.logger.info(f"Best params: {self.best_params_}")
            
            self._end_execution()
            return result
            
        except Exception as e:
            self._end_execution()
            raise HyperparameterTuningError(
                f"Tuning failed: {e}",
                cause=e
            )
    
    def _tune_optuna(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int,
        scoring: str,
        n_trials: int,
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """Tune using Optuna."""
        import optuna
        from optuna.samplers import TPESampler
        
        # Get parameter space
        param_space = model.get_tuning_param_space()
        
        # Create objective function
        def objective(trial: optuna.Trial) -> float:
            params = {}
            
            for param_name, param_config in param_space.items():
                param_type = param_config.get('type', 'float')
                
                if param_type == 'int':
                    step = param_config.get('step', 1)
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=step
                    )
                elif param_type == 'float':
                    log = param_config.get('log', False)
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=log
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Set params and evaluate
            model.set_params(**params)
            
            cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(
                model.model, X, y,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1
            )
            
            return scores.mean()
        
        # Run optimization
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Store results
        self.study_results_ = [
            {
                'trial': t.number,
                'params': t.params,
                'value': t.value
            }
            for t in study.trials
        ]
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'method': 'optuna'
        }
    
    def _tune_grid_search(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int,
        scoring: str
    ) -> Dict[str, Any]:
        """Tune using Grid Search."""
        from sklearn.model_selection import GridSearchCV
        
        # Get parameter grid
        param_grid = model.get_tuning_param_grid()
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        self.logger.info(f"Grid search with {total_combinations} combinations")
        
        # Run grid search
        cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            model.model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        # Store results
        results_df = pd.DataFrame(grid_search.cv_results_)
        self.study_results_ = results_df.to_dict('records')
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'n_combinations': total_combinations,
            'method': 'grid_search'
        }
    
    def get_tuning_history(self) -> Optional[pd.DataFrame]:
        """Get tuning history as DataFrame."""
        if self.study_results_ is None:
            return None
        return pd.DataFrame(self.study_results_)
