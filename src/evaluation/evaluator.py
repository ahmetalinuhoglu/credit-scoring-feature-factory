"""
Model Evaluator

Evaluates and compares models using credit scoring metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from src.core.base import PandasComponent
from src.models.base_model import BaseModel
from src.evaluation.metrics import CreditScoringMetrics


class ModelEvaluator(PandasComponent):
    """
    Evaluates models and compares performance.
    
    Features:
    - Evaluate single models
    - Compare multiple models
    - Generate comparison tables
    - Track evaluation history
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
            name: Optional evaluator name
        """
        super().__init__(config, name or "ModelEvaluator")
        
        self.primary_metric = self.get_config('primary_metric', 'gini')
        self.evaluation_history: List[Dict[str, Any]] = []
        
    def validate(self) -> bool:
        return True
    
    def run(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Run evaluation."""
        return self.evaluate(model, X, y)
    
    def evaluate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'test'
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model: Fitted model
            X: Features
            y: True labels
            dataset_name: Name for this dataset (train, test, validation)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self._start_execution()
        
        # Get predictions
        y_pred = model.predict(X)
        y_score = model.predict_proba(X)
        
        # Calculate all metrics
        metrics = CreditScoringMetrics.calculate_all_metrics(
            y.values, y_score, y_pred
        )
        
        # Add lift table
        lift_table = CreditScoringMetrics.lift_table(y.values, y_score)
        
        # Build result
        result = {
            'model_name': model.name,
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(y),
            'positive_ratio': float(y.mean()),
            'metrics': metrics,
            'lift_table': lift_table
        }
        
        # Store in history
        self.evaluation_history.append(result)
        
        self.logger.info(
            f"{model.name} on {dataset_name}: "
            f"Gini={metrics['gini']:.4f}, KS={metrics['ks_statistic']:.4f}, "
            f"AUC={metrics['auc']:.4f}"
        )
        
        self._end_execution()
        
        return result
    
    def evaluate_multiple(
        self,
        models: Dict[str, BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'test'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary of model name to model
            X: Features
            y: True labels
            dataset_name: Dataset name
            
        Returns:
            Dictionary of model name to evaluation results
        """
        results = {}
        
        for name, model in models.items():
            results[name] = self.evaluate(model, X, y, dataset_name)
        
        return results
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare model evaluation results.
        
        Args:
            results: Dictionary of model name to evaluation results
            
        Returns:
            Comparison DataFrame
        """
        comparison = []
        
        for model_name, result in results.items():
            metrics = result['metrics']
            row = {
                'model': model_name,
                'dataset': result['dataset'],
                'gini': metrics['gini'],
                'ks_statistic': metrics['ks_statistic'],
                'auc': metrics['auc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Sort by primary metric
        if self.primary_metric in df.columns:
            df = df.sort_values(self.primary_metric, ascending=False)
        
        return df
    
    def get_best_model(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model.
        
        Args:
            results: Evaluation results
            metric: Metric to use for comparison (defaults to primary_metric)
            
        Returns:
            Tuple of (best model name, its results)
        """
        metric = metric or self.primary_metric
        
        best_name = None
        best_score = -np.inf
        best_result = None
        
        for name, result in results.items():
            score = result['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = name
                best_result = result
        
        self.logger.info(f"Best model: {best_name} with {metric}={best_score:.4f}")
        
        return best_name, best_result
    
    def calculate_psi(
        self,
        model: BaseModel,
        X_dev: pd.DataFrame,
        X_val: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate PSI between development and validation scores.
        
        Args:
            model: Fitted model
            X_dev: Development features
            X_val: Validation features
            
        Returns:
            PSI results
        """
        dev_scores = model.predict_proba(X_dev)
        val_scores = model.predict_proba(X_val)
        
        psi_value, breakdown = CreditScoringMetrics.psi(dev_scores, val_scores)
        
        # Interpret PSI
        if psi_value < 0.1:
            interpretation = "No significant population shift"
        elif psi_value < 0.25:
            interpretation = "Some population shift - monitor closely"
        else:
            interpretation = "Significant population shift - model may need retraining"
        
        self.logger.info(f"PSI: {psi_value:.4f} - {interpretation}")
        
        return {
            'psi': round(psi_value, 4),
            'interpretation': interpretation,
            'breakdown': breakdown
        }
