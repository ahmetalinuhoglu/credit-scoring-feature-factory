"""
Tests for Model Evaluator

Tests ModelEvaluator for model performance evaluation.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from src.evaluation.evaluator import ModelEvaluator


class TestModelEvaluatorInit:
    """Test suite for ModelEvaluator initialization."""
    
    def test_init_basic(self, base_config):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(config=base_config)
        
        assert evaluator is not None
        assert evaluator.name == "ModelEvaluator"
    
    def test_init_with_custom_name(self, base_config):
        """Test ModelEvaluator with custom name."""
        evaluator = ModelEvaluator(config=base_config, name="CustomEvaluator")
        
        assert evaluator.name == "CustomEvaluator"
    
    def test_init_primary_metric(self, base_config):
        """Test primary metric is loaded from config."""
        config = base_config.copy()
        config['primary_metric'] = 'auc'
        
        evaluator = ModelEvaluator(config=config)
        
        assert evaluator.primary_metric == 'auc'
    
    def test_init_default_primary_metric(self, base_config):
        """Test default primary metric."""
        evaluator = ModelEvaluator(config=base_config)
        
        assert evaluator.primary_metric == 'gini'
    
    def test_validate(self, base_config):
        """Test validate method."""
        evaluator = ModelEvaluator(config=base_config)
        
        assert evaluator.validate() is True


class TestModelEvaluatorEvaluate:
    """Test suite for evaluate method."""
    
    def test_evaluate_single_model(self, base_config, train_test_data):
        """Test evaluating a single model."""
        X_train, X_test, y_train, y_test = train_test_data
        
        evaluator = ModelEvaluator(config=base_config)
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.name = "TestModel"
        mock_model.predict.return_value = np.random.choice([0, 1], len(y_test))
        mock_model.predict_proba.return_value = np.random.rand(len(y_test), 2)
        
        with patch('src.evaluation.evaluator.CreditScoringMetrics') as mock_metrics:
            mock_metrics.calculate_all_metrics.return_value = {
                'gini': 0.45,
                'ks_statistic': 0.30,
                'auc': 0.72,
                'accuracy': 0.85,
                'precision': 0.70,
                'recall': 0.60,
                'f1_score': 0.65
            }
            mock_metrics.lift_table.return_value = pd.DataFrame()
            
            result = evaluator.evaluate(mock_model, X_test, y_test)
            
            assert isinstance(result, dict)
            assert result['model_name'] == "TestModel"
            assert 'metrics' in result
    
    def test_evaluate_stores_in_history(self, base_config, train_test_data):
        """Test that evaluation is stored in history."""
        X_train, X_test, y_train, y_test = train_test_data
        
        evaluator = ModelEvaluator(config=base_config)
        
        mock_model = MagicMock()
        mock_model.name = "TestModel"
        mock_model.predict.return_value = np.zeros(len(y_test))
        mock_model.predict_proba.return_value = np.column_stack([
            np.ones(len(y_test)) * 0.5,
            np.ones(len(y_test)) * 0.5
        ])
        
        with patch('src.evaluation.evaluator.CreditScoringMetrics') as mock_metrics:
            mock_metrics.calculate_all_metrics.return_value = {
                'gini': 0.40, 'ks_statistic': 0.25, 'ks_threshold': 0.4,
                'auc': 0.70, 'accuracy': 0.80, 'precision': 0.65,
                'recall': 0.55, 'f1_score': 0.60, 'log_loss': 0.3,
                'confusion_matrix': {'tn': 40, 'fp': 10, 'fn': 15, 'tp': 35},
                'classification_report': {}
            }
            mock_metrics.lift_table.return_value = pd.DataFrame()
            
            evaluator.evaluate(mock_model, X_test, y_test)
            
            assert len(evaluator.evaluation_history) == 1
    
    def test_run_calls_evaluate(self, base_config, train_test_data):
        """Test run method calls evaluate."""
        X_train, X_test, y_train, y_test = train_test_data
        
        evaluator = ModelEvaluator(config=base_config)
        mock_model = MagicMock()
        
        with patch.object(evaluator, 'evaluate', return_value={}) as mock_eval:
            evaluator.run(mock_model, X_test, y_test)
            
            mock_eval.assert_called_once()


class TestModelEvaluatorMultiple:
    """Test suite for evaluate_multiple method."""
    
    def test_evaluate_multiple_models(self, base_config, train_test_data):
        """Test evaluating multiple models."""
        X_train, X_test, y_train, y_test = train_test_data
        
        evaluator = ModelEvaluator(config=base_config)
        
        # Create mock models
        mock_model1 = MagicMock()
        mock_model1.name = "Model1"
        mock_model2 = MagicMock()
        mock_model2.name = "Model2"
        
        models = {'model1': mock_model1, 'model2': mock_model2}
        
        with patch.object(evaluator, 'evaluate') as mock_eval:
            mock_eval.return_value = {'metrics': {'gini': 0.5}}
            
            results = evaluator.evaluate_multiple(models, X_test, y_test)
            
            assert 'model1' in results
            assert 'model2' in results
            assert mock_eval.call_count == 2


class TestModelEvaluatorCompare:
    """Test suite for compare_models method."""
    
    def test_compare_models(self, base_config):
        """Test model comparison."""
        evaluator = ModelEvaluator(config=base_config)
        
        results = {
            'model1': {
                'dataset': 'test',
                'metrics': {
                    'gini': 0.45,
                    'ks_statistic': 0.30,
                    'auc': 0.72,
                    'accuracy': 0.85,
                    'precision': 0.70,
                    'recall': 0.60,
                    'f1_score': 0.65
                }
            },
            'model2': {
                'dataset': 'test',
                'metrics': {
                    'gini': 0.50,
                    'ks_statistic': 0.35,
                    'auc': 0.75,
                    'accuracy': 0.87,
                    'precision': 0.72,
                    'recall': 0.62,
                    'f1_score': 0.67
                }
            }
        }
        
        comparison = evaluator.compare_models(results)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'gini' in comparison.columns
    
    def test_compare_models_sorted_by_primary_metric(self, base_config):
        """Test comparison is sorted by primary metric."""
        evaluator = ModelEvaluator(config=base_config)
        evaluator.primary_metric = 'gini'
        
        results = {
            'worse_model': {
                'dataset': 'test',
                'metrics': {'gini': 0.30, 'ks_statistic': 0.20, 'auc': 0.65,
                           'accuracy': 0.80, 'precision': 0.60, 'recall': 0.55, 'f1_score': 0.57}
            },
            'better_model': {
                'dataset': 'test',
                'metrics': {'gini': 0.55, 'ks_statistic': 0.40, 'auc': 0.77,
                           'accuracy': 0.88, 'precision': 0.75, 'recall': 0.65, 'f1_score': 0.70}
            }
        }
        
        comparison = evaluator.compare_models(results)
        
        # Better model should be first (higher gini)
        assert comparison.iloc[0]['model'] == 'better_model'


class TestModelEvaluatorBestModel:
    """Test suite for get_best_model method."""
    
    def test_get_best_model(self, base_config):
        """Test getting best model."""
        evaluator = ModelEvaluator(config=base_config)
        
        results = {
            'model1': {'metrics': {'gini': 0.45}},
            'model2': {'metrics': {'gini': 0.55}},
            'model3': {'metrics': {'gini': 0.40}}
        }
        
        best_name, best_result = evaluator.get_best_model(results)
        
        assert best_name == 'model2'
        assert best_result['metrics']['gini'] == 0.55
    
    def test_get_best_model_custom_metric(self, base_config):
        """Test getting best model with custom metric."""
        evaluator = ModelEvaluator(config=base_config)
        
        results = {
            'model1': {'metrics': {'gini': 0.45, 'auc': 0.80}},
            'model2': {'metrics': {'gini': 0.55, 'auc': 0.75}}
        }
        
        best_name, best_result = evaluator.get_best_model(results, metric='auc')
        
        assert best_name == 'model1'  # model1 has higher AUC


class TestModelEvaluatorPSI:
    """Test suite for PSI calculation."""
    
    def test_calculate_psi(self, base_config, train_test_data):
        """Test PSI calculation."""
        X_train, X_test, y_train, y_test = train_test_data
        
        evaluator = ModelEvaluator(config=base_config)
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(len(X_train), 2)
        
        with patch('src.evaluation.evaluator.CreditScoringMetrics') as mock_metrics:
            mock_metrics.psi.return_value = (0.05, {'bin1': 0.01, 'bin2': 0.02})
            
            result = evaluator.calculate_psi(mock_model, X_train, X_test)
            
            assert 'psi' in result
            assert 'interpretation' in result
            assert result['psi'] == 0.05
    
    def test_psi_interpretation_no_shift(self, base_config, train_test_data):
        """Test PSI interpretation for no significant shift."""
        X_train, X_test, y_train, y_test = train_test_data
        
        evaluator = ModelEvaluator(config=base_config)
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(len(X_train), 2)
        
        with patch('src.evaluation.evaluator.CreditScoringMetrics') as mock_metrics:
            mock_metrics.psi.return_value = (0.05, {})
            
            result = evaluator.calculate_psi(mock_model, X_train, X_test)
            
            assert 'No significant' in result['interpretation']
    
    def test_psi_interpretation_significant_shift(self, base_config, train_test_data):
        """Test PSI interpretation for significant shift."""
        X_train, X_test, y_train, y_test = train_test_data
        
        evaluator = ModelEvaluator(config=base_config)
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(len(X_train), 2)
        
        with patch('src.evaluation.evaluator.CreditScoringMetrics') as mock_metrics:
            mock_metrics.psi.return_value = (0.30, {})
            
            result = evaluator.calculate_psi(mock_model, X_train, X_test)
            
            assert 'Significant' in result['interpretation']
