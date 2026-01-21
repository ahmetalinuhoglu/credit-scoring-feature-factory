"""
Tests for Report Generator

Tests ReportGenerator for HTML evaluation reports.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path
import pandas as pd

from src.evaluation.report_generator import ReportGenerator


class TestReportGeneratorInit:
    """Test suite for ReportGenerator initialization."""
    
    def test_init_basic(self, base_config):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator(config=base_config)
        
        assert generator is not None
        assert generator.name == "ReportGenerator"
    
    def test_init_with_custom_name(self, base_config):
        """Test ReportGenerator with custom name."""
        generator = ReportGenerator(config=base_config, name="CustomGenerator")
        
        assert generator.name == "CustomGenerator"
    
    def test_init_with_output_dir(self, base_config, temp_output_dir):
        """Test ReportGenerator with output directory."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        assert generator.output_dir == temp_output_dir
    
    def test_validate(self, base_config):
        """Test validate method."""
        generator = ReportGenerator(config=base_config)
        
        assert generator.validate() is True


def create_full_evaluation_result(model_name: str, gini: float = 0.45) -> dict:
    """Helper to create a complete evaluation result with all required fields."""
    return {
        'model_name': model_name,
        'dataset': 'test',
        'sample_size': 1000,
        'positive_ratio': 0.15,
        'metrics': {
            'gini': gini,
            'ks_statistic': 0.30,
            'ks_threshold': 0.45,
            'auc': 0.72,
            'accuracy': 0.85,
            'precision': 0.60,
            'recall': 0.55,
            'f1_score': 0.57,
            'log_loss': 0.35,
            'confusion_matrix': {
                'tn': 800,
                'fp': 50,
                'fn': 100,
                'tp': 50
            },
            'classification_report': {}
        },
        'lift_table': pd.DataFrame({
            'decile': range(1, 11),
            'count': [100] * 10,
            'bads': [15, 12, 10, 8, 6, 5, 4, 3, 2, 1],
            'bad_rate': [0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
            'lift': [2.0, 1.6, 1.3, 1.0, 0.8, 0.67, 0.53, 0.4, 0.27, 0.13],
            'cum_lift': [2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85],
            'capture_rate': [0.2, 0.35, 0.48, 0.6, 0.68, 0.75, 0.8, 0.85, 0.9, 1.0]
        })
    }


class TestReportGeneratorGenerate:
    """Test suite for generate method."""
    
    def test_generate_basic(self, base_config, temp_output_dir):
        """Test basic report generation."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        evaluation_results = {
            'model1': create_full_evaluation_result('model1')
        }
        
        result = generator.generate(evaluation_results)
        
        assert result is not None
        assert isinstance(result, (str, Path))
        assert Path(result).exists()
    
    def test_generate_with_comparison(self, base_config, temp_output_dir):
        """Test report generation with comparison DataFrame."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        evaluation_results = {
            'model1': create_full_evaluation_result('model1', gini=0.40)
        }
        
        comparison_df = pd.DataFrame({
            'model': ['model1'],
            'gini': [0.40],
            'ks_statistic': [0.28],
            'auc': [0.70],
            'precision': [0.55],
            'recall': [0.50],
            'f1_score': [0.52]
        })
        
        result = generator.generate(
            evaluation_results,
            comparison_df=comparison_df
        )
        
        assert result is not None
        assert Path(result).exists()
    
    def test_generate_with_feature_importance(self, base_config, temp_output_dir):
        """Test report generation with feature importances."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        evaluation_results = {
            'model1': create_full_evaluation_result('model1', gini=0.50)
        }
        
        feature_importances = {
            'model1': {
                'feature_a': 0.35,
                'feature_b': 0.25,
                'feature_c': 0.20
            }
        }
        
        result = generator.generate(
            evaluation_results,
            feature_importances=feature_importances
        )
        
        assert result is not None
        assert Path(result).exists()
    
    def test_run_calls_generate(self, base_config, temp_output_dir):
        """Test run method calls generate."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        evaluation_results = {'model1': create_full_evaluation_result('model1')}
        
        with patch.object(generator, 'generate', return_value='path') as mock_gen:
            generator.run(evaluation_results)
            
            mock_gen.assert_called_once()


class TestReportGeneratorHTML:
    """Test suite for HTML report building."""
    
    def test_html_contains_model_info(self, base_config, temp_output_dir):
        """Test HTML report contains model information."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        evaluation_results = {
            'test_model': create_full_evaluation_result('test_model', gini=0.55)
        }
        
        result_path = generator.generate(evaluation_results, title="Test Report")
        
        content = Path(result_path).read_text()
        assert 'test_model' in content
        assert '0.55' in content or 'gini' in content.lower()
    
    def test_html_contains_custom_title(self, base_config, temp_output_dir):
        """Test HTML report contains custom title."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        evaluation_results = {
            'model1': create_full_evaluation_result('model1', gini=0.40)
        }
        
        result_path = generator.generate(
            evaluation_results,
            title="My Custom Report Title"
        )
        
        content = Path(result_path).read_text()
        assert 'My Custom Report Title' in content


class TestReportGeneratorMultipleModels:
    """Test suite for reports with multiple models."""
    
    def test_report_with_multiple_models(self, base_config, temp_output_dir):
        """Test report generation with multiple models."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        evaluation_results = {
            'xgboost': create_full_evaluation_result('xgboost', gini=0.55),
            'logistic': create_full_evaluation_result('logistic', gini=0.42)
        }
        
        comparison_df = pd.DataFrame({
            'model': ['xgboost', 'logistic'],
            'gini': [0.55, 0.42],
            'ks_statistic': [0.35, 0.25],
            'auc': [0.78, 0.71],
            'precision': [0.65, 0.55],
            'recall': [0.60, 0.50],
            'f1_score': [0.62, 0.52]
        })
        
        result_path = generator.generate(
            evaluation_results,
            comparison_df=comparison_df
        )
        
        content = Path(result_path).read_text()
        assert 'xgboost' in content
        assert 'logistic' in content


class TestReportGeneratorOutput:
    """Test suite for output handling."""
    
    def test_creates_output_directory(self, base_config):
        """Test report generator creates output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'reports' / 'evaluation'
            
            generator = ReportGenerator(
                config=base_config,
                output_dir=str(output_path)
            )
            
            evaluation_results = {
                'model1': create_full_evaluation_result('model1', gini=0.30)
            }
            
            generator.generate(evaluation_results)
            
            assert output_path.exists()
    
    def test_report_file_created(self, base_config, temp_output_dir):
        """Test that report file is actually created."""
        generator = ReportGenerator(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        evaluation_results = {
            'model1': create_full_evaluation_result('model1', gini=0.35)
        }
        
        result_path = generator.generate(evaluation_results)
        
        assert Path(result_path).exists()
        assert Path(result_path).suffix == '.html'
