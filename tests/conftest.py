"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for all test modules including:
- Sample configurations
- Test data (Pandas and Spark DataFrames)
- Temporary directories
- Mock objects
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from typing import Dict, Any

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Base configuration for testing."""
    return {
        'pipeline': {
            'random_state': 42,
            'date_format': '%Y-%m-%d',
            'max_rows_for_pandas': 100000
        },
        'spark': {
            'app_name': 'TestPipeline',
            'master': 'local[2]'
        },
        'logging': {
            'level': 'DEBUG'
        }
    }


@pytest.fixture
def data_config(base_config) -> Dict[str, Any]:
    """Data configuration for testing."""
    config = base_config.copy()
    config['data'] = {
        'sources': {
            'applications': {
                'required_columns': ['application_id', 'customer_id', 'application_date']
            },
            'credit_bureau': {
                'required_columns': ['customer_id', 'product_type', 'amount', 'open_date']
            }
        }
    }
    return config


@pytest.fixture
def model_config(base_config) -> Dict[str, Any]:
    """Model configuration for testing."""
    config = base_config.copy()
    config['model'] = {
        'training': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'stratify': True,
            'random_state': 42
        },
        'xgboost': {
            'default_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'logistic_regression': {
            'default_params': {
                'max_iter': 1000,
                'solver': 'lbfgs',
                'C': 1.0,
                'random_state': 42
            }
        }
    }
    return config


@pytest.fixture
def quality_config(base_config) -> Dict[str, Any]:
    """Quality configuration for testing."""
    config = base_config.copy()
    config['quality'] = {
        'null_checks': {
            'critical_columns': ['application_id', 'customer_id'],
            'warning_threshold': 0.1,
            'error_threshold': 0.5
        },
        'iv_thresholds': {
            'useless': 0.02,
            'weak': 0.1,
            'medium': 0.3,
            'strong': 0.5
        }
    }
    return config


@pytest.fixture
def full_config(base_config, data_config, model_config, quality_config) -> Dict[str, Any]:
    """Complete configuration combining all sections."""
    config = base_config.copy()
    config.update(data_config)
    config.update(model_config)
    config.update(quality_config)
    return config


# ═══════════════════════════════════════════════════════════════
# DATA FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_credit_data() -> pd.DataFrame:
    """Sample credit bureau data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        'application_id': [f'APP_{i:05d}' for i in range(n_samples)],
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'num_accounts': np.random.randint(0, 10, n_samples),
        'num_delinquencies': np.random.poisson(0.5, n_samples),
        'credit_history_months': np.random.randint(0, 360, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    })


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Sample features DataFrame for model testing."""
    np.random.seed(42)
    n_samples = 500
    
    # Create correlated features for realistic testing
    base = np.random.randn(n_samples)
    
    return pd.DataFrame({
        'feature_1': base + np.random.randn(n_samples) * 0.3,
        'feature_2': -base + np.random.randn(n_samples) * 0.3,
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.exponential(1, n_samples),
        'feature_5': np.random.uniform(0, 1, n_samples),
        'target': (base > 0).astype(int)
    })


@pytest.fixture
def binary_classification_data():
    """Generate binary classification data for model tests."""
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    X = pd.DataFrame({
        'feat_1': np.random.randn(n_samples),
        'feat_2': np.random.randn(n_samples),
        'feat_3': np.random.randn(n_samples),
        'feat_4': np.random.randn(n_samples),
        'feat_5': np.random.randn(n_samples),
    })
    
    # Generate target with some signal
    logits = 0.5 * X['feat_1'] - 0.3 * X['feat_2'] + 0.2 * X['feat_3']
    probs = 1 / (1 + np.exp(-logits))
    y = pd.Series((np.random.rand(n_samples) < probs).astype(int))
    
    return X, y


@pytest.fixture
def train_test_data(binary_classification_data):
    """Split data into train and test sets."""
    X, y = binary_classification_data
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════
# MOCK FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_spark_session():
    """Mock SparkSession for unit tests."""
    mock_session = MagicMock()
    mock_session.sparkContext.appName = "TestApp"
    return mock_session


@pytest.fixture
def mock_spark_dataframe(sample_credit_data):
    """Mock Spark DataFrame from Pandas data."""
    mock_df = MagicMock()
    mock_df.toPandas.return_value = sample_credit_data
    mock_df.count.return_value = len(sample_credit_data)
    mock_df.columns = list(sample_credit_data.columns)
    return mock_df


# ═══════════════════════════════════════════════════════════════
# DIRECTORY FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix='test_output_')
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_config_dir(base_config):
    """Create temporary config directory with test configs."""
    temp_dir = tempfile.mkdtemp(prefix='test_config_')
    config_path = Path(temp_dir)
    
    # Write base config
    import yaml
    with open(config_path / 'base_config.yaml', 'w') as f:
        yaml.dump(base_config, f)
    
    yield config_path
    shutil.rmtree(temp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# HELPER FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def assert_almost_equal():
    """Helper for floating point comparisons."""
    def _assert(actual, expected, tolerance=1e-6):
        assert abs(actual - expected) < tolerance, f"{actual} != {expected} (tolerance: {tolerance})"
    return _assert


@pytest.fixture
def prediction_data():
    """Generate prediction data for evaluation tests."""
    np.random.seed(42)
    n = 1000
    
    y_true = np.random.choice([0, 1], n, p=[0.8, 0.2])
    
    # Generate realistic probabilities (higher for actual positives)
    y_prob = np.where(
        y_true == 1,
        np.clip(np.random.beta(5, 2, n), 0.1, 0.99),
        np.clip(np.random.beta(2, 5, n), 0.01, 0.9)
    )
    
    y_pred = (y_prob > 0.5).astype(int)
    
    return y_true, y_prob, y_pred
