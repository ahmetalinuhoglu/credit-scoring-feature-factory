"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for all test modules including:
- Sample configurations (Pydantic-based)
- Synthetic test data with known properties
- Pre-split train/test/OOT data
- Temporary directories for output testing
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ===================================================================
# CONFIGURATION FIXTURES
# ===================================================================

@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Minimal valid config dict that can be loaded into PipelineConfig."""
    return {
        "data": {
            "input_path": "data/sample/sample_features.parquet",
            "target_column": "target",
            "date_column": "date",
            "id_columns": ["application_id", "customer_id"],
            "exclude_columns": ["applicant_type"],
        },
        "splitting": {
            "train_end_date": "2024-06-30",
            "test_size": 0.20,
            "stratify": True,
        },
        "steps": {
            "constant": {"enabled": True, "min_unique_values": 2},
            "missing": {"enabled": True, "threshold": 0.70},
            "iv": {
                "enabled": True,
                "min_iv": 0.02,
                "max_iv": 0.50,
                "n_bins": 10,
                "min_samples_per_bin": 50,
            },
            "psi": {
                "enabled": True,
                "threshold": 0.25,
                "n_bins": 10,
                "checks": [
                    {"type": "quarterly"},
                    {"type": "yearly"},
                    {"type": "consecutive"},
                ],
            },
            "correlation": {"enabled": True, "threshold": 0.90, "method": "pearson"},
            "selection": {
                "enabled": True,
                "method": "forward",
                "auc_threshold": 0.0001,
                "max_features": None,
            },
        },
        "model": {
            "algorithm": "xgboost",
            "params": {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_estimators": 50,
                "early_stopping_rounds": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
        "evaluation": {
            "metrics": ["auc", "gini", "ks"],
            "precision_at_k": [5, 10, 20],
            "n_deciles": 10,
            "calculate_score_psi": True,
        },
        "validation": {
            "enabled": True,
            "checks": {
                "min_auc": 0.65,
                "max_overfit_gap": 0.05,
                "max_oot_degradation": 0.08,
                "max_score_psi": 0.25,
                "max_feature_concentration": 0.50,
                "min_oot_samples": 30,
                "check_monotonicity": True,
            },
        },
        "output": {
            "base_dir": "outputs/model_development",
            "save_step_results": True,
            "save_model": True,
            "save_split_indices": True,
            "generate_excel": True,
            "save_correlation_matrix": True,
        },
        "reproducibility": {
            "global_seed": 42,
            "save_config": True,
            "save_metadata": True,
            "log_level": "DEBUG",
        },
    }


@pytest.fixture
def sample_config(sample_config_dict):
    """Create a PipelineConfig from the sample dict."""
    from src.config.schema import PipelineConfig

    return PipelineConfig(**sample_config_dict)


# ===================================================================
# DATA FIXTURES
# ===================================================================

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Small synthetic DataFrame (100 rows, 20 features) with known properties.

    Properties:
    - 100 rows, ~20% bad rate
    - Dates spanning 2023-01-01 to 2024-02-09 (4D frequency)
    - 2 constant features: const_feat_1 (always 1.0), const_feat_2 (always 0)
    - 2 high-missing features: high_missing_1 (~80%), high_missing_2 (~75%)
    - 2 low-IV features (random noise)
    - 2 high-IV features (correlated with target)
    - 2 highly correlated features (r > 0.99)
    - 8 normal features with varying predictive power
    """
    np.random.seed(42)
    n = 100

    # Create target with ~20% bad rate
    target = np.zeros(n)
    target[:20] = 1
    np.random.shuffle(target)

    # Create dates spanning 4 quarters
    dates = pd.date_range("2023-01-01", periods=n, freq="4D")

    data = {
        "application_id": range(1, n + 1),
        "customer_id": range(1001, 1001 + n),
        "applicant_type": ["individual"] * n,
        "date": dates,
        "target": target.astype(int),
    }

    # 2 constant features (to test constant filter)
    data["const_feat_1"] = 1.0
    data["const_feat_2"] = 0

    # 2 high-missing features (to test missing filter)
    data["high_missing_1"] = np.where(
        np.random.rand(n) < 0.80, np.nan, np.random.randn(n)
    )
    data["high_missing_2"] = np.where(
        np.random.rand(n) < 0.75, np.nan, np.random.randn(n)
    )

    # 2 low-IV features (random noise, no relationship to target)
    data["low_iv_1"] = np.random.randn(n)
    data["low_iv_2"] = np.random.randn(n)

    # 2 high-IV features (strongly correlated with target)
    data["high_iv_1"] = target * 2 + np.random.randn(n) * 0.5
    data["high_iv_2"] = target * 1.5 + np.random.randn(n) * 0.5

    # 2 highly correlated features (r > 0.99 with each other)
    base = np.random.randn(n)
    data["corr_feat_1"] = base + np.random.randn(n) * 0.01
    data["corr_feat_2"] = base + np.random.randn(n) * 0.01

    # 8 normal features with varying predictive power
    for i in range(8):
        data[f"normal_feat_{i+1}"] = target * (0.5 + i * 0.1) + np.random.randn(n)

    return pd.DataFrame(data)


@pytest.fixture
def feature_columns() -> List[str]:
    """List of feature column names in sample_data (excluding id/target/date/exclude)."""
    return [
        "const_feat_1", "const_feat_2",
        "high_missing_1", "high_missing_2",
        "low_iv_1", "low_iv_2",
        "high_iv_1", "high_iv_2",
        "corr_feat_1", "corr_feat_2",
    ] + [f"normal_feat_{i+1}" for i in range(8)]


@pytest.fixture
def sample_X(sample_data, feature_columns) -> pd.DataFrame:
    """Feature matrix from sample_data."""
    return sample_data[feature_columns]


@pytest.fixture
def sample_y(sample_data) -> pd.Series:
    """Target series from sample_data."""
    return sample_data["target"]


@pytest.fixture
def sample_split_data(sample_data):
    """Pre-split train/test/oot from sample_data.

    Uses date 2023-07-01 as cutoff (first ~45 rows are train period).
    """
    df = sample_data.copy()
    df["date"] = pd.to_datetime(df["date"])

    cutoff = pd.Timestamp("2023-07-01")
    train_period = df[df["date"] <= cutoff]
    oot_period = df[df["date"] > cutoff]

    # 80/20 split of train period
    split_idx = int(len(train_period) * 0.8)
    train = train_period.iloc[:split_idx].reset_index(drop=True)
    test = train_period.iloc[split_idx:].reset_index(drop=True)
    oot = oot_period.reset_index(drop=True)

    return train, test, oot


@pytest.fixture
def large_sample_data() -> pd.DataFrame:
    """Larger synthetic DataFrame (1000 rows) for tests needing more data.

    Has more statistical power for IV/PSI calculations.
    """
    np.random.seed(42)
    n = 1000

    target = np.zeros(n)
    target[:200] = 1
    np.random.shuffle(target)

    dates = pd.date_range("2022-01-01", periods=n, freq="1D")

    data = {
        "application_id": range(1, n + 1),
        "customer_id": range(1001, 1001 + n),
        "applicant_type": ["individual"] * n,
        "date": dates,
        "target": target.astype(int),
    }

    # Constant features
    data["const_feat_1"] = 1.0
    data["const_feat_2"] = 0

    # High-missing features
    data["high_missing_1"] = np.where(
        np.random.rand(n) < 0.80, np.nan, np.random.randn(n)
    )
    data["high_missing_2"] = np.where(
        np.random.rand(n) < 0.75, np.nan, np.random.randn(n)
    )

    # Low-IV features
    data["low_iv_1"] = np.random.randn(n)
    data["low_iv_2"] = np.random.randn(n)

    # High-IV features
    data["high_iv_1"] = target * 2 + np.random.randn(n) * 0.5
    data["high_iv_2"] = target * 1.5 + np.random.randn(n) * 0.5

    # Highly correlated features
    base = np.random.randn(n)
    data["corr_feat_1"] = base + np.random.randn(n) * 0.01
    data["corr_feat_2"] = base + np.random.randn(n) * 0.01

    # Normal features
    for i in range(8):
        data[f"normal_feat_{i+1}"] = target * (0.5 + i * 0.1) + np.random.randn(n)

    return pd.DataFrame(data)


# ===================================================================
# OUTPUT FIXTURES
# ===================================================================

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for output testing."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def tmp_config_yaml(tmp_path, sample_config_dict):
    """Write sample config to a temp YAML file and return its path."""
    import yaml

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f, default_flow_style=False)
    return config_path


# ===================================================================
# LEGACY FIXTURES (kept for compatibility with existing tests)
# ===================================================================

@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Base configuration for testing (legacy)."""
    return {
        "pipeline": {"random_state": 42, "date_format": "%Y-%m-%d", "max_rows_for_pandas": 100000},
        "spark": {"app_name": "TestPipeline", "master": "local[2]"},
        "logging": {"level": "DEBUG"},
    }


@pytest.fixture
def data_config(base_config) -> Dict[str, Any]:
    """Data configuration for testing (legacy)."""
    config = base_config.copy()
    config["data"] = {
        "sources": {
            "applications": {"required_columns": ["application_id", "customer_id", "application_date"]},
            "credit_bureau": {"required_columns": ["customer_id", "product_type", "amount", "open_date"]},
        }
    }
    return config


@pytest.fixture
def model_config(base_config) -> Dict[str, Any]:
    """Model configuration for testing (legacy)."""
    config = base_config.copy()
    config["model"] = {
        "training": {"test_size": 0.2, "validation_size": 0.1, "stratify": True, "random_state": 42},
        "xgboost": {"default_params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}},
        "logistic_regression": {"default_params": {"max_iter": 1000, "solver": "lbfgs", "C": 1.0, "random_state": 42}},
    }
    return config


@pytest.fixture
def quality_config(base_config) -> Dict[str, Any]:
    """Quality configuration for testing (legacy)."""
    config = base_config.copy()
    config["quality"] = {
        "null_checks": {"critical_columns": ["application_id", "customer_id"], "warning_threshold": 0.1, "error_threshold": 0.5},
        "iv_thresholds": {"useless": 0.02, "weak": 0.1, "medium": 0.3, "strong": 0.5},
    }
    return config


@pytest.fixture
def full_config(base_config, data_config, model_config, quality_config) -> Dict[str, Any]:
    """Complete configuration combining all sections (legacy)."""
    config = base_config.copy()
    config.update(data_config)
    config.update(model_config)
    config.update(quality_config)
    return config


@pytest.fixture
def sample_credit_data() -> pd.DataFrame:
    """Sample credit bureau data for testing (legacy)."""
    np.random.seed(42)
    n_samples = 1000
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(n_samples)],
        "application_id": [f"APP_{i:05d}" for i in range(n_samples)],
        "age": np.random.randint(18, 70, n_samples),
        "income": np.random.lognormal(10, 0.5, n_samples),
        "credit_score": np.random.randint(300, 850, n_samples),
        "debt_ratio": np.random.uniform(0, 1, n_samples),
        "num_accounts": np.random.randint(0, 10, n_samples),
        "num_delinquencies": np.random.poisson(0.5, n_samples),
        "credit_history_months": np.random.randint(0, 360, n_samples),
        "target": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    })


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Sample features DataFrame for model testing (legacy)."""
    np.random.seed(42)
    n_samples = 500
    base = np.random.randn(n_samples)
    return pd.DataFrame({
        "feature_1": base + np.random.randn(n_samples) * 0.3,
        "feature_2": -base + np.random.randn(n_samples) * 0.3,
        "feature_3": np.random.randn(n_samples),
        "feature_4": np.random.exponential(1, n_samples),
        "feature_5": np.random.uniform(0, 1, n_samples),
        "target": (base > 0).astype(int),
    })


@pytest.fixture
def binary_classification_data():
    """Generate binary classification data for model tests (legacy)."""
    np.random.seed(42)
    n_samples = 500
    X = pd.DataFrame({
        "feat_1": np.random.randn(n_samples),
        "feat_2": np.random.randn(n_samples),
        "feat_3": np.random.randn(n_samples),
        "feat_4": np.random.randn(n_samples),
        "feat_5": np.random.randn(n_samples),
    })
    logits = 0.5 * X["feat_1"] - 0.3 * X["feat_2"] + 0.2 * X["feat_3"]
    probs = 1 / (1 + np.exp(-logits))
    y = pd.Series((np.random.rand(n_samples) < probs).astype(int))
    return X, y


@pytest.fixture
def train_test_data(binary_classification_data):
    """Split data into train and test sets (legacy)."""
    X, y = binary_classification_data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def mock_spark_session():
    """Mock SparkSession for unit tests (legacy)."""
    mock_session = MagicMock()
    mock_session.sparkContext.appName = "TestApp"
    return mock_session


@pytest.fixture
def mock_spark_dataframe(sample_credit_data):
    """Mock Spark DataFrame from Pandas data (legacy)."""
    mock_df = MagicMock()
    mock_df.toPandas.return_value = sample_credit_data
    mock_df.count.return_value = len(sample_credit_data)
    mock_df.columns = list(sample_credit_data.columns)
    return mock_df


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests (legacy)."""
    temp_dir = tempfile.mkdtemp(prefix="test_output_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_config_dir(base_config):
    """Create temporary config directory with test configs (legacy)."""
    import yaml

    temp_dir = tempfile.mkdtemp(prefix="test_config_")
    config_path = Path(temp_dir)
    with open(config_path / "base_config.yaml", "w") as f:
        yaml.dump(base_config, f)
    yield config_path
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def assert_almost_equal():
    """Helper for floating point comparisons (legacy)."""
    def _assert(actual, expected, tolerance=1e-6):
        assert abs(actual - expected) < tolerance, f"{actual} != {expected} (tolerance: {tolerance})"
    return _assert


@pytest.fixture
def prediction_data():
    """Generate prediction data for evaluation tests (legacy)."""
    np.random.seed(42)
    n = 1000
    y_true = np.random.choice([0, 1], n, p=[0.8, 0.2])
    y_prob = np.where(
        y_true == 1,
        np.clip(np.random.beta(5, 2, n), 0.1, 0.99),
        np.clip(np.random.beta(2, 5, n), 0.01, 0.9),
    )
    y_pred = (y_prob > 0.5).astype(int)
    return y_true, y_prob, y_pred
