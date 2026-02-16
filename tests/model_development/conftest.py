"""
Shared fixtures for model_development test suite.

Provides pre-built DataSets, trained XGBoost model, canned EliminationResult,
and canned performance DataFrame fixtures.
"""

import pytest
import numpy as np
import pandas as pd
import xgboost as xgb

from src.model_development.data_loader import DataSets
from src.model_development.eliminators import EliminationResult


# ===================================================================
# datasets_fixture: small synthetic DataSets
# ===================================================================

@pytest.fixture
def datasets_fixture():
    """Pre-built DataSets with small synthetic data.

    100 rows, 10 features (feat_0..feat_9), ~20% bad rate,
    3 months spanning 2024-01 to 2024-09 so we get train + 1 OOT quarter.
    Includes 'applicant_type' column with values ['new', 'existing'].
    """
    rng = np.random.RandomState(42)
    n = 100

    # Target: ~20% bad rate
    target = np.zeros(n, dtype=int)
    target[:20] = 1
    rng.shuffle(target)

    # Dates spanning 2024-01 to 2024-09 (9 months)
    dates = pd.date_range("2024-01-01", "2024-09-30", periods=n)

    # 10 features with varying correlation to target
    features = {}
    for i in range(10):
        noise = rng.randn(n)
        signal_strength = 0.3 + i * 0.15
        features[f"feat_{i}"] = target * signal_strength + noise

    # Metadata columns
    applicant_type = rng.choice(["new", "existing"], size=n)

    df = pd.DataFrame(features)
    df["target"] = target
    df["application_date"] = dates
    df["applicant_type"] = applicant_type

    feature_columns = [f"feat_{i}" for i in range(10)]

    # Split: train_end_date = 2024-06-30 -> train+test period, after = OOT
    cutoff = pd.Timestamp("2024-06-30")
    train_period = df[df["application_date"] <= cutoff].copy()
    oot_period = df[df["application_date"] > cutoff].copy()

    # Stratified train/test split within train_period
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        train_period,
        test_size=0.20,
        stratify=train_period["target"],
        random_state=42,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Build OOT quarters
    oot_quarters = {}
    if len(oot_period) > 0:
        oot_period = oot_period.copy()
        oot_period["_quarter"] = oot_period["application_date"].dt.to_period("Q")
        for period, group in oot_period.groupby("_quarter"):
            label = str(period)
            oot_quarters[label] = group.drop(columns=["_quarter"]).reset_index(drop=True)

    return DataSets(
        train=train_df,
        test=test_df,
        oot_quarters=oot_quarters,
        feature_columns=feature_columns,
        id_columns=[],
        meta_columns=["applicant_type"],
        target_column="target",
        date_column="application_date",
    )


# ===================================================================
# trained_xgb_model: quick XGBoost on datasets_fixture
# ===================================================================

@pytest.fixture
def trained_xgb_model(datasets_fixture):
    """Quick XGBoost model trained on datasets_fixture features."""
    ds = datasets_fixture
    features = ds.feature_columns
    X_train = ds.train[features]
    y_train = ds.train[ds.target_column]
    X_test = ds.test[features]
    y_test = ds.test[ds.target_column]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 50,
        "early_stopping_rounds": 10,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": 1,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    return model


# ===================================================================
# elimination_result_fixture: canned EliminationResult
# ===================================================================

@pytest.fixture
def elimination_result_fixture():
    """Canned EliminationResult for testing."""
    kept = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
    eliminated = ["feat_5", "feat_6", "feat_7", "feat_8", "feat_9"]

    details_df = pd.DataFrame({
        "Feature": kept + eliminated,
        "IV_Score": [0.15, 0.12, 0.10, 0.08, 0.05, 0.01, 0.005, 0.002, 0.001, 0.0],
        "Status": ["Kept"] * 5 + ["Eliminated"] * 5,
    })

    return EliminationResult(
        step_name="03_IV_Analysis",
        kept_features=kept,
        eliminated_features=eliminated,
        details_df=details_df,
    )


# ===================================================================
# performance_df_fixture: canned performance DataFrame
# ===================================================================

@pytest.fixture
def performance_df_fixture():
    """Canned performance DataFrame with Train/Test/OOT rows."""
    return pd.DataFrame({
        "Period": ["Train", "Test", "OOT_2024Q3", "OOT_2024Q4"],
        "N_Samples": [60, 15, 15, 10],
        "N_Bads": [12, 3, 3, 2],
        "Bad_Rate": [0.2000, 0.2000, 0.2000, 0.2000],
        "AUC": [0.8500, 0.8200, 0.8000, 0.7800],
        "Gini": [0.7000, 0.6400, 0.6000, 0.5600],
        "KS": [0.5500, 0.5200, 0.5000, 0.4800],
    })
