"""
Tests for src.model_development.data_loader

Covers: load_and_split, stratified split, OOT quarter splitting,
feature_columns exclusion, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd

from src.model_development.data_loader import load_and_split, DataSets


# ===================================================================
# Helper to create a small parquet file
# ===================================================================

def _make_parquet(tmp_path, n=100, bad_rate=0.20, seed=42):
    """Write a small synthetic parquet and return its path."""
    rng = np.random.RandomState(seed)
    target = np.zeros(n, dtype=int)
    target[: int(n * bad_rate)] = 1
    rng.shuffle(target)

    dates = pd.date_range("2024-01-01", "2024-09-30", periods=n)

    data = {
        "application_id": range(n),
        "customer_id": range(1000, 1000 + n),
        "applicant_type": rng.choice(["new", "existing"], size=n),
        "application_date": dates,
        "target": target,
    }
    for i in range(5):
        data[f"feat_{i}"] = rng.randn(n)

    df = pd.DataFrame(data)
    path = str(tmp_path / "test_data.parquet")
    df.to_parquet(path, index=False)
    return path, df


# ===================================================================
# Basic load_and_split tests
# ===================================================================

class TestLoadAndSplit:
    def test_returns_datasets(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        assert isinstance(ds, DataSets)

    def test_train_not_empty(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        assert len(ds.train) > 0

    def test_test_not_empty(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        assert len(ds.test) > 0

    def test_oot_quarters_present(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        assert len(ds.oot_quarters) >= 1

    def test_train_dates_before_cutoff(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        cutoff = "2024-06-30"
        ds = load_and_split(path, train_end_date=cutoff)
        max_train_date = ds.train["application_date"].max()
        assert max_train_date <= pd.Timestamp(cutoff)

    def test_oot_dates_after_cutoff(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        cutoff = "2024-06-30"
        ds = load_and_split(path, train_end_date=cutoff)
        for label, qdf in ds.oot_quarters.items():
            min_oot_date = qdf["application_date"].min()
            assert min_oot_date > pd.Timestamp(cutoff)


# ===================================================================
# Stratified split tests
# ===================================================================

class TestStratifiedSplit:
    def test_maintains_bad_rate(self, tmp_path):
        path, _ = _make_parquet(tmp_path, n=200, bad_rate=0.20)
        ds = load_and_split(path, train_end_date="2024-06-30", stratify=True)
        train_rate = ds.train["target"].mean()
        test_rate = ds.test["target"].mean()
        # Should be within 5 percentage points of each other
        assert abs(train_rate - test_rate) < 0.10

    def test_deterministic_split(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds1 = load_and_split(path, train_end_date="2024-06-30", random_state=42)
        ds2 = load_and_split(path, train_end_date="2024-06-30", random_state=42)
        pd.testing.assert_frame_equal(ds1.train, ds2.train)


# ===================================================================
# OOT quarter tests
# ===================================================================

class TestOOTQuarters:
    def test_quarter_labels_format(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        for label in ds.oot_labels:
            # e.g. "2024Q3"
            assert "Q" in label

    def test_all_oot_combines_quarters(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        all_oot = ds.all_oot
        total_oot_rows = sum(len(q) for q in ds.oot_quarters.values())
        assert len(all_oot) == total_oot_rows

    def test_no_oot_when_cutoff_after_all_dates(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2025-12-31")
        assert len(ds.oot_quarters) == 0
        assert len(ds.all_oot) == 0


# ===================================================================
# Feature columns tests
# ===================================================================

class TestFeatureColumns:
    def test_excludes_target(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        assert "target" not in ds.feature_columns

    def test_excludes_date(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        assert "application_date" not in ds.feature_columns

    def test_excludes_id_columns(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        assert "application_id" not in ds.feature_columns
        assert "customer_id" not in ds.feature_columns

    def test_excludes_meta_columns(self, tmp_path):
        path, _ = _make_parquet(tmp_path)
        ds = load_and_split(path, train_end_date="2024-06-30")
        assert "applicant_type" not in ds.feature_columns


# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:
    def test_missing_target_raises(self, tmp_path):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "application_id": range(50),
            "customer_id": range(50),
            "applicant_type": ["new"] * 50,
            "application_date": pd.date_range("2024-01-01", periods=50),
            "feat_0": rng.randn(50),
        })
        path = str(tmp_path / "no_target.parquet")
        df.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="Target column"):
            load_and_split(path, train_end_date="2024-06-30")

    def test_missing_date_column_raises(self, tmp_path):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "application_id": range(50),
            "customer_id": range(50),
            "applicant_type": ["new"] * 50,
            "target": rng.choice([0, 1], 50),
            "feat_0": rng.randn(50),
        })
        path = str(tmp_path / "no_date.parquet")
        df.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="Date column"):
            load_and_split(path, train_end_date="2024-06-30")
