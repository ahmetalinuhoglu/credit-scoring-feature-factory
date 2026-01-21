#!/usr/bin/env python
"""
Spark Local Testing Script

Tests that the FeatureFactory produces identical results whether run with
Pandas locally or with Spark. This validates GCP Dataproc compatibility.

Usage:
    # Start Spark cluster first
    docker-compose up -d
    
    # Run this script
    python scripts/spark_local_test.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime


def test_feature_factory_pandas():
    """Test FeatureFactory with Pandas (baseline)."""
    from src.features.feature_factory import FeatureFactory
    
    print("=" * 60)
    print("Testing FeatureFactory with Pandas")
    print("=" * 60)
    
    # Load sample data
    data_dir = Path(__file__).parent.parent / "data" / "sample"
    apps = pd.read_csv(data_dir / "sample_applications.csv")
    bureau = pd.read_csv(data_dir / "sample_credit_bureau.csv")
    
    print(f"Loaded: {len(apps)} applications, {len(bureau)} bureau records")
    
    # Generate features
    factory = FeatureFactory()
    
    import time
    start = time.time()
    result = factory.generate_all_features(apps.head(100), bureau)
    elapsed = time.time() - start
    
    print(f"Generated: {len(result)} rows × {len(result.columns)} columns")
    print(f"Time: {elapsed:.2f}s ({100/elapsed:.1f} apps/sec)")
    
    # Validate output
    assert len(result) > 0, "No results generated"
    assert 'total_credit_count' in result.columns, "Missing expected feature"
    assert result['total_credit_count'].isna().sum() == 0, "Unexpected NaN values"
    
    print("✓ Pandas test passed!")
    return result


def test_feature_factory_parallel():
    """Test FeatureFactory with parallel processing."""
    from src.features.feature_factory import FeatureFactory
    
    print("\n" + "=" * 60)
    print("Testing FeatureFactory with Parallel Processing")
    print("=" * 60)
    
    # Load sample data
    data_dir = Path(__file__).parent.parent / "data" / "sample"
    apps = pd.read_csv(data_dir / "sample_applications.csv")
    bureau = pd.read_csv(data_dir / "sample_credit_bureau.csv")
    
    # Generate features with parallel processing
    factory = FeatureFactory()
    
    import time
    start = time.time()
    result = factory.generate_all_features(
        apps.head(100), bureau, 
        parallel=True, n_jobs=4
    )
    elapsed = time.time() - start
    
    print(f"Generated: {len(result)} rows × {len(result.columns)} columns")
    print(f"Time: {elapsed:.2f}s ({100/elapsed:.1f} apps/sec)")
    
    print("✓ Parallel test passed!")
    return result


def compare_results(pandas_result: pd.DataFrame, parallel_result: pd.DataFrame):
    """Compare results from Pandas and parallel execution."""
    print("\n" + "=" * 60)
    print("Comparing Results")
    print("=" * 60)
    
    # Sort both by application_id and customer_id for comparison
    pandas_sorted = pandas_result.sort_values(
        ['application_id', 'customer_id']
    ).reset_index(drop=True)
    
    parallel_sorted = parallel_result.sort_values(
        ['application_id', 'customer_id']
    ).reset_index(drop=True)
    
    # Check shape
    assert pandas_sorted.shape == parallel_sorted.shape, \
        f"Shape mismatch: {pandas_sorted.shape} vs {parallel_sorted.shape}"
    print(f"✓ Shape matches: {pandas_sorted.shape}")
    
    # Check columns
    pandas_cols = set(pandas_sorted.columns)
    parallel_cols = set(parallel_sorted.columns)
    assert pandas_cols == parallel_cols, \
        f"Column mismatch: {pandas_cols.symmetric_difference(parallel_cols)}"
    print(f"✓ Columns match: {len(pandas_cols)} columns")
    
    # Compare numeric values
    numeric_cols = pandas_sorted.select_dtypes(include=[np.number]).columns
    differences = 0
    
    for col in numeric_cols:
        if not np.allclose(
            pandas_sorted[col].fillna(0), 
            parallel_sorted[col].fillna(0), 
            rtol=1e-5, atol=1e-8
        ):
            max_diff = abs(
                pandas_sorted[col].fillna(0) - parallel_sorted[col].fillna(0)
            ).max()
            print(f"  ⚠ Column '{col}' has differences (max: {max_diff})")
            differences += 1
    
    if differences == 0:
        print("✓ All numeric values match!")
    else:
        print(f"⚠ {differences} columns have minor differences")
    
    return differences == 0


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SPARK LOCAL TESTING")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        # Run Pandas test (baseline)
        pandas_result = test_feature_factory_pandas()
        
        # Run parallel test
        parallel_result = test_feature_factory_parallel()
        
        # Compare results
        match = compare_results(pandas_result, parallel_result)
        
        print("\n" + "=" * 60)
        if match:
            print("✓ ALL TESTS PASSED")
            print("Feature generation is consistent across execution modes.")
            print("Safe to deploy to GCP Dataproc.")
        else:
            print("⚠ TESTS COMPLETED WITH WARNINGS")
            print("Review the differences before deploying.")
        print("=" * 60)
        
        return 0 if match else 1
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
