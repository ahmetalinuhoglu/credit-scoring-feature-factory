"""
Spark Feature Factory Test Script for Docker

Run this inside the Spark Docker container:
    docker exec -it spark-master python /app/scripts/spark_docker_test.py
"""

import sys
import time
sys.path.insert(0, '/app')

from pyspark.sql import SparkSession
import pandas as pd


def main():
    print("=" * 60)
    print("SPARK DOCKER FEATURE FACTORY TEST")
    print("=" * 60)
    
    # Create Spark session connecting to master
    spark = SparkSession.builder \
        .appName("FeatureFactoryDockerTest") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Load sample data
        print("\nLoading sample data...")
        
        apps_pdf = pd.read_csv("/app/data/sample/sample_applications.csv")
        bureau_pdf = pd.read_csv("/app/data/sample/sample_credit_bureau.csv")
        
        print(f"Applications: {len(apps_pdf)}")
        print(f"Bureau records: {len(bureau_pdf)}")
        
        # Convert to Spark DataFrames
        apps_sdf = spark.createDataFrame(apps_pdf)
        bureau_sdf = spark.createDataFrame(bureau_pdf)
        
        # Test 1: Simple Pandas processing (baseline)
        print("\n" + "=" * 60)
        print("Test 1: Local Pandas Processing")
        print("=" * 60)
        
        from src.features.feature_factory import FeatureFactory
        
        factory = FeatureFactory()
        
        start = time.time()
        result_pandas = factory.generate_all_features(
            apps_pdf.copy(), 
            bureau_pdf.copy(),
            parallel=False
        )
        elapsed = time.time() - start
        
        print(f"Generated: {len(result_pandas)} rows × {len(result_pandas.columns)} columns")
        print(f"Time: {elapsed:.2f}s ({len(apps_pdf)/elapsed:.1f} apps/sec)")
        
        # Test 2: Collect + Parallel Processing
        print("\n" + "=" * 60)
        print("Test 2: Spark Collect + Parallel Processing")
        print("=" * 60)
        
        start = time.time()
        
        # Collect from Spark
        apps_collected = apps_sdf.toPandas()
        bureau_collected = bureau_sdf.toPandas()
        
        # Process with parallel
        result_parallel = factory.generate_all_features(
            apps_collected,
            bureau_collected,
            parallel=True,
            n_jobs=-1
        )
        
        # Convert back to Spark
        result_sdf = spark.createDataFrame(result_parallel)
        row_count = result_sdf.count()
        
        elapsed = time.time() - start
        
        print(f"Generated: {row_count} rows × {len(result_parallel.columns)} columns")
        print(f"Time: {elapsed:.2f}s ({len(apps_pdf)/elapsed:.1f} apps/sec)")
        
        # Show sample results
        print("\nSample Features:")
        result_sdf.select(
            "application_id", 
            "total_credit_count", 
            "total_credit_amount",
            "default_count_ever",
            "installment_loan_count"
        ).show(5)
        
        # Verify values match
        print("\n" + "=" * 60)
        print("Verification")
        print("=" * 60)
        
        numeric_cols = [c for c in result_pandas.columns if result_pandas[c].dtype in ['float64', 'int64']]
        pandas_values = result_pandas[numeric_cols].values
        parallel_values = result_parallel[numeric_cols].values
        
        if pandas_values.shape == parallel_values.shape:
            diff = abs(pandas_values - parallel_values).sum()
            print(f"✓ Shapes match: {pandas_values.shape}")
            print(f"✓ Total numeric difference: {diff:.6f}")
            if diff < 0.001:
                print("✓ Values match!")
        else:
            print(f"✗ Shape mismatch: {pandas_values.shape} vs {parallel_values.shape}")
        
        print("\n" + "=" * 60)
        print("✓ SPARK DOCKER TEST COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
