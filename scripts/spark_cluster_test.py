#!/usr/bin/env python3
"""
Spark Cluster Test - Runs on the actual Spark cluster (not local mode)
This will show up in the Spark UI at http://localhost:8080
"""
import sys
import time
sys.path.insert(0, '/app/src')

from pyspark.sql import SparkSession

def main():
    print("=" * 60)
    print("SPARK CLUSTER TEST")
    print("=" * 60)
    
    # Create SparkSession connected to the cluster
    spark = SparkSession.builder \
        .appName("FeatureFactoryClusterTest") \
        .master("spark://spark-master:7077") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.ui.showConsoleProgress", "true") \
        .getOrCreate()
    
    print(f"\n✓ Connected to Spark: {spark.sparkContext.master}")
    print(f"  Application ID: {spark.sparkContext.applicationId}")
    print(f"  Web UI: http://localhost:4040")
    print(f"  Cluster UI: http://localhost:8080")
    
    # Load sample data
    print("\n[Step 1] Loading data...")
    apps_pdf = spark.read.parquet('/app/data/processed/applications.parquet').limit(100).toPandas()
    bureau_pdf = spark.read.parquet('/app/data/processed/bureau.parquet').toPandas()
    print(f"  Applications: {len(apps_pdf)}")
    print(f"  Bureau records: {len(bureau_pdf)}")
    
    # Import the feature factory
    from features.feature_factory import FeatureFactory
    from features.spark_feature_factory import SparkFeatureFactory
    import yaml
    
    with open('/app/configs/feature_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Create the factory
    factory = FeatureFactory(config)
    spark_factory = SparkFeatureFactory(spark, factory)
    
    print("\n[Step 2] Generating features (local parallel)...")
    start = time.time()
    result = spark_factory.generate_all_features_simple(apps_pdf, bureau_pdf)
    elapsed = time.time() - start
    
    print(f"\n✓ Generated: {result.shape[0]} rows × {result.shape[1]} columns")
    print(f"  Time: {elapsed:.2f}s ({len(apps_pdf)/elapsed:.1f} apps/sec)")
    
    # Keep the application alive for 60 seconds so you can see it in the UI
    print("\n" + "=" * 60)
    print("APPLICATION RUNNING - CHECK THE SPARK UI:")
    print("  - Cluster Master:  http://localhost:8080")
    print("  - Application UI:  http://localhost:4040")
    print("=" * 60)
    print("\nApplication will remain active for 60 seconds...")
    print("Press Ctrl+C to stop early.")
    
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping...")
    
    spark.stop()
    print("\n✓ SPARK CLUSTER TEST COMPLETED!")

if __name__ == "__main__":
    main()
