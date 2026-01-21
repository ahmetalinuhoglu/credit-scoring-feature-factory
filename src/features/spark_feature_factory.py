"""
Spark Feature Factory Adapter

This module provides a PySpark adapter for FeatureFactory, enabling
distributed feature generation across a Spark cluster.

Usage:
    from src.features.spark_feature_factory import SparkFeatureFactory
    
    spark_factory = SparkFeatureFactory(spark)
    features_df = spark_factory.generate_all_features(apps_sdf, bureau_sdf)
"""

from typing import Optional, Iterator
from datetime import datetime

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, 
    IntegerType, DoubleType, DateType
)
from pyspark.sql.functions import (
    pandas_udf, col, broadcast, lit, when,
    sum as spark_sum, count as spark_count, avg as spark_avg,
    max as spark_max, min as spark_min, stddev as spark_stddev
)

from src.features.feature_factory import FeatureFactory


class SparkFeatureFactory:
    """
    Spark adapter for FeatureFactory.
    
    Distributes feature generation across Spark workers using:
    1. Grouped Map Pandas UDF (mapInPandas) for complex features
    2. Native Spark aggregations where possible
    """
    
    def __init__(self, spark: SparkSession, config: Optional[dict] = None):
        """
        Initialize SparkFeatureFactory.
        
        Args:
            spark: SparkSession instance
            config: Optional configuration dictionary
        """
        self.spark = spark
        self.config = config or {}
        self._base_factory = FeatureFactory(config)
        self._schema = None
    
    def _get_output_schema(self) -> StructType:
        """Build output schema from FeatureFactory."""
        if self._schema is not None:
            return self._schema
        
        # Core identification columns
        fields = [
            StructField("application_id", StringType(), True),
            StructField("customer_id", StringType(), True),
            StructField("applicant_type", StringType(), True),
            StructField("application_date", DateType(), True),
            StructField("target", IntegerType(), True),
        ]
        
        # Get all feature names and add as DoubleType
        # Use dummy data to discover feature names
        dummy_apps = pd.DataFrame({
            'application_id': ['A1'],
            'customer_id': ['C1'],
            'applicant_type': ['PRIMARY'],
            'application_date': [datetime.now()],
            'target': [0]
        })
        dummy_bureau = pd.DataFrame({
            'application_id': ['A1'],
            'customer_id': ['C1'],
            'product_type': ['INSTALLMENT_LOAN'],
            'total_amount': [10000.0],
            'opening_date': [datetime.now()],
            'default_date': [None],
            'recovery_date': [None],
            'monthly_payment': [500.0],
            'remaining_term_months': [12],
            'closure_date': [None]
        })
        
        result = self._base_factory.generate_all_features(dummy_apps, dummy_bureau)
        
        for col_name in result.columns:
            if col_name not in ['application_id', 'customer_id', 'applicant_type', 
                               'application_date', 'target']:
                fields.append(StructField(col_name, DoubleType(), True))
        
        self._schema = StructType(fields)
        return self._schema
    
    def generate_all_features(
        self,
        applications_sdf: DataFrame,
        credit_bureau_sdf: DataFrame,
        reference_date_col: str = 'application_date',
        num_partitions: Optional[int] = None
    ) -> DataFrame:
        """
        Generate all features using Spark distributed processing.
        
        Uses pandas_udf with grouped map to process each application's
        data on individual workers.
        
        Args:
            applications_sdf: Spark DataFrame with applications
            credit_bureau_sdf: Spark DataFrame with credit bureau data
            reference_date_col: Column name for reference date
            num_partitions: Number of partitions (default: auto)
            
        Returns:
            Spark DataFrame with generated features
        """
        # Get output schema
        output_schema = self._get_output_schema()
        
        # Repartition by application_id for efficient grouping
        if num_partitions:
            applications_sdf = applications_sdf.repartition(num_partitions, "application_id")
        
        # Join applications with bureau data
        joined_df = applications_sdf.join(
            credit_bureau_sdf,
            on=["application_id", "customer_id"],
            how="left"
        )
        
        # Create a copy of the factory for use in UDF
        factory_config = self.config
        
        # Define the pandas UDF for grouped processing
        @pandas_udf(output_schema, functionType="grouped_map")
        def generate_features_udf(pdf: pd.DataFrame) -> pd.DataFrame:
            """Generate features for a group of applications."""
            # Initialize factory inside UDF (for Spark serialization)
            factory = FeatureFactory(factory_config)
            
            # Get unique applications in this partition
            app_cols = ['application_id', 'customer_id', 'applicant_type', 
                       'application_date', 'target']
            bureau_cols = ['application_id', 'customer_id', 'product_type',
                          'total_amount', 'opening_date', 'default_date',
                          'recovery_date', 'monthly_payment', 'remaining_term_months',
                          'closure_date']
            
            # Extract applications (unique rows)
            apps_df = pdf[app_cols].drop_duplicates()
            
            # Extract bureau data (may have duplicates from join)
            bureau_cols_present = [c for c in bureau_cols if c in pdf.columns]
            bureau_df = pdf[bureau_cols_present].drop_duplicates()
            
            # Generate features
            result = factory.generate_all_features(
                apps_df, 
                bureau_df,
                reference_date_col='application_date'
            )
            
            return result
        
        # Apply grouped map UDF
        result_df = joined_df.groupby("application_id").applyInPandas(
            lambda pdf: generate_features_udf.func(pdf),
            schema=output_schema
        )
        
        return result_df
    
    def generate_all_features_simple(
        self,
        applications_sdf: DataFrame,
        credit_bureau_sdf: DataFrame,
        reference_date_col: str = 'application_date'
    ) -> DataFrame:
        """
        Simple approach: collect to driver and process locally.
        
        Best for smaller datasets (< 100k applications).
        Uses optimized parallel processing on driver.
        
        Args:
            applications_sdf: Spark DataFrame with applications
            credit_bureau_sdf: Spark DataFrame with credit bureau data
            reference_date_col: Column name for reference date
            
        Returns:
            Spark DataFrame with generated features
        """
        # Collect to pandas
        apps_pdf = applications_sdf.toPandas()
        bureau_pdf = credit_bureau_sdf.toPandas()
        
        # Generate features using optimized local processing
        result_pdf = self._base_factory.generate_all_features(
            apps_pdf,
            bureau_pdf,
            reference_date_col=reference_date_col,
            parallel=True,
            n_jobs=-1
        )
        
        # Convert back to Spark DataFrame
        return self.spark.createDataFrame(result_pdf)


def test_spark_feature_factory():
    """Test SparkFeatureFactory with local Spark."""
    from pyspark.sql import SparkSession
    
    print("=" * 60)
    print("SPARK FEATURE FACTORY TEST")
    print("=" * 60)
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("FeatureFactoryTest") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Load sample data
        print("\nLoading sample data...")
        apps_sdf = spark.read.csv(
            "data/sample/sample_applications.csv",
            header=True,
            inferSchema=True
        ).limit(100)  # Limit for testing
        
        bureau_sdf = spark.read.csv(
            "data/sample/sample_credit_bureau.csv",
            header=True,
            inferSchema=True
        )
        
        print(f"Applications: {apps_sdf.count()}")
        print(f"Bureau records: {bureau_sdf.count()}")
        
        # Initialize SparkFeatureFactory
        factory = SparkFeatureFactory(spark)
        
        # Test simple approach (collect + local parallel)
        print("\n" + "=" * 60)
        print("Testing Simple Approach (collect + local parallel)")
        print("=" * 60)
        
        import time
        start = time.time()
        result_simple = factory.generate_all_features_simple(apps_sdf, bureau_sdf)
        elapsed = time.time() - start
        
        row_count = result_simple.count()
        col_count = len(result_simple.columns)
        print(f"Generated: {row_count} rows × {col_count} columns")
        print(f"Time: {elapsed:.2f}s ({row_count/elapsed:.1f} apps/sec)")
        
        # Show sample
        print("\nSample features:")
        result_simple.select(
            "application_id", "total_credit_count", "total_credit_amount",
            "defaulted_credit_count", "installment_loan_count"
        ).show(5)
        
        print("\n" + "=" * 60)
        print("✓ SPARK FEATURE FACTORY TEST PASSED")
        print("=" * 60)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    test_spark_feature_factory()
