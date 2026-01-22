#!/usr/bin/env python3
"""
Dataproc Feature Generation Job

This script runs on GCP Dataproc to generate credit scoring features.
It reads from BigQuery, processes with the FeatureFactory using Spark's
distributed computing, and saves results to GCS.

Usage:
    gcloud dataproc jobs submit pyspark dataproc_feature_job.py \
        --cluster=your-cluster \
        --region=your-region \
        --py-files=feature_factory.zip \
        -- \
        --project-id=your-project \
        --bq-dataset=your_dataset \
        --gcs-bucket=your-bucket \
        --output-path=features/output

Author: Feature Engineering Team
Version: 1.0.0
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType
import pyspark.sql.functions as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "FeatureFactoryJob") -> SparkSession:
    """
    Create a SparkSession configured for BigQuery and GCS access.
    
    On Dataproc, most configurations are pre-set. This function ensures
    the necessary connectors are available.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"SparkSession created: {spark.sparkContext.applicationId}")
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Master: {spark.sparkContext.master}")
    
    return spark


def read_from_bigquery(
    spark: SparkSession,
    project_id: str,
    dataset: str,
    table: str,
    filter_condition: Optional[str] = None
) -> DataFrame:
    """
    Read a table from BigQuery into a Spark DataFrame.
    
    Args:
        spark: SparkSession
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        filter_condition: Optional SQL WHERE clause for filtering
        
    Returns:
        Spark DataFrame with the table data
    """
    full_table = f"{project_id}.{dataset}.{table}"
    logger.info(f"Reading from BigQuery: {full_table}")
    
    reader = spark.read.format("bigquery") \
        .option("table", full_table) \
        .option("viewsEnabled", "true") \
        .option("materializationDataset", dataset)
    
    if filter_condition:
        reader = reader.option("filter", filter_condition)
        logger.info(f"Applied filter: {filter_condition}")
    
    df = reader.load()
    
    count = df.count()
    logger.info(f"Loaded {count:,} rows from {table}")
    
    return df


def save_to_gcs(
    df: DataFrame,
    gcs_path: str,
    format: str = "parquet",
    mode: str = "overwrite",
    partition_by: Optional[list] = None
) -> None:
    """
    Save a Spark DataFrame to GCS.
    
    Args:
        df: Spark DataFrame to save
        gcs_path: GCS path (gs://bucket/path)
        format: Output format (parquet, csv, json)
        mode: Save mode (overwrite, append, error, ignore)
        partition_by: Optional list of columns to partition by
    """
    logger.info(f"Saving to GCS: {gcs_path}")
    
    writer = df.write.format(format).mode(mode)
    
    if partition_by:
        writer = writer.partitionBy(*partition_by)
        logger.info(f"Partitioning by: {partition_by}")
    
    if format == "parquet":
        writer = writer.option("compression", "snappy")
    
    writer.save(gcs_path)
    logger.info(f"Successfully saved to {gcs_path}")


def generate_features_distributed(
    spark: SparkSession,
    applications_df: DataFrame,
    bureau_df: DataFrame,
    config: dict,
    num_partitions: int = 200
) -> DataFrame:
    """
    Generate features using distributed processing with pandas_udf.
    
    This method partitions the data by customer_id and processes each
    partition using the FeatureFactory's optimized Pandas logic.
    
    Args:
        spark: SparkSession
        applications_df: Spark DataFrame with application data
        bureau_df: Spark DataFrame with bureau/credit data
        config: Feature factory configuration
        num_partitions: Number of partitions for distributed processing
        
    Returns:
        Spark DataFrame with generated features
    """
    from features.feature_factory import FeatureFactory
    import pandas as pd
    
    logger.info("Starting distributed feature generation...")
    
    # Get the schema from the factory
    factory = FeatureFactory(config)
    feature_names = factory.feature_names
    
    # Build output schema dynamically
    schema_fields = [StructField("customer_id", StringType(), True)]
    for name in feature_names:
        schema_fields.append(StructField(name, DoubleType(), True))
    output_schema = StructType(schema_fields)
    
    # Prepare the data
    # Join applications with bureau on customer_id
    apps_with_key = applications_df.withColumn("_app_key", F.col("customer_id"))
    bureau_with_key = bureau_df.withColumn("_bureau_key", F.col("customer_id"))
    
    # Broadcast the config to all workers
    config_broadcast = spark.sparkContext.broadcast(config)
    
    # Define the pandas UDF for grouped processing
    @F.pandas_udf(output_schema, F.PandasUDFType.GROUPED_MAP)
    def generate_customer_features(pdf: pd.DataFrame) -> pd.DataFrame:
        """Process a single customer's data and generate features."""
        if pdf.empty:
            return pd.DataFrame(columns=[f.name for f in output_schema.fields])
        
        # Get config from broadcast
        cfg = config_broadcast.value
        
        # Create factory (cached per executor)
        factory = FeatureFactory(cfg)
        
        # Extract customer_id
        customer_id = pdf['customer_id'].iloc[0]
        
        # Split the joined data back into applications and bureau
        # (This depends on your data structure - adjust column names as needed)
        app_cols = ['customer_id', 'application_date', 'application_id']  # Add your app columns
        bureau_cols = [c for c in pdf.columns if c not in app_cols or c == 'customer_id']
        
        # Get application row (assuming one row per customer)
        app_row = pdf[app_cols].drop_duplicates().iloc[0]
        
        # Get bureau data
        bureau_data = pdf[bureau_cols].copy()
        
        # Generate features
        try:
            features = factory._generate_customer_features(
                customer_id=customer_id,
                credit_data=bureau_data,
                reference_date=pd.to_datetime(app_row.get('application_date', pd.Timestamp.now()))
            )
            
            # Create result row
            result = {'customer_id': customer_id}
            result.update(features)
            
            return pd.DataFrame([result])
        except Exception as e:
            # Log error and return empty features
            print(f"Error processing customer {customer_id}: {e}")
            result = {'customer_id': customer_id}
            for name in feature_names:
                result[name] = None
            return pd.DataFrame([result])
    
    # Join applications and bureau
    joined_df = applications_df.join(
        bureau_df,
        on="customer_id",
        how="left"
    )
    
    # Repartition by customer_id for optimal grouping
    repartitioned = joined_df.repartition(num_partitions, "customer_id")
    
    # Apply the UDF grouped by customer_id
    logger.info(f"Processing with {num_partitions} partitions...")
    start_time = time.time()
    
    result_df = repartitioned.groupby("customer_id").apply(generate_customer_features)
    
    # Cache the result for performance
    result_df = result_df.cache()
    
    # Trigger computation
    count = result_df.count()
    elapsed = time.time() - start_time
    
    logger.info(f"Generated features for {count:,} customers in {elapsed:.2f}s")
    logger.info(f"Throughput: {count/elapsed:.1f} customers/sec")
    
    return result_df


def generate_features_simple(
    spark: SparkSession,
    applications_df: DataFrame,
    bureau_df: DataFrame,
    config: dict
) -> DataFrame:
    """
    Generate features by collecting data to driver and using parallel processing.
    
    This is simpler but less scalable - suitable for datasets that fit in driver memory.
    
    Args:
        spark: SparkSession
        applications_df: Spark DataFrame with application data
        bureau_df: Spark DataFrame with bureau/credit data
        config: Feature factory configuration
        
    Returns:
        Spark DataFrame with generated features
    """
    from features.feature_factory import FeatureFactory
    import pandas as pd
    
    logger.info("Starting simple feature generation (collect to driver)...")
    logger.warning("This method collects all data to the driver - ensure sufficient memory!")
    
    # Collect to Pandas
    start_time = time.time()
    apps_pdf = applications_df.toPandas()
    bureau_pdf = bureau_df.toPandas()
    collect_time = time.time() - start_time
    logger.info(f"Collected data in {collect_time:.2f}s")
    logger.info(f"Applications: {len(apps_pdf):,}, Bureau records: {len(bureau_pdf):,}")
    
    # Create factory and generate features
    factory = FeatureFactory(config)
    
    start_time = time.time()
    result_pdf = factory.generate_all_features(
        applications_df=apps_pdf,
        credit_bureau_df=bureau_pdf,
        parallel=True
    )
    process_time = time.time() - start_time
    
    logger.info(f"Generated {result_pdf.shape[1]} features for {result_pdf.shape[0]:,} customers")
    logger.info(f"Processing time: {process_time:.2f}s ({len(apps_pdf)/process_time:.1f} apps/sec)")
    
    # Convert back to Spark DataFrame
    return spark.createDataFrame(result_pdf)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dataproc Feature Generation Job")
    
    # Required arguments
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--bq-dataset", required=True, help="BigQuery dataset name")
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket name (without gs://)")
    
    # Table configuration - Option 1: Separate tables
    parser.add_argument("--applications-table", default=None,
                        help="BigQuery table name for applications (use with --bureau-table)")
    parser.add_argument("--bureau-table", default=None,
                        help="BigQuery table name for bureau/credit data (use with --applications-table)")
    
    # Table configuration - Option 2: Pre-joined table
    parser.add_argument("--joined-table", default=None,
                        help="BigQuery table name for pre-joined data (applications + bureau)")
    
    # Column mapping for custom field names
    parser.add_argument("--column-mapping", default=None,
                        help="""JSON string to map your column names to expected names.
Example: '{"musteri_no": "customer_id", "basvuru_tarihi": "application_date", "kredi_tutari": "total_amount"}'

Expected columns for applications:
  - customer_id, application_id, application_date, applicant_type, target

Expected columns for bureau/credit:
  - customer_id, application_id, product_type, total_amount, opening_date,
    default_date, recovery_date, monthly_payment, remaining_term_months, closure_date
""")
    
    # Other arguments
    parser.add_argument("--output-path", default="features/output", 
                        help="Output path within the GCS bucket")
    parser.add_argument("--config-path", default="gs://",
                        help="Path to feature_config.yaml (GCS or local)")
    parser.add_argument("--date-filter", default=None,
                        help="Optional date filter (e.g., 'application_date >= \"2024-01-01\"')")
    parser.add_argument("--mode", choices=["distributed", "simple"], default="simple",
                        help="Processing mode: distributed (pandas_udf) or simple (collect)")
    parser.add_argument("--num-partitions", type=int, default=200,
                        help="Number of partitions for distributed mode")
    parser.add_argument("--save-format", choices=["parquet", "csv", "json"], default="parquet",
                        help="Output format")
    
    args = parser.parse_args()
    
    # Validate table arguments
    if args.joined_table:
        if args.applications_table or args.bureau_table:
            parser.error("Cannot use --joined-table with --applications-table or --bureau-table")
    else:
        if not (args.applications_table and args.bureau_table):
            # Default to separate tables
            args.applications_table = args.applications_table or "applications"
            args.bureau_table = args.bureau_table or "bureau"
    
    return args


def apply_column_mapping(df: DataFrame, column_mapping: dict) -> DataFrame:
    """
    Rename columns in DataFrame based on mapping.
    
    Args:
        df: Spark DataFrame
        column_mapping: Dict mapping source names to expected names
                       e.g., {"musteri_no": "customer_id", "basvuru_tarihi": "application_date"}
    
    Returns:
        DataFrame with renamed columns
    """
    if not column_mapping:
        return df
    
    logger.info(f"Applying column mapping: {column_mapping}")
    
    for source_name, target_name in column_mapping.items():
        if source_name in df.columns:
            df = df.withColumnRenamed(source_name, target_name)
            logger.info(f"  Renamed: {source_name} -> {target_name}")
        else:
            logger.warning(f"  Column not found: {source_name}")
    
    return df


def split_joined_data(joined_df: DataFrame) -> tuple:
    """
    Split pre-joined data into applications and bureau DataFrames.
    
    Args:
        joined_df: Pre-joined DataFrame with both application and bureau columns
        
    Returns:
        Tuple of (applications_df, bureau_df)
    """
    # Application columns - these define unique applications
    app_cols = ['customer_id', 'application_id', 'application_date', 'applicant_type', 'target']
    available_app_cols = [c for c in app_cols if c in joined_df.columns]
    
    # Bureau columns - credit history data
    bureau_cols = [
        'customer_id', 'application_id', 'product_type', 'total_amount', 
        'opening_date', 'default_date', 'recovery_date', 'monthly_payment',
        'remaining_term_months', 'closure_date', 'duration_months'
    ]
    available_bureau_cols = [c for c in bureau_cols if c in joined_df.columns]
    
    logger.info(f"Available application columns: {available_app_cols}")
    logger.info(f"Available bureau columns: {available_bureau_cols}")
    
    # Extract unique applications
    applications_df = joined_df.select(available_app_cols).distinct()
    
    # Extract bureau data (may have multiple rows per application)
    bureau_df = joined_df.select(available_bureau_cols)
    
    app_count = applications_df.count()
    bureau_count = bureau_df.count()
    logger.info(f"Split data: {app_count:,} applications, {bureau_count:,} bureau records")
    
    return applications_df, bureau_df


def load_config(config_path: str, spark: SparkSession) -> dict:
    """
    Load feature configuration from a YAML file.
    
    Args:
        config_path: Path to config file (GCS or local)
        spark: SparkSession (for reading from GCS)
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    if config_path.startswith("gs://"):
        # Read from GCS using Spark
        logger.info(f"Loading config from GCS: {config_path}")
        # Use subprocess to download config
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            tmp_path = tmp.name
        
        subprocess.run(["gsutil", "cp", config_path, tmp_path], check=True)
        
        with open(tmp_path) as f:
            config = yaml.safe_load(f)
    else:
        # Read from local path
        logger.info(f"Loading config from local: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    return config


def main():
    """Main entry point for the Dataproc job."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("DATAPROC FEATURE GENERATION JOB")
    logger.info("=" * 60)
    logger.info(f"Project: {args.project_id}")
    logger.info(f"Dataset: {args.bq_dataset}")
    logger.info(f"Output: gs://{args.gcs_bucket}/{args.output_path}")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Create SparkSession
    spark = create_spark_session("FeatureFactory_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    try:
        # Load configuration
        config = load_config(args.config_path, spark)
        logger.info(f"Loaded configuration with {len(config.get('product_types', []))} product types")
        
        # Parse column mapping if provided
        column_mapping = None
        if args.column_mapping:
            import json
            column_mapping = json.loads(args.column_mapping)
            logger.info(f"Using column mapping with {len(column_mapping)} mappings")
        
        # Read data from BigQuery
        if args.joined_table:
            # Option 2: Pre-joined table
            logger.info(f"Reading pre-joined data from: {args.joined_table}")
            joined_df = read_from_bigquery(
                spark, args.project_id, args.bq_dataset,
                args.joined_table, args.date_filter
            )
            
            # Apply column mapping if provided
            if column_mapping:
                joined_df = apply_column_mapping(joined_df, column_mapping)
            
            # Split into applications and bureau
            applications_df, bureau_df = split_joined_data(joined_df)
        else:
            # Option 1: Separate tables
            logger.info(f"Reading separate tables: {args.applications_table}, {args.bureau_table}")
            applications_df = read_from_bigquery(
                spark, args.project_id, args.bq_dataset, 
                args.applications_table, args.date_filter
            )
            
            bureau_df = read_from_bigquery(
                spark, args.project_id, args.bq_dataset,
                args.bureau_table
            )
            
            # Apply column mapping if provided
            if column_mapping:
                applications_df = apply_column_mapping(applications_df, column_mapping)
                bureau_df = apply_column_mapping(bureau_df, column_mapping)
        
        # Generate features
        if args.mode == "distributed":
            result_df = generate_features_distributed(
                spark, applications_df, bureau_df, config, args.num_partitions
            )
        else:
            result_df = generate_features_simple(
                spark, applications_df, bureau_df, config
            )
        
        # Save results to GCS
        output_path = f"gs://{args.gcs_bucket}/{args.output_path}"
        save_to_gcs(result_df, output_path, format=args.save_format)
        
        # Log summary
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("JOB COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"Output saved to: {output_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
