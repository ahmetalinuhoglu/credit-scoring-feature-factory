#!/usr/bin/env python3
"""
Spark Feature Extraction

Standalone script for generating credit scoring features via Spark.
Supports BigQuery, parquet, and CSV inputs. Outputs parquet.

Usage:
    # From BigQuery
    spark-submit spark_feature_extraction.py \
        --input-format bigquery \
        --project-id my-project \
        --bq-dataset credit_data \
        --applications-table applications \
        --bureau-table credit_bureau \
        --output-path gs://bucket/features/output \
        --mode simple

    # From local parquet
    spark-submit spark_feature_extraction.py \
        --input-format parquet \
        --applications-path /data/applications.parquet \
        --bureau-path /data/credit_bureau.parquet \
        --output-path /data/features \
        --mode simple

    # From CSV
    spark-submit spark_feature_extraction.py \
        --input-format csv \
        --applications-path /data/applications.csv \
        --bureau-path /data/bureau.csv \
        --output-path /data/features \
        --mode simple
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

# Add project root to path for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data Reading
# ──────────────────────────────────────────────────────────────

def read_bigquery(
    spark: SparkSession, project_id: str, dataset: str,
    table: str, filter_condition: Optional[str] = None,
) -> DataFrame:
    """Read a table from BigQuery."""
    full_table = f"{project_id}.{dataset}.{table}"
    logger.info(f"Reading BigQuery: {full_table}")
    reader = (
        spark.read.format("bigquery")
        .option("table", full_table)
        .option("viewsEnabled", "true")
        .option("materializationDataset", dataset)
    )
    if filter_condition:
        reader = reader.option("filter", filter_condition)
    df = reader.load()
    logger.info(f"  -> {df.count():,} rows")
    return df


def read_local(
    spark: SparkSession, path: str, fmt: str = "parquet",
) -> DataFrame:
    """Read from local/GCS parquet or CSV."""
    logger.info(f"Reading {fmt}: {path}")
    if fmt == "csv":
        df = spark.read.csv(path, header=True, inferSchema=True)
    else:
        df = spark.read.parquet(path)
    logger.info(f"  -> {df.count():,} rows, {len(df.columns)} columns")
    return df


def apply_column_mapping(df: DataFrame, mapping: dict) -> DataFrame:
    """Rename columns according to mapping dict."""
    for src, tgt in mapping.items():
        if src in df.columns:
            df = df.withColumnRenamed(src, tgt)
    return df


def split_joined_data(joined_df: DataFrame) -> tuple:
    """Split a pre-joined table into applications and bureau DataFrames."""
    app_cols = ['customer_id', 'application_id', 'application_date',
                'applicant_type', 'target']
    bureau_cols = ['customer_id', 'application_id', 'product_type',
                   'total_amount', 'opening_date', 'default_date',
                   'recovery_date', 'monthly_payment',
                   'remaining_term_months', 'closure_date', 'duration_months']

    avail_app = [c for c in app_cols if c in joined_df.columns]
    avail_bur = [c for c in bureau_cols if c in joined_df.columns]

    apps_df = joined_df.select(avail_app).distinct()
    bureau_df = joined_df.select(avail_bur)
    logger.info(
        f"Split: {apps_df.count():,} applications, "
        f"{bureau_df.count():,} bureau records"
    )
    return apps_df, bureau_df


# ──────────────────────────────────────────────────────────────
# Feature Generation
# ──────────────────────────────────────────────────────────────

def generate_features_simple(
    spark: SparkSession, apps_df: DataFrame, bureau_df: DataFrame,
    config: dict,
) -> DataFrame:
    """Collect to driver, generate features with FeatureFactory (parallel)."""
    from src.features.feature_factory import FeatureFactory

    logger.info("Mode: simple (collect to driver)")
    apps_pd = apps_df.toPandas()
    bureau_pd = bureau_df.toPandas()
    logger.info(f"Collected: {len(apps_pd):,} apps, {len(bureau_pd):,} bureau")

    factory = FeatureFactory(config)
    t0 = time.time()
    result = factory.generate_all_features(
        applications_df=apps_pd,
        credit_bureau_df=bureau_pd,
        parallel=True,
    )
    elapsed = time.time() - t0
    logger.info(
        f"Generated {result.shape[1]} features for {result.shape[0]:,} rows "
        f"in {elapsed:.1f}s ({len(apps_pd)/elapsed:.0f} apps/sec)"
    )
    return spark.createDataFrame(result)


def generate_features_distributed(
    spark: SparkSession, apps_df: DataFrame, bureau_df: DataFrame,
    config: dict, num_partitions: int = 200,
) -> DataFrame:
    """Use SparkFeatureFactory with pandas_udf for distributed processing."""
    from src.features.spark_feature_factory import SparkFeatureFactory

    logger.info(f"Mode: distributed ({num_partitions} partitions)")
    factory = SparkFeatureFactory(spark, config)
    t0 = time.time()
    result = factory.generate_all_features(
        applications_sdf=apps_df,
        credit_bureau_sdf=bureau_df,
        num_partitions=num_partitions,
    )
    result = result.cache()
    count = result.count()
    elapsed = time.time() - t0
    logger.info(
        f"Generated features for {count:,} rows in {elapsed:.1f}s "
        f"({count/elapsed:.0f} apps/sec)"
    )
    return result


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Spark Feature Extraction for Credit Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input format
    p.add_argument(
        '--input-format', choices=['bigquery', 'parquet', 'csv'],
        default='parquet', help='Input data format',
    )

    # BigQuery options
    p.add_argument('--project-id', help='GCP project ID (BigQuery)')
    p.add_argument('--bq-dataset', help='BigQuery dataset name')
    p.add_argument('--applications-table', help='Applications table (BigQuery)')
    p.add_argument('--bureau-table', help='Bureau table (BigQuery)')
    p.add_argument('--joined-table', help='Pre-joined table (BigQuery)')

    # Local file options
    p.add_argument('--applications-path', help='Path to applications file')
    p.add_argument('--bureau-path', help='Path to bureau file')

    # Output
    p.add_argument(
        '--output-path', required=True,
        help='Output path for parquet (local or gs://)',
    )

    # Processing
    p.add_argument(
        '--mode', choices=['simple', 'distributed'], default='simple',
        help='Processing mode',
    )
    p.add_argument(
        '--num-partitions', type=int, default=200,
        help='Partitions for distributed mode',
    )
    p.add_argument(
        '--column-mapping', default=None,
        help='JSON column mapping: \'{"src": "tgt", ...}\'',
    )
    p.add_argument(
        '--date-filter', default=None,
        help='SQL filter for BigQuery (e.g., \'application_date >= "2024-01-01"\')',
    )
    p.add_argument(
        '--config-path', default=None,
        help='Path to feature_config.yaml',
    )

    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("SPARK FEATURE EXTRACTION")
    logger.info(f"Format: {args.input_format} | Mode: {args.mode}")
    logger.info(f"Output: {args.output_path}")
    logger.info("=" * 60)

    spark = (
        SparkSession.builder
        .appName(f"FeatureExtraction_{datetime.now():%Y%m%d_%H%M%S}")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        # Load config
        config = {}
        if args.config_path:
            import yaml
            with open(args.config_path) as f:
                config = yaml.safe_load(f) or {}

        # Parse column mapping
        col_map = json.loads(args.column_mapping) if args.column_mapping else None

        # Read data
        if args.input_format == 'bigquery':
            if args.joined_table:
                joined = read_bigquery(
                    spark, args.project_id, args.bq_dataset,
                    args.joined_table, args.date_filter,
                )
                if col_map:
                    joined = apply_column_mapping(joined, col_map)
                apps_df, bureau_df = split_joined_data(joined)
            else:
                apps_df = read_bigquery(
                    spark, args.project_id, args.bq_dataset,
                    args.applications_table or 'applications',
                    args.date_filter,
                )
                bureau_df = read_bigquery(
                    spark, args.project_id, args.bq_dataset,
                    args.bureau_table or 'credit_bureau',
                )
                if col_map:
                    apps_df = apply_column_mapping(apps_df, col_map)
                    bureau_df = apply_column_mapping(bureau_df, col_map)
        else:
            fmt = args.input_format
            apps_df = read_local(spark, args.applications_path, fmt)
            bureau_df = read_local(spark, args.bureau_path, fmt)
            if col_map:
                apps_df = apply_column_mapping(apps_df, col_map)
                bureau_df = apply_column_mapping(bureau_df, col_map)

        # Generate features
        if args.mode == 'distributed':
            result = generate_features_distributed(
                spark, apps_df, bureau_df, config, args.num_partitions,
            )
        else:
            result = generate_features_simple(
                spark, apps_df, bureau_df, config,
            )

        # Save
        logger.info(f"Saving to {args.output_path}")
        result.write.parquet(args.output_path, mode="overwrite",
                             compression="snappy")
        logger.info("Done.")

    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
