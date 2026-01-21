# GCP Dataproc Feature Generation Guide

This guide provides step-by-step instructions for running the Credit Scoring Feature Factory on GCP Dataproc, reading data from BigQuery, and saving results to Google Cloud Storage (GCS).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [GCP Setup](#gcp-setup)
4. [Data Preparation](#data-preparation)
5. [Deploying Code to GCS](#deploying-code-to-gcs)
6. [Creating a Dataproc Cluster](#creating-a-dataproc-cluster)
7. [Running the Feature Generation Job](#running-the-feature-generation-job)
8. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
9. [Cost Optimization](#cost-optimization)
10. [Production Best Practices](#production-best-practices)

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│                 │     │                      │     │                 │
│    BigQuery     │────▶│   Dataproc Cluster   │────▶│      GCS        │
│   (Source)      │     │   (Spark + Python)   │     │   (Output)      │
│                 │     │                      │     │                 │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
        │                         │                          │
        ▼                         ▼                          ▼
   ┌─────────┐            ┌──────────────┐           ┌──────────────┐
   │ Tables: │            │ Components:  │           │ Outputs:     │
   │ - apps  │            │ - Master     │           │ - Parquet    │
   │ - bureau│            │ - Workers    │           │ - (or CSV)   │
   └─────────┘            │ - Feature    │           │ - Partitioned│
                          │   Factory    │           │   by date    │
                          └──────────────┘           └──────────────┘
```

### Processing Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Distributed** | Uses `pandas_udf` to process partitions across workers | Large datasets (>100k rows) |
| **Simple** | Collects data to driver, uses parallel processing | Smaller datasets (<100k rows) |

---

## Prerequisites

### Required Tools

```bash
# Install Google Cloud SDK
brew install google-cloud-sdk  # macOS
# or see: https://cloud.google.com/sdk/docs/install

# Verify installation
gcloud version
gsutil version
bq version
```

### Required Permissions

Your GCP account/service account needs:

- `roles/dataproc.editor` - Create and manage Dataproc clusters
- `roles/bigquery.dataViewer` - Read from BigQuery
- `roles/storage.objectAdmin` - Write to GCS
- `roles/iam.serviceAccountUser` - Use service accounts

### Python Dependencies

The Dataproc cluster needs these packages:
- `pandas>=1.5.0`
- `numpy>=1.21.0`
- `pyarrow>=10.0.0`
- `pyyaml>=6.0`

---

## GCP Setup

### 1. Set Environment Variables

```bash
# Set your project
export PROJECT_ID="your-project-id"
export REGION="europe-west1"  # Choose your region
export ZONE="${REGION}-b"
export BUCKET_NAME="your-bucket-name"
export DATASET_NAME="credit_scoring"
export CLUSTER_NAME="feature-factory-cluster"

# Authenticate
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE
```

### 2. Enable Required APIs

```bash
gcloud services enable \
    dataproc.googleapis.com \
    bigquery.googleapis.com \
    storage.googleapis.com \
    compute.googleapis.com
```

### 3. Create GCS Bucket

```bash
# Create bucket for code, configs, and outputs
gsutil mb -l $REGION gs://$BUCKET_NAME

# Create directory structure
gsutil cp /dev/null gs://$BUCKET_NAME/code/.keep
gsutil cp /dev/null gs://$BUCKET_NAME/configs/.keep
gsutil cp /dev/null gs://$BUCKET_NAME/features/.keep
gsutil cp /dev/null gs://$BUCKET_NAME/logs/.keep
```

### 4. Create Service Account (Optional but Recommended)

```bash
# Create service account for Dataproc
SA_NAME="dataproc-feature-factory"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create $SA_NAME \
    --display-name="Dataproc Feature Factory Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/dataproc.worker"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/storage.objectAdmin"
```

---

## Data Preparation

### BigQuery Table Schemas

#### Applications Table

```sql
CREATE TABLE `your-project.credit_scoring.applications` (
    customer_id STRING NOT NULL,
    application_id STRING NOT NULL,
    application_date DATE NOT NULL,
    -- Add other application fields as needed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Bureau Table

```sql
CREATE TABLE `your-project.credit_scoring.bureau` (
    customer_id STRING NOT NULL,
    credit_id STRING,
    product_type STRING,          -- 'consumer', 'mortgage', 'credit_card', etc.
    opening_date DATE,
    maturity_date DATE,
    total_amount FLOAT64,
    remaining_amount FLOAT64,
    monthly_payment FLOAT64,
    default_date DATE,            -- NULL if not defaulted
    recovery_date DATE,           -- NULL if not recovered
    closing_date DATE,            -- NULL if still active
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Load Sample Data to BigQuery

```bash
# From local parquet files
bq load --source_format=PARQUET \
    $DATASET_NAME.applications \
    data/processed/applications.parquet

bq load --source_format=PARQUET \
    $DATASET_NAME.bureau \
    data/processed/bureau.parquet

# Verify data
bq query --use_legacy_sql=false \
    "SELECT COUNT(*) as cnt FROM $PROJECT_ID.$DATASET_NAME.applications"

bq query --use_legacy_sql=false \
    "SELECT COUNT(*) as cnt FROM $PROJECT_ID.$DATASET_NAME.bureau"
```

---

## Deploying Code to GCS

### 1. Package the Feature Factory

```bash
# Navigate to project root
cd /path/to/an-model-development

# Create a zip file with the source code
cd src
zip -r ../feature_factory.zip features/ -x "*.pyc" -x "__pycache__/*"
cd ..

# Upload to GCS
gsutil cp feature_factory.zip gs://$BUCKET_NAME/code/
gsutil cp configs/feature_config.yaml gs://$BUCKET_NAME/configs/
gsutil cp scripts/dataproc_feature_job.py gs://$BUCKET_NAME/code/
```

### 2. Create Initialization Script

This script installs Python dependencies on cluster nodes:

```bash
cat > init_script.sh << 'EOF'
#!/bin/bash
set -e

# Install Python dependencies
pip install --upgrade pip
pip install pandas>=1.5.0 numpy>=1.21.0 pyarrow>=10.0.0 pyyaml>=6.0 pydantic>=2.0
EOF

# Upload to GCS
gsutil cp init_script.sh gs://$BUCKET_NAME/init/init_script.sh
```

---

## Creating a Dataproc Cluster

### Option 1: Standard Cluster (Persistent)

```bash
gcloud dataproc clusters create $CLUSTER_NAME \
    --region=$REGION \
    --zone=$ZONE \
    --image-version=2.1-debian11 \
    --master-machine-type=n1-standard-8 \
    --master-boot-disk-size=100GB \
    --num-workers=4 \
    --worker-machine-type=n1-standard-8 \
    --worker-boot-disk-size=100GB \
    --initialization-actions=gs://$BUCKET_NAME/init/init_script.sh \
    --optional-components=JUPYTER \
    --enable-component-gateway \
    --properties="spark:spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2,spark:spark.executor.memory=6g,spark:spark.driver.memory=8g" \
    --scopes=cloud-platform \
    --service-account=$SA_EMAIL
```

### Option 2: Ephemeral Cluster (Auto-delete)

```bash
gcloud dataproc clusters create $CLUSTER_NAME \
    --region=$REGION \
    --zone=$ZONE \
    --image-version=2.1-debian11 \
    --master-machine-type=n1-standard-4 \
    --num-workers=2 \
    --worker-machine-type=n1-standard-4 \
    --max-idle=30m \
    --max-age=2h \
    --initialization-actions=gs://$BUCKET_NAME/init/init_script.sh \
    --properties="spark:spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2"
```

### Cluster Size Recommendations

| Dataset Size | Master | Workers | Worker Machine |
|--------------|--------|---------|----------------|
| <50k rows | n1-standard-4 | 2 | n1-standard-4 |
| 50k-500k rows | n1-standard-8 | 4 | n1-standard-8 |
| 500k-5M rows | n1-standard-16 | 8 | n1-standard-8 |
| >5M rows | n1-highmem-16 | 16+ | n1-standard-8 |

---

## Running the Feature Generation Job

### Basic Job Submission

```bash
gcloud dataproc jobs submit pyspark \
    gs://$BUCKET_NAME/code/dataproc_feature_job.py \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --py-files=gs://$BUCKET_NAME/code/feature_factory.zip \
    --properties="spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2" \
    -- \
    --project-id=$PROJECT_ID \
    --bq-dataset=$DATASET_NAME \
    --gcs-bucket=$BUCKET_NAME \
    --output-path=features/$(date +%Y%m%d_%H%M%S) \
    --config-path=gs://$BUCKET_NAME/configs/feature_config.yaml \
    --mode=distributed \
    --num-partitions=200
```

### With Date Filter

```bash
gcloud dataproc jobs submit pyspark \
    gs://$BUCKET_NAME/code/dataproc_feature_job.py \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --py-files=gs://$BUCKET_NAME/code/feature_factory.zip \
    --properties="spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2" \
    -- \
    --project-id=$PROJECT_ID \
    --bq-dataset=$DATASET_NAME \
    --gcs-bucket=$BUCKET_NAME \
    --output-path=features/2024/january \
    --config-path=gs://$BUCKET_NAME/configs/feature_config.yaml \
    --date-filter="application_date >= '2024-01-01' AND application_date < '2024-02-01'" \
    --mode=distributed
```

### Simple Mode (For Smaller Datasets)

```bash
gcloud dataproc jobs submit pyspark \
    gs://$BUCKET_NAME/code/dataproc_feature_job.py \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --py-files=gs://$BUCKET_NAME/code/feature_factory.zip \
    --properties="spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2,spark.driver.memory=16g" \
    -- \
    --project-id=$PROJECT_ID \
    --bq-dataset=$DATASET_NAME \
    --gcs-bucket=$BUCKET_NAME \
    --output-path=features/simple_output \
    --config-path=gs://$BUCKET_NAME/configs/feature_config.yaml \
    --mode=simple
```

---

## Monitoring and Troubleshooting

### Viewing Job Status

```bash
# List running jobs
gcloud dataproc jobs list --region=$REGION --state-filter=RUNNING

# Get job details
gcloud dataproc jobs describe JOB_ID --region=$REGION

# Stream logs
gcloud dataproc jobs wait JOB_ID --region=$REGION
```

### Accessing Spark UI

```bash
# Create SSH tunnel to master
gcloud compute ssh $CLUSTER_NAME-m \
    --zone=$ZONE \
    --tunnel-through-iap \
    -- -L 8088:localhost:8088 -L 4040:localhost:4040
```

Then open:
- **YARN UI**: http://localhost:8088
- **Spark UI**: http://localhost:4040

Or use the Dataproc web interfaces in the GCP Console.

### Common Issues and Solutions

#### Issue: "BigQuery connector not found"

```bash
# Add the connector explicitly
--properties="spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2"
```

#### Issue: "Out of memory on driver"

```bash
# Increase driver memory for simple mode
--properties="spark.driver.memory=16g,spark.driver.maxResultSize=8g"
```

#### Issue: "py-files not found"

```bash
# Ensure the zip file is accessible
gsutil ls gs://$BUCKET_NAME/code/feature_factory.zip
```

#### Issue: "Python module not found"

```bash
# Verify initialization script ran
# SSH into master and check
pip list | grep pandas
```

### Viewing Logs

```bash
# Check driver logs
gcloud dataproc jobs describe JOB_ID --region=$REGION

# Download job logs
gsutil cp -r gs://$BUCKET_NAME-dataproc-staging/google-cloud-dataproc-metainfo/*/jobs/JOB_ID/driveroutput ./logs/
```

---

## Cost Optimization

### 1. Use Preemptible/Spot Workers

```bash
gcloud dataproc clusters create $CLUSTER_NAME \
    --region=$REGION \
    --num-secondary-workers=4 \
    --secondary-worker-type=preemptible \
    # ... other options
```

### 2. Auto-scaling

```bash
# Create autoscaling policy
cat > autoscaling-policy.yaml << EOF
workerConfig:
  minInstances: 2
  maxInstances: 10
  weight: 1
secondaryWorkerConfig:
  minInstances: 0
  maxInstances: 20
  weight: 1
basicAlgorithm:
  cooldownPeriod: 2m
  yarnConfig:
    scaleUpFactor: 0.05
    scaleDownFactor: 1.0
    scaleUpMinWorkerFraction: 0.0
    scaleDownMinWorkerFraction: 0.0
    gracefulDecommissionTimeout: 1h
EOF

gcloud dataproc autoscaling-policies create feature-factory-policy \
    --region=$REGION \
    --file=autoscaling-policy.yaml

# Apply to cluster
gcloud dataproc clusters create $CLUSTER_NAME \
    --autoscaling-policy=feature-factory-policy \
    # ... other options
```

### 3. Use Workflows for Ephemeral Clusters

```bash
# Create workflow template
gcloud dataproc workflow-templates create feature-generation-workflow \
    --region=$REGION

# Set managed cluster
gcloud dataproc workflow-templates set-managed-cluster feature-generation-workflow \
    --region=$REGION \
    --master-machine-type=n1-standard-8 \
    --num-workers=4 \
    --worker-machine-type=n1-standard-8 \
    --image-version=2.1-debian11

# Add job step
gcloud dataproc workflow-templates add-job pyspark \
    --workflow-template=feature-generation-workflow \
    --region=$REGION \
    --step-id=generate-features \
    --py-files=gs://$BUCKET_NAME/code/feature_factory.zip \
    gs://$BUCKET_NAME/code/dataproc_feature_job.py \
    -- --project-id=$PROJECT_ID --bq-dataset=$DATASET_NAME --gcs-bucket=$BUCKET_NAME

# Run workflow (creates ephemeral cluster)
gcloud dataproc workflow-templates instantiate feature-generation-workflow \
    --region=$REGION
```

### Estimated Costs

| Configuration | Hourly Cost | 1M Rows Processing Time | Total Cost |
|--------------|-------------|------------------------|------------|
| 2 × n1-standard-4 | ~$0.40 | ~3 hours | ~$1.20 |
| 4 × n1-standard-8 | ~$1.60 | ~1 hour | ~$1.60 |
| 8 × n1-standard-8 | ~$3.20 | ~30 min | ~$1.60 |

---

## Production Best Practices

### 1. Scheduling with Cloud Composer/Airflow

```python
# Example Airflow DAG
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'feature_generation_daily',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    submit_feature_job = DataprocSubmitJobOperator(
        task_id='submit_feature_generation',
        project_id='your-project',
        region='europe-west1',
        job={
            'placement': {'cluster_name': 'feature-factory-cluster'},
            'pyspark_job': {
                'main_python_file_uri': 'gs://your-bucket/code/dataproc_feature_job.py',
                'python_file_uris': ['gs://your-bucket/code/feature_factory.zip'],
                'args': [
                    '--project-id=your-project',
                    '--bq-dataset=credit_scoring',
                    '--gcs-bucket=your-bucket',
                    '--output-path=features/{{ ds }}',
                    '--date-filter=application_date = DATE("{{ ds }}")',
                ],
            }
        }
    )
```

### 2. CI/CD Pipeline

```yaml
# .github/workflows/deploy-feature-factory.yml
name: Deploy Feature Factory

on:
  push:
    branches: [main]
    paths:
      - 'src/features/**'
      - 'configs/feature_config.yaml'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - name: Package and Upload
        run: |
          cd src
          zip -r ../feature_factory.zip features/
          gsutil cp ../feature_factory.zip gs://${{ secrets.GCS_BUCKET }}/code/
          gsutil cp ../configs/feature_config.yaml gs://${{ secrets.GCS_BUCKET }}/configs/
```

### 3. Monitoring and Alerting

```bash
# Create alert policy for failed jobs
gcloud alpha monitoring policies create \
    --display-name="Dataproc Feature Job Failures" \
    --condition-display-name="Job failed" \
    --condition-filter='resource.type="cloud_dataproc_job" AND metric.type="dataproc.googleapis.com/job/state" AND metric.labels.state="ERROR"' \
    --notification-channels=YOUR_NOTIFICATION_CHANNEL \
    --combiner=OR \
    --condition-threshold-value=1 \
    --condition-threshold-comparison=COMPARISON_GE
```

### 4. Data Validation

Add validation steps to your job:

```python
def validate_output(result_df: DataFrame, min_rows: int = 100) -> bool:
    """Validate that the output meets quality standards."""
    count = result_df.count()
    if count < min_rows:
        raise ValueError(f"Output has only {count} rows, expected at least {min_rows}")
    
    # Check for null customer_ids
    null_count = result_df.filter(F.col("customer_id").isNull()).count()
    if null_count > 0:
        raise ValueError(f"Found {null_count} rows with null customer_id")
    
    # Check feature completeness
    null_features = result_df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in result_df.columns if c != "customer_id"
    ]).collect()[0]
    
    high_null_cols = [c for c, v in null_features.asDict().items() if v > count * 0.5]
    if high_null_cols:
        logger.warning(f"High null rate in columns: {high_null_cols}")
    
    return True
```

---

## Quick Reference Commands

```bash
# === Setup ===
export PROJECT_ID="your-project"
export REGION="europe-west1"
export BUCKET_NAME="your-bucket"
export CLUSTER_NAME="feature-factory"

# === Deploy Code ===
cd src && zip -r ../feature_factory.zip features/ && cd ..
gsutil cp feature_factory.zip gs://$BUCKET_NAME/code/
gsutil cp configs/feature_config.yaml gs://$BUCKET_NAME/configs/
gsutil cp scripts/dataproc_feature_job.py gs://$BUCKET_NAME/code/

# === Create Cluster ===
gcloud dataproc clusters create $CLUSTER_NAME \
    --region=$REGION --num-workers=4 --worker-machine-type=n1-standard-8

# === Run Job ===
gcloud dataproc jobs submit pyspark gs://$BUCKET_NAME/code/dataproc_feature_job.py \
    --cluster=$CLUSTER_NAME --region=$REGION \
    --py-files=gs://$BUCKET_NAME/code/feature_factory.zip \
    -- --project-id=$PROJECT_ID --bq-dataset=credit_scoring --gcs-bucket=$BUCKET_NAME

# === Check Output ===
gsutil ls gs://$BUCKET_NAME/features/

# === Delete Cluster ===
gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION
```

---

## Support

For issues specific to:
- **Feature Factory**: Check `src/features/feature_factory.py`
- **Spark Integration**: Check `src/features/spark_feature_factory.py`
- **Dataproc Job**: Check `scripts/dataproc_feature_job.py`

For GCP-specific issues, consult:
- [Dataproc Documentation](https://cloud.google.com/dataproc/docs)
- [BigQuery Spark Connector](https://github.com/GoogleCloudDataproc/spark-bigquery-connector)
