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

### Input Data Options

The feature factory supports **two data input modes**:

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Separate Tables** | Two tables: `applications` and `bureau` | Standard setup, data stored separately |
| **Pre-Joined Table** | Single table with all data joined | Data already joined in BigQuery, or using a view |

### Option 1: Separate Tables (Default)

#### Applications Table Schema

```sql
CREATE TABLE `your-project.credit_scoring.applications` (
    customer_id STRING NOT NULL,           -- Unique customer identifier
    application_id STRING NOT NULL,        -- Unique application identifier
    application_date DATE NOT NULL,        -- Date of credit application
    applicant_type STRING,                 -- 'PRIMARY' or 'CO_APPLICANT'
    target INT64,                          -- Target variable (1=default, 0=no default)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Bureau/Credit Table Schema

```sql
CREATE TABLE `your-project.credit_scoring.bureau` (
    customer_id STRING NOT NULL,           -- Links to applications.customer_id
    application_id STRING,                 -- Links to applications.application_id
    product_type STRING NOT NULL,          -- Credit type (see Product Types below)
    opening_date DATE NOT NULL,            -- Credit opening date
    total_amount FLOAT64,                  -- Original credit amount
    monthly_payment FLOAT64,               -- Monthly payment amount
    remaining_term_months INT64,           -- Remaining term in months
    default_date DATE,                     -- NULL if not defaulted
    recovery_date DATE,                    -- NULL if not recovered
    closure_date DATE,                     -- NULL if still active
    duration_months INT64,                 -- Original term in months
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Expected Product Types

The `product_type` column should contain one of these values:

| Code | Description |
|------|-------------|
| `IL` or `INSTALLMENT_LOAN` | Installment Loan (Bireysel Kredi) |
| `IS` or `INSTALLMENT_SALE` | Installment Sale (Taksitli Satış) |
| `CF` or `CASH_FACILITY` | Cash Facility / Credit Card (Nakit Avans) |
| `MG` or `MORTGAGE` | Mortgage (Konut Kredisi) |

### Option 2: Pre-Joined Table (Recommended for Existing Data)

If your data is already joined in BigQuery, use `--joined-table`:

```sql
-- Example: Create a view that joins your existing tables
CREATE VIEW `your-project.credit_scoring.credit_data_joined` AS
SELECT 
    a.customer_id,
    a.application_id,
    a.application_date,
    a.applicant_type,
    a.target,
    b.product_type,
    b.opening_date,
    b.total_amount,
    b.monthly_payment,
    b.remaining_term_months,
    b.default_date,
    b.recovery_date,
    b.closure_date,
    b.duration_months
FROM `your-project.credit_scoring.applications` a
LEFT JOIN `your-project.credit_scoring.bureau` b
    ON a.customer_id = b.customer_id 
    AND a.application_id = b.application_id;
```

### Column Mapping (For Custom Field Names)

If your columns have different names (e.g., Turkish names), use `--column-mapping`:

#### Expected Column Names

| Category | Expected Name | Description |
|----------|---------------|-------------|
| **Application** | `customer_id` | Unique customer ID |
| **Application** | `application_id` | Unique application ID |
| **Application** | `application_date` | Application date |
| **Application** | `applicant_type` | PRIMARY or CO_APPLICANT |
| **Application** | `target` | Target variable (0/1) |
| **Bureau** | `product_type` | Credit type code |
| **Bureau** | `opening_date` | Credit opening date |
| **Bureau** | `total_amount` | Original credit amount |
| **Bureau** | `monthly_payment` | Monthly payment |
| **Bureau** | `remaining_term_months` | Remaining term |
| **Bureau** | `default_date` | Default date (NULL if none) |
| **Bureau** | `recovery_date` | Recovery date (NULL if none) |
| **Bureau** | `closure_date` | Closure date (NULL if active) |
| **Bureau** | `duration_months` | Original term |

#### Column Mapping Example

```bash
# Turkish column names -> Expected names
--column-mapping='{
    "musteri_no": "customer_id",
    "basvuru_no": "application_id",
    "basvuru_tarihi": "application_date",
    "basvuru_tipi": "applicant_type",
    "hedef": "target",
    "urun_kodu": "product_type",
    "acilis_tarihi": "opening_date",
    "kredi_tutari": "total_amount",
    "aylik_taksit": "monthly_payment",
    "kalan_vade": "remaining_term_months",
    "temerrut_tarihi": "default_date",
    "tahsilat_tarihi": "recovery_date",
    "kapanis_tarihi": "closure_date",
    "vade_ay": "duration_months"
}'
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

### Command Line Arguments Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--project-id` | ✅ | - | GCP Project ID |
| `--bq-dataset` | ✅ | - | BigQuery dataset name |
| `--gcs-bucket` | ✅ | - | GCS bucket name (without gs://) |
| `--output-path` | ❌ | `features/output` | Output path in GCS bucket |
| `--applications-table` | ❌ | `applications` | Applications table name |
| `--bureau-table` | ❌ | `bureau` | Bureau table name |
| `--joined-table` | ❌ | - | **Pre-joined table name** (use instead of separate tables) |
| `--column-mapping` | ❌ | - | **JSON column name mapping** |
| `--config-path` | ❌ | - | Path to feature_config.yaml |
| `--date-filter` | ❌ | - | SQL WHERE clause for filtering |
| `--mode` | ❌ | `simple` | `simple` (recommended) or `distributed` |
| `--num-partitions` | ❌ | `200` | Partitions for distributed mode |
| `--save-format` | ❌ | `parquet` | Output format: parquet, csv, json |

### Example 1: Separate Tables (Default)

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
    --applications-table=applications \
    --bureau-table=bureau \
    --output-path=features/$(date +%Y%m%d_%H%M%S) \
    --config-path=gs://$BUCKET_NAME/configs/feature_config.yaml \
    --mode=simple
```

### Example 2: Pre-Joined Table (Recommended)

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
    --joined-table=credit_data_joined \
    --output-path=features/$(date +%Y%m%d_%H%M%S) \
    --config-path=gs://$BUCKET_NAME/configs/feature_config.yaml \
    --mode=simple
```

### Example 3: Pre-Joined Table with Column Mapping (Turkish Names)

```bash
# Create the column mapping JSON
COLUMN_MAPPING='{
    "musteri_no": "customer_id",
    "basvuru_no": "application_id",
    "basvuru_tarihi": "application_date",
    "basvuru_tipi": "applicant_type",
    "hedef": "target",
    "urun_kodu": "product_type",
    "acilis_tarihi": "opening_date",
    "kredi_tutari": "total_amount",
    "aylik_taksit": "monthly_payment",
    "kalan_vade": "remaining_term_months",
    "temerrut_tarihi": "default_date",
    "tahsilat_tarihi": "recovery_date",
    "kapanis_tarihi": "closure_date",
    "vade_ay": "duration_months"
}'

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
    --joined-table=kredi_verisi \
    --column-mapping="$COLUMN_MAPPING" \
    --output-path=features/$(date +%Y%m%d_%H%M%S) \
    --config-path=gs://$BUCKET_NAME/configs/feature_config.yaml \
    --mode=simple
```

### Example 4: With Date Filter

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
    --joined-table=credit_data_joined \
    --output-path=features/2024/january \
    --config-path=gs://$BUCKET_NAME/configs/feature_config.yaml \
    --date-filter="application_date >= '2024-01-01' AND application_date < '2024-02-01'" \
    --mode=simple
```

### Example 5: Distributed Mode (For Large Datasets)

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
    --joined-table=credit_data_joined \
    --output-path=features/$(date +%Y%m%d_%H%M%S) \
    --config-path=gs://$BUCKET_NAME/configs/feature_config.yaml \
    --mode=distributed \
    --num-partitions=200
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
export DATASET_NAME="credit_scoring"

# === Deploy Code ===
cd src && zip -r ../feature_factory.zip features/ && cd ..
gsutil cp feature_factory.zip gs://$BUCKET_NAME/code/
gsutil cp config/feature_config.yaml gs://$BUCKET_NAME/configs/
gsutil cp scripts/dataproc_feature_job.py gs://$BUCKET_NAME/code/

# === Create Cluster ===
gcloud dataproc clusters create $CLUSTER_NAME \
    --region=$REGION \
    --image-version=2.1-debian11 \
    --num-workers=4 \
    --worker-machine-type=n1-standard-8 \
    --initialization-actions=gs://$BUCKET_NAME/init/init_script.sh \
    --properties="spark:spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2"

# === Run Job (Pre-Joined Table) ===
gcloud dataproc jobs submit pyspark gs://$BUCKET_NAME/code/dataproc_feature_job.py \
    --cluster=$CLUSTER_NAME --region=$REGION \
    --py-files=gs://$BUCKET_NAME/code/feature_factory.zip \
    --properties="spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2,spark.driver.memory=16g" \
    -- \
    --project-id=$PROJECT_ID \
    --bq-dataset=$DATASET_NAME \
    --gcs-bucket=$BUCKET_NAME \
    --joined-table=credit_data_joined \
    --output-path=features/$(date +%Y%m%d_%H%M%S) \
    --mode=simple

# === Run Job with Column Mapping ===
COLUMN_MAPPING='{"musteri_no":"customer_id","basvuru_no":"application_id","basvuru_tarihi":"application_date","urun_kodu":"product_type","acilis_tarihi":"opening_date","kredi_tutari":"total_amount","temerrut_tarihi":"default_date"}'

gcloud dataproc jobs submit pyspark gs://$BUCKET_NAME/code/dataproc_feature_job.py \
    --cluster=$CLUSTER_NAME --region=$REGION \
    --py-files=gs://$BUCKET_NAME/code/feature_factory.zip \
    --properties="spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2,spark.driver.memory=16g" \
    -- \
    --project-id=$PROJECT_ID \
    --bq-dataset=$DATASET_NAME \
    --gcs-bucket=$BUCKET_NAME \
    --joined-table=kredi_verisi \
    --column-mapping="$COLUMN_MAPPING" \
    --output-path=features/$(date +%Y%m%d_%H%M%S) \
    --mode=simple

# === Check Output ===
gsutil ls gs://$BUCKET_NAME/features/
gsutil du -h gs://$BUCKET_NAME/features/

# === Delete Cluster ===
gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet
```

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Column not found: customer_id` | Column names don't match expected | Use `--column-mapping` to map your columns |
| `BigQuery connector not found` | Missing Spark package | Add `--properties="spark.jars.packages=..."` |
| `Out of memory on driver` | Dataset too large for simple mode | Increase `spark.driver.memory` or use distributed mode |
| `py-files not found` | Missing zip file | Run `gsutil ls gs://$BUCKET/code/feature_factory.zip` |
| `JSON parsing error` | Invalid column mapping JSON | Validate JSON with `echo $COLUMN_MAPPING | jq .` |

### Checking Your Column Names

Before running the job, verify your BigQuery table columns:

```bash
# List columns in your table
bq show --schema --format=prettyjson $PROJECT_ID:$DATASET_NAME.your_table | jq '.[].name'
```

### Verifying Output

```bash
# Check if output was created
gsutil ls gs://$BUCKET_NAME/features/

# Check row count in output
gsutil cat gs://$BUCKET_NAME/features/your_output/_SUCCESS && echo "Job succeeded!"

# Preview output data (first 10 rows)
bq load --source_format=PARQUET --autodetect temp_features gs://$BUCKET_NAME/features/your_output/*.parquet
bq head --max_rows=10 temp_features
bq rm -f temp_features
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

### Getting Help

1. **Check logs**: `gcloud dataproc jobs describe JOB_ID --region=$REGION`
2. **SSH to master**: `gcloud compute ssh $CLUSTER_NAME-m --zone=$ZONE`
3. **View Spark UI**: Use port forwarding with `gcloud compute ssh ... -- -L 4040:localhost:4040`

