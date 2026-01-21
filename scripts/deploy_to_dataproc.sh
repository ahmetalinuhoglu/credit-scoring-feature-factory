#!/bin/bash
#
# deploy_to_dataproc.sh
# 
# Deploys the Feature Factory code to GCS and optionally runs a job on Dataproc.
#
# Usage:
#   ./scripts/deploy_to_dataproc.sh                    # Deploy code only
#   ./scripts/deploy_to_dataproc.sh --run              # Deploy and run job
#   ./scripts/deploy_to_dataproc.sh --run --simple     # Deploy and run in simple mode
#

set -e

# ============================================================================
# Configuration - UPDATE THESE VALUES
# ============================================================================

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"
BUCKET_NAME="${BUCKET_NAME:-your-bucket-name}"
CLUSTER_NAME="${CLUSTER_NAME:-feature-factory-cluster}"
DATASET_NAME="${DATASET_NAME:-credit_scoring}"

# ============================================================================
# Script Logic
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
RUN_JOB=false
SIMPLE_MODE=false
DATE_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            RUN_JOB=true
            shift
            ;;
        --simple)
            SIMPLE_MODE=true
            shift
            ;;
        --date-filter)
            DATE_FILTER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--run] [--simple] [--date-filter 'SQL_CONDITION']"
            echo ""
            echo "Options:"
            echo "  --run          Run the feature generation job after deploying"
            echo "  --simple       Use simple mode (collect to driver) instead of distributed"
            echo "  --date-filter  Add a date filter to the query"
            echo ""
            echo "Environment Variables:"
            echo "  PROJECT_ID     GCP Project ID (default: your-project-id)"
            echo "  REGION         GCP Region (default: europe-west1)"
            echo "  BUCKET_NAME    GCS Bucket name (default: your-bucket-name)"
            echo "  CLUSTER_NAME   Dataproc cluster name (default: feature-factory-cluster)"
            echo "  DATASET_NAME   BigQuery dataset name (default: credit_scoring)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Step 1: Package the Feature Factory
# ============================================================================

log_info "Packaging Feature Factory..."

cd "$PROJECT_ROOT/src"
rm -f ../feature_factory.zip
zip -r ../feature_factory.zip features/ -x "*.pyc" -x "__pycache__/*" -x "*.DS_Store"
cd "$PROJECT_ROOT"

log_info "Created feature_factory.zip ($(du -h feature_factory.zip | cut -f1))"

# ============================================================================
# Step 2: Upload to GCS
# ============================================================================

log_info "Uploading to gs://$BUCKET_NAME/..."

# Upload code
gsutil -q cp feature_factory.zip gs://$BUCKET_NAME/code/
gsutil -q cp scripts/dataproc_feature_job.py gs://$BUCKET_NAME/code/
gsutil -q cp configs/feature_config.yaml gs://$BUCKET_NAME/configs/

log_info "Uploaded files:"
echo "  - gs://$BUCKET_NAME/code/feature_factory.zip"
echo "  - gs://$BUCKET_NAME/code/dataproc_feature_job.py"
echo "  - gs://$BUCKET_NAME/configs/feature_config.yaml"

# Clean up local zip
rm -f feature_factory.zip

# ============================================================================
# Step 3: Run Job (if requested)
# ============================================================================

if [ "$RUN_JOB" = true ]; then
    log_info "Submitting Dataproc job..."
    
    OUTPUT_PATH="features/$(date +%Y%m%d_%H%M%S)"
    
    if [ "$SIMPLE_MODE" = true ]; then
        MODE="simple"
        EXTRA_PROPS="spark.driver.memory=16g,spark.driver.maxResultSize=8g"
    else
        MODE="distributed"
        EXTRA_PROPS="spark.executor.memory=6g"
    fi
    
    JOB_ARGS=(
        "--project-id=$PROJECT_ID"
        "--bq-dataset=$DATASET_NAME"
        "--gcs-bucket=$BUCKET_NAME"
        "--output-path=$OUTPUT_PATH"
        "--config-path=gs://$BUCKET_NAME/configs/feature_config.yaml"
        "--mode=$MODE"
    )
    
    if [ -n "$DATE_FILTER" ]; then
        JOB_ARGS+=("--date-filter=$DATE_FILTER")
    fi
    
    log_info "Job configuration:"
    echo "  - Mode: $MODE"
    echo "  - Output: gs://$BUCKET_NAME/$OUTPUT_PATH"
    if [ -n "$DATE_FILTER" ]; then
        echo "  - Filter: $DATE_FILTER"
    fi
    
    # Submit job
    gcloud dataproc jobs submit pyspark \
        "gs://$BUCKET_NAME/code/dataproc_feature_job.py" \
        --cluster="$CLUSTER_NAME" \
        --region="$REGION" \
        --py-files="gs://$BUCKET_NAME/code/feature_factory.zip" \
        --properties="spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2,$EXTRA_PROPS" \
        -- "${JOB_ARGS[@]}"
    
    log_info "Job completed! Output saved to: gs://$BUCKET_NAME/$OUTPUT_PATH"
else
    log_info "Deployment complete!"
    echo ""
    echo "To run a job, use:"
    echo "  $0 --run"
    echo ""
    echo "Or submit manually with:"
    echo "  gcloud dataproc jobs submit pyspark gs://$BUCKET_NAME/code/dataproc_feature_job.py \\"
    echo "    --cluster=$CLUSTER_NAME --region=$REGION \\"
    echo "    --py-files=gs://$BUCKET_NAME/code/feature_factory.zip \\"
    echo "    --properties=\"spark.jars.packages=com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2\" \\"
    echo "    -- --project-id=$PROJECT_ID --bq-dataset=$DATASET_NAME --gcs-bucket=$BUCKET_NAME"
fi
