# Credit Scoring Feature Factory

High-performance feature engineering library for credit scoring models. Generates 900+ credit risk features with optimized execution for both local development and distributed Spark/Dataproc environments.

## 🚀 Features

- **941 credit risk features** across 34 feature categories
- **Optimized execution**: 360+ applications/second on a single machine
- **Dual execution modes**: Local Pandas or Distributed Spark
- **GCP Dataproc ready**: BigQuery → Spark → GCS pipeline
- **Config-driven**: YAML configuration for feature definitions

## 📊 Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| Amount Features | 300+ | Sum, avg, max, min, std by product/window/status |
| Count Features | 125 | Credit counts by multiple dimensions |
| Ratio Features | 80+ | Portfolio composition ratios |
| Temporal Features | 50+ | Credit age, history length, time to default |
| Trend Features | 40+ | Credit velocity, growth trends |
| Risk Signals | 20+ | Overdraft, overlimit, stress indicators |
| Behavioral Features | 50+ | Default patterns, recovery rates |
| Payment Features | 40+ | Monthly payments, DTI proxies |
| And more... | 200+ | Diversity, complexity, anomaly features |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FeatureFactory                               │
│  - Config-driven feature generation                              │
│  - Vectorized Pandas operations                                  │
│  - Parallel multiprocessing support                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SparkFeatureFactory                            │
│  - PySpark adapter for distributed processing                    │
│  - pandas_udf for worker-level execution                         │
│  - BigQuery/GCS integration                                      │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ahmetalinuhoglu/credit-scoring-feature-factory.git
cd credit-scoring-feature-factory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Quick Start

### Local Execution

```python
from src.features.feature_factory import FeatureFactory
import pandas as pd

# Load data
applications = pd.read_parquet('data/applications.parquet')
bureau = pd.read_parquet('data/bureau.parquet')

# Initialize factory
factory = FeatureFactory()

# Generate features (parallel execution)
features = factory.generate_all_features(
    applications_df=applications,
    credit_bureau_df=bureau,
    parallel=True,
    n_jobs=-1  # Use all CPU cores
)

print(f"Generated {features.shape[1]} features for {features.shape[0]} applications")
```

### Spark/Dataproc Execution

```python
from pyspark.sql import SparkSession
from src.features.spark_feature_factory import SparkFeatureFactory

# Create Spark session
spark = SparkSession.builder.appName("FeatureGeneration").getOrCreate()

# Load data
apps_sdf = spark.read.parquet("gs://bucket/applications")
bureau_sdf = spark.read.parquet("gs://bucket/bureau")

# Initialize Spark factory
factory = SparkFeatureFactory(spark)

# Generate features (distributed)
features_sdf = factory.generate_all_features(apps_sdf, bureau_sdf)

# Save to GCS
features_sdf.write.parquet("gs://bucket/features")
```

## ☁️ GCP Dataproc Deployment

See [DATAPROC_FEATURE_GENERATION_GUIDE.md](docs/DATAPROC_FEATURE_GENERATION_GUIDE.md) for complete instructions.

### Quick Deploy

```bash
# Set environment variables
export PROJECT_ID="your-project"
export BUCKET_NAME="your-bucket"
export CLUSTER_NAME="feature-factory"

# Deploy code to GCS
./scripts/deploy_to_dataproc.sh

# Run the job
./scripts/deploy_to_dataproc.sh --run
```

## 🧪 Testing

```bash
# Run unit tests
make test

# Run with coverage
make test-coverage

# Run benchmarks
make benchmark
```

## 📈 Performance

| Environment | Dataset | Throughput |
|-------------|---------|------------|
| Local (Sequential) | 11.5k apps | 65 apps/sec |
| Local (Parallel) | 11.5k apps | 360 apps/sec |
| Spark (Docker) | 11.5k apps | 242 apps/sec |
| Dataproc (4 workers) | 1M apps | ~5,000 apps/sec* |

*Estimated based on cluster size

## 📁 Project Structure

```
├── config/                    # Configuration files
│   └── feature_config.yaml    # Feature definitions
├── src/
│   └── features/
│       ├── feature_factory.py        # Core feature generator
│       └── spark_feature_factory.py  # Spark adapter
├── scripts/
│   ├── dataproc_feature_job.py      # Dataproc job script
│   └── deploy_to_dataproc.sh        # Deployment helper
├── docs/
│   └── DATAPROC_FEATURE_GENERATION_GUIDE.md
├── tests/
└── docker-compose.yml         # Local Spark testing
```

## 🔑 Key Configuration

### Product Types
- INSTALLMENT_LOAN
- INSTALLMENT_SALE  
- CASH_FACILITY
- MORTGAGE

### Time Windows
- All time
- Last 3 months
- Last 6 months
- Last 12 months
- Last 24 months

### Status Filters
- All
- Active
- Defaulted
- Recovered

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.
