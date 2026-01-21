# Credit Scoring Feature Factory

High-performance feature engineering library for credit scoring models. Generates 900+ credit risk features with optimized execution for both local development and distributed Spark/Dataproc environments.

## 🚀 Features

- **941 credit risk features** across 34 feature categories
- **Optimized execution**: 360+ applications/second on a single machine
- **Dual execution modes**: Local Pandas or Distributed Spark
- **GCP Dataproc ready**: BigQuery → Spark → GCS pipeline
- **Config-driven**: YAML configuration for feature definitions

## 📊 Feature Categories

The 941 features are organized into **7 main categories**:

| # | Category | Features | Description |
|---|----------|----------|-------------|
| 1 | **📈 Volume & Amount** | 350+ | Credit counts, amounts, and aggregations (sum, avg, max, min, std) by product type, time window, and status |
| 2 | **⏱️ Temporal & History** | 120+ | Credit age, history length, time-to-default, days since events, seasonal patterns |
| 3 | **📊 Ratios & Composition** | 100+ | Portfolio mix ratios, product concentration, secured vs unsecured proportions |
| 4 | **📉 Trends & Velocity** | 80+ | Credit growth trends, acquisition velocity, 6m vs 12m comparisons |
| 5 | **⚠️ Risk & Default** | 150+ | Default counts, recovery rates, stress signals, overlimit/overdraft events, default severity |
| 6 | **💳 Payment & Obligations** | 80+ | Monthly payments, remaining terms, DTI proxies, payment burden metrics |
| 7 | **🔍 Behavioral & Patterns** | 60+ | Credit sequences, burst detection, product transitions, complexity scores, anomaly indicators |

### Detailed Breakdown

<details>
<summary><b>1. Volume & Amount Features (350+)</b></summary>

- **Count Features**: Total credits, active credits, closed credits by product
- **Amount Aggregations**: Sum, average, max, min, std of credit amounts
- **Dimensional Slicing**: By product type (IL, IS, CF, MG), time window (3m, 6m, 12m, 24m), status (active, defaulted, recovered)
- **Examples**: `total_credit_count`, `installment_loan_total_amount_last_12m`, `mortgage_average_amount`
</details>

<details>
<summary><b>2. Temporal & History Features (120+)</b></summary>

- **Age Metrics**: Oldest credit age, newest credit age, average credit age
- **History Length**: Credit history span in months
- **Time-to-Event**: Days to default, recovery duration
- **Recency**: Days since last credit, days since last default
- **Examples**: `oldest_credit_age_months`, `days_since_last_default`, `avg_time_to_default_days`
</details>

<details>
<summary><b>3. Ratios & Composition Features (100+)</b></summary>

- **Product Mix**: Share of each product type in portfolio
- **Secured Ratio**: Proportion of mortgages vs unsecured credits
- **Concentration Metrics**: HHI index, diversity scores
- **Amount Ratios**: Amount-weighted proportions
- **Examples**: `secured_ratio`, `installment_loan_amt_to_total_ratio`, `product_diversity_ratio`
</details>

<details>
<summary><b>4. Trends & Velocity Features (80+)</b></summary>

- **Credit Velocity**: Rate of credit acquisition in recent periods
- **Amount Velocity**: Rate of credit amount growth
- **Period Comparisons**: 6m vs 12m trends, acceleration metrics
- **Growth Patterns**: Increasing/decreasing credit activity
- **Examples**: `credit_velocity_3m`, `amt_trend_6m_vs_12m`, `default_trend_6m_vs_12m`
</details>

<details>
<summary><b>5. Risk & Default Features (150+)</b></summary>

- **Default Counts**: Ever defaulted, multiple defaults, by product
- **Recovery Metrics**: Recovery rate, recovery time
- **Stress Signals**: Overdraft events, overlimit occurrences
- **Default Severity**: Average/max default amounts
- **Risk Flags**: Current default status, financial stress indicators
- **Examples**: `default_count_ever`, `recovery_success_rate`, `has_overlimit`, `max_default_severity`
</details>

<details>
<summary><b>6. Payment & Obligation Features (80+)</b></summary>

- **Payment Metrics**: Total monthly payments, average payment
- **Term Features**: Remaining term months, original term
- **DTI Proxies**: Payment burden indicators
- **Maturity Patterns**: Short-term vs long-term credit mix
- **Examples**: `total_monthly_payment`, `avg_remaining_term_months`, `payment_to_amount_ratio`
</details>

<details>
<summary><b>7. Behavioral & Pattern Features (60+)</b></summary>

- **Sequence Patterns**: First/last product type, product transitions
- **Burst Detection**: Multiple credits in short periods
- **Interval Analysis**: Time between credit applications
- **Complexity Scores**: Portfolio sophistication metrics
- **Anomaly Indicators**: Unusual patterns in credit behavior
- **Examples**: `first_product_installment_loan`, `credit_burst_3m`, `product_transition_count`
</details>

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
