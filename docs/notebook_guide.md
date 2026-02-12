# Notebook Guide: Running the Model Development Pipeline

This guide shows how to run the credit scoring model development pipeline interactively in a Jupyter notebook. There are two modes:

- **Step-by-step mode** — Run each component individually, inspect results between steps
- **Full pipeline mode** — Run everything in one call

Both modes use the same YAML config and produce identical results.

---

## Quick Start (Full Pipeline in 5 Lines)

```python
import sys; sys.path.insert(0, "..")
from src.config.loader import load_config
from src.pipeline.orchestrator import PipelineOrchestrator
from src.io.output_manager import OutputManager
import pandas as pd

config = load_config("../config/model_development.yaml")
df = pd.read_parquet(config.data.input_path)
om = OutputManager(config)
pipeline = PipelineOrchestrator(config, om)
results = pipeline.run_all(df)
print(results.summary())
```

---

## Step-by-Step Guide

### 1. Setup

```python
import sys
from pathlib import Path

# If running from notebooks/ directory
project_root = str(Path.cwd().parent) if Path.cwd().name == "notebooks" else str(Path.cwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import random

from src.config.loader import load_config
from src.components import (
    DataSplitter,
    ConstantFilter,
    MissingFilter,
    IVFilter,
    PSIFilter,
    CorrelationFilter,
    ForwardFeatureSelector,
    ModelEvaluator,
)
from src.io.output_manager import OutputManager
from src.validation.data_checks import DataValidator
from src.validation.model_checks import ModelValidator
from src.reporting.excel_reporter import ExcelReporter
from src.tracking.experiment_tracker import ExperimentTracker
```

### 2. Load Config

Load the master YAML config. Everything is driven by this single file.

```python
config = load_config("config/model_development.yaml")
```

**Override values for this session** (without editing the YAML):

```python
config = load_config(
    "config/model_development.yaml",
    overrides={
        "data.input_path": "data/sample/sample_features.parquet",
        "splitting.train_end_date": "2024-06-30",
        "steps.iv.min_iv": 0.03,
        "model.params.max_depth": 4,
        "reproducibility.global_seed": 123,
    },
)
```

**Inspect the config:**

```python
# View specific sections
print(f"Input: {config.data.input_path}")
print(f"Train end date: {config.splitting.train_end_date}")
print(f"IV range: [{config.steps.iv.min_iv}, {config.steps.iv.max_iv}]")
print(f"XGBoost params: {config.model.params}")
print(f"Seed: {config.reproducibility.global_seed}")
```

### 3. Set Seeds for Reproducibility

```python
seed = config.reproducibility.global_seed
np.random.seed(seed)
random.seed(seed)
```

### 4. Load Data

```python
df = pd.read_parquet(config.data.input_path)
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
print(f"Date range: {df[config.data.date_column].min()} to {df[config.data.date_column].max()}")
print(f"Bad rate: {df[config.data.target_column].mean():.2%}")
```

### 5. Data Quality Checks (Optional but Recommended)

Run automated data quality checks before the pipeline. These catch issues
like non-binary targets, duplicate IDs, missing date columns, or data leakage.

```python
validator = DataValidator(config)
data_report = validator.validate(df)
print(data_report.summary())

# If critical failures, inspect details:
if data_report.has_critical_failures:
    for check in data_report.checks:
        if check.status.value == "FAIL":
            print(f"  FAIL: {check.check_name} - {check.message}")
            print(f"        Recommendation: {check.recommendation}")
```

### 6. Split Data

```python
splitter = DataSplitter(
    data_config=config.data,
    splitting_config=config.splitting,
    seed=config.reproducibility.global_seed,
)
split = splitter.split(df)

# Unpack
X_train = split.train[split.feature_columns]
y_train = split.train[config.data.target_column]
X_test = split.test[split.feature_columns]
y_test = split.test[config.data.target_column]
features = list(split.feature_columns)

print(f"Train: {len(split.train):,} rows (bad rate: {y_train.mean():.2%})")
print(f"Test:  {len(split.test):,} rows (bad rate: {y_test.mean():.2%})")
for label in split.oot_labels:
    qdf = split.oot_quarters[label]
    print(f"OOT {label}: {len(qdf):,} rows (bad rate: {qdf[config.data.target_column].mean():.2%})")
```

### 7. Step 1 — Constant Filter

Remove features with zero or near-zero variance.

```python
const_filter = ConstantFilter(config.steps.constant)
const_result = const_filter.fit(X_train[features], y_train)
features = const_result.output_features
print(const_result.summary())

# Inspect eliminated features
print(const_result.results_df[const_result.results_df["Status"] == "Eliminated"][["Feature", "Unique_Count"]].head(10))
```

### 8. Step 2 — Missing Filter

Remove features with excessive missing values.

```python
missing_filter = MissingFilter(config.steps.missing)
missing_result = missing_filter.fit(
    X_train[features], y_train,
    X_test=X_test[features],
    X_oot={label: split.oot_quarters[label][features] for label in split.oot_labels},
)
features = missing_result.output_features
print(missing_result.summary())

# Inspect high-missing features
high_missing = missing_result.results_df[missing_result.results_df["Train_Missing_Rate"] > 0.3]
print(high_missing[["Feature", "Train_Missing_Rate", "Status"]].head(10))
```

### 9. Step 3 — IV Filter

Remove features with low or suspiciously high Information Value.

```python
iv_filter = IVFilter(config.steps.iv)
iv_result = iv_filter.fit(X_train[features], y_train)
features = iv_result.output_features
print(iv_result.summary())

# IV distribution
iv_df = iv_result.results_df
print("\nIV Distribution:")
print(iv_df["IV_Category"].value_counts())

# Top features by IV
kept = iv_df[iv_df["Status"] == "Kept"].sort_values("IV_Score", ascending=False)
print(f"\nTop 10 features by IV:")
print(kept[["Feature", "IV_Score", "IV_Category", "Univariate_AUC"]].head(10).to_string(index=False))

# Access IV scores for downstream steps
iv_scores = iv_filter.iv_scores_

# Access WoE binning (useful for scorecard deployment)
# iv_filter.woe_bins_  # Dict[feature_name -> List[WoEBin]]
```

### 10. Step 4 — PSI Filter

Remove features with unstable distributions within training data.

```python
psi_filter = PSIFilter(config.steps.psi)
psi_result = psi_filter.fit(
    X_train[features], y_train,
    train_dates=split.train[config.data.date_column],
)
features = psi_result.output_features
print(psi_result.summary())

# Inspect unstable features
unstable = psi_result.results_df[psi_result.results_df["Status"] == "Eliminated"]
if len(unstable) > 0:
    print(f"\nUnstable features: {len(unstable)}")
    print(unstable[["Feature", "Max_PSI", "Mean_PSI"]].head(10))
```

### 11. Step 5 — Correlation Filter

Remove correlated features, keeping the one with higher IV.

```python
corr_filter = CorrelationFilter(config.steps.correlation)
corr_result = corr_filter.fit(
    X_train[features], y_train,
    iv_scores=iv_scores,
)
features = corr_result.output_features
print(corr_result.summary())

# Inspect correlated pairs
if corr_filter.corr_pairs_df_ is not None and len(corr_filter.corr_pairs_df_) > 0:
    print("\nCorrelated pairs removed:")
    print(corr_filter.corr_pairs_df_[["Feature_A", "Feature_B", "Correlation", "Decision"]].head(10))

# VIF for surviving features
if corr_filter.vif_df_ is not None:
    print("\nVIF for surviving features:")
    print(corr_filter.vif_df_.sort_values("VIF", ascending=False).head(10))
```

### 12. Step 6 — Forward Feature Selection

Select features via forward stepwise selection with XGBoost.

```python
selector = ForwardFeatureSelector(
    config=config.steps.selection,
    model_config=config.model,
    seed=config.reproducibility.global_seed,
)
selection_result = selector.fit(
    X_train[features], y_train,
    X_test=X_test[features],
    y_test=y_test,
    iv_scores=iv_scores,
)
selected_features = selection_result.output_features
print(selection_result.summary())

# Selection steps
sel_df = selection_result.results_df
added = sel_df[sel_df["Status"] == "Added"]
print(f"\nSelected {len(added)} features:")
print(added[["Feature", "Feature_IV", "AUC_After", "AUC_Improvement"]].to_string(index=False))

# The trained model is stored on the selector
model = selector.model_
```

### 13. Step 7 — Model Evaluation

Evaluate the final model on train, test, and each OOT quarter.

```python
evaluator = ModelEvaluator(config.evaluation)
eval_result = evaluator.fit(
    X_train[selected_features], y_train,
    model=model,
    X_test=X_test[selected_features],
    y_test=y_test,
    oot_data={
        label: split.oot_quarters[label]
        for label in split.oot_labels
    },
    selected_features=selected_features,
    target_column=config.data.target_column,
)
print(eval_result.summary())

# Performance table
print("\nPerformance by period:")
print(evaluator.performance_df_.to_string(index=False))

# Feature importance
print("\nFeature importance:")
print(evaluator.importance_df_[["Feature", "Importance", "Rank"]].to_string(index=False))

# Lift tables (per period)
for period, lt in evaluator.lift_tables_.items():
    print(f"\nLift table — {period}:")
    print(lt.head(5).to_string(index=False))

# Score PSI (if enabled)
if evaluator.score_psi_df_ is not None:
    print("\nScore PSI:")
    print(evaluator.score_psi_df_.to_string(index=False))
```

### 14. Model Validation Checks

Automated quality checks on the final model.

```python
model_validator = ModelValidator(config)
model_report = model_validator.validate(
    performance_df=evaluator.performance_df_,
    importance_df=evaluator.importance_df_,
    woe_bins=iv_filter.woe_bins_,
    score_psi_df=evaluator.score_psi_df_,
)
print(model_report.summary())

# Detailed check results
report_df = model_report.to_dataframe()
print(report_df[["check_name", "status", "severity", "message"]].to_string(index=False))
```

### 15. Save Outputs and Generate Report

```python
import time

om = OutputManager(config)

# Save config snapshot
om.save_config_snapshot(config)

# Save split indices for reproducibility
om.save_artifact("train_test_split", split.split_indices_df, fmt="parquet", subdir="data")

# Save step results
for step_result in [const_result, missing_result, iv_result, psi_result, corr_result, selection_result, eval_result]:
    om.save_step_results(step_result.step_name, {"results": step_result.results_df})

# Save correlation matrix
if config.output.save_correlation_matrix and corr_filter.correlation_matrix_ is not None:
    om.save_artifact("correlation_matrix", corr_filter.correlation_matrix_, fmt="parquet", subdir="steps/05_correlation")

# Save model
if config.output.save_model and model is not None:
    model.save_model(str(om.get_step_dir("06_selection") / "model.json"))

# Generate Excel report
if config.output.generate_excel:
    reporter = ExcelReporter(config)
    excel_path = str(om.run_dir / "reports" / "model_development_report.xlsx")
    reporter.generate(excel_path, {
        "summary": {
            "Run ID": om.run_id,
            "Input File": config.data.input_path,
            "Train End Date": config.splitting.train_end_date,
            "Total Features": len(split.feature_columns),
            "Selected Features": len(selected_features),
        },
        "step_results": [
            {"step_name": r.step_name, "results_df": r.results_df}
            for r in [const_result, missing_result, iv_result, psi_result, corr_result]
        ],
        "corr_pairs_df": corr_filter.corr_pairs_df_,
        "selection_df": selection_result.results_df,
        "performance_df": evaluator.performance_df_,
        "lift_tables": evaluator.lift_tables_,
        "importance_df": evaluator.importance_df_,
        "validation_report": model_report,
        "config": config,
    })
    print(f"Excel report saved: {excel_path}")

# Save run metadata
om.save_run_metadata()
om.mark_complete()
print(f"All outputs saved to: {om.run_dir}")
```

### 16. Track Experiment

```python
tracker = ExperimentTracker()

# Extract key metrics
perf = evaluator.performance_df_
train_auc = float(perf[perf["Period"] == "Train"]["AUC"].iloc[0])
test_auc = float(perf[perf["Period"] == "Test"]["AUC"].iloc[0])
oot_rows = perf[perf["Period"].str.startswith("OOT")]
oot_mean_auc = float(oot_rows["AUC"].mean()) if len(oot_rows) > 0 else None

tracker.log_run(
    run_id=om.run_id,
    config=config,
    metrics={
        "n_features_selected": len(selected_features),
        "train_auc": train_auc,
        "test_auc": test_auc,
        "oot_mean_auc": oot_mean_auc,
    },
    duration=eval_result.duration_seconds,
    status="success",
    notes="Step-by-step notebook run",
)
print(f"Run logged. History: {tracker.get_history().shape[0]} total runs")
```

---

## Full Pipeline Mode (Alternative)

If you don't need to inspect intermediate results, run everything in one call:

```python
from src.config.loader import load_config
from src.pipeline.orchestrator import PipelineOrchestrator
from src.io.output_manager import OutputManager
import pandas as pd

# Load config (with optional overrides)
config = load_config("config/model_development.yaml")

# Load data
df = pd.read_parquet(config.data.input_path)

# Create output manager and pipeline
om = OutputManager(config)
pipeline = PipelineOrchestrator(config, om)

# Run everything
results = pipeline.run_all(df)

# Inspect results
print(results.summary())
print(f"Status: {results.status}")
print(f"Final features: {results.final_features}")
print(f"Total duration: {results.total_duration:.1f}s")

# Individual step summaries
for step in results.steps:
    print(f"  {step.summary()}")
```

---

## Common Recipes

### Run with different data

```python
config = load_config(
    "config/model_development.yaml",
    overrides={"data.input_path": "/path/to/your/features.parquet"},
)
```

### Change the train/OOT cutoff date

```python
config = load_config(
    "config/model_development.yaml",
    overrides={"splitting.train_end_date": "2024-09-30"},
)
```

### Tighten IV thresholds

```python
config = load_config(
    "config/model_development.yaml",
    overrides={
        "steps.iv.min_iv": 0.05,     # Keep only medium+ IV
        "steps.iv.max_iv": 0.40,     # Stricter suspicious threshold
    },
)
```

### Use Spearman correlation instead of Pearson

```python
config = load_config(
    "config/model_development.yaml",
    overrides={"steps.correlation.method": "spearman"},
)
```

### Cap the number of selected features

```python
config = load_config(
    "config/model_development.yaml",
    overrides={"steps.selection.max_features": 15},
)
```

### Tune XGBoost hyperparameters

```python
config = load_config(
    "config/model_development.yaml",
    overrides={
        "model.params.max_depth": 4,
        "model.params.learning_rate": 0.05,
        "model.params.n_estimators": 500,
        "model.params.subsample": 0.7,
    },
)
```

### Add a custom PSI date split check

Edit `config/model_development.yaml`:
```yaml
steps:
  psi:
    checks:
      - type: "quarterly"
      - type: "yearly"
      - type: "date_split"
        date: "2024-04-01"
        label: "Pre/Post Apr 2024"
```

### Compare two runs

```python
from src.tracking.run_comparison import RunComparison

rc = RunComparison()
print(rc.list_runs())  # List all runs

comparison = rc.compare(["20260212_143052_a1b2c3", "20260212_150000_d4e5f6"])
print(comparison)

# Config diff between two runs
diff = rc.diff_configs("20260212_143052_a1b2c3", "20260212_150000_d4e5f6")
print(diff)
```

### Resume from a specific step

If you have pre-computed results and want to resume from a later step:

```python
# Run only from correlation filter onwards
result = pipeline.run_from("05_correlation", df)
```

---

## Output Structure

Each run creates a self-contained directory:

```
outputs/model_development/{run_id}/
├── config.yaml                          # Frozen config snapshot
├── run_metadata.json                    # Git hash, versions, timing
├── data/
│   ├── train_test_split.parquet         # Split indices for reproducibility
│   └── feature_summary.parquet
├── steps/
│   ├── 01_constant/results.parquet
│   ├── 02_missing/results.parquet
│   ├── 03_iv/results.parquet
│   ├── 04_psi/results.parquet
│   ├── 05_correlation/
│   │   ├── results.parquet
│   │   └── correlation_matrix.parquet
│   ├── 06_selection/
│   │   ├── results.parquet
│   │   └── model.json
│   └── 07_evaluation/
│       └── results.parquet
├── reports/
│   └── model_development_report.xlsx    # 13-sheet Excel report
└── logs/
    └── pipeline.log
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: src` | Working directory wrong | Run `sys.path.insert(0, "..")` or start notebook from project root |
| `ValidationError` on config load | Invalid config values | Check error message — shows which field failed (e.g., iv_min > iv_max) |
| `ValueError: X_test required` | Missing kwargs in fit() | Pass `X_test=X_test, y_test=y_test` to ForwardFeatureSelector.fit() |
| `No OOT data found` | train_end_date too late | Set train_end_date earlier so data exists after it |
| `XGBoost early_stopping` error | Wrong XGBoost version | Ensure xgboost >= 2.0; early_stopping_rounds goes in constructor |
| Empty results after IV filter | Thresholds too strict | Relax iv_min (lower) or iv_max (higher) |
| PSI filter skips all checks | No dates provided | Pass `train_dates=split.train[config.data.date_column]` to PSIFilter.fit() |
