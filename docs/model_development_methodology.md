# Model Development Methodology

## 1. Purpose and Scope

This pipeline automates the development of credit scoring models for application-level default prediction. It takes a feature matrix with a binary target (0 = good, 1 = bad) and produces:

- A trained XGBoost model with forward-selected features.
- An Excel report covering every elimination step, model performance across time periods, validation results, and configuration details.
- Logs for full reproducibility.

The pipeline is designed for retail lending (consumer credit, auto loans, mortgages) but works for any binary classification use case with time-indexed observations.


## 2. Data Requirements

### 2.1 Input Format

The pipeline accepts a single flat file (CSV or Parquet) with one row per application. All features must already be computed.

### 2.2 Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `application_id` | string/int | Unique application identifier |
| `customer_id` | string/int | Customer identifier |
| `date` (configurable) | date string | Application date, used for train/OOT splitting |
| `target` (configurable) | int (0/1) | Binary default indicator |

All other columns are treated as candidate features unless listed in `id_columns` or `exclude_columns`.

### 2.3 Date Conventions

- Dates should be in `YYYY-MM-DD` format or any format parseable by `pd.to_datetime`.
- The pipeline splits data into Train (before `train_end_date`), Test (random holdout from Train), and OOT (after `train_end_date`, split by calendar quarter).
- At least 2 quarters of OOT data are recommended for meaningful stability assessment.


## 3. Pipeline Steps

### 3.1 Data Loading and Splitting

**What it does:** Loads the input file, identifies feature columns (excluding ID, date, target, and user-specified exclusions), and splits into Train, Test, and OOT datasets.

- **Train/Test split:** Stratified random split on the target variable within the training period. Default test size is 20%.
- **OOT split:** All data after `train_end_date` is grouped by calendar quarter (e.g., "OOT_2024Q3").

**Why:** Out-of-time validation is the gold standard for credit scoring model assessment. Random holdout tests generalization; OOT tests temporal stability.

### 3.2 Constant Feature Elimination

**What it does:** Removes features with fewer than `min_unique_values` (default: 2) distinct non-null values in the training set.

**Why:** Constant features provide zero information and waste computation. A feature that has the same value for every observation cannot discriminate between good and bad.

**Threshold:** `min_unique_values = 2`. Features with only one unique value (or all-null) are eliminated.

**How to interpret:** The `01_Constant` sheet lists each feature with its unique-value count and whether it was kept or eliminated.

### 3.3 Missing Value Elimination

**What it does:** Removes features where more than `threshold` (default: 70%) of training observations are missing.

**Why:** Features with excessive missing values are unreliable. Imputation on >70% missing data introduces more noise than signal.

**Threshold:** `missing_threshold = 0.70`. A feature is eliminated if its missing rate exceeds this value.

**How to interpret:** The `02_Missing` sheet shows each feature's missing rate. Eliminated features appear in pink.

### 3.4 Information Value (IV) Elimination

**What it does:** Computes the Information Value for each remaining feature and eliminates those outside the range `[min_iv, max_iv]` (default: [0.02, 0.50]).

**Why:**

- **Low IV (<0.02):** The feature has almost no predictive power for the target. Including it adds noise.
- **High IV (>0.50):** Suspiciously high IV may indicate data leakage, look-ahead bias, or an overly specific feature.

**Computation:** Features are binned into `n_bins` (default: 10) equal-frequency groups. For each bin, the Weight of Evidence (WoE) is computed, and IV is the sum of (Distribution of Goods - Distribution of Bads) * WoE across all bins.

**How to interpret:** The `03_IV` sheet lists features with their IV scores and classification (Useless, Weak, Medium, Strong, Suspicious). Green rows are kept; pink rows are eliminated.

### 3.5 PSI Stability Elimination

**What it does:** Computes the Population Stability Index (PSI) of each feature's distribution across time periods within the training data. Eliminates features exceeding `psi_threshold` (default: 0.25) in any configured check.

**Why:** A feature whose distribution shifts over time is unstable and may produce unreliable predictions on future data. PSI quantifies distributional shift.

**Configured checks:**
- **Quarterly:** Compares each quarter against the overall training distribution.
- **Yearly:** Compares year-over-year distributions.
- **Consecutive:** Compares adjacent quarters.

**Thresholds:** PSI interpretation:
| PSI | Interpretation |
|-----|---------------|
| < 0.10 | Stable |
| 0.10 - 0.25 | Moderate shift (warning) |
| > 0.25 | Significant shift (eliminated) |

**How to interpret:** The `04_PSI` sheet shows PSI values for each feature and check type. Features exceeding the critical threshold are eliminated.

### 3.6 Correlation Elimination

**What it does:** Identifies pairs of features with absolute correlation above `correlation_threshold` (default: 0.90). From each pair, the feature with lower IV is eliminated.

**Why:** Highly correlated features provide redundant information. Including both adds collinearity, inflates importance of the underlying signal, and wastes model capacity.

**Method:** Pearson correlation by default (configurable to Spearman or Kendall).

**How to interpret:** The `05_Corr_Pairs` sheet shows correlated pairs with their correlation coefficient and which feature was kept (higher IV wins).

### 3.7 Forward Feature Selection

**What it does:** Starting from an empty feature set, iteratively adds the feature that improves test AUC the most, stopping when no feature improves AUC by more than `auc_threshold` (default: 0.0001) or `max_features` is reached.

**Why:** Even after elimination, there may be 40-100+ candidate features. Forward selection builds a parsimonious model by greedily picking the most informative features one at a time.

**Model used:** XGBoost with the configured hyperparameters. At each step, a fresh model is trained with the current feature set plus each candidate.

**How to interpret:** The `06_Selection` sheet shows the order features were added, the AUC at each step, and the marginal AUC improvement.

### 3.8 Model Evaluation

**What it does:** Trains the final model on the selected features and evaluates it on Train, Test, and each OOT quarter. Produces metrics (AUC, Gini, KS), lift tables (decile analysis), and feature importance.

**Why:** Multi-period evaluation reveals both discrimination power and temporal stability. Lift tables show how well the model ranks borrowers.

**Metrics computed:**
- **AUC (Area Under ROC Curve):** Probability that the model ranks a random bad higher than a random good. Range: 0.5 (random) to 1.0 (perfect).
- **Gini:** 2 * AUC - 1. Equivalent to AUC but scaled to [0, 1].
- **KS (Kolmogorov-Smirnov):** Maximum separation between cumulative good and bad distributions.

**How to interpret:**
- `07_Performance`: One row per period with AUC, Gini, KS, sample size, and bad count.
- `07_Lift_Tables`: Decile analysis per period. Decile 1 should have the highest bad rate (top risk).
- `07_Importance`: Feature importance from the final model (gain-based for XGBoost).


## 4. Validation Checks

### 4.1 Pre-Pipeline Data Checks (DataValidator)

These run BEFORE the pipeline starts and can block execution on critical failures.

| Check | Severity | What it tests |
|-------|----------|--------------|
| Non-empty dataset | CRITICAL | DataFrame has rows |
| Target column exists | CRITICAL | Target column is present |
| Target is binary | CRITICAL | Only 0/1 values |
| Target has no nulls | CRITICAL | No missing target values |
| Bad rate within range | WARNING/CRITICAL | Bad rate between 1% and 50% |
| Date column exists | CRITICAL | Date column present and parseable |
| Date range coverage | WARNING | At least 2 quarters of data |
| Features are numeric | WARNING | All feature columns are numeric |
| No duplicate IDs | WARNING | No duplicate primary keys |
| Sufficient sample size | WARNING | At least 500 rows |
| Leakage detection | WARNING | No feature with single-variable AUC > 0.95 |

**What failures mean:**
- **CRITICAL failures block pipeline execution.** Fix the data issue before proceeding.
- **WARNING failures are logged** but do not block. They indicate potential data quality issues that should be investigated.

### 4.2 Post-Pipeline Model Checks (ModelValidator)

These run AFTER model evaluation and produce advisory results.

| Check | Severity | What it tests |
|-------|----------|--------------|
| Discrimination (AUC) | CRITICAL | AUC >= min_auc (default 0.65) for every period |
| Discrimination (Gini) | WARNING | Gini meets implied threshold |
| Overfitting | WARNING | Train-Test AUC gap <= max_overfit_gap (default 5pp) |
| OOT stability | WARNING | OOT AUC within max_oot_degradation (default 8pp) of test |
| Score PSI | WARNING | Score distribution PSI < max_score_psi (default 0.25) |
| Feature concentration | WARNING | No feature > max_feature_concentration (default 50%) of importance |
| WoE monotonicity | INFO | WoE trend is monotonic per feature |
| OOT sample size | WARNING | Each OOT period has >= min_oot_samples (default 30) bads |


## 5. Output Excel Report

### Sheet Reference

| Sheet | Content |
|-------|---------|
| 00_Summary | Run metadata, data stats, feature elimination funnel, key metrics, validation flag counts |
| 01_Constant | Constant feature elimination details |
| 02_Missing | Missing value elimination details |
| 03_IV | Information Value elimination details |
| 04_PSI | PSI stability elimination details |
| 05_Corr_Pairs | Correlated feature pairs and which was kept |
| 06_Selection | Forward selection order, cumulative AUC, marginal improvement |
| 07_Performance | Per-period AUC, Gini, KS, sample sizes |
| 07_Lift_Tables | Decile analysis per period |
| 07_Importance | Feature importance ranking |
| 08_Validation | Full validation check results with pass/fail/warning status |
| 09_Score_Distribution | Score histogram data per period, score PSI values |
| 10_Config | Complete pipeline configuration as section/parameter/value table |

### Color Coding

- **Blue header row:** Column headers.
- **Green rows:** Kept features, passing checks.
- **Pink rows:** Eliminated features.
- **Yellow rows (validation):** Warning checks.
- **Red rows (validation):** Failed checks.

### How to Read the Report

1. Start with **00_Summary** for the big picture: how many features survived each step, key AUC metrics, and whether validation passed.
2. Check **08_Validation** for any failures or warnings.
3. Review **07_Performance** to compare Train, Test, and OOT metrics. Stable AUC across periods indicates a robust model.
4. Use **07_Lift_Tables** to verify the model ranks risk correctly (Decile 1 should capture the highest bad rate).
5. Examine **07_Importance** for feature concentration -- no single feature should dominate.
6. Review elimination sheets (01-05) only if you need to understand why specific features were removed.
7. Check **10_Config** to confirm the correct thresholds were applied.


## 6. Configuration

The pipeline is configured via a YAML file or `PipelineConfig` Pydantic model. Key parameters:

### Data Section
| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_path` | `data/sample/sample_features.parquet` | Path to input file |
| `target_column` | `target` | Binary target column name |
| `date_column` | `date` | Date column for temporal splitting |
| `id_columns` | `[application_id, customer_id]` | Columns to exclude from features |

### Splitting Section
| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_end_date` | `2024-06-30` | Last date in training set |
| `test_size` | `0.20` | Fraction of training data held out as test |

### Steps Section
| Parameter | Default | Description |
|-----------|---------|-------------|
| `constant.min_unique_values` | `2` | Minimum distinct values |
| `missing.threshold` | `0.70` | Maximum allowed missing rate |
| `iv.min_iv` | `0.02` | Minimum IV to keep |
| `iv.max_iv` | `0.50` | Maximum IV to keep |
| `psi.threshold` | `0.25` | PSI elimination threshold |
| `correlation.threshold` | `0.90` | Maximum allowed correlation |
| `selection.auc_threshold` | `0.0001` | Minimum AUC improvement per feature |

### Validation Section
| Parameter | Default | Description |
|-----------|---------|-------------|
| `checks.min_auc` | `0.65` | Minimum acceptable AUC |
| `checks.max_overfit_gap` | `0.05` | Maximum Train-Test AUC gap |
| `checks.max_oot_degradation` | `0.08` | Maximum OOT AUC drop from test |
| `checks.max_score_psi` | `0.25` | Maximum score distribution PSI |
| `checks.max_feature_concentration` | `0.50` | Maximum single-feature importance share |
| `checks.min_oot_samples` | `30` | Minimum bads per OOT quarter |


## 7. Glossary

### AUC (Area Under the ROC Curve)
Measures the probability that the model assigns a higher risk score to a randomly chosen bad observation than to a randomly chosen good one. Range: 0.5 (no discrimination) to 1.0 (perfect). In credit scoring, AUC > 0.70 is generally considered acceptable for application scoring.

### Gini Coefficient
Gini = 2 * AUC - 1. Equivalent to the AUC but rescaled to [0, 1]. Sometimes preferred for reporting because it centers on zero for a random model.

### IV (Information Value)
Summarizes the predictive power of a single feature for a binary target. Computed from WoE bins:

    IV = SUM( (% of Goods_i - % of Bads_i) * WoE_i )

| IV Range | Interpretation |
|----------|---------------|
| < 0.02 | Useless -- no predictive power |
| 0.02 - 0.10 | Weak predictor |
| 0.10 - 0.30 | Medium predictor |
| 0.30 - 0.50 | Strong predictor |
| > 0.50 | Suspicious -- check for leakage |

### WoE (Weight of Evidence)
For each bin of a feature, WoE = ln(% of Goods / % of Bads). Positive WoE means the bin has more goods than average; negative means more bads. WoE transforms features into a scale that is directly related to log-odds.

### PSI (Population Stability Index)
Measures shift between two distributions. Computed similarly to IV but comparing two time periods rather than good vs. bad:

    PSI = SUM( (% Actual_i - % Expected_i) * ln(% Actual_i / % Expected_i) )

| PSI | Interpretation |
|-----|---------------|
| < 0.10 | Stable -- no significant change |
| 0.10 - 0.25 | Moderate shift -- monitor |
| > 0.25 | Significant shift -- investigate |

### KS (Kolmogorov-Smirnov Statistic)
Maximum absolute difference between the cumulative distribution of goods and bads when ordered by score. Higher KS means better separation. Typically reported as a percentage (e.g., KS = 45%).

### OOT (Out-of-Time)
Data from periods after the training window, used to assess temporal stability. In credit scoring, OOT validation is more important than random holdout because it simulates real deployment conditions.

### Lift
Ratio of the bad rate in a particular decile to the overall bad rate. Lift > 1 in top deciles means the model is successfully concentrating risk. Decile 1 (highest risk) should have the highest lift.

### Decile Analysis
Observations are ranked by model score and divided into 10 equal groups (deciles). For each decile, the bad rate, cumulative bad capture rate, and lift are computed. This is the primary tool for assessing model rank-ordering.

### Overfit / Overfitting
When a model performs significantly better on training data than on test/OOT data. Indicates the model has memorized training noise rather than learning general patterns. Measured by the Train-Test AUC gap.

### Forward Feature Selection
Greedy algorithm that starts with an empty feature set and iteratively adds the feature that most improves model performance (AUC on test set). Stops when marginal improvement falls below a threshold.

### Stratified Split
A train/test split that preserves the target class ratio in both sets. Important for imbalanced datasets (low bad rate) to ensure both sets have enough bads for reliable evaluation.

### Monotonicity
A feature exhibits monotonic WoE when the WoE trend consistently increases or decreases across ordered bins. Monotonic features are easier to interpret and often preferred in credit scoring for regulatory reasons.

### Bad Rate
The proportion of observations that defaulted (target = 1). Also called "default rate" or "event rate." The overall bad rate drives the baseline for lift calculations and influences the practical utility of model scores.

### ROC Curve (Receiver Operating Characteristic)
A plot of True Positive Rate (sensitivity) vs. False Positive Rate (1 - specificity) at all possible classification thresholds. The area under this curve is the AUC.

### Cumulative Capture Rate
In a lift table, the cumulative percentage of all bads captured by the top N deciles. For example, "Top 3 deciles capture 65% of bads" means that if you reject the riskiest 30% of applicants, you would have caught 65% of all defaults.

### Regularization
Techniques to prevent overfitting by penalizing model complexity. In XGBoost: `reg_alpha` (L1), `reg_lambda` (L2), `gamma` (minimum loss reduction for splits), `min_child_weight`, and `max_depth` all serve as regularization knobs.


## 8. Typical Workflow

### 8.1 Initial Run

1. Prepare the feature matrix (one row per application, all features pre-computed).
2. Set the `train_end_date` to leave at least 2-3 quarters for OOT.
3. Run the pipeline with default thresholds.
4. Review the Excel report, starting with 00_Summary and 08_Validation.
5. Check 07_Performance for AUC stability across periods.

### 8.2 Iterating on Thresholds

If the model underperforms, consider:

- **Too few features selected:** Lower `iv.min_iv` (e.g., from 0.02 to 0.01), or increase `correlation.threshold` (e.g., from 0.90 to 0.95).
- **Overfitting (Train AUC >> Test AUC):** Reduce `model.params.max_depth`, increase `model.params.min_child_weight`, or increase `model.params.reg_lambda`.
- **Unstable OOT performance:** Tighten `psi.threshold` (e.g., from 0.25 to 0.15) to remove shifting features. Consider shortening the training window.
- **Too many correlated features getting through:** Lower `correlation.threshold` (e.g., from 0.90 to 0.80).

### 8.3 Feature Engineering Feedback Loop

The elimination report tells you which features survive. Common patterns:

- If most features are eliminated at the IV step, the features may be too noisy or the target definition needs review.
- If many features fail PSI, the population may be changing rapidly. Consider time-windowed features or more robust aggregations.
- If correlation eliminates many features, the feature set may have too many similar variants of the same underlying signal. Consolidate before extraction.

### 8.4 When to Retrain

- When score PSI exceeds 0.25 on live data (population shift).
- When OOT AUC drops below the minimum threshold.
- When business rules or product definitions change materially.
- As a standard practice, at least annually.


## 9. XGBoost Hyperparameter Guide

The pipeline uses XGBoost for both forward selection and the final model. Key parameters and their impact:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_depth` | 6 | Maximum tree depth. Lower = simpler model, less overfitting. Start at 4-6 for credit scoring. |
| `learning_rate` | 0.1 | Step size shrinkage. Lower values need more `n_estimators` but generalize better. |
| `n_estimators` | 300 | Maximum number of boosting rounds. Early stopping prevents using all of them. |
| `early_stopping_rounds` | 30 | Stop if no improvement for N rounds on the test set. |
| `subsample` | 0.8 | Fraction of training data used per tree. Lower = more regularization. |
| `colsample_bytree` | 0.8 | Fraction of features sampled per tree. Lower = more diversity. |
| `min_child_weight` | 1 | Minimum sum of instance weight in a child. Higher = more conservative splits. |
| `gamma` | 0 | Minimum loss reduction required to make a split. Higher = fewer splits. |
| `reg_alpha` | 0 | L1 regularization on leaf weights. |
| `reg_lambda` | 1 | L2 regularization on leaf weights. Higher = smoother predictions. |

For credit scoring, a common starting point is `max_depth=4-6`, `learning_rate=0.05-0.1`, `n_estimators=300-500` with `early_stopping_rounds=30-50`.


## 10. Troubleshooting

### Pipeline Fails at Data Validation

**Symptom:** The pipeline stops before any elimination step and logs a CRITICAL failure.

**Solution:** Read the validation report (logged to console and saved in the 08_Validation sheet if the report gets generated). Common causes:
- Target column has non-binary values (e.g., 0, 1, 2): fix the target definition.
- Target has nulls: filter out rows with missing target.
- Date column not parseable: check date format.
- Fewer than 2 quarters of data: provide more data or adjust `train_end_date`.

### All Features Eliminated at IV Step

**Symptom:** 0 features remain after IV elimination.

**Possible causes:**
- Target is poorly defined (very low signal).
- Features are raw and need aggregation/transformation.
- `min_iv` is set too high (try 0.01).
- `max_iv` is set too low (try 1.0 to disable the upper cap temporarily for diagnosis).

### Forward Selection Selects Only 1-2 Features

**Symptom:** The model uses very few features.

**Possible causes:**
- `auc_threshold` is too high (try 0.00005 or 0.00001).
- The candidate features are all highly correlated with the first selected feature.
- Review `06_Selection` to see AUC gains for each candidate.

### OOT AUC Is Much Lower Than Test AUC

**Symptom:** Model works well on test set but degrades on OOT quarters.

**Possible causes:**
- Population has shifted (check feature PSI on OOT periods).
- The model overfits to the training period's distribution.
- Target definition or data collection changed after `train_end_date`.

**Solutions:**
- Tighten PSI threshold.
- Use a shorter, more recent training window.
- Add time-invariant features.
- Regularize the model more aggressively.

### Excel Report Is Missing Sheets

**Symptom:** Some expected sheets are not in the output workbook.

**Possible causes:**
- The corresponding pipeline step was disabled or produced no data.
- A step threw an exception that was caught but resulted in empty results.
- Check the log file for warnings during report generation.

### Memory Issues on Large Datasets

**Symptom:** The pipeline runs out of memory or is very slow.

**Solutions:**
- Use Parquet input instead of CSV (more memory-efficient).
- Reduce the number of candidate features before pipeline input.
- Increase `missing.threshold` to aggressively filter sparse features early.
- For forward selection, set `max_features` to limit search depth.


## 11. Best Practices for Credit Scoring

### Feature Design

- Use time-windowed aggregations (e.g., count of events in last 6/12/24 months).
- Include both level features (total count, average amount) and trend features (change over time).
- Avoid forward-looking features that would not be available at the time of decision.
- Test features for business interpretability: can you explain why this feature should predict default?

### Model Complexity

- Credit scoring models should be interpretable and stable, not maximally accurate.
- 8-15 features is a typical sweet spot for XGBoost credit scoring models.
- Deeper trees and more features improve training AUC but increase overfit risk.

### Temporal Validation

- Always evaluate on OOT data. A model that works on random holdout but fails OOT is useless in production.
- Look for consistent AUC across quarters, not just average AUC.
- A model with AUC 0.80 stable across 4 quarters is better than one with AUC 0.85 that drops to 0.70 in Q4.

### Monitoring After Deployment

- Track score PSI monthly. Trigger model review if PSI > 0.10.
- Monitor actual default rates by score band. Misalignment indicates calibration drift.
- Revalidate the model at least annually, or when PSI exceeds 0.25.

### Documentation Standards

- Record every model development run with its configuration and results.
- Keep the Excel report as the primary artifact for model governance.
- Use the 10_Config sheet to trace exactly which parameters produced the results.
- Store logs for audit trail purposes.


## 12. Pipeline Architecture Reference

### Module Layout

```
src/
  config/
    schema.py         # Pydantic config models (PipelineConfig, StepsConfig, etc.)
    loader.py         # YAML / dict config loading
  validation/
    data_checks.py    # DataValidator (pre-pipeline)
    model_checks.py   # ModelValidator (post-pipeline)
  components/
    data_splitter.py      # Train/Test/OOT splitting
    constant_filter.py    # Constant elimination
    missing_filter.py     # Missing value elimination
    iv_filter.py          # IV-based elimination
    psi_filter.py         # PSI stability elimination
    correlation_filter.py # Correlation elimination
    feature_selector.py   # Forward selection
    model_evaluator.py    # Multi-period evaluation
  reporting/
    excel_reporter.py     # Full Excel workbook generation
    report_exporter.py    # Charts (PDF/PNG) generation
  pipeline/
    orchestrator.py   # End-to-end pipeline execution
  model_development/
    pipeline.py       # Legacy pipeline (pre-refactor)
    excel_reporter.py # Legacy Excel reporter
```

### Data Flow

```
Input File (.csv/.parquet)
  |
  v
DataValidator.validate()  -->  ValidationReport (blocks on CRITICAL)
  |
  v
DataSplitter.run()  -->  Train, Test, OOT quarter DataFrames
  |
  v
ConstantFilter  -->  StepResult (eliminated features list)
  |
  v
MissingFilter   -->  StepResult
  |
  v
IVFilter        -->  StepResult (+ IV scores for correlation tiebreaker)
  |
  v
PSIFilter       -->  StepResult
  |
  v
CorrelationFilter -->  StepResult (+ correlation pairs detail)
  |
  v
ForwardFeatureSelector  -->  Selected features, selection history
  |
  v
ModelEvaluator  -->  Performance table, lift tables, importance
  |
  v
ModelValidator.validate()  -->  ValidationReport (advisory)
  |
  v
ExcelReporter.generate()  -->  model_dev_YYYYMMDD_HHMMSS.xlsx
```

### StepResult Interface

Each filter component returns a `StepResult` dataclass:

```python
@dataclass
class StepResult:
    step_name: str          # E.g., "01_Constant"
    input_features: List[str]
    output_features: List[str]
    eliminated_features: List[str]
    results_df: pd.DataFrame  # Detailed per-feature results
    metadata: Dict[str, Any]
    duration_seconds: float
```

The `results_df` DataFrame typically includes a "Status" column ("Kept" or "Eliminated") that the Excel reporter uses for color-coding.


## 13. Reproducibility

Every pipeline run is fully reproducible given:

1. **The input data file** (path recorded in 00_Summary and 10_Config).
2. **The configuration** (saved in 10_Config sheet and optionally as YAML).
3. **The random seed** (`reproducibility.global_seed`, default 42) -- controls train/test splitting and XGBoost randomness.
4. **The pipeline version** (use git commit hash in production).

The log file (`logs/model_dev_YYYYMMDD_HHMMSS.log`) captures every step with timestamps, feature counts, and threshold decisions at DEBUG level.

To reproduce a previous run, load the same input file and apply the same configuration (from the 10_Config sheet or saved YAML). The same seed guarantees identical train/test splits and model results.
