# Variable Elimination Methodology

## Overview

The pipeline applies six sequential elimination filters to reduce the initial
feature set to a manageable pool of high-quality candidates. Each filter targets
a specific deficiency. The order ensures that computationally cheap filters run
first, and features eliminated at earlier stages do not consume resources in
later stages.

## 1. Constant Feature Elimination

### Definition

A feature is classified as constant if it has fewer than `min_unique_values`
(default: 2) distinct non-null values in the training set.

### Rationale

A feature with only one unique value provides zero discriminatory information.
It cannot separate good from bad observations because every observation has the
same value. Including such features wastes computation and can cause numerical
issues in some algorithms.

### Implementation

The filter counts `nunique()` for each feature on the training data. Features
with count below the threshold are marked for elimination. Null values are
excluded from the unique count.

### Interpretation

The elimination report lists each feature with its unique value count and
status (Kept or Eliminated). Any feature showing only 1 unique value should
be reviewed in the feature extraction stage -- it may indicate a data pipeline
issue upstream.

## 2. Missing Rate Filter

### Definition

A feature is eliminated if its missing rate (fraction of null values in the
training set) exceeds `missing_threshold` (default: 0.70).

### Rationale

When more than 70% of observations are missing for a feature, the remaining
values are unlikely to provide reliable signal. Imputation on such sparse data
introduces noise rather than information. Additionally, a high missing rate
often indicates that the feature is only available for a specific subpopulation,
which can bias the model.

### Implementation

Missing rate is computed as `X[feature].isnull().mean()` on the training set.
The threshold is configurable per run.

### Interpretation

Features just below the threshold warrant investigation. A feature with 65%
missing may still be valuable if the non-missing population is well-defined
(e.g., a feature that only applies to returning customers). Consider
alternative thresholds or dedicated missing-indicator features in such cases.

## 3. Information Value (IV)

### Formula

Information Value measures the overall predictive power of a single feature
for a binary target. The feature is binned into `n_bins` equal-frequency
groups, and for each bin `i`:

    WoE_i = ln(Distribution_of_Goods_i / Distribution_of_Bads_i)

    IV = SUM_i (Distribution_of_Goods_i - Distribution_of_Bads_i) * WoE_i

Where:
- `Distribution_of_Goods_i` = (number of goods in bin i) / (total goods)
- `Distribution_of_Bads_i` = (number of bads in bin i) / (total bads)

### Interpretation Ranges

| IV Range | Classification | Action |
|----------|---------------|--------|
| < 0.02 | Useless | Eliminate -- no predictive power |
| 0.02 - 0.10 | Weak | Keep -- contributes marginally |
| 0.10 - 0.30 | Medium | Keep -- good predictor |
| 0.30 - 0.50 | Strong | Keep -- strong predictor |
| > 0.50 | Suspicious | Eliminate -- investigate for data leakage |

### Rationale for Upper Bound

Features with IV exceeding 0.50 are eliminated by default because
suspiciously high IV often indicates:

- **Data leakage:** The feature contains information from after the outcome
  period (e.g., account closure flags computed after default).
- **Look-ahead bias:** The feature uses future data not available at
  decision time.
- **Overly specific encoding:** The feature is a near-direct encoding of
  the target (e.g., a derived "risk_category" that was created from default
  data).

The upper threshold can be raised or disabled for diagnostic purposes if
leakage has been ruled out.

### Edge Cases

- Bins with zero goods or zero bads produce undefined WoE. The implementation
  applies Laplace smoothing (adding a small count to each bin) to avoid
  division by zero.
- Features with very few distinct values may produce fewer bins than requested.
  The `min_samples_per_bin` parameter prevents bins that are too small for
  stable estimates.

## 4. Population Stability Index (PSI)

### Formula

PSI measures the shift between two distributions of the same feature across
different time periods. For each bin `i`:

    PSI = SUM_i (Actual_i - Expected_i) * ln(Actual_i / Expected_i)

Where:
- `Actual_i` = proportion of observations in bin `i` for the recent period
- `Expected_i` = proportion of observations in bin `i` for the reference period

### Interpretation Thresholds

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | Stable | No action needed |
| 0.10 - 0.25 | Moderate shift | Monitor -- acceptable for most uses |
| > 0.25 | Significant shift | Eliminate -- feature distribution has changed materially |

### Configured Checks

The pipeline supports multiple PSI check strategies, each comparing different
time periods within the training data:

- **Quarterly:** Compares each calendar quarter against the overall training
  distribution. Detects seasonal or gradual drift.
- **Yearly:** Compares year-over-year distributions. Detects structural changes
  across annual cycles.
- **Consecutive:** Compares each quarter against its predecessor. Detects sudden
  distributional jumps.
- **Date split:** Compares before vs. after a user-specified date. Useful for
  known events (policy changes, COVID impact, etc.).
- **Half split:** Compares the first and second halves of the training period.

A feature is eliminated if it exceeds the PSI threshold in **any** configured
check. This conservative approach ensures that only temporally stable features
enter the model.

### Rationale

Features whose distributions shift over time are unreliable for prediction.
A model trained on one distribution will produce biased scores when applied
to a shifted distribution. PSI provides an early warning of such instability
before the model is deployed.

## 5. Temporal Performance Filter

### Definition

This optional filter evaluates each feature's individual predictive power across
time periods and eliminates features that show degrading performance.

### Criteria

A feature is eliminated if any of the following conditions are met:

- **Minimum quarterly AUC:** The feature's single-variable AUC in any quarter
  falls below `min_quarterly_auc` (default: 0.52).
- **Maximum AUC degradation:** The difference between the feature's best and
  worst quarterly AUC exceeds `max_auc_degradation` (default: 0.05).
- **Minimum trend slope:** A linear regression of quarterly AUC values over
  time yields a slope below `min_trend_slope` (default: -0.02), indicating
  a declining trend.

### Rationale

A feature may pass IV and PSI checks on the overall training set but still
show declining predictive power over time. This filter catches features
whose relationship with the target is weakening, which would make the model
less reliable on future data.

### When to Use

This filter is disabled by default because it is computationally expensive
(requires fitting a model per feature per quarter) and can be overly
aggressive on short time series. Enable it when the training period spans
at least 8 quarters and temporal stability is a priority.

## 6. Correlation Elimination

### Method

The filter computes the pairwise absolute correlation matrix for all remaining
features. For each pair with correlation exceeding `correlation_threshold`
(default: 0.80):

1. Compare the IV scores of the two features.
2. Eliminate the feature with the lower IV score.
3. If IV scores are equal, eliminate the feature that appears later in the
   feature list (arbitrary but deterministic tiebreaker).

### Algorithm

The implementation uses a greedy iterative approach:

1. Sort all features by IV in descending order.
2. Starting from the highest-IV feature, mark it as kept.
3. For each subsequent feature, check its correlation with all previously
   kept features. If any correlation exceeds the threshold, eliminate it.
4. Continue until all features are processed.

This ensures that the highest-IV features are always retained.

### Rationale

Highly correlated features provide redundant information. Including both
members of a correlated pair in the model:

- Inflates the apparent importance of the underlying signal.
- Increases variance of coefficient estimates (multicollinearity).
- Wastes model capacity on redundant dimensions.
- Can cause instability in feature importance rankings.

### Correlation Methods

The pipeline supports three correlation methods:

- **Pearson** (default): Measures linear correlation. Fast and appropriate for
  numeric features with approximately linear relationships.
- **Spearman**: Measures monotonic correlation. More robust to outliers and
  nonlinear monotonic relationships.
- **Kendall**: Measures concordance. Most robust but slowest. Best for ordinal
  features or small datasets.
