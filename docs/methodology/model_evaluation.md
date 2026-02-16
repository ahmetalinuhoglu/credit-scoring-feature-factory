# Model Evaluation Methodology

## Overview

After the final model is trained on the selected features, the pipeline evaluates
its performance across all data splits: Train, Test, and each out-of-time (OOT)
quarter. This document describes each evaluation metric and technique.

## Discrimination Metrics

### AUC (Area Under the ROC Curve)

AUC measures the probability that the model assigns a higher risk score to a
randomly chosen defaulter than to a randomly chosen non-defaulter.

- **Range:** 0.5 (random model) to 1.0 (perfect discrimination).
- **Typical targets:** AUC > 0.70 is acceptable for application scoring;
  AUC > 0.80 is strong.
- **Interpretation:** AUC is threshold-independent -- it evaluates the model's
  ranking ability without committing to a specific cutoff.

### Gini Coefficient

Gini is a rescaled version of AUC:

    Gini = 2 * AUC - 1

- **Range:** 0.0 (random) to 1.0 (perfect).
- **Usage:** Preferred in some organizations because it centers on zero for a
  random model, making the magnitude of improvement more intuitive.

### KS Statistic (Kolmogorov-Smirnov)

KS is the maximum absolute difference between the cumulative distribution
functions of scores for goods and bads:

    KS = max |F_good(s) - F_bad(s)|

- **Range:** 0.0 to 1.0.
- **Interpretation:** The score threshold at which KS is achieved represents
  the point of maximum separation between the two populations. Higher KS means
  better rank-ordering.
- **Relationship to AUC:** KS and AUC are related but not equivalent. A model
  can have high AUC but low KS if separation is spread across many thresholds
  rather than concentrated at one point.

## Lift and Capture Rate Analysis

### Decile Analysis

Observations are ranked by model score (highest risk first) and divided into
10 equal-sized groups (deciles). For each decile, the pipeline computes:

- **Bad rate:** Proportion of defaults within the decile.
- **Lift:** Bad rate in the decile divided by the overall bad rate. Decile 1
  should have the highest lift.
- **Cumulative capture rate:** Percentage of all defaults captured by the top
  N deciles.

### Interpretation

A well-performing model should show:

- Monotonically decreasing bad rates from Decile 1 to Decile 10.
- Lift > 2.0 in the top decile for a model with moderate discrimination.
- Top 3 deciles capturing 50-70% of all defaults.

Deviations from monotonicity in the middle deciles are acceptable if the top
and bottom deciles are well-separated. Non-monotonicity in the extremes
(e.g., Decile 10 having a higher bad rate than Decile 9) warrants
investigation.

## Bootstrap Confidence Intervals

### Method

The pipeline computes confidence intervals for AUC using bootstrap resampling:

1. Draw `n_iterations` (default: 1000) bootstrap samples with replacement
   from the evaluation dataset.
2. Compute AUC on each bootstrap sample.
3. Report the percentile-based confidence interval at the configured
   confidence level (default: 95%).

### Interpretation

Narrow confidence intervals indicate stable AUC estimates. Wide intervals
(e.g., more than 0.05 wide) suggest insufficient sample size or high
variance in the score distribution. OOT periods with small samples often
produce wider intervals.

## Score PSI (Model Score Stability)

### Definition

Score PSI measures the stability of the model's predicted probability
distribution between the training period and each OOT period. It uses the
same PSI formula as feature PSI but applied to model scores rather than
individual features.

### Thresholds

| Score PSI | Interpretation |
|-----------|---------------|
| < 0.10 | Stable scores |
| 0.10 - 0.25 | Moderate drift -- monitor closely |
| > 0.25 | Significant drift -- consider model recalibration or retraining |

### Rationale

Even if individual features are stable, the model's score distribution can
shift due to changes in the joint distribution of features. Score PSI provides
a holistic stability measure at the model output level.

## SHAP Interpretability

### Method

The pipeline computes SHAP (SHapley Additive exPlanations) values using the
TreeSHAP algorithm, which is efficient for tree-based models like XGBoost.

### Outputs

- **SHAP summary:** For each feature, the mean absolute SHAP value across all
  observations, indicating its average impact on model output.
- **SHAP plots:** Beeswarm and bar plots saved as image files.

### Interpretation

SHAP values decompose each prediction into feature-level contributions.
A feature with high mean absolute SHAP value has a large average impact on
the model's predicted probability. Unlike gain-based importance, SHAP accounts
for feature interactions and provides directional information (positive or
negative contribution).

## Calibration

### Purpose

Calibration ensures that the model's predicted probabilities reflect true
default rates. A model predicting 10% default probability should see
approximately 10% of such predictions actually default.

### Methods

The pipeline supports three calibration approaches:

- **Platt Scaling:** Fits a logistic regression on the model's raw
  probabilities to produce calibrated probabilities. Effective when the
  miscalibration is approximately sigmoid-shaped.
- **Isotonic Regression:** Fits a non-parametric isotonic regression
  (piecewise constant, non-decreasing) on the raw probabilities. More
  flexible than Platt scaling but prone to overfitting on small datasets.
- **Temperature Scaling:** Divides the model's logits by a learned
  temperature parameter. A single-parameter method that preserves the
  model's ranking while adjusting confidence.

### Metrics

- **Brier Score:** Mean squared error between predicted probabilities and
  actual outcomes. Lower is better. Reported before and after calibration.
- **Expected Calibration Error (ECE):** Average absolute difference between
  predicted probability and observed frequency within probability bins.
  Lower is better.

## Confusion Matrix Metrics

### Definition

The pipeline computes confusion matrix metrics at multiple probability
thresholds (configurable, default: 0.1, 0.2, 0.3, 0.4, 0.5):

- **Precision (Positive Predictive Value):** Of all observations predicted as
  bad, what fraction actually defaulted.
- **Recall (Sensitivity / True Positive Rate):** Of all actual defaults,
  what fraction did the model correctly identify.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **Type I Error (False Positive Rate):** Fraction of goods incorrectly
  classified as bad. Represents unnecessary rejections in a lending context.
- **Type II Error (False Negative Rate):** Fraction of bads incorrectly
  classified as good. Represents defaults that the model failed to flag.

### Interpretation

In credit scoring, the optimal threshold depends on the business cost
structure. A lower threshold increases Recall (catches more defaults) but
also increases Type I errors (rejects more good applicants). The confusion
matrix at multiple thresholds helps stakeholders select the operating point
that balances risk and volume.
