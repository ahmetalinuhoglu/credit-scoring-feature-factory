# Pipeline Methodology Overview

## Purpose

This document describes the methodology behind the credit scoring model development
pipeline. The pipeline takes a pre-computed feature matrix with a binary default
indicator and produces a trained model with a documented audit trail.

## Pipeline Architecture

The pipeline follows a sequential elimination-selection-tuning-evaluation workflow:

1. **Variable Elimination** -- Remove unsuitable features through a series of
   statistical filters applied in order of increasing computational cost.
2. **Feature Selection** -- From the surviving candidates, select a parsimonious
   subset using sequential (forward or backward) selection with cross-validation.
3. **Multicollinearity Check** -- Apply Variance Inflation Factor (VIF) analysis
   to the selected features and remove any that introduce excessive collinearity.
4. **Hyperparameter Tuning** -- Optimize XGBoost hyperparameters using Optuna
   with a stability-aware objective that penalizes AUC variance across periods.
5. **Model Evaluation** -- Evaluate the final model on Train, Test, and
   out-of-time (OOT) quarterly datasets.
6. **Reporting** -- Generate Excel and Markdown reports documenting every step.

## Data Splitting Strategy

The pipeline uses a three-way temporal split:

- **Training set:** All observations before `train_end_date`, minus the held-out
  test fraction. Used for model fitting.
- **Test set:** A stratified random sample (default 20%) drawn from the training
  period. Used for early stopping, feature selection scoring, and overfitting
  assessment.
- **Out-of-time (OOT) sets:** All observations after `train_end_date`, grouped
  by calendar quarter. Used for temporal stability assessment.

This design ensures that the test set shares the same distribution as training
data (for unbiased performance estimation), while OOT quarters simulate real
deployment conditions where the population may have shifted.

## Feature Elimination Steps

Features pass through six filters in sequence. Each filter removes features that
fail a specific criterion. The order is chosen so that cheap, broad filters run
first and expensive, narrow filters run last.

1. **Constant Elimination** -- Removes features with fewer than 2 unique non-null
   values. These carry zero information.
2. **Missing Rate Filter** -- Removes features where more than the configured
   threshold (default 70%) of training observations are missing.
3. **Information Value (IV)** -- Removes features with IV below the minimum
   threshold (weak predictors) or above the maximum threshold (suspiciously
   strong, potential leakage).
4. **Population Stability Index (PSI)** -- Removes features whose distribution
   shifts significantly over time within the training period.
5. **Temporal Performance Filter** (optional) -- Removes features whose
   individual predictive power degrades across OOT quarters.
6. **Correlation Elimination** -- From pairs of features with absolute
   correlation above the threshold, removes the one with lower IV.

See `docs/methodology/variable_elimination.md` for detailed formulas and
interpretation guides for each step.

## Feature Selection

After elimination, the surviving features (typically 30-100+) are passed to
a sequential feature selection algorithm:

- **Forward selection** (default): Starts with an empty set and greedily adds
  the feature that produces the largest AUC improvement on the test set.
- **Backward selection**: Starts with all candidates and greedily removes the
  feature whose removal causes the smallest AUC loss.

Selection uses k-fold cross-validation (default 5 folds) for stable AUC
estimates. Early stopping is controlled by two parameters:

- **Tolerance:** Minimum AUC improvement to continue adding features.
- **Patience:** Number of consecutive non-improving steps before stopping.

Features are evaluated in descending IV order to prioritize statistically
stronger predictors.

## Multicollinearity Check (VIF)

After selection, Variance Inflation Factor analysis detects multicollinearity
among the final feature set. VIF measures how much the variance of a
regression coefficient is inflated due to linear dependence on other features.

- VIF = 1: No collinearity with other features.
- VIF > 5 (default threshold): Concerning collinearity.
- VIF > 10: Severe collinearity.

When a feature exceeds the VIF threshold, the pipeline removes it
iteratively, preferring to drop the feature with lower IV (when `iv_aware`
mode is enabled).

## Hyperparameter Tuning

The pipeline uses Optuna for Bayesian hyperparameter optimization. The
objective function balances discrimination and stability:

    objective = mean_cv_auc - stability_weight * std_across_periods

Key tuned parameters include `max_depth`, `learning_rate`, `n_estimators`,
`subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`,
and `reg_lambda`.

## Model Evaluation

The final model is evaluated on every data split (Train, Test, each OOT
quarter). Computed metrics include:

- AUC, Gini coefficient, and KS statistic
- Decile-level lift tables and capture rate analysis
- Feature importance (gain-based by default)
- Bootstrap confidence intervals for AUC
- Score PSI between training and OOT periods
- Probability calibration (Platt scaling or isotonic regression)
- SHAP value analysis for interpretability
- Confusion matrix metrics at multiple thresholds

See `docs/methodology/model_evaluation.md` for detailed metric definitions.
