# Scorecard Methodology

## Overview

A credit risk scorecard converts a statistical model's output into an
integer-point scoring system that is interpretable, auditable, and easy to
implement in production systems. This document describes the classic scorecard
approach based on Weight of Evidence (WoE) transformation and logistic
regression.

## Weight of Evidence (WoE) Transformation

### Definition

WoE transforms each feature's bins into a scale that is linearly related to
the log-odds of the target:

    WoE_i = ln(Distribution_of_Goods_i / Distribution_of_Bads_i)

Where:
- `Distribution_of_Goods_i` = (number of goods in bin i) / (total goods)
- `Distribution_of_Bads_i` = (number of bads in bin i) / (total bads)

### Properties

- Positive WoE indicates the bin has a higher proportion of goods than the
  overall population (lower risk).
- Negative WoE indicates the bin has a higher proportion of bads (higher risk).
- WoE = 0 when the bin matches the overall good/bad distribution.

### Binning Requirements

- Bins should have at least `min_bin_size` (default: 5%) of observations to
  ensure statistical stability.
- Monotonic WoE is preferred for interpretability: the WoE should consistently
  increase or decrease across ordered bins.
- A separate bin is typically created for missing values.

## Logistic Regression Model

After WoE transformation, a logistic regression model is fitted:

    ln(odds) = beta_0 + beta_1 * WoE_1 + beta_2 * WoE_2 + ... + beta_k * WoE_k

Where `odds = P(Good) / P(Bad)`.

### Requirements

- All coefficients (`beta_i`) should be positive, confirming that higher WoE
  (more goods) corresponds to lower risk. Negative coefficients indicate a
  problematic feature and should be investigated.
- The model uses L2 regularization (Ridge) by default to prevent overfitting,
  with configurable regularization strength (`C` parameter).

## Scorecard Formula

### Point Calculation

The scorecard converts the logistic regression output into integer points:

    Score = Offset + Factor * (beta_0 + SUM(beta_i * WoE_i))

Where:
- **Factor** = PDO / ln(2)
- **Offset** = Target_Score - Factor * ln(Target_Odds)

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_score` | 600 | Score assigned at the target odds ratio |
| `target_odds` | 20:1 | Good-to-bad odds ratio at the target score |
| `pdo` | 50 | Points to Double the Odds |

### Points to Double the Odds (PDO)

PDO defines how many additional score points correspond to a doubling of the
odds of being good. For example, with PDO = 50:

- A score of 600 corresponds to 20:1 odds (good:bad).
- A score of 650 corresponds to 40:1 odds.
- A score of 550 corresponds to 10:1 odds.

### Per-Feature Points

Each feature contributes points based on the bin the applicant falls into:

    Points_i = -(beta_i * WoE_ij + beta_0 / k) * Factor

Where:
- `beta_i` is the logistic regression coefficient for feature `i`
- `WoE_ij` is the WoE value for the bin `j` that the applicant falls into
- `k` is the number of features
- `beta_0 / k` distributes the intercept evenly across features

Points are rounded to integers for the final scorecard. The total score is
the sum of the offset and all feature-level points.

## Scorecard Output

The final scorecard is a lookup table:

| Feature | Bin | WoE | Points |
|---------|-----|-----|--------|
| Age | < 25 | -0.45 | 12 |
| Age | 25-35 | 0.10 | 18 |
| Age | 35-50 | 0.35 | 21 |
| Age | > 50 | 0.55 | 24 |

This format is suitable for manual scoring, rule engine implementation, or
direct deployment in production systems without requiring the original model
object.

## Relationship to XGBoost Pipeline

The classic scorecard pipeline is an alternative to the XGBoost-based pipeline.
Key differences:

- **XGBoost:** Higher discrimination, non-linear feature interactions, requires
  model serialization for deployment.
- **Scorecard:** Lower discrimination ceiling, fully interpretable, deployable
  as a simple lookup table, preferred for regulatory environments.

Both pipelines share the same elimination steps (constant, missing, IV, PSI,
correlation). The scorecard pipeline replaces forward selection and XGBoost
with WoE transformation, logistic regression, and point-based scoring.
