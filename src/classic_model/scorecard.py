"""
Scorecard Generator

Converts Logistic Regression coefficients + WoE bins into a traditional
point-based credit scorecard.

Key formulas
------------
- ``Factor  = PDO / ln(2)``
- ``Offset  = target_score - Factor * ln(target_odds)``
- Per-bin points::

      Points_i = -(WoE_i * coef_j + intercept / n_features) * Factor
                 + Offset / n_features

The total score for an applicant is simply the sum of points across all
selected features (one bin per feature).
"""

from typing import Dict, List, Optional

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.woe_transformer import WoETransformer, WoEBin

logger = logging.getLogger(__name__)


class ScorecardGenerator:
    """Generate and apply a point-based scorecard.

    Parameters
    ----------
    target_score : int
        The score corresponding to ``target_odds``.  A common choice
        is 600.
    target_odds : float
        Odds of *good* at ``target_score`` (e.g. 20 means 20:1 good:bad).
    pdo : float
        Points to double the odds.  A common choice is 50.
    """

    def __init__(
        self,
        target_score: int = 600,
        target_odds: float = 20.0,
        pdo: float = 50.0,
    ):
        self.target_score = target_score
        self.target_odds = target_odds
        self.pdo = pdo

        # Derived constants
        self.factor = pdo / np.log(2)
        self.offset = target_score - self.factor * np.log(target_odds)

        # Populated after generate()
        self._scorecard_df: Optional[pd.DataFrame] = None
        self._scorecard_lookup: Dict[str, List[dict]] = {}  # feature -> list of bin dicts

        logger.info(
            "ScorecardGenerator: target_score=%d, target_odds=%.1f, pdo=%.1f, "
            "factor=%.4f, offset=%.4f",
            target_score, target_odds, pdo, self.factor, self.offset,
        )

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(
        self,
        model: LogisticRegression,
        scaler: StandardScaler,
        woe_transformer: WoETransformer,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Build the scorecard from a fitted LogReg, scaler, and WoE bins.

        The method accounts for the ``StandardScaler`` transformation that was
        applied during model training.  Specifically, if the model was trained
        on *scaled* WoE values, the raw coefficient for feature *j* in the
        *unscaled* WoE space is::

            effective_coef_j = coef_j / scale_j

        and the intercept absorbs the mean adjustments::

            effective_intercept = intercept - sum(coef_j * mean_j / scale_j)

        Parameters
        ----------
        model : LogisticRegression
            Fitted sklearn ``LogisticRegression``.
        scaler : StandardScaler
            Fitted ``StandardScaler`` used during training.
        woe_transformer : WoETransformer
            Fitted ``WoETransformer`` that produced the WoE encodings.
        feature_names : list of str
            The *original* feature names (before ``_woe`` suffix) that were
            selected by the pipeline.

        Returns
        -------
        pd.DataFrame
            Scorecard with columns: Feature, Bin, Lower, Upper, Count,
            Bad_Rate, WoE, IV_Contribution, Points.
        """
        coefs = model.coef_[0]  # shape (n_features,)
        intercept = float(model.intercept_[0])
        n_features = len(feature_names)
        scales = scaler.scale_  # std devs
        means = scaler.mean_  # means

        # Map WoE column names to indices in scaler / coef arrays
        woe_col_names = [f"{f}_woe" for f in feature_names]

        rows = []
        self._scorecard_lookup = {}

        for idx, (orig_feat, woe_feat) in enumerate(zip(feature_names, woe_col_names)):
            coef_j = coefs[idx]
            scale_j = scales[idx]
            mean_j = means[idx]

            # Effective coefficient in unscaled WoE space
            eff_coef = coef_j / scale_j
            # Contribution of this feature's mean to the intercept
            eff_intercept_contribution = intercept / n_features - coef_j * mean_j / scale_j

            bins = woe_transformer.get_woe_table(orig_feat)
            if bins is None:
                logger.warning("No WoE bins found for feature '%s', skipping", orig_feat)
                continue

            feature_lookup = []
            for bin_obj in bins:
                # Points formula (in the unscaled WoE space)
                raw_logit_contribution = eff_coef * bin_obj.woe + eff_intercept_contribution
                points = int(round(-raw_logit_contribution * self.factor + self.offset / n_features))

                bad_rate = (
                    bin_obj.bad_count / bin_obj.count
                    if bin_obj.count > 0
                    else 0.0
                )

                bin_label = self._bin_label(bin_obj)

                row = {
                    "Feature": orig_feat,
                    "Bin": bin_label,
                    "Lower": bin_obj.lower_bound,
                    "Upper": bin_obj.upper_bound,
                    "Count": bin_obj.count,
                    "Bad_Rate": round(bad_rate, 4),
                    "WoE": round(bin_obj.woe, 4),
                    "IV_Contribution": round(bin_obj.iv_contribution, 6),
                    "Points": points,
                }
                rows.append(row)
                feature_lookup.append(row)

            self._scorecard_lookup[orig_feat] = feature_lookup

        self._scorecard_df = pd.DataFrame(rows)

        logger.info(
            "Scorecard generated: %d features, %d rows, score range [%d, %d]",
            n_features,
            len(rows),
            self._scorecard_df["Points"].min() if len(rows) else 0,
            self._scorecard_df["Points"].sum() if len(rows) == 0
            else self._min_max_total_scores()[0],
        )

        return self._scorecard_df

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(
        self,
        df: pd.DataFrame,
        woe_transformer: WoETransformer,
        feature_names: List[str],
    ) -> pd.Series:
        """Apply the scorecard to new data.

        For each row, the total score is the sum of per-feature points
        determined by matching the raw feature value to the corresponding
        WoE bin.

        Parameters
        ----------
        df : pd.DataFrame
            Raw (un-encoded) data with the original feature columns.
        woe_transformer : WoETransformer
            The same fitted ``WoETransformer`` used during ``generate()``.
        feature_names : list of str
            Original feature names (before ``_woe``).

        Returns
        -------
        pd.Series
            Integer scores for each row.
        """
        if self._scorecard_df is None or self._scorecard_df.empty:
            raise RuntimeError("Scorecard not generated yet. Call generate() first.")

        total_scores = pd.Series(0, index=df.index, dtype=int)

        for feat in feature_names:
            if feat not in self._scorecard_lookup:
                logger.warning("Feature '%s' not in scorecard, contributing 0", feat)
                continue

            bins_info = self._scorecard_lookup[feat]
            feat_points = self._assign_points_for_feature(df[feat], bins_info)
            total_scores += feat_points

        return total_scores

    # ------------------------------------------------------------------
    # to_dataframe
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return the scorecard as a DataFrame.

        Raises
        ------
        RuntimeError
            If ``generate()`` has not been called yet.
        """
        if self._scorecard_df is None:
            raise RuntimeError("Scorecard not generated yet. Call generate() first.")
        return self._scorecard_df.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bin_label(bin_obj: WoEBin) -> str:
        """Human-readable label for a bin."""
        if bin_obj.bin_id < 0:
            return "Missing"
        low = bin_obj.lower_bound
        high = bin_obj.upper_bound
        if np.isnan(low) or np.isnan(high):
            return "Missing"
        return f"[{low:.4g}, {high:.4g}]"

    @staticmethod
    def _assign_points_for_feature(
        series: pd.Series,
        bins_info: List[dict],
    ) -> pd.Series:
        """Map raw feature values to scorecard points for a single feature.

        For each value, find the bin whose [Lower, Upper] range contains it.
        Missing / NaN values map to the "Missing" bin if one exists, otherwise
        they receive 0 points.
        """
        points = pd.Series(0, index=series.index, dtype=int)

        # Separate missing bin from regular bins
        regular_bins = [b for b in bins_info if b["Bin"] != "Missing"]
        missing_bin = next((b for b in bins_info if b["Bin"] == "Missing"), None)

        # Handle NaN
        if missing_bin is not None:
            points[series.isna()] = missing_bin["Points"]

        # Regular bins (sorted by Lower)
        regular_bins_sorted = sorted(regular_bins, key=lambda b: b["Lower"])
        for b in regular_bins_sorted:
            mask = (series >= b["Lower"]) & (series <= b["Upper"])
            points[mask] = b["Points"]

        return points

    def _min_max_total_scores(self):
        """Compute theoretical min and max total scores."""
        if not self._scorecard_lookup:
            return 0, 0

        min_total = 0
        max_total = 0
        for feat, bins_info in self._scorecard_lookup.items():
            pts = [b["Points"] for b in bins_info]
            if pts:
                min_total += min(pts)
                max_total += max(pts)
        return min_total, max_total
