"""
Classic Model Adapter

Wraps a fitted sklearn LogisticRegression + StandardScaler to match the
``model.predict_proba(X)[:, 1]`` interface expected by ``evaluator.py``.

The existing ``LogisticRegressionModel.predict_proba()`` returns a 1-D array
(positive class only). The evaluator, however, indexes with ``[:, 1]``, so
this adapter ensures a standard 2-D sklearn-style output.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ClassicModelAdapter:
    """Adapts LogReg + Scaler to the XGBoost-like ``predict_proba`` interface.

    The evaluator calls ``model.predict_proba(X)[:, 1]`` and
    ``model.predict(X)``.  This adapter transparently selects the right
    feature columns, scales them, and delegates to the underlying sklearn
    ``LogisticRegression``.

    Parameters
    ----------
    model : LogisticRegression
        A fitted sklearn ``LogisticRegression`` instance.
    scaler : StandardScaler
        A fitted ``StandardScaler`` used during training.
    feature_names : list of str
        The WoE-encoded feature column names the model was trained on.
    """

    def __init__(
        self,
        model: LogisticRegression,
        scaler: StandardScaler,
        feature_names: List[str],
    ):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)

    # ------------------------------------------------------------------
    # Public API (mirrors XGBClassifier)
    # ------------------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        """Return 2-D array ``[p_neg, p_pos]`` like XGBoost / sklearn.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.  If a DataFrame is passed, only
            ``self.feature_names`` columns are selected.

        Returns
        -------
        np.ndarray of shape ``(n_samples, 2)``
        """
        X_scaled = self._prepare(X)
        return self.model.predict_proba(X_scaled)  # sklearn returns 2-D

    def predict(self, X) -> np.ndarray:
        """Return binary class predictions ``{0, 1}``."""
        X_scaled = self._prepare(X)
        return self.model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def classes_(self) -> np.ndarray:
        """Expose underlying model classes for sklearn compatibility."""
        return self.model.classes_

    @property
    def coef_(self) -> np.ndarray:
        """Expose raw coefficients."""
        return self.model.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Expose raw intercept."""
        return self.model.intercept_

    @property
    def feature_importances_(self) -> np.ndarray:
        """Coefficient-based feature importances (absolute, normalised).

        This property is accessed by ``evaluator._feature_importance()`` as a
        fallback when ``model.get_booster()`` is not available.
        """
        abs_coefs = np.abs(self.model.coef_[0])
        total = abs_coefs.sum()
        if total > 0:
            return abs_coefs / total
        return abs_coefs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare(self, X) -> np.ndarray:
        """Select features and scale."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        return self.scaler.transform(X)
