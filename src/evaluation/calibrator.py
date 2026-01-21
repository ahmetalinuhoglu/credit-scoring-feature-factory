"""
Model Calibrator

Calibrates model probabilities to match actual default rates.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from scipy import stats


@dataclass
class CalibrationResult:
    """Result of probability calibration."""
    method: str
    brier_score_before: float
    brier_score_after: float
    ece_before: float
    ece_after: float
    hosmer_lemeshow_chi2: float
    hosmer_lemeshow_pvalue: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'brier_score_before': self.brier_score_before,
            'brier_score_after': self.brier_score_after,
            'ece_before': self.ece_before,
            'ece_after': self.ece_after,
            'hosmer_lemeshow_chi2': self.hosmer_lemeshow_chi2,
            'hosmer_lemeshow_pvalue': self.hosmer_lemeshow_pvalue
        }


class ModelCalibrator:
    """
    Calibrate model probabilities to match actual default rates.
    
    Methods:
    - Platt Scaling: Sigmoid fit on probabilities
    - Isotonic Regression: Non-parametric monotonic fit
    - Temperature Scaling: Single parameter scaling
    
    Critical for:
    - Regulatory capital calculations
    - Pricing decisions
    - Provisioning
    """
    
    def __init__(self, method: str = 'platt'):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('platt', 'isotonic', 'temperature')
        """
        self.method = method.lower()
        self._calibrator = None
        self._is_fitted = False
        self._temperature = 1.0  # For temperature scaling
    
    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> 'ModelCalibrator':
        """
        Fit calibrator on validation data.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Self
        """
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have same length")
        
        if self.method == 'platt':
            self._fit_platt(y_true, y_prob)
        elif self.method == 'isotonic':
            self._fit_isotonic(y_true, y_prob)
        elif self.method == 'temperature':
            self._fit_temperature(y_true, y_prob)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._is_fitted = True
        return self
    
    def _fit_platt(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit Platt scaling (sigmoid calibration)."""
        # Use logistic regression on log-odds
        X = y_prob.reshape(-1, 1)
        self._calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._calibrator.fit(X, y_true)
    
    def _fit_isotonic(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit isotonic regression calibration."""
        self._calibrator = IsotonicRegression(
            y_min=0, 
            y_max=1, 
            out_of_bounds='clip'
        )
        self._calibrator.fit(y_prob, y_true)
    
    def _fit_temperature(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit temperature scaling."""
        from scipy.optimize import minimize_scalar
        
        def nll_loss(temperature):
            """Negative log likelihood with temperature scaling."""
            # Convert probability to logit, scale, convert back
            eps = 1e-7
            y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
            logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
            scaled_logits = logits / temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            
            # Calculate NLL
            nll = -np.mean(
                y_true * np.log(scaled_probs + eps) + 
                (1 - y_true) * np.log(1 - scaled_probs + eps)
            )
            return nll
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self._temperature = result.x
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.
        
        Args:
            y_prob: Raw predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        
        y_prob = np.asarray(y_prob).ravel()
        
        if self.method == 'platt':
            X = y_prob.reshape(-1, 1)
            return self._calibrator.predict_proba(X)[:, 1]
        
        elif self.method == 'isotonic':
            return self._calibrator.predict(y_prob)
        
        elif self.method == 'temperature':
            eps = 1e-7
            y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
            logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
            scaled_logits = logits / self._temperature
            return 1 / (1 + np.exp(-scaled_logits))
        
        return y_prob
    
    @staticmethod
    def calibration_curve_data(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibration curve data (predicted vs actual).
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Tuple of (mean predicted, fraction of positives)
        """
        fraction_positives, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        return mean_predicted, fraction_positives
    
    @staticmethod
    def hosmer_lemeshow_test(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, float]:
        """
        Perform Hosmer-Lemeshow goodness of fit test.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Tuple of (chi-squared statistic, p-value)
        """
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        
        # Create bins based on predicted probabilities
        try:
            bins = pd.qcut(y_prob, q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            # Fall back to fewer bins
            bins = pd.qcut(y_prob, q=5, labels=False, duplicates='drop')
        
        df = pd.DataFrame({
            'y_true': y_true,
            'y_prob': y_prob,
            'bin': bins
        })
        
        grouped = df.groupby('bin').agg({
            'y_true': 'sum',
            'y_prob': ['sum', 'count']
        })
        
        grouped.columns = ['observed', 'expected', 'n']
        
        # Calculate chi-squared statistic
        chi2 = 0
        for _, row in grouped.iterrows():
            observed_bad = row['observed']
            expected_bad = row['expected']
            n = row['n']
            observed_good = n - observed_bad
            expected_good = n - expected_bad
            
            if expected_bad > 0:
                chi2 += (observed_bad - expected_bad) ** 2 / expected_bad
            if expected_good > 0:
                chi2 += (observed_good - expected_good) ** 2 / expected_good
        
        # Degrees of freedom = n_bins - 2
        df_stat = max(len(grouped) - 2, 1)
        p_value = 1 - stats.chi2.cdf(chi2, df_stat)
        
        return chi2, p_value
    
    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate Brier score.
        
        Brier Score = mean((y_true - y_prob)^2)
        Perfect score = 0, worst = 1
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Brier score
        """
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        return np.mean((y_true - y_prob) ** 2)
    
    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE = Σ (|bin| / n) * |accuracy(bin) - confidence(bin)|
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            ECE value (lower is better)
        """
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        
        # Create uniform bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        n_samples = len(y_true)
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() == 0:
                continue
            
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_size = mask.sum()
            
            ece += (bin_size / n_samples) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def get_calibration_result(
        self,
        y_true: np.ndarray,
        y_prob_before: np.ndarray
    ) -> CalibrationResult:
        """
        Get comprehensive calibration result.
        
        Args:
            y_true: True binary labels
            y_prob_before: Original probabilities before calibration
            
        Returns:
            CalibrationResult with before/after metrics
        """
        y_prob_after = self.calibrate(y_prob_before)
        
        brier_before = self.brier_score(y_true, y_prob_before)
        brier_after = self.brier_score(y_true, y_prob_after)
        
        ece_before = self.expected_calibration_error(y_true, y_prob_before)
        ece_after = self.expected_calibration_error(y_true, y_prob_after)
        
        chi2, p_value = self.hosmer_lemeshow_test(y_true, y_prob_after)
        
        return CalibrationResult(
            method=self.method,
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            ece_before=ece_before,
            ece_after=ece_after,
            hosmer_lemeshow_chi2=chi2,
            hosmer_lemeshow_pvalue=p_value
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save calibrator to file."""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'calibrator': self._calibrator,
                'temperature': self._temperature,
                'is_fitted': self._is_fitted
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModelCalibrator':
        """Load calibrator from file."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls(method=data['method'])
        calibrator._calibrator = data['calibrator']
        calibrator._temperature = data['temperature']
        calibrator._is_fitted = data['is_fitted']
        
        return calibrator
    
    @property
    def is_fitted(self) -> bool:
        """Check if calibrator is fitted."""
        return self._is_fitted
