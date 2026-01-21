"""
Cutoff Optimizer

Determines optimal score thresholds for approval decisions.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)


@dataclass
class CutoffAnalysis:
    """Analysis result for a specific cutoff."""
    cutoff: float
    approval_rate: float
    rejection_rate: float
    bad_rate_approved: float
    bad_rate_rejected: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cutoff': self.cutoff,
            'approval_rate': self.approval_rate,
            'rejection_rate': self.rejection_rate,
            'bad_rate_approved': self.bad_rate_approved,
            'bad_rate_rejected': self.bad_rate_rejected,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives
        }


class CutoffOptimizer:
    """
    Find optimal cutoff/threshold based on different strategies.
    
    Strategies:
    - KS Optimal: Maximum separation between good/bad distributions
    - Youden's J: Balanced sensitivity/specificity
    - Cost-Based: Minimize asymmetric costs
    - Target Approval Rate: Volume control
    - Target Bad Rate: Risk appetite control
    
    Note: Higher score = lower risk (good)
          Lower score = higher risk (bad)
    """
    
    def __init__(self):
        """Initialize cutoff optimizer."""
        pass
    
    @staticmethod
    def find_ks_optimal(
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find cutoff that maximizes KS statistic.
        
        Args:
            y_true: True binary labels (1 = bad, 0 = good)
            y_score: Predicted scores (higher = lower risk)
            
        Returns:
            Tuple of (optimal_cutoff, ks_statistic)
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        
        # Get ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # KS = max(TPR - FPR)
        ks_values = tpr - fpr
        max_idx = np.argmax(ks_values)
        
        return thresholds[max_idx], ks_values[max_idx]
    
    @staticmethod
    def find_youden_optimal(
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find cutoff that maximizes Youden's J statistic.
        
        Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
        
        Args:
            y_true: True binary labels
            y_score: Predicted scores
            
        Returns:
            Tuple of (optimal_cutoff, youden_j)
        """
        # Youden's J is equivalent to KS
        return CutoffOptimizer.find_ks_optimal(y_true, y_score)
    
    @staticmethod
    def find_cost_optimal(
        y_true: np.ndarray,
        y_score: np.ndarray,
        cost_fn: float,
        cost_fp: float
    ) -> Tuple[float, float]:
        """
        Find cutoff that minimizes total cost.
        
        Total Cost = cost_fn * FN + cost_fp * FP
        
        Args:
            y_true: True binary labels
            y_score: Predicted scores
            cost_fn: Cost of false negative (missing a bad)
            cost_fp: Cost of false positive (rejecting a good)
            
        Returns:
            Tuple of (optimal_cutoff, minimum_cost)
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        
        # Try different thresholds
        thresholds = np.percentile(y_score, np.linspace(1, 99, 99))
        
        min_cost = float('inf')
        optimal_cutoff = thresholds[len(thresholds) // 2]
        
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            y_pred = 1 - y_pred  # Invert: below threshold = predict bad (1)
            
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                cost = cost_fn * fn + cost_fp * fp
                
                if cost < min_cost:
                    min_cost = cost
                    optimal_cutoff = threshold
        
        return optimal_cutoff, min_cost
    
    @staticmethod
    def find_by_approval_rate(
        y_score: np.ndarray,
        target_rate: float
    ) -> float:
        """
        Find cutoff to achieve target approval rate.
        
        Args:
            y_score: Predicted scores (higher = lower risk)
            target_rate: Target approval rate (0-1)
            
        Returns:
            Cutoff score
        """
        y_score = np.asarray(y_score).ravel()
        
        if not 0 < target_rate < 1:
            raise ValueError("target_rate must be between 0 and 1")
        
        # Cutoff is the (1-target_rate) percentile
        # E.g., 60% approval = cutoff at 40th percentile
        cutoff = np.percentile(y_score, (1 - target_rate) * 100)
        
        return cutoff
    
    @staticmethod
    def find_by_bad_rate(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_bad_rate: float
    ) -> float:
        """
        Find cutoff to achieve target bad rate among approved.
        
        Args:
            y_true: True binary labels (1 = bad)
            y_score: Predicted scores (higher = lower risk)
            target_bad_rate: Target bad rate among approved
            
        Returns:
            Cutoff score
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        
        # Sort by score descending (best first)
        sorted_indices = np.argsort(-y_score)
        sorted_true = y_true[sorted_indices]
        sorted_score = y_score[sorted_indices]
        
        # Find cutoff where cumulative bad rate equals target
        cumsum_bads = np.cumsum(sorted_true)
        cumsum_total = np.arange(1, len(sorted_true) + 1)
        cumulative_bad_rates = cumsum_bads / cumsum_total
        
        # Find first point where bad rate exceeds target
        exceeds_target = cumulative_bad_rates > target_bad_rate
        
        if not exceeds_target.any():
            # Target never exceeded, return minimum score
            return sorted_score[-1]
        
        cutoff_idx = np.argmax(exceeds_target)
        if cutoff_idx > 0:
            cutoff_idx -= 1
        
        return sorted_score[cutoff_idx]
    
    @staticmethod
    def analyze_cutoff(
        y_true: np.ndarray,
        y_score: np.ndarray,
        cutoff: float
    ) -> CutoffAnalysis:
        """
        Analyze metrics at a specific cutoff.
        
        Args:
            y_true: True binary labels (1 = bad)
            y_score: Predicted scores (higher = lower risk)
            cutoff: Score cutoff for approval
            
        Returns:
            CutoffAnalysis with all metrics
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        
        # Approved = score >= cutoff, Rejected = score < cutoff
        approved_mask = y_score >= cutoff
        rejected_mask = ~approved_mask
        
        n_total = len(y_true)
        n_approved = approved_mask.sum()
        n_rejected = rejected_mask.sum()
        
        approval_rate = n_approved / n_total if n_total > 0 else 0
        rejection_rate = n_rejected / n_total if n_total > 0 else 0
        
        # Bad rates
        bad_rate_approved = (
            y_true[approved_mask].mean() 
            if n_approved > 0 else 0
        )
        bad_rate_rejected = (
            y_true[rejected_mask].mean() 
            if n_rejected > 0 else 0
        )
        
        # Classification metrics
        # Predict: approved (high score) = 0 (good), rejected (low score) = 1 (bad)
        y_pred = (~approved_mask).astype(int)
        
        # Handle edge cases
        if len(np.unique(y_pred)) == 1:
            prec = 0.0
            rec = 0.0
            f1 = 0.0
        else:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        return CutoffAnalysis(
            cutoff=cutoff,
            approval_rate=approval_rate,
            rejection_rate=rejection_rate,
            bad_rate_approved=bad_rate_approved,
            bad_rate_rejected=bad_rate_rejected,
            precision=prec,
            recall=rec,
            f1_score=f1,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn)
        )
    
    @staticmethod
    def generate_cutoff_table(
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_points: int = 20
    ) -> pd.DataFrame:
        """
        Generate cutoff analysis table across multiple thresholds.
        
        Args:
            y_true: True binary labels
            y_score: Predicted scores
            n_points: Number of cutoff points to analyze
            
        Returns:
            DataFrame with cutoff analysis at each point
        """
        y_score = np.asarray(y_score).ravel()
        
        # Generate cutoffs at percentiles
        percentiles = np.linspace(5, 95, n_points)
        cutoffs = np.percentile(y_score, percentiles)
        
        results = []
        for cutoff in cutoffs:
            analysis = CutoffOptimizer.analyze_cutoff(y_true, y_score, cutoff)
            results.append(analysis.to_dict())
        
        df = pd.DataFrame(results)
        
        # Add formatted columns
        df['approval_pct'] = (df['approval_rate'] * 100).round(1).astype(str) + '%'
        df['bad_rate_pct'] = (df['bad_rate_approved'] * 100).round(2).astype(str) + '%'
        
        return df
    
    @staticmethod
    def generate_score_distribution_table(
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_bands: int = 10
    ) -> pd.DataFrame:
        """
        Generate score distribution table by score bands.
        
        Args:
            y_true: True binary labels
            y_score: Predicted scores
            n_bands: Number of score bands
            
        Returns:
            DataFrame with distribution by score band
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        
        # Create score bands
        try:
            bands = pd.qcut(y_score, q=n_bands, labels=False, duplicates='drop')
        except ValueError:
            bands = pd.cut(y_score, bins=n_bands, labels=False)
        
        df = pd.DataFrame({
            'score': y_score,
            'target': y_true,
            'band': bands
        })
        
        # Aggregate by band
        result = df.groupby('band').agg({
            'score': ['min', 'max', 'count'],
            'target': ['sum', 'mean']
        })
        
        result.columns = ['score_min', 'score_max', 'count', 'bad_count', 'bad_rate']
        result = result.reset_index()
        
        # Calculate cumulative metrics
        result['cumul_count'] = result['count'].cumsum()
        result['cumul_bad'] = result['bad_count'].cumsum()
        result['cumul_bad_rate'] = result['cumul_bad'] / result['cumul_count']
        
        # Calculate capture rate (cumulative % of bads)
        total_bads = result['bad_count'].sum()
        result['capture_rate'] = result['cumul_bad'] / total_bads if total_bads > 0 else 0
        
        return result
    
    @staticmethod
    def find_multiple_cutoffs(
        y_true: np.ndarray,
        y_score: np.ndarray,
        methods: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Find optimal cutoffs using multiple methods.
        
        Args:
            y_true: True binary labels
            y_score: Predicted scores
            methods: List of methods to use
            
        Returns:
            Dictionary of method -> (cutoff, metric)
        """
        if methods is None:
            methods = ['ks', 'youden', 'approval_50', 'approval_70', 'bad_rate_5']
        
        results = {}
        
        for method in methods:
            if method == 'ks':
                cutoff, metric = CutoffOptimizer.find_ks_optimal(y_true, y_score)
                results['ks_optimal'] = {'cutoff': cutoff, 'ks_statistic': metric}
            
            elif method == 'youden':
                cutoff, metric = CutoffOptimizer.find_youden_optimal(y_true, y_score)
                results['youden_optimal'] = {'cutoff': cutoff, 'youden_j': metric}
            
            elif method.startswith('approval_'):
                rate = float(method.split('_')[1]) / 100
                cutoff = CutoffOptimizer.find_by_approval_rate(y_score, rate)
                analysis = CutoffOptimizer.analyze_cutoff(y_true, y_score, cutoff)
                results[method] = {
                    'cutoff': cutoff, 
                    'approval_rate': analysis.approval_rate,
                    'bad_rate': analysis.bad_rate_approved
                }
            
            elif method.startswith('bad_rate_'):
                target = float(method.split('_')[2]) / 100
                cutoff = CutoffOptimizer.find_by_bad_rate(y_true, y_score, target)
                analysis = CutoffOptimizer.analyze_cutoff(y_true, y_score, cutoff)
                results[method] = {
                    'cutoff': cutoff,
                    'approval_rate': analysis.approval_rate,
                    'bad_rate': analysis.bad_rate_approved
                }
        
        return results
