"""
Credit Scoring Metrics

Implements credit-specific evaluation metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, log_loss
)


class CreditScoringMetrics:
    """
    Credit scoring specific metrics.
    
    Includes:
    - Gini Coefficient
    - KS Statistic
    - Lift Analysis
    - PSI (Population Stability Index)
    - Standard classification metrics
    """
    
    @staticmethod
    def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Calculate Gini coefficient.
        
        Gini = 2 * AUC - 1
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            
        Returns:
            Gini coefficient (-1 to 1, higher is better)
        """
        auc = roc_auc_score(y_true, y_score)
        return 2 * auc - 1
    
    @staticmethod
    def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic.
        
        Measures maximum separation between cumulative distributions
        of goods and bads.
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            
        Returns:
            Tuple of (KS statistic, threshold at max separation)
        """
        # Sort by score descending
        df = pd.DataFrame({
            'score': y_score,
            'target': y_true
        }).sort_values('score', ascending=False)
        
        # Calculate cumulative distributions
        total_bads = df['target'].sum()
        total_goods = len(df) - total_bads
        
        df['cum_bads'] = df['target'].cumsum() / total_bads
        df['cum_goods'] = (1 - df['target']).cumsum() / total_goods
        
        # KS is max difference
        df['ks'] = abs(df['cum_bads'] - df['cum_goods'])
        
        max_idx = df['ks'].idxmax()
        ks_stat = df.loc[max_idx, 'ks']
        ks_threshold = df.loc[max_idx, 'score']
        
        return ks_stat, ks_threshold
    
    @staticmethod
    def lift_table(
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_deciles: int = 10
    ) -> pd.DataFrame:
        """
        Create decile lift table.
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            n_deciles: Number of deciles (default 10)
            
        Returns:
            DataFrame with lift analysis by decile
        """
        df = pd.DataFrame({
            'score': y_score,
            'target': y_true
        })
        
        # Assign deciles (10 = highest risk, 1 = lowest risk)
        df['decile'] = pd.qcut(
            df['score'],
            q=n_deciles,
            labels=list(range(n_deciles, 0, -1)),
            duplicates='drop'
        )
        
        # Calculate metrics per decile
        lift_table = df.groupby('decile').agg({
            'score': ['min', 'max', 'mean'],
            'target': ['count', 'sum', 'mean']
        }).round(4)
        
        lift_table.columns = [
            'score_min', 'score_max', 'score_mean',
            'count', 'bads', 'bad_rate'
        ]
        
        # Calculate lift
        overall_bad_rate = df['target'].mean()
        lift_table['lift'] = lift_table['bad_rate'] / overall_bad_rate
        
        # Calculate cumulative stats
        lift_table = lift_table.sort_index(ascending=False)
        lift_table['cum_count'] = lift_table['count'].cumsum()
        lift_table['cum_bads'] = lift_table['bads'].cumsum()
        lift_table['cum_bad_rate'] = lift_table['cum_bads'] / lift_table['cum_count']
        lift_table['cum_lift'] = lift_table['cum_bad_rate'] / overall_bad_rate
        
        # Capture rates
        total_bads = df['target'].sum()
        lift_table['capture_rate'] = lift_table['cum_bads'] / total_bads
        
        return lift_table.reset_index()
    
    @staticmethod
    def psi(
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate Population Stability Index.
        
        PSI measures distribution shift between populations.
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Some change
        PSI > 0.25: Significant change
        
        Args:
            expected: Expected distribution (e.g., development)
            actual: Actual distribution (e.g., validation)
            n_bins: Number of bins
            
        Returns:
            Tuple of (PSI value, breakdown DataFrame)
        """
        # Create bins based on expected distribution
        try:
            _, bins = pd.qcut(expected, q=n_bins, retbins=True, duplicates='drop')
        except ValueError:
            # Not enough unique values, use fewer bins
            _, bins = pd.qcut(expected, q=min(5, len(np.unique(expected))), retbins=True)
        
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # Bin both distributions
        expected_bins = pd.cut(expected, bins=bins)
        actual_bins = pd.cut(actual, bins=bins)
        
        # Calculate proportions (convert to Series to avoid Categorical.value_counts issues)
        expected_pct = pd.Series(expected_bins).value_counts(normalize=True).sort_index()
        actual_pct = pd.Series(actual_bins).value_counts(normalize=True).sort_index()
        
        # Align indices
        all_bins = expected_pct.index.union(actual_pct.index)
        expected_pct = expected_pct.reindex(all_bins, fill_value=0.0001)
        actual_pct = actual_pct.reindex(all_bins, fill_value=0.0001)
        
        # Calculate PSI
        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi_total = psi_values.sum()
        
        # Create breakdown
        breakdown = pd.DataFrame({
            'bin': [str(b) for b in all_bins],
            'expected_pct': expected_pct.values,
            'actual_pct': actual_pct.values,
            'psi': psi_values.values
        })
        
        return psi_total, breakdown
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate all credit scoring metrics.
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            y_pred: Predicted classes (optional, will be computed if not provided)
            threshold: Classification threshold
            
        Returns:
            Dictionary of all metrics
        """
        if y_pred is None:
            y_pred = (y_score >= threshold).astype(int)
        
        # Basic metrics
        auc = roc_auc_score(y_true, y_score)
        gini = 2 * auc - 1
        ks_stat, ks_threshold = CreditScoringMetrics.ks_statistic(y_true, y_score)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Derived metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        try:
            logloss = log_loss(y_true, y_score)
        except:
            logloss = None
        
        return {
            'auc': round(auc, 4),
            'gini': round(gini, 4),
            'ks_statistic': round(ks_stat, 4),
            'ks_threshold': round(ks_threshold, 4),
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'log_loss': round(logloss, 4) if logloss else None,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'classification_report': report
        }
