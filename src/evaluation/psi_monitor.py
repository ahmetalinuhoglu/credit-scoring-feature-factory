"""
PSI Monitor

Population Stability Index monitoring for production drift detection.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


@dataclass
class PSIResult:
    """PSI calculation result for a single feature."""
    feature_name: str
    psi_value: float
    status: str  # 'stable', 'warning', 'critical'
    bin_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'psi_value': self.psi_value,
            'status': self.status,
            'bin_details': self.bin_details
        }


class PSIMonitor:
    """
    Population Stability Index monitoring.
    
    Features:
    - Feature-level PSI tracking
    - Score distribution PSI
    - Threshold-based alerting
    - Historical tracking
    
    PSI Thresholds:
    - PSI < 0.10: Stable (no action needed)
    - 0.10 ≤ PSI < 0.25: Warning (investigate)
    - PSI ≥ 0.25: Critical (model retraining needed)
    
    PSI Formula:
        PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.10,
        critical_threshold: float = 0.25,
        n_bins: int = 10
    ):
        """
        Initialize PSI monitor.
        
        Args:
            warning_threshold: PSI threshold for warning status
            critical_threshold: PSI threshold for critical status
            n_bins: Number of bins for distribution comparison
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.n_bins = n_bins
        
        # Baseline storage
        self._baseline_distributions: Dict[str, Dict[str, Any]] = {}
        self._has_baseline = False
        
        # Tracking history
        self._history: List[Dict[str, Any]] = []
    
    def set_baseline(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> None:
        """
        Set baseline distributions from training/reference data.
        
        Args:
            df: Baseline DataFrame
            features: Features to track (None = all numeric)
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            series = df[feature].dropna()
            if len(series) < 50:
                continue
            
            # Calculate bin edges from baseline
            bin_edges = self._calculate_bin_edges(series)
            
            # Calculate baseline distribution
            baseline_dist = self._calculate_distribution(series, bin_edges)
            
            self._baseline_distributions[feature] = {
                'bin_edges': bin_edges,
                'distribution': baseline_dist,
                'sample_size': len(series),
                'timestamp': datetime.now().isoformat()
            }
        
        self._has_baseline = True
    
    def _calculate_bin_edges(self, series: pd.Series) -> np.ndarray:
        """Calculate bin edges using quantiles."""
        # Use quantile-based binning
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(series, percentiles)
        
        # Ensure unique edges
        edges = np.unique(edges)
        
        # Add -inf and inf for edge cases
        edges[0] = -np.inf
        edges[-1] = np.inf
        
        return edges
    
    def _calculate_distribution(
        self,
        series: pd.Series,
        bin_edges: np.ndarray
    ) -> np.ndarray:
        """Calculate distribution proportions."""
        # Bin the series
        binned = np.digitize(series, bin_edges[1:-1])
        
        # Count per bin
        n_bins = len(bin_edges) - 1
        counts = np.bincount(binned, minlength=n_bins)[:n_bins]
        
        # Convert to proportions
        total = len(series)
        proportions = counts / total if total > 0 else counts
        
        return proportions
    
    def calculate_feature_psi(
        self,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, PSIResult]:
        """
        Calculate PSI for features against baseline.
        
        Args:
            current_df: Current/production DataFrame
            features: Features to check (None = all with baseline)
            
        Returns:
            Dictionary of feature name to PSIResult
        """
        if not self._has_baseline:
            raise RuntimeError("No baseline set. Call set_baseline() first.")
        
        if features is None:
            features = list(self._baseline_distributions.keys())
        
        results = {}
        
        for feature in features:
            if feature not in self._baseline_distributions:
                continue
            
            if feature not in current_df.columns:
                continue
            
            baseline_info = self._baseline_distributions[feature]
            bin_edges = baseline_info['bin_edges']
            expected_dist = baseline_info['distribution']
            
            # Calculate current distribution
            current_series = current_df[feature].dropna()
            actual_dist = self._calculate_distribution(current_series, bin_edges)
            
            # Calculate PSI
            psi_value, bin_details = self._calculate_psi(expected_dist, actual_dist)
            
            # Determine status
            if psi_value >= self.critical_threshold:
                status = 'critical'
            elif psi_value >= self.warning_threshold:
                status = 'warning'
            else:
                status = 'stable'
            
            results[feature] = PSIResult(
                feature_name=feature,
                psi_value=psi_value,
                status=status,
                bin_details=bin_details
            )
        
        return results
    
    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray
    ) -> tuple:
        """Calculate PSI value and bin details."""
        epsilon = 1e-6
        
        # Ensure non-zero proportions
        expected = np.maximum(expected, epsilon)
        actual = np.maximum(actual, epsilon)
        
        # Normalize
        expected = expected / expected.sum()
        actual = actual / actual.sum()
        
        # Calculate PSI per bin
        psi_per_bin = (actual - expected) * np.log(actual / expected)
        total_psi = float(np.sum(psi_per_bin))
        
        # Create bin details
        bin_details = []
        for i, (exp, act, psi_contrib) in enumerate(zip(expected, actual, psi_per_bin)):
            bin_details.append({
                'bin': i,
                'expected_pct': float(exp),
                'actual_pct': float(act),
                'psi_contribution': float(psi_contrib)
            })
        
        return total_psi, bin_details
    
    def calculate_score_psi(
        self,
        baseline_scores: np.ndarray,
        current_scores: np.ndarray
    ) -> PSIResult:
        """
        Calculate PSI for score distributions.
        
        Args:
            baseline_scores: Reference/training scores
            current_scores: Current/production scores
            
        Returns:
            PSIResult for scores
        """
        baseline_scores = np.asarray(baseline_scores).ravel()
        current_scores = np.asarray(current_scores).ravel()
        
        # Calculate bin edges from baseline
        bin_edges = self._calculate_bin_edges(pd.Series(baseline_scores))
        
        # Calculate distributions
        expected_dist = self._calculate_distribution(
            pd.Series(baseline_scores), bin_edges
        )
        actual_dist = self._calculate_distribution(
            pd.Series(current_scores), bin_edges
        )
        
        # Calculate PSI
        psi_value, bin_details = self._calculate_psi(expected_dist, actual_dist)
        
        # Determine status
        if psi_value >= self.critical_threshold:
            status = 'critical'
        elif psi_value >= self.warning_threshold:
            status = 'warning'
        else:
            status = 'stable'
        
        return PSIResult(
            feature_name='model_score',
            psi_value=psi_value,
            status=status,
            bin_details=bin_details
        )
    
    def get_alerts(
        self,
        results: Dict[str, PSIResult]
    ) -> List[str]:
        """
        Get alert messages for features with issues.
        
        Args:
            results: PSI results from calculate_feature_psi
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        for feature, result in results.items():
            if result.status == 'critical':
                alerts.append(
                    f"CRITICAL: {feature} PSI = {result.psi_value:.4f} "
                    f"(>= {self.critical_threshold}). Model retraining recommended."
                )
            elif result.status == 'warning':
                alerts.append(
                    f"WARNING: {feature} PSI = {result.psi_value:.4f} "
                    f"(>= {self.warning_threshold}). Investigation recommended."
                )
        
        return alerts
    
    def record_monitoring(
        self,
        results: Dict[str, PSIResult],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record monitoring results for historical tracking.
        
        Args:
            results: PSI results
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            'timestamp': timestamp.isoformat(),
            'results': {
                name: result.to_dict()
                for name, result in results.items()
            },
            'summary': {
                'total_features': len(results),
                'stable': sum(1 for r in results.values() if r.status == 'stable'),
                'warning': sum(1 for r in results.values() if r.status == 'warning'),
                'critical': sum(1 for r in results.values() if r.status == 'critical')
            }
        }
        
        self._history.append(record)
    
    def get_summary_table(
        self,
        results: Dict[str, PSIResult]
    ) -> pd.DataFrame:
        """
        Get summary table of PSI results.
        
        Args:
            results: PSI results
            
        Returns:
            DataFrame with PSI summary
        """
        rows = []
        for feature, result in results.items():
            rows.append({
                'feature': feature,
                'psi': result.psi_value,
                'status': result.status
            })
        
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('psi', ascending=False)
        
        return df
    
    def save_baseline(self, path: Union[str, Path]) -> None:
        """Save baseline to JSON file."""
        path = Path(path)
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for feature, baseline in self._baseline_distributions.items():
            export_data[feature] = {
                'bin_edges': baseline['bin_edges'].tolist(),
                'distribution': baseline['distribution'].tolist(),
                'sample_size': baseline['sample_size'],
                'timestamp': baseline['timestamp']
            }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def load_baseline(self, path: Union[str, Path]) -> None:
        """Load baseline from JSON file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self._baseline_distributions = {}
        for feature, baseline in data.items():
            self._baseline_distributions[feature] = {
                'bin_edges': np.array(baseline['bin_edges']),
                'distribution': np.array(baseline['distribution']),
                'sample_size': baseline['sample_size'],
                'timestamp': baseline['timestamp']
            }
        
        self._has_baseline = True
    
    def generate_report(
        self,
        results: Dict[str, PSIResult],
        output_path: str
    ) -> None:
        """Generate HTML monitoring report."""
        summary = self.get_summary_table(results)
        alerts = self.get_alerts(results)
        
        # Count by status
        n_stable = sum(1 for r in results.values() if r.status == 'stable')
        n_warning = sum(1 for r in results.values() if r.status == 'warning')
        n_critical = sum(1 for r in results.values() if r.status == 'critical')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PSI Monitoring Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .stable {{ background-color: #dff0d8; }}
        .warning {{ background-color: #fcf8e3; }}
        .critical {{ background-color: #f2dede; }}
        .summary {{ margin: 20px 0; padding: 15px; background: #e7e7e7; }}
        .alert {{ padding: 10px; margin: 5px 0; border-left: 4px solid; }}
        .alert-warning {{ border-color: #f0ad4e; background: #fcf8e3; }}
        .alert-critical {{ border-color: #d9534f; background: #f2dede; }}
    </style>
</head>
<body>
    <h1>PSI Monitoring Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <strong>Summary:</strong><br>
        Total Features: {len(results)}<br>
        <span class="stable">Stable: {n_stable}</span> |
        <span class="warning">Warning: {n_warning}</span> |
        <span class="critical">Critical: {n_critical}</span>
    </div>
    
    {'<h2>Alerts</h2>' if alerts else ''}
    {''.join(f'<div class="alert alert-{"critical" if "CRITICAL" in a else "warning"}">{a}</div>' for a in alerts)}
    
    <h2>Feature PSI Summary</h2>
    {summary.to_html(index=False)}
</body>
</html>
"""
        
        Path(output_path).write_text(html)
    
    @property
    def has_baseline(self) -> bool:
        """Check if baseline is set."""
        return self._has_baseline
    
    @property
    def tracked_features(self) -> List[str]:
        """Get list of tracked features."""
        return list(self._baseline_distributions.keys())
