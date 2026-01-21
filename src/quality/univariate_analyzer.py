"""
Univariate Analyzer

Single-variable analysis for feature screening in credit scoring.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path

from src.core.base import PandasComponent
from src.core.exceptions import DataQualityError


@dataclass
class UnivariateResult:
    """Quality analysis result for a single feature."""
    feature_name: str
    dtype: str
    missing_count: int
    missing_rate: float
    unique_count: int
    iv_score: Optional[float]
    iv_category: Optional[str]
    gini_univariate: Optional[float]
    ks_univariate: Optional[float]
    distribution_stats: Dict[str, float] = field(default_factory=dict)
    is_recommended: bool = True
    rejection_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'dtype': self.dtype,
            'missing_count': self.missing_count,
            'missing_rate': self.missing_rate,
            'unique_count': self.unique_count,
            'iv_score': self.iv_score,
            'iv_category': self.iv_category,
            'gini_univariate': self.gini_univariate,
            'ks_univariate': self.ks_univariate,
            'distribution_stats': self.distribution_stats,
            'is_recommended': self.is_recommended,
            'rejection_reasons': self.rejection_reasons
        }


class UnivariateAnalyzer(PandasComponent):
    """
    Single-variable analysis for feature screening.
    
    Checks:
    - Missing rate threshold
    - IV score threshold
    - Univariate Gini/KS
    - Distribution analysis
    - Zero/constant variance detection
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None
    ):
        """
        Initialize univariate analyzer.
        
        Args:
            config: Configuration dictionary
            name: Optional analyzer name
        """
        super().__init__(config, name or "UnivariateAnalyzer")
        
        # Load thresholds from config
        ua_config = self.get_config('quality.univariate_analysis', {})
        thresholds = ua_config.get('thresholds', {})
        
        self.missing_rate_max = thresholds.get('missing_rate_max', 0.70)
        self.iv_min = thresholds.get('iv_min', 0.02)
        self.iv_max = thresholds.get('iv_max', 0.50)
        self.variance_min = thresholds.get('variance_min', 0.001)
        self.gini_min = thresholds.get('gini_min', 0.02)
        
        # Results storage
        self._results: Dict[str, UnivariateResult] = {}
        self._is_analyzed = False
    
    def validate(self) -> bool:
        """Validate analyzer configuration."""
        if not 0 < self.missing_rate_max <= 1:
            self.logger.error("missing_rate_max must be between 0 and 1")
            return False
        return True
    
    def run(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_column: str = 'target'
    ) -> Dict[str, 'UnivariateResult']:
        """
        Run the univariate analysis (satisfies abstract method requirement).
        
        This is a wrapper around analyze() to comply with PipelineComponent interface.
        
        Args:
            df: DataFrame with features and target
            features: List of feature columns to analyze
            target_column: Name of binary target column
            
        Returns:
            Dictionary of feature name to UnivariateResult
        """
        return self.analyze(df, features, target_column)
    
    def analyze(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_column: str = 'target'
    ) -> Dict[str, UnivariateResult]:
        """
        Analyze all features.
        
        Args:
            df: DataFrame with features and target
            features: List of feature columns to analyze
            target_column: Name of binary target column
            
        Returns:
            Dictionary of feature name to UnivariateResult
        """
        self._start_execution()
        
        if target_column not in df.columns:
            raise DataQualityError(
                f"Target column '{target_column}' not found",
                check_name="univariate_analysis"
            )
        
        self.logger.info(f"Analyzing {len(features)} features")
        
        for feature in features:
            if feature not in df.columns:
                self.logger.warning(f"Feature '{feature}' not found, skipping")
                continue
            
            try:
                result = self._analyze_feature(df, feature, target_column)
                self._results[feature] = result
            except Exception as e:
                self.logger.warning(f"Could not analyze '{feature}': {e}")
        
        self._is_analyzed = True
        
        # Summary logging
        recommended = sum(1 for r in self._results.values() if r.is_recommended)
        rejected = len(self._results) - recommended
        self.logger.info(
            f"Analysis complete: {recommended} recommended, {rejected} rejected"
        )
        
        self._end_execution()
        return self._results
    
    def _analyze_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        target_column: str
    ) -> UnivariateResult:
        """Analyze a single feature."""
        series = df[feature]
        target = df[target_column]
        
        # Basic statistics
        dtype = str(series.dtype)
        n_total = len(series)
        missing_count = series.isna().sum()
        missing_rate = missing_count / n_total if n_total > 0 else 0
        unique_count = series.nunique()
        
        # Distribution statistics for numeric features
        distribution_stats = {}
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                distribution_stats = {
                    'mean': float(non_null.mean()),
                    'std': float(non_null.std()),
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'p25': float(non_null.quantile(0.25)),
                    'p50': float(non_null.quantile(0.50)),
                    'p75': float(non_null.quantile(0.75)),
                    'variance': float(non_null.var())
                }
        
        # Calculate IV
        iv_score = self._calculate_iv(series, target)
        iv_category = self._get_iv_category(iv_score) if iv_score else None
        
        # Calculate univariate Gini and KS
        gini_uni, ks_uni = self._calculate_gini_ks(series, target)
        
        # Determine rejection reasons
        rejection_reasons = []
        
        if missing_rate > self.missing_rate_max:
            rejection_reasons.append(
                f"Missing rate {missing_rate:.1%} > {self.missing_rate_max:.1%}"
            )
        
        if iv_score is not None:
            if iv_score < self.iv_min:
                rejection_reasons.append(
                    f"IV {iv_score:.4f} < {self.iv_min} (useless)"
                )
            elif iv_score > self.iv_max:
                rejection_reasons.append(
                    f"IV {iv_score:.4f} > {self.iv_max} (suspicious - possible leakage)"
                )
        
        variance = distribution_stats.get('variance', 0)
        if variance < self.variance_min:
            rejection_reasons.append(
                f"Variance {variance:.6f} < {self.variance_min} (near-constant)"
            )
        
        if unique_count <= 1:
            rejection_reasons.append("Single unique value (constant)")
        
        is_recommended = len(rejection_reasons) == 0
        
        return UnivariateResult(
            feature_name=feature,
            dtype=dtype,
            missing_count=int(missing_count),
            missing_rate=float(missing_rate),
            unique_count=int(unique_count),
            iv_score=iv_score,
            iv_category=iv_category,
            gini_univariate=gini_uni,
            ks_univariate=ks_uni,
            distribution_stats=distribution_stats,
            is_recommended=is_recommended,
            rejection_reasons=rejection_reasons
        )
    
    def _calculate_iv(
        self,
        series: pd.Series,
        target: pd.Series,
        n_bins: int = 10
    ) -> Optional[float]:
        """Calculate Information Value for a feature."""
        try:
            # Prepare data
            data = pd.DataFrame({'feature': series, 'target': target}).dropna()
            
            if len(data) < 50:
                return None
            
            total_goods = (data['target'] == 0).sum()
            total_bads = (data['target'] == 1).sum()
            
            if total_goods == 0 or total_bads == 0:
                return None
            
            # Bin the feature
            try:
                data['bin'] = pd.qcut(
                    data['feature'], 
                    q=n_bins, 
                    labels=False, 
                    duplicates='drop'
                )
            except (ValueError, TypeError):
                return None
            
            # Calculate IV
            iv = 0.0
            epsilon = 1e-6
            
            for _, group in data.groupby('bin'):
                goods = (group['target'] == 0).sum()
                bads = (group['target'] == 1).sum()
                
                pct_goods = max(goods / total_goods, epsilon)
                pct_bads = max(bads / total_bads, epsilon)
                
                woe = np.log(pct_goods / pct_bads)
                iv += (pct_goods - pct_bads) * woe
            
            return float(iv)
            
        except Exception:
            return None
    
    def _get_iv_category(self, iv: float) -> str:
        """Categorize IV score."""
        if iv < 0.02:
            return "useless"
        elif iv < 0.1:
            return "weak"
        elif iv < 0.3:
            return "medium"
        elif iv < 0.5:
            return "strong"
        else:
            return "suspicious"
    
    def _calculate_gini_ks(
        self,
        series: pd.Series,
        target: pd.Series
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate univariate Gini and KS."""
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            
            # Prepare data
            data = pd.DataFrame({'feature': series, 'target': target}).dropna()
            
            if len(data) < 50:
                return None, None
            
            if data['target'].nunique() < 2:
                return None, None
            
            # Calculate AUC -> Gini
            auc = roc_auc_score(data['target'], data['feature'])
            gini = 2 * auc - 1
            
            # Calculate KS
            fpr, tpr, _ = roc_curve(data['target'], data['feature'])
            ks = max(tpr - fpr)
            
            return float(gini), float(ks)
            
        except Exception:
            return None, None
    
    def get_recommended_features(self) -> List[str]:
        """Get list of recommended features."""
        return [
            name for name, result in self._results.items()
            if result.is_recommended
        ]
    
    def get_rejected_features(self) -> Dict[str, List[str]]:
        """Get rejected features with reasons."""
        return {
            name: result.rejection_reasons
            for name, result in self._results.items()
            if not result.is_recommended
        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table of all features."""
        rows = []
        for name, result in self._results.items():
            rows.append({
                'feature': name,
                'dtype': result.dtype,
                'missing_rate': result.missing_rate,
                'unique_count': result.unique_count,
                'iv_score': result.iv_score,
                'iv_category': result.iv_category,
                'gini': result.gini_univariate,
                'ks': result.ks_univariate,
                'recommended': result.is_recommended,
                'rejection_reasons': '; '.join(result.rejection_reasons)
            })
        
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('iv_score', ascending=False, na_position='last')
        
        return df
    
    def generate_report(self, output_path: str) -> None:
        """Generate HTML report."""
        summary = self.get_summary_table()
        recommended = self.get_recommended_features()
        rejected = self.get_rejected_features()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Univariate Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .recommended {{ background-color: #dff0d8; }}
        .rejected {{ background-color: #f2dede; }}
        .summary {{ margin: 20px 0; padding: 15px; background: #e7e7e7; }}
    </style>
</head>
<body>
    <h1>Univariate Analysis Report</h1>
    
    <div class="summary">
        <strong>Summary:</strong><br>
        Total Features: {len(self._results)}<br>
        Recommended: {len(recommended)}<br>
        Rejected: {len(rejected)}
    </div>
    
    <h2>Feature Summary</h2>
    {summary.to_html(index=False, classes='summary-table')}
    
    <h2>Recommended Features ({len(recommended)})</h2>
    <ul>
        {''.join(f'<li>{f}</li>' for f in recommended)}
    </ul>
    
    <h2>Rejected Features ({len(rejected)})</h2>
    <ul>
        {''.join(f'<li><strong>{f}</strong>: {", ".join(reasons)}</li>' for f, reasons in rejected.items())}
    </ul>
</body>
</html>
"""
        
        Path(output_path).write_text(html)
        self.logger.info(f"Report saved to {output_path}")
    
    @property
    def is_analyzed(self) -> bool:
        """Check if analysis has been performed."""
        return self._is_analyzed
