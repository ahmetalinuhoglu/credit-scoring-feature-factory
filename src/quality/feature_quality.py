"""
Feature Quality Checker

Analyzes feature quality including IV, correlation, and distributions.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.core.base import SparkComponent


@dataclass
class FeatureQualityResult:
    """Quality analysis result for a single feature."""
    feature_name: str
    null_ratio: float
    unique_count: int
    iv_score: Optional[float] = None
    iv_category: Optional[str] = None  # useless, weak, medium, strong, suspicious
    variance: Optional[float] = None
    correlations: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'null_ratio': self.null_ratio,
            'unique_count': self.unique_count,
            'iv_score': self.iv_score,
            'iv_category': self.iv_category,
            'variance': self.variance,
            'correlations': self.correlations
        }


class FeatureQualityChecker(SparkComponent):
    """
    Analyzes feature quality for credit scoring.
    
    Calculates:
    - Information Value (IV) for predictive power
    - Correlation analysis
    - Variance analysis
    - Distribution statistics
    """
    
    # IV thresholds for credit scoring
    IV_THRESHOLDS = {
        'useless': 0.02,
        'weak': 0.10,
        'medium': 0.30,
        'strong': 0.50,
        'suspicious': float('inf')  # IV > 0.5 might indicate data leakage
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        name: Optional[str] = None
    ):
        """
        Initialize feature quality checker.
        
        Args:
            config: Configuration dictionary
            spark_session: Active SparkSession
            name: Optional checker name
        """
        super().__init__(config, spark_session, name or "FeatureQualityChecker")
        
        self.quality_config = self.get_config('quality.feature_quality', {})
        
    def validate(self) -> bool:
        return super().validate()
    
    def run(
        self,
        df: Any,
        features: List[str],
        target_column: str
    ) -> Dict[str, FeatureQualityResult]:
        """Run feature quality analysis."""
        return self.analyze_features(df, features, target_column)
    
    def analyze_features(
        self,
        df: Any,
        features: List[str],
        target_column: str = 'target'
    ) -> Dict[str, FeatureQualityResult]:
        """
        Analyze quality of all features.
        
        Args:
            df: Spark DataFrame with features
            features: List of feature column names
            target_column: Name of target column
            
        Returns:
            Dictionary of feature name to FeatureQualityResult
        """
        self._start_execution()
        self.logger.info(f"Analyzing {len(features)} features")
        
        results = {}
        total_rows = df.count()
        
        for feature in features:
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in DataFrame")
                continue
            
            # Basic stats
            null_count = df.filter(F.col(feature).isNull()).count()
            null_ratio = null_count / total_rows if total_rows > 0 else 0
            
            unique_count = df.select(feature).distinct().count()
            
            # Calculate IV
            iv_score = self._calculate_iv(df, feature, target_column)
            iv_category = self._get_iv_category(iv_score) if iv_score else None
            
            # Calculate variance for numeric features
            variance = self._calculate_variance(df, feature)
            
            results[feature] = FeatureQualityResult(
                feature_name=feature,
                null_ratio=null_ratio,
                unique_count=unique_count,
                iv_score=iv_score,
                iv_category=iv_category,
                variance=variance
            )
        
        self._end_execution()
        self.logger.info(f"Feature analysis complete")
        
        return results
    
    def _calculate_iv(
        self,
        df: Any,
        feature: str,
        target_column: str,
        n_bins: int = 10
    ) -> Optional[float]:
        """
        Calculate Information Value for a feature.
        
        IV = Σ (% of Goods - % of Bads) * ln(% of Goods / % of Bads)
        
        Args:
            df: DataFrame
            feature: Feature column name
            target_column: Target column name
            n_bins: Number of bins for numeric features
            
        Returns:
            IV score or None if calculation fails
        """
        try:
            # Filter out nulls
            df_clean = df.filter(F.col(feature).isNotNull())
            
            if df_clean.count() == 0:
                return None
            
            # Get total goods and bads
            totals = df_clean.groupBy(target_column).count().collect()
            total_goods = sum(r['count'] for r in totals if r[target_column] == 0)
            total_bads = sum(r['count'] for r in totals if r[target_column] == 1)
            
            if total_goods == 0 or total_bads == 0:
                return None
            
            # Check if numeric or categorical
            dtype = str(df_clean.schema[feature].dataType)
            is_numeric = any(t in dtype for t in ['Int', 'Long', 'Double', 'Float'])
            
            if is_numeric and df_clean.select(feature).distinct().count() > n_bins:
                # Bin numeric feature
                df_binned = self._bin_numeric_feature(df_clean, feature, n_bins)
                group_col = f"{feature}_bin"
            else:
                df_binned = df_clean
                group_col = feature
            
            # Calculate WoE and IV for each bin
            bin_stats = (
                df_binned
                .groupBy(group_col)
                .agg(
                    F.sum(F.when(F.col(target_column) == 0, 1).otherwise(0)).alias('goods'),
                    F.sum(F.when(F.col(target_column) == 1, 1).otherwise(0)).alias('bads')
                )
                .collect()
            )
            
            iv = 0.0
            for row in bin_stats:
                goods = row['goods']
                bads = row['bads']
                
                # Avoid division by zero
                goods_pct = (goods + 0.5) / (total_goods + 0.5)
                bads_pct = (bads + 0.5) / (total_bads + 0.5)
                
                woe = math.log(goods_pct / bads_pct)
                iv += (goods_pct - bads_pct) * woe
            
            return round(iv, 4)
            
        except Exception as e:
            self.logger.debug(f"IV calculation failed for {feature}: {e}")
            return None
    
    def _bin_numeric_feature(
        self,
        df: Any,
        feature: str,
        n_bins: int
    ) -> Any:
        """Bin a numeric feature into quantiles."""
        from pyspark.ml.feature import QuantileDiscretizer
        
        discretizer = QuantileDiscretizer(
            numBuckets=n_bins,
            inputCol=feature,
            outputCol=f"{feature}_bin",
            handleInvalid="keep"
        )
        
        return discretizer.fit(df).transform(df)
    
    def _get_iv_category(self, iv: float) -> str:
        """Categorize IV score."""
        if iv < self.IV_THRESHOLDS['useless']:
            return 'useless'
        elif iv < self.IV_THRESHOLDS['weak']:
            return 'weak'
        elif iv < self.IV_THRESHOLDS['medium']:
            return 'medium'
        elif iv < self.IV_THRESHOLDS['strong']:
            return 'strong'
        else:
            return 'suspicious'
    
    def _calculate_variance(
        self,
        df: Any,
        feature: str
    ) -> Optional[float]:
        """Calculate variance for numeric features."""
        dtype = str(df.schema[feature].dataType)
        is_numeric = any(t in dtype for t in ['Int', 'Long', 'Double', 'Float'])
        
        if not is_numeric:
            return None
        
        try:
            result = df.select(F.variance(feature)).collect()[0][0]
            return float(result) if result else None
        except:
            return None
    
    def calculate_correlation_matrix(
        self,
        df: Any,
        features: List[str],
        method: str = 'pearson'
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix for numeric features.
        
        Args:
            df: DataFrame
            features: List of feature names
            method: Correlation method (pearson or spearman)
            
        Returns:
            Dictionary of feature pairs to correlation values
        """
        from pyspark.ml.stat import Correlation
        from pyspark.ml.feature import VectorAssembler
        
        # Filter to numeric features only
        numeric_features = []
        for f in features:
            if f not in df.columns:
                continue
            dtype = str(df.schema[f].dataType)
            if any(t in dtype for t in ['Int', 'Long', 'Double', 'Float']):
                numeric_features.append(f)
        
        if len(numeric_features) < 2:
            return {}
        
        # Fill nulls with 0 for correlation calculation
        df_clean = df.select(numeric_features).fillna(0)
        
        # Assemble features into vector
        assembler = VectorAssembler(
            inputCols=numeric_features,
            outputCol="features"
        )
        df_vector = assembler.transform(df_clean)
        
        # Calculate correlation matrix
        corr_matrix = Correlation.corr(df_vector, "features", method).collect()[0][0]
        
        # Convert to dictionary
        result = {}
        for i, f1 in enumerate(numeric_features):
            result[f1] = {}
            for j, f2 in enumerate(numeric_features):
                result[f1][f2] = round(corr_matrix[i, j], 4)
        
        return result
    
    def find_high_correlations(
        self,
        correlation_matrix: Dict[str, Dict[str, float]],
        threshold: float = 0.90
    ) -> List[Tuple[str, str, float]]:
        """
        Find feature pairs with high correlation.
        
        Args:
            correlation_matrix: Correlation matrix dictionary
            threshold: Correlation threshold
            
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        high_corr = []
        seen = set()
        
        for f1, corrs in correlation_matrix.items():
            for f2, corr in corrs.items():
                if f1 == f2:
                    continue
                    
                pair = tuple(sorted([f1, f2]))
                if pair in seen:
                    continue
                seen.add(pair)
                
                if abs(corr) >= threshold:
                    high_corr.append((f1, f2, corr))
        
        return sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)
    
    def get_iv_summary(
        self,
        results: Dict[str, FeatureQualityResult]
    ) -> Dict[str, List[str]]:
        """
        Get summary of features by IV category.
        
        Args:
            results: Dictionary of feature quality results
            
        Returns:
            Dictionary of category to list of features
        """
        summary = {
            'useless': [],
            'weak': [],
            'medium': [],
            'strong': [],
            'suspicious': [],
            'unknown': []
        }
        
        for name, result in results.items():
            category = result.iv_category or 'unknown'
            summary[category].append(name)
        
        return summary
