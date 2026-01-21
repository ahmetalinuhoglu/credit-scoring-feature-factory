"""
Data Dictionary Generator

Automatically generates data dictionary from extracted features.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import json

import yaml
from pyspark.sql import functions as F

from src.core.base import SparkComponent


class DataDictionaryGenerator(SparkComponent):
    """
    Generates data dictionary from feature DataFrame.
    
    Creates comprehensive documentation including:
    - Feature names and descriptions
    - Data types
    - Statistics (min, max, mean, std, percentiles)
    - Null ratios
    - IV scores
    - Correlations with target
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        output_dir: Optional[str] = None,
        name: Optional[str] = None
    ):
        """
        Initialize data dictionary generator.
        
        Args:
            config: Configuration dictionary
            spark_session: Active SparkSession
            output_dir: Output directory for dictionary files
            name: Optional generator name
        """
        super().__init__(config, spark_session, name or "DataDictionaryGenerator")
        
        self.output_dir = Path(
            output_dir or
            self.get_config('features.data_dictionary.output_path', 'outputs/data_dictionary')
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature descriptions from config
        self.feature_descriptions = self._load_feature_descriptions()
        
    def validate(self) -> bool:
        return super().validate()
    
    def run(
        self,
        features_df: Any,
        target_column: str = 'target',
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate data dictionary."""
        return self.generate(features_df, target_column, exclude_columns)
    
    def generate(
        self,
        features_df: Any,
        target_column: str = 'target',
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data dictionary.
        
        Args:
            features_df: DataFrame with features
            target_column: Name of target column
            exclude_columns: Columns to exclude from dictionary
            
        Returns:
            Data dictionary as dictionary
        """
        self._start_execution()
        
        exclude = set(exclude_columns or [])
        exclude.update(['application_id', 'customer_id', 'applicant_type', 'application_date'])
        
        # Get feature columns
        feature_columns = [c for c in features_df.columns if c not in exclude and c != target_column]
        
        self.logger.info(f"Generating data dictionary for {len(feature_columns)} features")
        
        # Build dictionary
        dictionary = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_features': len(feature_columns),
                'total_rows': features_df.count(),
                'target_column': target_column
            },
            'features': []
        }
        
        # Calculate statistics for each feature
        for col in feature_columns:
            feature_info = self._analyze_feature(features_df, col, target_column)
            dictionary['features'].append(feature_info)
        
        # Sort by IV score (if available)
        dictionary['features'].sort(
            key=lambda x: x.get('iv_score') or 0,
            reverse=True
        )
        
        self._end_execution()
        
        return dictionary
    
    def _analyze_feature(
        self,
        df: Any,
        feature: str,
        target_column: str
    ) -> Dict[str, Any]:
        """Analyze a single feature and return its metadata."""
        
        total_rows = df.count()
        
        # Basic info
        dtype = str(df.schema[feature].dataType)
        is_numeric = any(t in dtype for t in ['Int', 'Long', 'Double', 'Float'])
        
        info = {
            'name': feature,
            'description': self._get_description(feature),
            'data_type': self._simplify_dtype(dtype),
            'category': self._get_category(feature)
        }
        
        # Null analysis
        null_count = df.filter(F.col(feature).isNull()).count()
        info['null_count'] = null_count
        info['null_ratio'] = round(null_count / total_rows, 4) if total_rows > 0 else 0
        
        # Statistics
        if is_numeric:
            stats = self._calculate_numeric_stats(df, feature)
            info['statistics'] = stats
        else:
            stats = self._calculate_categorical_stats(df, feature)
            info['statistics'] = stats
        
        # IV score (simplified calculation)
        if target_column in df.columns:
            iv = self._calculate_iv_simple(df, feature, target_column)
            info['iv_score'] = iv
            info['iv_category'] = self._categorize_iv(iv)
        
        return info
    
    def _calculate_numeric_stats(
        self,
        df: Any,
        feature: str
    ) -> Dict[str, Any]:
        """Calculate statistics for numeric feature."""
        
        # Basic stats
        stats_row = df.select(
            F.min(feature).alias('min'),
            F.max(feature).alias('max'),
            F.avg(feature).alias('mean'),
            F.stddev(feature).alias('std'),
            F.expr(f'percentile_approx({feature}, 0.25)').alias('p25'),
            F.expr(f'percentile_approx({feature}, 0.50)').alias('median'),
            F.expr(f'percentile_approx({feature}, 0.75)').alias('p75'),
            F.expr(f'percentile_approx({feature}, 0.95)').alias('p95'),
            F.expr(f'percentile_approx({feature}, 0.99)').alias('p99')
        ).collect()[0]
        
        return {
            'min': self._safe_float(stats_row['min']),
            'max': self._safe_float(stats_row['max']),
            'mean': self._safe_float(stats_row['mean']),
            'std': self._safe_float(stats_row['std']),
            'percentiles': {
                '25': self._safe_float(stats_row['p25']),
                '50': self._safe_float(stats_row['median']),
                '75': self._safe_float(stats_row['p75']),
                '95': self._safe_float(stats_row['p95']),
                '99': self._safe_float(stats_row['p99'])
            }
        }
    
    def _calculate_categorical_stats(
        self,
        df: Any,
        feature: str
    ) -> Dict[str, Any]:
        """Calculate statistics for categorical feature."""
        
        value_counts = (
            df.groupBy(feature)
            .count()
            .orderBy(F.desc('count'))
            .limit(10)
            .collect()
        )
        
        return {
            'cardinality': df.select(feature).distinct().count(),
            'top_values': {
                str(row[feature]): row['count'] 
                for row in value_counts
            }
        }
    
    def _calculate_iv_simple(
        self,
        df: Any,
        feature: str,
        target_column: str
    ) -> Optional[float]:
        """Simple IV calculation."""
        import math
        
        try:
            df_clean = df.filter(F.col(feature).isNotNull())
            
            totals = df_clean.groupBy(target_column).count().collect()
            total_goods = sum(r['count'] for r in totals if r[target_column] == 0)
            total_bads = sum(r['count'] for r in totals if r[target_column] == 1)
            
            if total_goods == 0 or total_bads == 0:
                return None
            
            # Simple binning based on value distribution
            dtype = str(df_clean.schema[feature].dataType)
            is_numeric = any(t in dtype for t in ['Int', 'Long', 'Double', 'Float'])
            
            if is_numeric:
                # Quantile-based binning
                quantiles = df_clean.approxQuantile(feature, [0.2, 0.4, 0.6, 0.8], 0.05)
                
                conditions = [
                    F.col(feature) <= quantiles[0],
                    (F.col(feature) > quantiles[0]) & (F.col(feature) <= quantiles[1]),
                    (F.col(feature) > quantiles[1]) & (F.col(feature) <= quantiles[2]),
                    (F.col(feature) > quantiles[2]) & (F.col(feature) <= quantiles[3]),
                    F.col(feature) > quantiles[3]
                ]
                
                df_binned = df_clean.withColumn(
                    'bin',
                    F.when(conditions[0], 0)
                     .when(conditions[1], 1)
                     .when(conditions[2], 2)
                     .when(conditions[3], 3)
                     .otherwise(4)
                )
            else:
                df_binned = df_clean.withColumn('bin', F.col(feature))
            
            bin_stats = (
                df_binned
                .groupBy('bin')
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
                
                goods_pct = (goods + 0.5) / (total_goods + 0.5)
                bads_pct = (bads + 0.5) / (total_bads + 0.5)
                
                woe = math.log(goods_pct / bads_pct)
                iv += (goods_pct - bads_pct) * woe
            
            return round(iv, 4)
            
        except Exception:
            return None
    
    def _categorize_iv(self, iv: Optional[float]) -> str:
        """Categorize IV score."""
        if iv is None:
            return 'unknown'
        elif iv < 0.02:
            return 'useless'
        elif iv < 0.10:
            return 'weak'
        elif iv < 0.30:
            return 'medium'
        elif iv < 0.50:
            return 'strong'
        else:
            return 'suspicious'
    
    def _get_description(self, feature: str) -> str:
        """Get feature description from config or generate."""
        if feature in self.feature_descriptions:
            return self.feature_descriptions[feature]
        
        # Auto-generate description based on feature name
        desc_parts = []
        
        if 'total' in feature.lower():
            desc_parts.append('Total')
        if 'avg' in feature.lower() or 'mean' in feature.lower():
            desc_parts.append('Average')
        if 'max' in feature.lower():
            desc_parts.append('Maximum')
        if 'min' in feature.lower():
            desc_parts.append('Minimum')
        if 'count' in feature.lower():
            desc_parts.append('Count of')
        if 'ratio' in feature.lower():
            desc_parts.append('Ratio of')
        if 'trend' in feature.lower():
            desc_parts.append('Trend in')
        
        # Add product type if present
        for product in ['INSTALLMENT_LOAN', 'INSTALLMENT_SALE', 'CASH_FACILITY', 'MORTGAGE']:
            if product in feature:
                desc_parts.append(product.replace('_', ' ').title())
        
        if desc_parts:
            return ' '.join(desc_parts)
        
        return feature.replace('_', ' ').title()
    
    def _get_category(self, feature: str) -> str:
        """Categorize feature based on name."""
        feature_lower = feature.lower()
        
        if 'amount' in feature_lower or 'exposure' in feature_lower:
            return 'amount'
        elif 'count' in feature_lower:
            return 'count'
        elif 'ratio' in feature_lower:
            return 'ratio'
        elif 'default' in feature_lower:
            return 'default'
        elif any(p.lower() in feature_lower for p in ['installment', 'cash', 'mortgage']):
            return 'product'
        elif 'age' in feature_lower or 'months' in feature_lower or 'days' in feature_lower:
            return 'temporal'
        elif 'trend' in feature_lower or 'velocity' in feature_lower:
            return 'trend'
        elif 'overdraft' in feature_lower or 'overlimit' in feature_lower:
            return 'non_credit_signal'
        else:
            return 'other'
    
    def _simplify_dtype(self, dtype: str) -> str:
        """Simplify Spark data type to readable format."""
        if 'Long' in dtype or 'Int' in dtype:
            return 'integer'
        elif 'Double' in dtype or 'Float' in dtype:
            return 'float'
        elif 'String' in dtype:
            return 'string'
        elif 'Date' in dtype:
            return 'date'
        elif 'Boolean' in dtype:
            return 'boolean'
        else:
            return dtype
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert to float."""
        if value is None:
            return None
        try:
            return round(float(value), 4)
        except:
            return None
    
    def _load_feature_descriptions(self) -> Dict[str, str]:
        """Load feature descriptions from config."""
        descriptions = {}
        
        feature_config = self.get_config('features.feature_engineering', {})
        
        # Parse descriptions from various feature groups
        for group_name, group in feature_config.items():
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, dict) and 'name' in item and 'description' in item:
                        descriptions[item['name']] = item['description']
        
        return descriptions
    
    def export(
        self,
        dictionary: Dict[str, Any],
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export data dictionary to files.
        
        Args:
            dictionary: Data dictionary
            formats: Output formats (yaml, json, html, excel)
            
        Returns:
            Dictionary of format to file path
        """
        formats = formats or self.get_config(
            'features.data_dictionary.formats',
            ['yaml', 'json', 'html']
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_files = {}
        
        if 'yaml' in formats:
            path = self.output_dir / f'data_dictionary_{timestamp}.yaml'
            with open(path, 'w') as f:
                yaml.dump(dictionary, f, default_flow_style=False, allow_unicode=True)
            output_files['yaml'] = str(path)
        
        if 'json' in formats:
            path = self.output_dir / f'data_dictionary_{timestamp}.json'
            with open(path, 'w') as f:
                json.dump(dictionary, f, indent=2, default=str)
            output_files['json'] = str(path)
        
        if 'html' in formats:
            path = self.output_dir / f'data_dictionary_{timestamp}.html'
            html = self._generate_html(dictionary)
            with open(path, 'w') as f:
                f.write(html)
            output_files['html'] = str(path)
        
        self.logger.info(f"Exported data dictionary: {list(output_files.keys())}")
        
        return output_files
    
    def _generate_html(self, dictionary: Dict[str, Any]) -> str:
        """Generate HTML data dictionary."""
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Dictionary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }
        .meta { background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; font-weight: bold; position: sticky; top: 0; }
        tr:hover { background: #f9f9f9; }
        .iv-useless { color: #999; }
        .iv-weak { color: #ff9800; }
        .iv-medium { color: #2196F3; }
        .iv-strong { color: #4CAF50; font-weight: bold; }
        .iv-suspicious { color: #f44336; font-weight: bold; }
        .category { background: #e8e8e8; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📖 Data Dictionary</h1>
        <div class="meta">
            <strong>Generated:</strong> """ + dictionary['metadata']['generated_at'] + """<br>
            <strong>Total Features:</strong> """ + str(dictionary['metadata']['total_features']) + """<br>
            <strong>Total Rows:</strong> """ + f"{dictionary['metadata']['total_rows']:,}" + """
        </div>
        
        <table>
            <tr>
                <th>Feature</th>
                <th>Description</th>
                <th>Type</th>
                <th>Category</th>
                <th>Null %</th>
                <th>IV Score</th>
                <th>IV Category</th>
                <th>Mean</th>
                <th>Std</th>
            </tr>
"""
        
        for feature in dictionary['features']:
            iv_class = f"iv-{feature.get('iv_category', 'unknown')}"
            iv_score = f"{feature.get('iv_score', 0):.4f}" if feature.get('iv_score') else "N/A"
            
            stats = feature.get('statistics', {})
            mean = f"{stats.get('mean', 0):.2f}" if stats.get('mean') is not None else "N/A"
            std = f"{stats.get('std', 0):.2f}" if stats.get('std') is not None else "N/A"
            
            html += f"""
            <tr>
                <td><code>{feature['name']}</code></td>
                <td>{feature.get('description', '')}</td>
                <td>{feature.get('data_type', '')}</td>
                <td><span class="category">{feature.get('category', '')}</span></td>
                <td>{feature.get('null_ratio', 0):.1%}</td>
                <td>{iv_score}</td>
                <td class="{iv_class}">{feature.get('iv_category', 'N/A')}</td>
                <td>{mean}</td>
                <td>{std}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        return html
