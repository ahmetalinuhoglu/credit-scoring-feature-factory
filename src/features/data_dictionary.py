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
    
    # 7 Main Feature Categories
    CATEGORY_DEFINITIONS = {
        'volume_amount': {
            'name': 'Volume & Amount',
            'emoji': '📈',
            'description': 'Credit counts, amounts, and aggregations by product type, time window, and status',
            'keywords': ['amount', 'count', 'total', 'sum', 'exposure', '_amt', '_cnt']
        },
        'temporal_history': {
            'name': 'Temporal & History', 
            'emoji': '⏱️',
            'description': 'Credit age, history length, time-to-default, days since events',
            'keywords': ['age', 'months', 'days', 'oldest', 'newest', 'history', 'since', 'time_to', 'duration', 'maturity', 'term', 'seasonal']
        },
        'ratios_composition': {
            'name': 'Ratios & Composition',
            'emoji': '📊', 
            'description': 'Portfolio mix ratios, product concentration, secured vs unsecured proportions',
            'keywords': ['ratio', 'proportion', 'share', 'concentration', 'hhi', 'diversity', 'secured', 'unsecured', 'revolving']
        },
        'trends_velocity': {
            'name': 'Trends & Velocity',
            'emoji': '📉',
            'description': 'Credit growth trends, acquisition velocity, period comparisons',
            'keywords': ['trend', 'velocity', 'growth', 'acceleration', 'vs_', '_vs_']
        },
        'risk_default': {
            'name': 'Risk & Default',
            'emoji': '⚠️',
            'description': 'Default counts, recovery rates, stress signals, default severity',
            'keywords': ['default', 'recovery', 'overdraft', 'overlimit', 'stress', 'risk', 'severity', 'ever_defaulted', 'current_default']
        },
        'payment_obligations': {
            'name': 'Payment & Obligations',
            'emoji': '💳',
            'description': 'Monthly payments, remaining terms, DTI proxies, payment burden',
            'keywords': ['payment', 'obligation', 'burden', 'dti', 'installment', 'remaining']
        },
        'behavioral_patterns': {
            'name': 'Behavioral & Patterns',
            'emoji': '🔍',
            'description': 'Credit sequences, burst detection, product transitions, anomalies',
            'keywords': ['sequence', 'burst', 'transition', 'pattern', 'first_product', 'last_product', 'anomaly', 'complexity', 'freshness', 'interval']
        }
    }
    
    def _get_category(self, feature: str) -> str:
        """
        Categorize feature based on name using 7 main categories.
        
        Categories:
        1. volume_amount - Credit counts, amounts, aggregations
        2. temporal_history - Age, history, time metrics
        3. ratios_composition - Portfolio ratios, concentrations
        4. trends_velocity - Growth trends, velocity metrics
        5. risk_default - Default, recovery, stress signals
        6. payment_obligations - Payments, terms, burden
        7. behavioral_patterns - Sequences, patterns, anomalies
        """
        feature_lower = feature.lower()
        
        # Check categories in priority order (more specific first)
        
        # 5. Risk & Default (check first as it's most important)
        if any(kw in feature_lower for kw in ['default', 'recovery', 'overdraft', 'overlimit', 'stress', 'severity']):
            return 'risk_default'
        
        # 6. Payment & Obligations
        if any(kw in feature_lower for kw in ['payment', 'obligation', 'burden', 'dti']):
            return 'payment_obligations'
        
        # 7. Behavioral & Patterns
        if any(kw in feature_lower for kw in ['sequence', 'burst', 'transition', 'first_product', 'last_product', 
                                               'anomaly', 'complexity', 'freshness', 'interval', 'pattern']):
            return 'behavioral_patterns'
        
        # 4. Trends & Velocity
        if any(kw in feature_lower for kw in ['trend', 'velocity', 'growth', 'acceleration', '_vs_']):
            return 'trends_velocity'
        
        # 3. Ratios & Composition
        if any(kw in feature_lower for kw in ['ratio', 'proportion', 'share', 'concentration', 'hhi', 
                                               'diversity', 'secured', 'unsecured', 'revolving']):
            return 'ratios_composition'
        
        # 2. Temporal & History
        if any(kw in feature_lower for kw in ['age', 'months', 'days', 'oldest', 'newest', 'history', 
                                               'since', 'time_to', 'duration', 'maturity', 'term', 'seasonal']):
            return 'temporal_history'
        
        # 1. Volume & Amount (default for count/amount features)
        if any(kw in feature_lower for kw in ['amount', 'count', 'total', 'sum', 'exposure', '_amt', '_cnt']):
            return 'volume_amount'
        
        # Default to volume_amount for unclassified features
        return 'volume_amount'
    
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
        """Generate HTML data dictionary grouped by 7 main categories."""
        
        # Group features by category
        features_by_category = {}
        for feature in dictionary['features']:
            cat = feature.get('category', 'volume_amount')
            if cat not in features_by_category:
                features_by_category[cat] = []
            features_by_category[cat].append(feature)
        
        # Category order and styling
        category_order = [
            'volume_amount', 'temporal_history', 'ratios_composition',
            'trends_velocity', 'risk_default', 'payment_obligations', 'behavioral_patterns'
        ]
        
        category_colors = {
            'volume_amount': '#4CAF50',
            'temporal_history': '#2196F3',
            'ratios_composition': '#9C27B0',
            'trends_velocity': '#FF9800',
            'risk_default': '#f44336',
            'payment_obligations': '#00BCD4',
            'behavioral_patterns': '#795548'
        }
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Dictionary - Credit Scoring Features</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .meta { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin: 20px 0; }
        .meta-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; text-align: center; }
        .meta-item { background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; }
        .meta-value { font-size: 2em; font-weight: bold; }
        .meta-label { font-size: 0.9em; opacity: 0.9; }
        
        .category-nav { display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; justify-content: center; }
        .category-btn { padding: 10px 20px; border-radius: 20px; color: white; text-decoration: none; font-weight: bold; transition: transform 0.2s; }
        .category-btn:hover { transform: scale(1.05); }
        
        .category-section { background: white; border-radius: 12px; margin: 20px 0; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .category-header { padding: 20px; color: white; }
        .category-header h2 { margin: 0 0 5px 0; }
        .category-header p { margin: 0; opacity: 0.9; font-size: 0.95em; }
        .category-count { background: rgba(255,255,255,0.3); padding: 5px 15px; border-radius: 20px; float: right; font-weight: bold; }
        
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; position: sticky; top: 0; }
        tr:hover { background: #f5f5f5; }
        
        .iv-useless { color: #999; }
        .iv-weak { color: #ff9800; }
        .iv-medium { color: #2196F3; font-weight: 500; }
        .iv-strong { color: #4CAF50; font-weight: bold; }
        .iv-suspicious { color: #f44336; font-weight: bold; }
        
        .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 500; }
        .badge-float { background: #e3f2fd; color: #1976D2; }
        .badge-integer { background: #e8f5e9; color: #388E3C; }
        
        code { background: #f5f5f5; padding: 2px 6px; border-radius: 4px; font-family: 'Monaco', monospace; font-size: 0.85em; }
        
        .footer { text-align: center; padding: 20px; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📖 Credit Scoring Feature Dictionary</h1>
        
        <div class="meta">
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-value">""" + str(dictionary['metadata']['total_features']) + """</div>
                    <div class="meta-label">Total Features</div>
                </div>
                <div class="meta-item">
                    <div class="meta-value">""" + str(len(features_by_category)) + """</div>
                    <div class="meta-label">Categories</div>
                </div>
                <div class="meta-item">
                    <div class="meta-value">""" + f"{dictionary['metadata']['total_rows']:,}" + """</div>
                    <div class="meta-label">Training Rows</div>
                </div>
                <div class="meta-item">
                    <div class="meta-value">""" + dictionary['metadata']['generated_at'][:10] + """</div>
                    <div class="meta-label">Generated Date</div>
                </div>
            </div>
        </div>
        
        <div class="category-nav">
"""
        
        # Generate category navigation
        for cat in category_order:
            if cat in features_by_category:
                cat_info = self.CATEGORY_DEFINITIONS.get(cat, {})
                color = category_colors.get(cat, '#666')
                count = len(features_by_category[cat])
                html += f'''<a href="#{cat}" class="category-btn" style="background: {color};">
                    {cat_info.get('emoji', '📋')} {cat_info.get('name', cat)} ({count})
                </a>'''
        
        html += """</div>"""
        
        # Generate category sections
        for cat in category_order:
            if cat not in features_by_category:
                continue
                
            cat_info = self.CATEGORY_DEFINITIONS.get(cat, {})
            color = category_colors.get(cat, '#666')
            features = features_by_category[cat]
            
            html += f'''
        <div class="category-section" id="{cat}">
            <div class="category-header" style="background: {color};">
                <span class="category-count">{len(features)} features</span>
                <h2>{cat_info.get('emoji', '📋')} {cat_info.get('name', cat)}</h2>
                <p>{cat_info.get('description', '')}</p>
            </div>
            <table>
                <tr>
                    <th>Feature Name</th>
                    <th>Description</th>
                    <th>Type</th>
                    <th>Null %</th>
                    <th>IV Score</th>
                    <th>IV Category</th>
                    <th>Mean</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
'''
            
            # Sort features by IV score within category
            sorted_features = sorted(features, key=lambda x: x.get('iv_score') or 0, reverse=True)
            
            for feature in sorted_features:
                iv_class = f"iv-{feature.get('iv_category', 'unknown')}"
                iv_score = f"{feature.get('iv_score', 0):.4f}" if feature.get('iv_score') else "N/A"
                
                stats = feature.get('statistics', {})
                mean_val = f"{stats.get('mean', 0):.2f}" if stats.get('mean') is not None else "N/A"
                min_val = f"{stats.get('min', 0):.2f}" if stats.get('min') is not None else "N/A"
                max_val = f"{stats.get('max', 0):.2f}" if stats.get('max') is not None else "N/A"
                
                dtype = feature.get('data_type', '')
                dtype_class = 'badge-integer' if dtype == 'integer' else 'badge-float'
                
                html += f'''
                <tr>
                    <td><code>{feature['name']}</code></td>
                    <td>{feature.get('description', '')}</td>
                    <td><span class="badge {dtype_class}">{dtype}</span></td>
                    <td>{feature.get('null_ratio', 0):.1%}</td>
                    <td>{iv_score}</td>
                    <td class="{iv_class}">{feature.get('iv_category', 'N/A')}</td>
                    <td>{mean_val}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                </tr>
'''
            
            html += """</table></div>"""
        
        html += """
        <div class="footer">
            <p>Generated by Credit Scoring Feature Factory | 
            <a href="https://github.com/ahmetalinuhoglu/credit-scoring-feature-factory">GitHub</a></p>
        </div>
    </div>
</body>
</html>
"""
        return html
