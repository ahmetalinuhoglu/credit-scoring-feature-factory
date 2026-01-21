#!/usr/bin/env python3
"""
Generate Data Dictionary with 7 Main Categories

This script generates a comprehensive data dictionary organized by 7 main
feature categories, outputting in HTML, YAML, and JSON formats.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
import json


def generate_data_dictionary():
    """Generate data dictionary for all features."""
    from src.features.feature_factory import FeatureFactory
    
    print("=" * 60)
    print("DATA DICTIONARY GENERATOR")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'feature_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize factory
    factory = FeatureFactory(config)
    
    # Load sample data
    data_dir = project_root / 'data' / 'sample'
    apps_df = pd.read_csv(data_dir / 'sample_applications.csv').head(100)
    bureau_df = pd.read_csv(data_dir / 'sample_credit_bureau.csv')
    
    print(f"Loaded {len(apps_df)} applications, {len(bureau_df)} bureau records")
    
    # Generate features
    print("\n[Step 1] Generating features...")
    features_df = factory.generate_all_features(apps_df, bureau_df, parallel=True)
    print(f"Generated {features_df.shape[1]} features for {features_df.shape[0]} applications")
    
    # Get feature names (excluding ID columns)
    exclude_cols = {'application_id', 'customer_id', 'applicant_type', 'application_date', 'target'}
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    print(f"\n[Step 2] Analyzing {len(feature_cols)} features...")
    
    # 7 Main Categories
    CATEGORY_DEFINITIONS = {
        'volume_amount': {
            'name': 'Volume & Amount',
            'emoji': '📈',
            'description': 'Credit counts, amounts, and aggregations by product type, time window, and status',
        },
        'temporal_history': {
            'name': 'Temporal & History', 
            'emoji': '⏱️',
            'description': 'Credit age, history length, time-to-default, days since events',
        },
        'ratios_composition': {
            'name': 'Ratios & Composition',
            'emoji': '📊', 
            'description': 'Portfolio mix ratios, product concentration, secured vs unsecured proportions',
        },
        'trends_velocity': {
            'name': 'Trends & Velocity',
            'emoji': '📉',
            'description': 'Credit growth trends, acquisition velocity, period comparisons',
        },
        'risk_default': {
            'name': 'Risk & Default',
            'emoji': '⚠️',
            'description': 'Default counts, recovery rates, stress signals, default severity',
        },
        'payment_obligations': {
            'name': 'Payment & Obligations',
            'emoji': '💳',
            'description': 'Monthly payments, remaining terms, DTI proxies, payment burden',
        },
        'behavioral_patterns': {
            'name': 'Behavioral & Patterns',
            'emoji': '🔍',
            'description': 'Credit sequences, burst detection, product transitions, anomalies',
        }
    }
    
    def get_category(feature: str) -> str:
        """Categorize feature based on name."""
        feature_lower = feature.lower()
        
        # Check categories in priority order
        if any(kw in feature_lower for kw in ['default', 'recovery', 'overdraft', 'overlimit', 'stress', 'severity']):
            return 'risk_default'
        if any(kw in feature_lower for kw in ['payment', 'obligation', 'burden', 'dti']):
            return 'payment_obligations'
        if any(kw in feature_lower for kw in ['sequence', 'burst', 'transition', 'first_product', 'last_product', 
                                               'anomaly', 'complexity', 'freshness', 'interval', 'pattern']):
            return 'behavioral_patterns'
        if any(kw in feature_lower for kw in ['trend', 'velocity', 'growth', 'acceleration', '_vs_']):
            return 'trends_velocity'
        if any(kw in feature_lower for kw in ['ratio', 'proportion', 'share', 'concentration', 'hhi', 
                                               'diversity', 'secured', 'unsecured', 'revolving']):
            return 'ratios_composition'
        if any(kw in feature_lower for kw in ['age', 'months', 'days', 'oldest', 'newest', 'history', 
                                               'since', 'time_to', 'duration', 'maturity', 'term', 'seasonal']):
            return 'temporal_history'
        return 'volume_amount'
    
    # Build dictionary
    dictionary = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_features': len(feature_cols),
            'total_rows': len(features_df),
            'categories': list(CATEGORY_DEFINITIONS.keys())
        },
        'categories': CATEGORY_DEFINITIONS,
        'features': []
    }
    
    # Analyze each feature
    for col in feature_cols:
        series = features_df[col]
        
        feature_info = {
            'name': col,
            'category': get_category(col),
            'data_type': 'integer' if series.dtype in [np.int64, np.int32] else 'float',
            'null_count': int(series.isna().sum()),
            'null_ratio': round(series.isna().mean(), 4),
            'statistics': {
                'min': round(float(series.min()), 4) if pd.notna(series.min()) else None,
                'max': round(float(series.max()), 4) if pd.notna(series.max()) else None,
                'mean': round(float(series.mean()), 4) if pd.notna(series.mean()) else None,
                'std': round(float(series.std()), 4) if pd.notna(series.std()) else None,
                'median': round(float(series.median()), 4) if pd.notna(series.median()) else None,
            }
        }
        dictionary['features'].append(feature_info)
    
    # Group by category
    features_by_category = {}
    for f in dictionary['features']:
        cat = f['category']
        if cat not in features_by_category:
            features_by_category[cat] = []
        features_by_category[cat].append(f)
    
    dictionary['summary'] = {cat: len(feats) for cat, feats in features_by_category.items()}
    
    print("\n[Step 3] Feature Distribution by Category:")
    print("-" * 50)
    for cat, feats in sorted(features_by_category.items(), key=lambda x: -len(x[1])):
        cat_info = CATEGORY_DEFINITIONS[cat]
        print(f"  {cat_info['emoji']} {cat_info['name']:25} : {len(feats):4} features")
    print("-" * 50)
    print(f"  {'Total':28} : {len(feature_cols):4} features")
    
    # Export
    output_dir = Path(__file__).parent.parent / 'outputs' / 'data_dictionary'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON
    json_path = output_dir / f'data_dictionary_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(dictionary, f, indent=2, default=str)
    print(f"\n✓ Saved: {json_path}")
    
    # YAML
    yaml_path = output_dir / f'data_dictionary_{timestamp}.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dictionary, f, default_flow_style=False, allow_unicode=True)
    print(f"✓ Saved: {yaml_path}")
    
    # HTML
    html_path = output_dir / f'data_dictionary_{timestamp}.html'
    html = generate_html(dictionary, CATEGORY_DEFINITIONS, features_by_category)
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"✓ Saved: {html_path}")
    
    print("\n" + "=" * 60)
    print("✓ DATA DICTIONARY GENERATED SUCCESSFULLY!")
    print("=" * 60)
    
    return dictionary


def generate_html(dictionary, categories, features_by_category):
    """Generate beautiful HTML data dictionary."""
    
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
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Credit Scoring Feature Dictionary</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
        
        .meta {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; margin: 20px 0; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; text-align: center; }}
        .meta-item {{ background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; }}
        .meta-value {{ font-size: 2em; font-weight: bold; }}
        .meta-label {{ font-size: 0.9em; opacity: 0.9; }}
        
        .category-nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; justify-content: center; }}
        .category-btn {{ padding: 10px 20px; border-radius: 20px; color: white; text-decoration: none; font-weight: bold; transition: all 0.2s; }}
        .category-btn:hover {{ transform: scale(1.05); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
        
        .category-section {{ background: white; border-radius: 12px; margin: 20px 0; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .category-header {{ padding: 20px; color: white; }}
        .category-header h2 {{ margin: 0 0 5px 0; }}
        .category-header p {{ margin: 0; opacity: 0.9; font-size: 0.95em; }}
        .category-count {{ background: rgba(255,255,255,0.3); padding: 5px 15px; border-radius: 20px; float: right; font-weight: bold; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; position: sticky; top: 0; }}
        tr:hover {{ background: #f9f9f9; }}
        
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-family: 'Monaco', 'Consolas', monospace; font-size: 0.85em; }}
        
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 500; }}
        .badge-float {{ background: #e3f2fd; color: #1976D2; }}
        .badge-integer {{ background: #e8f5e9; color: #388E3C; }}
        
        .footer {{ text-align: center; padding: 30px; color: #666; font-size: 0.9em; }}
        .footer a {{ color: #2196F3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📖 Credit Scoring Feature Dictionary</h1>
        
        <div class="meta">
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-value">{dictionary['metadata']['total_features']}</div>
                    <div class="meta-label">Total Features</div>
                </div>
                <div class="meta-item">
                    <div class="meta-value">7</div>
                    <div class="meta-label">Categories</div>
                </div>
                <div class="meta-item">
                    <div class="meta-value">{dictionary['metadata']['total_rows']:,}</div>
                    <div class="meta-label">Sample Rows</div>
                </div>
                <div class="meta-item">
                    <div class="meta-value">{dictionary['metadata']['generated_at'][:10]}</div>
                    <div class="meta-label">Generated</div>
                </div>
            </div>
        </div>
        
        <div class="category-nav">
"""
    
    # Navigation buttons
    for cat in category_order:
        if cat in features_by_category:
            cat_info = categories[cat]
            color = category_colors[cat]
            count = len(features_by_category[cat])
            html += f'<a href="#{cat}" class="category-btn" style="background: {color};">{cat_info["emoji"]} {cat_info["name"]} ({count})</a>\n'
    
    html += "</div>\n"
    
    # Category sections
    for cat in category_order:
        if cat not in features_by_category:
            continue
        
        cat_info = categories[cat]
        color = category_colors[cat]
        feats = features_by_category[cat]
        
        html += f'''
        <div class="category-section" id="{cat}">
            <div class="category-header" style="background: {color};">
                <span class="category-count">{len(feats)} features</span>
                <h2>{cat_info["emoji"]} {cat_info["name"]}</h2>
                <p>{cat_info["description"]}</p>
            </div>
            <table>
                <tr>
                    <th>Feature Name</th>
                    <th>Type</th>
                    <th>Null %</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Std</th>
                </tr>
'''
        
        # Sort by name
        for f in sorted(feats, key=lambda x: x['name']):
            stats = f.get('statistics', {})
            dtype = f.get('data_type', '')
            dtype_class = 'badge-integer' if dtype == 'integer' else 'badge-float'
            
            html += f'''
                <tr>
                    <td><code>{f['name']}</code></td>
                    <td><span class="badge {dtype_class}">{dtype}</span></td>
                    <td>{f['null_ratio']:.1%}</td>
                    <td>{stats.get('min', 'N/A')}</td>
                    <td>{stats.get('max', 'N/A')}</td>
                    <td>{stats.get('mean', 'N/A')}</td>
                    <td>{stats.get('std', 'N/A')}</td>
                </tr>
'''
        
        html += "</table></div>\n"
    
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


if __name__ == "__main__":
    generate_data_dictionary()
