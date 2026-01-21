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
    
    class DescriptionGenerator:
        """
        Intelligent feature description generator.
        
        Parses feature names systematically and generates natural,
        human-readable descriptions without any hard-coding.
        """
        
        # Mappings are defined as class attributes for easy extension
        AGGREGATIONS = {
            'total': 'Total',
            'sum': 'Sum of',
            'avg': 'Average',
            'average': 'Average',
            'mean': 'Mean',
            'max': 'Maximum',
            'min': 'Minimum',
            'std': 'Standard deviation of',
            'count': 'Number of',
            'cnt': 'Count of',
            'median': 'Median',
        }
        
        PRODUCTS = {
            'installment_loan': 'installment loan',
            'installment_sale': 'installment sale', 
            'cash_facility': 'cash facility',
            'mortgage': 'mortgage',
            'il': 'installment loan',
            'is': 'installment sale',
            'cf': 'cash facility',
            'mg': 'mortgage',
        }
        
        STATUSES = {
            'active': 'active',
            'defaulted': 'defaulted',
            'recovered': 'recovered',
            'current_default': 'currently defaulted',
            'ever_defaulted': 'ever defaulted',
        }
        
        TIME_WINDOWS = {
            'last_3m': 'in the last 3 months',
            'last_6m': 'in the last 6 months',
            'last_12m': 'in the last 12 months',
            'last_24m': 'in the last 24 months',
            '_3m': 'in the last 3 months',
            '_6m': 'in the last 6 months',
            '_12m': 'in the last 12 months',
            '_24m': 'in the last 24 months',
        }
        
        METRICS = {
            'amount': 'credit amount',
            'payment': 'payment',
            'monthly_payment': 'monthly payment',
            'exposure': 'credit exposure',
            'credit': 'credit',
            'age': 'age',
            'term': 'term',
            'remaining_term': 'remaining term',
        }
        
        SPECIAL_PATTERNS = {
            # Ratios
            'to_total_ratio': 'as a proportion of total credits',
            'amt_to_total_ratio': 'amount as a proportion of total credit amount',
            'ratio': 'ratio',
            
            # Velocity & Trends
            'velocity': 'acquisition rate',
            'trend': 'trend (change over time)',
            'vs': 'compared to',
            
            # Temporal
            'days_since': 'days since',
            'months': 'months',
            'age_months': 'age in months',
            'oldest': 'oldest',
            'newest': 'newest', 
            'history_length': 'history length',
            
            # Risk signals
            'overdraft': 'overdraft event',
            'overlimit': 'overlimit event',
            'stress': 'financial stress indicator',
            'severity': 'severity',
            
            # Behavioral
            'first_product': 'first credit product was',
            'last_product': 'most recent credit product was',
            'transition': 'product type transition',
            'burst': 'credit burst (multiple credits in short period)',
            'interval': 'interval between credits',
            'sequence': 'credit sequence',
            'complexity': 'portfolio complexity',
            'freshness': 'credit freshness',
            'anomaly': 'anomaly indicator',
            
            # Diversity
            'diversity': 'diversity measure',
            'concentration': 'concentration measure',
            'hhi': 'Herfindahl-Hirschman Index (concentration)',
            'distinct': 'distinct/unique',
            
            # Flags
            'has_': 'indicator flag for',
            'is_': 'indicator flag for',
            'flag': 'binary indicator',
            
            # Payment
            'dti': 'debt-to-income proxy',
            'burden': 'payment burden',
            'obligation': 'payment obligation',
        }
        
        @classmethod
        def generate(cls, feature_name: str) -> str:
            """Generate a natural, descriptive explanation for a feature."""
            original = feature_name
            name = feature_name.lower()
            
            # ==========================================
            # PHASE 1: Handle special complete patterns
            # ==========================================
            
            # Velocity patterns
            if 'velocity' in name:
                time_part = ''
                for tw in ['3m', '6m', '12m', '24m']:
                    if tw in name:
                        time_part = f" in the last {tw.replace('m', ' months')}"
                        break
                metric = 'credit count' if 'credit' in name or 'cnt' in name else 'credit amount'
                if 'amount' in name or 'amt' in name:
                    metric = 'credit amount'
                return f"Rate of {metric} acquisition relative to total portfolio{time_part}"
            
            # Growth rate patterns
            if 'growth_rate' in name:
                if 'amount' in name:
                    return "Percentage growth rate of credit amount over time"
                elif 'count' in name or 'credit_count' in name:
                    return "Percentage growth rate of credit count over time"
                return "Growth rate indicator measuring change over time"
            
            # Payment count pattern
            if name == 'payment_count':
                return "Number of credits with active payment obligations"
            
            # Concentration ratio patterns
            if 'concentration_ratio' in name or 'concentration' in name:
                if '3m' in name:
                    return "Concentration of credits in last 3 months relative to total portfolio"
                elif '6m' in name:
                    return "Concentration of credits in last 6 months relative to total portfolio"
                elif '12m' in name:
                    return "Concentration of credits in last 12 months relative to total portfolio"
                return "Credit concentration measure (higher = more concentrated)"
            
            # Trend patterns
            if 'trend' in name:
                if 'vs' in name:
                    # Parse "6m_vs_12m" pattern
                    if '6m_vs_12m' in name or '6m vs 12m' in name.replace('_', ' '):
                        metric = 'credit count' if 'cnt' in name else 'credit amount'
                        return f"Change in {metric}: last 6 months compared to previous 6 months (positive = increasing)"
                return f"Trend indicator measuring change over time for {original.replace('_trend', '').replace('trend_', '').replace('_', ' ')}"
            
            # Days since patterns
            if 'days_since' in name:
                event = name.replace('days_since_', '').replace('days_since', '').replace('_', ' ')
                if 'default' in event:
                    return "Number of days elapsed since the customer's most recent credit default"
                elif 'last_credit' in name or 'credit' in event:
                    return "Number of days elapsed since the customer's most recent credit was opened"
                return f"Number of days since {event.strip()}"
            
            # Oldest/newest age patterns
            if 'oldest' in name and ('age' in name or 'months' in name):
                return "Age of the customer's oldest credit in months (credit history length)"
            if 'newest' in name and ('age' in name or 'months' in name):
                return "Age of the customer's most recently opened credit in months"
            if 'avg' in name and 'age' in name and 'months' in name:
                return "Average age of all customer credits in months"
            if 'age_months' in name or ('age' in name and 'month' in name):
                return "Credit age measured in months"
            
            # History length
            if 'history_length' in name or 'history' in name and 'length' in name:
                return "Total length of credit history in months (from first credit to reference date)"
            
            # =============================================
            # PRODUCT RATIO PATTERNS - Check before flags!
            # =============================================
            # These patterns start with 2-letter product codes
            if 'to_total_ratio' in name or 'amt_to_total_ratio' in name:
                ratio_products = {
                    'il_': 'installment loan',
                    'is_': 'installment sale',
                    'cf_': 'cash facility',
                    'mg_': 'mortgage',
                }
                for prefix, prod_name in ratio_products.items():
                    if name.startswith(prefix):
                        if 'amt' in name:
                            return f"Proportion of total credit amount that is from {prod_name} products"
                        return f"Proportion of total credit count that is {prod_name} products"
            
            # Product monthly payment patterns (is_, il_, cf_, mg_ prefixes)
            if 'monthly_payment' in name:
                payment_products = {
                    'il_': 'installment loan',
                    'is_': 'installment sale',
                    'cf_': 'cash facility',
                    'mg_': 'mortgage',
                }
                for prefix, prod_name in payment_products.items():
                    if name.startswith(prefix):
                        if 'total' in name:
                            return f"Total monthly payment amount across all {prod_name} credits"
                        elif 'avg' in name or 'average' in name:
                            return f"Average monthly payment amount for {prod_name} credits"
                        elif 'max' in name:
                            return f"Maximum monthly payment amount among {prod_name} credits"
                        return f"Monthly payment for {prod_name} credits"
            
            # =============================================  
            # VS COMPARISON PATTERNS
            # =============================================
            if '_vs_' in name:
                if 'primary_vs_co' in name:
                    return "Ratio of primary applicant credit amount to co-applicant credit amount"
                elif 'latest_vs_first' in name:
                    return "Ratio of most recent credit amount to first credit amount (credit growth indicator)"
                elif '6m_vs_12m' in name:
                    metric = 'credit count' if 'cnt' in name else 'credit amount'
                    return f"Change in {metric}: last 6 months compared to previous 6 months (positive = increasing)"
                elif '3m_vs_6m' in name:
                    metric = 'credit count' if 'cnt' in name else 'credit amount'
                    return f"Change in {metric}: last 3 months compared to previous 3 months (positive = increasing)"
                # Generic vs pattern
                parts = name.split('_vs_')
                if len(parts) == 2:
                    left = parts[0].replace('_', ' ')
                    right = parts[1].replace('_', ' ')
                    return f"Comparison ratio: {left} relative to {right}"
            
            if 'vs_average' in name:
                return "Ratio of most recent credit amount to average credit amount (recent credit size indicator)"
            if 'vs_max' in name:
                return "Ratio of most recent credit amount to maximum credit amount"
            
            # =============================================
            # FLAG PATTERNS (has_, is_ for boolean flags)
            # =============================================
            if name.startswith('has_'):
                remainder = name.replace('has_', '').replace('_', ' ')
                if 'current_default' in name:
                    return "Binary flag: 1 if customer has at least one currently defaulted credit, 0 otherwise"
                elif 'ever_default' in name:
                    return "Binary flag: 1 if customer has ever had a defaulted credit in their history, 0 otherwise"
                elif 'multiple_default' in name:
                    return "Binary flag: 1 if customer has defaulted on more than one credit, 0 otherwise"
                elif 'overdraft' in name:
                    return "Binary flag: 1 if customer has any unauthorized overdraft events, 0 otherwise"
                elif 'overlimit' in name:
                    return "Binary flag: 1 if customer has any overlimit events, 0 otherwise"
                elif 'credit_burst' in name or 'burst' in name:
                    return "Binary flag: 1 if customer had a credit burst (3+ credits within 30 days), 0 otherwise"
                return f"Binary flag: 1 if customer has {remainder}, 0 otherwise"
            
            # is_ flag (only if NOT a product code context)
            if name.startswith('is_') and 'to_total' not in name and 'monthly_payment' not in name and 'amount' not in name:
                remainder = name.replace('is_', '').replace('_', ' ')
                return f"Binary flag: 1 if {remainder}, 0 otherwise"
            
            # First/last product patterns
            if 'first_product' in name or 'last_product' in name:
                which = 'first' if 'first' in name else 'most recent'
                for prod_code, prod_name in sorted(cls.PRODUCTS.items(), key=lambda x: -len(x[0])):
                    if prod_code in name:
                        return f"Binary flag: 1 if the customer's {which} credit product was a {prod_name}"
            
            # Product transition patterns
            if 'transition' in name:
                if 'count' in name:
                    return "Count of product type transitions (switches between different credit product types)"
                return "Product type transition indicator"
            
            # Status ratio patterns
            if name == 'active_ratio':
                return "Proportion of total credits that are currently active (not closed or defaulted)"
            if name == 'recovered_ratio':
                return "Proportion of defaulted credits that have subsequently recovered"
            if name == 'default_ratio' or name == 'defaulted_ratio':
                return "Proportion of total credits that are in default status"
            if '_ratio' in name and 'to_total' not in name and '_vs_' not in name:
                # Generic ratio pattern
                parts = name.replace('_ratio', '').replace('_', ' ')
                return f"Ratio measuring {parts}"
            
            # Diversity/concentration patterns
            if 'diversity' in name:
                if 'product' in name:
                    return "Product type diversity score (0-1): ratio of distinct product types used to max possible"
                return "Diversity measure of credit portfolio composition"
            if 'hhi' in name or 'concentration' in name:
                return "Herfindahl-Hirschman Index: concentration measure (higher = less diverse, 1 = single product type)"
            if 'distinct_product' in name:
                return "Count of distinct/unique credit product types in customer's portfolio"
            
            # Burst patterns
            if 'burst' in name:
                time_part = ''
                for tw in ['3m', '6m', '12m', '24m']:
                    if tw in name:
                        time_part = f" in {tw.replace('m', '-month')} window"
                        break
                return f"Count of credit acquisition bursts (3+ credits in 30 days){time_part}"
            
            # Interval patterns
            if 'interval' in name:
                agg = 'average' if 'avg' in name or 'mean' in name else 'minimum' if 'min' in name else 'maximum' if 'max' in name else ''
                return f"{'Average time' if not agg else agg.capitalize() + ' time'} interval between consecutive credit openings in days"
            
            # Complexity patterns
            if 'complexity' in name:
                return "Portfolio complexity score based on number of product types and credit patterns"
            
            # Freshness patterns
            if 'freshness' in name:
                return "Credit portfolio freshness: weighted score based on how recently credits were opened"
            
            # Anomaly patterns
            if 'anomaly' in name:
                return "Anomaly score indicating unusual patterns in credit behavior (higher = more unusual)"
            
            # Severity patterns
            if 'severity' in name:
                agg = 'average' if 'avg' in name else 'maximum' if 'max' in name else ''
                return f"{agg.capitalize() + ' ' if agg else ''}severity of defaults measured by defaulted credit amount"
            
            # DTI patterns
            if 'dti' in name:
                return "Debt-to-income proxy: ratio of monthly payment obligations to estimated income"
            
            # Burden patterns
            if 'burden' in name:
                return "Payment burden: ratio of total monthly payments to average credit amount"
            
            # Recovery patterns
            if 'recovery' in name:
                if 'rate' in name or 'ratio' in name:
                    return "Proportion of defaulted credits that have subsequently recovered"
                elif 'time' in name:
                    return "Average time in days from default to recovery"
                elif 'count' in name or 'cnt' in name:
                    return "Number of credits that have recovered from default status"
            
            # ==========================================
            # PHASE 2: Build description from components
            # ==========================================
            
            parts = []
            
            # Extract time window and remove from name for cleaner parsing
            time_window = None
            name_clean = name
            for tw_key, tw_desc in sorted(cls.TIME_WINDOWS.items(), key=lambda x: -len(x[0])):
                if tw_key in name:
                    time_window = tw_desc
                    name_clean = name.replace(tw_key, '')
                    break
            
            # Extract aggregation
            aggregation = None
            agg_order = sorted(cls.AGGREGATIONS.items(), key=lambda x: -len(x[0]))
            for agg_key, agg_desc in agg_order:
                # Check if it's a whole word match
                name_parts = name_clean.replace('_', ' ').split()
                if agg_key in name_parts:
                    aggregation = agg_desc
                    break
                elif name_clean.startswith(agg_key + '_'):
                    aggregation = agg_desc
                    break
            
            # Extract status
            status = None
            for stat_key, stat_desc in sorted(cls.STATUSES.items(), key=lambda x: -len(x[0])):
                if stat_key in name_clean:
                    status = stat_desc
                    break
            
            # Extract product type - be careful with 2-letter codes
            product = None
            # First check full product names
            for prod_key in ['installment_loan', 'installment_sale', 'cash_facility', 'mortgage']:
                if prod_key in name_clean:
                    product = cls.PRODUCTS[prod_key]
                    break
            # Then check 2-letter prefixes (must be at start followed by _)
            if not product:
                two_letter_products = [('il_', 'installment loan'), ('cf_', 'cash facility'), ('mg_', 'mortgage')]
                # Note: 'is_' is tricky, only match if specifically product context
                for prefix, prod_name in two_letter_products:
                    if name_clean.startswith(prefix):
                        product = prod_name
                        break
                # Handle is_ only in specific product contexts
                if not product and name_clean.startswith('is_') and ('to_total' in name_clean or 'amount' in name_clean or 'count' in name_clean):
                    product = 'installment sale'
            
            # Build the description
            if aggregation:
                parts.append(aggregation)
            
            if status:
                parts.append(status)
            
            if product:
                parts.append(product)
            
            # Determine the subject/metric
            if 'amount' in name_clean and 'count' not in name_clean:
                parts.append('credit amount')
            elif 'count' in name_clean or 'cnt' in name_clean:
                parts.append('credits')
            elif 'payment' in name_clean:
                parts.append('monthly payment amount')
            elif 'term' in name_clean:
                parts.append('term length in months')
            else:
                # Default subject based on context
                if not product and not status:
                    parts.append('credits')
                elif 'credit' not in ' '.join(parts).lower():
                    parts.append('credit')
            
            # Add time window
            if time_window:
                parts.append(time_window)
            
            # Build final description
            if parts:
                desc = ' '.join(parts)
                desc = desc[0].upper() + desc[1:]
                # Clean up grammar issues
                desc = desc.replace('  ', ' ')
                desc = desc.replace('credits credits', 'credits')
                desc = desc.replace('credit credit', 'credit')
                return desc
            
            # Fallback: convert feature name to readable format
            return original.replace('_', ' ').replace('-', ' ').title()
    
    def generate_description(feature: str) -> str:
        """Generate description using DescriptionGenerator class."""
        return DescriptionGenerator.generate(feature)
    
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
            'description': generate_description(col),
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
    
    # Excel (xlsx) with category-based sheets
    xlsx_path = output_dir / f'data_dictionary_{timestamp}.xlsx'
    generate_excel(dictionary, CATEGORY_DEFINITIONS, features_by_category, xlsx_path)
    print(f"✓ Saved: {xlsx_path}")
    
    print("\n" + "=" * 60)
    print("✓ DATA DICTIONARY GENERATED SUCCESSFULLY!")
    print("=" * 60)
    
    return dictionary


def generate_excel(dictionary, categories, features_by_category, output_path):
    """Generate Excel data dictionary with category-based sheets."""
    
    category_order = [
        'volume_amount', 'temporal_history', 'ratios_composition',
        'trends_velocity', 'risk_default', 'payment_obligations', 'behavioral_patterns'
    ]
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for cat in category_order:
            if cat in features_by_category:
                cat_info = categories[cat]
                summary_data.append({
                    'Category Code': cat,
                    'Category Name': cat_info['name'],
                    'Description': cat_info['description'],
                    'Feature Count': len(features_by_category[cat])
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.loc[len(summary_df)] = ['', 'TOTAL', '', summary_df['Feature Count'].sum()]
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # All features sheet
        all_features = []
        for f in dictionary['features']:
            cat_info = categories.get(f['category'], {})
            stats = f.get('statistics', {})
            all_features.append({
                'Feature Name': f['name'],
                'Description': f.get('description', ''),
                'Category': cat_info.get('name', f['category']),
                'Data Type': f['data_type'],
                'Null %': f['null_ratio'],
                'Min': stats.get('min'),
                'Max': stats.get('max'),
                'Mean': stats.get('mean'),
                'Std': stats.get('std'),
                'Median': stats.get('median'),
            })
        
        all_df = pd.DataFrame(all_features)
        all_df.to_excel(writer, sheet_name='All Features', index=False)
        
        # Category-specific sheets
        for cat in category_order:
            if cat not in features_by_category:
                continue
            
            cat_info = categories[cat]
            feats = features_by_category[cat]
            
            cat_data = []
            for f in sorted(feats, key=lambda x: x['name']):
                stats = f.get('statistics', {})
                cat_data.append({
                    'Feature Name': f['name'],
                    'Description': f.get('description', ''),
                    'Data Type': f['data_type'],
                    'Null %': f['null_ratio'],
                    'Min': stats.get('min'),
                    'Max': stats.get('max'),
                    'Mean': stats.get('mean'),
                    'Std': stats.get('std'),
                    'Median': stats.get('median'),
                })
            
            cat_df = pd.DataFrame(cat_data)
            # Truncate sheet name to 31 chars (Excel limit)
            sheet_name = cat_info['name'][:31]
            cat_df.to_excel(writer, sheet_name=sheet_name, index=False)


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
