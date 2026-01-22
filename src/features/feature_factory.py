"""
Feature Factory

Config-driven feature generator that creates hundreds of credit risk 
variables from template combinations (product types × time windows × 
status filters × aggregations).
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import itertools
import json
import yaml

import numpy as np
import pandas as pd


@dataclass
class FeatureDefinition:
    """Definition of a single feature with metadata."""
    name: str
    description: str
    formula: str
    category: str
    data_type: str = "float64"
    product: Optional[str] = None
    time_window: Optional[str] = None
    status_filter: Optional[str] = None
    aggregation: Optional[str] = None
    unit: str = "numeric"
    expected_range: Tuple[Optional[float], Optional[float]] = (None, None)
    null_handling: float = 0.0
    business_interpretation: str = ""
    risk_direction: str = "neutral"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            'name': self.name,
            'description': self.description,
            'formula': self.formula,
            'category': self.category,
            'data_type': self.data_type,
            'product': self.product,
            'time_window': self.time_window,
            'status_filter': self.status_filter,
            'aggregation': self.aggregation,
            'unit': self.unit,
            'expected_range': list(self.expected_range) if self.expected_range else None,
            'null_handling': self.null_handling,
            'business_interpretation': self.business_interpretation,
            'risk_direction': self.risk_direction
        }


@dataclass 
class DimensionValue:
    """A single value in a dimension."""
    code: str
    label: str
    filter_expr: Optional[str] = None
    months: Optional[int] = None


class FeatureFactory:
    """
    Config-driven feature generator for credit scoring.
    
    Generates features by cartesian expansion of dimensions:
    - Product types (all, installment_loan, installment_sale, cash_facility, mortgage)
    - Time windows (all, 3m, 6m, 12m, 24m)
    - Status filters (all, active, defaulted, recovered)
    - Aggregations (sum, avg, max, min, std, count)
    
    Also generates derived features (ratios, trends, flags).
    """
    
    # Dimension definitions with readable names
    TIME_WINDOWS = [
        DimensionValue("all", "all time", None, None),
        DimensionValue("3m", "last 3 months", None, 3),
        DimensionValue("6m", "last 6 months", None, 6),
        DimensionValue("12m", "last 12 months", None, 12),
        DimensionValue("24m", "last 24 months", None, 24),
    ]
    
    PRODUCT_TYPES = [
        DimensionValue("all", "all products", None),
        DimensionValue("il", "installment loan", "product_type == 'INSTALLMENT_LOAN'"),
        DimensionValue("is", "installment sale", "product_type == 'INSTALLMENT_SALE'"),
        DimensionValue("cf", "cash facility", "product_type == 'CASH_FACILITY'"),
        DimensionValue("mg", "mortgage", "product_type == 'MORTGAGE'"),
    ]
    
    STATUS_FILTERS = [
        DimensionValue("all", "all statuses", None),
        DimensionValue("active", "active", "default_date.isna()"),
        DimensionValue("defaulted", "defaulted", "default_date.notna()"),
        DimensionValue("recovered", "recovered", "recovery_date.notna()"),
    ]
    
    AGGREGATIONS = {
        'sum': ('sum', 'total'),
        'avg': ('mean', 'average'),
        'max': ('max', 'maximum'),
        'min': ('min', 'minimum'),
        'std': ('std', 'standard deviation'),
        'cnt': ('count', 'count'),
    }
    
    # Readable name mappings for human-friendly feature names
    PRODUCT_READABLE = {
        'all': '',  # Omit for "all products"
        'il': 'installment_loan_',
        'is': 'installment_sale_',
        'cf': 'cash_facility_',
        'mg': 'mortgage_',
    }
    
    STATUS_READABLE = {
        'all': '',  # Omit for "all statuses"
        'active': 'active_',
        'defaulted': 'defaulted_',
        'recovered': 'recovered_',
    }
    
    WINDOW_READABLE = {
        'all': '',  # Omit for "all time"
        '3m': '_last_3m',
        '6m': '_last_6m',
        '12m': '_last_12m',
        '24m': '_last_24m',
    }
    
    AGG_READABLE = {
        'sum': 'total_amount',
        'avg': 'average_amount',
        'max': 'max_amount',
        'min': 'min_amount',
        'std': 'amount_std',
        'cnt': 'count',
    }
    
    # Credit product types for filtering
    CREDIT_PRODUCTS = ['INSTALLMENT_LOAN', 'INSTALLMENT_SALE', 'CASH_FACILITY', 'MORTGAGE']
    NON_CREDIT_PRODUCTS = ['NON_AUTH_OVERDRAFT', 'OVERLIMIT']
    
    def _generate_readable_name(
        self, 
        product_code: str, 
        window_code: str, 
        status_code: str, 
        agg_code: str
    ) -> str:
        """
        Generate human-readable feature name from dimension codes.
        
        Examples:
            all, all, all, sum -> total_credit_amount
            il, all, all, cnt -> installment_loan_count
            all, 3m, defaulted, sum -> defaulted_total_amount_last_3m
            mg, 12m, active, avg -> mortgage_active_average_amount_last_12m
        """
        product = self.PRODUCT_READABLE.get(product_code, f'{product_code}_')
        status = self.STATUS_READABLE.get(status_code, f'{status_code}_')
        window = self.WINDOW_READABLE.get(window_code, f'_{window_code}')
        agg = self.AGG_READABLE.get(agg_code, agg_code)
        
        # Build name: {product}{status}{aggregation}{window}
        # Special case for base features (all products, all statuses, all time)
        if product_code == 'all' and status_code == 'all' and window_code == 'all':
            # e.g., total_credit_amount, credit_count
            if agg_code == 'cnt':
                return 'total_credit_count'
            return f'total_credit_{agg.replace("total_", "")}'
        
        # Build readable name
        name_parts = []
        
        if product:
            name_parts.append(product.rstrip('_'))
        
        if status:
            name_parts.append(status.rstrip('_'))
        
        if agg_code == 'cnt':
            name_parts.append('count')
        else:
            name_parts.append(agg)
        
        name = '_'.join(name_parts)
        
        # Add time window suffix
        if window:
            name = name + window
        
        return name
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature factory.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._feature_definitions: List[FeatureDefinition] = []
        self._generated_features: Dict[str, pd.Series] = {}
        
        # Pre-compute feature names for performance (avoid string ops in loops)
        self._feature_name_cache: Dict[tuple, str] = {}
        self._build_feature_name_cache()
        
    def generate_all_features(
        self,
        applications_df: pd.DataFrame,
        credit_bureau_df: pd.DataFrame,
        reference_date_col: str = 'application_date',
        parallel: bool = False,
        n_jobs: int = 4
    ) -> pd.DataFrame:
        """
        Generate all features from applications and credit bureau data.
        
        NaN values are preserved to maintain semantic meaning:
        - days_since_last_default=NaN means "never defaulted"
        - ratio features=NaN when denominator is 0
        - avg/std features=NaN when no values exist
        
        Downstream consumers (models, imputers) should handle NaN appropriately.
        
        Args:
            applications_df: Applications DataFrame with application_id, customer_id, application_date
            credit_bureau_df: Credit bureau DataFrame with credit records
            reference_date_col: Column to use as reference date for time calculations
            parallel: If True, use multiprocessing for feature generation (local dev only)
            n_jobs: Number of parallel workers (only used if parallel=True)
            
        Returns:
            DataFrame with one row per (application_id, customer_id) and all generated features
        """
        # Expand all feature definitions first
        self._expand_feature_definitions()
        
        # Pre-convert dates in bureau data once (avoid repeated conversion per row)
        bureau_copy = credit_bureau_df.copy()
        for col in ['opening_date', 'default_date', 'recovery_date', 'closure_date']:
            if col in bureau_copy.columns:
                bureau_copy[col] = pd.to_datetime(bureau_copy[col], errors='coerce')
        
        if parallel:
            return self._generate_features_parallel(
                applications_df, bureau_copy, reference_date_col, n_jobs
            )
        else:
            return self._generate_features_vectorized(
                applications_df, bureau_copy, reference_date_col
            )
    
    def _generate_features_vectorized(
        self,
        applications_df: pd.DataFrame,
        credit_bureau_df: pd.DataFrame,
        reference_date_col: str
    ) -> pd.DataFrame:
        """
        Vectorized feature generation using groupby().apply().
        
        More efficient than row-by-row iteration as it:
        1. Merges data once upfront
        2. Uses Pandas groupby aggregations
        3. Avoids Python-level loops where possible
        """
        # Create application lookup for quick access
        app_lookup = applications_df.set_index(['application_id', 'customer_id']).to_dict('index')
        
        # Group bureau by application/customer and generate features
        results = []
        
        # Get unique app/customer pairs from applications
        for _, app_row in applications_df.iterrows():
            app_id = app_row['application_id']
            cust_id = app_row['customer_id']
            ref_date = pd.to_datetime(app_row[reference_date_col])
            
            # Filter bureau data for this customer (already has pre-converted dates)
            customer_data = credit_bureau_df[
                (credit_bureau_df['application_id'] == app_id) &
                (credit_bureau_df['customer_id'] == cust_id)
            ]
            
            # Generate all features for this customer
            feature_row = self._generate_customer_features(
                app_row, customer_data, ref_date
            )
            results.append(feature_row)
        
        return pd.DataFrame(results)
    
    def _generate_features_parallel(
        self,
        applications_df: pd.DataFrame,
        credit_bureau_df: pd.DataFrame,
        reference_date_col: str,
        n_jobs: int
    ) -> pd.DataFrame:
        """
        Parallel feature generation using multiprocessing.
        
        Best for local development with large datasets.
        Note: Not recommended for Spark/Dataproc where data parallelism is handled differently.
        """
        from multiprocessing import Pool, cpu_count
        from functools import partial
        
        # Prepare tasks: list of (app_row_dict, bureau_subset)
        tasks = []
        for _, app_row in applications_df.iterrows():
            app_id = app_row['application_id']
            cust_id = app_row['customer_id']
            
            customer_data = credit_bureau_df[
                (credit_bureau_df['application_id'] == app_id) &
                (credit_bureau_df['customer_id'] == cust_id)
            ].copy()
            
            tasks.append((app_row.to_dict(), customer_data, reference_date_col))
        
        # Use min of requested workers and CPU count
        # Handle n_jobs=-1 like scikit-learn (use all CPU cores)
        max_cores = cpu_count()
        if n_jobs < 0:
            actual_workers = max(1, max_cores + 1 + n_jobs)  # -1 means all cores
        else:
            actual_workers = min(n_jobs, max_cores, len(tasks))
        
        if actual_workers <= 1 or len(tasks) < 10:
            # Too few tasks, just use sequential
            return self._generate_features_vectorized(
                applications_df, credit_bureau_df, reference_date_col
            )
        
        # Process in parallel
        with Pool(actual_workers) as pool:
            results = pool.starmap(
                self._process_single_application,
                tasks
            )
        
        return pd.DataFrame(results)
    
    def _process_single_application(
        self,
        app_row_dict: dict,
        customer_data: pd.DataFrame,
        reference_date_col: str
    ) -> dict:
        """Process a single application for parallel execution."""
        app_row = pd.Series(app_row_dict)
        ref_date = pd.to_datetime(app_row[reference_date_col])
        return self._generate_customer_features(app_row, customer_data, ref_date)
    
    def _build_feature_name_cache(self) -> None:
        """Pre-compute all feature names to avoid string operations in loops."""
        for product in self.PRODUCT_TYPES:
            for window in self.TIME_WINDOWS:
                for status in self.STATUS_FILTERS:
                    # Count name
                    key_cnt = (product.code, window.code, status.code, 'cnt')
                    self._feature_name_cache[key_cnt] = self._generate_readable_name(
                        product.code, window.code, status.code, 'cnt'
                    )
                    # Amount aggregation names
                    for agg_code in ['sum', 'avg', 'max', 'min', 'std']:
                        key_agg = (product.code, window.code, status.code, agg_code)
                        self._feature_name_cache[key_agg] = self._generate_readable_name(
                            product.code, window.code, status.code, agg_code
                        )
    
    def _get_cached_name(self, product_code: str, window_code: str, status_code: str, agg_code: str) -> str:
        """Get pre-computed feature name from cache."""
        key = (product_code, window_code, status_code, agg_code)
        return self._feature_name_cache.get(key, self._generate_readable_name(
            product_code, window_code, status_code, agg_code
        ))
    
    def _expand_feature_definitions(self) -> None:
        """Expand templates into concrete feature definitions."""
        self._feature_definitions = []
        
        # 1. Amount features (product × window × status × aggregation)
        self._expand_amount_features()
        
        # 2. Count features (product × window × status)
        self._expand_count_features()
        
        # 3. Ratio features
        self._expand_ratio_features()
        
        # 4. Temporal features
        self._expand_temporal_features()
        
        # 5. Trend features
        self._expand_trend_features()
        
        # 6. Risk signal features
        self._expand_risk_signal_features()
        
        # 7. Diversity features
        self._expand_diversity_features()
        
        # 8. Behavioral features
        self._expand_behavioral_features()
        
        # 9. Default pattern features
        self._expand_default_pattern_features()
        
        # 10. Credit sequence features
        self._expand_sequence_features()
        
        # 11. Size pattern features
        self._expand_size_pattern_features()
        
        # 12. Burst detection features
        self._expand_burst_features()
        
        # 13. Inter-credit interval features
        self._expand_interval_features()
        
        # 14. Seasonal/cyclical features (NEW)
        self._expand_seasonal_features()
        
        # 15. Weighted average features (NEW)
        self._expand_weighted_features()
        
        # 16. Co-applicant features (NEW)
        self._expand_co_applicant_features()
        
        # 17. Relative/percentile features (NEW)
        self._expand_relative_features()
        
        # 18. Complexity features (NEW)
        self._expand_complexity_features()
        
        # 19. Time-decay features (NEW)
        self._expand_time_decay_features()
        
        # 20. Default lifecycle features (EXPERT)
        self._expand_default_lifecycle_features()
        
        # 21. Early default detection features (EXPERT)
        self._expand_early_default_features()
        
        # 22. Amount tier behavior features (EXPERT)
        self._expand_amount_tier_features()
        
        # 23. Cross-product correlation features (EXPERT)
        self._expand_cross_product_features()
        
        # 24. Credit freshness at application features (EXPERT)
        self._expand_freshness_features()
        
        # 25. Risk concentration features (EXPERT)
        self._expand_concentration_features()
        
        # 26. Behavioral anomaly features (EXPERT)
        self._expand_anomaly_features()
        
        # === PAYMENT & DURATION FEATURES (using new fields) ===
        
        # 27. Payment features
        self._expand_payment_features()
        
        # 28. Remaining term features
        self._expand_remaining_term_features()
        
        # 29. Obligation burden features
        self._expand_obligation_features()
        
        # 30. DTI proxy features
        self._expand_dti_proxy_features()
        
        # 31. Maturity features (with 3-month closure rule)
        self._expand_maturity_features()
        
        # 32. Duration risk features
        self._expand_duration_risk_features()
        
        # 33. Payment behavior features
        self._expand_payment_behavior_features()
        
        # 34. Term structure features
        self._expand_term_structure_features()
    
    def _expand_amount_features(self) -> None:
        """Generate amount-based feature definitions with readable names."""
        # Skip these readable names (duplicates of better-named versions elsewhere)
        SKIP_READABLE = {
            'total_credit_max_amount',  # == max_single_credit_amount
            'total_credit_min_amount',  # == min_single_credit_amount
            'defaulted_average_amount', # == avg_default_severity
            'defaulted_max_amount',     # == max_default_severity
        }
        
        for product in self.PRODUCT_TYPES:
            for window in self.TIME_WINDOWS:
                for status in self.STATUS_FILTERS:
                    for agg_code, (agg_func, agg_label) in self.AGGREGATIONS.items():
                        if agg_code == 'cnt':
                            continue  # Count handled separately
                        
                        # Generate readable name
                        readable_name = self._generate_readable_name(
                            product.code, window.code, status.code, agg_code
                        )
                        
                        # Keep old name for internal mapping
                        old_name = f"{product.code}_{window.code}_{status.code}_{agg_code}_amt"
                        
                        # Skip duplicates
                        if readable_name in SKIP_READABLE:
                            continue
                        
                        desc = f"{product.label} {window.label} {status.label} {agg_label} amount"
                        
                        formula = self._build_formula(
                            f"{agg_func}(total_amount)",
                            product, window, status
                        )
                        
                        interpretation = self._get_amount_interpretation(
                            product, window, status, agg_code
                        )
                        
                        self._feature_definitions.append(FeatureDefinition(
                            name=readable_name,
                            description=desc,
                            formula=formula,
                            category="amount_feature",
                            product=product.code,
                            time_window=window.code,
                            status_filter=status.code,
                            aggregation=agg_code,
                            unit="currency",
                            expected_range=(0, None),
                            business_interpretation=interpretation,
                            risk_direction="higher_is_riskier" if agg_code in ['sum', 'max'] else "neutral"
                        ))
    
    def _expand_count_features(self) -> None:
        """Generate count-based feature definitions with readable names."""
        # Skip these readable names (duplicates of better-named versions elsewhere)
        SKIP_READABLE = {
            'defaulted_count',   # == default_count_ever
            'recovered_count',   # == recovery_cycle_count
        }
        
        for product in self.PRODUCT_TYPES:
            for window in self.TIME_WINDOWS:
                for status in self.STATUS_FILTERS:
                    # Generate readable name
                    readable_name = self._generate_readable_name(
                        product.code, window.code, status.code, 'cnt'
                    )
                    
                    # Skip duplicates
                    if readable_name in SKIP_READABLE:
                        continue
                    
                    desc = f"{product.label} {window.label} {status.label} count"
                    
                    formula = self._build_formula("count(*)", product, window, status)
                    
                    interpretation = f"Number of {status.label} {product.label} credits in {window.label}"
                    
                    self._feature_definitions.append(FeatureDefinition(
                        name=readable_name,
                        description=desc,
                        formula=formula,
                        category="count_feature",
                        product=product.code,
                        time_window=window.code,
                        status_filter=status.code,
                        aggregation="count",
                        unit="count",
                        expected_range=(0, None),
                        business_interpretation=interpretation,
                        risk_direction="context_dependent"
                    ))
    
    def _expand_ratio_features(self) -> None:
        """Generate ratio-based feature definitions."""
        # Product mix ratios
        for product in self.PRODUCT_TYPES[1:]:  # Skip 'all'
            # Count ratio
            name = f"{product.code}_to_total_ratio"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"{product.label} count as proportion of total credits",
                formula=f"{product.code}_all_all_cnt / all_all_all_cnt",
                category="ratio_feature",
                product=product.code,
                unit="ratio",
                expected_range=(0, 1),
                business_interpretation=f"Concentration of portfolio in {product.label}",
                risk_direction="neutral"
            ))
            
            # Amount ratio
            name = f"{product.code}_amt_to_total_ratio"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"{product.label} amount as proportion of total credit amount",
                formula=f"{product.code}_all_all_sum_amt / all_all_all_sum_amt",
                category="ratio_feature",
                product=product.code,
                unit="ratio",
                expected_range=(0, 1),
                business_interpretation=f"Amount concentration in {product.label}",
                risk_direction="neutral"
            ))
        
        # Status ratios
        for status in self.STATUS_FILTERS[1:]:  # Skip 'all'
            name = f"{status.code}_ratio"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"Proportion of credits that are {status.label}",
                formula=f"all_all_{status.code}_cnt / all_all_all_cnt",
                category="ratio_feature",
                status_filter=status.code,
                unit="ratio",
                expected_range=(0, 1),
                business_interpretation=f"Proportion of {status.label} credits in portfolio",
                risk_direction="higher_is_riskier" if status.code == "defaulted" else "lower_is_riskier"
            ))
        
        # Time window concentration ratios
        for window in self.TIME_WINDOWS[1:]:  # Skip 'all'
            name = f"{window.code}_concentration_ratio"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"Proportion of credits opened in {window.label}",
                formula=f"all_{window.code}_all_cnt / all_all_all_cnt",
                category="ratio_feature",
                time_window=window.code,
                unit="ratio",
                expected_range=(0, 1),
                business_interpretation=f"Recent credit activity concentration in {window.label}",
                risk_direction="context_dependent"
            ))
    
    def _expand_temporal_features(self) -> None:
        """Generate temporal feature definitions."""
        temporal_features = [
            ("oldest_credit_age_months", "Age of oldest credit in months", "max"),
            ("newest_credit_age_months", "Age of newest credit in months", "min"),
            ("avg_credit_age_months", "Average credit age in months", "mean"),
            ("credit_history_length_months", "Length of credit history in months", "max"),
            ("days_since_last_default", "Days since most recent default", "min"),
            ("days_since_last_credit", "Days since most recent credit opened", "min"),
            ("avg_time_to_default_days", "Average days from opening to default", "mean"),
            ("avg_recovery_time_days", "Average days from default to recovery", "mean"),
        ]
        
        for name, desc, agg in temporal_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"{agg}(date_diff)",
                category="temporal_feature",
                aggregation=agg,
                unit="months" if "months" in name else "days",
                expected_range=(0, None),
                business_interpretation=f"Measures {desc.lower()}",
                risk_direction="context_dependent"
            ))
    
    def _expand_trend_features(self) -> None:
        """Generate trend feature definitions."""
        # Velocity features
        for window in ['3m', '6m']:
            name = f"credit_velocity_{window}"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"Rate of credit acquisition in {window}",
                formula=f"all_{window}_all_cnt / all_all_all_cnt",
                category="trend_feature",
                time_window=window,
                unit="ratio",
                expected_range=(0, 1),
                business_interpretation=f"Measures acceleration of credit taking in {window}",
                risk_direction="higher_is_riskier"
            ))
            
            name = f"amount_velocity_{window}"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"Rate of credit amount acquisition in {window}",
                formula=f"all_{window}_all_sum_amt / all_all_all_sum_amt",
                category="trend_feature",
                time_window=window,
                unit="ratio",
                expected_range=(0, 1),
                business_interpretation=f"Measures acceleration of credit amount in {window}",
                risk_direction="higher_is_riskier"
            ))
        
        # Trend comparisons
        trends = [
            ("cnt_trend_6m_vs_12m", "Credit count trend (6m vs prior 6m)", 
             "all_6m_all_cnt - (all_12m_all_cnt - all_6m_all_cnt)"),
            ("amt_trend_6m_vs_12m", "Amount trend (6m vs prior 6m)",
             "all_6m_all_sum_amt - (all_12m_all_sum_amt - all_6m_all_sum_amt)"),
            ("default_trend_6m_vs_12m", "Default trend",
             "all_6m_defaulted_cnt - (all_12m_defaulted_cnt - all_6m_defaulted_cnt)"),
        ]
        
        for name, desc, formula in trends:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=formula,
                category="trend_feature",
                unit="change",
                business_interpretation=f"Positive values indicate increasing {name.split('_')[0]}",
                risk_direction="higher_is_riskier" if "default" in name else "context_dependent"
            ))
    
    def _expand_risk_signal_features(self) -> None:
        """Generate risk signal feature definitions."""
        signals = [
            ("has_overdraft", "Has non-authorized overdraft", "NON_AUTH_OVERDRAFT", True),
            ("has_overlimit", "Has overlimit event", "OVERLIMIT", True),
            ("financial_stress_flag", "Any overdraft or overlimit", None, True),
            ("overdraft_count", "Number of overdraft events", "NON_AUTH_OVERDRAFT", False),
            ("overlimit_count", "Number of overlimit events", "OVERLIMIT", False),
            ("overdraft_amount", "Total overdraft amount", "NON_AUTH_OVERDRAFT", False),
            ("overlimit_amount", "Total overlimit amount", "OVERLIMIT", False),
            ("has_current_default", "Has currently defaulted credit", None, True),
            ("ever_defaulted", "Has ever defaulted on any credit", None, True),
        ]
        
        for name, desc, product_filter, is_flag in signals:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"1 if {product_filter or 'condition'} else 0",
                category="risk_signal_feature",
                data_type="int64" if is_flag else "float64",
                unit="flag" if is_flag else "numeric",
                expected_range=(0, 1) if is_flag else (0, None),
                business_interpretation=f"Risk indicator: {desc.lower()}",
                risk_direction="higher_is_riskier"
            ))
    
    def _expand_diversity_features(self) -> None:
        """Generate diversity and concentration feature definitions."""
        diversity_features = [
            ("distinct_product_count", "Number of different product types used", "count_distinct(product_type)"),
            ("product_diversity_ratio", "Product diversity (0-1)", "distinct_product_count / 4"),
            ("hhi_product", "Herfindahl-Hirschman Index for product concentration", "sum(product_share^2)"),
            ("secured_ratio", "Proportion of secured credits (mortgage)", "mg_all_all_cnt / all_all_all_cnt"),
            ("unsecured_ratio", "Proportion of unsecured credits", "(il + is + cf) / all"),
            ("revolving_ratio", "Proportion of revolving credits (cash facility)", "cf_all_all_cnt / all_all_all_cnt"),
        ]
        
        for name, desc, formula in diversity_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=formula,
                category="diversity_feature",
                unit="ratio" if "ratio" in name else "numeric",
                expected_range=(0, 1) if "ratio" in name else (0, None),
                business_interpretation=f"Portfolio diversification: {desc.lower()}",
                risk_direction="context_dependent"
            ))
    
    def _expand_behavioral_features(self) -> None:
        """Generate behavioral pattern feature definitions."""
        behavioral_features = [
            # Time-to-default patterns by product
            ("min_time_to_default_days", "Minimum days from opening to default", "min"),
            ("max_time_to_default_days", "Maximum days from opening to default", "max"),
            ("std_time_to_default_days", "Variability in time to default", "std"),
            # Recovery patterns
            ("min_recovery_time_days", "Minimum recovery time", "min"),
            ("max_recovery_time_days", "Maximum recovery time", "max"),
            ("recovery_success_rate", "Proportion of defaults that recovered", "ratio"),
            # Credit maturity
            ("mature_credit_ratio", "Proportion of credits older than 24 months", "ratio"),
            ("new_credit_ratio", "Proportion of credits newer than 6 months", "ratio"),
        ]
        
        for name, desc, agg_or_type in behavioral_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"{agg_or_type}(behavioral_metric)",
                category="behavioral_feature",
                unit="days" if "days" in name else "ratio",
                expected_range=(0, 1) if "ratio" in name else (0, None),
                business_interpretation=f"Credit behavior: {desc.lower()}",
                risk_direction="higher_is_riskier" if "default" in name.lower() else "context_dependent"
            ))
        
        # Product-specific time-to-default
        for product in self.PRODUCT_TYPES[1:]:  # Skip 'all'
            for agg in ['min', 'max', 'avg']:
                name = f"{product.code}_time_to_default_{agg}_days"
                self._feature_definitions.append(FeatureDefinition(
                    name=name,
                    description=f"{product.label} {agg} time to default in days",
                    formula=f"{agg}(default_date - opening_date) WHERE product={product.code}",
                    category="behavioral_feature",
                    product=product.code,
                    unit="days",
                    expected_range=(0, None),
                    business_interpretation=f"How quickly {product.label} credits default",
                    risk_direction="lower_is_riskier"
                ))
    
    def _expand_default_pattern_features(self) -> None:
        """Generate default history and recurrence pattern features."""
        default_patterns = [
            ("default_count_ever", "Total number of defaults in history", "count"),
            ("default_recurrence_count", "Number of times customer has defaulted on multiple products", "count"),
            ("has_multiple_defaults", "Has defaulted on more than one credit", "flag"),
            ("has_recovered_default", "Has any recovered default", "flag"),
            ("all_defaults_recovered", "All past defaults have recovered", "flag"),
            ("default_amount_ratio", "Defaulted amount as proportion of total", "ratio"),
            ("avg_default_severity", "Average default amount", "currency"),
            ("max_default_severity", "Maximum single default amount", "currency"),
            ("default_to_active_ratio", "Ratio of defaulted to active credits", "ratio"),
        ]
        
        for name, desc, unit in default_patterns:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="default_pattern_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Default pattern: {desc.lower()}",
                risk_direction="higher_is_riskier"
            ))
        
        # Default patterns by product type
        for product in self.PRODUCT_TYPES[1:]:
            name = f"{product.code}_default_rate"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"Default rate for {product.label}",
                formula=f"{product.code}_defaulted_cnt / {product.code}_all_cnt",
                category="default_pattern_feature",
                product=product.code,
                unit="ratio",
                expected_range=(0, 1),
                business_interpretation=f"How often {product.label} credits default",
                risk_direction="higher_is_riskier"
            ))
    
    def _expand_sequence_features(self) -> None:
        """Generate credit sequence and transition pattern features."""
        sequence_features = [
            ("first_product_installment_loan", "First credit was installment loan", "flag"),
            ("first_product_installment_sale", "First credit was installment sale", "flag"),
            ("first_product_cash_facility", "First credit was cash facility", "flag"),
            ("first_product_mortgage", "First credit was mortgage", "flag"),
            ("last_product_installment_loan", "Most recent credit was installment loan", "flag"),
            ("last_product_installment_sale", "Most recent credit was installment sale", "flag"),
            ("last_product_cash_facility", "Most recent credit was cash facility", "flag"),
            ("last_product_mortgage", "Most recent credit was mortgage", "flag"),
            ("product_transition_count", "Number of product type transitions", "count"),
            ("moved_to_secured", "Progressed from unsecured to secured credit", "flag"),
            ("moved_to_unsecured", "Regressed from secured to unsecured only", "flag"),
            ("same_product_streak", "Consecutive same product credits", "count"),
        ]
        
        for name, desc, unit in sequence_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="sequence_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit == "flag" else (0, None),
                business_interpretation=f"Credit sequence: {desc.lower()}",
                risk_direction="context_dependent"
            ))
    
    def _expand_size_pattern_features(self) -> None:
        """Generate credit size pattern features."""
        size_patterns = [
            ("has_small_credit", "Has any credit under 5000", "flag"),
            ("has_large_credit", "Has any credit over 50000", "flag"),  
            ("has_very_large_credit", "Has any credit over 100000", "flag"),
            ("small_credit_count", "Number of credits under 5000", "count"),
            ("large_credit_count", "Number of credits over 50000", "count"),
            ("small_credit_ratio", "Proportion of small credits", "ratio"),
            ("large_credit_ratio", "Proportion of large credits", "ratio"),
            ("max_to_avg_amount_ratio", "Max credit amount / average", "ratio"),
            ("max_to_min_amount_ratio", "Max credit amount / min", "ratio"),
            ("amount_coefficient_of_variation", "Credit amount CV (std/mean)", "ratio"),
            ("amount_range", "Difference between max and min amounts", "currency"),
            ("median_credit_amount", "Median credit amount", "currency"),
            ("amount_skewness", "Skewness of credit amounts", "numeric"),
        ]
        
        for name, desc, unit in size_patterns:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="size_pattern_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Credit size pattern: {desc.lower()}",
                risk_direction="context_dependent"
            ))
        
        # Size patterns by product
        for product in self.PRODUCT_TYPES[1:]:
            name = f"{product.code}_avg_amount"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"Average {product.label} amount",
                formula=f"avg(total_amount) WHERE product={product.code}",
                category="size_pattern_feature",
                product=product.code,
                unit="currency",
                expected_range=(0, None),
                business_interpretation=f"Typical {product.label} credit size",
                risk_direction="context_dependent"
            ))
    
    def _expand_burst_features(self) -> None:
        """Generate credit burst detection features (rapid acquisition)."""
        burst_features = [
            ("credits_in_30_days", "Credits opened within 30 days of each other", "count"),
            ("credits_in_60_days", "Credits opened within 60 days of each other", "count"),
            ("credits_in_90_days", "Credits opened within 90 days of each other", "count"),
            ("has_credit_burst_30d", "2+ credits in 30 day window", "flag"),
            ("has_credit_burst_60d", "3+ credits in 60 day window", "flag"),
            ("max_credits_same_month", "Maximum credits opened in same month", "count"),
            ("amount_in_30_days", "Total amount in 30 day burst", "currency"),
            ("amount_in_60_days", "Total amount in 60 day burst", "currency"),
            ("burst_intensity_30d", "Credits per 30 days (max)", "ratio"),
            ("burst_intensity_60d", "Credits per 60 days (max)", "ratio"),
        ]
        
        for name, desc, unit in burst_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="burst_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit == "flag" else (0, None),
                business_interpretation=f"Credit burst: {desc.lower()}",
                risk_direction="higher_is_riskier"
            ))
        
        # Burst by product type
        for product in self.PRODUCT_TYPES[1:]:
            name = f"{product.code}_burst_count"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"Burst count for {product.label}",
                formula=f"count(*) WHERE product={product.code} AND burst",
                category="burst_feature",
                product=product.code,
                unit="count",
                expected_range=(0, None),
                business_interpretation=f"Rapid {product.label} acquisition",
                risk_direction="higher_is_riskier"
            ))
    
    def _expand_interval_features(self) -> None:
        """Generate inter-credit interval features."""
        interval_features = [
            ("avg_inter_credit_days", "Average days between credit openings", "days"),
            ("min_inter_credit_days", "Minimum days between credit openings", "days"),
            ("max_inter_credit_days", "Maximum days between credit openings", "days"),
            ("std_inter_credit_days", "Variability in credit opening intervals", "days"),
            ("median_inter_credit_days", "Median days between credit openings", "days"),
            ("inter_credit_cv", "Coefficient of variation of intervals", "ratio"),
            ("has_rapid_succession", "Any credits within 14 days of each other", "flag"),
            ("rapid_succession_count", "Count of credits within 14 days of previous", "count"),
            ("longest_credit_gap_days", "Longest gap between credits", "days"),
            ("recent_vs_historical_interval", "Recent interval / historical avg", "ratio"),
            # Acceleration/deceleration
            ("interval_trend", "Trend in credit intervals (accelerating < 0)", "numeric"),
            ("is_accelerating", "Credit taking is speeding up", "flag"),
            ("is_decelerating", "Credit taking is slowing down", "flag"),
        ]
        
        for name, desc, unit in interval_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="interval_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else None,
                business_interpretation=f"Credit interval: {desc.lower()}",
                risk_direction="lower_is_riskier" if "rapid" in name or "accelerating" in name else "context_dependent"
            ))
        
        # Interval by product type
        for product in self.PRODUCT_TYPES[1:]:
            name = f"{product.code}_avg_interval_days"
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=f"Average days between {product.label} credits",
                formula=f"avg(interval) WHERE product={product.code}",
                category="interval_feature",
                product=product.code,
                unit="days",
                expected_range=(0, None),
                business_interpretation=f"Typical gap between {product.label} credits",
                risk_direction="context_dependent"
            ))
    
    def _expand_seasonal_features(self) -> None:
        """Generate seasonal and cyclical feature definitions."""
        seasonal_features = [
            ("most_common_opening_month", "Most frequent credit opening month", "month"),
            ("most_common_opening_quarter", "Most frequent credit opening quarter", "quarter"),
            ("credits_in_q1", "Credits opened in Q1 (Jan-Mar)", "count"),
            ("credits_in_q2", "Credits opened in Q2 (Apr-Jun)", "count"),
            ("credits_in_q3", "Credits opened in Q3 (Jul-Sep)", "count"),
            ("credits_in_q4", "Credits opened in Q4 (Oct-Dec)", "count"),
            ("credits_in_december", "Credits opened in December", "count"),
            ("is_year_end_heavy", "More than 25% of credits in Q4", "flag"),
            ("is_summer_heavy", "More than 25% of credits in Q2-Q3", "flag"),
            ("monthly_spread", "How spread out credits are across months (0-1)", "ratio"),
            ("quarterly_concentration", "HHI of quarterly distribution", "ratio"),
        ]
        
        for name, desc, unit in seasonal_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="seasonal_feature",
                data_type="int64" if unit in ["count", "flag", "month", "quarter"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Seasonal pattern: {desc.lower()}",
                risk_direction="context_dependent"
            ))
    
    def _expand_weighted_features(self) -> None:
        """Generate amount-weighted feature definitions."""
        weighted_features = [
            ("amount_weighted_default_rate", "Default rate weighted by credit amount", "ratio"),
            ("amount_weighted_age_months", "Credit age weighted by amount", "months"),
            ("amount_weighted_product_score", "Product risk score weighted by amount", "numeric"),
            ("recent_amount_weight", "Last 6 months amount / total amount", "ratio"),
            ("large_credit_weight", "Amount in credits >50k / total", "ratio"),
            ("defaulted_amount_weighted_age", "Age of defaulted credits weighted by amount", "months"),
        ]
        
        for name, desc, unit in weighted_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="weighted_feature",
                unit=unit,
                expected_range=(0, 1) if unit == "ratio" else (0, None),
                business_interpretation=f"Weighted metric: {desc.lower()}",
                risk_direction="higher_is_riskier" if "default" in name else "context_dependent"
            ))
    
    def _expand_co_applicant_features(self) -> None:
        """Generate co-applicant relationship feature definitions."""
        co_applicant_features = [
            ("is_primary_applicant", "Is the primary applicant", "flag"),
            ("is_co_applicant", "Is a co-applicant", "flag"),
            ("application_has_co_applicant", "Application has a co-applicant", "flag"),
            ("co_applicant_credit_count", "Number of credits for co-applicant", "count"),
            ("co_applicant_default_count", "Number of defaults for co-applicant", "count"),
            ("co_applicant_default_rate", "Default rate of co-applicant", "ratio"),
            ("co_applicant_total_amount", "Total credit amount of co-applicant", "currency"),
            ("primary_vs_co_amount_ratio", "Primary applicant amount / co-applicant", "ratio"),
            ("combined_default_exposure", "Combined defaulted amount", "currency"),
            ("either_has_default", "Either primary or co-applicant has default", "flag"),
            ("both_have_default", "Both primary and co-applicant have defaults", "flag"),
        ]
        
        for name, desc, unit in co_applicant_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="co_applicant_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Co-applicant: {desc.lower()}",
                risk_direction="higher_is_riskier" if "default" in name else "context_dependent"
            ))
    
    def _expand_relative_features(self) -> None:
        """Generate relative and percentile-based feature definitions."""
        relative_features = [
            ("latest_amount_vs_average", "Latest credit amount / customer average", "ratio"),
            ("latest_amount_vs_max", "Latest credit amount / customer max", "ratio"),
            ("is_largest_credit_recent", "Largest credit was in last 6 months", "flag"),
            ("is_smallest_credit_recent", "Smallest credit was in last 6 months", "flag"),
            ("amount_growth_rate", "Trend in credit amounts over time", "numeric"),
            ("credit_count_growth_rate", "Trend in credit count over windows", "numeric"),
            ("default_rate_trend", "Trend in default rate over time", "numeric"),
            ("amount_percentile_latest", "Percentile rank of latest credit amount", "ratio"),
            ("recency_of_max_amount", "Months since largest credit", "months"),
            ("latest_vs_first_amount_ratio", "Latest credit / first credit amount", "ratio"),
        ]
        
        for name, desc, unit in relative_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="relative_feature",
                data_type="int64" if unit == "flag" else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else None,
                business_interpretation=f"Relative position: {desc.lower()}",
                risk_direction="context_dependent"
            ))
    
    def _expand_complexity_features(self) -> None:
        """Generate credit portfolio complexity feature definitions."""
        complexity_features = [
            ("product_entropy", "Shannon entropy of product distribution", "numeric"),
            ("amount_entropy", "Entropy of amount distribution (binned)", "numeric"),
            ("has_all_product_types", "Has used all 4 product types", "flag"),
            ("single_product_customer", "Uses only one product type", "flag"),
            ("dual_product_customer", "Uses exactly two product types", "flag"),
            ("product_status_combinations", "Unique product × status combinations", "count"),
            ("portfolio_complexity_score", "Combined complexity metric (0-10)", "numeric"),
            ("credit_frequency_regularity", "How regular credit timing is (CV inverse)", "ratio"),
            ("amount_regularity", "How regular credit amounts are (CV inverse)", "ratio"),
        ]
        
        for name, desc, unit in complexity_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="complexity_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Portfolio complexity: {desc.lower()}",
                risk_direction="context_dependent"
            ))
    
    def _expand_time_decay_features(self) -> None:
        """Generate time-decay weighted feature definitions."""
        decay_features = [
            ("recency_weighted_default_count", "Default count with recent defaults weighted higher", "numeric"),
            ("recency_weighted_amount", "Total amount with recent credits weighted higher", "currency"),
            ("exponential_decay_default_score", "Exponentially decayed default risk score", "numeric"),
            ("half_life_30d_amount", "Amount sum with 30-day half-life decay", "currency"),
            ("half_life_90d_amount", "Amount sum with 90-day half-life decay", "currency"),
            ("recency_score", "Overall recency score (0-1)", "ratio"),
            ("stale_portfolio_flag", "No new credits in 12+ months", "flag"),
            ("fresh_portfolio_flag", "Most recent credit < 3 months ago", "flag"),
        ]
        
        for name, desc, unit in decay_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="time_decay_feature",
                data_type="int64" if unit == "flag" else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Time-weighted: {desc.lower()}",
                risk_direction="higher_is_riskier" if "default" in name else "context_dependent"
            ))
    
    def _expand_default_lifecycle_features(self) -> None:
        """Generate default lifecycle and recovery pattern features."""
        lifecycle_features = [
            ("credits_after_first_recovery", "Credits opened after first recovery", "count"),
            ("defaulted_again_after_recovery", "Defaulted again after a previous recovery", "flag"),
            ("days_clean_before_first_default", "Days of clean history before first default", "days"),
            ("time_since_last_recovery_days", "Days since most recent recovery", "days"),
            ("recovery_to_new_credit_days", "Days from recovery to next new credit", "days"),
            ("multiple_recovery_cycles", "Has gone through default-recovery cycle more than once", "flag"),
            ("recovery_cycle_count", "Number of default-recovery cycles", "count"),
            ("clean_after_recovery", "No defaults after recovery", "flag"),
            ("first_default_age_months", "Age of first default in months", "months"),
            ("last_default_age_months", "Age of most recent default in months", "months"),
        ]
        
        for name, desc, unit in lifecycle_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="default_lifecycle_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit == "flag" else (0, None),
                business_interpretation=f"Default lifecycle: {desc.lower()}",
                risk_direction="higher_is_riskier" if "default" in name.lower() or "again" in name else "context_dependent"
            ))
    
    def _expand_early_default_features(self) -> None:
        """Generate early default detection features."""
        early_default_features = [
            ("has_early_default_90d", "Has any default within 90 days of opening", "flag"),
            ("has_early_default_180d", "Has any default within 180 days of opening", "flag"),
            ("has_early_default_365d", "Has any default within 1 year of opening", "flag"),
            ("early_default_count_90d", "Count of defaults within 90 days", "count"),
            ("early_default_count_180d", "Count of defaults within 180 days", "count"),
            ("early_default_ratio", "Proportion of defaults that were early (<180d)", "ratio"),
            ("fastest_default_days", "Minimum days from opening to default", "days"),
            ("slowest_default_days", "Maximum days from opening to default", "days"),
            ("early_default_amount", "Total amount of early defaults", "currency"),
            ("early_default_amount_ratio", "Early default amount / total default amount", "ratio"),
        ]
        
        for name, desc, unit in early_default_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="early_default_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Early default: {desc.lower()}",
                risk_direction="higher_is_riskier"
            ))
    
    def _expand_amount_tier_features(self) -> None:
        """Generate amount tier behavior features."""
        tier_features = [
            ("micro_credit_count", "Count of credits < 1000", "count"),
            ("micro_credit_default_rate", "Default rate for credits < 1000", "ratio"),
            ("small_tier_count", "Count of credits 1K-10K", "count"),
            ("small_tier_default_rate", "Default rate for credits 1K-10K", "ratio"),
            ("medium_tier_count", "Count of credits 10K-50K", "count"),
            ("medium_tier_default_rate", "Default rate for credits 10K-50K", "ratio"),
            ("large_tier_count", "Count of credits 50K+", "count"),
            ("large_tier_default_rate", "Default rate for credits 50K+", "ratio"),
            ("tier_with_most_defaults", "Which tier has most defaults (0-3)", "category"),
            ("tier_with_highest_default_rate", "Which tier has highest default rate (0-3)", "category"),
            ("tier_spread", "Number of tiers with at least one credit", "count"),
            ("dominant_tier", "Tier with most credit count (0-3)", "category"),
        ]
        
        for name, desc, unit in tier_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="amount_tier_feature",
                data_type="int64" if unit in ["count", "flag", "category"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit == "ratio" else (0, None),
                business_interpretation=f"Amount tier: {desc.lower()}",
                risk_direction="higher_is_riskier" if "default" in name else "context_dependent"
            ))
    
    def _expand_cross_product_features(self) -> None:
        """Generate cross-product correlation features."""
        cross_product_features = [
            ("cf_default_then_il_default", "Cash facility default followed by installment loan default", "flag"),
            ("il_default_then_cf_default", "Installment loan default followed by cash facility default", "flag"),
            ("multi_product_default", "Has defaults in multiple product types", "flag"),
            ("multi_product_default_count", "Number of product types with defaults", "count"),
            ("mortgage_with_default_elsewhere", "Has mortgage but default in other products", "flag"),
            ("all_products_clean", "All product types have no defaults", "flag"),
            ("unsecured_default_rate", "Default rate in unsecured products", "ratio"),
            ("secured_default_rate", "Default rate in secured products (mortgage)", "ratio"),
            ("high_risk_product_only", "Only uses high-risk products (CF, IS)", "flag"),
            ("low_risk_product_only", "Only uses low-risk products (mortgage)", "flag"),
            ("cross_product_default_spread", "Default spread across products (0-1)", "ratio"),
        ]
        
        for name, desc, unit in cross_product_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="cross_product_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Cross-product: {desc.lower()}",
                risk_direction="higher_is_riskier" if "default" in name else "context_dependent"
            ))
    
    def _expand_freshness_features(self) -> None:
        """Generate credit freshness at application time features."""
        freshness_features = [
            ("newest_credit_age_at_app_days", "Age of newest credit at application time", "days"),
            ("applied_with_fresh_credit_30d", "Has credit opened within 30 days of application", "flag"),
            ("applied_with_fresh_credit_60d", "Has credit opened within 60 days of application", "flag"),
            ("credits_near_application_before", "Credits opened 30 days before application", "count"),
            ("credits_near_application_after", "Credits opened 30 days after application", "count"),
            ("amount_near_application_before", "Amount opened 30 days before application", "currency"),
            ("credit_surge_at_application", "More than 2 credits near application time", "flag"),
            ("application_timing_score", "How 'normal' is the application timing (0-1)", "ratio"),
        ]
        
        for name, desc, unit in freshness_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="freshness_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Application freshness: {desc.lower()}",
                risk_direction="higher_is_riskier" if "surge" in name or "fresh" in name else "context_dependent"
            ))
    
    def _expand_concentration_features(self) -> None:
        """Generate risk concentration features."""
        concentration_features = [
            ("single_large_exposure_flag", "Single credit is >50% of total exposure", "flag"),
            ("top_credit_concentration", "Largest credit / total amount", "ratio"),
            ("top_2_concentration", "Top 2 credits / total amount", "ratio"),
            ("top_3_concentration", "Top 3 credits / total amount", "ratio"),
            ("max_single_credit_amount", "Maximum single credit amount", "currency"),
            ("min_single_credit_amount", "Minimum single credit amount", "currency"),
            ("defaulted_concentration", "Defaulted amount / total amount", "ratio"),
            ("active_concentration", "Active amount / total amount", "ratio"),
            ("single_product_concentration", "Largest product type / total", "ratio"),
            ("concentration_risk_score", "Combined concentration risk (0-10)", "numeric"),
        ]
        
        for name, desc, unit in concentration_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="concentration_feature",
                data_type="int64" if unit == "flag" else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Concentration: {desc.lower()}",
                risk_direction="higher_is_riskier" if "concentration" in name or "defaulted" in name else "context_dependent"
            ))
    
    def _expand_anomaly_features(self) -> None:
        """Generate behavioral anomaly detection features."""
        anomaly_features = [
            ("latest_amount_zscore", "Z-score of latest credit amount vs history", "numeric"),
            ("is_latest_amount_outlier", "Latest amount is statistical outlier (>2 std)", "flag"),
            ("amount_iqr_outlier_count", "Count of amount outliers (IQR method)", "count"),
            ("behavior_change_flag", "Significant behavior change detected", "flag"),
            ("sudden_large_credit_flag", "Recent credit >3x previous average", "flag"),
            ("sudden_product_change", "Changed to new product type recently", "flag"),
            ("unusual_timing_flag", "Credit timing is unusual vs history", "flag"),
            ("risk_score_change", "Change in implied risk score", "numeric"),
            ("velocity_anomaly_flag", "Unusual acceleration in credit taking", "flag"),
            ("dormancy_break_flag", "Took credit after 12+ months of inactivity", "flag"),
        ]
        
        for name, desc, unit in anomaly_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="anomaly_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit == "flag" else None,
                business_interpretation=f"Anomaly detection: {desc.lower()}",
                risk_direction="higher_is_riskier"
            ))
    
    # === NEW PAYMENT & DURATION FEATURE EXPANSIONS ===
    
    def _expand_payment_features(self) -> None:
        """Generate monthly payment-based features."""
        payment_features = [
            ("total_monthly_payment", "Sum of all monthly payments", "currency"),
            ("avg_monthly_payment", "Average payment per credit", "currency"),
            ("max_monthly_payment", "Largest single payment", "currency"),
            ("min_monthly_payment", "Smallest payment", "currency"),
            ("std_monthly_payment", "Payment volatility", "currency"),
            ("il_monthly_payment_total", "IL total monthly payment", "currency"),
            ("mg_monthly_payment_total", "Mortgage monthly payment", "currency"),
            ("is_monthly_payment_total", "IS monthly payment", "currency"),
            ("active_monthly_payment", "Total payment on active credits", "currency"),
            ("payment_count", "Count of credits with payments", "count"),
            ("payment_concentration", "Max payment / total payments", "ratio"),
            ("largest_payment_product", "Product with largest payment (0-3)", "category"),
            ("high_payment_credit_count", "Credits with payment > 2000", "count"),
            ("low_payment_credit_count", "Credits with payment < 500", "count"),
            ("median_monthly_payment", "Median payment amount", "currency"),
        ]
        
        for name, desc, unit in payment_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="payment_feature",
                data_type="int64" if unit in ["count", "flag", "category"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit == "ratio" else (0, None),
                business_interpretation=f"Payment: {desc.lower()}",
                risk_direction="higher_is_riskier" if "payment" in name and "total" in name else "context_dependent"
            ))
    
    def _expand_remaining_term_features(self) -> None:
        """Generate remaining term features."""
        remaining_term_features = [
            ("avg_remaining_months", "Average remaining term (amount-weighted)", "months"),
            ("max_remaining_months", "Longest remaining term", "months"),
            ("min_remaining_months", "Shortest remaining term", "months"),
            ("total_remaining_months", "Sum of all remaining terms", "months"),
            ("has_maturing_3m", "Credit maturing in 3 months", "flag"),
            ("has_maturing_6m", "Credit maturing in 6 months", "flag"),
            ("has_maturing_12m", "Credit maturing in 12 months", "flag"),
            ("maturing_amount_3m", "Amount maturing in 3 months", "currency"),
            ("maturing_amount_6m", "Amount maturing in 6 months", "currency"),
            ("maturing_count_6m", "Count of credits maturing in 6 months", "count"),
            ("weighted_remaining_term", "Amount-weighted remaining term", "months"),
            ("remaining_term_spread", "Max - min remaining term", "months"),
        ]
        
        for name, desc, unit in remaining_term_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="remaining_term_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit == "flag" else (0, None),
                business_interpretation=f"Remaining term: {desc.lower()}",
                risk_direction="context_dependent"
            ))
    
    def _expand_obligation_features(self) -> None:
        """Generate obligation burden features."""
        obligation_features = [
            ("total_credit_burden", "Total amount + future payments", "currency"),
            ("future_payment_obligation", "Sum(remaining × payment)", "currency"),
            ("past_payment_made", "Estimated already paid", "currency"),
            ("single_credit_burden_ratio", "Largest credit burden / total", "ratio"),
            ("mortgage_burden_ratio", "Mortgage burden / total", "ratio"),
            ("unsecured_burden_ratio", "Unsecured burden / total", "ratio"),
            ("new_burden_12m", "Recently added burden (12m)", "currency"),
            ("burden_growth_rate", "Burden growth rate", "numeric"),
            ("burden_per_credit", "Average burden per credit", "currency"),
            ("monthly_burden_intensity", "Total payment / payment count", "currency"),
            ("high_burden_flag", "Total burden > threshold", "flag"),
            ("burden_to_amount_ratio", "Total burden / total amount", "ratio"),
        ]
        
        for name, desc, unit in obligation_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="obligation_feature",
                data_type="int64" if unit == "flag" else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Obligation: {desc.lower()}",
                risk_direction="higher_is_riskier" if "burden" in name else "context_dependent"
            ))
    
    def _expand_dti_proxy_features(self) -> None:
        """Generate DTI proxy features (no income, so use ratios)."""
        dti_features = [
            ("payment_to_amount_ratio", "Total payment / Total amount", "ratio"),
            ("payment_intensity", "Monthly payment / credit count", "currency"),
            ("avg_payment_per_credit", "Payment burden per credit", "currency"),
            ("payment_to_exposure_ratio", "Payment / max(amount)", "ratio"),
            ("high_payment_credit_ratio", "High payment credits / total", "ratio"),
            ("unsustainable_payment_flag", "Payment/Amount > threshold", "flag"),
            ("payment_vs_historical_avg", "Current vs past avg payment", "ratio"),
            ("payment_growth_rate", "Payment growth over time", "numeric"),
            ("annualized_payment_ratio", "12 × monthly_pmt / amount", "ratio"),
            ("payment_efficiency_score", "Combined payment efficiency (0-10)", "numeric"),
        ]
        
        for name, desc, unit in dti_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="dti_proxy_feature",
                data_type="int64" if unit == "flag" else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else None,
                business_interpretation=f"DTI proxy: {desc.lower()}",
                risk_direction="higher_is_riskier" if "unsustainable" in name or "high" in name else "context_dependent"
            ))
    
    def _expand_maturity_features(self) -> None:
        """Generate maturity features (considering 3-month closure rule)."""
        maturity_features = [
            ("recently_closed_count", "Credits closed in last 90 days", "count"),
            ("recently_closed_amount", "Amount closed in last 90 days", "currency"),
            ("closed_without_default", "Clean closures in 90 days", "count"),
            ("closed_with_default", "Defaulted closures (recovered)", "count"),
            ("closure_success_rate", "Clean closures / total closures", "ratio"),
            ("avg_closure_to_plan", "Actual vs planned closure", "numeric"),
            ("early_closure_count", "Paid off early", "count"),
            ("on_time_closure_count", "Closed on schedule", "count"),
            ("maturity_behavior_score", "Combined maturity behavior (0-10)", "numeric"),
            ("has_recent_closure", "Any closure in last 90 days", "flag"),
        ]
        
        for name, desc, unit in maturity_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="maturity_feature",
                data_type="int64" if unit in ["count", "flag"] else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Maturity: {desc.lower()}",
                risk_direction="lower_is_riskier" if "success" in name else "context_dependent"
            ))
    
    def _expand_duration_risk_features(self) -> None:
        """Generate duration risk features."""
        duration_risk_features = [
            ("avg_duration_months", "Average term taken", "months"),
            ("max_duration_months", "Longest term", "months"),
            ("min_duration_months", "Shortest term", "months"),
            ("duration_spread", "Max - min duration", "months"),
            ("duration_cv", "Duration coefficient of variation", "ratio"),
            ("prefers_long_term", ">60m credits dominate", "flag"),
            ("prefers_short_term", "<24m credits dominate", "flag"),
            ("mixed_duration_portfolio", "High duration variety", "flag"),
            ("duration_to_amount_ratio", "Months per 10K amount", "numeric"),
            ("high_duration_high_amount", "Long term + large amount flag", "flag"),
        ]
        
        for name, desc, unit in duration_risk_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="duration_risk_feature",
                data_type="int64" if unit == "flag" else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else (0, None),
                business_interpretation=f"Duration risk: {desc.lower()}",
                risk_direction="higher_is_riskier" if "long" in name or "high" in name else "context_dependent"
            ))
    
    def _expand_payment_behavior_features(self) -> None:
        """Generate payment behavior features."""
        payment_behavior_features = [
            ("payment_amount_cv", "Payment amount variation", "ratio"),
            ("payment_consistency_score", "How consistent payments are (0-1)", "ratio"),
            ("has_minimum_payments", "Very small payments exist", "flag"),
            ("first_payment_vs_last", "First vs last payment ratio", "ratio"),
            ("payment_trend_slope", "Payment trend direction", "numeric"),
            ("avg_payment_by_vintage", "Payment by credit age", "currency"),
            ("payment_spread", "Max - min payment", "currency"),
            ("payment_skewness", "Payment distribution skew", "numeric"),
        ]
        
        for name, desc, unit in payment_behavior_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="payment_behavior_feature",
                data_type="int64" if unit == "flag" else "float64",
                unit=unit,
                expected_range=(0, 1) if unit in ["ratio", "flag"] else None,
                business_interpretation=f"Payment behavior: {desc.lower()}",
                risk_direction="context_dependent"
            ))
    
    def _expand_term_structure_features(self) -> None:
        """Generate term structure features."""
        term_structure_features = [
            ("weighted_avg_term", "Amount-weighted duration", "months"),
            ("term_concentration_hhi", "HHI by term bucket", "ratio"),
            ("term_diversity_score", "Variety of terms (0-1)", "ratio"),
            ("short_term_ratio", "<12m credits / total", "ratio"),
            ("medium_term_ratio", "12-60m credits / total", "ratio"),
            ("long_term_ratio", ">60m credits / total", "ratio"),
            ("amortizing_ratio", "Amortizing products ratio", "ratio"),
            ("term_structure_score", "Combined term structure (0-10)", "numeric"),
        ]
        
        for name, desc, unit in term_structure_features:
            self._feature_definitions.append(FeatureDefinition(
                name=name,
                description=desc,
                formula=f"compute({name})",
                category="term_structure_feature",
                data_type="float64",
                unit=unit,
                expected_range=(0, 1) if unit == "ratio" else (0, None),
                business_interpretation=f"Term structure: {desc.lower()}",
                risk_direction="context_dependent"
            ))

    
    def _build_formula(
        self, 
        base: str, 
        product: DimensionValue, 
        window: DimensionValue, 
        status: DimensionValue
    ) -> str:
        """Build formula string from dimensions."""
        conditions = []
        
        if product.filter_expr:
            conditions.append(product.filter_expr)
        if window.months:
            conditions.append(f"opening_date >= ref_date - {window.months} months")
        if status.filter_expr:
            conditions.append(status.filter_expr)
        
        if conditions:
            where_clause = " AND ".join(conditions)
            return f"{base} WHERE {where_clause}"
        return base
    
    def _get_amount_interpretation(
        self,
        product: DimensionValue,
        window: DimensionValue,
        status: DimensionValue,
        agg: str
    ) -> str:
        """Generate business interpretation for amount features."""
        agg_meanings = {
            'sum': 'total exposure',
            'avg': 'typical credit size',
            'max': 'largest single credit',
            'min': 'smallest single credit',
            'std': 'variability in credit sizes'
        }
        return f"Measures {agg_meanings.get(agg, agg)} for {status.label} {product.label} in {window.label}"
    
    def _generate_customer_features(
        self,
        app_row: pd.Series,
        customer_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Generate all features for a single customer."""
        features = {
            'application_id': app_row['application_id'],
            'customer_id': app_row['customer_id'],
            'applicant_type': app_row.get('applicant_type', 'PRIMARY'),
            'application_date': app_row.get('application_date'),
            'target': app_row.get('target', 0),
        }
        
        # Filter to credit products only for most features
        credit_data = customer_data[
            customer_data['product_type'].isin(self.CREDIT_PRODUCTS)
        ]
        
        # Non-credit signals data
        non_credit_data = customer_data[
            customer_data['product_type'].isin(self.NON_CREDIT_PRODUCTS)
        ]
        
        # Skip list for duplicates (keep better-named versions)
        SKIP_READABLE_COUNTS = {'defaulted_count', 'recovered_count'}
        SKIP_READABLE_AMOUNTS = {
            'total_credit_max_amount', 'total_credit_min_amount',
            'defaulted_average_amount', 'defaulted_max_amount'
        }
        
        # === OPTIMIZED: Pre-compute all masks once ===
        n_credit = len(credit_data)
        
        if n_credit > 0:
            # Pre-compute product masks
            product_masks = {'all': pd.Series(True, index=credit_data.index)}
            for product in self.PRODUCT_TYPES[1:]:  # Skip 'all'
                product_type = product.filter_expr.split("'")[1] if product.filter_expr else None
                if product_type:
                    product_masks[product.code] = credit_data['product_type'] == product_type
                else:
                    product_masks[product.code] = pd.Series(True, index=credit_data.index)
            
            # Pre-compute time window masks
            window_masks = {'all': pd.Series(True, index=credit_data.index)}
            if 'opening_date' in credit_data.columns:
                for window in self.TIME_WINDOWS[1:]:  # Skip 'all'
                    if window.months:
                        cutoff = ref_date - pd.DateOffset(months=window.months)
                        window_masks[window.code] = credit_data['opening_date'] >= cutoff
                    else:
                        window_masks[window.code] = pd.Series(True, index=credit_data.index)
            else:
                for window in self.TIME_WINDOWS[1:]:
                    window_masks[window.code] = pd.Series(True, index=credit_data.index)
            
            # Pre-compute status masks
            status_masks = {'all': pd.Series(True, index=credit_data.index)}
            status_masks['active'] = credit_data['default_date'].isna()
            status_masks['defaulted'] = credit_data['default_date'].notna()
            status_masks['recovered'] = credit_data['recovery_date'].notna()
            
            # Get amounts array once
            amounts = credit_data['total_amount'].values if 'total_amount' in credit_data.columns else None
            
            # === Generate features using pre-computed masks (no DataFrame copying!) ===
            for product in self.PRODUCT_TYPES:
                p_mask = product_masks[product.code]
                for window in self.TIME_WINDOWS:
                    w_mask = window_masks[window.code]
                    pw_mask = p_mask & w_mask
                    for status in self.STATUS_FILTERS:
                        s_mask = status_masks[status.code]
                        final_mask = pw_mask & s_mask
                        count = final_mask.sum()
                        
                        # Count with cached readable name
                        cnt_readable = self._get_cached_name(
                            product.code, window.code, status.code, 'cnt'
                        )
                        if cnt_readable not in SKIP_READABLE_COUNTS:
                            features[cnt_readable] = count
                        
                        # Amount aggregations
                        if amounts is not None and count > 0:
                            filtered_amounts = amounts[final_mask.values]
                            for agg_code in ['sum', 'avg', 'max', 'min', 'std']:
                                amt_readable = self._get_cached_name(
                                    product.code, window.code, status.code, agg_code
                                )
                                if amt_readable in SKIP_READABLE_AMOUNTS:
                                    continue
                                if agg_code == 'sum':
                                    features[amt_readable] = filtered_amounts.sum()
                                elif agg_code == 'avg':
                                    features[amt_readable] = filtered_amounts.mean()
                                elif agg_code == 'max':
                                    features[amt_readable] = filtered_amounts.max()
                                elif agg_code == 'min':
                                    features[amt_readable] = filtered_amounts.min()
                                elif agg_code == 'std':
                                    features[amt_readable] = filtered_amounts.std() if len(filtered_amounts) > 1 else 0
                        else:
                            for agg_code in ['sum', 'avg', 'max', 'min', 'std']:
                                amt_readable = self._get_cached_name(
                                    product.code, window.code, status.code, agg_code
                                )
                                if amt_readable not in SKIP_READABLE_AMOUNTS:
                                    features[amt_readable] = 0
        else:
            # Empty credit data - set all to zero
            for product in self.PRODUCT_TYPES:
                for window in self.TIME_WINDOWS:
                    for status in self.STATUS_FILTERS:
                        cnt_readable = self._get_cached_name(
                            product.code, window.code, status.code, 'cnt'
                        )
                        if cnt_readable not in SKIP_READABLE_COUNTS:
                            features[cnt_readable] = 0
                        for agg_code in ['sum', 'avg', 'max', 'min', 'std']:
                            amt_readable = self._get_cached_name(
                                product.code, window.code, status.code, agg_code
                            )
                            if amt_readable not in SKIP_READABLE_AMOUNTS:
                                features[amt_readable] = 0
        
        # === COMPREHENSIVE PRE-COMPUTED CACHE FOR ALL HELPER METHODS ===
        # This eliminates redundant filtering across 27 helper methods
        if n_credit > 0:
            # Pre-compute common filtered subsets (used by 15+ methods)
            defaulted = credit_data[credit_data['default_date'].notna()]
            recovered = credit_data[credit_data['recovery_date'].notna()]
            active = credit_data[credit_data['default_date'].isna()]
            
            # Pre-compute sorted data (used by 8+ methods)
            sorted_data = credit_data.sort_values('opening_date') if 'opening_date' in credit_data.columns else credit_data
            
            # Pre-compute ages array (used by 6+ methods)
            if 'opening_date' in credit_data.columns:
                ages_days = (ref_date - credit_data['opening_date']).dt.days
                ages_months = ages_days / 30.44
            else:
                ages_days = pd.Series([0] * n_credit, index=credit_data.index)
                ages_months = ages_days
            
            # Pre-compute amounts (used by 10+ methods)
            amounts = credit_data['total_amount'] if 'total_amount' in credit_data.columns else pd.Series([0] * n_credit, index=credit_data.index)
            total_amount = amounts.sum()
            
            # Pre-compute opening dates list (used by interval/burst features)
            dates_list = sorted_data['opening_date'].tolist() if 'opening_date' in sorted_data.columns else []
            
            # Pre-compute product masks for helper methods
            product_data = {}
            for product in self.PRODUCT_TYPES[1:]:
                prod_code = product.filter_expr.split("'")[1] if product.filter_expr else None
                if prod_code:
                    product_data[product.code] = credit_data[credit_data['product_type'] == prod_code]
                else:
                    product_data[product.code] = pd.DataFrame()
            
            _cache = {
                'defaulted': defaulted,
                'recovered': recovered,
                'active': active,
                'sorted_data': sorted_data,
                'ages_days': ages_days,
                'ages_months': ages_months,
                'amounts': amounts,
                'total_amount': total_amount,
                'dates_list': dates_list,
                'product_data': product_data,
                'n_credit': n_credit,
                'ref_date': ref_date,
            }
        else:
            # Empty cache for zero-credit case
            _cache = {
                'defaulted': credit_data,
                'recovered': credit_data,
                'active': credit_data,
                'sorted_data': credit_data,
                'ages_days': pd.Series([], dtype=float),
                'ages_months': pd.Series([], dtype=float),
                'amounts': pd.Series([], dtype=float),
                'total_amount': 0,
                'dates_list': [],
                'product_data': {p.code: pd.DataFrame() for p in self.PRODUCT_TYPES[1:]},
                'n_credit': 0,
                'ref_date': ref_date,
            }
        
        # Generate ratio features (uses existing features dict only)
        features = self._add_ratio_features(features)
        
        # Generate temporal features (uses cache)
        features = self._add_temporal_features_optimized(features, credit_data, ref_date, _cache)
        
        # Generate trend features (uses existing features dict only)
        features = self._add_trend_features(features)
        
        # Generate risk signal features
        features = self._add_risk_signal_features(features, non_credit_data, credit_data)
        
        # Generate diversity features
        features = self._add_diversity_features(features, credit_data)
        
        # Generate behavioral features (uses cache)
        features = self._add_behavioral_features_optimized(features, credit_data, ref_date, _cache)
        
        # Generate default pattern features (uses cache)
        features = self._add_default_pattern_features_optimized(features, credit_data, _cache)
        
        # Generate sequence features (uses cache)
        features = self._add_sequence_features_optimized(features, credit_data, _cache)
        
        # Generate size pattern features
        features = self._add_size_pattern_features(features, credit_data)
        
        # Generate burst features (uses cache)
        features = self._add_burst_features_optimized(features, credit_data, _cache)
        
        # Generate interval features (uses cache)
        features = self._add_interval_features_optimized(features, credit_data, _cache)
        
        # Generate seasonal features
        features = self._add_seasonal_features(features, credit_data)
        
        # Generate weighted features (uses cache)
        features = self._add_weighted_features_optimized(features, credit_data, ref_date, _cache)
        
        # Generate co-applicant features  
        features = self._add_co_applicant_features(features, app_row, credit_data)
        
        # Generate relative features (uses cache)
        features = self._add_relative_features_optimized(features, credit_data, ref_date, _cache)
        
        # Generate complexity features
        features = self._add_complexity_features(features, credit_data)
        
        # Generate time-decay features (uses cache)
        features = self._add_time_decay_features_optimized(features, credit_data, ref_date, _cache)
        
        # EXPERT: Generate default lifecycle features
        features = self._add_default_lifecycle_features(features, credit_data, ref_date)
        
        # EXPERT: Generate early default features
        features = self._add_early_default_features(features, credit_data)
        
        # EXPERT: Generate amount tier features
        features = self._add_amount_tier_features(features, credit_data)
        
        # EXPERT: Generate cross-product features
        features = self._add_cross_product_features(features, credit_data)
        
        # EXPERT: Generate freshness features
        features = self._add_freshness_features(features, credit_data, ref_date)
        
        # EXPERT: Generate concentration features
        features = self._add_concentration_features(features, credit_data)
        
        # EXPERT: Generate anomaly features (uses cache)
        features = self._add_anomaly_features_optimized(features, credit_data, ref_date, _cache)
        
        # === PAYMENT & DURATION FEATURES (using new fields) ===
        
        # Payment features
        features = self._add_payment_features(features, credit_data)
        
        # Remaining term features
        features = self._add_remaining_term_features(features, credit_data, ref_date)
        
        # Obligation burden features
        features = self._add_obligation_features(features, credit_data, ref_date)
        
        # DTI proxy features
        features = self._add_dti_proxy_features(features, credit_data)
        
        # Maturity features
        features = self._add_maturity_features(features, credit_data, ref_date)
        
        # Duration risk features
        features = self._add_duration_risk_features(features, credit_data)
        
        # Payment behavior features
        features = self._add_payment_behavior_features(features, credit_data)
        
        # Term structure features
        features = self._add_term_structure_features(features, credit_data)
        
        return features
    
    def _apply_filters(
        self,
        data: pd.DataFrame,
        product: DimensionValue,
        window: DimensionValue,
        status: DimensionValue,
        ref_date: datetime
    ) -> pd.DataFrame:
        """Apply dimension filters to data."""
        filtered = data.copy()
        
        # Product filter
        if product.code != 'all' and product.filter_expr:
            product_type = product.filter_expr.split("'")[1]
            filtered = filtered[filtered['product_type'] == product_type]
        
        # Time window filter
        if window.months and 'opening_date' in filtered.columns:
            cutoff = ref_date - pd.DateOffset(months=window.months)
            filtered = filtered[filtered['opening_date'] >= cutoff]
        
        # Status filter
        if status.code == 'active':
            filtered = filtered[filtered['default_date'].isna()]
        elif status.code == 'defaulted':
            filtered = filtered[filtered['default_date'].notna()]
        elif status.code == 'recovered':
            filtered = filtered[filtered['recovery_date'].notna()]
        
        return filtered
    
    def _add_ratio_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Add ratio features based on existing counts and amounts.
        
        Uses NaN when denominator is 0 (ratio is undefined, not 0).
        """
        # Use the actual generated feature names
        total_cnt = features.get('total_credit_count', 0)
        total_amt = features.get('total_credit_amount', 0)
        
        # Product ratios (using correct feature names from the generator)
        product_to_feature = {
            'il': 'installment_loan',
            'is': 'installment_sale', 
            'cf': 'cash_facility',
            'mg': 'mortgage'
        }
        
        for code, prod_name in product_to_feature.items():
            # Try different possible feature name patterns
            prod_cnt = features.get(f'{prod_name}_count', 0)
            prod_amt = features.get(f'{prod_name}_total_amount', 0)
            
            # Also try alternative names
            if prod_cnt == 0:
                prod_cnt = features.get(f'{prod_name}_all_all_cnt', 0)
            if prod_amt == 0:
                prod_amt = features.get(f'{prod_name}_all_all_sum_amt', 0)
            
            # Use NaN when denominator is 0 (undefined ratio)
            features[f'{code}_to_total_ratio'] = prod_cnt / total_cnt if total_cnt > 0 else np.nan
            features[f'{code}_amt_to_total_ratio'] = prod_amt / total_amt if total_amt > 0 else np.nan
        
        # Status ratios
        for status in self.STATUS_FILTERS[1:]:
            status_cnt = features.get(f'{status.code}_count', 0)
            features[f'{status.code}_ratio'] = status_cnt / total_cnt if total_cnt > 0 else np.nan
        
        # Time window concentration
        for window in self.TIME_WINDOWS[1:]:
            window_cnt = features.get(f'count_{window.code}', 0)
            if window_cnt == 0:
                window_cnt = features.get(f'{window.code}_count', 0)
            features[f'{window.code}_concentration_ratio'] = window_cnt / total_cnt if total_cnt > 0 else np.nan
        
        return features
    
    def _add_temporal_features(
        self, 
        features: Dict[str, Any], 
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add temporal features.
        
        Uses NaN (not 0) when a feature is semantically undefined:
        - days_since_last_default = NaN when customer has never defaulted
        - avg_time_to_default_days = NaN when no defaults exist
        - avg_recovery_time_days = NaN when no recoveries exist
        """
        if len(credit_data) == 0 or 'opening_date' not in credit_data.columns:
            # No credit history at all
            features['oldest_credit_age_months'] = np.nan  # No credits = undefined age
            features['newest_credit_age_months'] = np.nan
            features['avg_credit_age_months'] = np.nan
            features['credit_history_length_months'] = 0     # 0 months of history is valid
            features['days_since_last_default'] = np.nan    # Never defaulted = undefined
            features['days_since_last_credit'] = np.nan     # No credits = undefined
            features['avg_time_to_default_days'] = np.nan   # Never defaulted = undefined
            features['avg_recovery_time_days'] = np.nan     # Never recovered = undefined
            return features
        
        # Credit ages - valid when credits exist
        ages = (ref_date - credit_data['opening_date']).dt.days / 30.44
        features['oldest_credit_age_months'] = ages.max() if len(ages) > 0 else np.nan
        features['newest_credit_age_months'] = ages.min() if len(ages) > 0 else np.nan
        features['avg_credit_age_months'] = ages.mean() if len(ages) > 0 else np.nan
        features['credit_history_length_months'] = features['oldest_credit_age_months']
        
        # Days since last credit - valid when credits exist
        if len(credit_data) > 0:
            days_since = (ref_date - credit_data['opening_date'].max()).days
            features['days_since_last_credit'] = max(days_since, 0)
        else:
            features['days_since_last_credit'] = np.nan
        
        # Days since last default - NaN if never defaulted (0 means "defaulted today")
        defaulted = credit_data[credit_data['default_date'].notna()]
        if len(defaulted) > 0:
            days_since_default = (ref_date - defaulted['default_date'].max()).days
            features['days_since_last_default'] = max(days_since_default, 0)
        else:
            # Customer has never defaulted - this is semantically different from 0!
            features['days_since_last_default'] = np.nan
        
        # Average time to default - NaN if no defaults
        if len(defaulted) > 0:
            time_to_default = (defaulted['default_date'] - defaulted['opening_date']).dt.days
            features['avg_time_to_default_days'] = time_to_default.mean()
        else:
            features['avg_time_to_default_days'] = np.nan
        
        # Average recovery time - NaN if no recoveries
        recovered = credit_data[credit_data['recovery_date'].notna()]
        if len(recovered) > 0:
            recovery_time = (recovered['recovery_date'] - recovered['default_date']).dt.days
            features['avg_recovery_time_days'] = recovery_time.mean()
        else:
            features['avg_recovery_time_days'] = np.nan
        
        return features
    
    def _add_trend_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Add trend features based on existing time-windowed features."""
        total_cnt = features.get('all_all_all_cnt', 0)
        total_amt = features.get('all_all_all_sum_amt', 0)
        
        # Velocity
        for window in ['3m', '6m']:
            window_cnt = features.get(f'all_{window}_all_cnt', 0)
            window_amt = features.get(f'all_{window}_all_sum_amt', 0)
            
            features[f'credit_velocity_{window}'] = window_cnt / total_cnt if total_cnt > 0 else 0
            features[f'amount_velocity_{window}'] = window_amt / total_amt if total_amt > 0 else 0
        
        # Trend comparisons (6m vs prior 6m)
        cnt_6m = features.get('all_6m_all_cnt', 0)
        cnt_12m = features.get('all_12m_all_cnt', 0)
        features['cnt_trend_6m_vs_12m'] = cnt_6m - (cnt_12m - cnt_6m)
        
        amt_6m = features.get('all_6m_all_sum_amt', 0)
        amt_12m = features.get('all_12m_all_sum_amt', 0)
        features['amt_trend_6m_vs_12m'] = amt_6m - (amt_12m - amt_6m)
        
        def_6m = features.get('all_6m_defaulted_cnt', 0)
        def_12m = features.get('all_12m_defaulted_cnt', 0)
        features['default_trend_6m_vs_12m'] = def_6m - (def_12m - def_6m)
        
        return features
    
    def _add_risk_signal_features(
        self,
        features: Dict[str, Any],
        non_credit_data: pd.DataFrame,
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add risk signal features."""
        # Overdraft signals
        overdraft = non_credit_data[non_credit_data['product_type'] == 'NON_AUTH_OVERDRAFT']
        features['has_overdraft'] = 1 if len(overdraft) > 0 else 0
        features['overdraft_count'] = len(overdraft)
        features['overdraft_amount'] = overdraft['total_amount'].sum() if len(overdraft) > 0 else 0
        
        # Overlimit signals
        overlimit = non_credit_data[non_credit_data['product_type'] == 'OVERLIMIT']
        features['has_overlimit'] = 1 if len(overlimit) > 0 else 0
        features['overlimit_count'] = len(overlimit)
        features['overlimit_amount'] = overlimit['total_amount'].sum() if len(overlimit) > 0 else 0
        
        # Combined stress flag
        features['financial_stress_flag'] = 1 if (features['has_overdraft'] or features['has_overlimit']) else 0
        
        # Default signals
        features['has_current_default'] = 1 if features.get('all_all_defaulted_cnt', 0) > 0 else 0
        features['ever_defaulted'] = features['has_current_default']  # Simplified
        
        return features
    
    def _add_diversity_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add diversity and concentration features."""
        total_cnt = features.get('all_all_all_cnt', 0)
        
        # Distinct product count
        if len(credit_data) > 0:
            features['distinct_product_count'] = credit_data['product_type'].nunique()
        else:
            features['distinct_product_count'] = 0
        
        features['product_diversity_ratio'] = features['distinct_product_count'] / 4.0
        
        # HHI
        if total_cnt > 0:
            shares = []
            for product in self.PRODUCT_TYPES[1:]:
                share = features.get(f'{product.code}_to_total_ratio', 0)
                shares.append(share ** 2)
            features['hhi_product'] = sum(shares)
        else:
            features['hhi_product'] = 0
        
        # Secured/unsecured ratios
        features['secured_ratio'] = features.get('mg_to_total_ratio', 0)
        features['unsecured_ratio'] = 1 - features['secured_ratio']
        features['revolving_ratio'] = features.get('cf_to_total_ratio', 0)
        
        return features
    
    def _add_behavioral_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add behavioral pattern features."""
        defaulted = credit_data[credit_data['default_date'].notna()]
        recovered = credit_data[credit_data['recovery_date'].notna()]
        
        # Time to default statistics
        if len(defaulted) > 0:
            ttd = (defaulted['default_date'] - defaulted['opening_date']).dt.days
            features['min_time_to_default_days'] = ttd.min()
            features['max_time_to_default_days'] = ttd.max()
            features['std_time_to_default_days'] = ttd.std() if len(ttd) > 1 else 0
        else:
            features['min_time_to_default_days'] = 0
            features['max_time_to_default_days'] = 0
            features['std_time_to_default_days'] = 0
        
        # Recovery time statistics
        if len(recovered) > 0:
            rec_time = (recovered['recovery_date'] - recovered['default_date']).dt.days
            features['min_recovery_time_days'] = rec_time.min()
            features['max_recovery_time_days'] = rec_time.max()
        else:
            features['min_recovery_time_days'] = 0
            features['max_recovery_time_days'] = 0
        
        # Recovery success rate
        default_cnt = len(defaulted)
        features['recovery_success_rate'] = len(recovered) / default_cnt if default_cnt > 0 else 0
        
        # Credit maturity
        total_cnt = len(credit_data)
        if total_cnt > 0 and 'opening_date' in credit_data.columns:
            ages = (ref_date - credit_data['opening_date']).dt.days / 30.44
            features['mature_credit_ratio'] = (ages > 24).sum() / total_cnt
            features['new_credit_ratio'] = (ages < 6).sum() / total_cnt
        else:
            features['mature_credit_ratio'] = 0
            features['new_credit_ratio'] = 0
        
        # Product-specific time-to-default
        for product in self.PRODUCT_TYPES[1:]:
            prod_code = product.filter_expr.split("'")[1] if product.filter_expr else None
            if prod_code:
                prod_defaults = defaulted[defaulted['product_type'] == prod_code]
                if len(prod_defaults) > 0:
                    ttd = (prod_defaults['default_date'] - prod_defaults['opening_date']).dt.days
                    features[f'{product.code}_time_to_default_min_days'] = ttd.min()
                    features[f'{product.code}_time_to_default_max_days'] = ttd.max()
                    features[f'{product.code}_time_to_default_avg_days'] = ttd.mean()
                else:
                    features[f'{product.code}_time_to_default_min_days'] = 0
                    features[f'{product.code}_time_to_default_max_days'] = 0
                    features[f'{product.code}_time_to_default_avg_days'] = 0
        
        return features
    
    def _add_default_pattern_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add default pattern features."""
        defaulted = credit_data[credit_data['default_date'].notna()]
        recovered = credit_data[credit_data['recovery_date'].notna()]
        active = credit_data[credit_data['default_date'].isna()]
        
        default_cnt = len(defaulted)
        features['default_count_ever'] = default_cnt
        features['default_recurrence_count'] = max(0, default_cnt - 1)
        features['has_multiple_defaults'] = 1 if default_cnt > 1 else 0
        features['has_recovered_default'] = 1 if len(recovered) > 0 else 0
        features['all_defaults_recovered'] = 1 if default_cnt > 0 and len(recovered) == default_cnt else 0
        
        # Default amounts
        total_amt = credit_data['total_amount'].sum() if len(credit_data) > 0 else 0
        default_amt = defaulted['total_amount'].sum() if len(defaulted) > 0 else 0
        features['default_amount_ratio'] = default_amt / total_amt if total_amt > 0 else 0
        features['avg_default_severity'] = defaulted['total_amount'].mean() if len(defaulted) > 0 else 0
        features['max_default_severity'] = defaulted['total_amount'].max() if len(defaulted) > 0 else 0
        
        # Default to active ratio
        active_cnt = len(active)
        features['default_to_active_ratio'] = default_cnt / active_cnt if active_cnt > 0 else 0
        
        # Product-specific default rates
        for product in self.PRODUCT_TYPES[1:]:
            prod_code = product.filter_expr.split("'")[1] if product.filter_expr else None
            if prod_code:
                prod_all = credit_data[credit_data['product_type'] == prod_code]
                prod_def = prod_all[prod_all['default_date'].notna()]
                features[f'{product.code}_default_rate'] = len(prod_def) / len(prod_all) if len(prod_all) > 0 else 0
        
        return features
    
    def _add_sequence_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add sequence and transition features."""
        if len(credit_data) == 0 or 'opening_date' not in credit_data.columns:
            # Set all to 0
            for prod in ['installment_loan', 'installment_sale', 'cash_facility', 'mortgage']:
                features[f'first_product_{prod}'] = 0
                features[f'last_product_{prod}'] = 0
            features['product_transition_count'] = 0
            features['moved_to_secured'] = 0
            features['moved_to_unsecured'] = 0
            features['same_product_streak'] = 0
            return features
        
        sorted_data = credit_data.sort_values('opening_date')
        product_seq = sorted_data['product_type'].tolist()
        
        # First and last product
        first_prod = product_seq[0] if product_seq else None
        last_prod = product_seq[-1] if product_seq else None
        
        for prod_name, prod_type in [('installment_loan', 'INSTALLMENT_LOAN'), 
                                      ('installment_sale', 'INSTALLMENT_SALE'),
                                      ('cash_facility', 'CASH_FACILITY'),
                                      ('mortgage', 'MORTGAGE')]:
            features[f'first_product_{prod_name}'] = 1 if first_prod == prod_type else 0
            features[f'last_product_{prod_name}'] = 1 if last_prod == prod_type else 0
        
        # Transition count
        transitions = sum(1 for i in range(1, len(product_seq)) if product_seq[i] != product_seq[i-1])
        features['product_transition_count'] = transitions
        
        # Moved to secured (mortgage)
        has_mortgage = 'MORTGAGE' in product_seq
        mortgage_idx = product_seq.index('MORTGAGE') if has_mortgage else -1
        unsecured_before = any(p != 'MORTGAGE' for p in product_seq[:mortgage_idx]) if mortgage_idx > 0 else False
        features['moved_to_secured'] = 1 if unsecured_before else 0
        
        # Moved to unsecured (had mortgage early, now only unsecured)
        features['moved_to_unsecured'] = 0  # Simplified
        
        # Same product streak
        max_streak = 1
        current_streak = 1
        for i in range(1, len(product_seq)):
            if product_seq[i] == product_seq[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        features['same_product_streak'] = max_streak if len(product_seq) > 0 else 0
        
        return features
    
    def _add_size_pattern_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add credit size pattern features."""
        if len(credit_data) == 0 or 'total_amount' not in credit_data.columns:
            features['has_small_credit'] = 0
            features['has_large_credit'] = 0
            features['has_very_large_credit'] = 0
            features['small_credit_count'] = 0
            features['large_credit_count'] = 0
            features['small_credit_ratio'] = 0
            features['large_credit_ratio'] = 0
            features['max_to_avg_amount_ratio'] = 0
            features['max_to_min_amount_ratio'] = 0
            features['amount_coefficient_of_variation'] = 0
            features['amount_range'] = 0
            features['median_credit_amount'] = 0
            features['amount_skewness'] = 0
            for product in self.PRODUCT_TYPES[1:]:
                features[f'{product.code}_avg_amount'] = 0
            return features
        
        amounts = credit_data['total_amount']
        total_cnt = len(amounts)
        
        features['has_small_credit'] = 1 if (amounts < 5000).any() else 0
        features['has_large_credit'] = 1 if (amounts > 50000).any() else 0
        features['has_very_large_credit'] = 1 if (amounts > 100000).any() else 0
        
        small_cnt = (amounts < 5000).sum()
        large_cnt = (amounts > 50000).sum()
        features['small_credit_count'] = small_cnt
        features['large_credit_count'] = large_cnt
        features['small_credit_ratio'] = small_cnt / total_cnt if total_cnt > 0 else 0
        features['large_credit_ratio'] = large_cnt / total_cnt if total_cnt > 0 else 0
        
        avg_amt = amounts.mean()
        max_amt = amounts.max()
        min_amt = amounts.min()
        std_amt = amounts.std()
        
        features['max_to_avg_amount_ratio'] = max_amt / avg_amt if avg_amt > 0 else 0
        features['max_to_min_amount_ratio'] = max_amt / min_amt if min_amt > 0 else 0
        features['amount_coefficient_of_variation'] = std_amt / avg_amt if avg_amt > 0 else 0
        features['amount_range'] = max_amt - min_amt
        features['median_credit_amount'] = amounts.median()
        features['amount_skewness'] = amounts.skew() if len(amounts) > 2 else 0
        
        # Product-specific averages
        for product in self.PRODUCT_TYPES[1:]:
            prod_code = product.filter_expr.split("'")[1] if product.filter_expr else None
            if prod_code:
                prod_data = credit_data[credit_data['product_type'] == prod_code]
                features[f'{product.code}_avg_amount'] = prod_data['total_amount'].mean() if len(prod_data) > 0 else 0
        
        return features
    
    def _add_burst_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add credit burst detection features."""
        if len(credit_data) < 2 or 'opening_date' not in credit_data.columns:
            features['credits_in_30_days'] = len(credit_data)
            features['credits_in_60_days'] = len(credit_data)
            features['credits_in_90_days'] = len(credit_data)
            features['has_credit_burst_30d'] = 0
            features['has_credit_burst_60d'] = 0
            features['max_credits_same_month'] = len(credit_data)
            features['amount_in_30_days'] = credit_data['total_amount'].sum() if len(credit_data) > 0 else 0
            features['amount_in_60_days'] = credit_data['total_amount'].sum() if len(credit_data) > 0 else 0
            features['burst_intensity_30d'] = 0
            features['burst_intensity_60d'] = 0
            for product in self.PRODUCT_TYPES[1:]:
                features[f'{product.code}_burst_count'] = 0
            return features
        
        sorted_data = credit_data.sort_values('opening_date')
        dates = sorted_data['opening_date'].tolist()
        amounts = sorted_data['total_amount'].tolist()
        
        # Count credits within various windows
        max_30d = max_60d = max_90d = 1
        max_30d_amt = max_60d_amt = 0
        
        for i in range(len(dates)):
            cnt_30 = cnt_60 = cnt_90 = 1
            amt_30 = amt_60 = amounts[i]
            
            for j in range(i + 1, len(dates)):
                diff = (dates[j] - dates[i]).days
                if diff <= 30:
                    cnt_30 += 1
                    amt_30 += amounts[j]
                if diff <= 60:
                    cnt_60 += 1
                    amt_60 += amounts[j]
                if diff <= 90:
                    cnt_90 += 1
            
            max_30d = max(max_30d, cnt_30)
            max_60d = max(max_60d, cnt_60)
            max_90d = max(max_90d, cnt_90)
            max_30d_amt = max(max_30d_amt, amt_30)
            max_60d_amt = max(max_60d_amt, amt_60)
        
        features['credits_in_30_days'] = max_30d
        features['credits_in_60_days'] = max_60d
        features['credits_in_90_days'] = max_90d
        features['has_credit_burst_30d'] = 1 if max_30d >= 2 else 0
        features['has_credit_burst_60d'] = 1 if max_60d >= 3 else 0
        features['amount_in_30_days'] = max_30d_amt
        features['amount_in_60_days'] = max_60d_amt
        features['burst_intensity_30d'] = max_30d / 1  # Per 30 days
        features['burst_intensity_60d'] = max_60d / 2  # Per 30 days
        
        # Same month
        if 'opening_date' in sorted_data.columns:
            months = sorted_data['opening_date'].dt.to_period('M')
            features['max_credits_same_month'] = months.value_counts().max() if len(months) > 0 else 0
        else:
            features['max_credits_same_month'] = 0
        
        # By product
        for product in self.PRODUCT_TYPES[1:]:
            prod_code = product.filter_expr.split("'")[1] if product.filter_expr else None
            if prod_code:
                prod_data = credit_data[credit_data['product_type'] == prod_code]
                features[f'{product.code}_burst_count'] = len(prod_data) if len(prod_data) >= 2 else 0
        
        return features
    
    def _add_interval_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add inter-credit interval features."""
        if len(credit_data) < 2 or 'opening_date' not in credit_data.columns:
            features['avg_inter_credit_days'] = 0
            features['min_inter_credit_days'] = 0
            features['max_inter_credit_days'] = 0
            features['std_inter_credit_days'] = 0
            features['median_inter_credit_days'] = 0
            features['inter_credit_cv'] = 0
            features['has_rapid_succession'] = 0
            features['rapid_succession_count'] = 0
            features['longest_credit_gap_days'] = 0
            features['recent_vs_historical_interval'] = 0
            features['interval_trend'] = 0
            features['is_accelerating'] = 0
            features['is_decelerating'] = 0
            for product in self.PRODUCT_TYPES[1:]:
                features[f'{product.code}_avg_interval_days'] = 0
            return features
        
        sorted_data = credit_data.sort_values('opening_date')
        dates = sorted_data['opening_date'].tolist()
        
        # Calculate intervals
        intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        
        if intervals:
            features['avg_inter_credit_days'] = np.mean(intervals)
            features['min_inter_credit_days'] = min(intervals)
            features['max_inter_credit_days'] = max(intervals)
            features['std_inter_credit_days'] = np.std(intervals) if len(intervals) > 1 else 0
            features['median_inter_credit_days'] = np.median(intervals)
            
            avg_interval = features['avg_inter_credit_days']
            features['inter_credit_cv'] = features['std_inter_credit_days'] / avg_interval if avg_interval > 0 else 0
            
            # Rapid succession (within 14 days)
            rapid = sum(1 for i in intervals if i <= 14)
            features['has_rapid_succession'] = 1 if rapid > 0 else 0
            features['rapid_succession_count'] = rapid
            features['longest_credit_gap_days'] = max(intervals)
            
            # Recent vs historical
            if len(intervals) >= 3:
                recent = intervals[-1]
                historical_avg = np.mean(intervals[:-1])
                features['recent_vs_historical_interval'] = recent / historical_avg if historical_avg > 0 else 0
                
                # Trend (simple linear fit direction)
                half = len(intervals) // 2
                first_half_avg = np.mean(intervals[:half]) if half > 0 else 0
                second_half_avg = np.mean(intervals[half:]) if half > 0 else 0
                features['interval_trend'] = second_half_avg - first_half_avg
                features['is_accelerating'] = 1 if features['interval_trend'] < -7 else 0
                features['is_decelerating'] = 1 if features['interval_trend'] > 7 else 0
            else:
                features['recent_vs_historical_interval'] = 0
                features['interval_trend'] = 0
                features['is_accelerating'] = 0
                features['is_decelerating'] = 0
        else:
            features['avg_inter_credit_days'] = 0
            features['min_inter_credit_days'] = 0
            features['max_inter_credit_days'] = 0
            features['std_inter_credit_days'] = 0
            features['median_inter_credit_days'] = 0
            features['inter_credit_cv'] = 0
            features['has_rapid_succession'] = 0
            features['rapid_succession_count'] = 0
            features['longest_credit_gap_days'] = 0
            features['recent_vs_historical_interval'] = 0
            features['interval_trend'] = 0
            features['is_accelerating'] = 0
            features['is_decelerating'] = 0
        
        # By product
        for product in self.PRODUCT_TYPES[1:]:
            prod_code = product.filter_expr.split("'")[1] if product.filter_expr else None
            if prod_code:
                prod_data = credit_data[credit_data['product_type'] == prod_code].sort_values('opening_date')
                if len(prod_data) >= 2:
                    prod_dates = prod_data['opening_date'].tolist()
                    prod_intervals = [(prod_dates[i+1] - prod_dates[i]).days for i in range(len(prod_dates)-1)]
                    features[f'{product.code}_avg_interval_days'] = np.mean(prod_intervals)
                else:
                    features[f'{product.code}_avg_interval_days'] = 0
        
        return features
    
    def _add_seasonal_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add seasonal pattern features."""
        if len(credit_data) == 0 or 'opening_date' not in credit_data.columns:
            features['most_common_opening_month'] = 0
            features['most_common_opening_quarter'] = 0
            for i in range(1, 5):
                features[f'credits_in_q{i}'] = 0
            features['credits_in_december'] = 0
            features['is_year_end_heavy'] = 0
            features['is_summer_heavy'] = 0
            features['monthly_spread'] = 0
            features['quarterly_concentration'] = 0
            return features
        
        months = credit_data['opening_date'].dt.month
        quarters = credit_data['opening_date'].dt.quarter
        total = len(credit_data)
        
        features['most_common_opening_month'] = months.mode().iloc[0] if len(months) > 0 else 0
        features['most_common_opening_quarter'] = quarters.mode().iloc[0] if len(quarters) > 0 else 0
        
        # Quarterly counts
        for q in range(1, 5):
            features[f'credits_in_q{q}'] = (quarters == q).sum()
        
        features['credits_in_december'] = (months == 12).sum()
        features['is_year_end_heavy'] = 1 if features['credits_in_q4'] / total > 0.25 else 0
        features['is_summer_heavy'] = 1 if (features['credits_in_q2'] + features['credits_in_q3']) / total > 0.5 else 0
        
        # Monthly spread (12 months - unique months used) / 12
        features['monthly_spread'] = months.nunique() / 12.0
        
        # Quarterly HHI
        q_shares = [features[f'credits_in_q{q}'] / total for q in range(1, 5)]
        features['quarterly_concentration'] = sum(s**2 for s in q_shares)
        
        return features
    
    def _add_weighted_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add amount-weighted features."""
        if len(credit_data) == 0:
            features['amount_weighted_default_rate'] = 0
            features['amount_weighted_age_months'] = 0
            features['amount_weighted_product_score'] = 0
            features['recent_amount_weight'] = 0
            features['large_credit_weight'] = 0
            features['defaulted_amount_weighted_age'] = 0
            return features
        
        total_amount = credit_data['total_amount'].sum()
        
        # Amount-weighted default rate
        defaulted = credit_data[credit_data['default_date'].notna()]
        features['amount_weighted_default_rate'] = defaulted['total_amount'].sum() / total_amount if total_amount > 0 else 0
        
        # Amount-weighted age
        if 'opening_date' in credit_data.columns and total_amount > 0:
            ages = (ref_date - credit_data['opening_date']).dt.days / 30.44
            weighted_age = (ages * credit_data['total_amount']).sum() / total_amount
            features['amount_weighted_age_months'] = weighted_age
        else:
            features['amount_weighted_age_months'] = 0
        
        # Product risk score (simplified)
        product_scores = {'INSTALLMENT_SALE': 3, 'CASH_FACILITY': 2, 'INSTALLMENT_LOAN': 1, 'MORTGAGE': 0}
        if 'product_type' in credit_data.columns and total_amount > 0:
            scores = credit_data['product_type'].map(lambda x: product_scores.get(x, 1))
            features['amount_weighted_product_score'] = (scores * credit_data['total_amount']).sum() / total_amount
        else:
            features['amount_weighted_product_score'] = 0
        
        # Recent amount weight
        if 'opening_date' in credit_data.columns:
            cutoff = ref_date - pd.DateOffset(months=6)
            recent = credit_data[credit_data['opening_date'] >= cutoff]
            features['recent_amount_weight'] = recent['total_amount'].sum() / total_amount if total_amount > 0 else 0
        else:
            features['recent_amount_weight'] = 0
        
        # Large credit weight
        large = credit_data[credit_data['total_amount'] > 50000]
        features['large_credit_weight'] = large['total_amount'].sum() / total_amount if total_amount > 0 else 0
        
        # Defaulted amount weighted age
        if len(defaulted) > 0 and 'opening_date' in defaulted.columns:
            def_ages = (ref_date - defaulted['opening_date']).dt.days / 30.44
            def_total = defaulted['total_amount'].sum()
            features['defaulted_amount_weighted_age'] = (def_ages * defaulted['total_amount']).sum() / def_total if def_total > 0 else 0
        else:
            features['defaulted_amount_weighted_age'] = 0
        
        return features
    
    def _add_co_applicant_features(
        self,
        features: Dict[str, Any],
        app_row: pd.Series,
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add co-applicant features."""
        applicant_type = app_row.get('applicant_type', 'PRIMARY')
        
        features['is_primary_applicant'] = 1 if applicant_type == 'PRIMARY' else 0
        features['is_co_applicant'] = 1 if applicant_type == 'CO_APPLICANT' else 0
        
        # These would need the full application data to compute properly
        # For now, set defaults
        features['application_has_co_applicant'] = 0  # Would need to check application
        features['co_applicant_credit_count'] = 0
        features['co_applicant_default_count'] = 0
        features['co_applicant_default_rate'] = 0
        features['co_applicant_total_amount'] = 0
        features['primary_vs_co_amount_ratio'] = 0
        features['combined_default_exposure'] = 0
        
        # Calculate from this customer's data
        defaulted = credit_data[credit_data['default_date'].notna()]
        has_default = len(defaulted) > 0
        
        features['either_has_default'] = 1 if has_default else 0
        features['both_have_default'] = 0  # Would need co-applicant data
        
        return features
    
    def _add_relative_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add relative and percentile-based features."""
        if len(credit_data) == 0 or 'opening_date' not in credit_data.columns:
            features['latest_amount_vs_average'] = 0
            features['latest_amount_vs_max'] = 0
            features['is_largest_credit_recent'] = 0
            features['is_smallest_credit_recent'] = 0
            features['amount_growth_rate'] = 0
            features['credit_count_growth_rate'] = 0
            features['default_rate_trend'] = 0
            features['amount_percentile_latest'] = 0
            features['recency_of_max_amount'] = 0
            features['latest_vs_first_amount_ratio'] = 0
            return features
        
        sorted_data = credit_data.sort_values('opening_date')
        amounts = sorted_data['total_amount']
        
        latest_amt = amounts.iloc[-1]
        first_amt = amounts.iloc[0]
        avg_amt = amounts.mean()
        max_amt = amounts.max()
        
        features['latest_amount_vs_average'] = latest_amt / avg_amt if avg_amt > 0 else 0
        features['latest_amount_vs_max'] = latest_amt / max_amt if max_amt > 0 else 0
        features['latest_vs_first_amount_ratio'] = latest_amt / first_amt if first_amt > 0 else 0
        
        # Is largest/smallest recent
        cutoff_6m = ref_date - pd.DateOffset(months=6)
        recent_data = sorted_data[sorted_data['opening_date'] >= cutoff_6m]
        
        features['is_largest_credit_recent'] = 1 if len(recent_data) > 0 and recent_data['total_amount'].max() == max_amt else 0
        features['is_smallest_credit_recent'] = 1 if len(recent_data) > 0 and recent_data['total_amount'].min() == amounts.min() else 0
        
        # Growth rate (simple linear slope direction)
        if len(amounts) >= 2:
            half = len(amounts) // 2
            first_half_avg = amounts.iloc[:half].mean() if half > 0 else 0
            second_half_avg = amounts.iloc[half:].mean() if half > 0 else 0
            features['amount_growth_rate'] = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
        else:
            features['amount_growth_rate'] = 0
        
        features['credit_count_growth_rate'] = 0  # Would need windowed data
        features['default_rate_trend'] = 0  # Would need windowed data
        
        # Percentile of latest
        features['amount_percentile_latest'] = (amounts < latest_amt).sum() / len(amounts) if len(amounts) > 0 else 0
        
        # Recency of max
        max_idx = amounts.idxmax()
        max_date = sorted_data.loc[max_idx, 'opening_date']
        features['recency_of_max_amount'] = (ref_date - max_date).days / 30.44
        
        return features
    
    def _add_complexity_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add portfolio complexity features."""
        if len(credit_data) == 0:
            features['product_entropy'] = 0
            features['amount_entropy'] = 0
            features['has_all_product_types'] = 0
            features['single_product_customer'] = 0
            features['dual_product_customer'] = 0
            features['product_status_combinations'] = 0
            features['portfolio_complexity_score'] = 0
            features['credit_frequency_regularity'] = 0
            features['amount_regularity'] = 0
            return features
        
        # Product entropy
        product_counts = credit_data['product_type'].value_counts(normalize=True)
        entropy = -sum(p * np.log2(p) for p in product_counts if p > 0)
        features['product_entropy'] = entropy
        
        # Amount entropy (binned)
        amounts = credit_data['total_amount']
        if len(amounts) > 1:
            bins = pd.cut(amounts, bins=5, labels=False)
            bin_counts = pd.Series(bins).value_counts(normalize=True)
            features['amount_entropy'] = -sum(p * np.log2(p) for p in bin_counts if p > 0)
        else:
            features['amount_entropy'] = 0
        
        # Product type counts
        unique_products = credit_data['product_type'].nunique()
        features['has_all_product_types'] = 1 if unique_products >= 4 else 0
        features['single_product_customer'] = 1 if unique_products == 1 else 0
        features['dual_product_customer'] = 1 if unique_products == 2 else 0
        
        # Product × status combinations
        if 'default_date' in credit_data.columns:
            credit_data_copy = credit_data.copy()
            credit_data_copy['status'] = credit_data_copy['default_date'].apply(lambda x: 'defaulted' if pd.notna(x) else 'active')
            combos = credit_data_copy.groupby(['product_type', 'status']).size().shape[0]
            features['product_status_combinations'] = combos
        else:
            features['product_status_combinations'] = unique_products
        
        # Complexity score (0-10)
        features['portfolio_complexity_score'] = min(10, unique_products * 2 + entropy * 2)
        
        # Regularity (inverse CV)
        cv = features.get('amount_coefficient_of_variation', 0)
        features['amount_regularity'] = 1 / (1 + cv)
        
        cv_interval = features.get('inter_credit_cv', 0)
        features['credit_frequency_regularity'] = 1 / (1 + cv_interval)
        
        return features
    
    def _add_time_decay_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add time-decay weighted features."""
        if len(credit_data) == 0 or 'opening_date' not in credit_data.columns:
            features['recency_weighted_default_count'] = 0
            features['recency_weighted_amount'] = 0
            features['exponential_decay_default_score'] = 0
            features['half_life_30d_amount'] = 0
            features['half_life_90d_amount'] = 0
            features['recency_score'] = 0
            features['stale_portfolio_flag'] = 1
            features['fresh_portfolio_flag'] = 0
            return features
        
        # Calculate days since each credit
        days_since = (ref_date - credit_data['opening_date']).dt.days
        
        # Recency-weighted default count (more recent = higher weight)
        defaulted = credit_data[credit_data['default_date'].notna()]
        if len(defaulted) > 0:
            def_days = (ref_date - defaulted['opening_date']).dt.days
            # Weight = 1 / (1 + days/365)
            weights = 1 / (1 + def_days / 365)
            features['recency_weighted_default_count'] = weights.sum()
        else:
            features['recency_weighted_default_count'] = 0
        
        # Recency-weighted amount
        weights = 1 / (1 + days_since / 365)
        features['recency_weighted_amount'] = (credit_data['total_amount'] * weights).sum()
        
        # Exponential decay default score
        if len(defaulted) > 0:
            def_days = (ref_date - defaulted['opening_date']).dt.days
            decay = np.exp(-def_days / 365)  # 1-year half-life
            features['exponential_decay_default_score'] = (defaulted['total_amount'] * decay).sum()
        else:
            features['exponential_decay_default_score'] = 0
        
        # Half-life amounts
        decay_30 = np.exp(-days_since * np.log(2) / 30)
        decay_90 = np.exp(-days_since * np.log(2) / 90)
        features['half_life_30d_amount'] = (credit_data['total_amount'] * decay_30).sum()
        features['half_life_90d_amount'] = (credit_data['total_amount'] * decay_90).sum()
        
        # Recency score (0-1)
        max_age = days_since.max()
        features['recency_score'] = 1 - (days_since.min() / max_age) if max_age > 0 else 1
        
        # Stale/fresh flags
        most_recent_days = days_since.min()
        features['stale_portfolio_flag'] = 1 if most_recent_days > 365 else 0
        features['fresh_portfolio_flag'] = 1 if most_recent_days < 90 else 0
        
        return features
    
    def _add_default_lifecycle_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add default lifecycle and recovery pattern features."""
        defaulted = credit_data[credit_data['default_date'].notna()]
        recovered = credit_data[credit_data['recovery_date'].notna()]
        
        if len(credit_data) == 0 or 'opening_date' not in credit_data.columns:
            features['credits_after_first_recovery'] = 0
            features['defaulted_again_after_recovery'] = 0
            features['days_clean_before_first_default'] = 0
            features['time_since_last_recovery_days'] = 0
            features['recovery_to_new_credit_days'] = 0
            features['multiple_recovery_cycles'] = 0
            features['recovery_cycle_count'] = len(recovered)
            features['clean_after_recovery'] = 0
            features['first_default_age_months'] = 0
            features['last_default_age_months'] = 0
            return features
        
        # Recovery cycle count
        features['recovery_cycle_count'] = len(recovered)
        features['multiple_recovery_cycles'] = 1 if len(recovered) > 1 else 0
        
        if len(recovered) > 0:
            first_recovery = recovered['recovery_date'].min()
            
            # Credits after first recovery
            credits_after = credit_data[credit_data['opening_date'] > first_recovery]
            features['credits_after_first_recovery'] = len(credits_after)
            
            # Defaulted again after recovery
            defaults_after = defaulted[defaulted['default_date'] > first_recovery]
            features['defaulted_again_after_recovery'] = 1 if len(defaults_after) > 0 else 0
            features['clean_after_recovery'] = 1 if len(defaults_after) == 0 else 0
            
            # Time since last recovery
            last_recovery = recovered['recovery_date'].max()
            features['time_since_last_recovery_days'] = (ref_date - last_recovery).days
            
            # Recovery to new credit
            credits_post_recovery = credit_data[credit_data['opening_date'] > first_recovery]
            if len(credits_post_recovery) > 0:
                first_post = credits_post_recovery['opening_date'].min()
                features['recovery_to_new_credit_days'] = (first_post - first_recovery).days
            else:
                features['recovery_to_new_credit_days'] = 0
        else:
            features['credits_after_first_recovery'] = 0
            features['defaulted_again_after_recovery'] = 0
            features['time_since_last_recovery_days'] = 0
            features['recovery_to_new_credit_days'] = 0
            features['clean_after_recovery'] = 1 if len(defaulted) == 0 else 0
        
        # Days clean before first default
        if len(defaulted) > 0:
            first_credit = credit_data['opening_date'].min()
            first_default = defaulted['default_date'].min()
            features['days_clean_before_first_default'] = (first_default - first_credit).days
            features['first_default_age_months'] = (ref_date - first_default).days / 30.44
            features['last_default_age_months'] = (ref_date - defaulted['default_date'].max()).days / 30.44
        else:
            features['days_clean_before_first_default'] = (ref_date - credit_data['opening_date'].min()).days
            features['first_default_age_months'] = 0
            features['last_default_age_months'] = 0
        
        return features
    
    def _add_early_default_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add early default detection features."""
        defaulted = credit_data[credit_data['default_date'].notna()]
        
        if len(defaulted) == 0:
            features['has_early_default_90d'] = 0
            features['has_early_default_180d'] = 0
            features['has_early_default_365d'] = 0
            features['early_default_count_90d'] = 0
            features['early_default_count_180d'] = 0
            features['early_default_ratio'] = 0
            features['fastest_default_days'] = 0
            features['slowest_default_days'] = 0
            features['early_default_amount'] = 0
            features['early_default_amount_ratio'] = 0
            return features
        
        # Time to default
        ttd = (defaulted['default_date'] - defaulted['opening_date']).dt.days
        
        features['fastest_default_days'] = ttd.min()
        features['slowest_default_days'] = ttd.max()
        
        # Early defaults
        early_90 = ttd <= 90
        early_180 = ttd <= 180
        early_365 = ttd <= 365
        
        features['has_early_default_90d'] = 1 if early_90.any() else 0
        features['has_early_default_180d'] = 1 if early_180.any() else 0
        features['has_early_default_365d'] = 1 if early_365.any() else 0
        features['early_default_count_90d'] = early_90.sum()
        features['early_default_count_180d'] = early_180.sum()
        features['early_default_ratio'] = early_180.sum() / len(defaulted) if len(defaulted) > 0 else 0
        
        # Early default amount
        early_defaults = defaulted[early_180]
        features['early_default_amount'] = early_defaults['total_amount'].sum() if len(early_defaults) > 0 else 0
        total_default_amt = defaulted['total_amount'].sum()
        features['early_default_amount_ratio'] = features['early_default_amount'] / total_default_amt if total_default_amt > 0 else 0
        
        return features
    
    def _add_amount_tier_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add amount tier behavior features."""
        if len(credit_data) == 0 or 'total_amount' not in credit_data.columns:
            for tier in ['micro', 'small_tier', 'medium_tier', 'large_tier']:
                features[f'{tier}_count'] = 0
                features[f'{tier}_default_rate'] = 0
            features['tier_with_most_defaults'] = 0
            features['tier_with_highest_default_rate'] = 0
            features['tier_spread'] = 0
            features['dominant_tier'] = 0
            return features
        
        amounts = credit_data['total_amount']
        defaulted = credit_data['default_date'].notna()
        
        # Define tiers
        tiers = {
            'micro': (amounts < 1000),
            'small_tier': (amounts >= 1000) & (amounts < 10000),
            'medium_tier': (amounts >= 10000) & (amounts < 50000),
            'large_tier': (amounts >= 50000)
        }
        
        tier_counts = {}
        tier_defaults = {}
        tier_default_rates = {}
        
        for tier_name, tier_mask in tiers.items():
            count = tier_mask.sum()
            tier_counts[tier_name] = count
            features[f'{tier_name}_count'] = count
            
            if count > 0:
                default_count = (tier_mask & defaulted).sum()
                tier_defaults[tier_name] = default_count
                tier_default_rates[tier_name] = default_count / count
                features[f'{tier_name}_default_rate'] = default_count / count
            else:
                tier_defaults[tier_name] = 0
                tier_default_rates[tier_name] = 0
                features[f'{tier_name}_default_rate'] = 0
        
        # Tier with most defaults
        tier_names = list(tiers.keys())
        if max(tier_defaults.values()) > 0:
            features['tier_with_most_defaults'] = tier_names.index(max(tier_defaults, key=tier_defaults.get))
        else:
            features['tier_with_most_defaults'] = 0
        
        # Tier with highest default rate
        valid_rates = {k: v for k, v in tier_default_rates.items() if tier_counts[k] > 0}
        if valid_rates:
            features['tier_with_highest_default_rate'] = tier_names.index(max(valid_rates, key=valid_rates.get))
        else:
            features['tier_with_highest_default_rate'] = 0
        
        # Tier spread
        features['tier_spread'] = sum(1 for c in tier_counts.values() if c > 0)
        
        # Dominant tier
        features['dominant_tier'] = tier_names.index(max(tier_counts, key=tier_counts.get))
        
        return features
    
    def _add_cross_product_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add cross-product correlation features."""
        defaulted = credit_data[credit_data['default_date'].notna()]
        
        if len(credit_data) == 0:
            features['cf_default_then_il_default'] = 0
            features['il_default_then_cf_default'] = 0
            features['multi_product_default'] = 0
            features['multi_product_default_count'] = 0
            features['mortgage_with_default_elsewhere'] = 0
            features['all_products_clean'] = 1
            features['unsecured_default_rate'] = 0
            features['secured_default_rate'] = 0
            features['high_risk_product_only'] = 0
            features['low_risk_product_only'] = 0
            features['cross_product_default_spread'] = 0
            return features
        
        products_used = credit_data['product_type'].unique()
        defaulted_products = defaulted['product_type'].unique() if len(defaulted) > 0 else []
        
        features['multi_product_default_count'] = len(defaulted_products)
        features['multi_product_default'] = 1 if len(defaulted_products) > 1 else 0
        features['all_products_clean'] = 1 if len(defaulted) == 0 else 0
        
        # Mortgage with default elsewhere
        has_mortgage = 'MORTGAGE' in products_used
        non_mortgage_defaults = len([p for p in defaulted_products if p != 'MORTGAGE'])
        features['mortgage_with_default_elsewhere'] = 1 if has_mortgage and non_mortgage_defaults > 0 else 0
        
        # Unsecured vs secured default rates
        unsecured = credit_data[credit_data['product_type'].isin(['INSTALLMENT_LOAN', 'INSTALLMENT_SALE', 'CASH_FACILITY'])]
        secured = credit_data[credit_data['product_type'] == 'MORTGAGE']
        
        if len(unsecured) > 0:
            unsecured_defaults = unsecured[unsecured['default_date'].notna()]
            features['unsecured_default_rate'] = len(unsecured_defaults) / len(unsecured)
        else:
            features['unsecured_default_rate'] = 0
        
        if len(secured) > 0:
            secured_defaults = secured[secured['default_date'].notna()]
            features['secured_default_rate'] = len(secured_defaults) / len(secured)
        else:
            features['secured_default_rate'] = 0
        
        # High/low risk only
        high_risk_products = {'CASH_FACILITY', 'INSTALLMENT_SALE'}
        features['high_risk_product_only'] = 1 if set(products_used).issubset(high_risk_products) else 0
        features['low_risk_product_only'] = 1 if set(products_used) == {'MORTGAGE'} else 0
        
        # Cross-product default spread
        unique_products = len(set(products_used))
        features['cross_product_default_spread'] = len(defaulted_products) / unique_products if unique_products > 0 else 0
        
        # Sequence: CF default then IL default, etc
        features['cf_default_then_il_default'] = 0
        features['il_default_then_cf_default'] = 0
        
        if len(defaulted) >= 2 and 'default_date' in defaulted.columns:
            sorted_defaults = defaulted.sort_values('default_date')
            default_seq = sorted_defaults['product_type'].tolist()
            for i in range(len(default_seq) - 1):
                if default_seq[i] == 'CASH_FACILITY' and default_seq[i+1] == 'INSTALLMENT_LOAN':
                    features['cf_default_then_il_default'] = 1
                if default_seq[i] == 'INSTALLMENT_LOAN' and default_seq[i+1] == 'CASH_FACILITY':
                    features['il_default_then_cf_default'] = 1
        
        return features
    
    def _add_freshness_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add credit freshness at application time features."""
        if len(credit_data) == 0 or 'opening_date' not in credit_data.columns:
            features['newest_credit_age_at_app_days'] = 0
            features['applied_with_fresh_credit_30d'] = 0
            features['applied_with_fresh_credit_60d'] = 0
            features['credits_near_application_before'] = 0
            features['credits_near_application_after'] = 0
            features['amount_near_application_before'] = 0
            features['credit_surge_at_application'] = 0
            features['application_timing_score'] = 0
            return features
        
        # Age of newest credit at application
        ages = (ref_date - credit_data['opening_date']).dt.days
        features['newest_credit_age_at_app_days'] = ages.min()
        
        # Fresh credit flags
        features['applied_with_fresh_credit_30d'] = 1 if ages.min() <= 30 else 0
        features['applied_with_fresh_credit_60d'] = 1 if ages.min() <= 60 else 0
        
        # Credits near application
        cutoff_30_before = ref_date - pd.DateOffset(days=30)
        cutoff_30_after = ref_date + pd.DateOffset(days=30)
        
        near_before = credit_data[(credit_data['opening_date'] >= cutoff_30_before) & 
                                   (credit_data['opening_date'] < ref_date)]
        near_after = credit_data[(credit_data['opening_date'] > ref_date) & 
                                  (credit_data['opening_date'] <= cutoff_30_after)]
        
        features['credits_near_application_before'] = len(near_before)
        features['credits_near_application_after'] = len(near_after)
        features['amount_near_application_before'] = near_before['total_amount'].sum() if len(near_before) > 0 else 0
        
        # Credit surge
        total_near = len(near_before) + len(near_after)
        features['credit_surge_at_application'] = 1 if total_near > 2 else 0
        
        # Application timing score (inverse of how unusual)
        avg_interval = features.get('avg_inter_credit_days', 30)
        features['application_timing_score'] = min(1.0, avg_interval / 365) if avg_interval > 0 else 0
        
        return features
    
    def _add_concentration_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add risk concentration features."""
        if len(credit_data) == 0 or 'total_amount' not in credit_data.columns:
            features['single_large_exposure_flag'] = 0
            features['top_credit_concentration'] = 0
            features['top_2_concentration'] = 0
            features['top_3_concentration'] = 0
            features['max_single_credit_amount'] = 0
            features['min_single_credit_amount'] = 0
            features['defaulted_concentration'] = 0
            features['active_concentration'] = 0
            features['single_product_concentration'] = 0
            features['concentration_risk_score'] = 0
            return features
        
        amounts = credit_data['total_amount'].sort_values(ascending=False)
        total = amounts.sum()
        
        features['max_single_credit_amount'] = amounts.max()
        features['min_single_credit_amount'] = amounts.min()
        
        # Top N concentration
        features['top_credit_concentration'] = amounts.iloc[0] / total if total > 0 else 0
        features['top_2_concentration'] = amounts.iloc[:2].sum() / total if len(amounts) >= 2 and total > 0 else features['top_credit_concentration']
        features['top_3_concentration'] = amounts.iloc[:3].sum() / total if len(amounts) >= 3 and total > 0 else features['top_2_concentration']
        
        features['single_large_exposure_flag'] = 1 if features['top_credit_concentration'] > 0.5 else 0
        
        # Defaulted vs active concentration
        defaulted = credit_data[credit_data['default_date'].notna()]
        active = credit_data[credit_data['default_date'].isna()]
        
        features['defaulted_concentration'] = defaulted['total_amount'].sum() / total if total > 0 else 0
        features['active_concentration'] = active['total_amount'].sum() / total if total > 0 else 0
        
        # Single product concentration
        product_amounts = credit_data.groupby('product_type')['total_amount'].sum()
        features['single_product_concentration'] = product_amounts.max() / total if total > 0 else 0
        
        # Concentration risk score (0-10)
        score = (features['top_credit_concentration'] * 3 + 
                 features['defaulted_concentration'] * 5 + 
                 features['single_product_concentration'] * 2)
        features['concentration_risk_score'] = min(10, score * 10)
        
        return features
    
    def _add_anomaly_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add behavioral anomaly detection features."""
        if len(credit_data) < 2 or 'total_amount' not in credit_data.columns:
            features['latest_amount_zscore'] = 0
            features['is_latest_amount_outlier'] = 0
            features['amount_iqr_outlier_count'] = 0
            features['behavior_change_flag'] = 0
            features['sudden_large_credit_flag'] = 0
            features['sudden_product_change'] = 0
            features['unusual_timing_flag'] = 0
            features['risk_score_change'] = 0
            features['velocity_anomaly_flag'] = 0
            features['dormancy_break_flag'] = 0
            return features
        
        sorted_data = credit_data.sort_values('opening_date')
        amounts = sorted_data['total_amount']
        
        # Z-score of latest amount
        latest_amt = amounts.iloc[-1]
        mean_amt = amounts.iloc[:-1].mean() if len(amounts) > 1 else amounts.mean()
        std_amt = amounts.iloc[:-1].std() if len(amounts) > 1 else 1
        
        if std_amt > 0:
            features['latest_amount_zscore'] = (latest_amt - mean_amt) / std_amt
        else:
            features['latest_amount_zscore'] = 0
        
        features['is_latest_amount_outlier'] = 1 if abs(features['latest_amount_zscore']) > 2 else 0
        
        # IQR outliers
        q1 = amounts.quantile(0.25)
        q3 = amounts.quantile(0.75)
        iqr = q3 - q1
        outliers = ((amounts < q1 - 1.5 * iqr) | (amounts > q3 + 1.5 * iqr)).sum()
        features['amount_iqr_outlier_count'] = outliers
        
        # Sudden large credit
        avg_prev = amounts.iloc[:-1].mean() if len(amounts) > 1 else amounts.mean()
        features['sudden_large_credit_flag'] = 1 if latest_amt > avg_prev * 3 else 0
        
        # Sudden product change
        products = sorted_data['product_type'].tolist()
        if len(products) >= 2:
            latest_product = products[-1]
            prev_products = set(products[:-1])
            features['sudden_product_change'] = 1 if latest_product not in prev_products else 0
        else:
            features['sudden_product_change'] = 0
        
        # Behavior change (simplified)
        features['behavior_change_flag'] = 1 if (features['is_latest_amount_outlier'] or features['sudden_product_change']) else 0
        
        # Unusual timing
        features['unusual_timing_flag'] = 1 if features.get('has_rapid_succession', 0) or features.get('is_accelerating', 0) else 0
        
        # Risk score change (simplified)
        features['risk_score_change'] = features.get('recency_weighted_default_count', 0)
        
        # Velocity anomaly
        features['velocity_anomaly_flag'] = 1 if features.get('is_accelerating', 0) else 0
        
        # Dormancy break
        if 'opening_date' in sorted_data.columns and len(sorted_data) >= 2:
            dates = sorted_data['opening_date'].tolist()
            last_gap = (dates[-1] - dates[-2]).days
            features['dormancy_break_flag'] = 1 if last_gap > 365 else 0
        else:
            features['dormancy_break_flag'] = 0
        
        return features
    
    # === NEW PAYMENT & DURATION COMPUTATION METHODS ===
    
    def _add_payment_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add monthly payment-based features."""
        payments = credit_data['monthly_payment'].dropna()
        
        if len(payments) == 0:
            features['total_monthly_payment'] = 0
            features['avg_monthly_payment'] = 0
            features['max_monthly_payment'] = 0
            features['min_monthly_payment'] = 0
            features['std_monthly_payment'] = 0
            features['il_monthly_payment_total'] = 0
            features['mg_monthly_payment_total'] = 0
            features['is_monthly_payment_total'] = 0
            features['active_monthly_payment'] = 0
            features['payment_count'] = 0
            features['payment_concentration'] = 0
            features['largest_payment_product'] = 0
            features['high_payment_credit_count'] = 0
            features['low_payment_credit_count'] = 0
            features['median_monthly_payment'] = 0
            return features
        
        features['total_monthly_payment'] = payments.sum()
        features['avg_monthly_payment'] = payments.mean()
        features['max_monthly_payment'] = payments.max()
        features['min_monthly_payment'] = payments.min()
        features['std_monthly_payment'] = payments.std() if len(payments) > 1 else 0
        features['payment_count'] = len(payments)
        features['median_monthly_payment'] = payments.median()
        
        # By product
        for prod, col in [('INSTALLMENT_LOAN', 'il'), ('MORTGAGE', 'mg'), ('INSTALLMENT_SALE', 'is')]:
            prod_data = credit_data[credit_data['product_type'] == prod]
            features[f'{col}_monthly_payment_total'] = prod_data['monthly_payment'].sum()
        
        # Active payments
        active = credit_data[credit_data['default_date'].isna() & credit_data['closure_date'].isna()]
        features['active_monthly_payment'] = active['monthly_payment'].sum()
        
        # Concentration
        total = features['total_monthly_payment']
        features['payment_concentration'] = features['max_monthly_payment'] / total if total > 0 else 0
        
        # Largest payment product
        product_payments = credit_data.groupby('product_type')['monthly_payment'].sum()
        if len(product_payments) > 0:
            products = ['INSTALLMENT_LOAN', 'MORTGAGE', 'INSTALLMENT_SALE', 'CASH_FACILITY']
            features['largest_payment_product'] = products.index(product_payments.idxmax()) if product_payments.idxmax() in products else 0
        else:
            features['largest_payment_product'] = 0
        
        # High/low payment counts
        features['high_payment_credit_count'] = (payments > 2000).sum()
        features['low_payment_credit_count'] = (payments < 500).sum()
        
        return features
    
    def _add_remaining_term_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add remaining term features."""
        if len(credit_data) == 0 or 'duration_months' not in credit_data.columns:
            for f in ['avg_remaining_months', 'max_remaining_months', 'min_remaining_months', 
                     'total_remaining_months', 'weighted_remaining_term', 'remaining_term_spread']:
                features[f] = 0
            for f in ['has_maturing_3m', 'has_maturing_6m', 'has_maturing_12m']:
                features[f] = 0
            for f in ['maturing_amount_3m', 'maturing_amount_6m', 'maturing_count_6m']:
                features[f] = 0
            return features
        
        # Calculate remaining months for active term products
        term_products = credit_data[
            (credit_data['duration_months'].notna()) & 
            (credit_data['closure_date'].isna()) &
            (credit_data['default_date'].isna())
        ].copy()
        
        if len(term_products) == 0:
            for f in ['avg_remaining_months', 'max_remaining_months', 'min_remaining_months', 
                     'total_remaining_months', 'weighted_remaining_term', 'remaining_term_spread']:
                features[f] = 0
            for f in ['has_maturing_3m', 'has_maturing_6m', 'has_maturing_12m']:
                features[f] = 0
            for f in ['maturing_amount_3m', 'maturing_amount_6m', 'maturing_count_6m']:
                features[f] = 0
            return features
        
        # Calculate elapsed and remaining months
        term_products['elapsed_months'] = (ref_date - term_products['opening_date']).dt.days / 30.44
        term_products['remaining_months'] = term_products['duration_months'] - term_products['elapsed_months']
        term_products['remaining_months'] = term_products['remaining_months'].clip(lower=0)
        
        remaining = term_products['remaining_months']
        amounts = term_products['total_amount']
        
        features['avg_remaining_months'] = remaining.mean()
        features['max_remaining_months'] = remaining.max()
        features['min_remaining_months'] = remaining.min()
        features['total_remaining_months'] = remaining.sum()
        features['remaining_term_spread'] = remaining.max() - remaining.min()
        
        # Weighted remaining term
        total_amt = amounts.sum()
        features['weighted_remaining_term'] = (remaining * amounts).sum() / total_amt if total_amt > 0 else 0
        
        # Maturing soon
        features['has_maturing_3m'] = 1 if (remaining <= 3).any() else 0
        features['has_maturing_6m'] = 1 if (remaining <= 6).any() else 0
        features['has_maturing_12m'] = 1 if (remaining <= 12).any() else 0
        
        maturing_3m = term_products[remaining <= 3]
        maturing_6m = term_products[remaining <= 6]
        features['maturing_amount_3m'] = maturing_3m['total_amount'].sum()
        features['maturing_amount_6m'] = maturing_6m['total_amount'].sum()
        features['maturing_count_6m'] = len(maturing_6m)
        
        return features
    
    def _add_obligation_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add obligation burden features."""
        if len(credit_data) == 0:
            for f in ['total_credit_burden', 'future_payment_obligation', 'past_payment_made',
                     'single_credit_burden_ratio', 'mortgage_burden_ratio', 'unsecured_burden_ratio',
                     'new_burden_12m', 'burden_growth_rate', 'burden_per_credit', 
                     'monthly_burden_intensity', 'burden_to_amount_ratio']:
                features[f] = 0
            features['high_burden_flag'] = 0
            return features
        
        total_amount = credit_data['total_amount'].sum()
        
        # Future payment obligation
        term_products = credit_data[credit_data['duration_months'].notna()].copy()
        if len(term_products) > 0 and 'opening_date' in term_products.columns:
            term_products['elapsed_months'] = (ref_date - term_products['opening_date']).dt.days / 30.44
            term_products['remaining_months'] = (term_products['duration_months'] - term_products['elapsed_months']).clip(lower=0)
            term_products['remaining_obligation'] = term_products['remaining_months'] * term_products['monthly_payment'].fillna(0)
            features['future_payment_obligation'] = term_products['remaining_obligation'].sum()
            features['past_payment_made'] = (term_products['elapsed_months'] * term_products['monthly_payment'].fillna(0)).sum()
        else:
            features['future_payment_obligation'] = 0
            features['past_payment_made'] = 0
        
        features['total_credit_burden'] = total_amount + features['future_payment_obligation']
        
        # Burden ratios
        total_burden = features['total_credit_burden']
        
        if total_burden > 0:
            # Single credit burden
            max_single = credit_data['total_amount'].max()
            features['single_credit_burden_ratio'] = max_single / total_burden
            
            # Mortgage burden
            mortgage = credit_data[credit_data['product_type'] == 'MORTGAGE']
            features['mortgage_burden_ratio'] = mortgage['total_amount'].sum() / total_burden
            
            # Unsecured burden
            unsecured = credit_data[credit_data['product_type'].isin(['INSTALLMENT_LOAN', 'INSTALLMENT_SALE', 'CASH_FACILITY'])]
            features['unsecured_burden_ratio'] = unsecured['total_amount'].sum() / total_burden
        else:
            features['single_credit_burden_ratio'] = 0
            features['mortgage_burden_ratio'] = 0
            features['unsecured_burden_ratio'] = 0
        
        # New burden (12m)
        if 'opening_date' in credit_data.columns:
            cutoff = ref_date - pd.DateOffset(months=12)
            recent = credit_data[credit_data['opening_date'] >= cutoff]
            features['new_burden_12m'] = recent['total_amount'].sum()
        else:
            features['new_burden_12m'] = 0
        
        features['burden_growth_rate'] = features['new_burden_12m'] / total_amount if total_amount > 0 else 0
        features['burden_per_credit'] = total_burden / len(credit_data)
        
        payment_count = features.get('payment_count', 1)
        features['monthly_burden_intensity'] = features.get('total_monthly_payment', 0) / payment_count if payment_count > 0 else 0
        
        features['high_burden_flag'] = 1 if total_burden > 200000 else 0
        features['burden_to_amount_ratio'] = total_burden / total_amount if total_amount > 0 else 0
        
        return features
    
    def _add_dti_proxy_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add DTI proxy features."""
        total_payment = features.get('total_monthly_payment', 0)
        total_amount = credit_data['total_amount'].sum() if len(credit_data) > 0 else 0
        credit_count = len(credit_data)
        
        features['payment_to_amount_ratio'] = total_payment / total_amount if total_amount > 0 else 0
        features['payment_intensity'] = total_payment / credit_count if credit_count > 0 else 0
        features['avg_payment_per_credit'] = features['payment_intensity']
        
        max_amount = credit_data['total_amount'].max() if len(credit_data) > 0 else 0
        features['payment_to_exposure_ratio'] = total_payment / max_amount if max_amount > 0 else 0
        
        high_payment_count = features.get('high_payment_credit_count', 0)
        features['high_payment_credit_ratio'] = high_payment_count / credit_count if credit_count > 0 else 0
        
        # Unsustainable if annualized payment > 30% of amount
        annualized = total_payment * 12
        features['annualized_payment_ratio'] = annualized / total_amount if total_amount > 0 else 0
        features['unsustainable_payment_flag'] = 1 if features['annualized_payment_ratio'] > 0.5 else 0
        
        features['payment_vs_historical_avg'] = 1.0  # Would need historical data
        features['payment_growth_rate'] = 0  # Would need historical data
        
        # Payment efficiency score (0-10)
        score = min(10, features['annualized_payment_ratio'] * 10)
        features['payment_efficiency_score'] = score
        
        return features
    
    def _add_maturity_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame,
        ref_date: datetime
    ) -> Dict[str, Any]:
        """Add maturity features (considering 3-month closure rule)."""
        # Recently closed (within 90 days - max visible due to deletion rule)
        closed = credit_data[credit_data['closure_date'].notna()]
        
        features['recently_closed_count'] = len(closed)
        features['recently_closed_amount'] = closed['total_amount'].sum() if len(closed) > 0 else 0
        
        # Clean vs defaulted closures
        if len(closed) > 0:
            clean_closed = closed[closed['default_date'].isna()]
            defaulted_closed = closed[closed['default_date'].notna()]
            
            features['closed_without_default'] = len(clean_closed)
            features['closed_with_default'] = len(defaulted_closed)
            features['closure_success_rate'] = len(clean_closed) / len(closed) if len(closed) > 0 else 0
        else:
            features['closed_without_default'] = 0
            features['closed_with_default'] = 0
            features['closure_success_rate'] = 0
        
        features['avg_closure_to_plan'] = 1.0  # On-time assumption
        features['early_closure_count'] = 0  # Would need original closure_date
        features['on_time_closure_count'] = len(closed)
        
        # Maturity behavior score
        features['maturity_behavior_score'] = min(10, features['closure_success_rate'] * 10)
        features['has_recent_closure'] = 1 if len(closed) > 0 else 0
        
        return features
    
    def _add_duration_risk_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add duration risk features."""
        durations = credit_data['duration_months'].dropna()
        
        if len(durations) == 0:
            for f in ['avg_duration_months', 'max_duration_months', 'min_duration_months',
                     'duration_spread', 'duration_cv', 'duration_to_amount_ratio']:
                features[f] = 0
            for f in ['prefers_long_term', 'prefers_short_term', 'mixed_duration_portfolio', 'high_duration_high_amount']:
                features[f] = 0
            return features
        
        features['avg_duration_months'] = durations.mean()
        features['max_duration_months'] = durations.max()
        features['min_duration_months'] = durations.min()
        features['duration_spread'] = durations.max() - durations.min()
        features['duration_cv'] = durations.std() / durations.mean() if durations.mean() > 0 else 0
        
        # Preferences
        long_term = (durations > 60).sum()
        short_term = (durations < 24).sum()
        total = len(durations)
        
        features['prefers_long_term'] = 1 if long_term / total > 0.5 else 0
        features['prefers_short_term'] = 1 if short_term / total > 0.5 else 0
        features['mixed_duration_portfolio'] = 1 if features['duration_cv'] > 0.5 else 0
        
        # Duration to amount
        term_products = credit_data[credit_data['duration_months'].notna()]
        total_amount = term_products['total_amount'].sum()
        total_duration = durations.sum()
        features['duration_to_amount_ratio'] = total_duration / (total_amount / 10000) if total_amount > 0 else 0
        
        # High duration + high amount
        high_both = term_products[(term_products['duration_months'] > 60) & (term_products['total_amount'] > 50000)]
        features['high_duration_high_amount'] = 1 if len(high_both) > 0 else 0
        
        return features
    
    def _add_payment_behavior_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add payment behavior features."""
        payments = credit_data['monthly_payment'].dropna()
        
        if len(payments) == 0:
            for f in ['payment_amount_cv', 'payment_consistency_score', 'first_payment_vs_last',
                     'payment_trend_slope', 'avg_payment_by_vintage', 'payment_spread', 'payment_skewness']:
                features[f] = 0
            features['has_minimum_payments'] = 0
            return features
        
        mean_pmt = payments.mean()
        std_pmt = payments.std() if len(payments) > 1 else 0
        
        features['payment_amount_cv'] = std_pmt / mean_pmt if mean_pmt > 0 else 0
        features['payment_consistency_score'] = 1 / (1 + features['payment_amount_cv'])
        features['has_minimum_payments'] = 1 if (payments < 100).any() else 0
        
        # First vs last (by opening date order)
        if 'opening_date' in credit_data.columns and len(payments) >= 2:
            sorted_data = credit_data[credit_data['monthly_payment'].notna()].sort_values('opening_date')
            first_pmt = sorted_data['monthly_payment'].iloc[0]
            last_pmt = sorted_data['monthly_payment'].iloc[-1]
            features['first_payment_vs_last'] = first_pmt / last_pmt if last_pmt > 0 else 0
        else:
            features['first_payment_vs_last'] = 0
        
        features['payment_trend_slope'] = 0  # Simplified
        features['avg_payment_by_vintage'] = mean_pmt
        features['payment_spread'] = payments.max() - payments.min()
        features['payment_skewness'] = payments.skew() if len(payments) >= 3 else 0
        
        return features
    
    def _add_term_structure_features(
        self,
        features: Dict[str, Any],
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add term structure features."""
        durations = credit_data['duration_months'].dropna()
        amounts = credit_data[credit_data['duration_months'].notna()]['total_amount']
        
        if len(durations) == 0:
            for f in ['weighted_avg_term', 'term_concentration_hhi', 'term_diversity_score',
                     'short_term_ratio', 'medium_term_ratio', 'long_term_ratio',
                     'amortizing_ratio', 'term_structure_score']:
                features[f] = 0
            return features
        
        # Weighted average term
        total_amt = amounts.sum()
        features['weighted_avg_term'] = (durations * amounts).sum() / total_amt if total_amt > 0 else 0
        
        # Term buckets
        total = len(durations)
        short = (durations < 12).sum()
        medium = ((durations >= 12) & (durations <= 60)).sum()
        long = (durations > 60).sum()
        
        features['short_term_ratio'] = short / total
        features['medium_term_ratio'] = medium / total
        features['long_term_ratio'] = long / total
        
        # HHI concentration
        shares = [features['short_term_ratio'], features['medium_term_ratio'], features['long_term_ratio']]
        features['term_concentration_hhi'] = sum(s**2 for s in shares)
        
        # Diversity score (inverse of HHI)
        features['term_diversity_score'] = 1 - features['term_concentration_hhi']
        
        # Amortizing ratio (term products vs revolving)
        amortizing = credit_data[credit_data['duration_months'].notna()]
        features['amortizing_ratio'] = len(amortizing) / len(credit_data) if len(credit_data) > 0 else 0
        
        # Term structure score (0-10)
        features['term_structure_score'] = features['term_diversity_score'] * 10
        
        return features
    
    def build_data_dictionary(self) -> Dict[str, Any]:
        """
        Build comprehensive data dictionary from feature definitions.
        
        Returns:
            Dictionary with metadata and feature documentation
        """
        if not self._feature_definitions:
            self._expand_feature_definitions()
        
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_features': len(self._feature_definitions),
                'generator': 'FeatureFactory',
                'version': '1.0.0'
            },
            'categories': {
                'amount_feature': 'Amount-based features measuring credit exposure',
                'count_feature': 'Count-based features measuring credit activity',
                'ratio_feature': 'Ratio features measuring proportions and concentrations',
                'temporal_feature': 'Time-based features measuring credit history timing',
                'trend_feature': 'Trend features measuring changes over time',
                'risk_signal_feature': 'Binary and numeric risk indicators',
                'diversity_feature': 'Portfolio diversification and concentration measures',
                'behavioral_feature': 'Credit taking behavior and patterns',
                'default_pattern_feature': 'Default history and recurrence patterns',
                'sequence_feature': 'Product sequence and transition patterns',
                'size_pattern_feature': 'Credit size patterns and anomalies',
                'burst_feature': 'Rapid credit acquisition detection',
                'interval_feature': 'Time between credit events',
                'seasonal_feature': 'Seasonal and cyclical credit patterns',
                'weighted_feature': 'Amount-weighted aggregate features',
                'co_applicant_feature': 'Co-applicant relationship features',
                'relative_feature': 'Relative and percentile-based features',
                'complexity_feature': 'Portfolio complexity and regularity',
                'time_decay_feature': 'Recency-weighted features with time decay',
                'default_lifecycle_feature': 'Default and recovery lifecycle patterns',
                'early_default_feature': 'Early default detection and fast failures',
                'amount_tier_feature': 'Amount tier-specific risk behavior',
                'cross_product_feature': 'Cross-product correlation and contagion',
                'freshness_feature': 'Credit freshness at application time',
                'concentration_feature': 'Exposure concentration and single-name risk',
                'anomaly_feature': 'Behavioral anomaly and outlier detection',
                # NEW: Payment & Duration features
                'payment_feature': 'Monthly payment-based features',
                'remaining_term_feature': 'Remaining term and maturity features',
                'obligation_feature': 'Obligation burden and future payment features',
                'dti_proxy_feature': 'DTI proxy and payment intensity features',
                'maturity_feature': 'Credit maturity and closure behavior',
                'duration_risk_feature': 'Duration and term preference features',
                'payment_behavior_feature': 'Payment consistency and patterns',
                'term_structure_feature': 'Portfolio term structure features'
            },
            'dimensions': {
                'products': [p.code for p in self.PRODUCT_TYPES],
                'time_windows': [w.code for w in self.TIME_WINDOWS],
                'status_filters': [s.code for s in self.STATUS_FILTERS],
                'aggregations': list(self.AGGREGATIONS.keys())
            },
            'features': {
                f.name: f.to_dict() for f in self._feature_definitions
            }
        }
    
    def export_data_dictionary(
        self,
        output_dir: str,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export data dictionary to multiple formats.
        
        Args:
            output_dir: Output directory path
            formats: List of formats (yaml, json, excel, html). Default: all.
            
        Returns:
            Dictionary of format to file path
        """
        formats = formats or ['yaml', 'json', 'excel', 'html']
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dictionary = self.build_data_dictionary()
        exported = {}
        
        if 'yaml' in formats:
            yaml_path = output_path / 'data_dictionary.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(dictionary, f, default_flow_style=False, allow_unicode=True)
            exported['yaml'] = str(yaml_path)
        
        if 'json' in formats:
            json_path = output_path / 'data_dictionary.json'
            with open(json_path, 'w') as f:
                json.dump(dictionary, f, indent=2)
            exported['json'] = str(json_path)
        
        if 'excel' in formats:
            excel_path = output_path / 'data_dictionary.xlsx'
            self._export_excel(dictionary, excel_path)
            exported['excel'] = str(excel_path)
        
        if 'html' in formats:
            html_path = output_path / 'data_dictionary.html'
            self._export_html(dictionary, html_path)
            exported['html'] = str(html_path)
        
        return exported
    
    def _export_excel(self, dictionary: Dict[str, Any], path: Path) -> None:
        """Export data dictionary to Excel."""
        features_list = []
        for name, feature in dictionary['features'].items():
            features_list.append({
                'Feature Name': name,
                'Description': feature.get('description', ''),
                'Category': feature.get('category', ''),
                'Data Type': feature.get('data_type', ''),
                'Formula': feature.get('formula', ''),
                'Unit': feature.get('unit', ''),
                'Product': feature.get('product', ''),
                'Time Window': feature.get('time_window', ''),
                'Status Filter': feature.get('status_filter', ''),
                'Business Interpretation': feature.get('business_interpretation', ''),
                'Risk Direction': feature.get('risk_direction', ''),
            })
        
        df = pd.DataFrame(features_list)
        df.to_excel(path, index=False, sheet_name='Features')
    
    def _export_html(self, dictionary: Dict[str, Any], path: Path) -> None:
        """Export data dictionary to HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Dictionary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metadata {{ background: #e7e7e7; padding: 15px; margin: 20px 0; }}
        .category {{ margin: 10px 0; padding: 5px; background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Credit Scoring Feature Data Dictionary</h1>
    
    <div class="metadata">
        <strong>Generated:</strong> {dictionary['metadata']['generated_at']}<br>
        <strong>Total Features:</strong> {dictionary['metadata']['total_features']}
    </div>
    
    <h2>Feature Categories</h2>
    <ul>
"""
        for cat, desc in dictionary['categories'].items():
            count = sum(1 for f in dictionary['features'].values() if f.get('category') == cat)
            html += f"        <li><strong>{cat}</strong> ({count}): {desc}</li>\n"
        
        html += """    </ul>
    
    <h2>Features</h2>
    <table>
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Category</th>
            <th>Formula</th>
            <th>Business Interpretation</th>
        </tr>
"""
        for name, feature in dictionary['features'].items():
            html += f"""        <tr>
            <td><code>{name}</code></td>
            <td>{feature.get('description', '')}</td>
            <td>{feature.get('category', '')}</td>
            <td><code>{feature.get('formula', '')}</code></td>
            <td>{feature.get('business_interpretation', '')}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>
"""
        path.write_text(html)
    
    @property
    def feature_count(self) -> int:
        """Get total number of feature definitions."""
        if not self._feature_definitions:
            self._expand_feature_definitions()
        return len(self._feature_definitions)
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of all feature names."""
        if not self._feature_definitions:
            self._expand_feature_definitions()
        return [f.name for f in self._feature_definitions]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIMIZED HELPER METHODS - Use pre-computed cache for maximum performance
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _add_temporal_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame,
        ref_date: datetime, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized temporal features using pre-computed cache."""
        if _cache['n_credit'] == 0:
            features.update({'oldest_credit_age_months': 0, 'newest_credit_age_months': 0,
                'avg_credit_age_months': 0, 'credit_history_length_months': 0,
                'days_since_last_default': 0, 'days_since_last_credit': 0,
                'avg_time_to_default_days': 0, 'avg_recovery_time_days': 0})
            return features
        
        ages = _cache['ages_months']
        features['oldest_credit_age_months'] = ages.max()
        features['newest_credit_age_months'] = ages.min()
        features['avg_credit_age_months'] = ages.mean()
        features['credit_history_length_months'] = features['oldest_credit_age_months']
        
        dates_list = _cache['dates_list']
        features['days_since_last_credit'] = max((ref_date - dates_list[-1]).days, 0) if dates_list else 0
        
        defaulted = _cache['defaulted']
        if len(defaulted) > 0:
            features['days_since_last_default'] = max((ref_date - defaulted['default_date'].max()).days, 0)
            features['avg_time_to_default_days'] = (defaulted['default_date'] - defaulted['opening_date']).dt.days.mean()
        else:
            features['days_since_last_default'] = 0
            features['avg_time_to_default_days'] = 0
        
        recovered = _cache['recovered']
        features['avg_recovery_time_days'] = (recovered['recovery_date'] - recovered['default_date']).dt.days.mean() if len(recovered) > 0 else 0
        return features
    
    def _add_behavioral_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame,
        ref_date: datetime, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized behavioral features using pre-computed cache."""
        defaulted, recovered = _cache['defaulted'], _cache['recovered']
        ages_months, n_credit = _cache['ages_months'], _cache['n_credit']
        
        if len(defaulted) > 0:
            ttd = (defaulted['default_date'] - defaulted['opening_date']).dt.days
            features.update({'min_time_to_default_days': ttd.min(), 'max_time_to_default_days': ttd.max(),
                'std_time_to_default_days': ttd.std() if len(ttd) > 1 else 0})
        else:
            features.update({'min_time_to_default_days': 0, 'max_time_to_default_days': 0, 'std_time_to_default_days': 0})
        
        if len(recovered) > 0:
            rec_time = (recovered['recovery_date'] - recovered['default_date']).dt.days
            features.update({'min_recovery_time_days': rec_time.min(), 'max_recovery_time_days': rec_time.max()})
        else:
            features.update({'min_recovery_time_days': 0, 'max_recovery_time_days': 0})
        
        features['recovery_success_rate'] = len(recovered) / len(defaulted) if len(defaulted) > 0 else 0
        
        if n_credit > 0 and len(ages_months) > 0:
            features['mature_credit_ratio'] = (ages_months > 24).sum() / n_credit
            features['new_credit_ratio'] = (ages_months < 6).sum() / n_credit
        else:
            features.update({'mature_credit_ratio': 0, 'new_credit_ratio': 0})
        
        for product in self.PRODUCT_TYPES[1:]:
            prod_data = _cache['product_data'].get(product.code, pd.DataFrame())
            if len(prod_data) > 0:
                prod_def = prod_data[prod_data['default_date'].notna()]
                if len(prod_def) > 0:
                    ttd = (prod_def['default_date'] - prod_def['opening_date']).dt.days
                    features.update({f'{product.code}_time_to_default_min_days': ttd.min(),
                        f'{product.code}_time_to_default_max_days': ttd.max(),
                        f'{product.code}_time_to_default_avg_days': ttd.mean()})
                else:
                    features.update({f'{product.code}_time_to_default_min_days': 0,
                        f'{product.code}_time_to_default_max_days': 0, f'{product.code}_time_to_default_avg_days': 0})
            else:
                features.update({f'{product.code}_time_to_default_min_days': 0,
                    f'{product.code}_time_to_default_max_days': 0, f'{product.code}_time_to_default_avg_days': 0})
        return features
    
    def _add_default_pattern_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized default pattern features using pre-computed cache."""
        defaulted, recovered, active = _cache['defaulted'], _cache['recovered'], _cache['active']
        total_amount = _cache['total_amount']
        default_cnt = len(defaulted)
        
        features.update({'default_count_ever': default_cnt, 'default_recurrence_count': max(0, default_cnt - 1),
            'has_multiple_defaults': 1 if default_cnt > 1 else 0,
            'has_recovered_default': 1 if len(recovered) > 0 else 0,
            'all_defaults_recovered': 1 if default_cnt > 0 and len(recovered) == default_cnt else 0})
        
        default_amt = defaulted['total_amount'].sum() if len(defaulted) > 0 else 0
        features.update({'default_amount_ratio': default_amt / total_amount if total_amount > 0 else 0,
            'avg_default_severity': defaulted['total_amount'].mean() if len(defaulted) > 0 else 0,
            'max_default_severity': defaulted['total_amount'].max() if len(defaulted) > 0 else 0,
            'default_to_active_ratio': default_cnt / len(active) if len(active) > 0 else 0})
        
        for product in self.PRODUCT_TYPES[1:]:
            prod_data = _cache['product_data'].get(product.code, pd.DataFrame())
            if len(prod_data) > 0:
                prod_def = prod_data[prod_data['default_date'].notna()]
                features[f'{product.code}_default_rate'] = len(prod_def) / len(prod_data)
            else:
                features[f'{product.code}_default_rate'] = 0
        return features
    
    def _add_sequence_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized sequence features using pre-computed cache."""
        sorted_data = _cache['sorted_data']
        if _cache['n_credit'] == 0 or 'product_type' not in sorted_data.columns:
            for prod in ['installment_loan', 'installment_sale', 'cash_facility', 'mortgage']:
                features.update({f'first_product_{prod}': 0, f'last_product_{prod}': 0})
            features.update({'product_transition_count': 0, 'moved_to_secured': 0, 'moved_to_unsecured': 0, 'same_product_streak': 0})
            return features
        
        product_seq = sorted_data['product_type'].tolist()
        first_prod, last_prod = product_seq[0], product_seq[-1]
        
        for pn, pt in [('installment_loan', 'INSTALLMENT_LOAN'), ('installment_sale', 'INSTALLMENT_SALE'),
                       ('cash_facility', 'CASH_FACILITY'), ('mortgage', 'MORTGAGE')]:
            features[f'first_product_{pn}'] = 1 if first_prod == pt else 0
            features[f'last_product_{pn}'] = 1 if last_prod == pt else 0
        
        features['product_transition_count'] = sum(1 for i in range(1, len(product_seq)) if product_seq[i] != product_seq[i-1])
        has_mg = 'MORTGAGE' in product_seq
        mg_idx = product_seq.index('MORTGAGE') if has_mg else -1
        features['moved_to_secured'] = 1 if mg_idx > 0 and any(p != 'MORTGAGE' for p in product_seq[:mg_idx]) else 0
        features['moved_to_unsecured'] = 0
        
        max_streak, cur = 1, 1
        for i in range(1, len(product_seq)):
            cur = cur + 1 if product_seq[i] == product_seq[i-1] else 1
            max_streak = max(max_streak, cur)
        features['same_product_streak'] = max_streak
        return features
    
    def _add_burst_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized burst features using pre-computed cache."""
        sorted_data, dates_list = _cache['sorted_data'], _cache['dates_list']
        if len(dates_list) < 2:
            features.update({'credits_in_30_days': 0, 'credits_in_60_days': 0, 'credits_in_90_days': 0,
                'has_credit_burst_30d': 0, 'has_credit_burst_60d': 0, 'amount_in_30_days': 0,
                'amount_in_60_days': 0, 'burst_intensity_30d': 0, 'burst_intensity_60d': 0,
                'max_credits_same_month': 1 if _cache['n_credit'] > 0 else 0})
            for p in self.PRODUCT_TYPES[1:]: features[f'{p.code}_burst_count'] = 0
            return features
        
        amounts = sorted_data['total_amount'].values if 'total_amount' in sorted_data.columns else [0]*len(dates_list)
        max_30d, max_60d, max_90d, max_30d_amt, max_60d_amt = 0, 0, 0, 0, 0
        for i in range(len(dates_list)):
            c30, c60, c90, a30, a60 = 0, 0, 0, 0, 0
            for j in range(i, len(dates_list)):
                diff = (dates_list[j] - dates_list[i]).days
                if diff > 90: break
                if diff <= 30: c30 += 1; a30 += amounts[j]
                if diff <= 60: c60 += 1; a60 += amounts[j]
                if diff <= 90: c90 += 1
            max_30d, max_60d, max_90d = max(max_30d, c30), max(max_60d, c60), max(max_90d, c90)
            max_30d_amt, max_60d_amt = max(max_30d_amt, a30), max(max_60d_amt, a60)
        
        features.update({'credits_in_30_days': max_30d, 'credits_in_60_days': max_60d, 'credits_in_90_days': max_90d,
            'has_credit_burst_30d': 1 if max_30d >= 2 else 0, 'has_credit_burst_60d': 1 if max_60d >= 3 else 0,
            'amount_in_30_days': max_30d_amt, 'amount_in_60_days': max_60d_amt,
            'burst_intensity_30d': max_30d, 'burst_intensity_60d': max_60d / 2})
        
        if 'opening_date' in sorted_data.columns:
            months = sorted_data['opening_date'].dt.to_period('M')
            features['max_credits_same_month'] = months.value_counts().max() if len(months) > 0 else 0
        else:
            features['max_credits_same_month'] = 0
        
        for p in self.PRODUCT_TYPES[1:]:
            pd_data = _cache['product_data'].get(p.code, pd.DataFrame())
            features[f'{p.code}_burst_count'] = len(pd_data) if len(pd_data) >= 2 else 0
        return features
    
    def _add_interval_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized interval features using pre-computed cache."""
        dates_list = _cache['dates_list']
        if len(dates_list) < 2:
            features.update({'avg_inter_credit_days': 0, 'min_inter_credit_days': 0, 'max_inter_credit_days': 0,
                'std_inter_credit_days': 0, 'median_inter_credit_days': 0, 'inter_credit_cv': 0,
                'has_rapid_succession': 0, 'rapid_succession_count': 0, 'longest_credit_gap_days': 0,
                'recent_vs_historical_interval': 0, 'interval_trend': 0, 'is_accelerating': 0, 'is_decelerating': 0})
            for p in self.PRODUCT_TYPES[1:]: features[f'{p.code}_avg_interval_days'] = 0
            return features
        
        intervals = [(dates_list[i+1] - dates_list[i]).days for i in range(len(dates_list)-1)]
        avg_i = np.mean(intervals)
        features.update({'avg_inter_credit_days': avg_i, 'min_inter_credit_days': min(intervals),
            'max_inter_credit_days': max(intervals), 'std_inter_credit_days': np.std(intervals) if len(intervals) > 1 else 0,
            'median_inter_credit_days': np.median(intervals),
            'inter_credit_cv': np.std(intervals) / avg_i if avg_i > 0 else 0})
        
        rapid = sum(1 for i in intervals if i <= 14)
        features.update({'has_rapid_succession': 1 if rapid > 0 else 0, 'rapid_succession_count': rapid,
            'longest_credit_gap_days': max(intervals)})
        
        if len(intervals) >= 3:
            features['recent_vs_historical_interval'] = intervals[-1] / np.mean(intervals[:-1]) if np.mean(intervals[:-1]) > 0 else 0
            half = len(intervals) // 2
            trend = np.mean(intervals[half:]) - np.mean(intervals[:half]) if half > 0 else 0
            features.update({'interval_trend': trend, 'is_accelerating': 1 if trend < -7 else 0, 'is_decelerating': 1 if trend > 7 else 0})
        else:
            features.update({'recent_vs_historical_interval': 0, 'interval_trend': 0, 'is_accelerating': 0, 'is_decelerating': 0})
        
        for p in self.PRODUCT_TYPES[1:]:
            pd_data = _cache['product_data'].get(p.code, pd.DataFrame())
            if len(pd_data) >= 2 and 'opening_date' in pd_data.columns:
                pd_dates = pd_data.sort_values('opening_date')['opening_date'].tolist()
                features[f'{p.code}_avg_interval_days'] = np.mean([(pd_dates[i+1]-pd_dates[i]).days for i in range(len(pd_dates)-1)])
            else:
                features[f'{p.code}_avg_interval_days'] = 0
        return features
    
    def _add_weighted_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame, ref_date: datetime, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized weighted features using pre-computed cache."""
        n_credit, total_amount = _cache['n_credit'], _cache['total_amount']
        if n_credit == 0:
            features.update({'amount_weighted_default_rate': 0, 'amount_weighted_age_months': 0,
                'amount_weighted_product_score': 0, 'recent_amount_weight': 0, 'large_credit_weight': 0, 'defaulted_amount_weighted_age': 0})
            return features
        
        defaulted, amounts, ages_months = _cache['defaulted'], _cache['amounts'], _cache['ages_months']
        features['amount_weighted_default_rate'] = defaulted['total_amount'].sum() / total_amount if total_amount > 0 else 0
        features['amount_weighted_age_months'] = (ages_months * amounts).sum() / total_amount if total_amount > 0 else 0
        
        ps = {'INSTALLMENT_SALE': 3, 'CASH_FACILITY': 2, 'INSTALLMENT_LOAN': 1, 'MORTGAGE': 0}
        if 'product_type' in credit_data.columns:
            scores = credit_data['product_type'].map(lambda x: ps.get(x, 1))
            features['amount_weighted_product_score'] = (scores * amounts).sum() / total_amount if total_amount > 0 else 0
        else:
            features['amount_weighted_product_score'] = 0
        
        if 'opening_date' in credit_data.columns:
            recent_mask = credit_data['opening_date'] >= (ref_date - pd.DateOffset(months=6))
            features['recent_amount_weight'] = amounts[recent_mask].sum() / total_amount if total_amount > 0 else 0
        else:
            features['recent_amount_weight'] = 0
        
        features['large_credit_weight'] = amounts[amounts > 50000].sum() / total_amount if total_amount > 0 else 0
        
        if len(defaulted) > 0 and 'opening_date' in defaulted.columns:
            def_ages = (ref_date - defaulted['opening_date']).dt.days / 30.44
            def_total = defaulted['total_amount'].sum()
            features['defaulted_amount_weighted_age'] = (def_ages * defaulted['total_amount']).sum() / def_total if def_total > 0 else 0
        else:
            features['defaulted_amount_weighted_age'] = 0
        return features
    
    def _add_relative_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame, ref_date: datetime, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized relative features using pre-computed cache."""
        sorted_data, n_credit = _cache['sorted_data'], _cache['n_credit']
        if n_credit == 0 or 'total_amount' not in sorted_data.columns:
            features.update({'latest_amount_vs_average': 0, 'latest_amount_vs_max': 0, 'is_largest_credit_recent': 0,
                'is_smallest_credit_recent': 0, 'amount_growth_rate': 0, 'credit_count_growth_rate': 0,
                'default_rate_trend': 0, 'amount_percentile_latest': 0, 'recency_of_max_amount': 0, 'latest_vs_first_amount_ratio': 0})
            return features
        
        amts = sorted_data['total_amount']
        latest, first, avg, mx = amts.iloc[-1], amts.iloc[0], amts.mean(), amts.max()
        features.update({'latest_amount_vs_average': latest/avg if avg > 0 else 0, 'latest_amount_vs_max': latest/mx if mx > 0 else 0,
            'latest_vs_first_amount_ratio': latest/first if first > 0 else 0})
        
        if 'opening_date' in sorted_data.columns:
            recent = sorted_data[sorted_data['opening_date'] >= (ref_date - pd.DateOffset(months=6))]
            features['is_largest_credit_recent'] = 1 if len(recent) > 0 and recent['total_amount'].max() == mx else 0
            features['is_smallest_credit_recent'] = 1 if len(recent) > 0 and recent['total_amount'].min() == amts.min() else 0
        else:
            features.update({'is_largest_credit_recent': 0, 'is_smallest_credit_recent': 0})
        
        if len(amts) >= 2:
            h = len(amts) // 2
            f_avg, s_avg = amts.iloc[:h].mean() if h > 0 else 0, amts.iloc[h:].mean() if h > 0 else 0
            features['amount_growth_rate'] = (s_avg - f_avg) / f_avg if f_avg > 0 else 0
        else:
            features['amount_growth_rate'] = 0
        
        features.update({'credit_count_growth_rate': 0, 'default_rate_trend': 0,
            'amount_percentile_latest': (amts < latest).sum() / len(amts) if len(amts) > 0 else 0})
        
        if 'opening_date' in sorted_data.columns:
            features['recency_of_max_amount'] = (ref_date - sorted_data.loc[amts.idxmax(), 'opening_date']).days / 30.44
        else:
            features['recency_of_max_amount'] = 0
        return features
    
    def _add_time_decay_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame, ref_date: datetime, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized time-decay features using pre-computed cache."""
        n_credit, defaulted, ages_days, amounts = _cache['n_credit'], _cache['defaulted'], _cache['ages_days'], _cache['amounts']
        if n_credit == 0:
            features.update({'recency_weighted_default_count': 0, 'recency_weighted_default_amount': 0,
                'time_decay_total_amount': 0, 'time_decay_default_severity': 0, 'recent_activity_score': 0, 'historical_activity_score': 0})
            return features
        
        hl = 180
        dw = np.exp(-0.693 * ages_days.values / hl)
        dw = dw / dw.sum() if dw.sum() > 0 else dw
        
        if len(defaulted) > 0 and 'opening_date' in defaulted.columns:
            def_ages = (ref_date - defaulted['opening_date']).dt.days.values
            def_w = np.exp(-0.693 * def_ages / hl)
            features['recency_weighted_default_count'] = def_w.sum()
            features['recency_weighted_default_amount'] = (def_w * defaulted['total_amount'].values).sum() if 'total_amount' in defaulted.columns else 0
        else:
            features.update({'recency_weighted_default_count': 0, 'recency_weighted_default_amount': 0})
        
        features['time_decay_total_amount'] = (dw * amounts.values).sum() if len(amounts) > 0 else 0
        features['time_decay_default_severity'] = features['recency_weighted_default_amount']
        
        recent_mask = ages_days <= 180
        features['recent_activity_score'] = recent_mask.sum() / n_credit if n_credit > 0 else 0
        features['historical_activity_score'] = (~recent_mask).sum() / n_credit if n_credit > 0 else 0
        return features
    
    def _add_anomaly_features_optimized(
        self, features: Dict[str, Any], credit_data: pd.DataFrame, ref_date: datetime, _cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized anomaly features using pre-computed cache."""
        sorted_data, n_credit, dates_list = _cache['sorted_data'], _cache['n_credit'], _cache['dates_list']
        if n_credit < 2 or 'total_amount' not in sorted_data.columns:
            features.update({'latest_amount_zscore': 0, 'is_latest_amount_outlier': 0, 'amount_iqr_outlier_count': 0,
                'behavior_change_flag': 0, 'sudden_large_credit_flag': 0, 'sudden_product_change': 0,
                'unusual_timing_flag': 0, 'risk_score_change': 0, 'velocity_anomaly_flag': 0, 'dormancy_break_flag': 0})
            return features
        
        amts = sorted_data['total_amount']
        latest, mean_prev = amts.iloc[-1], amts.iloc[:-1].mean() if len(amts) > 1 else amts.mean()
        std_prev = amts.iloc[:-1].std() if len(amts) > 1 else 1
        
        features['latest_amount_zscore'] = (latest - mean_prev) / std_prev if std_prev > 0 else 0
        features['is_latest_amount_outlier'] = 1 if abs(features['latest_amount_zscore']) > 2 else 0
        
        q1, q3 = amts.quantile(0.25), amts.quantile(0.75)
        iqr = q3 - q1
        features['amount_iqr_outlier_count'] = ((amts < q1 - 1.5*iqr) | (amts > q3 + 1.5*iqr)).sum()
        features['sudden_large_credit_flag'] = 1 if latest > mean_prev * 3 else 0
        
        prods = sorted_data['product_type'].tolist()
        features['sudden_product_change'] = 1 if len(prods) >= 2 and prods[-1] not in set(prods[:-1]) else 0
        features['behavior_change_flag'] = 1 if features['is_latest_amount_outlier'] or features['sudden_product_change'] else 0
        features['unusual_timing_flag'] = 1 if features.get('has_rapid_succession', 0) or features.get('is_accelerating', 0) else 0
        features['risk_score_change'] = features.get('recency_weighted_default_count', 0)
        features['velocity_anomaly_flag'] = 1 if features.get('is_accelerating', 0) else 0
        features['dormancy_break_flag'] = 1 if len(dates_list) >= 2 and (dates_list[-1] - dates_list[-2]).days > 365 else 0
        return features
