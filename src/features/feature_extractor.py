"""
Feature Extractor

Main feature extraction engine that orchestrates all transformations.
Generates credit bureau features from raw data using config-driven rules.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.core.base import SparkComponent
from src.core.exceptions import FeatureEngineeringError


class FeatureExtractor(SparkComponent):
    """
    Main feature extraction engine for credit scoring.
    
    Generates features from credit bureau data including:
    - Amount-based features (sum, avg, max, min, std)
    - Product-specific features (per product type)
    - Default status features
    - Temporal features (age, recency)
    - Period-based features (3m, 6m, 12m, 24m)
    - Trend features (velocity, change ratios)
    - Ratio features (secured, revolving, high-risk)
    - Non-credit signals (overdraft, overlimit)
    """
    
    # Credit product types
    CREDIT_PRODUCTS = ['INSTALLMENT_LOAN', 'INSTALLMENT_SALE', 'CASH_FACILITY', 'MORTGAGE']
    NON_CREDIT_PRODUCTS = ['NON_AUTH_OVERDRAFT', 'OVERLIMIT']
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        name: Optional[str] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary
            spark_session: Active SparkSession
            name: Optional extractor name
        """
        super().__init__(config, spark_session, name or "FeatureExtractor")
        
        self.feature_config = self.get_config('features.feature_engineering', {})
        self._generated_features: List[Dict[str, Any]] = []
        
    def validate(self) -> bool:
        return super().validate()
    
    def run(
        self,
        applications_df: Any,
        credit_bureau_df: Any
    ) -> Any:
        """Run feature extraction."""
        return self.extract_features(applications_df, credit_bureau_df)
    
    def extract_features(
        self,
        applications_df: Any,
        credit_bureau_df: Any
    ) -> Any:
        """
        Extract all features from credit bureau data.
        
        Args:
            applications_df: Applications DataFrame with application_id, customer_id, application_date
            credit_bureau_df: Credit bureau DataFrame with credit records
            
        Returns:
            DataFrame with one row per (application_id, customer_id) and all features
        """
        self._start_execution()
        self._generated_features = []
        
        self.logger.info("Starting feature extraction")
        
        # Add application_date to credit bureau for temporal calculations
        bureau_with_date = credit_bureau_df.join(
            applications_df.select('application_id', 'customer_id', 'application_date'),
            on=['application_id', 'customer_id'],
            how='inner'
        )
        
        # Calculate credit age in months for each record
        bureau_with_date = bureau_with_date.withColumn(
            'credit_age_months',
            F.months_between(
                F.col('application_date'),
                F.col('opening_date')
            )
        )
        
        # Add credit product flag
        bureau_with_date = bureau_with_date.withColumn(
            'is_credit_product',
            F.col('product_type').isin(self.CREDIT_PRODUCTS)
        )
        
        # Generate all feature groups
        self.logger.info("Generating amount features")
        amount_features = self._generate_amount_features(bureau_with_date)
        
        self.logger.info("Generating count features")
        count_features = self._generate_count_features(bureau_with_date)
        
        self.logger.info("Generating product-specific features")
        product_features = self._generate_product_features(bureau_with_date)
        
        self.logger.info("Generating default features")
        default_features = self._generate_default_features(bureau_with_date)
        
        self.logger.info("Generating temporal features")
        temporal_features = self._generate_temporal_features(bureau_with_date)
        
        self.logger.info("Generating period features")
        period_features = self._generate_period_features(bureau_with_date)
        
        self.logger.info("Generating non-credit signal features")
        non_credit_features = self._generate_non_credit_features(bureau_with_date)
        
        # Join all feature groups
        key_cols = ['application_id', 'customer_id']
        
        features_df = amount_features
        
        for df in [count_features, product_features, default_features, 
                   temporal_features, period_features, non_credit_features]:
            features_df = features_df.join(df, on=key_cols, how='outer')
        
        # Fill nulls with 0 for count/amount features
        features_df = features_df.fillna(0)
        
        # Generate derived features (ratios, trends)
        self.logger.info("Generating derived features")
        features_df = self._generate_derived_features(features_df)
        
        # Join back with applications to get target
        final_df = applications_df.join(
            features_df,
            on=key_cols,
            how='left'
        )
        
        # Fill any remaining nulls
        final_df = final_df.fillna(0)

        # Preserve uid from input data, or create if not present
        if 'uid' not in final_df.columns:
            final_df = final_df.withColumn(
                'uid',
                F.concat(F.col('application_id'), F.lit('||'), F.col('customer_id'))
            )

        # Ensure uniqueness at uid level
        final_df = final_df.dropDuplicates(['uid'])

        feature_count = len([c for c in final_df.columns if c not in ['uid', 'application_id', 'customer_id', 'applicant_type', 'application_date', 'target']])
        self.logger.info(f"Generated {feature_count} features")

        self._end_execution()

        return final_df
    
    def _generate_amount_features(self, df: Any) -> Any:
        """Generate amount-based features."""
        key_cols = ['application_id', 'customer_id']
        
        # Filter to credit products only for main amount features
        credit_df = df.filter(F.col('is_credit_product'))
        
        features = credit_df.groupBy(key_cols).agg(
            F.sum('total_amount').alias('total_credit_amount'),
            F.avg('total_amount').alias('avg_credit_amount'),
            F.max('total_amount').alias('max_credit_amount'),
            F.min('total_amount').alias('min_credit_amount'),
            F.stddev('total_amount').alias('std_credit_amount')
        )
        
        # Total exposure (all products including non-credit)
        total_exposure = df.groupBy(key_cols).agg(
            F.sum('total_amount').alias('total_exposure')
        )
        
        features = features.join(total_exposure, on=key_cols, how='outer')
        
        self._register_features([
            'total_credit_amount', 'avg_credit_amount', 'max_credit_amount',
            'min_credit_amount', 'std_credit_amount', 'total_exposure'
        ])
        
        return features
    
    def _generate_count_features(self, df: Any) -> Any:
        """Generate count-based features."""
        key_cols = ['application_id', 'customer_id']
        
        # Credit products only
        credit_df = df.filter(F.col('is_credit_product'))
        
        features = credit_df.groupBy(key_cols).agg(
            F.count('*').alias('total_credit_count'),
            F.countDistinct('product_type').alias('distinct_product_count')
        )
        
        # Active credits (no default)
        active = credit_df.filter(F.col('default_date').isNull()).groupBy(key_cols).agg(
            F.count('*').alias('active_credit_count')
        )
        
        # Currently defaulted (default but no recovery)
        defaulted = credit_df.filter(
            F.col('default_date').isNotNull() & 
            F.col('recovery_date').isNull()
        ).groupBy(key_cols).agg(
            F.count('*').alias('defaulted_credit_count')
        )
        
        # Recovered from default
        recovered = credit_df.filter(
            F.col('recovery_date').isNotNull()
        ).groupBy(key_cols).agg(
            F.count('*').alias('recovered_credit_count')
        )
        
        # Ever defaulted
        ever_defaulted = credit_df.filter(
            F.col('default_date').isNotNull()
        ).groupBy(key_cols).agg(
            F.count('*').alias('ever_defaulted_count')
        )
        
        # Total record count (including non-credit)
        total_records = df.groupBy(key_cols).agg(
            F.count('*').alias('total_record_count')
        )
        
        # Join all
        for agg_df in [active, defaulted, recovered, ever_defaulted, total_records]:
            features = features.join(agg_df, on=key_cols, how='outer')
        
        self._register_features([
            'total_credit_count', 'distinct_product_count', 'active_credit_count',
            'defaulted_credit_count', 'recovered_credit_count', 'ever_defaulted_count',
            'total_record_count'
        ])
        
        return features
    
    def _generate_product_features(self, df: Any) -> Any:
        """Generate product-specific features."""
        key_cols = ['application_id', 'customer_id']
        
        features = None
        
        for product in self.CREDIT_PRODUCTS:
            product_df = df.filter(F.col('product_type') == product)
            
            product_features = product_df.groupBy(key_cols).agg(
                F.count('*').alias(f'{product}_count'),
                F.sum('total_amount').alias(f'{product}_total_amount'),
                F.avg('total_amount').alias(f'{product}_avg_amount'),
                F.sum(F.when(F.col('default_date').isNotNull(), 1).otherwise(0)).alias(f'{product}_default_count')
            )
            
            if features is None:
                features = product_features
            else:
                features = features.join(product_features, on=key_cols, how='outer')
            
            self._register_features([
                f'{product}_count', f'{product}_total_amount',
                f'{product}_avg_amount', f'{product}_default_count'
            ])
        
        return features
    
    def _generate_default_features(self, df: Any) -> Any:
        """Generate default status features."""
        key_cols = ['application_id', 'customer_id']
        
        credit_df = df.filter(F.col('is_credit_product'))
        
        # Defaulted amount
        defaulted_df = credit_df.filter(
            F.col('default_date').isNotNull() &
            F.col('recovery_date').isNull()
        ).groupBy(key_cols).agg(
            F.sum('total_amount').alias('total_defaulted_amount')
        )
        
        self._register_features(['total_defaulted_amount'])
        
        return defaulted_df
    
    def _generate_temporal_features(self, df: Any) -> Any:
        """Generate time-based features."""
        key_cols = ['application_id', 'customer_id']
        
        credit_df = df.filter(F.col('is_credit_product'))
        
        features = credit_df.groupBy(key_cols).agg(
            F.max('credit_age_months').alias('oldest_credit_age_months'),
            F.min('credit_age_months').alias('newest_credit_age_months'),
            F.avg('credit_age_months').alias('avg_credit_age_months')
        )
        
        # Time to default (for defaulted credits)
        default_df = credit_df.filter(F.col('default_date').isNotNull())
        default_time = default_df.withColumn(
            'time_to_default_days',
            F.datediff(F.col('default_date'), F.col('opening_date'))
        ).groupBy(key_cols).agg(
            F.avg('time_to_default_days').alias('avg_time_to_default_days')
        )
        
        features = features.join(default_time, on=key_cols, how='outer')
        
        # Days since last default
        last_default = default_df.groupBy(key_cols).agg(
            F.max('default_date').alias('last_default_date')
        )
        
        last_default = last_default.join(
            df.select('application_id', 'customer_id', 'application_date').distinct(),
            on=key_cols
        ).withColumn(
            'days_since_last_default',
            F.datediff(F.col('application_date'), F.col('last_default_date'))
        ).select(key_cols + ['days_since_last_default'])
        
        features = features.join(last_default, on=key_cols, how='outer')
        
        self._register_features([
            'oldest_credit_age_months', 'newest_credit_age_months',
            'avg_credit_age_months', 'avg_time_to_default_days',
            'days_since_last_default'
        ])
        
        return features
    
    def _generate_period_features(self, df: Any) -> Any:
        """Generate period-based features (3m, 6m, 12m, 24m)."""
        key_cols = ['application_id', 'customer_id']
        
        periods = [3, 6, 12, 24]
        credit_df = df.filter(F.col('is_credit_product'))
        
        features = None
        
        for months in periods:
            period_name = f'last_{months}m'
            
            # Credits opened in period
            period_df = credit_df.filter(F.col('credit_age_months') <= months)
            
            period_features = period_df.groupBy(key_cols).agg(
                F.count('*').alias(f'credits_opened_{period_name}'),
                F.sum('total_amount').alias(f'amount_opened_{period_name}'),
                F.sum(F.when(F.col('default_date').isNotNull(), 1).otherwise(0)).alias(f'defaults_{period_name}')
            )
            
            if features is None:
                features = period_features
            else:
                features = features.join(period_features, on=key_cols, how='outer')
            
            self._register_features([
                f'credits_opened_{period_name}',
                f'amount_opened_{period_name}',
                f'defaults_{period_name}'
            ])
        
        # Overdraft events in periods
        non_credit_df = df.filter(~F.col('is_credit_product'))
        
        for months in periods:
            period_name = f'last_{months}m'
            
            period_df = non_credit_df.filter(F.col('credit_age_months') <= months)
            
            overdraft_features = period_df.groupBy(key_cols).agg(
                F.count('*').alias(f'overdraft_events_{period_name}')
            )
            
            features = features.join(overdraft_features, on=key_cols, how='outer')
            
            self._register_features([f'overdraft_events_{period_name}'])
        
        return features
    
    def _generate_non_credit_features(self, df: Any) -> Any:
        """Generate non-credit product signal features."""
        key_cols = ['application_id', 'customer_id']
        
        non_credit_df = df.filter(~F.col('is_credit_product'))
        
        # Overdraft features
        overdraft_df = df.filter(F.col('product_type') == 'NON_AUTH_OVERDRAFT')
        overdraft_features = overdraft_df.groupBy(key_cols).agg(
            F.count('*').alias('non_auth_overdraft_count'),
            F.sum('total_amount').alias('non_auth_overdraft_amount')
        )
        
        # Overlimit features
        overlimit_df = df.filter(F.col('product_type') == 'OVERLIMIT')
        overlimit_features = overlimit_df.groupBy(key_cols).agg(
            F.count('*').alias('overlimit_count'),
            F.sum('total_amount').alias('overlimit_amount')
        )
        
        features = overdraft_features.join(overlimit_features, on=key_cols, how='outer')
        
        self._register_features([
            'non_auth_overdraft_count', 'non_auth_overdraft_amount',
            'overlimit_count', 'overlimit_amount'
        ])
        
        return features
    
    def _generate_derived_features(self, df: Any) -> Any:
        """Generate derived features (ratios, flags, trends)."""
        
        # Binary flags
        df = df.withColumn(
            'has_current_default',
            F.when(F.col('defaulted_credit_count') > 0, 1).otherwise(0)
        )
        df = df.withColumn(
            'has_ever_defaulted',
            F.when(F.col('ever_defaulted_count') > 0, 1).otherwise(0)
        )
        df = df.withColumn(
            'has_non_auth_overdraft',
            F.when(F.col('non_auth_overdraft_count') > 0, 1).otherwise(0)
        )
        df = df.withColumn(
            'has_overlimit',
            F.when(F.col('overlimit_count') > 0, 1).otherwise(0)
        )
        df = df.withColumn(
            'financial_stress_flag',
            F.when(
                (F.col('has_non_auth_overdraft') == 1) | 
                (F.col('has_overlimit') == 1), 1
            ).otherwise(0)
        )
        
        # Ratios
        df = df.withColumn(
            'default_ratio',
            F.when(F.col('total_credit_count') > 0,
                   F.col('defaulted_credit_count') / F.col('total_credit_count')
            ).otherwise(0)
        )
        df = df.withColumn(
            'recovery_ratio',
            F.when(F.col('ever_defaulted_count') > 0,
                   F.col('recovered_credit_count') / F.col('ever_defaulted_count')
            ).otherwise(0)
        )
        df = df.withColumn(
            'default_amount_ratio',
            F.when(F.col('total_credit_amount') > 0,
                   F.col('total_defaulted_amount') / F.col('total_credit_amount')
            ).otherwise(0)
        )
        
        # Product ratios
        for product in self.CREDIT_PRODUCTS:
            df = df.withColumn(
                f'{product}_ratio',
                F.when(F.col('total_credit_count') > 0,
                       F.col(f'{product}_count') / F.col('total_credit_count')
                ).otherwise(0)
            )
            df = df.withColumn(
                f'{product}_amount_ratio',
                F.when(F.col('total_credit_amount') > 0,
                       F.col(f'{product}_total_amount') / F.col('total_credit_amount')
                ).otherwise(0)
            )
        
        # Secured/unsecured ratios
        df = df.withColumn(
            'secured_ratio',
            F.when(F.col('total_credit_count') > 0,
                   F.col('MORTGAGE_count') / F.col('total_credit_count')
            ).otherwise(0)
        )
        df = df.withColumn(
            'revolving_ratio',
            F.when(F.col('total_credit_count') > 0,
                   F.col('CASH_FACILITY_count') / F.col('total_credit_count')
            ).otherwise(0)
        )
        df = df.withColumn(
            'high_risk_product_ratio',
            F.when(F.col('total_credit_count') > 0,
                   F.col('INSTALLMENT_SALE_count') / F.col('total_credit_count')
            ).otherwise(0)
        )
        
        # Velocity features
        df = df.withColumn(
            'credit_velocity',
            F.when(F.col('total_credit_count') > 0,
                   F.col('credits_opened_last_3m') / F.col('total_credit_count')
            ).otherwise(0)
        )
        df = df.withColumn(
            'amount_velocity',
            F.when(F.col('total_credit_amount') > 0,
                   F.col('amount_opened_last_3m') / F.col('total_credit_amount')
            ).otherwise(0)
        )
        
        # Trend features (6m vs previous 6m)
        df = df.withColumn(
            'credit_count_trend_6m_vs_12m',
            F.col('credits_opened_last_6m') - 
            (F.col('credits_opened_last_12m') - F.col('credits_opened_last_6m'))
        )
        df = df.withColumn(
            'credit_amount_trend_6m_vs_12m',
            F.col('amount_opened_last_6m') - 
            (F.col('amount_opened_last_12m') - F.col('amount_opened_last_6m'))
        )
        df = df.withColumn(
            'default_trend_6m_vs_12m',
            F.col('defaults_last_6m') - 
            (F.col('defaults_last_12m') - F.col('defaults_last_6m'))
        )
        
        # Diversity
        df = df.withColumn(
            'product_diversity_ratio',
            F.col('distinct_product_count') / 4.0  # 4 credit products
        )
        
        # Alias
        df = df.withColumn(
            'credit_history_length_months',
            F.col('oldest_credit_age_months')
        )
        
        self._register_features([
            'has_current_default', 'has_ever_defaulted', 'has_non_auth_overdraft',
            'has_overlimit', 'financial_stress_flag', 'default_ratio',
            'recovery_ratio', 'default_amount_ratio', 'secured_ratio',
            'revolving_ratio', 'high_risk_product_ratio', 'credit_velocity',
            'amount_velocity', 'credit_count_trend_6m_vs_12m',
            'credit_amount_trend_6m_vs_12m', 'default_trend_6m_vs_12m',
            'product_diversity_ratio', 'credit_history_length_months'
        ] + [f'{p}_ratio' for p in self.CREDIT_PRODUCTS] 
          + [f'{p}_amount_ratio' for p in self.CREDIT_PRODUCTS])
        
        return df
    
    def _register_features(self, features: List[str]) -> None:
        """Register generated features."""
        for f in features:
            self._generated_features.append({
                'name': f,
                'generated_at': datetime.now().isoformat()
            })
    
    @property
    def generated_features(self) -> List[Dict[str, Any]]:
        """Get list of generated features."""
        return self._generated_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [f['name'] for f in self._generated_features]
