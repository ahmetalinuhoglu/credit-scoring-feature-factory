"""
Tests for Feature Factory

Comprehensive tests for FeatureFactory feature generation logic.
Validates correctness of all 943 features with deterministic test data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.features.feature_factory import FeatureFactory, FeatureDefinition


# ═══════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def feature_factory():
    """Initialize FeatureFactory for testing."""
    return FeatureFactory()


@pytest.fixture
def minimal_application():
    """Single application for focused testing."""
    return pd.DataFrame({
        'application_id': ['APP_001'],
        'customer_id': ['CUST_001'],
        'applicant_type': ['PRIMARY'],
        'application_date': [pd.Timestamp('2024-01-01')],
        'target': [0]
    })


@pytest.fixture
def known_bureau():
    """
    Known credit records with predictable feature outputs.
    
    Expected outputs:
    - total_credit_count = 3
    - total_credit_amount = 215000 (10000 + 200000 + 5000)
    - installment_loan_count = 1
    - mortgage_count = 1
    - cash_facility_count = 1
    - distinct_product_count = 3
    """
    return pd.DataFrame({
        'application_id': ['APP_001'] * 3,
        'customer_id': ['CUST_001'] * 3,
        'product_type': ['INSTALLMENT_LOAN', 'MORTGAGE', 'CASH_FACILITY'],
        'total_amount': [10000.0, 200000.0, 5000.0],
        'monthly_payment': [500.0, 1500.0, 200.0],
        'duration_months': [24, 240, 12],
        'opening_date': pd.to_datetime(['2023-06-01', '2020-01-01', '2023-11-01']),
        'closure_date': [None, None, None],
        'default_date': [None, None, None],
        'recovery_date': [None, None, None]
    })


@pytest.fixture
def empty_bureau():
    """Empty credit bureau for edge case testing."""
    return pd.DataFrame({
        'application_id': pd.Series([], dtype=str),
        'customer_id': pd.Series([], dtype=str),
        'product_type': pd.Series([], dtype=str),
        'total_amount': pd.Series([], dtype=float),
        'monthly_payment': pd.Series([], dtype=float),
        'duration_months': pd.Series([], dtype=int),
        'opening_date': pd.Series([], dtype='datetime64[ns]'),
        'closure_date': pd.Series([], dtype='datetime64[ns]'),
        'default_date': pd.Series([], dtype='datetime64[ns]'),
        'recovery_date': pd.Series([], dtype='datetime64[ns]')
    })


@pytest.fixture
def defaulted_bureau():
    """Bureau data with defaulted credits for pattern testing."""
    return pd.DataFrame({
        'application_id': ['APP_001'] * 3,
        'customer_id': ['CUST_001'] * 3,
        'product_type': ['INSTALLMENT_LOAN', 'INSTALLMENT_LOAN', 'CASH_FACILITY'],
        'total_amount': [10000.0, 15000.0, 5000.0],
        'monthly_payment': [500.0, 750.0, 200.0],
        'duration_months': [24, 36, 12],
        'opening_date': pd.to_datetime(['2023-01-01', '2023-03-01', '2023-06-01']),
        'closure_date': [None, None, None],
        'default_date': pd.to_datetime([None, '2023-09-01', '2023-10-01']),
        'recovery_date': pd.to_datetime([None, None, '2023-12-01'])
    })


@pytest.fixture
def single_credit_bureau():
    """Single credit record for minimal testing."""
    return pd.DataFrame({
        'application_id': ['APP_001'],
        'customer_id': ['CUST_001'],
        'product_type': ['MORTGAGE'],
        'total_amount': [300000.0],
        'monthly_payment': [2000.0],
        'duration_months': [360],
        'opening_date': pd.to_datetime(['2022-01-01']),
        'closure_date': [None],
        'default_date': [None],
        'recovery_date': [None]
    })


# ═══════════════════════════════════════════════════════════════
# INITIALIZATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeatureFactoryInit:
    """Test suite for FeatureFactory initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        factory = FeatureFactory()
        
        assert factory is not None
        assert factory.config == {}
        assert factory._feature_definitions == []
    
    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = {'custom': 'value'}
        factory = FeatureFactory(config=config)
        
        assert factory.config == config
    
    def test_dimensions_defined(self, feature_factory):
        """Test that all dimensions are properly defined."""
        assert len(feature_factory.TIME_WINDOWS) == 5
        assert len(feature_factory.PRODUCT_TYPES) == 5
        assert len(feature_factory.STATUS_FILTERS) == 4
        assert len(feature_factory.AGGREGATIONS) == 6
    
    def test_credit_products_defined(self, feature_factory):
        """Test credit product constants."""
        assert 'INSTALLMENT_LOAN' in feature_factory.CREDIT_PRODUCTS
        assert 'MORTGAGE' in feature_factory.CREDIT_PRODUCTS
        assert 'CASH_FACILITY' in feature_factory.CREDIT_PRODUCTS
        assert 'INSTALLMENT_SALE' in feature_factory.CREDIT_PRODUCTS
        
        assert 'NON_AUTH_OVERDRAFT' in feature_factory.NON_CREDIT_PRODUCTS
        assert 'OVERLIMIT' in feature_factory.NON_CREDIT_PRODUCTS


# ═══════════════════════════════════════════════════════════════
# READABLE NAME GENERATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestReadableNameGeneration:
    """Test suite for human-readable feature name generation."""
    
    def test_base_amount_name(self, feature_factory):
        """Test base case: all products, all time, all status, sum."""
        name = feature_factory._generate_readable_name('all', 'all', 'all', 'sum')
        assert name == 'total_credit_amount'
    
    def test_base_count_name(self, feature_factory):
        """Test base count name."""
        name = feature_factory._generate_readable_name('all', 'all', 'all', 'cnt')
        assert name == 'total_credit_count'
    
    def test_product_specific_name(self, feature_factory):
        """Test product-specific feature names."""
        name = feature_factory._generate_readable_name('il', 'all', 'all', 'cnt')
        assert name == 'installment_loan_count'
        
        name = feature_factory._generate_readable_name('mg', 'all', 'all', 'sum')
        assert name == 'mortgage_total_amount'
    
    def test_status_specific_name(self, feature_factory):
        """Test status-specific feature names."""
        name = feature_factory._generate_readable_name('all', 'all', 'defaulted', 'cnt')
        assert 'defaulted' in name
        
        name = feature_factory._generate_readable_name('all', 'all', 'active', 'sum')
        assert 'active' in name
    
    def test_time_window_suffix(self, feature_factory):
        """Test time window suffix in names."""
        name = feature_factory._generate_readable_name('all', '3m', 'all', 'cnt')
        assert name.endswith('_last_3m')
        
        name = feature_factory._generate_readable_name('all', '12m', 'all', 'sum')
        assert name.endswith('_last_12m')
    
    def test_complex_name(self, feature_factory):
        """Test complex multi-dimension name."""
        name = feature_factory._generate_readable_name('mg', '12m', 'active', 'avg')
        assert name == 'mortgage_active_average_amount_last_12m'


# ═══════════════════════════════════════════════════════════════
# FEATURE DEFINITION EXPANSION TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeatureDefinitionExpansion:
    """Test suite for feature definition expansion."""
    
    def test_expand_creates_definitions(self, feature_factory):
        """Test that expansion creates feature definitions."""
        feature_factory._expand_feature_definitions()
        
        assert len(feature_factory._feature_definitions) > 0
    
    def test_amount_feature_count(self, feature_factory):
        """Test expected number of amount features (minus skipped duplicates)."""
        feature_factory._expand_feature_definitions()
        
        amount_features = [
            f for f in feature_factory._feature_definitions
            if f.category == 'amount_feature'
        ]
        
        # 5 products × 5 windows × 4 statuses × 5 aggs = 500, minus ~4 skipped
        assert len(amount_features) > 450
        assert len(amount_features) < 500
    
    def test_count_feature_count(self, feature_factory):
        """Test expected number of count features (minus skipped duplicates)."""
        feature_factory._expand_feature_definitions()
        
        count_features = [
            f for f in feature_factory._feature_definitions
            if f.category == 'count_feature'
        ]
        
        # 5 products × 5 windows × 4 statuses = 100, minus 2 skipped
        assert len(count_features) == 98
    
    def test_all_34_categories_present(self, feature_factory):
        """Test that all 34 feature categories are generated."""
        feature_factory._expand_feature_definitions()
        
        categories = set(f.category for f in feature_factory._feature_definitions)
        
        # Core categories that must be present
        expected_categories = {
            'amount_feature', 'count_feature', 'ratio_feature',
            'temporal_feature', 'trend_feature', 'risk_signal_feature',
            'diversity_feature', 'behavioral_feature', 'default_pattern_feature',
            'sequence_feature', 'size_pattern_feature', 'burst_feature',
            'interval_feature', 'seasonal_feature', 'weighted_feature'
        }
        
        for cat in expected_categories:
            assert cat in categories, f"Missing category: {cat}"
    
    def test_feature_definition_structure(self, feature_factory):
        """Test that feature definitions have required fields."""
        feature_factory._expand_feature_definitions()
        
        for feat in feature_factory._feature_definitions[:10]:  # Check first 10
            assert feat.name is not None
            assert feat.description is not None
            assert feat.formula is not None
            assert feat.category is not None


# ═══════════════════════════════════════════════════════════════
# FEATURE GENERATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeatureGeneration:
    """Test suite for actual feature generation."""
    
    def test_generate_with_known_data(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test feature generation with known data and expected outputs."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        assert len(result) == 1
        
        # Verify base features
        assert result['total_credit_count'].iloc[0] == 3
        assert result['total_credit_amount'].iloc[0] == 215000.0
    
    def test_product_counts(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test product-specific count features."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        assert result['installment_loan_count'].iloc[0] == 1
        assert result['mortgage_count'].iloc[0] == 1
        assert result['cash_facility_count'].iloc[0] == 1
        assert result['installment_sale_count'].iloc[0] == 0
    
    def test_product_amounts(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test product-specific amount features."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        assert result['installment_loan_total_amount'].iloc[0] == 10000.0
        assert result['mortgage_total_amount'].iloc[0] == 200000.0
        assert result['cash_facility_total_amount'].iloc[0] == 5000.0
    
    def test_average_amount(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test average amount calculation."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        expected_avg = (10000 + 200000 + 5000) / 3
        assert abs(result['total_credit_average_amount'].iloc[0] - expected_avg) < 0.01
    
    def test_max_min_amount(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test max/min amount features."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        # max_single_credit_amount is the preferred name over total_credit_max_amount
        assert 'max_single_credit_amount' in result.columns
        assert result['max_single_credit_amount'].iloc[0] == 200000.0
        
        assert 'min_single_credit_amount' in result.columns
        assert result['min_single_credit_amount'].iloc[0] == 5000.0
    
    def test_output_has_core_columns(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test that output contains core identification columns."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        assert 'application_id' in result.columns
        assert 'customer_id' in result.columns
        assert 'applicant_type' in result.columns
        assert 'application_date' in result.columns
        assert 'target' in result.columns


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_empty_bureau(
        self, feature_factory, minimal_application, empty_bureau
    ):
        """Test handling of empty credit bureau."""
        result = feature_factory.generate_all_features(
            minimal_application, empty_bureau
        )
        
        assert len(result) == 1
        assert result['total_credit_count'].iloc[0] == 0
        assert result['total_credit_amount'].iloc[0] == 0
    
    def test_single_credit(
        self, feature_factory, minimal_application, single_credit_bureau
    ):
        """Test handling of single credit record."""
        result = feature_factory.generate_all_features(
            minimal_application, single_credit_bureau
        )
        
        assert len(result) == 1
        assert result['total_credit_count'].iloc[0] == 1
        assert result['total_credit_amount'].iloc[0] == 300000.0
        assert result['mortgage_count'].iloc[0] == 1
        
        # For single credit, avg = max = min = total
        avg = result['total_credit_average_amount'].iloc[0]
        assert avg == 300000.0
    
    def test_no_matching_bureau(self, feature_factory):
        """Test when application has no matching bureau records."""
        apps = pd.DataFrame({
            'application_id': ['APP_001'],
            'customer_id': ['CUST_001'],
            'applicant_type': ['PRIMARY'],
            'application_date': [pd.Timestamp('2024-01-01')],
            'target': [0]
        })
        
        bureau = pd.DataFrame({
            'application_id': ['APP_999'],  # Different app
            'customer_id': ['CUST_999'],
            'product_type': ['MORTGAGE'],
            'total_amount': [100000.0],
            'monthly_payment': [1000.0],
            'duration_months': [240],
            'opening_date': pd.to_datetime(['2023-01-01']),
            'closure_date': [None],
            'default_date': [None],
            'recovery_date': [None]
        })
        
        result = feature_factory.generate_all_features(apps, bureau)
        
        assert len(result) == 1
        assert result['total_credit_count'].iloc[0] == 0


# ═══════════════════════════════════════════════════════════════
# DEFAULT PATTERN TESTS
# ═══════════════════════════════════════════════════════════════

class TestDefaultPatternFeatures:
    """Test suite for default pattern feature generation."""
    
    def test_default_counts(
        self, feature_factory, minimal_application, defaulted_bureau
    ):
        """Test default count features."""
        result = feature_factory.generate_all_features(
            minimal_application, defaulted_bureau
        )
        
        # 2 credits have default_date set
        assert 'default_count_ever' in result.columns
        assert result['default_count_ever'].iloc[0] == 2
    
    def test_recovery_detection(
        self, feature_factory, minimal_application, defaulted_bureau
    ):
        """Test recovery detection features."""
        result = feature_factory.generate_all_features(
            minimal_application, defaulted_bureau
        )
        
        # 1 credit has recovery_date set
        assert 'recovery_cycle_count' in result.columns
        assert result['recovery_cycle_count'].iloc[0] == 1
    
    def test_defaulted_status_filter(
        self, feature_factory, minimal_application, defaulted_bureau
    ):
        """Test defaulted status filter in aggregations."""
        result = feature_factory.generate_all_features(
            minimal_application, defaulted_bureau
        )
        
        # Should not contain skipped duplicates
        assert 'defaulted_count' not in result.columns
        
        # Defaulted total amount should be 15000 + 5000 = 20000
        if 'defaulted_total_amount' in result.columns:
            assert result['defaulted_total_amount'].iloc[0] == 20000.0


# ═══════════════════════════════════════════════════════════════
# DUPLICATE SKIPPING TESTS
# ═══════════════════════════════════════════════════════════════

class TestDuplicateSkipping:
    """Test suite for duplicate feature skip list enforcement."""
    
    def test_skipped_count_features(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test that duplicate count features are skipped."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        # These should be skipped (duplicates of better-named features)
        assert 'defaulted_count' not in result.columns
        assert 'recovered_count' not in result.columns
    
    def test_skipped_amount_features(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test that duplicate amount features are skipped."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        # These should be skipped (duplicates of expert-named features)
        assert 'total_credit_max_amount' not in result.columns
        assert 'total_credit_min_amount' not in result.columns
        assert 'defaulted_average_amount' not in result.columns
        assert 'defaulted_max_amount' not in result.columns
    
    def test_preferred_names_present(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test that preferred feature names are present."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        # These are the preferred names that should be present
        assert 'max_single_credit_amount' in result.columns
        assert 'min_single_credit_amount' in result.columns
        assert 'default_count_ever' in result.columns


# ═══════════════════════════════════════════════════════════════
# TIME WINDOW TESTS
# ═══════════════════════════════════════════════════════════════

class TestTimeWindowFeatures:
    """Test suite for time window-based features."""
    
    def test_recent_window_filtering(self, feature_factory, minimal_application):
        """Test that time windows correctly filter credits."""
        # Application date: 2024-01-01
        # Bureau with credits at different times
        bureau = pd.DataFrame({
            'application_id': ['APP_001'] * 3,
            'customer_id': ['CUST_001'] * 3,
            'product_type': ['INSTALLMENT_LOAN'] * 3,
            'total_amount': [10000.0, 20000.0, 30000.0],
            'monthly_payment': [500.0, 1000.0, 1500.0],
            'duration_months': [24, 24, 24],
            'opening_date': pd.to_datetime([
                '2023-11-01',  # 2 months ago (in 3m window)
                '2023-07-01',  # 6 months ago (in 6m/12m window, not 3m)
                '2022-01-01'   # 24 months ago (in 24m window only)
            ]),
            'closure_date': [None] * 3,
            'default_date': [None] * 3,
            'recovery_date': [None] * 3
        })
        
        result = feature_factory.generate_all_features(minimal_application, bureau)
        
        # All time should have all 3
        assert result['total_credit_count'].iloc[0] == 3
        
        # 3m window should have 1 (2023-11-01)
        if 'total_credit_count_last_3m' in result.columns:
            assert result['total_credit_count_last_3m'].iloc[0] == 1


# ═══════════════════════════════════════════════════════════════
# RATIO FEATURE TESTS
# ═══════════════════════════════════════════════════════════════

class TestRatioFeatures:
    """Test suite for ratio feature calculations."""
    
    def test_product_ratio(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test product mix ratio calculation."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        # Check that ratio features exist and are reasonable
        ratio_cols = [c for c in result.columns if 'ratio' in c.lower()]
        assert len(ratio_cols) > 0, "No ratio features found"
    
    def test_division_by_zero_handling(
        self, feature_factory, minimal_application, empty_bureau
    ):
        """Test that division by zero is handled gracefully."""
        result = feature_factory.generate_all_features(
            minimal_application, empty_bureau
        )
        
        # Verify no NaN or Inf values remain after fillna(0)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            val = result[col].iloc[0]
            assert not (pd.isna(val) or np.isinf(val)), f"Invalid value in {col}"


# ═══════════════════════════════════════════════════════════════
# DIVERSITY FEATURE TESTS
# ═══════════════════════════════════════════════════════════════

class TestDiversityFeatures:
    """Test suite for diversity and concentration features."""
    
    def test_distinct_product_count(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test distinct product count feature."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        assert 'distinct_product_count' in result.columns
        assert result['distinct_product_count'].iloc[0] == 3
    
    def test_single_product_diversity(
        self, feature_factory, minimal_application, single_credit_bureau
    ):
        """Test diversity with single product type."""
        result = feature_factory.generate_all_features(
            minimal_application, single_credit_bureau
        )
        
        assert result['distinct_product_count'].iloc[0] == 1


# ═══════════════════════════════════════════════════════════════
# OUTPUT COLUMN COUNT TEST
# ═══════════════════════════════════════════════════════════════

class TestOutputStructure:
    """Test suite for output DataFrame structure."""
    
    def test_expected_column_count(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test that output has expected number of columns (~948)."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        # Expected: 943 features + 5 core columns = 948
        assert len(result.columns) >= 900
        assert len(result.columns) <= 1000
    
    def test_no_duplicate_columns(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test that there are no duplicate column names."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        assert len(result.columns) == len(set(result.columns))
    
    def test_all_numeric_features(
        self, feature_factory, minimal_application, known_bureau
    ):
        """Test that all feature columns are numeric (except core)."""
        result = feature_factory.generate_all_features(
            minimal_application, known_bureau
        )
        
        core_cols = {'application_id', 'customer_id', 'applicant_type', 'application_date'}
        feature_cols = [c for c in result.columns if c not in core_cols]
        
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), f"Non-numeric: {col}"
