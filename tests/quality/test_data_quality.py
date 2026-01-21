"""
Tests for Data Quality Checker

Tests DataQualityChecker for data quality analysis.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.quality.data_quality import DataQualityChecker, QualityCheckResult, DataQualityReport


class TestQualityCheckResult:
    """Test suite for QualityCheckResult dataclass."""
    
    def test_result_creation(self):
        """Test QualityCheckResult creation."""
        result = QualityCheckResult(
            check_name='null_check',
            check_type='null_check',
            passed=True,
            severity='info',
            message='Check passed',
            details={'null_ratio': 0.01}
        )
        
        assert result.check_name == 'null_check'
        assert result.passed is True
        assert result.details['null_ratio'] == 0.01
    
    def test_result_with_failed(self):
        """Test QualityCheckResult for failed check."""
        result = QualityCheckResult(
            check_name='range_check',
            check_type='range_check',
            passed=False,
            severity='error',
            message='Values out of range',
            details={'out_of_range': 50}
        )
        
        assert result.passed is False
        assert 'out of range' in result.message.lower()
    
    def test_result_to_dict(self):
        """Test QualityCheckResult to_dict method."""
        result = QualityCheckResult(
            check_name='test',
            check_type='test_type',
            passed=True,
            severity='warning',
            message='Test message'
        )
        
        d = result.to_dict()
        
        assert d['check_name'] == 'test'
        assert d['passed'] is True
        assert d['severity'] == 'warning'


class TestDataQualityReport:
    """Test suite for DataQualityReport dataclass."""
    
    def test_report_creation(self):
        """Test DataQualityReport creation."""
        check1 = QualityCheckResult('c1', 'null_check', True, 'error', 'OK')
        check2 = QualityCheckResult('c2', 'range_check', False, 'warning', 'Failed')
        
        report = DataQualityReport(
            table_name='test_table',
            timestamp=datetime.now(),
            total_rows=1000,
            checks=[check1, check2]
        )
        
        assert report.table_name == 'test_table'
        assert len(report.checks) == 2
        assert report.total_rows == 1000
    
    def test_report_passed_property(self):
        """Test report passed property."""
        check1 = QualityCheckResult('c1', 't', True, 'error', 'OK')
        check2 = QualityCheckResult('c2', 't', True, 'error', 'OK')
        
        report = DataQualityReport(
            table_name='test',
            timestamp=datetime.now(),
            total_rows=100,
            checks=[check1, check2]
        )
        
        assert report.passed is True
    
    def test_report_failed_on_error(self):
        """Test report failed when error check fails."""
        check1 = QualityCheckResult('c1', 't', False, 'error', 'Failed')
        check2 = QualityCheckResult('c2', 't', True, 'warning', 'OK')
        
        report = DataQualityReport(
            table_name='test',
            timestamp=datetime.now(),
            total_rows=100,
            checks=[check1, check2]
        )
        
        assert report.passed is False
        assert report.error_count == 1


class TestDataQualityCheckerInit:
    """Test suite for DataQualityChecker initialization."""
    
    def test_init_basic(self, base_config, mock_spark_session):
        """Test DataQualityChecker initialization."""
        checker = DataQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker is not None
        assert checker.name == "DataQualityChecker"
    
    def test_init_with_custom_name(self, base_config, mock_spark_session):
        """Test DataQualityChecker with custom name."""
        checker = DataQualityChecker(
            config=base_config,
            spark_session=mock_spark_session,
            name="CustomChecker"
        )
        
        assert checker.name == "CustomChecker"
    
    def test_validate(self, base_config, mock_spark_session):
        """Test validate method."""
        checker = DataQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert checker.validate() is True


class TestDataQualityCheckerRun:
    """Test suite for run method."""
    
    def test_run_calls_check_all(self, base_config, mock_spark_session):
        """Test run method calls check_all."""
        checker = DataQualityChecker(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        mock_df = MagicMock()
        
        with patch.object(checker, 'check_all') as mock_check:
            mock_report = DataQualityReport(
                table_name='test',
                timestamp=datetime.now(),
                total_rows=100,
                checks=[]
            )
            mock_check.return_value = mock_report
            
            checker.run(mock_df, 'test_table')
            
            mock_check.assert_called_once()
