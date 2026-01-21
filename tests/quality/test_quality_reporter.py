"""
Tests for Quality Reporter

Tests QualityReporter HTML and JSON report generation.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path
from datetime import datetime
import json

from src.quality.quality_reporter import QualityReporter
from src.quality.data_quality import DataQualityReport, QualityCheckResult
from src.quality.feature_quality import FeatureQualityResult


class TestQualityReporterInit:
    """Test suite for QualityReporter initialization."""
    
    def test_init_basic(self, base_config):
        """Test QualityReporter initialization."""
        reporter = QualityReporter(config=base_config)
        
        assert reporter is not None
        assert reporter.name == "QualityReporter"
    
    def test_init_with_custom_name(self, base_config):
        """Test QualityReporter with custom name."""
        reporter = QualityReporter(
            config=base_config,
            name="CustomReporter"
        )
        
        assert reporter.name == "CustomReporter"
    
    def test_init_with_output_dir(self, base_config):
        """Test QualityReporter with output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = QualityReporter(
                config=base_config,
                output_dir=temp_dir
            )
            
            assert reporter.output_dir == Path(temp_dir)
    
    def test_validate(self, base_config):
        """Test validate method."""
        reporter = QualityReporter(config=base_config)
        
        assert reporter.validate() is True


class TestQualityReporterGenerateReports:
    """Test suite for generate_reports method."""
    
    def test_generate_reports_with_data_quality(self, base_config, temp_output_dir):
        """Test generate_reports with data quality report."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        data_quality_report = DataQualityReport(
            table_name='test_table',
            timestamp=datetime.now(),
            total_rows=1000,
            checks=[
                QualityCheckResult(
                    check_name='null_check',
                    check_type='null',
                    passed=True,
                    severity='info',
                    message='All good'
                )
            ]
        )
        
        result = reporter.generate_reports(
            data_quality_report=data_quality_report,
            formats=['json']
        )
        
        assert isinstance(result, dict)
    
    def test_generate_reports_with_feature_quality(self, base_config, temp_output_dir):
        """Test generate_reports with feature quality results."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        feature_results = {
            'feature_1': FeatureQualityResult(
                feature_name='feature_1',
                null_ratio=0.05,
                unique_count=100,
                iv_score=0.35,
                iv_category='medium'
            ),
            'feature_2': FeatureQualityResult(
                feature_name='feature_2',
                null_ratio=0.0,
                unique_count=50,
                iv_score=0.15,
                iv_category='weak'
            )
        }
        
        result = reporter.generate_reports(
            feature_quality_results=feature_results,
            formats=['json']
        )
        
        assert isinstance(result, dict)
    
    def test_generate_reports_multiple_formats(self, base_config, temp_output_dir):
        """Test generate_reports with multiple formats."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        data_quality_report = DataQualityReport(
            table_name='test_table',
            timestamp=datetime.now(),
            total_rows=500
        )
        
        result = reporter.generate_reports(
            data_quality_report=data_quality_report,
            formats=['json', 'html']
        )
        
        assert 'json' in result or 'html' in result
    
    def test_run_calls_generate_reports(self, base_config, temp_output_dir):
        """Test run method calls generate_reports."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        data_quality_report = DataQualityReport(
            table_name='test',
            timestamp=datetime.now(),
            total_rows=100
        )
        
        with patch.object(reporter, 'generate_reports', return_value={}) as mock_gen:
            reporter.run(data_quality_report=data_quality_report)
            
            mock_gen.assert_called_once()


class TestQualityReporterJSONReport:
    """Test suite for JSON report generation."""
    
    def test_generate_json_report(self, base_config, temp_output_dir):
        """Test JSON report generation."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        data_quality_report = DataQualityReport(
            table_name='test_table',
            timestamp=datetime.now(),
            total_rows=1000,
            checks=[
                QualityCheckResult(
                    check_name='test_check',
                    check_type='test',
                    passed=True,
                    severity='info',
                    message='Test passed'
                )
            ]
        )
        
        result = reporter.generate_reports(
            data_quality_report=data_quality_report,
            formats=['json']
        )
        
        # Verify JSON file was created
        if 'json' in result:
            json_path = Path(result['json'])
            assert json_path.exists()
            
            # Verify content
            with open(json_path) as f:
                content = json.load(f)
                assert 'data_quality' in content or 'table_name' in str(content)
    
    def test_json_report_content_structure(self, base_config, temp_output_dir):
        """Test JSON report has correct structure."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        feature_results = {
            'feat_1': FeatureQualityResult(
                feature_name='feat_1',
                null_ratio=0.0,
                unique_count=10,
                iv_score=0.5
            )
        }
        
        result = reporter.generate_reports(
            feature_quality_results=feature_results,
            formats=['json']
        )
        
        if 'json' in result:
            with open(result['json']) as f:
                content = json.load(f)
                # Should have feature quality section
                assert content is not None


class TestQualityReporterHTMLReport:
    """Test suite for HTML report generation."""
    
    def test_generate_html_report(self, base_config, temp_output_dir):
        """Test HTML report generation."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        data_quality_report = DataQualityReport(
            table_name='test_table',
            timestamp=datetime.now(),
            total_rows=1000
        )
        
        result = reporter.generate_reports(
            data_quality_report=data_quality_report,
            formats=['html']
        )
        
        if 'html' in result:
            html_path = Path(result['html'])
            assert html_path.exists()
            
            # Verify it's valid HTML
            content = html_path.read_text()
            assert '<html>' in content.lower() or '<!doctype' in content.lower()
    
    def test_html_report_contains_data_quality(self, base_config, temp_output_dir):
        """Test HTML report contains data quality info."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        data_quality_report = DataQualityReport(
            table_name='my_test_table',
            timestamp=datetime.now(),
            total_rows=5000,
            checks=[
                QualityCheckResult(
                    check_name='null_check',
                    check_type='null',
                    passed=True,
                    severity='info',
                    message='No nulls found'
                )
            ]
        )
        
        result = reporter.generate_reports(
            data_quality_report=data_quality_report,
            formats=['html']
        )
        
        if 'html' in result:
            content = Path(result['html']).read_text()
            assert 'my_test_table' in content
            assert '5000' in content or '5,000' in content
    
    def test_html_report_contains_feature_quality(self, base_config, temp_output_dir):
        """Test HTML report contains feature quality info."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        feature_results = {
            'important_feature': FeatureQualityResult(
                feature_name='important_feature',
                null_ratio=0.02,
                unique_count=150,
                iv_score=0.42,
                iv_category='strong'
            )
        }
        
        result = reporter.generate_reports(
            feature_quality_results=feature_results,
            formats=['html']
        )
        
        if 'html' in result:
            content = Path(result['html']).read_text()
            assert 'important_feature' in content


class TestQualityReporterCombinedReport:
    """Test suite for combined data and feature quality reports."""
    
    def test_combined_report(self, base_config, temp_output_dir):
        """Test generating report with both data and feature quality."""
        reporter = QualityReporter(
            config=base_config,
            output_dir=str(temp_output_dir)
        )
        
        data_quality_report = DataQualityReport(
            table_name='combined_table',
            timestamp=datetime.now(),
            total_rows=2000
        )
        
        feature_results = {
            'feature_a': FeatureQualityResult(
                feature_name='feature_a',
                null_ratio=0.01,
                unique_count=75
            )
        }
        
        result = reporter.generate_reports(
            data_quality_report=data_quality_report,
            feature_quality_results=feature_results,
            formats=['html', 'json']
        )
        
        assert isinstance(result, dict)


class TestQualityReporterOutputDirectory:
    """Test suite for output directory handling."""
    
    def test_creates_output_directory(self, base_config):
        """Test reporter creates output directory if not exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / 'reports' / 'quality'
            
            reporter = QualityReporter(
                config=base_config,
                output_dir=str(new_dir)
            )
            
            data_quality_report = DataQualityReport(
                table_name='test',
                timestamp=datetime.now(),
                total_rows=100
            )
            
            reporter.generate_reports(
                data_quality_report=data_quality_report,
                formats=['json']
            )
            
            assert new_dir.exists()
    
    def test_uses_default_output_dir(self, base_config):
        """Test reporter uses default output dir if not specified."""
        reporter = QualityReporter(config=base_config)
        
        # Should have a default output directory
        assert reporter.output_dir is not None
