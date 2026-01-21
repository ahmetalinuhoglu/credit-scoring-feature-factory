"""
Quality Reporter

Generates quality reports in various formats.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import json

from src.core.base import PandasComponent
from src.quality.data_quality import DataQualityReport
from src.quality.feature_quality import FeatureQualityResult


class QualityReporter(PandasComponent):
    """
    Generates quality reports in HTML, JSON, and other formats.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
        name: Optional[str] = None
    ):
        """
        Initialize quality reporter.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for reports
            name: Optional reporter name
        """
        super().__init__(config, name or "QualityReporter")
        
        self.output_dir = Path(
            output_dir or 
            self.get_config('quality.reporting.output_path', 'outputs/quality_reports')
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate(self) -> bool:
        return True
    
    def run(
        self,
        data_quality_report: Optional[DataQualityReport] = None,
        feature_quality_results: Optional[Dict[str, FeatureQualityResult]] = None
    ) -> Dict[str, str]:
        """Generate all reports."""
        return self.generate_reports(data_quality_report, feature_quality_results)
    
    def generate_reports(
        self,
        data_quality_report: Optional[DataQualityReport] = None,
        feature_quality_results: Optional[Dict[str, FeatureQualityResult]] = None,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate quality reports in specified formats.
        
        Args:
            data_quality_report: Data quality report
            feature_quality_results: Feature quality results
            formats: Output formats (html, json)
            
        Returns:
            Dictionary of format to file path
        """
        self._start_execution()
        
        formats = formats or self.get_config(
            'quality.reporting.formats', 
            ['html', 'json']
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_files = {}
        
        # Generate JSON report
        if 'json' in formats:
            json_path = self._generate_json_report(
                data_quality_report,
                feature_quality_results,
                timestamp
            )
            output_files['json'] = str(json_path)
        
        # Generate HTML report
        if 'html' in formats:
            html_path = self._generate_html_report(
                data_quality_report,
                feature_quality_results,
                timestamp
            )
            output_files['html'] = str(html_path)
        
        self._end_execution()
        self.logger.info(f"Generated reports: {list(output_files.keys())}")
        
        return output_files
    
    def _generate_json_report(
        self,
        data_quality_report: Optional[DataQualityReport],
        feature_quality_results: Optional[Dict[str, FeatureQualityResult]],
        timestamp: str
    ) -> Path:
        """Generate JSON format report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_quality': None,
            'feature_quality': None
        }
        
        if data_quality_report:
            report['data_quality'] = data_quality_report.to_dict()
        
        if feature_quality_results:
            report['feature_quality'] = {
                name: result.to_dict() 
                for name, result in feature_quality_results.items()
            }
        
        output_path = self.output_dir / f"quality_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path
    
    def _generate_html_report(
        self,
        data_quality_report: Optional[DataQualityReport],
        feature_quality_results: Optional[Dict[str, FeatureQualityResult]],
        timestamp: str
    ) -> Path:
        """Generate HTML format report."""
        html_content = self._build_html_report(
            data_quality_report,
            feature_quality_results
        )
        
        output_path = self.output_dir / f"quality_report_{timestamp}.html"
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _build_html_report(
        self,
        data_quality_report: Optional[DataQualityReport],
        feature_quality_results: Optional[Dict[str, FeatureQualityResult]]
    ) -> str:
        """Build HTML report content."""
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #666; margin-top: 30px; }
        .summary { display: flex; gap: 20px; margin: 20px 0; }
        .card { background: #f9f9f9; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }
        .card.success { border-left: 4px solid #4CAF50; }
        .card.error { border-left: 4px solid #f44336; }
        .card.warning { border-left: 4px solid #ff9800; }
        .card h3 { margin: 0; font-size: 2em; }
        .card p { margin: 5px 0 0; color: #666; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; font-weight: bold; }
        tr:hover { background: #f9f9f9; }
        .pass { color: #4CAF50; font-weight: bold; }
        .fail { color: #f44336; font-weight: bold; }
        .iv-useless { color: #999; }
        .iv-weak { color: #ff9800; }
        .iv-medium { color: #2196F3; }
        .iv-strong { color: #4CAF50; }
        .iv-suspicious { color: #f44336; }
        .timestamp { color: #999; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Data Quality Report</h1>
        <p class="timestamp">Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
"""
        
        # Data Quality Section
        if data_quality_report:
            status_class = 'success' if data_quality_report.passed else 'error'
            status_text = '✅ PASSED' if data_quality_report.passed else '❌ FAILED'
            
            html += f"""
        <h2>Data Quality - {data_quality_report.table_name}</h2>
        <div class="summary">
            <div class="card {status_class}">
                <h3>{status_text}</h3>
                <p>Overall Status</p>
            </div>
            <div class="card">
                <h3>{data_quality_report.total_rows:,}</h3>
                <p>Total Rows</p>
            </div>
            <div class="card error">
                <h3>{data_quality_report.error_count}</h3>
                <p>Errors</p>
            </div>
            <div class="card warning">
                <h3>{data_quality_report.warning_count}</h3>
                <p>Warnings</p>
            </div>
        </div>
        
        <table>
            <tr>
                <th>Check</th>
                <th>Type</th>
                <th>Status</th>
                <th>Severity</th>
                <th>Message</th>
            </tr>
"""
            for check in data_quality_report.checks:
                status = '<span class="pass">✓ Pass</span>' if check.passed else '<span class="fail">✗ Fail</span>'
                html += f"""
            <tr>
                <td>{check.check_name}</td>
                <td>{check.check_type}</td>
                <td>{status}</td>
                <td>{check.severity}</td>
                <td>{check.message}</td>
            </tr>
"""
            html += "</table>"
        
        # Feature Quality Section
        if feature_quality_results:
            html += """
        <h2>Feature Quality Analysis</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Null Ratio</th>
                <th>Unique Values</th>
                <th>IV Score</th>
                <th>IV Category</th>
                <th>Variance</th>
            </tr>
"""
            # Sort by IV score descending
            sorted_features = sorted(
                feature_quality_results.items(),
                key=lambda x: x[1].iv_score or 0,
                reverse=True
            )
            
            for name, result in sorted_features:
                iv_class = f"iv-{result.iv_category}" if result.iv_category else ""
                iv_score = f"{result.iv_score:.4f}" if result.iv_score else "N/A"
                variance = f"{result.variance:.4f}" if result.variance else "N/A"
                
                html += f"""
            <tr>
                <td>{name}</td>
                <td>{result.null_ratio:.2%}</td>
                <td>{result.unique_count:,}</td>
                <td>{iv_score}</td>
                <td class="{iv_class}">{result.iv_category or 'N/A'}</td>
                <td>{variance}</td>
            </tr>
"""
            html += "</table>"
        
        html += """
    </div>
</body>
</html>
"""
        return html
