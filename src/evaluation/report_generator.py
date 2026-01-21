"""
Report Generator

Generates comprehensive evaluation reports in HTML format.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import json

import pandas as pd

from src.core.base import PandasComponent


class ReportGenerator(PandasComponent):
    """
    Generates HTML evaluation reports.
    
    Creates comprehensive reports including:
    - Model performance summary
    - Metric comparisons
    - Lift charts
    - Feature importance
    - ROC curves
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
        name: Optional[str] = None
    ):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for reports
            name: Optional generator name
        """
        super().__init__(config, name or "ReportGenerator")
        
        self.output_dir = Path(
            output_dir or
            self.get_config('evaluation.reports.output_path', 'outputs/reports')
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate(self) -> bool:
        return True
    
    def run(
        self,
        evaluation_results: Dict[str, Any],
        **kwargs
    ) -> str:
        """Generate report."""
        return self.generate(evaluation_results, **kwargs)
    
    def generate(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        comparison_df: Optional[pd.DataFrame] = None,
        feature_importances: Optional[Dict[str, Dict[str, float]]] = None,
        title: str = "Model Evaluation Report"
    ) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            evaluation_results: Dictionary of model name to results
            comparison_df: Model comparison DataFrame
            feature_importances: Feature importances per model
            title: Report title
            
        Returns:
            Path to generated report
        """
        self._start_execution()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'evaluation_report_{timestamp}.html'
        
        html = self._build_html(
            evaluation_results,
            comparison_df,
            feature_importances,
            title
        )
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"Report generated: {report_path}")
        
        self._end_execution()
        
        return str(report_path)
    
    def _build_html(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        comparison_df: Optional[pd.DataFrame],
        feature_importances: Optional[Dict[str, Dict[str, float]]],
        title: str
    ) -> str:
        """Build HTML content."""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 15px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        .card {{ background: white; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .metric-card.success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .metric-card.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        .best {{ background: #d4edda !important; font-weight: bold; }}
        .importance-bar {{ background: #e9ecef; border-radius: 4px; overflow: hidden; height: 20px; }}
        .importance-fill {{ background: linear-gradient(90deg, #1a73e8, #34a853); height: 100%; border-radius: 4px; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        .section-divider {{ border-top: 2px solid #e0e0e0; margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        # Model Comparison Section
        if comparison_df is not None:
            html += """
        <div class="card">
            <h2>🏆 Model Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Gini</th>
                    <th>KS</th>
                    <th>AUC</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                </tr>
"""
            best_gini = comparison_df['gini'].max()
            
            for _, row in comparison_df.iterrows():
                is_best = row['gini'] == best_gini
                row_class = 'best' if is_best else ''
                
                html += f"""
                <tr class="{row_class}">
                    <td>{row['model']}</td>
                    <td>{row['gini']:.4f}</td>
                    <td>{row['ks_statistic']:.4f}</td>
                    <td>{row['auc']:.4f}</td>
                    <td>{row['precision']:.4f}</td>
                    <td>{row['recall']:.4f}</td>
                    <td>{row['f1_score']:.4f}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""
        
        # Per-Model Details
        for model_name, result in evaluation_results.items():
            metrics = result['metrics']
            
            html += f"""
        <div class="card">
            <h2>📈 {model_name}</h2>
            <p><strong>Dataset:</strong> {result['dataset']} | 
               <strong>Samples:</strong> {result['sample_size']:,} | 
               <strong>Positive Rate:</strong> {result['positive_ratio']:.2%}</p>
            
            <div class="metric-grid">
                <div class="metric-card success">
                    <div class="metric-value">{metrics['gini']:.3f}</div>
                    <div class="metric-label">Gini</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['ks_statistic']:.3f}</div>
                    <div class="metric-label">KS Statistic</div>
                </div>
                <div class="metric-card success">
                    <div class="metric-value">{metrics['auc']:.3f}</div>
                    <div class="metric-label">AUC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['accuracy']:.3f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
            </div>
            
            <h3>Confusion Matrix</h3>
            <table style="width: 300px;">
                <tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>
                <tr><th>Actual 0</th><td>{metrics['confusion_matrix']['tn']:,}</td><td>{metrics['confusion_matrix']['fp']:,}</td></tr>
                <tr><th>Actual 1</th><td>{metrics['confusion_matrix']['fn']:,}</td><td>{metrics['confusion_matrix']['tp']:,}</td></tr>
            </table>
"""
            
            # Lift Table
            if 'lift_table' in result:
                lift_df = result['lift_table']
                html += """
            <h3>Lift Table (Decile Analysis)</h3>
            <table>
                <tr>
                    <th>Decile</th>
                    <th>Count</th>
                    <th>Bads</th>
                    <th>Bad Rate</th>
                    <th>Lift</th>
                    <th>Cum Lift</th>
                    <th>Capture Rate</th>
                </tr>
"""
                for _, row in lift_df.iterrows():
                    html += f"""
                <tr>
                    <td>{int(row['decile'])}</td>
                    <td>{int(row['count']):,}</td>
                    <td>{int(row['bads']):,}</td>
                    <td>{row['bad_rate']:.2%}</td>
                    <td>{row['lift']:.2f}x</td>
                    <td>{row['cum_lift']:.2f}x</td>
                    <td>{row['capture_rate']:.1%}</td>
                </tr>
"""
                html += """
            </table>
"""
            
            html += """
        </div>
"""
        
        # Feature Importance Section
        if feature_importances:
            html += """
        <div class="card">
            <h2>🔑 Feature Importance (Top 20)</h2>
"""
            for model_name, importances in feature_importances.items():
                html += f"""
            <h3>{model_name}</h3>
            <table>
                <tr><th>Feature</th><th>Importance</th><th></th></tr>
"""
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
                max_importance = max(v for _, v in sorted_features) if sorted_features else 1
                
                for feature, importance in sorted_features:
                    pct = (importance / max_importance) * 100
                    html += f"""
                <tr>
                    <td><code>{feature}</code></td>
                    <td>{importance:.4f}</td>
                    <td style="width: 200px;">
                        <div class="importance-bar">
                            <div class="importance-fill" style="width: {pct}%;"></div>
                        </div>
                    </td>
                </tr>
"""
                html += """
            </table>
"""
            html += """
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        return html
