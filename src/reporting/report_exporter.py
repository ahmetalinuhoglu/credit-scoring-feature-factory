"""
Report Exporter

Generates Excel tables and PDF/PNG charts for pipeline stage outputs.
Optimized for executive presentations and business reporting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import seaborn as sns

from src.core.base import PandasComponent


# Set consistent styling for all charts
plt.style.use('seaborn-v0_8-whitegrid')
CHART_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'gradient': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
}


class ReportExporter(PandasComponent):
    """
    Exports pipeline stage results as Excel tables and PDF/PNG charts.
    
    Output Formats:
    - Excel (.xlsx): All tabular data with formatted sheets
    - PDF: Vector charts for high-quality printing
    - PNG: Raster charts for easy copy-paste into slides
    
    Chart Settings:
    - PDF: Vector (infinite resolution)
    - PNG: 300 DPI for crisp display
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
        name: Optional[str] = None
    ):
        """
        Initialize report exporter.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for reports
            name: Optional exporter name
        """
        super().__init__(config, name or "ReportExporter")
        
        self.output_dir = Path(output_dir or self.get_config(
            'outputs.base_path', 'outputs'
        )) / 'reports'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart settings
        self.dpi = 300
        self.figsize = (10, 6)
        self.title_fontsize = 14
        self.label_fontsize = 12
        
        self.logger.info(f"Report outputs will be saved to: {self.output_dir}")
    
    def validate(self) -> bool:
        return True
    
    def run(self, results: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Generate all reports."""
        return self.generate_all_reports(results)
    
    # ═══════════════════════════════════════════════════════════════
    # EXCEL GENERATORS
    # ═══════════════════════════════════════════════════════════════
    
    def generate_data_summary_excel(
        self,
        applications_stats: Dict[str, Any],
        credit_bureau_stats: Dict[str, Any],
        filename: str = "01_data_summary.xlsx"
    ) -> str:
        """Generate data summary Excel file."""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Applications summary
            apps_df = pd.DataFrame([{
                'Metric': 'Total Records',
                'Value': applications_stats.get('total_rows', 0)
            }, {
                'Metric': 'Unique Applications',
                'Value': applications_stats.get('unique_applications', 0)
            }, {
                'Metric': 'Unique Customers',
                'Value': applications_stats.get('unique_customers', 0)
            }, {
                'Metric': 'Joint Application Rate',
                'Value': f"{applications_stats.get('joint_rate', 0)*100:.1f}%"
            }, {
                'Metric': 'Target Rate (Default)',
                'Value': f"{applications_stats.get('target_rate', 0)*100:.2f}%"
            }])
            apps_df.to_excel(writer, sheet_name='Applications Summary', index=False)
            
            # Credit bureau summary
            if credit_bureau_stats:
                cb_df = pd.DataFrame([{
                    'Metric': 'Total Credit Records',
                    'Value': credit_bureau_stats.get('total_rows', 0)
                }, {
                    'Metric': 'Avg Credits per Customer',
                    'Value': f"{credit_bureau_stats.get('avg_credits_per_customer', 0):.1f}"
                }])
                cb_df.to_excel(writer, sheet_name='Credit Bureau Summary', index=False)
                
                # Product distribution
                if 'product_distribution' in credit_bureau_stats:
                    prod_df = pd.DataFrame(credit_bureau_stats['product_distribution'])
                    prod_df.to_excel(writer, sheet_name='Product Distribution', index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    def generate_feature_stats_excel(
        self,
        feature_stats: pd.DataFrame,
        filename: str = "02_feature_statistics.xlsx"
    ) -> str:
        """Generate feature statistics Excel file."""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            feature_stats.to_excel(writer, sheet_name='All Features', index=False)
            
            # Add category-wise sheets
            if 'category' in feature_stats.columns:
                for category in feature_stats['category'].unique():
                    cat_df = feature_stats[feature_stats['category'] == category]
                    sheet_name = category[:31]  # Excel sheet name limit
                    cat_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    def generate_quality_report_excel(
        self,
        quality_results: List[Dict[str, Any]],
        filename: str = "03_data_quality.xlsx"
    ) -> str:
        """Generate data quality report Excel file."""
        filepath = self.output_dir / filename
        
        # Convert to DataFrame
        df = pd.DataFrame(quality_results)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Full report
            df.to_excel(writer, sheet_name='All Checks', index=False)
            
            # Summary by severity
            if 'severity' in df.columns:
                summary = df.groupby('severity').agg({
                    'passed': ['sum', 'count']
                }).reset_index()
                summary.columns = ['Severity', 'Passed', 'Total']
                summary['Failed'] = summary['Total'] - summary['Passed']
                summary['Pass Rate'] = (summary['Passed'] / summary['Total'] * 100).round(1).astype(str) + '%'
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Failed checks only
            if 'passed' in df.columns:
                failed = df[df['passed'] == False]
                if len(failed) > 0:
                    failed.to_excel(writer, sheet_name='Failed Checks', index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    def generate_univariate_excel(
        self,
        univariate_results: Dict[str, Dict[str, Any]],
        filename: str = "04_univariate_analysis.xlsx"
    ) -> str:
        """Generate univariate analysis Excel file."""
        filepath = self.output_dir / filename
        
        # Convert to DataFrame
        records = []
        for feature, stats in univariate_results.items():
            record = {'feature': feature}
            record.update(stats)
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Sort by IV score
        if 'iv_score' in df.columns:
            df = df.sort_values('iv_score', ascending=False)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Univariate Analysis', index=False)
            
            # Recommended features
            if 'is_recommended' in df.columns:
                recommended = df[df['is_recommended'] == True]
                recommended.to_excel(writer, sheet_name='Recommended', index=False)
                
                rejected = df[df['is_recommended'] == False]
                if len(rejected) > 0:
                    rejected.to_excel(writer, sheet_name='Rejected', index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    def generate_woe_binning_excel(
        self,
        woe_results: Dict[str, Dict[str, Any]],
        filename: str = "05_woe_binning.xlsx"
    ) -> str:
        """Generate WoE binning Excel file (ideal for scorecards)."""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_records = []
            for feature, result in woe_results.items():
                summary_records.append({
                    'Feature': feature,
                    'IV Score': result.get('iv_score', 0),
                    'IV Category': result.get('iv_category', ''),
                    'Is Monotonic': result.get('is_monotonic', False),
                    'Num Bins': len(result.get('bins', []))
                })
            
            summary_df = pd.DataFrame(summary_records)
            summary_df = summary_df.sort_values('IV Score', ascending=False)
            summary_df.to_excel(writer, sheet_name='IV Summary', index=False)
            
            # Detailed bins per feature
            for feature, result in woe_results.items():
                if 'bins' in result:
                    bins_df = pd.DataFrame(result['bins'])
                    sheet_name = feature[:31]  # Excel limit
                    bins_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    def generate_model_metrics_excel(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        filename: str = "06_model_metrics.xlsx"
    ) -> str:
        """Generate model metrics comparison Excel file."""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Comparison table
            comparison_records = []
            for model_name, results in evaluation_results.items():
                metrics = results.get('metrics', {})
                record = {
                    'Model': model_name,
                    'Dataset': results.get('dataset', 'test'),
                    'Gini': metrics.get('gini', 0),
                    'KS Statistic': metrics.get('ks_statistic', 0),
                    'AUC': metrics.get('auc', 0),
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1 Score': metrics.get('f1_score', 0),
                    'Brier Score': metrics.get('brier_score', 0),
                    'Log Loss': metrics.get('log_loss', 0)
                }
                comparison_records.append(record)
            
            comparison_df = pd.DataFrame(comparison_records)
            comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
            
            # Lift tables per model
            for model_name, results in evaluation_results.items():
                if 'lift_table' in results:
                    lift_df = pd.DataFrame(results['lift_table'])
                    sheet_name = f"Lift_{model_name[:25]}"
                    lift_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    def generate_scorecard_excel(
        self,
        scorecard_data: Dict[str, List[Dict[str, Any]]],
        score_stats: Dict[str, Any],
        filename: str = "07_scorecard.xlsx"
    ) -> str:
        """Generate credit scorecard Excel file."""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Full scorecard
            all_bins = []
            for feature, bins in scorecard_data.items():
                for bin_data in bins:
                    record = {'Feature': feature}
                    record.update(bin_data)
                    all_bins.append(record)
            
            scorecard_df = pd.DataFrame(all_bins)
            scorecard_df.to_excel(writer, sheet_name='Scorecard', index=False)
            
            # Score statistics
            if score_stats:
                stats_df = pd.DataFrame([{
                    'Metric': k,
                    'Value': v
                } for k, v in score_stats.items()])
                stats_df.to_excel(writer, sheet_name='Score Statistics', index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    def generate_cutoff_analysis_excel(
        self,
        cutoff_table: pd.DataFrame,
        optimal_cutoffs: Dict[str, Dict[str, float]],
        filename: str = "08_cutoff_analysis.xlsx"
    ) -> str:
        """Generate cutoff analysis Excel file."""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            cutoff_table.to_excel(writer, sheet_name='Cutoff Table', index=False)
            
            # Optimal cutoffs summary
            optimal_records = []
            for method, values in optimal_cutoffs.items():
                optimal_records.append({
                    'Method': method,
                    'Optimal Cutoff': values.get('cutoff', 0),
                    'Metric Value': values.get('metric', 0)
                })
            
            optimal_df = pd.DataFrame(optimal_records)
            optimal_df.to_excel(writer, sheet_name='Optimal Cutoffs', index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    def generate_psi_report_excel(
        self,
        psi_results: Dict[str, Dict[str, Any]],
        filename: str = "09_psi_monitoring.xlsx"
    ) -> str:
        """Generate PSI monitoring Excel file."""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary
            summary_records = []
            for feature, result in psi_results.items():
                summary_records.append({
                    'Feature': feature,
                    'PSI Value': result.get('psi_value', 0),
                    'Status': result.get('status', 'Unknown')
                })
            
            summary_df = pd.DataFrame(summary_records)
            summary_df = summary_df.sort_values('PSI Value', ascending=False)
            summary_df.to_excel(writer, sheet_name='PSI Summary', index=False)
            
            # Bin details per feature (if available)
            for feature, result in psi_results.items():
                if 'bin_details' in result and result['bin_details']:
                    bins_df = pd.DataFrame(result['bin_details'])
                    sheet_name = f"PSI_{feature[:27]}"
                    bins_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Generated: {filepath}")
        return str(filepath)
    
    # ═══════════════════════════════════════════════════════════════
    # CHART GENERATORS (PDF + PNG)
    # ═══════════════════════════════════════════════════════════════
    
    def _save_chart(
        self,
        fig: Figure,
        base_filename: str,
        save_pdf: bool = True,
        save_png: bool = True
    ) -> Dict[str, str]:
        """Save chart in PDF and PNG formats."""
        paths = {}
        
        if save_pdf:
            pdf_path = self.output_dir / f"{base_filename}.pdf"
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=self.dpi)
            paths['pdf'] = str(pdf_path)
        
        if save_png:
            png_path = self.output_dir / f"{base_filename}.png"
            fig.savefig(png_path, format='png', bbox_inches='tight', dpi=self.dpi)
            paths['png'] = str(png_path)
        
        plt.close(fig)
        self.logger.info(f"Generated charts: {base_filename}")
        return paths
    
    def generate_target_distribution_chart(
        self,
        target_counts: Dict[int, int],
        base_filename: str = "chart_01_target_distribution"
    ) -> Dict[str, str]:
        """Generate target variable distribution chart."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        labels = ['Non-Default (0)', 'Default (1)']
        values = [target_counts.get(0, 0), target_counts.get(1, 0)]
        colors = [CHART_COLORS['success'], CHART_COLORS['danger']]
        
        bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                   f'{val:,}', ha='center', va='bottom', fontsize=self.label_fontsize, fontweight='bold')
        
        # Add percentage
        total = sum(values)
        for bar, val in zip(bars, values):
            pct = val/total*100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   f'{pct:.1f}%', ha='center', va='center', fontsize=self.label_fontsize,
                   color='white', fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=self.label_fontsize)
        ax.set_title('Target Distribution', fontsize=self.title_fontsize, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15)
        
        return self._save_chart(fig, base_filename)
    
    def generate_iv_ranking_chart(
        self,
        iv_scores: Dict[str, float],
        top_n: int = 20,
        base_filename: str = "chart_02_iv_ranking"
    ) -> Dict[str, str]:
        """Generate Information Value ranking bar chart."""
        # Sort and take top N
        sorted_iv = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [x[0] for x in sorted_iv]
        scores = [x[1] for x in sorted_iv]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by IV category
        colors = []
        for iv in scores:
            if iv < 0.02:
                colors.append('#DC3545')  # Useless - red
            elif iv < 0.1:
                colors.append('#FFC107')  # Weak - yellow
            elif iv < 0.3:
                colors.append('#28A745')  # Medium - green
            elif iv < 0.5:
                colors.append('#2E86AB')  # Strong - blue
            else:
                colors.append('#A23B72')  # Suspicious - purple
        
        bars = ax.barh(range(len(features)), scores, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Information Value (IV)', fontsize=self.label_fontsize)
        ax.set_title(f'Top {top_n} Features by Information Value', fontsize=self.title_fontsize, fontweight='bold')
        ax.invert_yaxis()
        
        # Add IV thresholds
        for threshold, label, color in [(0.02, 'Useless', '#DC3545'), 
                                         (0.1, 'Weak', '#FFC107'),
                                         (0.3, 'Medium', '#28A745'),
                                         (0.5, 'Strong', '#A23B72')]:
            ax.axvline(x=threshold, color=color, linestyle='--', alpha=0.5, linewidth=1)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='#DC3545', label='Useless (<0.02)'),
            mpatches.Patch(color='#FFC107', label='Weak (0.02-0.10)'),
            mpatches.Patch(color='#28A745', label='Medium (0.10-0.30)'),
            mpatches.Patch(color='#2E86AB', label='Strong (0.30-0.50)'),
            mpatches.Patch(color='#A23B72', label='Suspicious (>0.50)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        return self._save_chart(fig, base_filename)
    
    def generate_roc_curve_chart(
        self,
        roc_data: Dict[str, Dict[str, Any]],
        base_filename: str = "chart_03_roc_curves"
    ) -> Dict[str, str]:
        """Generate ROC curves for model comparison."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = CHART_COLORS['gradient']
        
        for idx, (model_name, data) in enumerate(roc_data.items()):
            fpr = data.get('fpr', [])
            tpr = data.get('tpr', [])
            auc = data.get('auc', 0)
            
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', 
                   color=color, linewidth=2)
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
        
        ax.set_xlabel('False Positive Rate', fontsize=self.label_fontsize)
        ax.set_ylabel('True Positive Rate', fontsize=self.label_fontsize)
        ax.set_title('ROC Curve Comparison', fontsize=self.title_fontsize, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        return self._save_chart(fig, base_filename)
    
    def generate_ks_chart(
        self,
        ks_data: Dict[str, Any],
        model_name: str = "Model",
        base_filename: str = "chart_04_ks_plot"
    ) -> Dict[str, str]:
        """Generate KS (Kolmogorov-Smirnov) chart."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        thresholds = ks_data.get('thresholds', [])
        cum_good = ks_data.get('cum_good', [])
        cum_bad = ks_data.get('cum_bad', [])
        ks_value = ks_data.get('ks_statistic', 0)
        ks_threshold = ks_data.get('ks_threshold', 0.5)
        
        ax.plot(thresholds, cum_good, label='Cumulative Good %', 
               color=CHART_COLORS['success'], linewidth=2)
        ax.plot(thresholds, cum_bad, label='Cumulative Bad %', 
               color=CHART_COLORS['danger'], linewidth=2)
        
        # Mark KS point
        ax.axvline(x=ks_threshold, color=CHART_COLORS['primary'], 
                  linestyle='--', linewidth=1.5, label=f'KS = {ks_value:.3f}')
        
        ax.set_xlabel('Score Threshold', fontsize=self.label_fontsize)
        ax.set_ylabel('Cumulative Percentage', fontsize=self.label_fontsize)
        ax.set_title(f'KS Plot - {model_name}', fontsize=self.title_fontsize, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return self._save_chart(fig, base_filename)
    
    def generate_lift_chart(
        self,
        lift_data: pd.DataFrame,
        model_name: str = "Model",
        base_filename: str = "chart_05_lift_curve"
    ) -> Dict[str, str]:
        """Generate lift curve chart."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        deciles = lift_data.get('decile', range(1, 11))
        lift = lift_data.get('lift', [1] * 10)
        cumulative_lift = lift_data.get('cumulative_lift', [1] * 10)
        
        ax.bar(deciles, lift, color=CHART_COLORS['primary'], 
              alpha=0.7, label='Decile Lift', edgecolor='black')
        ax.plot(deciles, cumulative_lift, color=CHART_COLORS['secondary'], 
               linewidth=2, marker='o', label='Cumulative Lift')
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Baseline (Lift=1)')
        
        ax.set_xlabel('Decile (1 = Highest Risk)', fontsize=self.label_fontsize)
        ax.set_ylabel('Lift', fontsize=self.label_fontsize)
        ax.set_title(f'Lift Chart - {model_name}', fontsize=self.title_fontsize, fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._save_chart(fig, base_filename)
    
    def generate_calibration_chart(
        self,
        calibration_data: Dict[str, Any],
        model_name: str = "Model",
        base_filename: str = "chart_06_calibration"
    ) -> Dict[str, str]:
        """Generate probability calibration chart."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        mean_predicted = calibration_data.get('mean_predicted', [])
        fraction_positive = calibration_data.get('fraction_positive', [])
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        ax.plot(mean_predicted, fraction_positive, 'o-', 
               color=CHART_COLORS['primary'], linewidth=2, markersize=8,
               label=f'{model_name}')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=self.label_fontsize)
        ax.set_ylabel('Fraction of Positives', fontsize=self.label_fontsize)
        ax.set_title(f'Calibration Curve - {model_name}', fontsize=self.title_fontsize, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        return self._save_chart(fig, base_filename)
    
    def generate_score_distribution_chart(
        self,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        base_filename: str = "chart_07_score_distribution"
    ) -> Dict[str, str]:
        """Generate score distribution comparison chart."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(train_scores, bins=50, alpha=0.6, label='Train', 
               color=CHART_COLORS['primary'], edgecolor='black', linewidth=0.5)
        ax.hist(test_scores, bins=50, alpha=0.6, label='Test', 
               color=CHART_COLORS['secondary'], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Score', fontsize=self.label_fontsize)
        ax.set_ylabel('Frequency', fontsize=self.label_fontsize)
        ax.set_title('Score Distribution: Train vs Test', fontsize=self.title_fontsize, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._save_chart(fig, base_filename)
    
    def generate_psi_chart(
        self,
        psi_results: Dict[str, float],
        top_n: int = 15,
        base_filename: str = "chart_08_psi_monitoring"
    ) -> Dict[str, str]:
        """Generate PSI monitoring bar chart."""
        # Sort and take top N
        sorted_psi = sorted(psi_results.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [x[0] for x in sorted_psi]
        values = [x[1] for x in sorted_psi]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by PSI status
        colors = []
        for psi in values:
            if psi < 0.1:
                colors.append(CHART_COLORS['success'])  # Stable
            elif psi < 0.25:
                colors.append(CHART_COLORS['warning'])  # Warning
            else:
                colors.append(CHART_COLORS['danger'])   # Critical
        
        bars = ax.barh(range(len(features)), values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('PSI Value', fontsize=self.label_fontsize)
        ax.set_title('Population Stability Index by Feature', fontsize=self.title_fontsize, fontweight='bold')
        ax.invert_yaxis()
        
        # Add threshold lines
        ax.axvline(x=0.1, color=CHART_COLORS['warning'], linestyle='--', 
                  linewidth=2, label='Warning (0.10)')
        ax.axvline(x=0.25, color=CHART_COLORS['danger'], linestyle='--', 
                  linewidth=2, label='Critical (0.25)')
        
        ax.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        
        return self._save_chart(fig, base_filename)
    
    def generate_cutoff_tradeoff_chart(
        self,
        cutoff_table: pd.DataFrame,
        base_filename: str = "chart_09_cutoff_tradeoff"
    ) -> Dict[str, str]:
        """Generate cutoff trade-off chart (approval rate vs bad rate)."""
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        cutoffs = cutoff_table.get('cutoff', [])
        approval_rate = cutoff_table.get('approval_rate', [])
        bad_rate = cutoff_table.get('bad_rate_approved', [])
        
        # Primary axis - Approval Rate
        color1 = CHART_COLORS['primary']
        ax1.plot(cutoffs, approval_rate, color=color1, linewidth=2, marker='o', label='Approval Rate')
        ax1.set_xlabel('Score Cutoff', fontsize=self.label_fontsize)
        ax1.set_ylabel('Approval Rate', color=color1, fontsize=self.label_fontsize)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Secondary axis - Bad Rate
        ax2 = ax1.twinx()
        color2 = CHART_COLORS['danger']
        ax2.plot(cutoffs, bad_rate, color=color2, linewidth=2, marker='s', label='Bad Rate (Approved)')
        ax2.set_ylabel('Bad Rate Among Approved', color=color2, fontsize=self.label_fontsize)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.set_title('Cutoff Trade-off Analysis', fontsize=self.title_fontsize, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
        
        return self._save_chart(fig, base_filename)
    
    def generate_feature_importance_chart(
        self,
        feature_importance: Dict[str, float],
        model_name: str = "Model",
        top_n: int = 20,
        base_filename: str = "chart_10_feature_importance"
    ) -> Dict[str, str]:
        """Generate feature importance bar chart."""
        # Sort and take top N
        sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [x[0] for x in sorted_fi]
        importance = [x[1] for x in sorted_fi]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
        bars = ax.barh(range(len(features)), importance, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Importance', fontsize=self.label_fontsize)
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}', 
                    fontsize=self.title_fontsize, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return self._save_chart(fig, base_filename)
    
    def generate_model_comparison_chart(
        self,
        comparison_data: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        base_filename: str = "chart_11_model_comparison"
    ) -> Dict[str, str]:
        """Generate model comparison radar/bar chart."""
        if metrics is None:
            metrics = ['gini', 'ks_statistic', 'auc', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        models = list(comparison_data.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        colors = CHART_COLORS['gradient']
        
        for i, model in enumerate(models):
            values = [comparison_data[model].get(m, 0) for m in metrics]
            offset = width * i - width * (len(models) - 1) / 2
            bars = ax.bar(x + offset, values, width, label=model, 
                         color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_ylabel('Score', fontsize=self.label_fontsize)
        ax.set_title('Model Performance Comparison', fontsize=self.title_fontsize, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._save_chart(fig, base_filename)
    
    # ═══════════════════════════════════════════════════════════════
    # MASTER REPORT GENERATOR
    # ═══════════════════════════════════════════════════════════════
    
    def generate_all_reports(
        self,
        pipeline_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate all Excel files and charts from pipeline results.
        
        Args:
            pipeline_results: Complete pipeline results dictionary
            
        Returns:
            Dictionary of output type to file path(s)
        """
        outputs = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.logger.info("Generating PowerPoint-friendly reports...")
        
        # Generate based on available results
        if 'data' in pipeline_results:
            data = pipeline_results['data']
            if 'applications_stats' in data:
                outputs['data_summary_excel'] = self.generate_data_summary_excel(
                    data.get('applications_stats', {}),
                    data.get('credit_bureau_stats', {})
                )
            if 'target_distribution' in data:
                outputs['target_chart'] = self.generate_target_distribution_chart(
                    data['target_distribution']
                )
        
        if 'univariate' in pipeline_results:
            outputs['univariate_excel'] = self.generate_univariate_excel(
                pipeline_results['univariate']
            )
            # Extract IV scores for chart
            iv_scores = {f: r.get('iv_score', 0) 
                        for f, r in pipeline_results['univariate'].items()
                        if r.get('iv_score')}
            if iv_scores:
                outputs['iv_chart'] = self.generate_iv_ranking_chart(iv_scores)
        
        if 'woe' in pipeline_results:
            outputs['woe_excel'] = self.generate_woe_binning_excel(
                pipeline_results['woe']
            )
        
        if 'evaluation' in pipeline_results:
            eval_results = pipeline_results['evaluation']
            outputs['metrics_excel'] = self.generate_model_metrics_excel(eval_results)
            
            # Generate comparison chart
            comparison_data = {}
            for model, results in eval_results.items():
                if 'metrics' in results:
                    comparison_data[model] = results['metrics']
            if comparison_data:
                outputs['comparison_chart'] = self.generate_model_comparison_chart(comparison_data)
        
        if 'scorecard' in pipeline_results:
            sc = pipeline_results['scorecard']
            outputs['scorecard_excel'] = self.generate_scorecard_excel(
                sc.get('scorecard_data', {}),
                sc.get('score_stats', {})
            )
        
        if 'cutoff' in pipeline_results:
            cutoff = pipeline_results['cutoff']
            if 'cutoff_table' in cutoff:
                outputs['cutoff_excel'] = self.generate_cutoff_analysis_excel(
                    pd.DataFrame(cutoff['cutoff_table']),
                    cutoff.get('optimal_cutoffs', {})
                )
                outputs['cutoff_chart'] = self.generate_cutoff_tradeoff_chart(
                    pd.DataFrame(cutoff['cutoff_table'])
                )
        
        if 'psi' in pipeline_results:
            psi = pipeline_results['psi']
            outputs['psi_excel'] = self.generate_psi_report_excel(psi)
            psi_values = {f: r.get('psi_value', 0) for f, r in psi.items()}
            if psi_values:
                outputs['psi_chart'] = self.generate_psi_chart(psi_values)
        
        # Generate index file
        index_path = self._generate_index(outputs, timestamp)
        outputs['index'] = index_path
        
        self.logger.info(f"Generated {len(outputs)} output files in {self.output_dir}")
        return outputs
    
    def _generate_index(
        self,
        outputs: Dict[str, str],
        timestamp: str
    ) -> str:
        """Generate index file listing all outputs."""
        index_path = self.output_dir / "00_index.txt"
        
        with open(index_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CREDIT SCORING PIPELINE - OUTPUT FILES\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXCEL FILES (Tables for PowerPoint):\n")
            f.write("-" * 40 + "\n")
            for key, path in outputs.items():
                if 'excel' in key:
                    f.write(f"  • {Path(path).name}\n")
            
            f.write("\nCHART FILES (PDF + PNG for PowerPoint):\n")
            f.write("-" * 40 + "\n")
            for key, path in outputs.items():
                if 'chart' in key:
                    if isinstance(path, dict):
                        for fmt, p in path.items():
                            f.write(f"  • {Path(p).name}\n")
                    else:
                        f.write(f"  • {Path(path).name}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("TIP: Copy Excel tables directly into PowerPoint.\n")
            f.write("TIP: Insert PNG charts for easy editing in slides.\n")
            f.write("TIP: Use PDF charts for high-quality printing.\n")
        
        return str(index_path)
