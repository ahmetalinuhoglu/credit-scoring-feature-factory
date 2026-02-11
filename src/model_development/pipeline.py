"""
Model Development Pipeline

Orchestrates the 6-step variable elimination and model development process.
Each run is logged to a separate datetime-stamped file.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging
import sys

import pandas as pd

from src.model_development.data_loader import load_and_split, DataSets
from src.model_development.eliminators import (
    ConstantEliminator,
    MissingEliminator,
    IVEliminator,
    PSIEliminator,
    PSICheck,
    QuarterlyPSICheck,
    CorrelationEliminator,
    EliminationResult,
)
from src.model_development.feature_selector import forward_feature_selection
from src.model_development.evaluator import evaluate_model_quarterly
from src.model_development import excel_reporter


logger = logging.getLogger(__name__)


class ModelDevelopmentPipeline:
    """
    End-to-end model development pipeline.

    Steps:
    1. Load data and split (Train/Test/OOT)
    2. Constant feature elimination
    3. Missing value elimination
    4. IV-based elimination
    5. PSI stability elimination
    6. Correlation elimination
    7. XGBoost forward feature selection
    8. Quarterly model evaluation
    9. Generate Excel report
    """

    def __init__(
        self,
        input_path: str,
        train_end_date: str,
        output_dir: str = 'outputs',
        iv_min: float = 0.02,
        iv_max: float = 0.50,
        missing_threshold: float = 0.70,
        psi_threshold: float = 0.25,
        correlation_threshold: float = 0.90,
        auc_threshold: float = 0.0001,
        test_size: float = 0.20,
        xgb_params: Optional[Dict] = None,
        target_column: str = 'target',
        date_column: str = 'application_date',
        psi_checks: Optional[List['PSICheck']] = None,
    ):
        self.input_path = input_path
        self.train_end_date = train_end_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.iv_min = iv_min
        self.iv_max = iv_max
        self.missing_threshold = missing_threshold
        self.psi_threshold = psi_threshold
        self.correlation_threshold = correlation_threshold
        self.auc_threshold = auc_threshold
        self.test_size = test_size
        self.xgb_params = xgb_params
        self.psi_checks = psi_checks
        self.target_column = target_column
        self.date_column = date_column

        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup run-specific log file."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'model_dev_{self.run_id}.log'

        # Create file handler for this run
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Configure root logger for this run
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        # Clear existing handlers to avoid duplication
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        self.log_file = str(log_file)
        logger.info(f"INIT | Pipeline started, run_id={self.run_id}")
        logger.info(f"INIT | Input: {self.input_path}")
        logger.info(f"INIT | Log file: {self.log_file}")

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline."""
        results = {
            'run_id': self.run_id,
            'input_path': self.input_path,
            'train_end_date': self.train_end_date,
        }

        # Step 0: Load data
        logger.info("DATA | Loading and splitting data")
        datasets = load_and_split(
            input_path=self.input_path,
            train_end_date=self.train_end_date,
            target_column=self.target_column,
            date_column=self.date_column,
            test_size=self.test_size,
        )

        features = datasets.feature_columns
        X_train = datasets.train[features]
        y_train = datasets.train[self.target_column]
        X_test = datasets.test[features]
        y_test = datasets.test[self.target_column]

        results['total_features'] = len(features)
        logger.info(f"DATA | {len(features)} feature columns identified")

        elimination_results: List[EliminationResult] = []

        # Step 1: Constant elimination
        const_elim = ConstantEliminator()
        const_result = const_elim.eliminate(X_train, y_train, features)
        elimination_results.append(const_result)
        features = const_result.kept_features
        results['after_constant'] = len(features)

        # Step 2: Missing elimination
        missing_elim = MissingEliminator(max_missing_rate=self.missing_threshold)
        missing_result = missing_elim.eliminate(X_train, y_train, features)
        elimination_results.append(missing_result)
        features = missing_result.kept_features
        results['after_missing'] = len(features)

        # Step 3: IV elimination
        iv_elim = IVEliminator(
            min_iv=self.iv_min, max_iv=self.iv_max
        )
        iv_result = iv_elim.eliminate(X_train, y_train, features)
        elimination_results.append(iv_result)
        features = iv_result.kept_features
        results['after_iv'] = len(features)

        # Extract IV scores for downstream use
        iv_scores = {}
        for _, row in iv_result.details_df.iterrows():
            if row.get('IV_Score') is not None:
                iv_scores[row['Feature']] = row['IV_Score']

        # Step 4: PSI stability (within training data only — OOT untouched)
        psi_elim = PSIEliminator(
            critical_threshold=self.psi_threshold,
            checks=self.psi_checks,
        )
        psi_result = psi_elim.eliminate(
            X_train, y_train, features,
            train_dates=datasets.train[self.date_column],
        )
        elimination_results.append(psi_result)
        features = psi_result.kept_features
        results['after_psi'] = len(features)

        # Step 5: Correlation elimination
        corr_elim = CorrelationEliminator(
            max_correlation=self.correlation_threshold
        )
        corr_result = corr_elim.eliminate(
            X_train, y_train, features, iv_scores=iv_scores
        )
        elimination_results.append(corr_result)
        features = corr_result.kept_features
        results['after_correlation'] = len(features)
        corr_pairs_df = getattr(corr_elim, 'corr_pairs_df', None)

        # Step 6: Sequential feature selection
        logger.info(
            f"SELECTION | Starting forward selection with {len(features)} features"
        )
        selected_features, selection_df, final_model = forward_feature_selection(
            X_train=X_train[features],
            y_train=y_train,
            X_test=X_test[features],
            y_test=y_test,
            features=features,
            iv_scores=iv_scores,
            auc_threshold=self.auc_threshold,
            xgb_params=self.xgb_params,
        )
        results['after_selection'] = len(selected_features)
        results['selected_features'] = selected_features

        # Step 7: Quarterly evaluation
        logger.info("EVAL | Evaluating model across all periods")
        performance_df, lift_tables, importance_df = evaluate_model_quarterly(
            model=final_model,
            selected_features=selected_features,
            train_df=datasets.train,
            test_df=datasets.test,
            oot_quarters=datasets.oot_quarters,
            target_column=self.target_column,
        )

        # Extract key metrics for summary
        for _, row in performance_df.iterrows():
            period = row['Period']
            results[f'AUC_{period}'] = row['AUC']
            results[f'Gini_{period}'] = row['Gini']

        # Step 8: Generate Excel report
        excel_path = str(
            self.output_dir / f'model_dev_{self.run_id}.xlsx'
        )

        # Build summary dict
        summary = self._build_summary(results, datasets)

        excel_reporter.generate_report(
            output_path=excel_path,
            summary=summary,
            elimination_results=elimination_results,
            corr_pairs_df=corr_pairs_df,
            selection_df=selection_df,
            performance_df=performance_df,
            lift_tables=lift_tables,
            importance_df=importance_df,
        )

        results['excel_path'] = excel_path
        results['log_file'] = self.log_file
        results['status'] = 'success'

        logger.info(f"COMPLETE | Pipeline finished successfully")
        logger.info(f"COMPLETE | Excel: {excel_path}")
        logger.info(f"COMPLETE | Log: {self.log_file}")

        return results

    def _build_summary(
        self, results: Dict[str, Any], datasets: DataSets
    ) -> Dict[str, Any]:
        """Build the summary dict for the 00_Summary sheet."""
        train_dates = datasets.train[self.date_column]
        oot_labels = ', '.join(datasets.oot_labels) if datasets.oot_labels else 'None'

        summary = {
            'Run Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Run ID': self.run_id,
            'Input File': self.input_path,
            'Train End Date': self.train_end_date,
            'Train Period': (
                f"{train_dates.min().strftime('%Y-%m-%d')} to "
                f"{train_dates.max().strftime('%Y-%m-%d')}"
            ),
            'OOT Periods': oot_labels,
            'Train Rows': len(datasets.train),
            'Test Rows': len(datasets.test),
            'Train Bad Rate': f"{datasets.train[self.target_column].mean():.2%}",
            'Test Bad Rate': f"{datasets.test[self.target_column].mean():.2%}",
            '': '',  # separator
            'Total Features': results['total_features'],
            'After Constant Elimination': (
                f"{results['after_constant']} "
                f"({results['total_features'] - results['after_constant']} eliminated)"
            ),
            'After Missing Elimination': (
                f"{results['after_missing']} "
                f"({results['after_constant'] - results['after_missing']} eliminated)"
            ),
            'After IV Elimination': (
                f"{results['after_iv']} "
                f"({results['after_missing'] - results['after_iv']} eliminated)"
            ),
            'After PSI Elimination': (
                f"{results['after_psi']} "
                f"({results['after_iv'] - results['after_psi']} eliminated)"
            ),
            'After Correlation Elimination': (
                f"{results['after_correlation']} "
                f"({results['after_psi'] - results['after_correlation']} eliminated)"
            ),
            'After Sequential Selection': (
                f"{results['after_selection']} "
                f"({results['after_correlation'] - results['after_selection']} skipped)"
            ),
            ' ': '',  # separator
        }

        # Add performance metrics
        for key, value in results.items():
            if key.startswith('AUC_') or key.startswith('Gini_'):
                summary[key.replace('_', ' ')] = value

        # Settings
        summary['  '] = ''  # separator
        summary['IV Range'] = f"[{self.iv_min}, {self.iv_max}]"
        summary['Missing Threshold'] = f"{self.missing_threshold:.0%}"
        summary['PSI Threshold'] = str(self.psi_threshold)
        summary['Correlation Threshold'] = str(self.correlation_threshold)
        summary['AUC Threshold'] = str(self.auc_threshold)

        return summary
