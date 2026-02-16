"""
Model Development Pipeline

Orchestrates the 10-step variable elimination and model development process:
1. Constant elimination
2. Missing elimination
3. IV elimination
4. PSI stability elimination
5. Correlation elimination
6. Sequential feature selection (forward/backward with CV)
7. VIF multicollinearity check
8. Optuna hyperparameter tuning
9. Quarterly model evaluation
10. Excel report generation
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config.schema import PipelineConfig
from src.io.output_manager import OutputManager
from src.model_development.data_loader import load_and_split, DataSets
from src.model_development.eliminators import (
    ConstantEliminator,
    MissingEliminator,
    IVEliminator,
    PSIEliminator,
    PSICheck,
    QuarterlyPSICheck,
    CorrelationEliminator,
    VIFEliminator,
    TemporalPerformanceEliminator,
    EliminationResult,
)
from src.model_development.feature_selector import sequential_feature_selection
from src.model_development.hyperparameter_tuner import tune_hyperparameters
from src.model_development.evaluator import (
    evaluate_model_quarterly,
    evaluate_model_summary,
    evaluate_quarterly_chronological,
    compute_variable_quarterly_auc,
    bootstrap_auc_ci,
    compute_score_psi,
    compute_quarterly_trend,
    compute_confusion_metrics,
)
from src.model_development.subsegment_evaluator import (
    evaluate_by_subsegment,
    compute_confusion_by_subsegment,
)
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
    7. Sequential feature selection (forward/backward with CV)
    8. VIF multicollinearity check
    9. Optuna hyperparameter tuning
    10. Quarterly model evaluation
    11. Generate Excel report
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
        correlation_threshold: float = 0.80,
        test_size: float = 0.20,
        xgb_params: Optional[Dict] = None,
        target_column: str = 'target',
        date_column: str = 'application_date',
        psi_checks: Optional[List['PSICheck']] = None,
        # Selection params
        selection_method: str = 'forward',
        selection_cv: int = 5,
        selection_max_features: int = 20,
        selection_min_features: int = 1,
        selection_tolerance: float = 0.001,
        selection_patience: int = 3,
        # VIF params
        vif_enabled: bool = True,
        vif_threshold: float = 5.0,
        vif_iv_aware: bool = True,
        # Tuning params
        tuning_enabled: bool = True,
        tuning_n_trials: int = 100,
        tuning_timeout: Optional[int] = None,
        tuning_cv: int = 5,
        # Optional config and output manager (Phase 2 enhancements)
        config: Optional[PipelineConfig] = None,
        output_manager: Optional[OutputManager] = None,
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
        self.test_size = test_size
        self.xgb_params = xgb_params
        self.psi_checks = psi_checks
        self.target_column = target_column
        self.date_column = date_column

        # Selection
        self.selection_method = selection_method
        self.selection_cv = selection_cv
        self.selection_max_features = selection_max_features
        self.selection_min_features = selection_min_features
        self.selection_tolerance = selection_tolerance
        self.selection_patience = selection_patience

        # VIF
        self.vif_enabled = vif_enabled
        self.vif_threshold = vif_threshold
        self.vif_iv_aware = vif_iv_aware

        # Tuning
        self.tuning_enabled = tuning_enabled
        self.tuning_n_trials = tuning_n_trials
        self.tuning_timeout = tuning_timeout
        self.tuning_cv = tuning_cv

        # Config and output manager for enhanced steps
        self.config = config
        self.output_manager = output_manager

        # Parallelism
        self.n_jobs = config.reproducibility.n_jobs if config else 1

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
        stratify = self.config.splitting.stratify if self.config else True
        datasets = load_and_split(
            input_path=self.input_path,
            train_end_date=self.train_end_date,
            target_column=self.target_column,
            date_column=self.date_column,
            test_size=self.test_size,
            stratify=stratify,
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
            min_iv=self.iv_min, max_iv=self.iv_max,
            n_jobs=self.n_jobs,
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

        # Step 4: PSI stability (within training data only -- OOT untouched)
        psi_elim = PSIEliminator(
            critical_threshold=self.psi_threshold,
            checks=self.psi_checks,
            n_jobs=self.n_jobs,
        )
        psi_result = psi_elim.eliminate(
            X_train, y_train, features,
            train_dates=datasets.train[self.date_column],
        )
        elimination_results.append(psi_result)
        features = psi_result.kept_features
        results['after_psi'] = len(features)

        # Step 5a: ALWAYS compute variable quarterly AUC (for 11b sheet)
        variable_quarterly_df = None
        try:
            logger.info("VAR_AUC | Computing variable quarterly AUC for all features")
            variable_quarterly_df = compute_variable_quarterly_auc(
                features=features,
                datasets=datasets,
                target_column=self.target_column,
                date_column=self.date_column,
                n_jobs=self.n_jobs,
            )
        except Exception as e:
            logger.warning("VAR_AUC | Variable quarterly AUC failed: %s", e)

        # Step 5b: Temporal performance filter (enabled by default)
        temporal_cfg = self.config.steps.temporal_filter if self.config else None
        if temporal_cfg and temporal_cfg.enabled:
            logger.info("TEMPORAL | Running temporal performance filter")
            temporal_elim = TemporalPerformanceEliminator(
                min_quarterly_auc=temporal_cfg.min_quarterly_auc,
                max_auc_degradation=temporal_cfg.max_auc_degradation,
                min_trend_slope=temporal_cfg.min_trend_slope,
                n_jobs=self.n_jobs,
            )
            temporal_result = temporal_elim.eliminate(
                X_train, y_train, features,
                train_dates=datasets.train[self.date_column],
                oot_quarters=datasets.oot_quarters,
                target_column=self.target_column,
            )
            elimination_results.append(temporal_result)
            features = temporal_result.kept_features
            results['after_temporal'] = len(features)
        else:
            results['after_temporal'] = len(features)

        # Step 6: Correlation elimination
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

        # Step 7: Sequential feature selection (CV-based)
        logger.info(
            f"SELECTION | Starting {self.selection_method} selection "
            f"with {len(features)} features"
        )
        selection_output_dir = str(self.output_dir)
        if self.output_manager:
            selection_output_dir = str(self.output_manager.run_dir / "reports")

        selected_features, selection_df, chart_path = sequential_feature_selection(
            X_train=X_train[features],
            y_train=y_train,
            X_test=X_test[features],
            y_test=y_test,
            features=features,
            direction=self.selection_method,
            cv=self.selection_cv,
            min_features=self.selection_min_features,
            max_features=self.selection_max_features,
            tolerance=self.selection_tolerance,
            patience=self.selection_patience,
            iv_scores=iv_scores,
            xgb_params=self.xgb_params,
            n_jobs=self.n_jobs,
            output_dir=selection_output_dir,
        )
        results['after_selection'] = len(selected_features)
        results['selected_features'] = selected_features
        results['chart_path'] = chart_path

        # Step 8: VIF check (post-selection, when few features remain)
        vif_result = None
        if self.vif_enabled and len(selected_features) > 2:
            logger.info(
                f"VIF | Running VIF check on {len(selected_features)} "
                f"selected features"
            )
            vif_elim = VIFEliminator(
                threshold=self.vif_threshold,
                iv_aware=self.vif_iv_aware,
            )
            vif_result = vif_elim.eliminate(
                X_train, y_train, selected_features,
                iv_scores=iv_scores,
            )
            elimination_results.append(vif_result)
            selected_features = vif_result.kept_features
            results['after_vif'] = len(selected_features)
        else:
            results['after_vif'] = len(selected_features)
            logger.info("VIF | Skipped (disabled or too few features)")

        # Step 9: Hyperparameter tuning
        tuning_df = None
        best_params = None
        if self.tuning_enabled:
            logger.info("TUNING | Starting hyperparameter optimization")
            tuning_cfg = self.config.model.tuning if self.config else None
            stability_weight = tuning_cfg.stability_weight if tuning_cfg else 1.0
            best_params, tuning_df, final_model = tune_hyperparameters(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                features=selected_features,
                n_trials=self.tuning_n_trials,
                timeout=self.tuning_timeout,
                cv=self.tuning_cv,
                n_jobs=self.n_jobs,
                oot_quarters=datasets.oot_quarters,
                target_column=self.target_column,
                stability_weight=stability_weight,
            )
            results['tuning_best_params'] = best_params
            results['tuning_n_trials'] = self.tuning_n_trials
        else:
            logger.info("TUNING | Skipped (disabled)")
            final_model = self._train_default_model(
                X_train[selected_features], y_train,
                X_test[selected_features], y_test,
            )

        # Step 10: Quarterly evaluation
        eval_cfg = self.config.evaluation if self.config else None
        importance_type = eval_cfg.importance_type if eval_cfg else 'gain'

        logger.info("EVAL | Evaluating model across all periods")
        performance_df, lift_tables, importance_df = evaluate_model_quarterly(
            model=final_model,
            selected_features=selected_features,
            train_df=datasets.train,
            test_df=datasets.test,
            oot_quarters=datasets.oot_quarters,
            target_column=self.target_column,
            importance_type=importance_type,
        )

        # NEW: 3-row summary for Excel 11_Performance
        logger.info("EVAL | Computing 3-row summary (Train, Test, OOT)")
        summary_perf_df, summary_lift_tables, importance_df = evaluate_model_summary(
            model=final_model,
            selected_features=selected_features,
            train_df=datasets.train,
            test_df=datasets.test,
            oot_quarters=datasets.oot_quarters,
            target_column=self.target_column,
            importance_type=importance_type,
        )

        # NEW: Chronological quarterly for Excel 11a_Quarterly_Perf
        quarterly_perf_df = None
        try:
            logger.info("EVAL | Computing chronological quarterly performance")
            quarterly_perf_df = evaluate_quarterly_chronological(
                model=final_model,
                selected_features=selected_features,
                datasets=datasets,
                target_column=self.target_column,
                date_column=self.date_column,
            )
        except Exception as e:
            logger.warning("EVAL | Quarterly chronological failed: %s", e)

        # Extract key metrics from summary (Train, Test, OOT)
        for _, row in summary_perf_df.iterrows():
            period = row['Period']
            results[f'AUC_{period}'] = row['AUC']
            results[f'Gini_{period}'] = row['Gini']

        # ── Post-evaluation enhanced steps ──────────────────────────
        score_psi_df = None
        bootstrap_df = None
        calibration_dict = None
        shap_summary = None
        shap_plot_paths = None
        validation_report_df = None
        has_critical_failures = False

        # Step 9a: Score PSI
        if eval_cfg is None or eval_cfg.calculate_score_psi:
            logger.info("EVAL | Computing score PSI between train and OOT periods")
            try:
                train_probs = final_model.predict_proba(
                    datasets.train[selected_features]
                )[:, 1]
                oot_scores = {}
                for label in sorted(datasets.oot_quarters.keys()):
                    qdf = datasets.oot_quarters[label]
                    oot_scores[f'OOT_{label}'] = final_model.predict_proba(
                        qdf[selected_features]
                    )[:, 1]
                score_psi_df = compute_score_psi(train_probs, oot_scores)
                results['score_psi_df'] = score_psi_df
                logger.info("EVAL | Score PSI computed for %d OOT periods", len(oot_scores))
            except Exception as e:
                logger.warning("EVAL | Score PSI failed: %s", e)

        # Step 9b: Bootstrap CI (on 3-row summary: Train, Test, OOT combined)
        if eval_cfg and eval_cfg.bootstrap.enabled:
            logger.info("EVAL | Computing bootstrap AUC confidence intervals")
            try:
                oot_combined = pd.concat(
                    list(datasets.oot_quarters.values()), ignore_index=True
                ) if datasets.oot_quarters else pd.DataFrame()
                periods_for_bootstrap = [
                    ('Train', datasets.train),
                    ('Test', datasets.test),
                ]
                if len(oot_combined) > 0:
                    periods_for_bootstrap.append(('OOT', oot_combined))
                bootstrap_df = bootstrap_auc_ci(
                    model=final_model,
                    selected_features=selected_features,
                    datasets=periods_for_bootstrap,
                    target_column=self.target_column,
                    n_iterations=eval_cfg.bootstrap.n_iterations,
                    confidence_level=eval_cfg.bootstrap.confidence_level,
                    n_jobs=self.n_jobs,
                )
                # Merge CI columns into summary_perf_df (3 rows)
                if bootstrap_df is not None and not bootstrap_df.empty:
                    ci_cols = bootstrap_df[['Period', 'CI_Lower', 'CI_Upper']].copy()
                    summary_perf_df = summary_perf_df.merge(ci_cols, on='Period', how='left')
                results['bootstrap_df'] = bootstrap_df
                logger.info("EVAL | Bootstrap CI computed")
            except Exception as e:
                logger.warning("EVAL | Bootstrap CI failed: %s", e)

        # Step 9c: Calibration
        if eval_cfg and eval_cfg.calibration.enabled:
            logger.info("EVAL | Running probability calibration")
            try:
                from src.evaluation.calibrator import ModelCalibrator
                calibrator = ModelCalibrator(method=eval_cfg.calibration.method)
                test_probs = final_model.predict_proba(
                    datasets.test[selected_features]
                )[:, 1]
                y_test_vals = datasets.test[self.target_column].values
                calibrator.fit(y_test_vals, test_probs)
                cal_result = calibrator.get_calibration_result(y_test_vals, test_probs)
                calibration_dict = cal_result.to_dict()
                results['calibration'] = calibration_dict
                logger.info(
                    "EVAL | Calibration (%s): Brier %.4f -> %.4f, ECE %.4f -> %.4f",
                    cal_result.method,
                    cal_result.brier_score_before,
                    cal_result.brier_score_after,
                    cal_result.ece_before,
                    cal_result.ece_after,
                )
            except Exception as e:
                logger.warning("EVAL | Calibration failed: %s", e)

        # Step 9d: SHAP
        if eval_cfg and eval_cfg.shap.enabled:
            logger.info("EVAL | Computing SHAP values")
            try:
                from src.model_development.shap_analyzer import (
                    compute_shap_values,
                    shap_summary_df,
                    save_shap_plots,
                )
                shap_vals, feat_names, X_shap = compute_shap_values(
                    model=final_model,
                    X=datasets.train[selected_features],
                    max_samples=eval_cfg.shap.max_samples,
                )
                shap_summary = shap_summary_df(shap_vals, feat_names)
                results['shap_summary'] = shap_summary

                # Save SHAP plots
                shap_output_dir = str(self.output_dir)
                if self.output_manager:
                    shap_output_dir = str(self.output_manager.run_dir / "reports")
                shap_plot_paths = save_shap_plots(shap_vals, X_shap, shap_output_dir)
                results['shap_plot_paths'] = shap_plot_paths
                logger.info("EVAL | SHAP analysis complete, %d plots saved", len(shap_plot_paths))
            except Exception as e:
                logger.warning("EVAL | SHAP analysis failed: %s", e)

        # Step 9e: Quarterly trend
        quarterly_trend_df = None
        try:
            quarterly_trend_df = compute_quarterly_trend(performance_df)
            if quarterly_trend_df is not None and len(quarterly_trend_df) > 0:
                results['quarterly_trend_df'] = quarterly_trend_df
                logger.info("EVAL | Quarterly trend computed for %d OOT periods", len(quarterly_trend_df))
        except Exception as e:
            logger.warning("EVAL | Quarterly trend failed: %s", e)

        # Step 9f: Confusion matrix
        confusion_matrix_df = None
        if eval_cfg and eval_cfg.confusion_matrix.enabled:
            logger.info("EVAL | Computing confusion matrix at multiple thresholds")
            try:
                # Use test set for overall confusion matrix
                test_probs_cm = final_model.predict_proba(
                    datasets.test[selected_features]
                )[:, 1]
                confusion_matrix_df = compute_confusion_metrics(
                    datasets.test[self.target_column].values,
                    test_probs_cm,
                    thresholds=eval_cfg.confusion_matrix.thresholds,
                )
                results['confusion_matrix_df'] = confusion_matrix_df
                logger.info("EVAL | Confusion matrix computed at %d thresholds",
                            len(eval_cfg.confusion_matrix.thresholds))
            except Exception as e:
                logger.warning("EVAL | Confusion matrix failed: %s", e)

        # Step 9g: Subsegment analysis
        subsegment_perf = None
        subsegment_confusion = None
        if eval_cfg and eval_cfg.subsegment.enabled and eval_cfg.subsegment.columns:
            logger.info("EVAL | Running subsegment analysis for columns: %s",
                        eval_cfg.subsegment.columns)
            try:
                subsegment_perf = evaluate_by_subsegment(
                    model=final_model,
                    datasets=datasets,
                    selected_features=selected_features,
                    subsegment_columns=eval_cfg.subsegment.columns,
                    target_column=self.target_column,
                )
                results['subsegment_perf'] = subsegment_perf
                logger.info("EVAL | Subsegment performance computed")
            except Exception as e:
                logger.warning("EVAL | Subsegment performance failed: %s", e)

            if eval_cfg.confusion_matrix.enabled:
                try:
                    subsegment_confusion = compute_confusion_by_subsegment(
                        model=final_model,
                        datasets=datasets,
                        selected_features=selected_features,
                        subsegment_columns=eval_cfg.subsegment.columns,
                        target_column=self.target_column,
                        thresholds=eval_cfg.confusion_matrix.thresholds,
                    )
                    results['subsegment_confusion'] = subsegment_confusion
                    logger.info("EVAL | Subsegment confusion matrices computed")
                except Exception as e:
                    logger.warning("EVAL | Subsegment confusion failed: %s", e)

        # Step 9i: Validation
        if self.config and self.config.validation.enabled:
            logger.info("EVAL | Running model validation checks")
            try:
                from src.validation.model_checks import ModelValidator
                validator = ModelValidator(self.config)
                val_report = validator.validate(
                    performance_df=performance_df,
                    importance_df=importance_df,
                    score_psi_df=score_psi_df,
                )
                validation_report_df = val_report.to_dataframe()
                has_critical_failures = val_report.has_critical_failures
                results['validation_report'] = validation_report_df
                results['has_critical_failures'] = has_critical_failures
                logger.info("EVAL | Validation: %s", val_report.summary().split('\n')[0])
            except Exception as e:
                logger.warning("EVAL | Validation failed: %s", e)

        # Step 9j: Save model artifact
        if self.output_manager and self.config and self.config.output.save_model:
            logger.info("EVAL | Saving model artifact")
            try:
                model_path = self.output_manager.save_artifact(
                    'model', final_model, fmt='joblib'
                )
                results['model_path'] = str(model_path)
                logger.info("EVAL | Model saved: %s", model_path)
            except Exception as e:
                logger.warning("EVAL | Model save failed: %s", e)

        # Step 11: Generate Excel report
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
            performance_df=summary_perf_df,
            lift_tables=summary_lift_tables,
            importance_df=importance_df,
            vif_df=vif_result.details_df if vif_result else None,
            tuning_df=tuning_df,
            tuning_best_params=best_params,
            chart_path=chart_path,
            score_psi_df=score_psi_df,
            bootstrap_df=bootstrap_df,
            shap_summary_df=shap_summary,
            shap_plot_path=shap_plot_paths[0] if shap_plot_paths else None,
            calibration_dict=calibration_dict,
            validation_report_df=validation_report_df,
            quarterly_perf_df=quarterly_perf_df,
            variable_quarterly_df=variable_quarterly_df,
            confusion_matrix_df=confusion_matrix_df,
            subsegment_perf=subsegment_perf,
            subsegment_confusion=subsegment_confusion,
        )

        results['excel_path'] = excel_path
        results['log_file'] = self.log_file
        results['status'] = 'success'

        logger.info(f"COMPLETE | Pipeline finished successfully")
        logger.info(f"COMPLETE | Excel: {excel_path}")
        logger.info(f"COMPLETE | Log: {self.log_file}")
        if chart_path:
            logger.info(f"COMPLETE | Chart: {chart_path}")

        return results

    def _train_default_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> xgb.XGBClassifier:
        """Train XGBoost with default/configured params (no tuning)."""
        params = (self.xgb_params or {}).copy()
        if not params:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
            }

        # Auto-balance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        if params.pop('scale_pos_weight', None) == 'auto':
            params['scale_pos_weight'] = neg_count / pos_count

        # early_stopping_rounds to constructor (xgboost >= 2.0)
        early_stopping_rounds = params.pop('early_stopping_rounds', 30)
        params['early_stopping_rounds'] = early_stopping_rounds

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        return model

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
            'After Temporal Filter': (
                f"{results['after_temporal']} "
                f"({results['after_psi'] - results['after_temporal']} eliminated)"
            ),
            'After Correlation Elimination': (
                f"{results['after_correlation']} "
                f"({results['after_temporal'] - results['after_correlation']} eliminated)"
            ),
            'After Sequential Selection': (
                f"{results['after_selection']} "
                f"({results['after_correlation'] - results['after_selection']} skipped)"
            ),
            'After VIF Check': (
                f"{results['after_vif']} "
                f"({results['after_selection'] - results['after_vif']} eliminated)"
            ),
            ' ': '',  # separator
        }

        # Selection method info
        summary['Selection Method'] = self.selection_method
        summary['Selection CV Folds'] = self.selection_cv

        # Tuning info
        if self.tuning_enabled:
            best_params = results.get('tuning_best_params', {})
            summary['Tuning Enabled'] = 'Yes'
            summary['Tuning Trials'] = results.get('tuning_n_trials', self.tuning_n_trials)
            if best_params:
                best_auc = best_params.get('_best_cv_auc', 'N/A')
                summary['Tuning Best CV AUC'] = best_auc
        else:
            summary['Tuning Enabled'] = 'No'

        summary['  '] = ''  # separator

        # Add performance metrics
        for key, value in results.items():
            if key.startswith('AUC_') or key.startswith('Gini_'):
                summary[key.replace('_', ' ')] = value

        # Settings
        summary['   '] = ''  # separator
        summary['IV Range'] = f"[{self.iv_min}, {self.iv_max}]"
        summary['Missing Threshold'] = f"{self.missing_threshold:.0%}"
        summary['PSI Threshold'] = str(self.psi_threshold)
        summary['Correlation Threshold'] = str(self.correlation_threshold)
        summary['VIF Threshold'] = str(self.vif_threshold) if self.vif_enabled else 'Disabled'
        summary['Selection Tolerance'] = str(self.selection_tolerance)
        summary['Selection Patience'] = str(self.selection_patience)

        return summary
