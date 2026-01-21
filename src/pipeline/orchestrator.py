"""
Pipeline Orchestrator

Orchestrates the end-to-end ML pipeline execution.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np

from src.core.base import PandasComponent
from src.core.logger import get_logger, PipelineLogger


class PipelineOrchestrator(PandasComponent):
    """
    Orchestrates the credit scoring pipeline.
    
    Coordinates execution of:
    1. Data loading
    2. Feature extraction
    3. Data quality checks
    4. Feature selection
    5. Model training
    6. Evaluation
    7. Reporting
    """
    
    # Available stages in order
    # New risk pipeline stages: univariate, woe, calibration, cutoff, scorecard, psi_baseline
    STAGES = [
        'data', 'features', 'quality', 'univariate', 'selection', 
        'woe', 'models', 'calibration', 'cutoff', 'scorecard', 
        'evaluation', 'psi_baseline'
    ]
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = 'outputs',
        use_sample_data: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Full configuration dictionary
            output_dir: Output directory for results
            use_sample_data: Use sample CSV data instead of BigQuery
            name: Optional orchestrator name
        """
        super().__init__(config, name or "PipelineOrchestrator")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_sample_data = use_sample_data
        self.pipeline_logger = PipelineLogger('Pipeline')
        
        self._spark = None
        self._results: Dict[str, Any] = {}
        
        # Initialize report exporter
        self._report_exporter = None
        self._init_report_exporter()
    
    def _init_report_exporter(self) -> None:
        """Initialize the report exporter for generating Excel/PDF/PNG outputs."""
        try:
            from src.reporting import ReportExporter
            self._report_exporter = ReportExporter(
                config=self.config,
                output_dir=str(self.output_dir)
            )
            self.pipeline_logger.info("Report exporter initialized")
        except Exception as e:
            self.pipeline_logger.warning(f"Could not initialize report exporter: {e}")
        
    def validate(self) -> bool:
        return True
    
    def run(
        self,
        stages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the pipeline.
        
        Args:
            stages: Specific stages to run (None = all)
            
        Returns:
            Dictionary of stage results
        """
        self._start_execution()
        
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.pipeline_logger.set_context(run_id=run_id)
        
        self.pipeline_logger.info(f"Starting pipeline run: {run_id}")
        
        # Determine stages to run
        stages_to_run = stages if stages else self.STAGES
        
        # Validate stages
        for stage in stages_to_run:
            if stage not in self.STAGES:
                raise ValueError(f"Unknown stage: {stage}. Available: {self.STAGES}")
        
        self._results = {'run_id': run_id, 'stages': {}}
        
        try:
            # Run each stage
            for stage in self.STAGES:
                if stage not in stages_to_run:
                    continue
                
                self.pipeline_logger.step_start(stage.upper())
                
                result = self._run_stage(stage)
                self._results['stages'][stage] = result
                
                # Generate reports for this stage
                self._generate_stage_reports(stage, result)
                
                self.pipeline_logger.step_complete(stage.upper())
            
            # Save results
            self._save_results()
            
            self._results['status'] = 'success'
            
        except Exception as e:
            self.pipeline_logger.exception(f"Pipeline failed: {e}")
            self._results['status'] = 'failed'
            self._results['error'] = str(e)
            raise
        
        finally:
            self._cleanup()
        
        self._end_execution()
        
        return self._results
    
    def _run_stage(self, stage: str) -> Dict[str, Any]:
        """Run a specific pipeline stage."""
        
        stage_methods = {
            'data': self._run_data_stage,
            'features': self._run_features_stage,
            'quality': self._run_quality_stage,
            'univariate': self._run_univariate_stage,
            'selection': self._run_selection_stage,
            'woe': self._run_woe_stage,
            'models': self._run_models_stage,
            'calibration': self._run_calibration_stage,
            'cutoff': self._run_cutoff_stage,
            'scorecard': self._run_scorecard_stage,
            'evaluation': self._run_evaluation_stage,
            'psi_baseline': self._run_psi_baseline_stage,
        }
        
        if stage not in stage_methods:
            raise ValueError(f"Unknown stage: {stage}")
        
        return stage_methods[stage]()
    
    def _run_data_stage(self) -> Dict[str, Any]:
        """Run data loading stage."""
        self.pipeline_logger.info("Loading data")
        
        if self.use_sample_data:
            # Load from CSV
            data_dir = Path(self.get_config('data.sample_data_path', 'data/sample'))
            
            applications = pd.read_csv(data_dir / 'sample_applications.csv')
            credit_bureau = pd.read_csv(data_dir / 'sample_credit_bureau.csv')
            
            self.pipeline_logger.data_stats('Applications', len(applications))
            self.pipeline_logger.data_stats('Credit Bureau', len(credit_bureau))
            
            # Store for next stages
            self._results['data'] = {
                'applications': applications,
                'credit_bureau': credit_bureau
            }
            
            return {
                'source': 'sample_csv',
                'applications_count': len(applications),
                'credit_bureau_count': len(credit_bureau)
            }
        else:
            # Use Spark for BigQuery
            self.pipeline_logger.info("BigQuery loading requires Spark session")
            raise NotImplementedError("BigQuery loading not implemented in sample mode")
    
    def _run_features_stage(self) -> Dict[str, Any]:
        """Run feature extraction stage using FeatureFactory."""
        self.pipeline_logger.info("Extracting features")
        
        # Get data from previous stage
        data = self._results.get('data', {})
        applications = data.get('applications')
        credit_bureau = data.get('credit_bureau')
        
        if applications is None or credit_bureau is None:
            raise ValueError("Data not loaded. Run 'data' stage first.")
        
        # Use FeatureFactory for automated feature generation
        from src.features.feature_factory import FeatureFactory
        
        factory = FeatureFactory(self.config)
        features_df = factory.generate_all_features(applications, credit_bureau)
        
        self.pipeline_logger.data_stats('Features', len(features_df), len(features_df.columns))
        self.pipeline_logger.info(f"Generated {factory.feature_count} feature definitions")
        
        # Export data dictionary
        dict_path = self.output_dir / 'data_dictionary'
        exported = factory.export_data_dictionary(str(dict_path))
        self.pipeline_logger.info(f"Data dictionary exported to {dict_path}")
        
        self._results['features'] = features_df
        self._results['feature_factory'] = factory
        
        return {
            'feature_count': len(features_df.columns) - 5,  # Exclude ID columns and target
            'sample_count': len(features_df),
            'data_dictionary_path': str(dict_path)
        }
    
    def _extract_features_pandas(
        self,
        applications: pd.DataFrame,
        credit_bureau: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract features using Pandas (for sample data)."""
        
        # Define credit products
        credit_products = ['INSTALLMENT_LOAN', 'INSTALLMENT_SALE', 'CASH_FACILITY', 'MORTGAGE']
        
        features = []
        
        for _, app_row in applications.iterrows():
            app_id = app_row['application_id']
            cust_id = app_row['customer_id']
            
            # Get customer's credit history
            customer_bureau = credit_bureau[
                (credit_bureau['application_id'] == app_id) &
                (credit_bureau['customer_id'] == cust_id)
            ]
            
            # Credit products only
            credit_only = customer_bureau[
                customer_bureau['product_type'].isin(credit_products)
            ]
            
            feature_row = {
                'application_id': app_id,
                'customer_id': cust_id,
                'applicant_type': app_row['applicant_type'],
                'application_date': app_row['application_date'],
                'target': app_row['target'],
                
                # Count features
                'total_credit_count': len(credit_only),
                'total_record_count': len(customer_bureau),
                
                # Amount features
                'total_credit_amount': credit_only['total_amount'].sum() if len(credit_only) > 0 else 0,
                'avg_credit_amount': credit_only['total_amount'].mean() if len(credit_only) > 0 else 0,
                'max_credit_amount': credit_only['total_amount'].max() if len(credit_only) > 0 else 0,
                'min_credit_amount': credit_only['total_amount'].min() if len(credit_only) > 0 else 0,
                'std_credit_amount': credit_only['total_amount'].std() if len(credit_only) > 1 else 0,
                
                # Default features
                'defaulted_credit_count': credit_only['default_date'].notna().sum(),
                'recovered_credit_count': credit_only['recovery_date'].notna().sum(),
            }
            
            # Product-specific features
            for product in credit_products:
                product_data = credit_only[credit_only['product_type'] == product]
                feature_row[f'{product}_count'] = len(product_data)
                feature_row[f'{product}_total_amount'] = product_data['total_amount'].sum()
            
            # Non-credit signals
            overdraft = customer_bureau[customer_bureau['product_type'] == 'NON_AUTH_OVERDRAFT']
            overlimit = customer_bureau[customer_bureau['product_type'] == 'OVERLIMIT']
            
            feature_row['non_auth_overdraft_count'] = len(overdraft)
            feature_row['overlimit_count'] = len(overlimit)
            feature_row['has_non_auth_overdraft'] = 1 if len(overdraft) > 0 else 0
            feature_row['has_overlimit'] = 1 if len(overlimit) > 0 else 0
            feature_row['financial_stress_flag'] = 1 if (len(overdraft) > 0 or len(overlimit) > 0) else 0
            
            # Derived features
            total = feature_row['total_credit_count']
            feature_row['default_ratio'] = feature_row['defaulted_credit_count'] / total if total > 0 else 0
            feature_row['has_current_default'] = 1 if feature_row['defaulted_credit_count'] > 0 else 0
            
            features.append(feature_row)
        
        return pd.DataFrame(features).fillna(0)
    
    def _run_quality_stage(self) -> Dict[str, Any]:
        """Run data quality checks."""
        self.pipeline_logger.info("Running quality checks")
        
        features_df = self._results.get('features')
        if features_df is None:
            raise ValueError("Features not extracted. Run 'features' stage first.")
        
        # Basic quality stats
        null_stats = features_df.isnull().sum()
        null_ratio = null_stats / len(features_df)
        
        high_null_cols = null_ratio[null_ratio > 0.5].index.tolist()
        
        self.pipeline_logger.info(f"Columns with >50% nulls: {len(high_null_cols)}")
        
        return {
            'total_columns': len(features_df.columns),
            'high_null_columns': high_null_cols,
            'quality_passed': len(high_null_cols) == 0
        }
    
    def _run_selection_stage(self) -> Dict[str, Any]:
        """Run feature selection."""
        self.pipeline_logger.info("Running feature selection")
        
        features_df = self._results.get('features')
        if features_df is None:
            raise ValueError("Features not available")
        
        # Identify feature columns
        exclude_cols = ['application_id', 'customer_id', 'applicant_type', 'application_date', 'target']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        # Simple selection: remove low variance features
        variances = features_df[feature_cols].var()
        low_var = variances[variances < 0.001].index.tolist()
        
        selected_features = [f for f in feature_cols if f not in low_var]
        
        self._results['selected_features'] = selected_features
        
        self.pipeline_logger.info(f"Selected {len(selected_features)}/{len(feature_cols)} features")
        
        return {
            'total_features': len(feature_cols),
            'selected_features': len(selected_features),
            'removed_low_variance': len(low_var)
        }
    
    def _run_models_stage(self) -> Dict[str, Any]:
        """Run model training."""
        self.pipeline_logger.info("Training models")
        
        features_df = self._results.get('features')
        selected_features = self._results.get('selected_features')
        
        if features_df is None or selected_features is None:
            raise ValueError("Features not prepared")
        
        # Prepare data
        X = features_df[selected_features]
        y = features_df['target']
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        self.pipeline_logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train models
        from src.models import XGBoostModel, LogisticRegressionModel
        
        models = {}
        
        # XGBoost
        xgb_config = self.get_config('model.models.xgboost', {})
        if xgb_config.get('enabled', True):
            xgb_model = XGBoostModel(xgb_config)
            xgb_model.fit(X_train, y_train)
            models['xgboost'] = xgb_model
            self.pipeline_logger.info("XGBoost trained")
        
        # Logistic Regression
        lr_config = self.get_config('model.models.logistic_regression', {})
        if lr_config.get('enabled', True):
            lr_model = LogisticRegressionModel(lr_config)
            lr_model.fit(X_train, y_train)
            models['logistic_regression'] = lr_model
            self.pipeline_logger.info("Logistic Regression trained")
        
        self._results['models'] = models
        self._results['test_data'] = (X_test, y_test)
        
        return {
            'models_trained': list(models.keys())
        }
    
    def _run_evaluation_stage(self) -> Dict[str, Any]:
        """Run model evaluation."""
        self.pipeline_logger.info("Evaluating models")
        
        models = self._results.get('models', {})
        test_data = self._results.get('test_data')
        
        if not models or test_data is None:
            raise ValueError("Models not trained")
        
        X_test, y_test = test_data
        
        from src.evaluation import ModelEvaluator, ReportGenerator
        
        evaluator = ModelEvaluator(self.config)
        
        # Evaluate each model
        eval_results = evaluator.evaluate_multiple(models, X_test, y_test)
        
        # Compare models
        comparison = evaluator.compare_models(eval_results)
        
        # Get best model
        best_name, best_result = evaluator.get_best_model(eval_results)
        
        # Generate report
        feature_importances = {
            name: model.get_feature_importance(top_n=20)
            for name, model in models.items()
        }
        
        report_gen = ReportGenerator(self.config, str(self.output_dir))
        report_path = report_gen.generate(
            eval_results,
            comparison,
            feature_importances
        )
        
        self.pipeline_logger.info(f"Report saved: {report_path}")
        
        # Save models
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            model.save(str(models_dir / f'{name}.joblib'))
        
        return {
            'best_model': best_name,
            'best_gini': best_result['metrics']['gini'],
            'comparison': comparison.to_dict('records'),
            'report_path': report_path
        }
    
    # ═══════════════════════════════════════════════════════════════
    # NEW RISK PIPELINE STAGES
    # ═══════════════════════════════════════════════════════════════
    
    def _run_univariate_stage(self) -> Dict[str, Any]:
        """Run univariate analysis for feature screening."""
        self.pipeline_logger.info("Running univariate analysis")
        
        features_df = self._results.get('features')
        if features_df is None:
            raise ValueError("Features not available. Run 'features' stage first.")
        
        # Check if enabled
        if not self.get_config('quality.univariate_analysis.enabled', True):
            self.pipeline_logger.info("Univariate analysis disabled, skipping")
            return {'skipped': True}
        
        from src.quality.univariate_analyzer import UnivariateAnalyzer
        
        # Get feature columns
        exclude_cols = ['application_id', 'customer_id', 'applicant_type', 'application_date', 'target']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        analyzer = UnivariateAnalyzer(self.config)
        results = analyzer.analyze(features_df, feature_cols, 'target')
        
        recommended = analyzer.get_recommended_features()
        rejected = analyzer.get_rejected_features()
        
        # Store for feature selection
        self._results['univariate_results'] = results
        self._results['univariate_recommended'] = recommended
        
        # Generate report
        report_path = self.output_dir / 'univariate_report.html'
        analyzer.generate_report(str(report_path))
        
        self.pipeline_logger.info(f"Recommended: {len(recommended)}, Rejected: {len(rejected)}")
        
        return {
            'total_features': len(feature_cols),
            'recommended': len(recommended),
            'rejected': len(rejected),
            'report_path': str(report_path)
        }
    
    def _run_woe_stage(self) -> Dict[str, Any]:
        """Run WoE binning and transformation."""
        self.pipeline_logger.info("Running WoE transformation")
        
        features_df = self._results.get('features')
        selected_features = self._results.get('selected_features')
        
        if features_df is None or selected_features is None:
            raise ValueError("Features not selected. Run 'selection' stage first.")
        
        # Check if LR and WoE enabled
        lr_config = self.get_config('model.models.logistic_regression', {})
        woe_config = lr_config.get('woe_binning', {})
        
        if not woe_config.get('enabled', False):
            self.pipeline_logger.info("WoE binning disabled, skipping")
            return {'skipped': True}
        
        from src.features.woe_transformer import WoETransformer
        
        # Only numeric features for WoE
        numeric_features = [f for f in selected_features 
                          if features_df[f].dtype in ['int64', 'float64']]
        
        woe_transformer = WoETransformer(self.config)
        woe_transformer.fit(features_df, numeric_features, 'target')
        
        # Store transformer for later use
        self._results['woe_transformer'] = woe_transformer
        
        # Export binning
        binning_path = self.output_dir / 'woe_binning.json'
        woe_transformer.export_binning(str(binning_path))
        
        iv_summary = woe_transformer.get_iv_summary()
        
        self.pipeline_logger.info(f"WoE fitted for {len(woe_transformer.fitted_features)} features")
        
        return {
            'features_binned': len(woe_transformer.fitted_features),
            'iv_summary': {k: round(v, 4) for k, v in list(iv_summary.items())[:5]},
            'binning_path': str(binning_path)
        }
    
    def _run_calibration_stage(self) -> Dict[str, Any]:
        """Run model probability calibration."""
        self.pipeline_logger.info("Running model calibration")
        
        models = self._results.get('models', {})
        test_data = self._results.get('test_data')
        
        if not models or test_data is None:
            raise ValueError("Models not trained. Run 'models' stage first.")
        
        # Check if enabled
        if not self.get_config('calibration.enabled', True):
            self.pipeline_logger.info("Calibration disabled, skipping")
            return {'skipped': True}
        
        from src.evaluation.calibrator import ModelCalibrator
        
        X_test, y_test = test_data
        method = self.get_config('calibration.default_method', 'platt')
        
        calibrators = {}
        calibration_results = {}
        
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)
            
            calibrator = ModelCalibrator(method=method)
            calibrator.fit(y_test.values, y_prob)
            
            result = calibrator.get_calibration_result(y_test.values, y_prob)
            
            calibrators[name] = calibrator
            calibration_results[name] = result.to_dict()
            
            self.pipeline_logger.info(
                f"{name}: Brier {result.brier_score_before:.4f} -> {result.brier_score_after:.4f}"
            )
        
        self._results['calibrators'] = calibrators
        
        return {
            'method': method,
            'results': calibration_results
        }
    
    def _run_cutoff_stage(self) -> Dict[str, Any]:
        """Run cutoff optimization."""
        self.pipeline_logger.info("Running cutoff optimization")
        
        models = self._results.get('models', {})
        test_data = self._results.get('test_data')
        
        if not models or test_data is None:
            raise ValueError("Models not trained")
        
        # Check if enabled
        if not self.get_config('cutoff_optimization.enabled', True):
            self.pipeline_logger.info("Cutoff optimization disabled, skipping")
            return {'skipped': True}
        
        from src.evaluation.cutoff_optimizer import CutoffOptimizer
        
        X_test, y_test = test_data
        cutoff_results = {}
        
        for name, model in models.items():
            y_score = model.predict_proba(X_test)
            
            # Find optimal cutoffs using multiple methods
            ks_cutoff, ks_stat = CutoffOptimizer.find_ks_optimal(y_test.values, y_score)
            
            # Generate cutoff table
            cutoff_table = CutoffOptimizer.generate_cutoff_table(y_test.values, y_score)
            cutoff_table.to_csv(self.output_dir / f'{name}_cutoff_table.csv', index=False)
            
            cutoff_results[name] = {
                'ks_optimal_cutoff': round(ks_cutoff, 4),
                'ks_statistic': round(ks_stat, 4)
            }
            
            self.pipeline_logger.info(f"{name}: KS cutoff = {ks_cutoff:.4f}, KS = {ks_stat:.4f}")
        
        self._results['cutoff_results'] = cutoff_results
        
        return cutoff_results
    
    def _run_scorecard_stage(self) -> Dict[str, Any]:
        """Run scorecard building for Logistic Regression."""
        self.pipeline_logger.info("Building scorecard")
        
        models = self._results.get('models', {})
        woe_transformer = self._results.get('woe_transformer')
        
        # Only for Logistic Regression with WoE
        lr_model = models.get('logistic_regression')
        
        if lr_model is None:
            self.pipeline_logger.info("No Logistic Regression model, skipping scorecard")
            return {'skipped': True, 'reason': 'no_lr_model'}
        
        if woe_transformer is None:
            self.pipeline_logger.info("No WoE transformer, skipping scorecard")
            return {'skipped': True, 'reason': 'no_woe'}
        
        # Check if enabled
        lr_config = self.get_config('model.models.logistic_regression', {})
        scorecard_config = lr_config.get('scorecard', {})
        
        if not scorecard_config.get('enabled', False):
            self.pipeline_logger.info("Scorecard disabled, skipping")
            return {'skipped': True}
        
        from src.models.scorecard_builder import ScorecardBuilder
        
        builder = ScorecardBuilder(
            lr_model=lr_model,
            woe_transformer=woe_transformer,
            pdo=scorecard_config.get('pdo', 20),
            base_score=scorecard_config.get('base_score', 600),
            base_odds=scorecard_config.get('base_odds', 50)
        )
        
        try:
            builder.build()
            
            # Export scorecard
            scorecard_path = self.output_dir / 'scorecard.json'
            builder.export_scorecard(str(scorecard_path), format='json')
            
            self._results['scorecard_builder'] = builder
            
            self.pipeline_logger.info(f"Scorecard built with {len(builder.features)} features")
            
            return {
                'features': len(builder.features),
                'scorecard_path': str(scorecard_path)
            }
        except Exception as e:
            self.pipeline_logger.warning(f"Could not build scorecard: {e}")
            return {'skipped': True, 'error': str(e)}
    
    def _run_psi_baseline_stage(self) -> Dict[str, Any]:
        """Set PSI baseline from training data."""
        self.pipeline_logger.info("Setting PSI baseline")
        
        features_df = self._results.get('features')
        selected_features = self._results.get('selected_features')
        
        if features_df is None or selected_features is None:
            raise ValueError("Features not available")
        
        # Check if enabled
        if not self.get_config('quality.psi_monitoring.enabled', True):
            self.pipeline_logger.info("PSI monitoring disabled, skipping")
            return {'skipped': True}
        
        from src.evaluation.psi_monitor import PSIMonitor
        
        monitor = PSIMonitor(
            warning_threshold=self.get_config('quality.psi_monitoring.thresholds.warning', 0.10),
            critical_threshold=self.get_config('quality.psi_monitoring.thresholds.critical', 0.25)
        )
        
        # Set baseline
        monitor.set_baseline(features_df, selected_features)
        
        # Save baseline
        baseline_path = self.output_dir / 'psi_baseline.json'
        monitor.save_baseline(str(baseline_path))
        
        self._results['psi_monitor'] = monitor
        
        self.pipeline_logger.info(f"PSI baseline set for {len(monitor.tracked_features)} features")
        
        return {
            'tracked_features': len(monitor.tracked_features),
            'baseline_path': str(baseline_path)
        }
    
    # ═══════════════════════════════════════════════════════════════
    # REPORT GENERATION
    # ═══════════════════════════════════════════════════════════════
    
    def _generate_stage_reports(self, stage: str, result: Dict[str, Any]) -> None:
        """
        Generate reports for a completed pipeline stage.
        
        Automatically generates Excel tables and PDF/PNG charts based on stage.
        """
        if self._report_exporter is None:
            return
        
        if result.get('skipped'):
            return
        
        try:
            if stage == 'data':
                self._report_data_stage(result)
            elif stage == 'features':
                self._report_features_stage(result)
            elif stage == 'quality':
                self._report_quality_stage(result)
            elif stage == 'univariate':
                self._report_univariate_stage(result)
            elif stage == 'woe':
                self._report_woe_stage(result)
            elif stage == 'models':
                self._report_models_stage(result)
            elif stage == 'calibration':
                self._report_calibration_stage(result)
            elif stage == 'cutoff':
                self._report_cutoff_stage(result)
            elif stage == 'scorecard':
                self._report_scorecard_stage(result)
            elif stage == 'evaluation':
                self._report_evaluation_stage(result)
            elif stage == 'psi_baseline':
                self._report_psi_stage(result)
                
        except Exception as e:
            self.pipeline_logger.warning(f"Could not generate reports for {stage}: {e}")
    
    def _report_data_stage(self, result: Dict[str, Any]) -> None:
        """Generate data stage reports."""
        data = self._results.get('data', {})
        applications = data.get('applications')
        credit_bureau = data.get('credit_bureau')
        
        if applications is None:
            return
        
        # Prepare stats
        apps_stats = {
            'total_rows': len(applications),
            'unique_applications': applications['application_id'].nunique(),
            'unique_customers': applications['customer_id'].nunique(),
            'joint_rate': (applications['applicant_type'] == 'CO_APPLICANT').mean(),
            'target_rate': applications['target'].mean()
        }
        
        cb_stats = {}
        if credit_bureau is not None:
            cb_stats = {
                'total_rows': len(credit_bureau),
                'avg_credits_per_customer': len(credit_bureau) / applications['customer_id'].nunique()
            }
            # Product distribution
            prod_dist = credit_bureau['product_type'].value_counts()
            cb_stats['product_distribution'] = [
                {'Product': k, 'Count': v, 'Percentage': f"{v/len(credit_bureau)*100:.1f}%"}
                for k, v in prod_dist.items()
            ]
        
        # Generate Excel
        self._report_exporter.generate_data_summary_excel(apps_stats, cb_stats)
        
        # Generate target distribution chart
        if 'target' in applications.columns:
            target_counts = applications['target'].value_counts().to_dict()
            self._report_exporter.generate_target_distribution_chart(target_counts)
    
    def _report_features_stage(self, result: Dict[str, Any]) -> None:
        """Generate features stage reports."""
        features_df = self._results.get('features')
        if features_df is None:
            return
        
        # Feature statistics
        exclude_cols = ['application_id', 'customer_id', 'applicant_type', 'application_date', 'target']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        stats_records = []
        for col in feature_cols:
            stats_records.append({
                'feature': col,
                'dtype': str(features_df[col].dtype),
                'missing': features_df[col].isnull().sum(),
                'missing_pct': f"{features_df[col].isnull().mean()*100:.1f}%",
                'unique': features_df[col].nunique(),
                'mean': round(features_df[col].mean(), 4) if features_df[col].dtype in ['int64', 'float64'] else None,
                'std': round(features_df[col].std(), 4) if features_df[col].dtype in ['int64', 'float64'] else None,
                'min': features_df[col].min() if features_df[col].dtype in ['int64', 'float64'] else None,
                'max': features_df[col].max() if features_df[col].dtype in ['int64', 'float64'] else None,
            })
        
        stats_df = pd.DataFrame(stats_records)
        self._report_exporter.generate_feature_stats_excel(stats_df)
    
    def _report_quality_stage(self, result: Dict[str, Any]) -> None:
        """Generate quality stage reports."""
        # Create quality check records from result
        quality_records = [
            {
                'check_name': 'High Null Columns',
                'check_type': 'null',
                'passed': len(result.get('high_null_columns', [])) == 0,
                'severity': 'warning',
                'message': f"Found {len(result.get('high_null_columns', []))} columns with >50% nulls",
                'details': str(result.get('high_null_columns', []))
            },
            {
                'check_name': 'Overall Quality',
                'check_type': 'summary',
                'passed': result.get('quality_passed', False),
                'severity': 'error' if not result.get('quality_passed', False) else 'info',
                'message': 'Quality passed' if result.get('quality_passed', False) else 'Quality issues found',
                'details': ''
            }
        ]
        self._report_exporter.generate_quality_report_excel(quality_records)
    
    def _report_univariate_stage(self, result: Dict[str, Any]) -> None:
        """Generate univariate analysis reports."""
        univariate_results = self._results.get('univariate_results', {})
        
        if not univariate_results:
            return
        
        # Convert to exportable format
        export_results = {}
        for feature, res in univariate_results.items():
            if hasattr(res, 'to_dict'):
                export_results[feature] = res.to_dict()
            elif isinstance(res, dict):
                export_results[feature] = res
        
        self._report_exporter.generate_univariate_excel(export_results)
        
        # Generate IV ranking chart
        iv_scores = {}
        for feature, res in univariate_results.items():
            if hasattr(res, 'iv_score') and res.iv_score is not None:
                iv_scores[feature] = res.iv_score
            elif isinstance(res, dict) and 'iv_score' in res:
                iv_scores[feature] = res['iv_score']
        
        if iv_scores:
            self._report_exporter.generate_iv_ranking_chart(iv_scores)
    
    def _report_woe_stage(self, result: Dict[str, Any]) -> None:
        """Generate WoE binning reports."""
        woe_transformer = self._results.get('woe_transformer')
        
        if woe_transformer is None:
            return
        
        # Get binning results
        woe_results = {}
        for feature in woe_transformer.fitted_features:
            binning = woe_transformer.get_binning(feature)
            if binning:
                woe_results[feature] = {
                    'iv_score': binning.get('iv', 0),
                    'iv_category': binning.get('iv_category', ''),
                    'is_monotonic': binning.get('is_monotonic', False),
                    'bins': binning.get('bins', [])
                }
        
        if woe_results:
            self._report_exporter.generate_woe_binning_excel(woe_results)
    
    def _report_models_stage(self, result: Dict[str, Any]) -> None:
        """Generate models stage reports."""
        models = self._results.get('models', {})
        
        # Feature importance charts for each model
        for name, model in models.items():
            importance = model.get_feature_importance(top_n=20)
            if importance:
                self._report_exporter.generate_feature_importance_chart(
                    importance, 
                    model_name=name,
                    base_filename=f"chart_feature_importance_{name}"
                )
    
    def _report_calibration_stage(self, result: Dict[str, Any]) -> None:
        """Generate calibration reports."""
        # Calibration results are already in the result dict
        calibration_results = result.get('results', {})
        
        for model_name, cal_result in calibration_results.items():
            if 'mean_predicted' in cal_result and 'fraction_positive' in cal_result:
                self._report_exporter.generate_calibration_chart(
                    cal_result,
                    model_name=model_name,
                    base_filename=f"chart_calibration_{model_name}"
                )
    
    def _report_cutoff_stage(self, result: Dict[str, Any]) -> None:
        """Generate cutoff analysis reports."""
        models = self._results.get('models', {})
        test_data = self._results.get('test_data')
        
        if not models or test_data is None:
            return
        
        X_test, y_test = test_data
        
        for name, model in models.items():
            # Read cutoff table if exists
            cutoff_file = self.output_dir / f'{name}_cutoff_table.csv'
            if cutoff_file.exists():
                cutoff_df = pd.read_csv(cutoff_file)
                
                # Generate cutoff tradeoff chart
                if 'cutoff' in cutoff_df.columns:
                    self._report_exporter.generate_cutoff_tradeoff_chart(
                        cutoff_df,
                        base_filename=f"chart_cutoff_tradeoff_{name}"
                    )
    
    def _report_scorecard_stage(self, result: Dict[str, Any]) -> None:
        """Generate scorecard reports."""
        scorecard_builder = self._results.get('scorecard_builder')
        
        if scorecard_builder is None:
            return
        
        # Get scorecard data
        if hasattr(scorecard_builder, 'scorecard') and scorecard_builder.scorecard is not None:
            scorecard_data = {}
            score_stats = {
                'pdo': scorecard_builder.pdo,
                'base_score': scorecard_builder.base_score,
                'base_odds': scorecard_builder.base_odds
            }
            
            # Convert scorecard to dict format
            for col in scorecard_builder.scorecard.columns:
                if col != 'Feature':
                    continue
                # Group by feature
                for feature in scorecard_builder.features:
                    feature_rows = scorecard_builder.scorecard[
                        scorecard_builder.scorecard['Feature'] == feature
                    ].to_dict('records')
                    scorecard_data[feature] = feature_rows
            
            if scorecard_data:
                self._report_exporter.generate_scorecard_excel(scorecard_data, score_stats)
    
    def _report_evaluation_stage(self, result: Dict[str, Any]) -> None:
        """Generate evaluation reports."""
        models = self._results.get('models', {})
        test_data = self._results.get('test_data')
        
        if not models or test_data is None:
            return
        
        X_test, y_test = test_data
        
        # Prepare evaluation results for Excel
        eval_results = {}
        roc_data = {}
        
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            metrics = {
                'auc': roc_auc,
                'gini': 2 * roc_auc - 1,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            }
            
            eval_results[name] = {'metrics': metrics, 'dataset': 'test'}
            roc_data[name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }
        
        # Generate Excel report
        self._report_exporter.generate_model_metrics_excel(eval_results)
        
        # Generate ROC curves chart
        if roc_data:
            self._report_exporter.generate_roc_curve_chart(roc_data)
        
        # Generate model comparison chart
        comparison_data = {name: res['metrics'] for name, res in eval_results.items()}
        if comparison_data:
            self._report_exporter.generate_model_comparison_chart(comparison_data)
    
    def _report_psi_stage(self, result: Dict[str, Any]) -> None:
        """Generate PSI monitoring reports."""
        psi_monitor = self._results.get('psi_monitor')
        
        if psi_monitor is None:
            return
        
        # For baseline setting, we don't have PSI values yet
        # Just log that baseline was set
        self.pipeline_logger.info("PSI baseline set - no drift reports to generate yet")
    
    def _save_results(self) -> None:
        """Save pipeline results to JSON."""
        results_path = self.output_dir / 'pipeline_results.json'
        
        # Filter out non-serializable objects
        serializable_results = {
            'run_id': self._results.get('run_id'),
            'status': self._results.get('status'),
            'stages': {}
        }
        
        for stage, result in self._results.get('stages', {}).items():
            if isinstance(result, dict):
                serializable_results['stages'][stage] = {
                    k: v for k, v in result.items()
                    if not isinstance(v, (pd.DataFrame, pd.Series))
                }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
