#!/usr/bin/env python3
"""
Model Development CLI

Usage:
    # Run with YAML config (recommended):
    python scripts/run_model_development.py --config config/model_development.yaml

    # Override specific settings via CLI:
    python scripts/run_model_development.py \
        --config config/model_development.yaml \
        --input data/features.parquet \
        --train-end-date 2024-06-30

    # Use backward selection with tuning disabled:
    python scripts/run_model_development.py \
        --config config/model_development.yaml \
        --selection-method backward \
        --no-tuning

    # Legacy mode (no YAML, all CLI args):
    python scripts/run_model_development.py \
        --input data/features.parquet \
        --train-end-date 2024-06-30
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.loader import load_config, save_config
from src.config.schema import PipelineConfig
from src.io.output_manager import OutputManager
from src.model_development.pipeline import ModelDevelopmentPipeline
from src.model_development.eliminators import (
    QuarterlyPSICheck,
    DateSplitPSICheck,
    YearlyPSICheck,
    ConsecutiveQuarterPSICheck,
    HalfSplitPSICheck,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Credit Scoring Model Development Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (new)
    parser.add_argument(
        '--config', default=None,
        help='Path to YAML config file (e.g., config/model_development.yaml)',
    )

    # Data overrides
    parser.add_argument(
        '--input', default=None,
        help='Path to features parquet file (overrides config)',
    )
    parser.add_argument(
        '--train-end-date', default=None,
        help='Cutoff date for training period (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Base directory for output files',
    )

    # Elimination threshold overrides
    parser.add_argument(
        '--iv-min', type=float, default=None,
        help='Minimum IV score to keep a feature',
    )
    parser.add_argument(
        '--iv-max', type=float, default=None,
        help='Maximum IV score (above is suspicious)',
    )
    parser.add_argument(
        '--missing-threshold', type=float, default=None,
        help='Maximum missing rate to keep a feature',
    )
    parser.add_argument(
        '--psi-threshold', type=float, default=None,
        help='Maximum PSI value to keep a feature',
    )
    parser.add_argument(
        '--psi-date-split', action='append', default=None, metavar='DATE:LABEL',
        help='Custom PSI date split. Format: "YYYY-MM-DD:Label"',
    )
    parser.add_argument(
        '--psi-checks', nargs='+', default=None,
        choices=['quarterly', 'yearly', 'consecutive', 'halfsplit'],
        help='Built-in PSI check strategies to use',
    )
    parser.add_argument(
        '--correlation-threshold', type=float, default=None,
        help='Maximum absolute correlation between features',
    )

    # Selection overrides
    parser.add_argument(
        '--selection-method', choices=['forward', 'backward'], default=None,
        help='Feature selection direction',
    )
    parser.add_argument(
        '--selection-cv', type=int, default=None,
        help='Number of CV folds for feature selection',
    )
    parser.add_argument(
        '--selection-max-features', type=int, default=None,
        help='Maximum number of features to select',
    )
    parser.add_argument(
        '--selection-min-features', type=int, default=None,
        help='Minimum number of features to select',
    )
    parser.add_argument(
        '--selection-tolerance', type=float, default=None,
        help='Minimum AUC improvement to avoid early stopping',
    )
    parser.add_argument(
        '--selection-patience', type=int, default=None,
        help='Consecutive non-improving steps before early stopping',
    )

    # VIF overrides
    parser.add_argument(
        '--vif-threshold', type=float, default=None,
        help='VIF threshold for multicollinearity elimination',
    )
    parser.add_argument(
        '--no-vif', action='store_true', default=False,
        help='Disable VIF multicollinearity check',
    )

    # Tuning overrides
    parser.add_argument(
        '--tuning-trials', type=int, default=None,
        help='Number of Optuna tuning trials',
    )
    parser.add_argument(
        '--tuning-timeout', type=int, default=None,
        help='Maximum time in seconds for hyperparameter tuning',
    )
    parser.add_argument(
        '--tuning-cv', type=int, default=None,
        help='Number of CV folds for hyperparameter tuning',
    )
    parser.add_argument(
        '--no-tuning', action='store_true', default=False,
        help='Disable hyperparameter tuning (use default params)',
    )
    parser.add_argument(
        '--stability-weight', type=float, default=None,
        help='Penalty multiplier for AUC deviation across periods during tuning '
             '(0 = pure mean AUC, 1 = equal weight on stability)',
    )

    # Data settings overrides
    parser.add_argument(
        '--test-size', type=float, default=None,
        help='Fraction of training period for test set',
    )
    parser.add_argument(
        '--target-column', default=None,
        help='Name of the binary target column',
    )
    parser.add_argument(
        '--date-column', default=None,
        help='Name of the application date column',
    )

    # Evaluation enhancement overrides
    parser.add_argument(
        '--no-calibration', action='store_true', default=False,
        help='Disable probability calibration',
    )
    parser.add_argument(
        '--no-shap', action='store_true', default=False,
        help='Disable SHAP analysis',
    )
    parser.add_argument(
        '--no-bootstrap', action='store_true', default=False,
        help='Disable bootstrap AUC confidence intervals',
    )
    parser.add_argument(
        '--calibration-method', choices=['platt', 'isotonic', 'temperature'],
        default=None, help='Calibration method',
    )
    parser.add_argument(
        '--importance-type', choices=['gain', 'weight', 'cover', 'total_gain', 'total_cover'],
        default=None, help='Feature importance type for XGBoost',
    )
    parser.add_argument(
        '--temporal-split', action='store_true', default=False,
        help='Use temporal (date-based) train/test split instead of stratified random',
    )

    # Parallelism
    parser.add_argument(
        '--n-jobs', type=int, default=None,
        help='Number of parallel jobs (-1 = all cores, 1 = sequential)',
    )

    return parser.parse_args()


def _build_cli_overrides(args) -> dict:
    """Build a flat dot-notation override dict from CLI args."""
    overrides = {}

    if args.input is not None:
        overrides["data.input_path"] = args.input
    if args.train_end_date is not None:
        overrides["splitting.train_end_date"] = args.train_end_date
    if args.output_dir is not None:
        overrides["output.base_dir"] = args.output_dir
    if args.target_column is not None:
        overrides["data.target_column"] = args.target_column
    if args.date_column is not None:
        overrides["data.date_column"] = args.date_column
    if args.test_size is not None:
        overrides["splitting.test_size"] = args.test_size
    if args.iv_min is not None:
        overrides["steps.iv.min_iv"] = args.iv_min
    if args.iv_max is not None:
        overrides["steps.iv.max_iv"] = args.iv_max
    if args.missing_threshold is not None:
        overrides["steps.missing.threshold"] = args.missing_threshold
    if args.psi_threshold is not None:
        overrides["steps.psi.threshold"] = args.psi_threshold
    if args.correlation_threshold is not None:
        overrides["steps.correlation.threshold"] = args.correlation_threshold

    # Selection overrides
    if args.selection_method is not None:
        overrides["steps.selection.method"] = args.selection_method
    if args.selection_cv is not None:
        overrides["steps.selection.cv"] = args.selection_cv
    if args.selection_max_features is not None:
        overrides["steps.selection.max_features"] = args.selection_max_features
    if args.selection_min_features is not None:
        overrides["steps.selection.min_features"] = args.selection_min_features
    if args.selection_tolerance is not None:
        overrides["steps.selection.tolerance"] = args.selection_tolerance
    if args.selection_patience is not None:
        overrides["steps.selection.patience"] = args.selection_patience

    # VIF overrides
    if args.no_vif:
        overrides["steps.vif.enabled"] = False
    if args.vif_threshold is not None:
        overrides["steps.vif.threshold"] = args.vif_threshold

    # Tuning overrides
    if args.no_tuning:
        overrides["model.tuning.enabled"] = False
    if args.tuning_trials is not None:
        overrides["model.tuning.n_trials"] = args.tuning_trials
    if args.tuning_timeout is not None:
        overrides["model.tuning.timeout"] = args.tuning_timeout
    if args.tuning_cv is not None:
        overrides["model.tuning.cv"] = args.tuning_cv
    if args.stability_weight is not None:
        overrides["model.tuning.stability_weight"] = args.stability_weight

    # Evaluation enhancement overrides
    if args.no_calibration:
        overrides["evaluation.calibration.enabled"] = False
    if args.no_shap:
        overrides["evaluation.shap.enabled"] = False
    if args.no_bootstrap:
        overrides["evaluation.bootstrap.enabled"] = False
    if args.calibration_method is not None:
        overrides["evaluation.calibration.method"] = args.calibration_method
    if args.importance_type is not None:
        overrides["evaluation.importance_type"] = args.importance_type
    if args.temporal_split:
        overrides["splitting.stratify"] = False

    # Parallelism
    if args.n_jobs is not None:
        overrides["reproducibility.n_jobs"] = args.n_jobs

    return overrides


def _build_psi_checks(config: PipelineConfig, args):
    """Build PSI check list from config and CLI arguments."""
    check_map = {
        'quarterly': QuarterlyPSICheck,
        'yearly': YearlyPSICheck,
        'consecutive': ConsecutiveQuarterPSICheck,
        'halfsplit': HalfSplitPSICheck,
    }

    # CLI --psi-checks takes precedence over config
    if args.psi_checks is not None:
        check_names = args.psi_checks
    else:
        check_names = [c.type for c in config.steps.psi.checks if c.type in check_map]

    checks = []
    for name in check_names:
        if name in check_map:
            checks.append(check_map[name]())

    # Add date-split checks from config
    for c in config.steps.psi.checks:
        if c.type == "date_split" and c.date:
            checks.append(DateSplitPSICheck(c.date, label=c.label))

    # Add custom date splits from CLI
    if args.psi_date_split:
        for spec in args.psi_date_split:
            if ':' in spec:
                date_str, label = spec.split(':', 1)
            else:
                date_str, label = spec, None
            checks.append(
                DateSplitPSICheck(date_str.strip(), label=label.strip() if label else None)
            )

    return checks


def main():
    args = parse_args()

    # Load config: YAML + CLI overrides
    cli_overrides = _build_cli_overrides(args)
    config = load_config(yaml_path=args.config, cli_overrides=cli_overrides)

    # Create output manager
    output_manager = OutputManager(config)

    # Save config snapshot
    if config.reproducibility.save_config:
        save_config(config, str(output_manager.run_dir / "config" / "pipeline_config.yaml"))

    # Build PSI checks
    psi_checks = _build_psi_checks(config, args)

    # Run the pipeline with config-derived parameters
    pipeline = ModelDevelopmentPipeline(
        input_path=config.data.input_path,
        train_end_date=config.splitting.train_end_date,
        output_dir=str(output_manager.run_dir / "reports"),
        iv_min=config.steps.iv.min_iv,
        iv_max=config.steps.iv.max_iv,
        missing_threshold=config.steps.missing.threshold,
        psi_threshold=config.steps.psi.threshold,
        correlation_threshold=config.steps.correlation.threshold,
        test_size=config.splitting.test_size,
        target_column=config.data.target_column,
        date_column=config.data.date_column,
        xgb_params=config.model.params,
        psi_checks=psi_checks,
        # Selection params
        selection_method=config.steps.selection.method,
        selection_cv=config.steps.selection.cv,
        selection_max_features=config.steps.selection.max_features,
        selection_min_features=config.steps.selection.min_features,
        selection_tolerance=config.steps.selection.tolerance,
        selection_patience=config.steps.selection.patience,
        # VIF params
        vif_enabled=config.steps.vif.enabled,
        vif_threshold=config.steps.vif.threshold,
        vif_iv_aware=config.steps.vif.iv_aware,
        # Tuning params
        tuning_enabled=config.model.tuning.enabled,
        tuning_n_trials=config.model.tuning.n_trials,
        tuning_timeout=config.model.tuning.timeout,
        tuning_cv=config.model.tuning.cv,
        # Config and output manager for enhanced steps
        config=config,
        output_manager=output_manager,
    )

    results = pipeline.run()

    # Save run metadata
    output_manager.mark_complete(results.get("status", "unknown"))
    if config.reproducibility.save_metadata:
        output_manager.save_run_metadata()

    print(f"\n{'='*60}")
    print(f"Pipeline completed: {results['status']}")
    print(f"Selected features: {results['after_selection']}")
    if results.get('after_vif') != results.get('after_selection'):
        print(f"After VIF check: {results['after_vif']}")
    if results.get('tuning_best_params'):
        print(f"Tuning: {results.get('tuning_n_trials', '?')} trials completed")
    print(f"Excel report: {results['excel_path']}")
    if results.get('chart_path'):
        print(f"Selection chart: {results['chart_path']}")
    if results.get('model_path'):
        print(f"Model artifact: {results['model_path']}")
    if results.get('shap_plot_paths'):
        print(f"SHAP plots: {', '.join(results['shap_plot_paths'])}")
    if results.get('has_critical_failures'):
        print("WARNING: Validation found critical failures!")
    print(f"Run directory: {output_manager.run_dir}")
    print(f"Log file: {results['log_file']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
