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
    parser.add_argument(
        '--auc-threshold', type=float, default=None,
        help='Minimum AUC improvement for feature selection',
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
    if args.auc_threshold is not None:
        overrides["steps.selection.auc_threshold"] = args.auc_threshold

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

    # Run the existing pipeline with config-derived parameters
    pipeline = ModelDevelopmentPipeline(
        input_path=config.data.input_path,
        train_end_date=config.splitting.train_end_date,
        output_dir=str(output_manager.run_dir / "reports"),
        iv_min=config.steps.iv.min_iv,
        iv_max=config.steps.iv.max_iv,
        missing_threshold=config.steps.missing.threshold,
        psi_threshold=config.steps.psi.threshold,
        correlation_threshold=config.steps.correlation.threshold,
        auc_threshold=config.steps.selection.auc_threshold,
        test_size=config.splitting.test_size,
        target_column=config.data.target_column,
        date_column=config.data.date_column,
        xgb_params=config.model.params,
        psi_checks=psi_checks,
    )

    results = pipeline.run()

    # Save run metadata
    output_manager.mark_complete(results.get("status", "unknown"))
    if config.reproducibility.save_metadata:
        output_manager.save_run_metadata()

    print(f"\n{'='*60}")
    print(f"Pipeline completed: {results['status']}")
    print(f"Selected features: {results['after_selection']}")
    print(f"Excel report: {results['excel_path']}")
    print(f"Run directory: {output_manager.run_dir}")
    print(f"Log file: {results['log_file']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
