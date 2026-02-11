#!/usr/bin/env python3
"""
Model Development CLI

Usage:
    python scripts/run_model_development.py \
        --input /data/features.parquet \
        --output-dir outputs/ \
        --train-end-date 2023-06-30
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

    # Required
    parser.add_argument(
        '--input', required=True,
        help='Path to features parquet file',
    )
    parser.add_argument(
        '--train-end-date', required=True,
        help='Cutoff date for training period (YYYY-MM-DD). '
             'Data after this date becomes OOT quarters.',
    )

    # Output
    parser.add_argument(
        '--output-dir', default='outputs',
        help='Directory for output Excel and model files',
    )

    # Elimination thresholds
    parser.add_argument(
        '--iv-min', type=float, default=0.02,
        help='Minimum IV score to keep a feature',
    )
    parser.add_argument(
        '--iv-max', type=float, default=0.50,
        help='Maximum IV score (above is suspicious)',
    )
    parser.add_argument(
        '--missing-threshold', type=float, default=0.70,
        help='Maximum missing rate to keep a feature',
    )
    parser.add_argument(
        '--psi-threshold', type=float, default=0.25,
        help='Maximum PSI value to keep a feature (across all checks)',
    )
    parser.add_argument(
        '--psi-date-split', action='append', default=None, metavar='DATE:LABEL',
        help='Custom PSI date split. Format: "YYYY-MM-DD:Label". '
             'Can be used multiple times. Example: '
             '--psi-date-split "2024-04-01:Pre/Post Apr 2024"',
    )
    parser.add_argument(
        '--psi-checks', nargs='+', default=['quarterly'],
        choices=['quarterly', 'yearly', 'consecutive', 'halfsplit'],
        help='Built-in PSI check strategies to use (default: quarterly)',
    )
    parser.add_argument(
        '--correlation-threshold', type=float, default=0.90,
        help='Maximum absolute correlation between features',
    )
    parser.add_argument(
        '--auc-threshold', type=float, default=0.0001,
        help='Minimum AUC improvement for feature selection',
    )

    # Data settings
    parser.add_argument(
        '--test-size', type=float, default=0.20,
        help='Fraction of training period for test set',
    )
    parser.add_argument(
        '--target-column', default='target',
        help='Name of the binary target column',
    )
    parser.add_argument(
        '--date-column', default='application_date',
        help='Name of the application date column',
    )

    return parser.parse_args()


def _build_psi_checks(args):
    """Build PSI check list from CLI arguments."""
    check_map = {
        'quarterly': QuarterlyPSICheck,
        'yearly': YearlyPSICheck,
        'consecutive': ConsecutiveQuarterPSICheck,
        'halfsplit': HalfSplitPSICheck,
    }

    checks = []
    for name in args.psi_checks:
        checks.append(check_map[name]())

    # Add custom date splits
    if args.psi_date_split:
        for spec in args.psi_date_split:
            if ':' in spec:
                date_str, label = spec.split(':', 1)
            else:
                date_str, label = spec, None
            checks.append(DateSplitPSICheck(date_str.strip(), label=label.strip() if label else None))

    return checks


def main():
    args = parse_args()

    psi_checks = _build_psi_checks(args)

    pipeline = ModelDevelopmentPipeline(
        input_path=args.input,
        train_end_date=args.train_end_date,
        output_dir=args.output_dir,
        iv_min=args.iv_min,
        iv_max=args.iv_max,
        missing_threshold=args.missing_threshold,
        psi_threshold=args.psi_threshold,
        correlation_threshold=args.correlation_threshold,
        auc_threshold=args.auc_threshold,
        test_size=args.test_size,
        target_column=args.target_column,
        date_column=args.date_column,
        psi_checks=psi_checks,
    )

    results = pipeline.run()

    print(f"\n{'='*60}")
    print(f"Pipeline completed: {results['status']}")
    print(f"Selected features: {results['after_selection']}")
    print(f"Excel report: {results['excel_path']}")
    print(f"Log file: {results['log_file']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
