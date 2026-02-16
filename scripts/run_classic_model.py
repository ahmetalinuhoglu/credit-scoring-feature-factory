#!/usr/bin/env python3
"""
Classic Credit Risk Model CLI

Runs the WoE + Logistic Regression + Scorecard pipeline.

Usage:
    # Run with YAML config (recommended):
    python scripts/run_classic_model.py --config config/classic_model.yaml

    # Override specific settings via CLI:
    python scripts/run_classic_model.py \\
        --config config/classic_model.yaml \\
        --input data/sample/sample_features.parquet \\
        --train-end-date 2024-06-30

    # Adjust WoE / LogReg / Scorecard params:
    python scripts/run_classic_model.py \\
        --config config/classic_model.yaml \\
        --woe-n-bins 8 \\
        --logistic-c 0.5 \\
        --scorecard-pdo 40
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.schema import ClassicPipelineConfig
from src.classic_model.pipeline import ClassicModelPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classic Credit Risk Model (WoE + LogReg + Scorecard)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        default="config/classic_model.yaml",
        help="Path to YAML config file",
    )

    # Data overrides
    parser.add_argument(
        "--input",
        default=None,
        help="Path to features parquet/csv file (overrides config)",
    )
    parser.add_argument(
        "--train-end-date",
        default=None,
        help="Cutoff date for training period (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base directory for output files",
    )

    # Data settings
    parser.add_argument(
        "--target-column",
        default=None,
        help="Name of the binary target column",
    )
    parser.add_argument(
        "--date-column",
        default=None,
        help="Name of the application date column",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of training period for test set",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        default=False,
        help="Use temporal split instead of stratified random",
    )

    # WoE overrides
    parser.add_argument(
        "--woe-n-bins",
        type=int,
        default=None,
        help="Number of WoE bins",
    )
    parser.add_argument(
        "--woe-min-bin-size",
        type=float,
        default=None,
        help="Minimum bin size as fraction of data",
    )
    parser.add_argument(
        "--woe-min-iv",
        type=float,
        default=None,
        help="Minimum IV to keep a feature",
    )
    parser.add_argument(
        "--woe-max-iv",
        type=float,
        default=None,
        help="Maximum IV (above is suspicious)",
    )
    parser.add_argument(
        "--no-monotonic",
        action="store_true",
        default=False,
        help="Disable monotonic WoE constraint",
    )

    # LogReg overrides
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=None,
        help="Regularization strength (inverse of C)",
    )
    parser.add_argument(
        "--logistic-solver",
        choices=["lbfgs", "liblinear", "saga"],
        default=None,
        help="Solver for LogisticRegression",
    )
    parser.add_argument(
        "--logistic-penalty",
        choices=["l1", "l2", "none"],
        default=None,
        help="Penalty type for LogisticRegression",
    )

    # Scorecard overrides
    parser.add_argument(
        "--scorecard-target-score",
        type=int,
        default=None,
        help="Target score for scorecard (default 600)",
    )
    parser.add_argument(
        "--scorecard-target-odds",
        type=float,
        default=None,
        help="Target odds (good:bad) at target score",
    )
    parser.add_argument(
        "--scorecard-pdo",
        type=float,
        default=None,
        help="Points to double the odds",
    )

    # Bootstrap
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        default=False,
        help="Disable bootstrap AUC confidence intervals",
    )

    # Parallelism
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (-1 = all cores)",
    )

    return parser.parse_args()


def _load_classic_config(args) -> ClassicPipelineConfig:
    """Load YAML and apply CLI overrides, returning a ClassicPipelineConfig."""
    yaml_path = Path(args.config)
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # CLI overrides
    if args.input is not None:
        raw.setdefault("data", {})["input_path"] = args.input
    if args.train_end_date is not None:
        raw.setdefault("splitting", {})["train_end_date"] = args.train_end_date
    if args.output_dir is not None:
        raw.setdefault("output", {})["base_dir"] = args.output_dir
    if args.target_column is not None:
        raw.setdefault("data", {})["target_column"] = args.target_column
    if args.date_column is not None:
        raw.setdefault("data", {})["date_column"] = args.date_column
    if args.test_size is not None:
        raw.setdefault("splitting", {})["test_size"] = args.test_size
    if args.no_stratify:
        raw.setdefault("splitting", {})["stratify"] = False

    # WoE overrides
    woe_cfg = raw.setdefault("woe", {})
    if args.woe_n_bins is not None:
        woe_cfg["n_bins"] = args.woe_n_bins
    if args.woe_min_bin_size is not None:
        woe_cfg["min_bin_size"] = args.woe_min_bin_size
    if args.woe_min_iv is not None:
        woe_cfg["min_iv"] = args.woe_min_iv
    if args.woe_max_iv is not None:
        woe_cfg["max_iv"] = args.woe_max_iv
    if args.no_monotonic:
        woe_cfg["monotonic"] = False

    # LogReg overrides
    log_cfg = raw.setdefault("logistic", {})
    if args.logistic_c is not None:
        log_cfg["C"] = args.logistic_c
    if args.logistic_solver is not None:
        log_cfg["solver"] = args.logistic_solver
    if args.logistic_penalty is not None:
        log_cfg["penalty"] = args.logistic_penalty

    # Scorecard overrides
    sc_cfg = raw.setdefault("scorecard", {})
    if args.scorecard_target_score is not None:
        sc_cfg["target_score"] = args.scorecard_target_score
    if args.scorecard_target_odds is not None:
        sc_cfg["target_odds"] = args.scorecard_target_odds
    if args.scorecard_pdo is not None:
        sc_cfg["pdo"] = args.scorecard_pdo

    # Bootstrap
    if args.no_bootstrap:
        raw.setdefault("evaluation", {}).setdefault("bootstrap", {})["enabled"] = False

    # Parallelism
    if args.n_jobs is not None:
        raw.setdefault("reproducibility", {})["n_jobs"] = args.n_jobs

    return ClassicPipelineConfig(**raw)


def _setup_logging(config: ClassicPipelineConfig) -> str:
    """Configure logging and return the log file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.output.base_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / f"classic_model_{timestamp}.log")

    log_level = getattr(logging, config.reproducibility.log_level, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
        force=True,
    )
    return log_file


def main():
    args = parse_args()
    config = _load_classic_config(args)
    log_file = _setup_logging(config)

    logger = logging.getLogger("run_classic_model")
    logger.info("Classic Credit Risk Model Pipeline")
    logger.info("Config: %s", args.config)
    logger.info("Log file: %s", log_file)

    # Determine paths
    input_path = config.data.input_path
    train_end_date = config.splitting.train_end_date
    output_dir = config.output.base_dir

    pipeline = ClassicModelPipeline(
        input_path=input_path,
        train_end_date=train_end_date,
        output_dir=output_dir,
        config=config,
    )

    results = pipeline.run()

    print(f"\n{'=' * 60}")
    print(f"Pipeline completed: {results['status']}")
    print(f"Selected features: {results.get('n_selected', 0)}")
    if results.get("selected_features"):
        for i, feat in enumerate(results["selected_features"], 1):
            print(f"  {i}. {feat}")
    if results.get("excel_path"):
        print(f"Excel report: {results['excel_path']}")
    if results.get("model_path"):
        print(f"Model artefact: {results['model_path']}")
    if results.get("woe_path"):
        print(f"WoE binning: {results['woe_path']}")
    print(f"Run directory: {results.get('run_dir', 'N/A')}")
    print(f"Log file: {log_file}")
    print(f"Duration: {results.get('duration_seconds', 0)} seconds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
