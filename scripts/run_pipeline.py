#!/usr/bin/env python3
"""
Run Pipeline Script

Main entry point for running the credit scoring pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_manager import ConfigManager
from src.core.logger import setup_logging, get_logger
from src.pipeline.orchestrator import PipelineOrchestrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Scoring ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline in dev environment
  python scripts/run_pipeline.py --env dev

  # Run only feature extraction
  python scripts/run_pipeline.py --env dev --stage features

  # Run with custom config directory
  python scripts/run_pipeline.py --env prod --config /path/to/config

  # Run specific stages
  python scripts/run_pipeline.py --env dev --stage data,features,models
        """
    )
    
    parser.add_argument(
        '--env', '-e',
        type=str,
        default='dev',
        choices=['dev', 'staging', 'prod'],
        help='Environment to run (default: dev)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config',
        help='Path to configuration directory (default: config)'
    )
    
    parser.add_argument(
        '--stage', '-s',
        type=str,
        default='all',
        help='Pipeline stage(s) to run: all, data, features, quality, models, evaluation (comma-separated)'
    )
    
    parser.add_argument(
        '--sample-data',
        action='store_true',
        help='Use sample data instead of BigQuery'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running pipeline'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager(
        config_dir=args.config,
        environment=args.env
    )
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else config_manager.get('logging.level', 'INFO')
    setup_logging(
        config=config_manager.get_section('logging'),
        log_level=log_level
    )
    
    logger = get_logger('run_pipeline')
    logger.info(f"Starting pipeline in {args.env} environment")
    
    # Dry run - just validate
    if args.dry_run:
        logger.info("Dry run mode - validating configuration")
        logger.info(f"Config loaded: {list(config_manager.to_dict().keys())}")
        logger.info("Configuration valid!")
        return 0
    
    # Parse stages
    if args.stage == 'all':
        stages = None  # Run all
    else:
        stages = [s.strip() for s in args.stage.split(',')]
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config=config_manager.to_dict(),
            output_dir=args.output_dir,
            use_sample_data=args.sample_data
        )
        
        # Run pipeline
        results = orchestrator.run(stages=stages)
        
        # Log results
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        if 'evaluation' in results:
            eval_results = results['evaluation']
            logger.info(f"Best model: {eval_results.get('best_model', 'N/A')}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
