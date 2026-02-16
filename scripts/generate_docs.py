#!/usr/bin/env python3
"""
Regenerate documentation from a saved pipeline run.

Reads pipeline artifacts (config, metadata, results) from a run directory
and generates a Markdown report.

Usage:
    python scripts/generate_docs.py --run-dir outputs/model_development/20250101_120000_abc123
    python scripts/generate_docs.py --run-dir outputs/model_development/20250101_120000_abc123 --output-dir docs/runs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.reporting.doc_generator import MarkdownReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_metadata(run_dir: Path) -> dict:
    """Load run_metadata.json if it exists.

    Args:
        run_dir: Path to the pipeline run directory.

    Returns:
        Parsed metadata dict, or empty dict if the file is missing.
    """
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_config(run_dir: Path):
    """Load the saved PipelineConfig YAML if available.

    Args:
        run_dir: Path to the pipeline run directory.

    Returns:
        PipelineConfig instance or None.
    """
    config_path = run_dir / "config" / "pipeline_config.yaml"
    if not config_path.is_file():
        return None

    try:
        from src.config.loader import load_config
        return load_config(yaml_path=str(config_path))
    except Exception as exc:
        logger.warning("Could not load config from %s: %s", config_path, exc)
        return None


def _build_results_from_artifacts(run_dir: Path, metadata: dict) -> dict:
    """Reconstruct a results dict from saved run artifacts.

    Looks for known files in the run directory (metadata, Excel reports,
    logs, step results) and assembles them into a dict compatible with
    ``MarkdownReportGenerator.generate()``.

    Args:
        run_dir: Path to the pipeline run directory.
        metadata: Parsed run_metadata.json dict.

    Returns:
        Assembled results dict.
    """
    results = {}

    # Basic metadata
    results["run_id"] = metadata.get("run_id", run_dir.name)
    results["status"] = metadata.get("status", "unknown")

    # Look for Excel report in the reports/ subdirectory
    reports_dir = run_dir / "reports"
    if reports_dir.is_dir():
        excel_files = list(reports_dir.glob("model_dev_*.xlsx"))
        if excel_files:
            results["excel_path"] = str(excel_files[0])

        # Look for selection chart
        chart_files = list(reports_dir.glob("*.png"))
        if chart_files:
            results["chart_path"] = str(chart_files[0])

    # Look for log files
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            results["log_file"] = str(log_files[0])

    # Look for model artifact
    data_dir = run_dir / "data"
    if data_dir.is_dir():
        model_files = list(data_dir.glob("model.*"))
        if model_files:
            results["model_path"] = str(model_files[0])

    # Try to load step result summaries
    steps_dir = run_dir / "steps"
    if steps_dir.is_dir():
        for step_dir in sorted(steps_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            # Look for JSON summaries
            for json_file in step_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        step_data = json.load(f)
                    if isinstance(step_data, dict):
                        results.update(step_data)
                except Exception:
                    continue

    return results


def main():
    """Entry point for the documentation generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown documentation from a saved pipeline run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the pipeline run directory (e.g., outputs/model_development/20250101_120000_abc123)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the generated report. Defaults to {run-dir}/docs.",
    )
    parser.add_argument(
        "--template-dir",
        default="docs/templates",
        help="Directory containing Jinja2 templates.",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        logger.error("Run directory does not exist: %s", run_dir)
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(run_dir / "docs")

    logger.info("Loading artifacts from: %s", run_dir)

    # Load metadata and config
    metadata = _load_metadata(run_dir)
    config = _load_config(run_dir)

    # Build results dict from available artifacts
    results = _build_results_from_artifacts(run_dir, metadata)
    logger.info("Assembled results with %d keys", len(results))

    # Generate the report
    generator = MarkdownReportGenerator(template_dir=args.template_dir)
    report_path = generator.generate(
        results=results,
        output_dir=output_dir,
        config=config,
    )

    print(f"\nDocumentation generated successfully.")
    print(f"Report: {report_path}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
