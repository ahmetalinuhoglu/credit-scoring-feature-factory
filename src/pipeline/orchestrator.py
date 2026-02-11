"""
Pipeline Orchestrator

Runs registered pipeline steps in order, passing narrowing feature lists
through each step, with timing, logging, and intermediate result saving.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import sys
import time

import pandas as pd

from src.config.schema import PipelineConfig
from src.io.output_manager import OutputManager
from src.pipeline.base import BaseComponent, StepResult


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Aggregate result of a full pipeline run.

    Attributes:
        steps: Ordered list of StepResult from each step.
        final_features: Feature names surviving all steps.
        final_model: The trained model object (if selection step produced one).
        performance: Performance metrics dict (from evaluation step).
        total_duration: Total wall-clock time in seconds.
        status: 'success' or 'failed'.
    """

    steps: List[StepResult] = field(default_factory=list)
    final_features: List[str] = field(default_factory=list)
    final_model: Any = None
    performance: Dict[str, Any] = field(default_factory=dict)
    total_duration: float = 0.0
    status: str = "pending"

    def summary(self) -> str:
        """Human-readable multi-line summary of the full run."""
        lines = [f"Pipeline {self.status} in {self.total_duration:.1f}s"]
        for step in self.steps:
            lines.append(f"  {step.summary()}")
        lines.append(f"  Final features: {len(self.final_features)}")
        return "\n".join(lines)


class PipelineOrchestrator:
    """Orchestrates the model development pipeline.

    Registers BaseComponent steps in order, then runs them sequentially,
    narrowing the feature list at each step. Saves intermediate results
    via the OutputManager.

    Args:
        config: Frozen pipeline configuration.
        output_manager: OutputManager for the current run.
    """

    def __init__(self, config: PipelineConfig, output_manager: OutputManager):
        self._config = config
        self._output_manager = output_manager
        self._steps: List[BaseComponent] = []
        self._results: List[StepResult] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure file and console logging for this run."""
        log_path = self._output_manager.get_log_path()

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        log_level = getattr(logging, self._config.reproducibility.log_level, logging.INFO)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logger.info("INIT | Run ID: %s", self._output_manager.run_id)

    def register_step(self, component: BaseComponent) -> None:
        """Add a step component to the pipeline.

        Steps are run in the order they are registered.

        Args:
            component: A BaseComponent subclass instance.
        """
        self._steps.append(component)
        logger.debug(
            "Registered step: %s (order=%d)", component.step_name, component.step_order
        )

    def run_all(self, df: pd.DataFrame) -> PipelineResult:
        """Run all registered steps in order.

        Args:
            df: Full DataFrame (train split) with features and target.

        Returns:
            PipelineResult with all step results and final features.
        """
        pipeline_result = PipelineResult()
        start_time = time.time()

        target = self._config.data.target_column
        y = df[target]
        features = [
            c for c in df.columns
            if c != target
            and c not in self._config.data.id_columns
            and c not in self._config.data.exclude_columns
            and c != self._config.data.date_column
        ]

        logger.info("PIPELINE | Starting with %d features, %d steps", len(features), len(self._steps))

        X = df[features]

        try:
            for component in self._steps:
                step_start = time.time()
                logger.info("STEP | %s starting (%d features)", component.step_name, len(features))

                result = component.fit(X[features], y, train_dates=df.get(self._config.data.date_column))
                result.duration_seconds = time.time() - step_start

                features = result.output_features
                self._results.append(result)
                pipeline_result.steps.append(result)

                logger.info("STEP | %s", result.summary())

                # Save intermediate results if configured
                if self._config.output.save_step_results:
                    self._save_step(component, result)

                # Store model if produced
                if result.metadata.get("model"):
                    pipeline_result.final_model = result.metadata["model"]
                if result.metadata.get("performance"):
                    pipeline_result.performance = result.metadata["performance"]

            pipeline_result.final_features = features
            pipeline_result.status = "success"

        except Exception as e:
            logger.exception("PIPELINE | Failed at step: %s", e)
            pipeline_result.status = "failed"
            pipeline_result.final_features = features

            # Save partial results
            if self._config.output.save_step_results:
                for component, result in zip(self._steps, self._results):
                    self._save_step(component, result)

        pipeline_result.total_duration = time.time() - start_time

        logger.info("PIPELINE | %s", pipeline_result.summary())

        # Save metadata
        self._output_manager.mark_complete(pipeline_result.status)
        if self._config.reproducibility.save_metadata:
            self._output_manager.save_run_metadata()

        return pipeline_result

    def run_step(
        self, step_name: str, df: pd.DataFrame, y: pd.Series
    ) -> StepResult:
        """Run a single named step.

        Args:
            step_name: The step_name attribute to match.
            df: Feature DataFrame.
            y: Target Series.

        Returns:
            StepResult from the matched step.

        Raises:
            ValueError: If no step matches the name.
        """
        for component in self._steps:
            if component.step_name == step_name:
                result = component.fit(df, y)
                return result
        raise ValueError(f"Step not found: {step_name}")

    def run_from(
        self, step_name: str, df: pd.DataFrame, y: pd.Series
    ) -> PipelineResult:
        """Resume the pipeline from a specific step.

        Skips all steps before the named step and runs from there onwards.

        Args:
            step_name: The step_name to start from.
            df: Feature DataFrame (should already be narrowed to this point).
            y: Target Series.

        Returns:
            PipelineResult from the resumed point.

        Raises:
            ValueError: If no step matches the name.
        """
        start_idx = None
        for i, component in enumerate(self._steps):
            if component.step_name == step_name:
                start_idx = i
                break

        if start_idx is None:
            raise ValueError(f"Step not found: {step_name}")

        pipeline_result = PipelineResult()
        start_time = time.time()
        features = list(df.columns)

        try:
            for component in self._steps[start_idx:]:
                step_start = time.time()
                logger.info("STEP | %s starting (%d features)", component.step_name, len(features))

                result = component.fit(df[features], y)
                result.duration_seconds = time.time() - step_start

                features = result.output_features
                pipeline_result.steps.append(result)

                logger.info("STEP | %s", result.summary())

                if self._config.output.save_step_results:
                    self._save_step(component, result)

                if result.metadata.get("model"):
                    pipeline_result.final_model = result.metadata["model"]

            pipeline_result.final_features = features
            pipeline_result.status = "success"

        except Exception as e:
            logger.exception("PIPELINE | Failed at step: %s", e)
            pipeline_result.status = "failed"
            pipeline_result.final_features = features

        pipeline_result.total_duration = time.time() - start_time
        return pipeline_result

    def _save_step(self, component: BaseComponent, result: StepResult) -> None:
        """Save a step's results to the output directory.

        Args:
            component: The component that produced the result.
            result: The StepResult to save.
        """
        step_dir_name = component.step_name
        artifacts: Dict[str, Any] = {
            "results": result.results_df,
            "output_features": result.output_features,
            "eliminated_features": result.eliminated_features,
        }
        if result.metadata:
            # Only save serializable metadata
            serializable_meta = {
                k: v for k, v in result.metadata.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            if serializable_meta:
                artifacts["metadata"] = serializable_meta

        self._output_manager.save_step_results(step_dir_name, artifacts)
