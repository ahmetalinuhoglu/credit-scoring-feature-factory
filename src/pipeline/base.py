"""
Pipeline Base Classes

Defines the contract (BaseComponent and StepResult) that all pipeline step
components must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import time

import pandas as pd


@dataclass
class StepResult:
    """Result of a pipeline step.

    Attributes:
        step_name: Identifier for the step (e.g., '01_constant').
        input_features: Feature names passed into the step.
        output_features: Feature names surviving the step.
        eliminated_features: Feature names removed by the step.
        results_df: Detailed per-feature results DataFrame.
        metadata: Arbitrary extra data (timings, thresholds used, etc.).
        duration_seconds: Wall-clock time the step took.
    """

    step_name: str
    input_features: List[str]
    output_features: List[str]
    eliminated_features: List[str]
    results_df: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def n_input(self) -> int:
        """Number of features passed into the step."""
        return len(self.input_features)

    @property
    def n_output(self) -> int:
        """Number of features surviving the step."""
        return len(self.output_features)

    @property
    def n_eliminated(self) -> int:
        """Number of features removed by the step."""
        return len(self.eliminated_features)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"{self.step_name}: {self.n_input} -> {self.n_output} features "
            f"({self.n_eliminated} eliminated) in {self.duration_seconds:.1f}s"
        )


class BaseComponent(ABC):
    """Base class for all pipeline step components.

    Subclasses must implement fit() and transform(). The step_name and
    step_order attributes are used by the orchestrator for ordering and
    directory naming.
    """

    step_name: str = ""
    step_order: int = 0

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> StepResult:
        """Fit the component on training data and return results.

        Args:
            X: Training feature DataFrame.
            y: Training target Series.
            **kwargs: Additional arguments (e.g., date columns, IV scores).

        Returns:
            StepResult with details of the fitting.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted filter to a DataFrame (drop eliminated features).

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with only surviving features.
        """
        pass

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, **kwargs: Any
    ) -> Tuple[pd.DataFrame, StepResult]:
        """Convenience: fit + transform in one call.

        Args:
            X: Training feature DataFrame.
            y: Training target Series.
            **kwargs: Additional arguments forwarded to fit().

        Returns:
            Tuple of (transformed DataFrame, StepResult).
        """
        result = self.fit(X, y, **kwargs)
        X_out = self.transform(X)
        return X_out, result
