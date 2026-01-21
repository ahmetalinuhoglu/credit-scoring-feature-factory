"""
Credit Scoring Pipeline - Core Package

This package provides the core infrastructure for the ML pipeline:
- Base classes for all components
- Configuration management
- Logging utilities
- Custom exceptions
"""

from src.core.base import PipelineComponent, SparkComponent, PandasComponent
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger, setup_logging
from src.core.exceptions import (
    PipelineException,
    ConfigurationError,
    DataValidationError,
    FeatureEngineeringError,
    ModelTrainingError,
    EvaluationError
)

__all__ = [
    # Base classes
    "PipelineComponent",
    "SparkComponent", 
    "PandasComponent",
    # Config
    "ConfigManager",
    # Logging
    "get_logger",
    "setup_logging",
    # Exceptions
    "PipelineException",
    "ConfigurationError",
    "DataValidationError",
    "FeatureEngineeringError",
    "ModelTrainingError",
    "EvaluationError",
]
