"""
Custom Exceptions for the Pipeline

Provides a hierarchy of exceptions for different error types,
enabling precise error handling throughout the pipeline.
"""

from typing import Any, Dict, List, Optional


class PipelineException(Exception):
    """
    Base exception for all pipeline errors.
    
    All custom exceptions inherit from this class.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        
    def __str__(self) -> str:
        result = self.message
        if self.details:
            result += f" | Details: {self.details}"
        if self.cause:
            result += f" | Caused by: {self.cause}"
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


class ConfigurationError(PipelineException):
    """
    Raised when there's a configuration error.
    
    Examples:
    - Missing required configuration
    - Invalid configuration values
    - Configuration file not found
    """
    pass


class DataValidationError(PipelineException):
    """
    Raised when data validation fails.
    
    Examples:
    - Schema mismatch
    - Missing required columns
    - Invalid data types
    - Data quality check failures
    """
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize the data validation error.
        
        Args:
            message: Error message
            validation_errors: List of validation error details
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []
        
    def __str__(self) -> str:
        result = super().__str__()
        if self.validation_errors:
            error_count = len(self.validation_errors)
            result += f" | {error_count} validation error(s)"
        return result


class SchemaValidationError(DataValidationError):
    """
    Raised when schema validation fails.
    
    Examples:
    - Missing columns
    - Extra columns
    - Type mismatches
    """
    
    def __init__(
        self,
        message: str,
        expected_schema: Optional[Dict[str, Any]] = None,
        actual_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.expected_schema = expected_schema
        self.actual_schema = actual_schema


class DataQualityError(DataValidationError):
    """
    Raised when data quality checks fail.
    
    Examples:
    - Null ratio exceeds threshold
    - Values outside expected range
    - Duplicate records
    """
    
    def __init__(
        self,
        message: str,
        quality_report: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.quality_report = quality_report


class FeatureEngineeringError(PipelineException):
    """
    Raised when feature engineering fails.
    
    Examples:
    - Feature calculation error
    - Invalid transformation
    - Missing source data
    """
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.feature_name = feature_name
        
    def __str__(self) -> str:
        result = super().__str__()
        if self.feature_name:
            result += f" | Feature: {self.feature_name}"
        return result


class FeatureSelectionError(FeatureEngineeringError):
    """
    Raised when feature selection fails.
    
    Examples:
    - No features selected
    - Feature selection criteria not met
    """
    pass


class ModelTrainingError(PipelineException):
    """
    Raised when model training fails.
    
    Examples:
    - Training data issues
    - Convergence failure
    - Invalid hyperparameters
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        
    def __str__(self) -> str:
        result = super().__str__()
        if self.model_name:
            result += f" | Model: {self.model_name}"
        return result


class HyperparameterTuningError(ModelTrainingError):
    """
    Raised when hyperparameter tuning fails.
    
    Examples:
    - All trials failed
    - Timeout exceeded
    - Invalid parameter space
    """
    pass


class EvaluationError(PipelineException):
    """
    Raised when model evaluation fails.
    
    Examples:
    - Metric calculation error
    - Invalid predictions
    - Missing ground truth
    """
    
    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.metric_name = metric_name


class DataReaderError(PipelineException):
    """
    Raised when data reading fails.
    
    Examples:
    - BigQuery connection error
    - Table not found
    - Permission denied
    """
    
    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.source = source
        
    def __str__(self) -> str:
        result = super().__str__()
        if self.source:
            result += f" | Source: {self.source}"
        return result


class SparkError(PipelineException):
    """
    Raised when Spark operations fail.
    
    Examples:
    - Spark session initialization error
    - Executor failures
    - Memory issues
    """
    pass


class ArtifactError(PipelineException):
    """
    Raised when artifact operations fail.
    
    Examples:
    - Model save/load error
    - Invalid artifact format
    - Storage permission error
    """
    
    def __init__(
        self,
        message: str,
        artifact_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.artifact_path = artifact_path
