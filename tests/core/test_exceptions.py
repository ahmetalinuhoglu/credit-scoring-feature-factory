"""
Tests for Custom Exceptions

Tests exception attributes, inheritance, and chaining.
"""

import pytest

from src.core.exceptions import (
    PipelineException,
    ConfigurationError,
    DataValidationError,
    SchemaValidationError,
    DataQualityError,
    FeatureEngineeringError,
    FeatureSelectionError,
    ModelTrainingError,
    HyperparameterTuningError,
    EvaluationError,
    DataReaderError,
    SparkError,
    ArtifactError
)


class TestPipelineException:
    """Test suite for base PipelineException."""
    
    def test_pipeline_exception_message(self):
        """Test PipelineException stores message correctly."""
        error = PipelineException("Test error message")
        
        assert "Test error message" in str(error)
        assert error.message == "Test error message"
    
    def test_pipeline_exception_with_details(self):
        """Test PipelineException with additional details."""
        error = PipelineException(
            "Test error",
            details={'key': 'value', 'count': 42}
        )
        
        assert error.details == {'key': 'value', 'count': 42}
    
    def test_pipeline_exception_with_cause(self):
        """Test PipelineException with cause."""
        original = ValueError("Original error")
        error = PipelineException("Wrapper error", cause=original)
        
        assert error.cause is original
    
    def test_pipeline_exception_is_exception(self):
        """Test PipelineException is proper Exception subclass."""
        error = PipelineException("Test")
        
        assert isinstance(error, Exception)
        
        with pytest.raises(PipelineException):
            raise error
    
    def test_pipeline_exception_to_dict(self):
        """Test to_dict method."""
        error = PipelineException(
            "Test error",
            details={'key': 'value'}
        )
        
        result = error.to_dict()
        
        assert result['type'] == 'PipelineException'
        assert result['message'] == 'Test error'
        assert result['details'] == {'key': 'value'}


class TestConfigurationError:
    """Test suite for ConfigurationError."""
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from PipelineException."""
        error = ConfigurationError("Config error")
        
        assert isinstance(error, PipelineException)
        assert isinstance(error, Exception)
    
    def test_configuration_error_with_details(self):
        """Test ConfigurationError with details."""
        error = ConfigurationError(
            "Missing configuration",
            details={'config_key': 'model.xgboost.max_depth'}
        )
        
        assert error.details['config_key'] == "model.xgboost.max_depth"


class TestDataValidationError:
    """Test suite for DataValidationError."""
    
    def test_data_validation_error_inheritance(self):
        """Test DataValidationError inherits correctly."""
        error = DataValidationError("Validation failed")
        
        assert isinstance(error, PipelineException)
    
    def test_data_validation_with_errors(self):
        """Test DataValidationError with validation errors list."""
        error = DataValidationError(
            "Invalid data type",
            validation_errors=[
                {'column': 'amount', 'error': 'type mismatch'},
                {'column': 'date', 'error': 'invalid format'}
            ]
        )
        
        assert len(error.validation_errors) == 2
        assert error.validation_errors[0]['column'] == 'amount'
    
    def test_data_validation_str_shows_error_count(self):
        """Test string representation shows error count."""
        error = DataValidationError(
            "Validation failed",
            validation_errors=[{'error': 'test'}]
        )
        
        error_str = str(error)
        assert "1 validation error" in error_str


class TestSchemaValidationError:
    """Test suite for SchemaValidationError."""
    
    def test_schema_error_inheritance(self):
        """Test SchemaValidationError inherits from DataValidationError."""
        error = SchemaValidationError("Schema mismatch")
        
        assert isinstance(error, DataValidationError)
        assert isinstance(error, PipelineException)
    
    def test_schema_error_with_schemas(self):
        """Test SchemaValidationError with expected/actual schemas."""
        error = SchemaValidationError(
            "Schema mismatch",
            expected_schema={'id': 'int', 'name': 'string'},
            actual_schema={'id': 'string', 'name': 'string'}
        )
        
        assert error.expected_schema == {'id': 'int', 'name': 'string'}
        assert error.actual_schema == {'id': 'string', 'name': 'string'}


class TestDataQualityError:
    """Test suite for DataQualityError."""
    
    def test_data_quality_error_basic(self):
        """Test DataQualityError basic functionality."""
        error = DataQualityError("Quality check failed")
        
        assert isinstance(error, DataValidationError)
        assert isinstance(error, PipelineException)
    
    def test_data_quality_error_with_report(self):
        """Test DataQualityError with quality report."""
        error = DataQualityError(
            "Null threshold exceeded",
            quality_report={
                'check_name': 'null_check',
                'threshold': 0.1,
                'actual_value': 0.25
            }
        )
        
        assert error.quality_report['check_name'] == 'null_check'
        assert error.quality_report['threshold'] == 0.1


class TestFeatureEngineeringError:
    """Test suite for FeatureEngineeringError."""
    
    def test_feature_error_basic(self):
        """Test FeatureEngineeringError basic functionality."""
        error = FeatureEngineeringError("Feature extraction failed")
        
        assert isinstance(error, PipelineException)
    
    def test_feature_error_with_feature_name(self):
        """Test FeatureEngineeringError with feature name."""
        error = FeatureEngineeringError(
            "Cannot compute feature",
            feature_name="debt_ratio"
        )
        
        assert error.feature_name == "debt_ratio"
        assert "Feature: debt_ratio" in str(error)
    
    def test_feature_error_with_cause(self):
        """Test FeatureEngineeringError with cause."""
        cause = ValueError("Division by zero")
        error = FeatureEngineeringError(
            "Cannot compute feature",
            feature_name="debt_ratio",
            cause=cause
        )
        
        assert error.cause is cause


class TestFeatureSelectionError:
    """Test suite for FeatureSelectionError."""
    
    def test_feature_selection_error_inheritance(self):
        """Test FeatureSelectionError inherits from FeatureEngineeringError."""
        error = FeatureSelectionError("No features selected")
        
        assert isinstance(error, FeatureEngineeringError)
        assert isinstance(error, PipelineException)


class TestModelTrainingError:
    """Test suite for ModelTrainingError."""
    
    def test_model_error_basic(self):
        """Test ModelTrainingError basic functionality."""
        error = ModelTrainingError("Training failed")
        
        assert isinstance(error, PipelineException)
    
    def test_model_error_with_model_name(self):
        """Test ModelTrainingError with model name."""
        error = ModelTrainingError(
            "Fitting failed",
            model_name="XGBoostModel"
        )
        
        assert error.model_name == "XGBoostModel"
        assert "Model: XGBoostModel" in str(error)
    
    def test_model_error_with_cause(self):
        """Test ModelTrainingError with underlying cause."""
        original_error = RuntimeError("Out of memory")
        error = ModelTrainingError(
            "Training failed",
            model_name="LargeModel",
            cause=original_error
        )
        
        assert error.cause is original_error
        assert isinstance(error.cause, RuntimeError)


class TestHyperparameterTuningError:
    """Test suite for HyperparameterTuningError."""
    
    def test_hyperparameter_error_inheritance(self):
        """Test HyperparameterTuningError inherits from ModelTrainingError."""
        error = HyperparameterTuningError("All trials failed")
        
        assert isinstance(error, ModelTrainingError)
        assert isinstance(error, PipelineException)


class TestEvaluationError:
    """Test suite for EvaluationError."""
    
    def test_evaluation_error_basic(self):
        """Test EvaluationError basic functionality."""
        error = EvaluationError("Metric calculation failed")
        
        assert isinstance(error, PipelineException)
    
    def test_evaluation_error_with_metric(self):
        """Test EvaluationError with metric name."""
        error = EvaluationError(
            "Cannot compute",
            metric_name="gini"
        )
        
        assert error.metric_name == "gini"


class TestDataReaderError:
    """Test suite for DataReaderError."""
    
    def test_data_reader_error_basic(self):
        """Test DataReaderError basic functionality."""
        error = DataReaderError("Connection failed")
        
        assert isinstance(error, PipelineException)
    
    def test_data_reader_error_with_source(self):
        """Test DataReaderError with source."""
        error = DataReaderError(
            "Table not found",
            source="project.dataset.table"
        )
        
        assert error.source == "project.dataset.table"
        assert "Source: project.dataset.table" in str(error)


class TestSparkError:
    """Test suite for SparkError."""
    
    def test_spark_error_inheritance(self):
        """Test SparkError inherits from PipelineException."""
        error = SparkError("Executor failure")
        
        assert isinstance(error, PipelineException)


class TestArtifactError:
    """Test suite for ArtifactError."""
    
    def test_artifact_error_basic(self):
        """Test ArtifactError basic functionality."""
        error = ArtifactError("Save failed")
        
        assert isinstance(error, PipelineException)
    
    def test_artifact_error_with_path(self):
        """Test ArtifactError with artifact path."""
        error = ArtifactError(
            "Model save failed",
            artifact_path="/models/xgboost.pkl"
        )
        
        assert error.artifact_path == "/models/xgboost.pkl"


class TestExceptionChaining:
    """Test exception chaining and cause tracking."""
    
    def test_exception_chain(self):
        """Test that exceptions properly chain causes."""
        original = ValueError("Original error")
        
        try:
            try:
                raise original
            except ValueError as e:
                raise ModelTrainingError("Training failed", cause=e) from e
        except ModelTrainingError as chain_error:
            assert chain_error.cause is original
            assert chain_error.__cause__ is original
    
    def test_nested_exception_chain(self):
        """Test multi-level exception chaining."""
        level1 = ValueError("Level 1")
        level2 = DataValidationError("Level 2", cause=level1)
        level3 = ModelTrainingError("Level 3", cause=level2)
        
        assert level3.cause is level2
        assert level2.cause is level1


class TestExceptionStringRepresentation:
    """Test string representation of exceptions."""
    
    def test_error_str_with_details(self):
        """Test __str__ includes details."""
        error = PipelineException(
            "Test error",
            details={'key': 'value'}
        )
        
        error_str = str(error)
        assert "Test error" in error_str
        assert "Details:" in error_str
    
    def test_error_str_with_cause(self):
        """Test __str__ includes cause."""
        cause = ValueError("Original")
        error = PipelineException(
            "Wrapper error",
            cause=cause
        )
        
        error_str = str(error)
        assert "Wrapper error" in error_str
        assert "Caused by:" in error_str
    
    def test_error_repr_representation(self):
        """Test __repr__ of exceptions."""
        error = ConfigurationError("Missing key")
        
        repr_str = repr(error)
        assert "ConfigurationError" in repr_str
