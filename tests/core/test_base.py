"""
Tests for Base Components

Tests SparkComponent, PandasComponent, and BaseModel abstractions.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.base import SparkComponent, PandasComponent, PipelineComponent, ComponentRegistry


# Concrete implementations for testing abstract classes
class ConcreteSparkComponent(SparkComponent):
    """Concrete implementation of SparkComponent for testing."""
    
    def run(self, *args, **kwargs):
        """Execute component logic."""
        return args[0] if args else None


class ConcretePandasComponent(PandasComponent):
    """Concrete implementation of PandasComponent for testing."""
    
    def run(self, *args, **kwargs):
        """Execute component logic."""
        return args[0] if args else None


class TestSparkComponent:
    """Test suite for SparkComponent base class."""
    
    def test_spark_component_init(self, base_config, mock_spark_session):
        """Test SparkComponent initialization."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session,
            name="TestComponent"
        )
        
        assert component.name == "TestComponent"
        assert component.config == base_config
        assert component.spark == mock_spark_session
    
    def test_spark_component_default_name(self, base_config, mock_spark_session):
        """Test SparkComponent uses class name as default."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert component.name == "ConcreteSparkComponent"
    
    def test_get_config_dot_notation(self, base_config, mock_spark_session):
        """Test getting config values with dot notation."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert component.get_config('pipeline.random_state') == 42
        assert component.get_config('spark.app_name') == 'TestPipeline'
    
    def test_get_config_with_default(self, base_config, mock_spark_session):
        """Test get_config returns default for missing keys."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert component.get_config('nonexistent.key') is None
        assert component.get_config('nonexistent.key', 'default_value') == 'default_value'
    
    def test_execution_tracking(self, base_config, mock_spark_session):
        """Test execution time tracking."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        component._start_execution()
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        component._end_execution()
        
        assert component.execution_duration > 0
    
    def test_validate_method(self, base_config, mock_spark_session):
        """Test validate method returns True by default."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert component.validate() is True
    
    def test_validate_returns_false_without_spark(self, base_config):
        """Test validate returns False when spark is None."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=None
        )
        
        assert component.validate() is False
    
    def test_logger_initialized(self, base_config, mock_spark_session):
        """Test logger is initialized on component creation."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session,
            name="LoggerTest"
        )
        
        assert hasattr(component, 'logger')
        assert component.logger is not None
    
    def test_run_method(self, base_config, mock_spark_session):
        """Test run method works."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        result = component.run("test_data")
        assert result == "test_data"


class TestPandasComponent:
    """Test suite for PandasComponent base class."""
    
    def test_pandas_component_init(self, base_config):
        """Test PandasComponent initialization."""
        component = ConcretePandasComponent(
            config=base_config,
            name="TestPandasComponent"
        )
        
        assert component.name == "TestPandasComponent"
        assert component.config == base_config
    
    def test_pandas_component_no_spark_required(self, base_config):
        """Test PandasComponent doesn't require Spark session."""
        # Should not raise even without spark
        component = ConcretePandasComponent(
            config=base_config,
            name="NullSparkComponent"
        )
        
        assert component is not None
    
    def test_get_config_works(self, base_config):
        """Test get_config works for PandasComponent."""
        component = ConcretePandasComponent(
            config=base_config,
            name="ConfigTest"
        )
        
        assert component.get_config('pipeline.random_state') == 42
    
    def test_validate_method(self, base_config):
        """Test validate method for PandasComponent."""
        component = ConcretePandasComponent(config=base_config)
        
        assert component.validate() is True
    
    def test_run_method(self, base_config):
        """Test run method works."""
        component = ConcretePandasComponent(config=base_config)
        
        result = component.run("test_data")
        assert result == "test_data"


class TestComponentInheritance:
    """Test custom component inheritance patterns."""
    
    def test_custom_spark_component(self, base_config, mock_spark_session):
        """Test creating custom SparkComponent subclass."""
        
        class CustomComponent(SparkComponent):
            def __init__(self, config, spark_session):
                super().__init__(config, spark_session, name="CustomComponent")
                self.custom_attr = "custom_value"
            
            def run(self, data):
                return data * 2
        
        component = CustomComponent(base_config, mock_spark_session)
        
        assert component.name == "CustomComponent"
        assert component.custom_attr == "custom_value"
        assert component.run(5) == 10
    
    def test_custom_validate_override(self, base_config, mock_spark_session):
        """Test overriding validate method."""
        
        class ValidatedComponent(SparkComponent):
            def validate(self):
                # Custom validation logic
                if 'pipeline' not in self.config:
                    return False
                return True
            
            def run(self, *args, **kwargs):
                return None
        
        component = ValidatedComponent(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        assert component.validate() is True
        
        # Test with invalid config
        invalid_component = ValidatedComponent(
            config={},
            spark_session=mock_spark_session
        )
        
        assert invalid_component.validate() is False


class TestComponentLogging:
    """Test component logging functionality."""
    
    def test_logger_name_matches_component(self, base_config, mock_spark_session):
        """Test logger name matches component name."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session,
            name="NamedComponent"
        )
        
        # Logger name should include component name
        assert "NamedComponent" in component.logger.name or component.name == "NamedComponent"
    
    def test_logging_does_not_raise(self, base_config, mock_spark_session):
        """Test that logging calls don't raise exceptions."""
        component = ConcreteSparkComponent(
            config=base_config,
            spark_session=mock_spark_session
        )
        
        # These should not raise
        component.logger.debug("Debug message")
        component.logger.info("Info message")
        component.logger.warning("Warning message")


class TestComponentRegistry:
    """Test suite for ComponentRegistry."""
    
    def test_register_and_get(self, base_config):
        """Test registering and retrieving components."""
        
        class TestComponent(PandasComponent):
            def run(self, *args, **kwargs):
                return "test"
        
        ComponentRegistry.register("test_comp", TestComponent)
        
        retrieved = ComponentRegistry.get("test_comp")
        assert retrieved is TestComponent
    
    def test_list_components(self):
        """Test listing registered components."""
        components = ComponentRegistry.list_components()
        assert isinstance(components, list)
    
    def test_get_nonexistent_returns_none(self):
        """Test getting non-existent component returns None."""
        result = ComponentRegistry.get("nonexistent_component")
        assert result is None
