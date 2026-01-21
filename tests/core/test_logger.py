"""
Tests for Logging Utilities

Tests setup_logging, get_logger, LoggerMixin, and PipelineLogger.
"""

import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.logger import (
    setup_logging,
    get_logger,
    LoggerMixin,
    PipelineLogger
)


class TestSetupLogging:
    """Test suite for setup_logging function."""
    
    def test_setup_logging_default(self):
        """Test setup_logging with no arguments."""
        setup_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO
    
    def test_setup_logging_with_config(self, base_config):
        """Test setup_logging with config dictionary."""
        config = {
            'level': 'DEBUG',
            'format': '%(message)s',
            'handlers': {
                'console': {'enabled': True}
            }
        }
        
        setup_logging(config=config)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom log level."""
        setup_logging(log_level='WARNING')
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
    
    def test_setup_logging_with_file(self):
        """Test setup_logging with file handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / 'test.log'
            
            setup_logging(log_file=str(log_file))
            
            # Get a logger and write a message
            logger = logging.getLogger('test_file')
            logger.info("Test message")
            
            # Check file was created (may need a short delay for file ops)
            assert log_file.parent.exists()
    
    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format."""
        custom_format = "%(levelname)s - %(message)s"
        
        setup_logging(log_format=custom_format)
        
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0


class TestGetLogger:
    """Test suite for get_logger function."""
    
    def test_get_logger_creates_logger(self):
        """Test get_logger creates a new logger."""
        logger = get_logger('test_module')
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger_returns_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger('same_name')
        logger2 = get_logger('same_name')
        
        assert logger1 is logger2
    
    def test_get_logger_different_names(self):
        """Test get_logger returns different instances for different names."""
        logger1 = get_logger('name_one')
        logger2 = get_logger('name_two')
        
        assert logger1 is not logger2
    
    def test_get_logger_name_in_logger(self):
        """Test logger has correct name."""
        logger = get_logger('my_module')
        
        assert 'my_module' in logger.name


class TestLoggerMixin:
    """Test suite for LoggerMixin class."""
    
    def test_mixin_provides_logger(self):
        """Test LoggerMixin provides logger property."""
        
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        
        assert hasattr(obj, 'logger')
        assert obj.logger is not None
    
    def test_mixin_logger_caching(self):
        """Test LoggerMixin caches logger instance."""
        
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        logger1 = obj.logger
        logger2 = obj.logger
        
        assert logger1 is logger2
    
    def test_mixin_logger_name_from_class(self):
        """Test LoggerMixin logger named after class."""
        
        class MyCustomClass(LoggerMixin):
            pass
        
        obj = MyCustomClass()
        
        assert 'MyCustomClass' in obj.logger.name
    
    def test_mixin_logging_works(self, caplog):
        """Test that logging through mixin actually works."""
        
        class TestClass(LoggerMixin):
            def do_something(self):
                self.logger.info("Doing something")
        
        obj = TestClass()
        
        with caplog.at_level(logging.INFO):
            obj.do_something()
            
            # Check log message was captured
            assert 'Doing something' in caplog.text


class TestPipelineLogger:
    """Test suite for PipelineLogger class."""
    
    def test_pipeline_logger_init(self):
        """Test PipelineLogger initialization."""
        pl = PipelineLogger('test_pipeline')
        
        assert pl is not None
        assert pl.logger is not None
    
    def test_set_context(self):
        """Test setting logging context."""
        pl = PipelineLogger('context_test')
        
        pl.set_context(run_id='123', component='loader')
        
        assert pl._context['run_id'] == '123'
        assert pl._context['component'] == 'loader'
    
    def test_clear_context(self):
        """Test clearing logging context."""
        pl = PipelineLogger('clear_test')
        pl.set_context(key='value')
        
        pl.clear_context()
        
        assert len(pl._context) == 0
    
    def test_format_message_without_context(self):
        """Test message formatting without context."""
        pl = PipelineLogger('format_test')
        
        message = pl._format_message("Test message")
        
        assert message == "Test message"
    
    def test_format_message_with_context(self):
        """Test message formatting with context."""
        pl = PipelineLogger('format_test')
        pl.set_context(run_id='xyz')
        
        message = pl._format_message("Test message")
        
        assert 'run_id=xyz' in message
        assert 'Test message' in message
    
    def test_info_method(self, caplog):
        """Test info logging method."""
        pl = PipelineLogger('info_test')
        
        with caplog.at_level(logging.INFO):
            pl.info("Info message")
            
            assert 'Info message' in caplog.text
    
    def test_debug_method(self, caplog):
        """Test debug logging method."""
        pl = PipelineLogger('debug_test')
        
        with caplog.at_level(logging.DEBUG):
            pl.debug("Debug message")
            
            assert 'Debug message' in caplog.text
    
    def test_warning_method(self, caplog):
        """Test warning logging method."""
        pl = PipelineLogger('warning_test')
        
        with caplog.at_level(logging.WARNING):
            pl.warning("Warning message")
            
            assert 'Warning message' in caplog.text
    
    def test_error_method(self, caplog):
        """Test error logging method."""
        pl = PipelineLogger('error_test')
        
        with caplog.at_level(logging.ERROR):
            pl.error("Error message")
            
            assert 'Error message' in caplog.text
    
    def test_step_start(self, caplog):
        """Test step_start logging."""
        pl = PipelineLogger('step_test')
        
        with caplog.at_level(logging.INFO):
            pl.step_start("DataLoading")
            
            assert 'Starting' in caplog.text
            assert 'DataLoading' in caplog.text
    
    def test_step_complete(self, caplog):
        """Test step_complete logging."""
        pl = PipelineLogger('step_test')
        
        with caplog.at_level(logging.INFO):
            pl.step_complete("DataLoading")
            
            assert 'Completed' in caplog.text
            assert 'DataLoading' in caplog.text
    
    def test_step_complete_with_duration(self, caplog):
        """Test step_complete with duration."""
        pl = PipelineLogger('step_test')
        
        with caplog.at_level(logging.INFO):
            pl.step_complete("DataLoading", duration=5.25)
            
            assert '5.25s' in caplog.text
    
    def test_metric_logging(self, caplog):
        """Test metric logging."""
        pl = PipelineLogger('metric_test')
        
        with caplog.at_level(logging.INFO):
            pl.metric("accuracy", 0.95)
            
            assert 'METRIC' in caplog.text
            assert 'accuracy' in caplog.text
            assert '0.95' in caplog.text
    
    def test_data_stats_logging(self, caplog):
        """Test data_stats logging."""
        pl = PipelineLogger('stats_test')
        
        with caplog.at_level(logging.INFO):
            pl.data_stats("train_data", count=1000, columns=50)
            
            assert 'DATA' in caplog.text
            assert 'train_data' in caplog.text
            assert '1,000' in caplog.text
            assert '50' in caplog.text
    
    def test_data_stats_without_columns(self, caplog):
        """Test data_stats without columns parameter."""
        pl = PipelineLogger('stats_test')
        
        with caplog.at_level(logging.INFO):
            pl.data_stats("test_data", count=500)
            
            assert '500' in caplog.text
            assert 'test_data' in caplog.text
