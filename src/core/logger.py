"""
Logging Utilities

Provides centralized logging configuration for the pipeline.
Supports console and file handlers with configurable levels.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


# Global logger registry
_loggers: Dict[str, logging.Logger] = {}


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Set up logging for the pipeline.
    
    Args:
        config: Logging configuration dictionary
        log_level: Default log level
        log_file: Path to log file (optional)
        log_format: Log message format
    """
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get config values or use defaults
    if config:
        log_level = config.get('level', log_level)
        log_format = config.get('format', log_format)
        handlers_config = config.get('handlers', {})
    else:
        handlers_config = {}
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_config = handlers_config.get('console', {'enabled': True})
    if console_config.get('enabled', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = console_config.get('level', log_level)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    file_config = handlers_config.get('file', {})
    if file_config.get('enabled', False) or log_file:
        file_path = log_file or file_config.get('path', 'logs/pipeline.log')
        
        # Create log directory if needed
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        max_bytes = file_config.get('max_bytes', 10 * 1024 * 1024)  # 10MB default
        backup_count = file_config.get('backup_count', 5)
        
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_level = file_config.get('level', log_level)
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ['py4j', 'pyspark', 'google', 'urllib3']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name (typically module or class name)
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


class LoggerMixin:
    """
    Mixin class that provides logging capabilities to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Doing something")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


class PipelineLogger:
    """
    Structured logger for pipeline execution.
    
    Provides methods for logging pipeline-specific events
    with consistent formatting.
    """
    
    def __init__(self, name: str):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """Set logging context (e.g., run_id, component)."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context."""
        self._context = {}
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self._context:
            context_str = " ".join(f"{k}={v}" for k, v in self._context.items())
            return f"[{context_str}] {message}"
        return message
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message), **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message), **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(self._format_message(message), **kwargs)
    
    def step_start(self, step_name: str) -> None:
        """Log the start of a pipeline step."""
        self.info(f"{'='*20} Starting: {step_name} {'='*20}")
    
    def step_complete(self, step_name: str, duration: Optional[float] = None) -> None:
        """Log the completion of a pipeline step."""
        if duration:
            self.info(f"{'='*20} Completed: {step_name} ({duration:.2f}s) {'='*20}")
        else:
            self.info(f"{'='*20} Completed: {step_name} {'='*20}")
    
    def metric(self, name: str, value: Any) -> None:
        """Log a metric."""
        self.info(f"METRIC | {name}: {value}")
    
    def data_stats(self, name: str, count: int, columns: Optional[int] = None) -> None:
        """Log data statistics."""
        if columns:
            self.info(f"DATA | {name}: {count:,} rows, {columns} columns")
        else:
            self.info(f"DATA | {name}: {count:,} rows")
