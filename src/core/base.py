"""
Base Classes for Pipeline Components

Provides abstract base classes that define the interface for all pipeline components.
All components inherit from these base classes to ensure consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime
import logging


class PipelineComponent(ABC):
    """
    Abstract base class for all pipeline components.
    
    Provides common functionality:
    - Configuration access
    - Logging
    - Validation interface
    - Execution tracking
    """
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None):
        """
        Initialize the pipeline component.
        
        Args:
            config: Configuration dictionary for this component
            name: Optional name for the component (defaults to class name)
        """
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.name)
        self._execution_start: Optional[datetime] = None
        self._execution_end: Optional[datetime] = None
        
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Execute the component's main logic.
        
        Must be implemented by all subclasses.
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the component's configuration and state.
        
        Returns:
            True if validation passes, False otherwise
        """
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.xgboost.max_depth')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def _start_execution(self) -> None:
        """Mark the start of execution."""
        self._execution_start = datetime.now()
        self.logger.info(f"Starting {self.name}")
        
    def _end_execution(self) -> None:
        """Mark the end of execution and log duration."""
        self._execution_end = datetime.now()
        
        if self._execution_start:
            duration = (self._execution_end - self._execution_start).total_seconds()
            self.logger.info(f"Completed {self.name} in {duration:.2f} seconds")
    
    @property
    def execution_duration(self) -> Optional[float]:
        """Get the execution duration in seconds."""
        if self._execution_start and self._execution_end:
            return (self._execution_end - self._execution_start).total_seconds()
        return None


class SparkComponent(PipelineComponent):
    """
    Base class for Spark-based pipeline components.
    
    Extends PipelineComponent with Spark-specific functionality:
    - SparkSession management
    - DataFrame operations
    - Distributed processing utilities
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        spark_session: Any,  # SparkSession type hint avoided for import flexibility
        name: Optional[str] = None
    ):
        """
        Initialize the Spark component.
        
        Args:
            config: Configuration dictionary
            spark_session: Active SparkSession instance
            name: Optional component name
        """
        super().__init__(config, name)
        self.spark = spark_session
        
    def validate(self) -> bool:
        """Validate Spark session is available."""
        if self.spark is None:
            self.logger.error("SparkSession is not initialized")
            return False
        return True
    
    def cache_dataframe(self, df: Any, name: str) -> Any:
        """
        Cache a DataFrame with logging.
        
        Args:
            df: Spark DataFrame to cache
            name: Name for logging purposes
            
        Returns:
            Cached DataFrame
        """
        self.logger.debug(f"Caching DataFrame: {name}")
        return df.cache()
    
    def unpersist_dataframe(self, df: Any, name: str) -> None:
        """
        Unpersist a cached DataFrame.
        
        Args:
            df: Spark DataFrame to unpersist
            name: Name for logging purposes
        """
        self.logger.debug(f"Unpersisting DataFrame: {name}")
        df.unpersist()
        
    def get_partition_count(self, df: Any) -> int:
        """Get the number of partitions in a DataFrame."""
        return df.rdd.getNumPartitions()
    
    def repartition_if_needed(
        self, 
        df: Any, 
        target_partitions: Optional[int] = None
    ) -> Any:
        """
        Repartition DataFrame if needed based on data size.
        
        Args:
            df: Spark DataFrame
            target_partitions: Target number of partitions (optional)
            
        Returns:
            Repartitioned DataFrame
        """
        current_partitions = self.get_partition_count(df)
        
        if target_partitions is None:
            target_partitions = self.get_config(
                'spark.sql.shuffle.partitions', 
                200
            )
            
        if current_partitions != target_partitions:
            self.logger.debug(
                f"Repartitioning from {current_partitions} to {target_partitions}"
            )
            return df.repartition(target_partitions)
            
        return df


class PandasComponent(PipelineComponent):
    """
    Base class for Pandas-based pipeline components.
    
    Extends PipelineComponent with Pandas-specific functionality:
    - DataFrame operations for single-node processing
    - Memory management
    - Model training utilities
    """
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None):
        """
        Initialize the Pandas component.
        
        Args:
            config: Configuration dictionary
            name: Optional component name
        """
        super().__init__(config, name)
        
    def validate(self) -> bool:
        """Default validation - always passes for Pandas components."""
        return True
    
    def check_memory_usage(self, df: Any) -> Dict[str, Any]:
        """
        Check memory usage of a Pandas DataFrame.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Dictionary with memory usage information
        """
        memory_usage = df.memory_usage(deep=True)
        total_bytes = memory_usage.sum()
        
        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'per_column': memory_usage.to_dict()
        }
    
    def optimize_dtypes(self, df: Any) -> Any:
        """
        Optimize DataFrame dtypes to reduce memory usage.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Optimized DataFrame
        """
        import pandas as pd
        import numpy as np
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if low cardinality
                num_unique = df[col].nunique()
                num_total = len(df[col])
                
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
                    
            elif col_type == 'float64':
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
                
            elif col_type == 'int64':
                # Downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')
                
        return df


class ComponentRegistry:
    """
    Registry for pipeline components.
    
    Allows dynamic component registration and retrieval,
    enabling plug-and-play architecture.
    """
    
    _components: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, component_class: type) -> None:
        """
        Register a component class.
        
        Args:
            name: Unique name for the component
            component_class: Component class to register
        """
        if not issubclass(component_class, PipelineComponent):
            raise TypeError(
                f"{component_class.__name__} must be a subclass of PipelineComponent"
            )
        cls._components[name] = component_class
        
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """
        Get a registered component class.
        
        Args:
            name: Component name
            
        Returns:
            Component class or None if not found
        """
        return cls._components.get(name)
    
    @classmethod
    def list_components(cls) -> List[str]:
        """List all registered component names."""
        return list(cls._components.keys())
    
    @classmethod
    def create(
        cls, 
        name: str, 
        config: Dict[str, Any], 
        **kwargs
    ) -> PipelineComponent:
        """
        Create a component instance.
        
        Args:
            name: Component name
            config: Configuration dictionary
            **kwargs: Additional arguments for the component
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If component not found
        """
        component_class = cls.get(name)
        
        if component_class is None:
            raise ValueError(f"Component '{name}' not found in registry")
            
        return component_class(config, **kwargs)


def register_component(name: str):
    """
    Decorator to register a component class.
    
    Usage:
        @register_component('my_component')
        class MyComponent(PipelineComponent):
            ...
    """
    def decorator(cls):
        ComponentRegistry.register(name, cls)
        return cls
    return decorator
