"""
Base Data Reader

Abstract base class for all data readers.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from src.core.base import SparkComponent


class BaseDataReader(SparkComponent):
    """
    Abstract base class for data readers.
    
    All data readers (BigQuery, GCS, local) inherit from this class.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        name: Optional[str] = None
    ):
        """
        Initialize the data reader.
        
        Args:
            config: Data configuration dictionary
            spark_session: Active SparkSession
            name: Optional reader name
        """
        super().__init__(config, spark_session, name)
        
    @abstractmethod
    def read(
        self,
        source: str,
        **kwargs
    ) -> Any:
        """
        Read data from the source.
        
        Args:
            source: Data source identifier (table name, path, etc.)
            **kwargs: Additional reading options
            
        Returns:
            Spark DataFrame
        """
        pass
    
    @abstractmethod
    def read_with_filter(
        self,
        source: str,
        filter_column: str,
        filter_values: List[Any],
        **kwargs
    ) -> Any:
        """
        Read data with filtering.
        
        Args:
            source: Data source identifier
            filter_column: Column to filter on
            filter_values: Values to include
            **kwargs: Additional reading options
            
        Returns:
            Filtered Spark DataFrame
        """
        pass
    
    def validate(self) -> bool:
        """Validate the reader is properly configured."""
        return super().validate()
    
    def run(self, *args, **kwargs) -> Any:
        """Run is implemented as read for data readers."""
        return self.read(*args, **kwargs)
