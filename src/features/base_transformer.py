"""
Base Transformer

Abstract base class for feature transformers.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from src.core.base import SparkComponent


class BaseTransformer(SparkComponent):
    """
    Abstract base class for feature transformers.
    
    All feature transformers inherit from this class.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        name: Optional[str] = None
    ):
        """
        Initialize the transformer.
        
        Args:
            config: Configuration dictionary
            spark_session: Active SparkSession
            name: Optional transformer name
        """
        super().__init__(config, spark_session, name)
        self._output_columns: List[str] = []
        
    @abstractmethod
    def transform(self, df: Any) -> Any:
        """
        Apply transformation to DataFrame.
        
        Args:
            df: Input Spark DataFrame
            
        Returns:
            Transformed Spark DataFrame
        """
        pass
    
    def run(self, df: Any) -> Any:
        """Run is implemented as transform."""
        return self.transform(df)
    
    def validate(self) -> bool:
        """Default validation."""
        return super().validate()
    
    @property
    def output_columns(self) -> List[str]:
        """Get list of output column names created by this transformer."""
        return self._output_columns
    
    def _add_output_column(self, column: str) -> None:
        """Register an output column."""
        if column not in self._output_columns:
            self._output_columns.append(column)
    
    def get_feature_info(self) -> List[Dict[str, Any]]:
        """
        Get information about features created by this transformer.
        
        Returns:
            List of feature info dictionaries
        """
        return [{'name': col} for col in self._output_columns]
