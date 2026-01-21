"""
BigQuery Data Reader

Reads data from Google BigQuery using Spark BigQuery connector.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from src.data.base_reader import BaseDataReader
from src.core.exceptions import DataReaderError


class BigQueryReader(BaseDataReader):
    """
    Reads data from Google BigQuery.
    
    Uses the Spark BigQuery connector for distributed reading.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        project_id: Optional[str] = None,
        dataset: Optional[str] = None
    ):
        """
        Initialize the BigQuery reader.
        
        Args:
            config: Configuration dictionary
            spark_session: Active SparkSession
            project_id: GCP project ID (optional, uses config if not provided)
            dataset: BigQuery dataset (optional, uses config if not provided)
        """
        super().__init__(config, spark_session, name="BigQueryReader")
        
        # Get GCP settings from config or parameters
        gcp_config = self.get_config('gcp', {})
        self.project_id = project_id or gcp_config.get('project_id')
        self.dataset = dataset or gcp_config.get('bigquery', {}).get('dataset')
        
        # Materialization settings
        bq_config = self.get_config('spark.bigquery', {})
        self.materialization_dataset = bq_config.get(
            'materializationDataset', 
            'temp_spark_bq'
        )
        
    def validate(self) -> bool:
        """Validate BigQuery connection settings."""
        if not super().validate():
            return False
            
        if not self.project_id:
            self.logger.error("GCP project_id not configured")
            return False
            
        if not self.dataset:
            self.logger.error("BigQuery dataset not configured")
            return False
            
        return True
    
    def _get_full_table_name(self, table: str) -> str:
        """Get fully qualified table name."""
        if '.' in table:
            return table  # Already fully qualified
        return f"{self.project_id}.{self.dataset}.{table}"
    
    def read(
        self,
        source: str,
        columns: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        sample_fraction: Optional[float] = None
    ) -> Any:
        """
        Read data from BigQuery table.
        
        Args:
            source: Table name (can be simple name or fully qualified)
            columns: Optional list of columns to select
            filter_expr: Optional SQL filter expression
            sample_fraction: Optional sampling fraction (0.0 to 1.0)
            
        Returns:
            Spark DataFrame
        """
        self._start_execution()
        
        try:
            full_table = self._get_full_table_name(source)
            self.logger.info(f"Reading from BigQuery: {full_table}")
            
            # Build read options
            reader = self.spark.read.format("bigquery")
            reader = reader.option("table", full_table)
            reader = reader.option(
                "materializationDataset", 
                self.materialization_dataset
            )
            
            # Apply column projection if specified
            if columns:
                reader = reader.option("selectedFields", ",".join(columns))
            
            # Apply filter if specified
            if filter_expr:
                reader = reader.option("filter", filter_expr)
            
            # Read the data
            df = reader.load()
            
            # Apply sampling if specified
            if sample_fraction and 0 < sample_fraction < 1:
                seed = self.get_config('pipeline.random_state', 42)
                df = df.sample(fraction=sample_fraction, seed=seed)
                self.logger.info(f"Sampled {sample_fraction*100:.1f}% of data")
            
            row_count = df.count()
            self.logger.info(f"Read {row_count:,} rows from {source}")
            
            self._end_execution()
            return df
            
        except Exception as e:
            self._end_execution()
            raise DataReaderError(
                f"Failed to read from BigQuery table: {source}",
                source=source,
                cause=e
            )
    
    def read_with_filter(
        self,
        source: str,
        filter_column: str,
        filter_values: List[Any],
        **kwargs
    ) -> Any:
        """
        Read data with column filtering.
        
        Args:
            source: Table name
            filter_column: Column to filter on
            filter_values: Values to include
            **kwargs: Additional read options
            
        Returns:
            Filtered Spark DataFrame
        """
        # Build IN clause for filter
        if isinstance(filter_values[0], str):
            values_str = ", ".join([f"'{v}'" for v in filter_values])
        else:
            values_str = ", ".join([str(v) for v in filter_values])
            
        filter_expr = f"{filter_column} IN ({values_str})"
        
        # Combine with any existing filter
        existing_filter = kwargs.pop('filter_expr', None)
        if existing_filter:
            filter_expr = f"({existing_filter}) AND ({filter_expr})"
        
        return self.read(source, filter_expr=filter_expr, **kwargs)
    
    def read_with_date_range(
        self,
        source: str,
        date_column: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Any:
        """
        Read data with date range filtering.
        
        Args:
            source: Table name
            date_column: Date column to filter on
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            **kwargs: Additional read options
            
        Returns:
            Filtered Spark DataFrame
        """
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        filter_expr = f"{date_column} >= '{start_str}' AND {date_column} <= '{end_str}'"
        
        # Combine with any existing filter
        existing_filter = kwargs.pop('filter_expr', None)
        if existing_filter:
            filter_expr = f"({existing_filter}) AND ({filter_expr})"
        
        self.logger.info(f"Reading {source} for date range: {start_str} to {end_str}")
        
        return self.read(source, filter_expr=filter_expr, **kwargs)
    
    def read_applications(
        self,
        sample_fraction: Optional[float] = None,
        date_range: Optional[tuple] = None
    ) -> Any:
        """
        Read applications table with optional filtering.
        
        Args:
            sample_fraction: Optional sampling fraction
            date_range: Optional (start_date, end_date) tuple
            
        Returns:
            Applications DataFrame
        """
        source = self.get_config('data.data_sources.applications.table', 'applications')
        
        if date_range:
            return self.read_with_date_range(
                source,
                date_column='application_date',
                start_date=date_range[0],
                end_date=date_range[1],
                sample_fraction=sample_fraction
            )
        
        return self.read(source, sample_fraction=sample_fraction)
    
    def read_credit_bureau(
        self,
        application_ids: Optional[List[str]] = None,
        sample_fraction: Optional[float] = None
    ) -> Any:
        """
        Read credit bureau table with optional filtering.
        
        Args:
            application_ids: Optional list of application IDs to filter
            sample_fraction: Optional sampling fraction
            
        Returns:
            Credit bureau DataFrame
        """
        source = self.get_config(
            'data.data_sources.credit_bureau.table', 
            'credit_bureau'
        )
        
        if application_ids:
            return self.read_with_filter(
                source,
                filter_column='application_id',
                filter_values=application_ids,
                sample_fraction=sample_fraction
            )
        
        return self.read(source, sample_fraction=sample_fraction)
