"""
Data Splitter

Handles train/test/validation splitting with stratification support.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.core.base import SparkComponent


@dataclass
class DataSplit:
    """Container for data splits."""
    train: Any  # DataFrame
    test: Any  # DataFrame
    validation: Optional[Any] = None  # DataFrame
    
    @property
    def has_validation(self) -> bool:
        return self.validation is not None


class DataSplitter(SparkComponent):
    """
    Handles train/test/validation splitting of data.
    
    Supports:
    - Stratified splitting by target variable
    - Reproducible splits with random seed
    - Optional validation set
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        name: Optional[str] = None
    ):
        """
        Initialize the data splitter.
        
        Args:
            config: Configuration dictionary
            spark_session: Active SparkSession
            name: Optional splitter name
        """
        super().__init__(config, spark_session, name or "DataSplitter")
        
        # Get split settings from config
        training_config = self.get_config('model.training', {})
        self.test_size = training_config.get('test_size', 0.2)
        self.validation_size = training_config.get('validation_size', 0.1)
        self.stratify = training_config.get('stratify', True)
        self.random_state = training_config.get(
            'random_state',
            self.get_config('pipeline.random_state', 42)
        )
        
    def validate(self) -> bool:
        """Validate split configuration."""
        if not super().validate():
            return False
            
        total_split = self.test_size + self.validation_size
        if total_split >= 1.0:
            self.logger.error(
                f"test_size ({self.test_size}) + validation_size ({self.validation_size}) >= 1.0"
            )
            return False
            
        return True
    
    def run(
        self,
        df: Any,
        target_column: str = 'target'
    ) -> DataSplit:
        """Run the splitting."""
        return self.split(df, target_column)
    
    def split(
        self,
        df: Any,
        target_column: str = 'target',
        include_validation: bool = True
    ) -> DataSplit:
        """
        Split data into train/test/validation sets.
        
        Args:
            df: Spark DataFrame to split
            target_column: Name of target column for stratification
            include_validation: Whether to create validation set
            
        Returns:
            DataSplit containing train, test, and optionally validation DataFrames
        """
        self._start_execution()
        
        total_count = df.count()
        self.logger.info(f"Splitting {total_count:,} rows")
        
        if self.stratify and target_column in df.columns:
            result = self._stratified_split(
                df, target_column, include_validation
            )
        else:
            result = self._random_split(df, include_validation)
        
        # Log split sizes
        train_count = result.train.count()
        test_count = result.test.count()
        
        self.logger.info(f"Train set: {train_count:,} rows ({train_count/total_count*100:.1f}%)")
        self.logger.info(f"Test set: {test_count:,} rows ({test_count/total_count*100:.1f}%)")
        
        if result.has_validation:
            val_count = result.validation.count()
            self.logger.info(f"Validation set: {val_count:,} rows ({val_count/total_count*100:.1f}%)")
        
        self._end_execution()
        return result
    
    def _random_split(
        self,
        df: Any,
        include_validation: bool
    ) -> DataSplit:
        """
        Perform random split without stratification.
        
        Args:
            df: DataFrame to split
            include_validation: Whether to include validation set
            
        Returns:
            DataSplit
        """
        if include_validation:
            train_ratio = 1.0 - self.test_size - self.validation_size
            weights = [train_ratio, self.validation_size, self.test_size]
            splits = df.randomSplit(weights, seed=self.random_state)
            return DataSplit(train=splits[0], validation=splits[1], test=splits[2])
        else:
            train_ratio = 1.0 - self.test_size
            weights = [train_ratio, self.test_size]
            splits = df.randomSplit(weights, seed=self.random_state)
            return DataSplit(train=splits[0], test=splits[1])
    
    def _stratified_split(
        self,
        df: Any,
        target_column: str,
        include_validation: bool
    ) -> DataSplit:
        """
        Perform stratified split maintaining target distribution.
        
        Args:
            df: DataFrame to split
            target_column: Column to stratify by
            include_validation: Whether to include validation set
            
        Returns:
            DataSplit
        """
        from pyspark.sql import functions as F
        
        # Get target distribution
        target_dist = df.groupBy(target_column).count().collect()
        self.logger.debug(f"Target distribution: {target_dist}")
        
        # Add random column for splitting
        df_with_rand = df.withColumn(
            "_split_rand",
            F.rand(seed=self.random_state)
        )
        
        # Calculate cumulative thresholds
        test_threshold = self.test_size
        if include_validation:
            val_threshold = self.test_size + self.validation_size
            train_threshold = 1.0
            
            # Split based on random values
            test_df = df_with_rand.filter(
                F.col("_split_rand") < test_threshold
            ).drop("_split_rand")
            
            val_df = df_with_rand.filter(
                (F.col("_split_rand") >= test_threshold) & 
                (F.col("_split_rand") < val_threshold)
            ).drop("_split_rand")
            
            train_df = df_with_rand.filter(
                F.col("_split_rand") >= val_threshold
            ).drop("_split_rand")
            
            return DataSplit(train=train_df, validation=val_df, test=test_df)
        else:
            test_df = df_with_rand.filter(
                F.col("_split_rand") < test_threshold
            ).drop("_split_rand")
            
            train_df = df_with_rand.filter(
                F.col("_split_rand") >= test_threshold
            ).drop("_split_rand")
            
            return DataSplit(train=train_df, test=test_df)
    
    def to_pandas(
        self,
        data_split: DataSplit,
        max_rows: Optional[int] = None
    ) -> Tuple[Any, Any, Optional[Any]]:
        """
        Convert DataSplit to Pandas DataFrames.
        
        Args:
            data_split: DataSplit with Spark DataFrames
            max_rows: Maximum rows to convert (samples if exceeded)
            
        Returns:
            Tuple of (train_pd, test_pd, validation_pd or None)
        """
        max_rows = max_rows or self.get_config('pipeline.max_rows_for_pandas', 1000000)
        
        def convert_or_sample(spark_df: Any, name: str) -> Any:
            count = spark_df.count()
            if count > max_rows:
                self.logger.warning(
                    f"{name} has {count:,} rows, sampling to {max_rows:,}"
                )
                fraction = max_rows / count
                spark_df = spark_df.sample(fraction=fraction, seed=self.random_state)
            
            return spark_df.toPandas()
        
        self.logger.info("Converting Spark DataFrames to Pandas")
        
        train_pd = convert_or_sample(data_split.train, "Train set")
        test_pd = convert_or_sample(data_split.test, "Test set")
        
        val_pd = None
        if data_split.has_validation:
            val_pd = convert_or_sample(data_split.validation, "Validation set")
        
        return train_pd, test_pd, val_pd
