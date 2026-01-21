"""
Data Quality Checker

Performs comprehensive data quality checks on raw data.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from pyspark.sql import functions as F

from src.core.base import SparkComponent
from src.core.exceptions import DataQualityError


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""
    check_name: str
    check_type: str
    passed: bool
    severity: str  # error, warning, info
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'check_name': self.check_name,
            'check_type': self.check_type,
            'passed': self.passed,
            'severity': self.severity,
            'message': self.message,
            'details': self.details
        }


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    table_name: str
    timestamp: datetime
    total_rows: int
    checks: List[QualityCheckResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Check if all critical checks passed."""
        return all(
            c.passed for c in self.checks 
            if c.severity == 'error'
        )
    
    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == 'error')
    
    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == 'warning')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'table_name': self.table_name,
            'timestamp': self.timestamp.isoformat(),
            'total_rows': self.total_rows,
            'passed': self.passed,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'checks': [c.to_dict() for c in self.checks]
        }


class DataQualityChecker(SparkComponent):
    """
    Performs data quality checks on Spark DataFrames.
    
    Checks include:
    - Null value analysis
    - Range validation
    - Date consistency
    - Uniqueness
    - Categorical value validation
    - Duplicate detection
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        name: Optional[str] = None
    ):
        """
        Initialize the data quality checker.
        
        Args:
            config: Configuration dictionary
            spark_session: Active SparkSession
            name: Optional checker name
        """
        super().__init__(config, spark_session, name or "DataQualityChecker")
        
        # Load quality config
        self.quality_config = self.get_config('quality.data_quality', {})
        self.fail_on_critical = self.quality_config.get('fail_on_critical_errors', True)
        
    def validate(self) -> bool:
        """Validate checker configuration."""
        return super().validate()
    
    def run(self, df: Any, table_name: str) -> DataQualityReport:
        """Run all quality checks."""
        return self.check_all(df, table_name)
    
    def check_all(
        self,
        df: Any,
        table_name: str
    ) -> DataQualityReport:
        """
        Run all configured quality checks.
        
        Args:
            df: Spark DataFrame to check
            table_name: Name of the table for reporting
            
        Returns:
            DataQualityReport with all check results
        """
        self._start_execution()
        
        total_rows = df.count()
        self.logger.info(f"Running quality checks on {table_name} ({total_rows:,} rows)")
        
        report = DataQualityReport(
            table_name=table_name,
            timestamp=datetime.now(),
            total_rows=total_rows
        )
        
        checks_config = self.quality_config.get('checks', {})
        
        # Run null checks
        if 'null_checks' in checks_config:
            null_results = self._run_null_checks(df, table_name, checks_config['null_checks'])
            report.checks.extend(null_results)
        
        # Run range checks
        if 'range_checks' in checks_config:
            range_results = self._run_range_checks(df, table_name, checks_config['range_checks'])
            report.checks.extend(range_results)
        
        # Run categorical checks
        if 'categorical_checks' in checks_config:
            cat_results = self._run_categorical_checks(df, table_name, checks_config['categorical_checks'])
            report.checks.extend(cat_results)
        
        # Run duplicate checks
        if 'duplicate_checks' in checks_config:
            dup_results = self._run_duplicate_checks(df, table_name, checks_config['duplicate_checks'])
            report.checks.extend(dup_results)
        
        # Log summary
        self.logger.info(
            f"Quality check complete: {len(report.checks)} checks, "
            f"{report.error_count} errors, {report.warning_count} warnings"
        )
        
        self._end_execution()
        
        # Raise if critical errors and configured to fail
        if self.fail_on_critical and not report.passed:
            raise DataQualityError(
                f"Data quality checks failed for {table_name}",
                quality_report=report.to_dict()
            )
        
        return report
    
    def _run_null_checks(
        self,
        df: Any,
        table_name: str,
        config: Dict[str, Any]
    ) -> List[QualityCheckResult]:
        """Run null value checks."""
        results = []
        severity = config.get('severity', 'error')
        rules = config.get('rules', [])
        
        for rule in rules:
            # Skip rules for other tables
            if rule.get('table') and rule['table'] != table_name:
                continue
                
            column = rule['column']
            max_null_ratio = rule.get('max_null_ratio', 0.0)
            
            if column not in df.columns:
                continue
            
            # Calculate null ratio
            total = df.count()
            null_count = df.filter(F.col(column).isNull()).count()
            null_ratio = null_count / total if total > 0 else 0
            
            passed = null_ratio <= max_null_ratio
            
            results.append(QualityCheckResult(
                check_name=f"null_check_{column}",
                check_type="null_check",
                passed=passed,
                severity=severity,
                message=f"Column '{column}' null ratio: {null_ratio:.4f} (max: {max_null_ratio})",
                details={
                    'column': column,
                    'null_count': null_count,
                    'null_ratio': null_ratio,
                    'max_null_ratio': max_null_ratio
                }
            ))
            
            if not passed:
                self.logger.warning(
                    f"Null check failed for {column}: {null_ratio:.4f} > {max_null_ratio}"
                )
        
        return results
    
    def _run_range_checks(
        self,
        df: Any,
        table_name: str,
        config: Dict[str, Any]
    ) -> List[QualityCheckResult]:
        """Run range validation checks."""
        results = []
        severity = config.get('severity', 'warning')
        rules = config.get('rules', [])
        
        for rule in rules:
            if rule.get('table') and rule['table'] != table_name:
                continue
                
            column = rule['column']
            min_val = rule.get('min')
            max_val = rule.get('max')
            
            if column not in df.columns:
                continue
            
            # Count violations
            total = df.count()
            violations = 0
            
            if min_val is not None:
                violations += df.filter(F.col(column) < min_val).count()
            if max_val is not None:
                violations += df.filter(F.col(column) > max_val).count()
            
            violation_ratio = violations / total if total > 0 else 0
            passed = violations == 0
            
            results.append(QualityCheckResult(
                check_name=f"range_check_{column}",
                check_type="range_check",
                passed=passed,
                severity=severity,
                message=f"Column '{column}' range violations: {violations} ({violation_ratio:.4f})",
                details={
                    'column': column,
                    'min': min_val,
                    'max': max_val,
                    'violations': violations,
                    'violation_ratio': violation_ratio
                }
            ))
        
        return results
    
    def _run_categorical_checks(
        self,
        df: Any,
        table_name: str,
        config: Dict[str, Any]
    ) -> List[QualityCheckResult]:
        """Run categorical value checks."""
        results = []
        severity = config.get('severity', 'error')
        rules = config.get('rules', [])
        
        for rule in rules:
            if rule.get('table') and rule['table'] != table_name:
                continue
                
            column = rule['column']
            allowed_values = rule.get('allowed_values', [])
            
            if column not in df.columns or not allowed_values:
                continue
            
            # Find invalid values
            total = df.count()
            invalid_count = df.filter(~F.col(column).isin(allowed_values)).count()
            
            passed = invalid_count == 0
            
            results.append(QualityCheckResult(
                check_name=f"categorical_check_{column}",
                check_type="categorical_check",
                passed=passed,
                severity=severity,
                message=f"Column '{column}' invalid values: {invalid_count}",
                details={
                    'column': column,
                    'allowed_values': allowed_values,
                    'invalid_count': invalid_count
                }
            ))
            
            if not passed:
                # Get sample of invalid values
                invalid_sample = (
                    df.filter(~F.col(column).isin(allowed_values))
                    .select(column)
                    .distinct()
                    .limit(5)
                    .collect()
                )
                self.logger.warning(
                    f"Invalid values in {column}: {[r[0] for r in invalid_sample]}"
                )
        
        return results
    
    def _run_duplicate_checks(
        self,
        df: Any,
        table_name: str,
        config: Dict[str, Any]
    ) -> List[QualityCheckResult]:
        """Run duplicate detection checks."""
        results = []
        severity = config.get('severity', 'warning')
        rules = config.get('rules', [])
        
        for rule in rules:
            if rule.get('table') and rule['table'] != table_name:
                continue
            
            columns = rule.get('columns', 'all')
            max_dup_ratio = rule.get('max_duplicate_ratio', 0.01)
            
            # Calculate duplicates
            total = df.count()
            
            if columns == 'all':
                distinct_count = df.distinct().count()
            else:
                distinct_count = df.select(columns).distinct().count()
            
            duplicate_count = total - distinct_count
            dup_ratio = duplicate_count / total if total > 0 else 0
            
            passed = dup_ratio <= max_dup_ratio
            
            results.append(QualityCheckResult(
                check_name=f"duplicate_check",
                check_type="duplicate_check",
                passed=passed,
                severity=severity,
                message=f"Duplicate ratio: {dup_ratio:.4f} (max: {max_dup_ratio})",
                details={
                    'columns': columns,
                    'total_rows': total,
                    'distinct_rows': distinct_count,
                    'duplicate_count': duplicate_count,
                    'duplicate_ratio': dup_ratio
                }
            ))
        
        return results
    
    def check_nulls(
        self,
        df: Any,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Quick null analysis for specified columns.
        
        Args:
            df: Spark DataFrame
            columns: Columns to analyze (all if None)
            
        Returns:
            Dictionary with null statistics per column
        """
        if columns is None:
            columns = df.columns
        
        total = df.count()
        results = {}
        
        for col in columns:
            null_count = df.filter(F.col(col).isNull()).count()
            results[col] = {
                'null_count': null_count,
                'null_ratio': null_count / total if total > 0 else 0,
                'non_null_count': total - null_count
            }
        
        return results
    
    def get_column_stats(
        self,
        df: Any,
        column: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a single column.
        
        Args:
            df: Spark DataFrame
            column: Column name
            
        Returns:
            Dictionary with column statistics
        """
        stats = df.select(
            F.count(column).alias('count'),
            F.countDistinct(column).alias('distinct'),
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias('nulls')
        ).collect()[0]
        
        result = {
            'count': stats['count'],
            'distinct': stats['distinct'],
            'nulls': stats['nulls'],
            'null_ratio': stats['nulls'] / df.count() if df.count() > 0 else 0
        }
        
        # Add numeric stats if applicable
        dtype = str(df.schema[column].dataType)
        if 'Int' in dtype or 'Long' in dtype or 'Double' in dtype or 'Float' in dtype:
            num_stats = df.select(
                F.min(column).alias('min'),
                F.max(column).alias('max'),
                F.avg(column).alias('mean'),
                F.stddev(column).alias('std')
            ).collect()[0]
            
            result.update({
                'min': num_stats['min'],
                'max': num_stats['max'],
                'mean': num_stats['mean'],
                'std': num_stats['std']
            })
        
        return result
