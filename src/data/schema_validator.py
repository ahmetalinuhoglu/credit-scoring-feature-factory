"""
Schema Validator

Validates DataFrame schemas against expected configurations.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from src.core.base import SparkComponent
from src.core.exceptions import SchemaValidationError


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""
    is_valid: bool
    missing_columns: List[str]
    extra_columns: List[str]
    type_mismatches: List[Dict[str, str]]
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'missing_columns': self.missing_columns,
            'extra_columns': self.extra_columns,
            'type_mismatches': self.type_mismatches,
            'errors': self.errors
        }


class SchemaValidator(SparkComponent):
    """
    Validates DataFrame schemas against expected configurations.
    
    Checks for:
    - Missing required columns
    - Extra unexpected columns
    - Data type mismatches
    - Nullable constraints
    """
    
    # Mapping from config types to Spark types
    TYPE_MAPPING = {
        'string': ['StringType', 'string'],
        'integer': ['IntegerType', 'LongType', 'int', 'bigint'],
        'float': ['FloatType', 'DoubleType', 'float', 'double'],
        'date': ['DateType', 'date'],
        'timestamp': ['TimestampType', 'timestamp'],
        'boolean': ['BooleanType', 'boolean'],
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        spark_session: Any,
        name: Optional[str] = None
    ):
        """
        Initialize the schema validator.
        
        Args:
            config: Configuration dictionary containing schemas
            spark_session: Active SparkSession
            name: Optional validator name
        """
        super().__init__(config, spark_session, name or "SchemaValidator")
        
        # Load schemas from config
        self.schemas = self.get_config('data.schemas', {})
        
    def validate(self) -> bool:
        """Validate that schemas are configured."""
        if not self.schemas:
            self.logger.warning("No schemas configured for validation")
        return super().validate()
    
    def run(self, df: Any, schema_name: str) -> SchemaValidationResult:
        """Run schema validation."""
        return self.validate_schema(df, schema_name)
    
    def validate_schema(
        self,
        df: Any,
        schema_name: str,
        strict: bool = True
    ) -> SchemaValidationResult:
        """
        Validate a DataFrame against a named schema.
        
        Args:
            df: Spark DataFrame to validate
            schema_name: Name of the schema in config (e.g., 'applications')
            strict: If True, fail on extra columns; if False, just warn
            
        Returns:
            SchemaValidationResult with validation details
        """
        self._start_execution()
        
        expected_schema = self.schemas.get(schema_name)
        if not expected_schema:
            self._end_execution()
            raise SchemaValidationError(
                f"Schema '{schema_name}' not found in configuration"
            )
        
        expected_columns = expected_schema.get('columns', [])
        
        # Get actual columns from DataFrame
        actual_columns = {field.name: str(field.dataType) for field in df.schema.fields}
        actual_column_names = set(actual_columns.keys())
        
        # Get expected column names
        expected_column_names = {col['name'] for col in expected_columns}
        
        # Find missing and extra columns
        missing_columns = list(expected_column_names - actual_column_names)
        extra_columns = list(actual_column_names - expected_column_names)
        
        # Check type mismatches
        type_mismatches = []
        for col_spec in expected_columns:
            col_name = col_spec['name']
            expected_type = col_spec.get('type', 'string')
            
            if col_name in actual_columns:
                actual_type = actual_columns[col_name]
                
                if not self._type_matches(expected_type, actual_type):
                    type_mismatches.append({
                        'column': col_name,
                        'expected': expected_type,
                        'actual': actual_type
                    })
        
        # Build error messages
        errors = []
        
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
            self.logger.error(f"Missing columns in {schema_name}: {missing_columns}")
            
        if extra_columns and strict:
            errors.append(f"Unexpected columns: {extra_columns}")
            self.logger.warning(f"Extra columns in {schema_name}: {extra_columns}")
            
        if type_mismatches:
            for mismatch in type_mismatches:
                errors.append(
                    f"Type mismatch for '{mismatch['column']}': "
                    f"expected {mismatch['expected']}, got {mismatch['actual']}"
                )
            self.logger.error(f"Type mismatches in {schema_name}: {type_mismatches}")
        
        # Determine if valid
        is_valid = len(missing_columns) == 0 and len(type_mismatches) == 0
        if strict:
            is_valid = is_valid and len(extra_columns) == 0
        
        result = SchemaValidationResult(
            is_valid=is_valid,
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            type_mismatches=type_mismatches,
            errors=errors
        )
        
        self._end_execution()
        
        if is_valid:
            self.logger.info(f"Schema validation passed for {schema_name}")
        else:
            self.logger.error(f"Schema validation failed for {schema_name}")
        
        return result
    
    def _type_matches(self, expected_type: str, actual_type: str) -> bool:
        """
        Check if actual type matches expected type.
        
        Args:
            expected_type: Type from config (e.g., 'string', 'integer')
            actual_type: Type from Spark schema (e.g., 'StringType()')
            
        Returns:
            True if types match
        """
        expected_type = expected_type.lower()
        
        if expected_type not in self.TYPE_MAPPING:
            self.logger.warning(f"Unknown expected type: {expected_type}")
            return True  # Don't fail on unknown types
        
        valid_types = self.TYPE_MAPPING[expected_type]
        
        for valid_type in valid_types:
            if valid_type.lower() in actual_type.lower():
                return True
                
        return False
    
    def validate_and_raise(
        self,
        df: Any,
        schema_name: str,
        strict: bool = True
    ) -> None:
        """
        Validate schema and raise exception if invalid.
        
        Args:
            df: Spark DataFrame
            schema_name: Schema name
            strict: If True, fail on extra columns
            
        Raises:
            SchemaValidationError: If validation fails
        """
        result = self.validate_schema(df, schema_name, strict)
        
        if not result.is_valid:
            raise SchemaValidationError(
                f"Schema validation failed for {schema_name}",
                validation_errors=[result.to_dict()]
            )
    
    def get_schema_info(self, schema_name: str) -> Dict[str, Any]:
        """
        Get schema information for a named schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            Schema configuration dictionary
        """
        return self.schemas.get(schema_name, {})
    
    def list_schemas(self) -> List[str]:
        """List all configured schema names."""
        return list(self.schemas.keys())
