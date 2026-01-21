"""
Configuration Manager

Handles loading, merging, and validating configuration files.
Supports hierarchical configuration with environment-specific overrides.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import yaml
from pydantic import BaseModel, Field, validator
from pydantic import ValidationError as PydanticValidationError

from src.core.exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class SparkConfig(BaseModel):
    """Spark configuration schema."""
    app_name: str = "CreditScoringPipeline"
    master: str = "local[*]"
    config: Dict[str, Any] = Field(default_factory=dict)


class GCPConfig(BaseModel):
    """GCP configuration schema."""
    project_id: str
    region: str = "europe-west1"
    bigquery: Dict[str, str] = Field(default_factory=dict)
    storage: Dict[str, str] = Field(default_factory=dict)
    dataproc: Dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Logging configuration schema."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Pipeline configuration schema."""
    random_state: int = 42
    date_format: str = "%Y-%m-%d"
    max_rows_for_pandas: int = 1000000


class ConfigManager:
    """
    Manages configuration loading and access.
    
    Features:
    - Hierarchical config loading (base + environment overrides)
    - Environment variable interpolation
    - Dot notation access
    - Schema validation with Pydantic
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path] = "config",
        environment: str = "dev"
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (dev, staging, prod)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment
        self._config: Dict[str, Any] = {}
        self._loaded = False
        
        # Load configurations
        self._load_configs()
        
    def _load_configs(self) -> None:
        """Load and merge all configuration files."""
        try:
            # Load base configs
            base_config = self._load_yaml(self.config_dir / "base_config.yaml")
            data_config = self._load_yaml(self.config_dir / "data_config.yaml")
            quality_config = self._load_yaml(self.config_dir / "quality_config.yaml")
            model_config = self._load_yaml(self.config_dir / "model_config.yaml")
            
            # Load optional feature config
            feature_config_path = self.config_dir / "feature_config.yaml"
            feature_config = self._load_yaml(feature_config_path) if feature_config_path.exists() else {}
            
            # Load environment-specific overrides
            env_config_path = self.config_dir / "environments" / f"{self.environment}.yaml"
            env_config = self._load_yaml(env_config_path) if env_config_path.exists() else {}
            
            # Merge all configs
            self._config = self._deep_merge(
                base_config,
                {"data": data_config},
                {"quality": quality_config},
                {"model": model_config},
                {"features": feature_config} if feature_config else {},
                env_config
            )
            
            # Interpolate environment variables
            self._config = self._interpolate_env_vars(self._config)
            
            self._loaded = True
            logger.info(f"Configuration loaded for environment: {self.environment}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            path: Path to the YAML file
            
        Returns:
            Dictionary with configuration
        """
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return {}
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                return content if content else {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}")
    
    def _deep_merge(self, *dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge multiple dictionaries.
        
        Later dictionaries override earlier ones.
        
        Args:
            *dicts: Dictionaries to merge
            
        Returns:
            Merged dictionary
        """
        result: Dict[str, Any] = {}
        
        for d in dicts:
            if not d:
                continue
                
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = value
                    
        return result
    
    def _interpolate_env_vars(self, config: Any) -> Any:
        """
        Interpolate environment variables in configuration values.
        
        Supports ${VAR_NAME} syntax.
        
        Args:
            config: Configuration value (dict, list, or scalar)
            
        Returns:
            Configuration with interpolated values
        """
        if isinstance(config, dict):
            return {k: self._interpolate_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._interpolate_env_vars(v) for v in config]
        elif isinstance(config, str):
            # Find all ${VAR_NAME} patterns
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, config)
            
            for var_name in matches:
                env_value = os.environ.get(var_name, "")
                config = config.replace(f"${{{var_name}}}", env_value)
                
            return config
        else:
            return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.xgboost.max_depth')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Enable bracket notation: config['key']."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key not found: {key}")
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name (e.g., 'model', 'data')
            
        Returns:
            Section configuration dictionary
        """
        return self.get(section, {})
    
    @property
    def spark_config(self) -> Dict[str, Any]:
        """Get Spark configuration."""
        return self.get_section("spark")
    
    @property
    def gcp_config(self) -> Dict[str, Any]:
        """Get GCP configuration."""
        return self.get_section("gcp")
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get_section("data")
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get_section("model")
    
    @property
    def quality_config(self) -> Dict[str, Any]:
        """Get quality configuration."""
        return self.get_section("quality")
    
    @property
    def feature_config(self) -> Dict[str, Any]:
        """Get feature configuration."""
        return self.get_section("features")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary."""
        return self._config.copy()
    
    def validate_required_keys(self, required_keys: List[str]) -> bool:
        """
        Validate that required configuration keys are present.
        
        Args:
            required_keys: List of required keys (dot notation supported)
            
        Returns:
            True if all keys present
            
        Raises:
            ConfigurationError: If any required key is missing
        """
        missing_keys = []
        
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
                
        if missing_keys:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing_keys}"
            )
            
        return True
    
    def reload(self) -> None:
        """Reload all configuration files."""
        self._config = {}
        self._loaded = False
        self._load_configs()
        logger.info("Configuration reloaded")
    
    def set_environment(self, environment: str) -> None:
        """
        Change the environment and reload configuration.
        
        Args:
            environment: New environment name
        """
        self.environment = environment
        self.reload()
        
    def __repr__(self) -> str:
        return f"ConfigManager(environment={self.environment}, loaded={self._loaded})"
