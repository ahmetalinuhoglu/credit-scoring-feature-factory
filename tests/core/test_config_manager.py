"""
Tests for ConfigManager

Tests configuration loading, merging, and access patterns.
"""

import pytest
import tempfile
import os
import copy
from pathlib import Path

import yaml

from src.core.config_manager import ConfigManager
from src.core.exceptions import ConfigurationError


class TestConfigManager:
    """Test suite for ConfigManager."""
    
    @pytest.fixture
    def config_dir(self):
        """Create temporary config directory with test files."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir)
        
        # Create base config (matches expected structure)
        base_config = {
            'pipeline': {
                'random_state': 42,
                'date_format': '%Y-%m-%d',
                'max_rows_for_pandas': 100000
            },
            'spark': {
                'app_name': 'TestApp',
                'master': 'local[*]'
            }
        }
        
        with open(config_path / 'base_config.yaml', 'w') as f:
            yaml.dump(base_config, f)
        
        # Create data config (required by ConfigManager)
        data_config = {
            'sources': {
                'applications': {'table': 'apps'},
                'credit_bureau': {'table': 'bureau'}
            }
        }
        
        with open(config_path / 'data_config.yaml', 'w') as f:
            yaml.dump(data_config, f)
        
        # Create quality config (required by ConfigManager)
        quality_config = {
            'null_threshold': 0.3
        }
        
        with open(config_path / 'quality_config.yaml', 'w') as f:
            yaml.dump(quality_config, f)
        
        # Create model config
        model_config = {
            'xgboost': {
                'max_depth': 5,
                'n_estimators': 100
            }
        }
        
        with open(config_path / 'model_config.yaml', 'w') as f:
            yaml.dump(model_config, f)
        
        # Create environments directory
        env_dir = config_path / 'environments'
        env_dir.mkdir()
        
        # Create dev environment override
        dev_config = {
            'pipeline': {
                'random_state': 123  # Override for dev
            },
            'spark': {
                'master': 'local[2]'  # Override
            }
        }
        
        with open(env_dir / 'dev.yaml', 'w') as f:
            yaml.dump(dev_config, f)
        
        yield str(config_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_init_creates_config_manager(self, config_dir):
        """Test ConfigManager initialization."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        assert cm is not None
        assert cm.environment == 'dev'
    
    def test_load_yaml_file(self, config_dir):
        """Test loading a single YAML file."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        # Should have loaded base_config.yaml
        assert 'pipeline' in cm.to_dict()
        assert 'spark' in cm.to_dict()
    
    def test_deep_merge_configs(self, config_dir):
        """Test deep merging of configuration files."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        config = cm.to_dict()
        
        # Base config values should be present
        assert config['pipeline']['date_format'] == '%Y-%m-%d'
        
        # Environment override should be applied
        assert config['pipeline']['random_state'] == 123  # From dev.yaml
        assert config['spark']['master'] == 'local[2]'  # From dev.yaml
    
    def test_get_with_dot_notation(self, config_dir):
        """Test accessing nested config with dot notation."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        assert cm.get('pipeline.random_state') == 123
        assert cm.get('spark.app_name') == 'TestApp'
        # Model config is nested under 'model' key
        assert cm.get('model.xgboost.max_depth') == 5
    
    def test_get_with_default_value(self, config_dir):
        """Test get returns default for missing keys."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        assert cm.get('nonexistent.key') is None
        assert cm.get('nonexistent.key', 'default') == 'default'
        assert cm.get('pipeline.nonexistent', 999) == 999
    
    def test_get_section(self, config_dir):
        """Test getting entire config section."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        pipeline_section = cm.get_section('pipeline')
        
        assert isinstance(pipeline_section, dict)
        assert 'random_state' in pipeline_section
        assert 'date_format' in pipeline_section
    
    def test_bracket_notation_access(self, config_dir):
        """Test bracket notation for config access."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        assert cm['pipeline.random_state'] == 123
        assert cm['spark.app_name'] == 'TestApp'
    
    def test_environment_variable_interpolation(self, config_dir):
        """Test environment variable interpolation in config values."""
        # Set an environment variable
        os.environ['TEST_CONFIG_VALUE'] = 'test_value'
        
        # Create config with env var in base_config
        config_path = Path(config_dir)
        
        # Update base config to include env var
        base_config = {
            'pipeline': {
                'random_state': 42,
                'date_format': '%Y-%m-%d'
            },
            'spark': {
                'app_name': 'TestApp',
                'master': 'local[*]'
            },
            'test': {
                'value': '${TEST_CONFIG_VALUE}',
                'nested': {
                    'var': '${TEST_CONFIG_VALUE}'
                }
            }
        }
        
        with open(config_path / 'base_config.yaml', 'w') as f:
            yaml.dump(base_config, f)
        
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        assert cm.get('test.value') == 'test_value'
        assert cm.get('test.nested.var') == 'test_value'
        
        # Cleanup
        del os.environ['TEST_CONFIG_VALUE']
    
    def test_validate_required_keys(self, config_dir):
        """Test validation of required configuration keys."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        # These keys should exist
        assert cm.validate_required_keys(['pipeline.random_state', 'spark.app_name'])
    
    def test_validate_required_keys_raises_on_missing(self, config_dir):
        """Test that missing required keys raise an error."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        with pytest.raises(ConfigurationError):
            cm.validate_required_keys(['nonexistent.required.key'])
    
    def test_reload_configuration(self, config_dir):
        """Test reloading configuration files."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        original_value = cm.get('pipeline.random_state')
        
        # Modify the config file
        config_path = Path(config_dir) / 'environments' / 'dev.yaml'
        with open(config_path, 'w') as f:
            yaml.dump({'pipeline': {'random_state': 999}}, f)
        
        # Reload
        cm.reload()
        
        # Should have new value
        assert cm.get('pipeline.random_state') == 999
    
    def test_to_dict_returns_shallow_copy(self, config_dir):
        """Test that to_dict returns a shallow copy (not deep copy)."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        config1 = cm.to_dict()
        config2 = cm.to_dict()
        
        # Top-level dict should be different objects
        assert config1 is not config2
        
        # Nested dicts may be shared (shallow copy behavior)
        # Just verify both have same values
        assert config1['pipeline']['random_state'] == config2['pipeline']['random_state']
    
    def test_set_environment(self, config_dir):
        """Test changing environment and reloading."""
        # Create prod environment
        env_dir = Path(config_dir) / 'environments'
        prod_config = {
            'pipeline': {'random_state': 999}
        }
        
        with open(env_dir / 'prod.yaml', 'w') as f:
            yaml.dump(prod_config, f)
        
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        assert cm.get('pipeline.random_state') == 123  # dev value
        
        cm.set_environment('prod')
        assert cm.get('pipeline.random_state') == 999  # prod value
    
    def test_repr(self, config_dir):
        """Test string representation."""
        cm = ConfigManager(config_dir=config_dir, environment='dev')
        
        repr_str = repr(cm)
        assert 'ConfigManager' in repr_str
        assert 'dev' in repr_str
