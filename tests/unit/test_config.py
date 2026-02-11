"""
Unit Tests for Config System

Tests config schema validation, YAML loading, CLI overrides,
config immutability, and save/load round-trips.
"""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from src.config.schema import (
    PipelineConfig,
    DataConfig,
    SplittingConfig,
    StepsConfig,
    ConstantConfig,
    MissingConfig,
    IVConfig,
    PSIConfig,
    PSICheckConfig,
    CorrelationConfig,
    SelectionConfig,
    ModelConfig,
    EvaluationConfig,
    ValidationConfig,
    ValidationChecksConfig,
    OutputConfig,
    ReproducibilityConfig,
)
from src.config.loader import load_config, save_config, _set_nested, _deep_merge


# ===================================================================
# Schema Defaults
# ===================================================================

class TestSchemaDefaults:
    """Test that all config models can be created with defaults."""

    def test_data_config_defaults(self):
        cfg = DataConfig()
        assert cfg.target_column == "target"
        assert cfg.date_column == "date"
        assert "application_id" in cfg.id_columns
        assert "customer_id" in cfg.id_columns

    def test_splitting_config_defaults(self):
        cfg = SplittingConfig()
        assert cfg.test_size == 0.20
        assert cfg.stratify is True

    def test_constant_config_defaults(self):
        cfg = ConstantConfig()
        assert cfg.enabled is True
        assert cfg.min_unique_values == 2

    def test_missing_config_defaults(self):
        cfg = MissingConfig()
        assert cfg.threshold == 0.70

    def test_iv_config_defaults(self):
        cfg = IVConfig()
        assert cfg.min_iv == 0.02
        assert cfg.max_iv == 0.50
        assert cfg.n_bins == 10

    def test_psi_config_defaults(self):
        cfg = PSIConfig()
        assert cfg.threshold == 0.25
        assert len(cfg.checks) == 3

    def test_correlation_config_defaults(self):
        cfg = CorrelationConfig()
        assert cfg.threshold == 0.90
        assert cfg.method == "pearson"

    def test_selection_config_defaults(self):
        cfg = SelectionConfig()
        assert cfg.method == "forward"
        assert cfg.max_features is None

    def test_model_config_defaults(self):
        cfg = ModelConfig()
        assert cfg.algorithm == "xgboost"
        assert "objective" in cfg.params

    def test_evaluation_config_defaults(self):
        cfg = EvaluationConfig()
        assert "auc" in cfg.metrics
        assert cfg.n_deciles == 10

    def test_validation_config_defaults(self):
        cfg = ValidationConfig()
        assert cfg.enabled is True
        assert cfg.checks.min_auc == 0.65

    def test_output_config_defaults(self):
        cfg = OutputConfig()
        assert cfg.save_step_results is True
        assert cfg.generate_excel is True

    def test_reproducibility_config_defaults(self):
        cfg = ReproducibilityConfig()
        assert cfg.global_seed == 42
        assert cfg.log_level == "DEBUG"

    def test_pipeline_config_all_defaults(self):
        cfg = PipelineConfig()
        assert cfg.data.target_column == "target"
        assert cfg.steps.constant.enabled is True
        assert cfg.model.algorithm == "xgboost"
        assert cfg.reproducibility.global_seed == 42


# ===================================================================
# Schema Validation
# ===================================================================

class TestSchemaValidation:
    """Test that validation rules correctly reject invalid configs."""

    def test_iv_min_greater_than_max_rejected(self):
        with pytest.raises(ValidationError, match="min_iv.*must be less than.*max_iv"):
            IVConfig(min_iv=0.5, max_iv=0.02)

    def test_iv_min_equals_max_rejected(self):
        with pytest.raises(ValidationError, match="min_iv.*must be less than.*max_iv"):
            IVConfig(min_iv=0.10, max_iv=0.10)

    def test_negative_missing_threshold_rejected(self):
        with pytest.raises(ValidationError):
            MissingConfig(threshold=-0.1)

    def test_missing_threshold_above_one_rejected(self):
        with pytest.raises(ValidationError):
            MissingConfig(threshold=1.5)

    def test_test_size_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            SplittingConfig(test_size=-0.1)

    def test_test_size_above_one_rejected(self):
        with pytest.raises(ValidationError):
            SplittingConfig(test_size=1.5)

    def test_negative_correlation_threshold_rejected(self):
        with pytest.raises(ValidationError):
            CorrelationConfig(threshold=-0.5)

    def test_correlation_threshold_above_one_rejected(self):
        with pytest.raises(ValidationError):
            CorrelationConfig(threshold=1.5)

    def test_invalid_correlation_method_rejected(self):
        with pytest.raises(ValidationError):
            CorrelationConfig(method="invalid_method")

    def test_invalid_log_level_rejected(self):
        with pytest.raises(ValidationError):
            ReproducibilityConfig(log_level="VERBOSE")

    def test_invalid_algorithm_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(algorithm="random_forest")

    def test_min_unique_values_zero_rejected(self):
        with pytest.raises(ValidationError):
            ConstantConfig(min_unique_values=0)

    def test_n_bins_below_two_rejected(self):
        with pytest.raises(ValidationError):
            IVConfig(n_bins=1)

    def test_negative_auc_threshold_rejected(self):
        with pytest.raises(ValidationError):
            SelectionConfig(auc_threshold=-0.001)

    def test_min_auc_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ValidationChecksConfig(min_auc=1.5)


# ===================================================================
# Config Immutability
# ===================================================================

class TestConfigImmutability:
    """Test that frozen configs are immutable after creation."""

    def test_pipeline_config_frozen(self, sample_config):
        with pytest.raises(ValidationError):
            sample_config.reproducibility = ReproducibilityConfig(global_seed=99)

    def test_data_config_frozen(self):
        cfg = DataConfig()
        with pytest.raises(ValidationError):
            cfg.target_column = "new_target"

    def test_steps_config_frozen(self):
        cfg = StepsConfig()
        with pytest.raises(ValidationError):
            cfg.constant = ConstantConfig(min_unique_values=5)

    def test_iv_config_frozen(self):
        cfg = IVConfig()
        with pytest.raises(ValidationError):
            cfg.min_iv = 0.10


# ===================================================================
# Config Loading
# ===================================================================

class TestConfigLoading:
    """Test YAML loading and config creation."""

    def test_load_config_from_yaml(self, tmp_config_yaml):
        config = load_config(str(tmp_config_yaml))
        assert isinstance(config, PipelineConfig)
        assert config.data.target_column == "target"
        assert config.steps.iv.min_iv == 0.02

    def test_load_config_with_defaults(self):
        config = load_config()
        assert isinstance(config, PipelineConfig)
        assert config.data.target_column == "target"

    def test_load_config_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_empty_yaml(self, tmp_path):
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("")
        config = load_config(str(empty_path))
        assert isinstance(config, PipelineConfig)


# ===================================================================
# CLI Overrides
# ===================================================================

class TestCLIOverrides:
    """Test CLI override merging with dot-notation keys."""

    def test_cli_override_simple_value(self, tmp_config_yaml):
        config = load_config(
            str(tmp_config_yaml),
            cli_overrides={"data.target_column": "label"},
        )
        assert config.data.target_column == "label"

    def test_cli_override_nested_value(self, tmp_config_yaml):
        config = load_config(
            str(tmp_config_yaml),
            cli_overrides={"steps.iv.min_iv": 0.05},
        )
        assert config.steps.iv.min_iv == 0.05

    def test_cli_override_deep_nested(self, tmp_config_yaml):
        config = load_config(
            str(tmp_config_yaml),
            cli_overrides={"validation.checks.min_auc": 0.70},
        )
        assert config.validation.checks.min_auc == 0.70

    def test_cli_override_none_value_ignored(self, tmp_config_yaml):
        config = load_config(
            str(tmp_config_yaml),
            cli_overrides={"data.target_column": None},
        )
        assert config.data.target_column == "target"

    def test_cli_overrides_multiple(self, tmp_config_yaml):
        config = load_config(
            str(tmp_config_yaml),
            cli_overrides={
                "steps.missing.threshold": 0.50,
                "steps.correlation.threshold": 0.80,
                "reproducibility.global_seed": 123,
            },
        )
        assert config.steps.missing.threshold == 0.50
        assert config.steps.correlation.threshold == 0.80
        assert config.reproducibility.global_seed == 123


# ===================================================================
# Programmatic Overrides
# ===================================================================

class TestProgrammaticOverrides:
    """Test nested dict overrides."""

    def test_nested_override(self, tmp_config_yaml):
        config = load_config(
            str(tmp_config_yaml),
            overrides={"steps": {"iv": {"min_iv": 0.05}}},
        )
        assert config.steps.iv.min_iv == 0.05
        # Other iv fields should remain from YAML
        assert config.steps.iv.max_iv == 0.50

    def test_override_replaces_section(self, tmp_config_yaml):
        config = load_config(
            str(tmp_config_yaml),
            overrides={"reproducibility": {"global_seed": 99}},
        )
        assert config.reproducibility.global_seed == 99


# ===================================================================
# Save / Load Round-Trip
# ===================================================================

class TestConfigSaveLoad:
    """Test config serialization and deserialization round-trips."""

    def test_save_config_yaml(self, sample_config, tmp_path):
        yaml_path = tmp_path / "out_config.yaml"
        save_config(sample_config, str(yaml_path))
        assert yaml_path.exists()

        # Reload and compare
        reloaded = load_config(str(yaml_path))
        assert reloaded.data.target_column == sample_config.data.target_column
        assert reloaded.steps.iv.min_iv == sample_config.steps.iv.min_iv
        assert reloaded.reproducibility.global_seed == sample_config.reproducibility.global_seed

    def test_save_config_json(self, sample_config, tmp_path):
        import json

        json_path = tmp_path / "out_config.json"
        save_config(sample_config, str(json_path))
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["data"]["target_column"] == "target"

    def test_save_creates_parent_dirs(self, sample_config, tmp_path):
        nested_path = tmp_path / "a" / "b" / "config.yaml"
        save_config(sample_config, str(nested_path))
        assert nested_path.exists()

    def test_round_trip_preserves_all_fields(self, sample_config, tmp_path):
        yaml_path = tmp_path / "round_trip.yaml"
        save_config(sample_config, str(yaml_path))
        reloaded = load_config(str(yaml_path))

        original_dict = sample_config.model_dump()
        reloaded_dict = reloaded.model_dump()

        # Compare all top-level sections
        for section in original_dict:
            assert original_dict[section] == reloaded_dict[section], (
                f"Section '{section}' mismatch after round-trip"
            )


# ===================================================================
# Helper Functions
# ===================================================================

class TestHelperFunctions:
    """Test internal helper functions."""

    def test_set_nested_single_level(self):
        d = {}
        _set_nested(d, "key", "value")
        assert d == {"key": "value"}

    def test_set_nested_multi_level(self):
        d = {}
        _set_nested(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_set_nested_overwrites_existing(self):
        d = {"a": {"b": 1}}
        _set_nested(d, "a.b", 2)
        assert d["a"]["b"] == 2

    def test_deep_merge_basic(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": {"c": 99, "d": 3}}

    def test_deep_merge_adds_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": 2}

    def test_deep_merge_replaces_non_dict(self):
        base = {"a": {"b": 1}}
        override = {"a": "replaced"}
        _deep_merge(base, override)
        assert base == {"a": "replaced"}


# ===================================================================
# PipelineConfig from Dict
# ===================================================================

class TestPipelineConfigFromDict:
    """Test creating PipelineConfig from various dict shapes."""

    def test_from_complete_dict(self, sample_config_dict):
        config = PipelineConfig(**sample_config_dict)
        assert config.steps.iv.min_iv == 0.02
        assert config.model.params["max_depth"] == 4

    def test_from_partial_dict_uses_defaults(self):
        config = PipelineConfig(data={"target_column": "my_target"})
        assert config.data.target_column == "my_target"
        assert config.data.date_column == "date"  # default
        assert config.steps.constant.enabled is True  # default

    def test_from_empty_dict(self):
        config = PipelineConfig()
        assert config.data.target_column == "target"
        assert config.reproducibility.global_seed == 42

    def test_model_dump_contains_all_sections(self):
        config = PipelineConfig()
        dump = config.model_dump()
        expected_keys = {"data", "splitting", "steps", "model", "evaluation", "validation", "output", "reproducibility"}
        assert expected_keys == set(dump.keys())
