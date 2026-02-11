"""
Config Loader

Loads pipeline configuration from YAML files, with CLI and programmatic overrides.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging

import yaml

from src.config.schema import PipelineConfig


logger = logging.getLogger(__name__)


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-notation key.

    Args:
        d: The dictionary to modify.
        dotted_key: Key in dot notation, e.g. "steps.iv.min_iv".
        value: Value to set.
    """
    keys = dotted_key.split(".")
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _resolve_paths(raw: Dict[str, Any], yaml_dir: Path) -> Dict[str, Any]:
    """Resolve relative paths in the config against the YAML file's directory.

    Only resolves the data.input_path field.

    Args:
        raw: Raw config dictionary.
        yaml_dir: Directory containing the YAML file.

    Returns:
        Config dict with resolved paths.
    """
    data_cfg = raw.get("data", {})
    input_path = data_cfg.get("input_path")
    if input_path and not Path(input_path).is_absolute():
        resolved = (yaml_dir / input_path).resolve()
        if resolved.exists():
            data_cfg["input_path"] = str(resolved)
        # If it doesn't exist, keep the original (might be relative to cwd)
    return raw


def load_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """Load pipeline configuration from YAML with optional overrides.

    Args:
        yaml_path: Path to the YAML config file. If None, uses defaults.
        cli_overrides: Flat dict of dot-notation keys from CLI args.
            Example: {"data.input_path": "/data/features.parquet"}
        overrides: Nested dict of programmatic overrides merged on top.

    Returns:
        Frozen PipelineConfig instance.
    """
    raw: Dict[str, Any] = {}

    if yaml_path is not None:
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_file, "r") as f:
            raw = yaml.safe_load(f) or {}

        logger.info("Loaded config from %s", yaml_path)

        # Resolve relative paths against the YAML file's directory
        raw = _resolve_paths(raw, yaml_file.parent)

    # Apply CLI overrides (flat dot-notation)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                _set_nested(raw, key, value)

    # Apply programmatic overrides (nested dict)
    if overrides:
        _deep_merge(raw, overrides)

    config = PipelineConfig(**raw)
    logger.debug("Pipeline config loaded successfully")
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override dict into base dict (in-place).

    Args:
        base: Base dictionary to merge into.
        override: Override dictionary whose values take priority.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_config(config: PipelineConfig, path: str) -> None:
    """Save a PipelineConfig to a YAML file.

    Args:
        config: The pipeline configuration to save.
        path: Output file path (.yaml or .json).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump()

    if out_path.suffix in (".yaml", ".yml"):
        with open(out_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    else:
        with open(out_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    logger.info("Config saved to %s", path)
