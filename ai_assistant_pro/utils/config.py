"""
Configuration management for AI Assistant Pro

Supports YAML and JSON configuration files.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file

    Supports YAML and JSON formats.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config("config.yaml")
        >>> srf_config = config["srf"]
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Determine file type
    suffix = config_path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping, got {type(data).__name__}")

    return data


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to file

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration

    Example:
        >>> config = {"srf": {"alpha": 0.3, "beta": 0.2}}
        >>> save_config(config, "config.yaml")
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = config_path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif suffix == ".json":
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration

    Returns:
        Default configuration dictionary
    """
    return {
        "srf": {
            "alpha": 0.3,
            "beta": 0.2,
            "gamma": 0.25,
            "delta": 0.15,
            "decay_half_life": 3600.0,
            "top_k": 10,
            "use_gpu": True,
            "use_triton": True,
        },
        "engine": {
            "model_name": "gpt2",
            "use_triton": True,
            "use_fp8": False,
            "enable_paged_attention": True,
            "max_batch_size": 32,
            "max_num_blocks": 1024,
            "block_size": 16,
        },
        "serving": {
            "host": "0.0.0.0",
            "port": 8000,
        },
        "logging": {
            "level": "INFO",
            "log_file": "logs/ai_assistant_pro.log",
            "use_rich": True,
        },
    }
