"""
Utility functions for loading and managing configurations.
"""
import yaml
from easydict import EasyDict
from pathlib import Path


def load_config(config_path: str) -> EasyDict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Configuration as EasyDict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return EasyDict(config)


def save_config(config: dict, output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Saved configuration to {output_path}")


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
