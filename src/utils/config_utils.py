"""
Utility functions for loading and managing configurations.
"""

import os
import yaml
from pathlib import Path
from omegaconf import OmegaConf


def load_config(config_path: str = "config/default.yaml", overrides: dict = None):
    """Load configuration from a YAML file and apply overrides (e.g., CLI args).

    Returns a dict with final config values.
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return OmegaConf.to_container(cfg, resolve=True)
