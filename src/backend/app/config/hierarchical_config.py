"""
Hierarchical Summary Configuration Loader
==========================================

Loads configuration for hierarchical summarization system.
"""

import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any


def load_hierarchical_config() -> SimpleNamespace:
    """
    Load hierarchical summary configuration from YAML file.

    Returns:
        SimpleNamespace object with nested config attributes
    """
    config_path = Path(__file__).parent / "hierarchical_summary.yaml"

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert nested dicts to SimpleNamespace for dot notation access
    def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
        """Recursively convert dict to SimpleNamespace"""
        namespace = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace

    return dict_to_namespace(config_dict)


# Singleton instance
_config = None

def get_hierarchical_config() -> SimpleNamespace:
    """Get cached config instance (singleton pattern)"""
    global _config
    if _config is None:
        _config = load_hierarchical_config()
    return _config
