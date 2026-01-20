"""
Dashboard Configuration Loader
===============================

Loads and validates dashboard-specific configuration files.
Each dashboard has a config.yaml with search, LLM, audio, and privacy settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.backend_logger import get_logger
logger = get_logger("dashboard_config")


class DashboardConfig:
    """Dashboard configuration with validation and defaults"""

    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        self.dashboard = config_dict.get('dashboard', {})
        self.search = config_dict.get('search', {})
        self.llm = config_dict.get('llm', {})
        self.audio = config_dict.get('audio', {})
        self.filters = config_dict.get('filters', {})
        self.privacy = config_dict.get('privacy', {})

    @property
    def id(self) -> str:
        return self.dashboard.get('id', 'unknown')

    @property
    def name(self) -> str:
        return self.dashboard.get('name', 'Unknown Dashboard')

    @property
    def project(self) -> str:
        """Primary project for this dashboard"""
        return self.dashboard.get('project', '')

    @property
    def languages(self) -> list:
        return self.dashboard.get('languages', ['en'])

    @property
    def embedding_model(self) -> str:
        """Model used for query embeddings"""
        return self.search.get('embedding_model', 'qwen3:1.7b')

    @property
    def embedding_dim(self) -> int:
        return self.search.get('embedding_dim', 1024)

    @property
    def use_alt_embeddings(self) -> bool:
        """Use alternative embeddings (embedding_alt field in database)"""
        return self.search.get('use_alt_embeddings', False)

    @property
    def time_windows(self) -> list:
        """Available time windows in days"""
        return self.search.get('time_windows', [7, 30, 365])

    @property
    def default_window(self) -> int:
        """Default time window in days"""
        return self.search.get('default_window', 7)

    @property
    def max_results(self) -> int:
        return self.search.get('max_results', 200)

    @property
    def similarity_threshold(self) -> float:
        return self.search.get('similarity_threshold', 0.7)

    @property
    def max_per_channel(self) -> Optional[int]:
        """Maximum results per channel (None = no limit)"""
        return self.search.get('max_per_channel', None)

    @property
    def llm_enabled(self) -> bool:
        return self.llm.get('enabled', True)

    @property
    def llm_model(self) -> str:
        return self.llm.get('model', 'qwen3:4b-instruct')

    @property
    def llm_max_sample_segments(self) -> int:
        """Number of segments to sample for RAG"""
        return self.llm.get('max_sample_segments', 20)

    @property
    def llm_temperature(self) -> float:
        return self.llm.get('temperature', 0.3)

    @property
    def query_expansion_strategy(self) -> str:
        """Query expansion strategy: 'query2doc' or 'multi_query'"""
        return self.llm.get('query_expansion', 'multi_query')

    @property
    def audio_enabled(self) -> bool:
        return self.audio.get('enabled', True)

    @property
    def default_padding(self) -> float:
        return self.audio.get('default_padding', 2.0)

    @property
    def max_padding(self) -> float:
        return self.audio.get('max_padding', 10.0)

    @property
    def allowed_projects(self) -> list:
        """Projects this dashboard can access (privacy control)"""
        return self.privacy.get('allowed_projects', [self.project])

    @property
    def discourse_projects(self) -> list:
        """Projects for discourse_summary workflow (defaults to allowed_projects)"""
        return self.privacy.get('discourse_projects', self.allowed_projects)

    @property
    def allow_downloads(self) -> bool:
        return self.privacy.get('allow_downloads', True)

    @property
    def show_speaker_names(self) -> bool:
        return self.privacy.get('show_speaker_names', True)

    @property
    def audit_searches(self) -> bool:
        return self.privacy.get('audit_searches', False)


def load_dashboard_config(dashboard_id: str) -> DashboardConfig:
    """
    Load dashboard-specific configuration from config.yaml

    Args:
        dashboard_id: Dashboard identifier (e.g., 'cprmv-practitioner')

    Returns:
        DashboardConfig object with validated settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Path to dashboard config in backend (dashboards/ subdirectory)
    dashboard_dir = dashboard_id.replace('-', '_')
    config_path = Path(__file__).parent / f"dashboards/{dashboard_dir}/config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Dashboard config not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if not config_dict:
            raise ValueError(f"Empty config file: {config_path}")

        config = DashboardConfig(config_dict)
        logger.info(f"Loaded config for dashboard: {config.name} (project: {config.project})")

        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config: {e}")
    except Exception as e:
        raise ValueError(f"Error loading dashboard config: {e}")


def get_cache_dir(dashboard_id: str) -> Path:
    """Get cache directory for dashboard"""
    dashboard_dir = dashboard_id.replace('-', '_')
    cache_dir = Path(__file__).parent.parent / f"reports/{dashboard_dir}/.cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_all_dashboard_ids() -> list:
    """
    Get list of all configured dashboard IDs
    
    Returns:
        List of dashboard IDs (e.g., ['cprmv-practitioner'])
    """
    dashboards_dir = Path(__file__).parent / "dashboards"
    
    if not dashboards_dir.exists():
        logger.warning(f"Dashboards directory not found: {dashboards_dir}")
        return []
    
    dashboard_ids = []
    for dashboard_path in dashboards_dir.iterdir():
        if dashboard_path.is_dir():
            config_file = dashboard_path / "config.yaml"
            if config_file.exists():
                # Convert directory name back to dashboard ID format
                dashboard_id = dashboard_path.name.replace('_', '-')
                dashboard_ids.append(dashboard_id)
    
    return sorted(dashboard_ids)
