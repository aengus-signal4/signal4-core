"""
Config Manager - Handles configuration loading and hot-reloading
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List

from src.utils.config import load_config as load_config_with_env

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and hot-reloading"""
    
    def __init__(self, config_path: str = "config/orchestrator_config.yaml"):
        """Initialize config manager

        Args:
            config_path: Path to config file (default: orchestrator_config.yaml)
                        Falls back to config.yaml if orchestrator_config.yaml not found
        """
        self.config_path = config_path

        # Try orchestrator_config.yaml first, fall back to config.yaml
        if not Path(config_path).exists() and config_path == "config/orchestrator_config.yaml":
            fallback_path = "config/config.yaml"
            if Path(fallback_path).exists():
                logger.warning(f"orchestrator_config.yaml not found, using {fallback_path}")
                self.config_path = fallback_path

        self.config = self._load_config()
        self.config_mtime = self._get_config_mtime()
        self.reload_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Extract commonly accessed config values
        self._update_cached_values()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with env var substitution"""
        try:
            config = load_config_with_env(Path(self.config_path), substitute_env=True)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {str(e)}")
            raise
    
    def _get_config_mtime(self) -> float:
        """Get the last modification time of the config file"""
        try:
            return Path(self.config_path).stat().st_mtime
        except Exception as e:
            logger.error(f"Error getting config file mtime: {str(e)}")
            return 0.0
    
    def _update_cached_values(self):
        """Update commonly accessed cached values"""
        orchestrator_config = self.config.get('processing', {}).get('orchestrator', {})
        self.api_host = orchestrator_config.get('host', '0.0.0.0')
        self.api_port = orchestrator_config.get('port', 8001)
        self.callback_host = orchestrator_config.get('callback_host', 'localhost')
        self.callback_url = f"http://{self.callback_host}:{self.api_port}/api/task_callback"
        
        # Task assignment settings
        self.task_assignment_interval = self.config.get('processing', {}).get('task_assignment_interval', 5)
        self.health_check_interval = self.config.get('processing', {}).get('health_check_interval', 30)
        self.config_check_interval = self.config.get('processing', {}).get('config_check_interval', 30)
        self.idle_sleep_interval = self.config.get('processing', {}).get('idle_sleep_interval', 10)
    
    def register_reload_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback to be called when config is reloaded"""
        self.reload_callbacks.append(callback)
    
    def has_config_changed(self) -> bool:
        """Check if config file has been modified"""
        current_mtime = self._get_config_mtime()
        return current_mtime > self.config_mtime
    
    def reload_config(self) -> bool:
        """Reload configuration if changed"""
        try:
            if not self.has_config_changed():
                return False
            
            logger.info("Configuration file changed, reloading...")
            new_config = self._load_config()
            
            # Update modification time
            self.config_mtime = self._get_config_mtime()
            
            # Update configuration
            self.config = new_config
            self._update_cached_values()
            
            # Call registered callbacks
            for callback in self.reload_callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"Error in config reload callback: {str(e)}")
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {str(e)}")
            return False
    
    def get_worker_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get worker configurations"""
        return self.config.get('processing', {}).get('workers', {})
    
    def get_active_projects(self) -> Dict[str, Any]:
        """Get active project configurations"""
        return self.config.get('active_projects', {})
    
    def get_s3_config(self) -> Dict[str, Any]:
        """Get S3 storage configuration"""
        return self.config.get('storage', {}).get('s3', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config.get('database', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.config.get('processing', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value