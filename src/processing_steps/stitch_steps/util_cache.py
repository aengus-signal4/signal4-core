#!/usr/bin/env python3
"""
Stage Cache System for Stitch Pipeline
=====================================

Caches results after each stage to enable faster development and testing.
Allows running individual stages without re-running the entire pipeline.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd
from datetime import datetime

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('stitch')


class StageCache:
    """Manages caching of stage results for the stitch pipeline."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the stage cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, uses .cache/stitch/
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / '.cache' / 'stitch'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Stage cache initialized at: {self.cache_dir}")
    
    def get_cache_key(self, content_id: str, stage_name: str, 
                      params: Optional[Dict] = None) -> str:
        """
        Generate a unique cache key for a stage result.
        
        Args:
            content_id: Content ID being processed
            stage_name: Name of the stage (e.g., 'stage1_load', 'stage6_context')
            params: Optional parameters that affect the stage output
            
        Returns:
            Cache key string
        """
        key_parts = [content_id, stage_name]
        
        if params:
            # Sort params for consistent hashing
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            key_parts.append(param_hash)
        
        return "_".join(key_parts)
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def save_stage_result(self, content_id: str, stage_name: str, 
                         result: Dict[str, Any], params: Optional[Dict] = None) -> bool:
        """
        Save a stage result to cache using pickle.
        This handles all object types including WordTable and SegmentTable automatically.
        
        Args:
            content_id: Content ID being processed
            stage_name: Name of the stage
            result: Stage result dictionary
            params: Optional parameters that affect the stage output
            
        Returns:
            True if saved successfully
        """
        try:
            cache_key = self.get_cache_key(content_id, stage_name, params)
            cache_path = self.get_cache_path(cache_key)
            
            # Add metadata
            cache_data = {
                'content_id': content_id,
                'stage_name': stage_name,
                'params': params,
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'version': '2.0'  # Version to handle backward compatibility
            }
            
            # Save entire object structure with pickle
            # This preserves WordTable, SegmentTable, and all other objects as-is
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Cached {stage_name} result for {content_id} at: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache {stage_name} result: {e}")
            logger.error(f"Error details:", exc_info=True)
            return False
    
    def load_stage_result(self, content_id: str, stage_name: str, 
                         params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Load a stage result from cache.
        Handles both new pickle format (v2.0) and old manual reconstruction format.
        
        Args:
            content_id: Content ID being processed
            stage_name: Name of the stage
            params: Optional parameters that affect the stage output
            
        Returns:
            Stage result dictionary if found, None otherwise
        """
        try:
            cache_key = self.get_cache_key(content_id, stage_name, params)
            cache_path = self.get_cache_path(cache_key)
            
            if not cache_path.exists():
                logger.debug(f"No cache found for {stage_name} ({content_id})")
                return None
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check cache version
            cache_version = cache_data.get('version', '1.0')
            
            if cache_version == '2.0':
                # New format - objects are already properly pickled
                result = cache_data['result']
                if self.validate_stage_result(result, stage_name):
                    logger.info(f"Loaded cached {stage_name} result for {content_id} (v2.0 format, cached: {cache_data['timestamp']})")
                    return result
                else:
                    logger.error(f"Cached {stage_name} result failed validation, returning None")
                    return None
            else:
                # Old format - need to reconstruct WordTable and SegmentTable
                logger.warning(f"Cache for {stage_name} ({content_id}) has outdated module references")
                logger.debug(f"Loading legacy cache format for {stage_name}")
                if 'result' in cache_data and 'data' in cache_data['result']:
                    if 'word_table' in cache_data['result']['data']:
                        wt_data = cache_data['result']['data']['word_table']
                        if isinstance(wt_data, dict) and wt_data.get('type') == 'WordTable':
                            # Reconstruct WordTable
                            from .stage3_tables import WordTable
                            word_table = WordTable(wt_data['content_id'])
                            word_table.df = pd.DataFrame(wt_data['df'])
                            # Restore proper dtypes
                            for col, dtype in wt_data['dtypes'].items():
                                if col in word_table.df.columns:
                                    try:
                                        word_table.df[col] = word_table.df[col].astype(dtype)
                                    except:
                                        pass  # Some dtypes might not convert back perfectly
                            cache_data['result']['data']['word_table'] = word_table
                    
                    if 'segment_table' in cache_data['result']['data']:
                        st_data = cache_data['result']['data']['segment_table']
                        if isinstance(st_data, dict) and st_data.get('type') == 'SegmentTable':
                            # Reconstruct SegmentTable
                            from .stage3_tables import SegmentTable
                            segment_table = SegmentTable(st_data['content_id'])
                            segment_table.df = pd.DataFrame(st_data['df'])
                            # Restore proper dtypes
                            for col, dtype in st_data['dtypes'].items():
                                if col in segment_table.df.columns:
                                    try:
                                        segment_table.df[col] = segment_table.df[col].astype(dtype)
                                    except:
                                        pass  # Some dtypes might not convert back perfectly
                            cache_data['result']['data']['segment_table'] = segment_table
                
                logger.info(f"Loaded cached {stage_name} result for {content_id} (legacy format, cached: {cache_data['timestamp']})")
                return cache_data['result']
            
        except (ImportError, ModuleNotFoundError) as e:
            # Handle import errors from old cached objects with changed module paths
            logger.warning(f"Cache for {stage_name} ({content_id}) has outdated module references: {e}")
            logger.error(f"Cannot load cache due to module changes. Please regenerate by running:")
            logger.error(f"  python src/processing_steps/stitch.py --content {content_id} --test --stages 5")
            logger.error(f"")
            logger.error(f"If you want to clear the cache manually, run:")
            logger.error(f"  rm {cache_path}")
            
            # Don't remove the cache automatically - let the user decide
            return None
        except Exception as e:
            logger.error(f"Failed to load cached {stage_name} result: {e}")
            logger.error(f"Error details:", exc_info=True)
            return None
    
    def clear_stage_cache(self, content_id: str, stage_name: Optional[str] = None):
        """
        Clear cache for a specific stage or all stages of a content ID.
        
        Args:
            content_id: Content ID to clear cache for
            stage_name: Specific stage to clear, or None for all stages
        """
        pattern = f"{content_id}_*" if stage_name is None else f"{content_id}_{stage_name}_*"
        
        cleared = 0
        for cache_file in self.cache_dir.glob(f"{pattern}.pkl"):
            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {cleared} cache files for {content_id}" + 
                   (f" stage {stage_name}" if stage_name else ""))
    
    def list_cached_stages(self, content_id: str) -> List[str]:
        """
        List all cached stages for a content ID.
        
        Args:
            content_id: Content ID to check
            
        Returns:
            List of cached stage names
        """
        stages = []
        for cache_file in self.cache_dir.glob(f"{content_id}_*.pkl"):
            parts = cache_file.stem.split('_')
            if len(parts) >= 2:
                # Extract stage name (everything after content_id)
                stage_name = '_'.join(parts[1:])
                # Remove parameter hash if present
                if len(parts) > 2 and len(parts[-1]) == 8:
                    stage_name = '_'.join(parts[1:-1])
                stages.append(stage_name)
        
        return sorted(list(set(stages)))
    
    def validate_stage_result(self, result: Dict[str, Any], stage_name: str) -> bool:
        """
        Validate that a stage result has the expected structure and objects.
        
        Args:
            result: Stage result dictionary
            stage_name: Name of the stage for context
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic structure validation
            if not isinstance(result, dict):
                logger.error(f"Stage {stage_name} result is not a dictionary")
                return False
                
            if 'status' not in result:
                logger.error(f"Stage {stage_name} result missing 'status' field")
                return False
                
            if 'data' not in result:
                logger.error(f"Stage {stage_name} result missing 'data' field")
                return False
                
            # Check for WordTable if expected
            if 'word_table' in result['data']:
                wt = result['data']['word_table']
                if not hasattr(wt, 'df'):
                    logger.error(f"Stage {stage_name} word_table missing 'df' attribute, got type: {type(wt)}")
                    return False
                if not hasattr(wt, 'content_id'):
                    logger.error(f"Stage {stage_name} word_table missing 'content_id' attribute")
                    return False
                    
            # Check for SegmentTable if expected
            if 'segment_table' in result['data']:
                st = result['data']['segment_table']
                if not hasattr(st, 'df'):
                    logger.error(f"Stage {stage_name} segment_table missing 'df' attribute, got type: {type(st)}")
                    return False
                if not hasattr(st, 'content_id'):
                    logger.error(f"Stage {stage_name} segment_table missing 'content_id' attribute")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating stage {stage_name} result: {e}")
            return False
    
    def get_cache_info(self, content_id: str) -> Dict[str, Any]:
        """
        Get information about cached stages for a content ID.
        
        Args:
            content_id: Content ID to check
            
        Returns:
            Dictionary with cache information
        """
        info = {
            'content_id': content_id,
            'cached_stages': [],
            'total_size': 0
        }
        
        for cache_file in self.cache_dir.glob(f"{content_id}_*.pkl"):
            try:
                stat = cache_file.stat()
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                stage_info = {
                    'stage_name': cache_data['stage_name'],
                    'timestamp': cache_data['timestamp'],
                    'size': stat.st_size,
                    'params': cache_data.get('params'),
                    'file': cache_file.name
                }
                
                info['cached_stages'].append(stage_info)
                info['total_size'] += stat.st_size
                
            except Exception as e:
                logger.error(f"Failed to read cache info from {cache_file}: {e}")
        
        # Sort by stage name
        info['cached_stages'].sort(key=lambda x: x['stage_name'])
        
        return info


# Global cache instance
_stage_cache = None

def get_stage_cache(cache_dir: Optional[Path] = None) -> StageCache:
    """Get or create the global stage cache instance."""
    global _stage_cache
    if _stage_cache is None:
        _stage_cache = StageCache(cache_dir)
    return _stage_cache


def with_stage_cache(stage_name: str, cache_params: Optional[Dict] = None):
    """
    Decorator to automatically cache stage results.
    
    Usage:
        @with_stage_cache('stage6_context')
        def context_stage(content_id: str, word_table: WordTable, ...) -> Dict:
            ...
    """
    def decorator(func):
        def wrapper(content_id: str, *args, **kwargs):
            # Check if caching is disabled
            if kwargs.get('use_cache', True) is False:
                return func(content_id, *args, **kwargs)
            
            cache = get_stage_cache()
            
            # Try to load from cache
            cached_result = cache.load_stage_result(content_id, stage_name, cache_params)
            if cached_result is not None and kwargs.get('force_recompute', False) is False:
                logger.debug(f"Using cached result for {stage_name}")
                return cached_result
            
            # Run the stage
            result = func(content_id, *args, **kwargs)
            
            # Cache the result if successful
            if result and result.get('status') == 'success':
                cache.save_stage_result(content_id, stage_name, result, cache_params)
            
            return result
        
        return wrapper
    return decorator