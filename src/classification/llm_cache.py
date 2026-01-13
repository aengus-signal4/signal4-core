#!/usr/bin/env python3
"""
LLM Result Caching System

Provides persistent caching for LLM classification results to avoid duplicate API calls.
Uses SQLite for efficient storage and retrieval with hash-based lookups.
"""

import sqlite3
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CachedThemeResult:
    """Cached theme classification result"""
    segment_id: int
    text_hash: str
    theme_ids: List[int]
    theme_names: List[str]
    confidence: float
    reasoning: str
    timestamp: str
    model_name: str

@dataclass
class CachedSubthemeResult:
    """Cached subtheme classification result"""
    segment_id: int
    text_hash: str
    theme_id: int
    subtheme_ids: List[int]
    subtheme_names: List[str]
    confidence: float
    reasoning: str
    timestamp: str
    model_name: str

class LLMResultCache:
    """
    Manages caching of LLM classification results.
    
    Uses SQLite for persistent storage with hash-based lookups for efficiency.
    Separate tables for theme and subtheme classifications.
    """
    
    def __init__(self, cache_dir: str = "cache", project: Optional[str] = None):
        """
        Initialize the LLM result cache.
        
        Args:
            cache_dir: Directory to store cache database
            project: Optional project name for project-specific caching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use project-specific cache if provided
        db_name = f"llm_cache_{project}.db" if project else "llm_cache.db"
        self.db_path = self.cache_dir / db_name
        
        self.conn = None
        self._init_database()
        
        # Statistics
        self.stats = {
            'theme_hits': 0,
            'theme_misses': 0,
            'subtheme_hits': 0,
            'subtheme_misses': 0,
            'total_saved_calls': 0
        }
    
    def _init_database(self):
        """Initialize SQLite database with tables for caching"""
        self.conn = sqlite3.connect(str(self.db_path))
        
        # Enable WAL mode for better concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        # Create theme cache table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS theme_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER NOT NULL,
                text_hash TEXT NOT NULL,
                theme_ids TEXT NOT NULL,  -- JSON array
                theme_names TEXT NOT NULL,  -- JSON array
                confidence REAL NOT NULL,
                reasoning TEXT,
                model_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(segment_id, text_hash, model_name)
            )
        """)
        
        # Create subtheme cache table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS subtheme_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER NOT NULL,
                text_hash TEXT NOT NULL,
                theme_id INTEGER NOT NULL,
                subtheme_ids TEXT NOT NULL,  -- JSON array
                subtheme_names TEXT NOT NULL,  -- JSON array
                confidence REAL NOT NULL,
                reasoning TEXT,
                model_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(segment_id, text_hash, theme_id, model_name)
            )
        """)
        
        # Create indexes for fast lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_theme_lookup 
            ON theme_cache(segment_id, text_hash, model_name)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_subtheme_lookup 
            ON subtheme_cache(segment_id, text_hash, theme_id, model_name)
        """)
        
        # Create batch processing table for tracking processed batches
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_hash TEXT NOT NULL UNIQUE,
                segment_ids TEXT NOT NULL,  -- JSON array
                phase TEXT NOT NULL,  -- 'theme' or 'subtheme'
                timestamp TEXT NOT NULL
            )
        """)
        
        self.conn.commit()
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute a hash of the text for cache lookup"""
        # Use first 2000 chars (same as what's sent to LLM)
        text_for_hash = text[:2000] if len(text) > 2000 else text
        return hashlib.sha256(text_for_hash.encode()).hexdigest()[:16]
    
    def get_theme_result(self, segment_id: int, text: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached theme classification result.
        
        Args:
            segment_id: ID of the segment
            text: Text content of the segment
            model_name: Name of the LLM model
            
        Returns:
            Cached result dict or None if not found
        """
        text_hash = self._compute_text_hash(text)
        
        cursor = self.conn.execute("""
            SELECT theme_ids, theme_names, confidence, reasoning, timestamp
            FROM theme_cache
            WHERE segment_id = ? AND text_hash = ? AND model_name = ?
        """, (segment_id, text_hash, model_name))
        
        row = cursor.fetchone()
        if row:
            self.stats['theme_hits'] += 1
            self.stats['total_saved_calls'] += 1
            logger.debug(f"Cache hit for theme classification: segment {segment_id}")
            
            return {
                'theme_ids': json.loads(row[0]),
                'theme_names': json.loads(row[1]),
                'confidence': row[2],
                'reasoning': row[3],
                'cached': True,
                'cache_timestamp': row[4]
            }
        
        self.stats['theme_misses'] += 1
        return None
    
    def save_theme_result(self, segment_id: int, text: str, model_name: str,
                         theme_ids: List[int], theme_names: List[str],
                         confidence: float, reasoning: str):
        """Save theme classification result to cache"""
        text_hash = self._compute_text_hash(text)
        timestamp = datetime.now().isoformat()
        
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO theme_cache
                (segment_id, text_hash, theme_ids, theme_names, confidence, reasoning, model_name, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment_id, text_hash,
                json.dumps(theme_ids), json.dumps(theme_names),
                confidence, reasoning, model_name, timestamp
            ))
            self.conn.commit()
            logger.debug(f"Cached theme result for segment {segment_id}")
        except Exception as e:
            logger.error(f"Failed to cache theme result: {e}")
    
    def get_subtheme_result(self, segment_id: int, text: str, theme_id: int, 
                           model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached subtheme classification result.
        
        Args:
            segment_id: ID of the segment
            text: Text content of the segment
            theme_id: ID of the parent theme
            model_name: Name of the LLM model
            
        Returns:
            Cached result dict or None if not found
        """
        text_hash = self._compute_text_hash(text)
        
        cursor = self.conn.execute("""
            SELECT subtheme_ids, subtheme_names, confidence, reasoning, timestamp
            FROM subtheme_cache
            WHERE segment_id = ? AND text_hash = ? AND theme_id = ? AND model_name = ?
        """, (segment_id, text_hash, theme_id, model_name))
        
        row = cursor.fetchone()
        if row:
            self.stats['subtheme_hits'] += 1
            self.stats['total_saved_calls'] += 1
            logger.debug(f"Cache hit for subtheme classification: segment {segment_id}, theme {theme_id}")
            
            return {
                'subtheme_ids': json.loads(row[0]),
                'subtheme_names': json.loads(row[1]),
                'confidence': row[2],
                'reasoning': row[3],
                'cached': True,
                'cache_timestamp': row[4]
            }
        
        self.stats['subtheme_misses'] += 1
        return None
    
    def save_subtheme_result(self, segment_id: int, text: str, theme_id: int,
                            model_name: str, subtheme_ids: List[int],
                            subtheme_names: List[str], confidence: float, reasoning: str):
        """Save subtheme classification result to cache"""
        text_hash = self._compute_text_hash(text)
        timestamp = datetime.now().isoformat()
        
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO subtheme_cache
                (segment_id, text_hash, theme_id, subtheme_ids, subtheme_names, 
                 confidence, reasoning, model_name, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment_id, text_hash, theme_id,
                json.dumps(subtheme_ids), json.dumps(subtheme_names),
                confidence, reasoning, model_name, timestamp
            ))
            self.conn.commit()
            logger.debug(f"Cached subtheme result for segment {segment_id}, theme {theme_id}")
        except Exception as e:
            logger.error(f"Failed to cache subtheme result: {e}")
    
    def get_batch_results(self, segment_ids: List[int], texts: Dict[int, str], 
                         model_name: str, phase: str = 'theme') -> Dict[int, Dict[str, Any]]:
        """
        Get cached results for a batch of segments.
        
        Args:
            segment_ids: List of segment IDs
            texts: Dict mapping segment_id to text content
            model_name: Name of the LLM model
            phase: 'theme' or 'subtheme'
            
        Returns:
            Dict mapping segment_id to cached result (if found)
        """
        cached_results = {}
        
        for segment_id in segment_ids:
            if segment_id in texts:
                if phase == 'theme':
                    result = self.get_theme_result(segment_id, texts[segment_id], model_name)
                    if result:
                        cached_results[segment_id] = result
                # For subtheme, would need theme_id as well
        
        if cached_results:
            logger.info(f"Found {len(cached_results)}/{len(segment_ids)} cached {phase} results")
        
        return cached_results
    
    def mark_batch_processed(self, segment_ids: List[int], phase: str):
        """Mark a batch of segments as processed"""
        batch_hash = hashlib.sha256(
            f"{sorted(segment_ids)}_{phase}".encode()
        ).hexdigest()[:16]
        
        timestamp = datetime.now().isoformat()
        
        try:
            self.conn.execute("""
                INSERT OR IGNORE INTO batch_cache (batch_hash, segment_ids, phase, timestamp)
                VALUES (?, ?, ?, ?)
            """, (batch_hash, json.dumps(segment_ids), phase, timestamp))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to mark batch as processed: {e}")
    
    def is_batch_processed(self, segment_ids: List[int], phase: str) -> bool:
        """Check if a batch has already been processed"""
        batch_hash = hashlib.sha256(
            f"{sorted(segment_ids)}_{phase}".encode()
        ).hexdigest()[:16]
        
        cursor = self.conn.execute("""
            SELECT 1 FROM batch_cache WHERE batch_hash = ? AND phase = ?
        """, (batch_hash, phase))
        
        return cursor.fetchone() is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        # Query database for counts
        cursor = self.conn.execute("SELECT COUNT(*) FROM theme_cache")
        total_theme_cached = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM subtheme_cache")
        total_subtheme_cached = cursor.fetchone()[0]
        
        return {
            **self.stats,
            'total_theme_cached': total_theme_cached,
            'total_subtheme_cached': total_subtheme_cached,
            'cache_hit_rate_theme': (
                self.stats['theme_hits'] / max(1, self.stats['theme_hits'] + self.stats['theme_misses'])
            ),
            'cache_hit_rate_subtheme': (
                self.stats['subtheme_hits'] / max(1, self.stats['subtheme_hits'] + self.stats['subtheme_misses'])
            )
        }
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear the cache.
        
        Args:
            older_than_days: If provided, only clear entries older than this many days
        """
        if older_than_days:
            cutoff_date = datetime.now().timestamp() - (older_than_days * 86400)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
            
            self.conn.execute("DELETE FROM theme_cache WHERE timestamp < ?", (cutoff_iso,))
            self.conn.execute("DELETE FROM subtheme_cache WHERE timestamp < ?", (cutoff_iso,))
            self.conn.execute("DELETE FROM batch_cache WHERE timestamp < ?", (cutoff_iso,))
        else:
            self.conn.execute("DELETE FROM theme_cache")
            self.conn.execute("DELETE FROM subtheme_cache")
            self.conn.execute("DELETE FROM batch_cache")
        
        self.conn.commit()
        logger.info("Cache cleared")
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EmbeddingCache:
    """Cache for theme description embeddings"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, descriptions: List[str], model_name: str, language: str) -> Path:
        """Generate cache file path for embeddings"""
        # Create hash of descriptions for cache key
        desc_hash = hashlib.sha256(
            "".join(sorted(descriptions)).encode()
        ).hexdigest()[:16]
        
        # Include model name and language in filename
        model_safe = model_name.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"embeddings_{model_safe}_{language}_{desc_hash}.npz"
    
    def load_embeddings(self, descriptions: List[str], model_name: str, 
                       language: str) -> Optional[Any]:
        """Load cached embeddings if available"""
        import numpy as np
        
        cache_path = self.get_cache_path(descriptions, model_name, language)
        
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                cached_descriptions = data['descriptions'].tolist()
                
                # Verify descriptions match
                if cached_descriptions == descriptions:
                    logger.info(f"Loaded cached embeddings from {cache_path}")
                    return data['embeddings']
                else:
                    logger.warning("Cached descriptions don't match, recomputing")
            except Exception as e:
                logger.error(f"Failed to load cached embeddings: {e}")
        
        return None
    
    def save_embeddings(self, descriptions: List[str], embeddings: Any, 
                       model_name: str, language: str):
        """Save embeddings to cache"""
        import numpy as np
        
        cache_path = self.get_cache_path(descriptions, model_name, language)
        
        try:
            np.savez_compressed(
                cache_path,
                descriptions=descriptions,
                embeddings=embeddings,
                model_name=model_name,
                language=language,
                timestamp=datetime.now().isoformat()
            )
            logger.info(f"Saved embeddings to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")


class CSVCache:
    """Cache for CSV files with modification time tracking"""
    
    _cache: Dict[str, Tuple[Any, float]] = {}
    
    @classmethod
    def load_csv(cls, csv_path: str) -> Any:
        """Load CSV with caching based on modification time"""
        import pandas as pd
        import os
        
        csv_path = str(Path(csv_path).absolute())
        mtime = os.path.getmtime(csv_path)
        
        # Check if cached and still valid
        if csv_path in cls._cache:
            cached_df, cached_mtime = cls._cache[csv_path]
            if cached_mtime == mtime:
                logger.debug(f"Using cached CSV: {csv_path}")
                return cached_df.copy()
        
        # Load from disk
        logger.info(f"Loading CSV from disk: {csv_path}")
        df = pd.read_csv(csv_path)
        cls._cache[csv_path] = (df, mtime)
        
        return df.copy()
    
    @classmethod
    def clear_cache(cls):
        """Clear the CSV cache"""
        cls._cache.clear()