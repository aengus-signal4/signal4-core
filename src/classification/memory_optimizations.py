#!/usr/bin/env python3
"""
Memory optimization utilities for theme classification
"""

import gc
import logging
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from functools import lru_cache

logger = logging.getLogger(__name__)

class MemoryEfficientBatchProcessor:
    """Process large datasets in memory-efficient batches"""
    
    @staticmethod
    def process_in_chunks(items: List[Any], chunk_size: int = 100) -> Iterator[List[Any]]:
        """Yield chunks of items to process"""
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]
    
    @staticmethod
    def lazy_load_csv(csv_path: str, chunksize: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Lazy load large CSV files in chunks
        """
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            yield chunk
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize pandas DataFrame memory usage
        """
        # Convert string columns to category if they have few unique values
        for col in df.select_dtypes(include=['object']):
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        
        # Downcast numeric types
        for col in df.select_dtypes(include=['int']):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']):
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    @staticmethod
    def clear_memory():
        """Force garbage collection to free memory"""
        gc.collect()


class EmbeddingPool:
    """
    Pool embeddings to avoid redundant model calls and memory duplication
    """
    
    def __init__(self, model, max_cache_size: int = 10000):
        self.model = model
        self.cache = {}
        self.max_cache_size = max_cache_size
        
    @lru_cache(maxsize=1000)
    def _encode_single(self, text: str) -> np.ndarray:
        """Encode single text with caching"""
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode texts in optimized batches
        """
        embeddings = []
        
        # Check cache first
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)
        
        # Batch encode uncached texts
        if uncached_texts:
            # Process in smaller batches to avoid memory issues
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                
                # Update cache and results
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    text_hash = hash(text)
                    
                    # Manage cache size
                    if len(self.cache) >= self.max_cache_size:
                        # Remove oldest item (simple FIFO)
                        self.cache.pop(next(iter(self.cache)))
                    
                    self.cache[text_hash] = embedding
                    original_index = uncached_indices[batch_start + j]
                    embeddings[original_index] = embedding
        
        return embeddings


class StreamingCSVWriter:
    """
    Write large CSV files efficiently using streaming
    """
    
    def __init__(self, output_path: str, columns: List[str], buffer_size: int = 1000):
        self.output_path = output_path
        self.columns = columns
        self.buffer_size = buffer_size
        self.buffer = []
        self.file_handle = None
        self.writer = None
        
    def __enter__(self):
        import csv
        self.file_handle = open(self.output_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.columns)
        self.writer.writeheader()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        if self.file_handle:
            self.file_handle.close()
    
    def write_row(self, row_dict: Dict[str, Any]):
        """Add row to buffer"""
        self.buffer.append(row_dict)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffer to disk"""
        if self.buffer and self.writer:
            self.writer.writerows(self.buffer)
            self.buffer = []
            self.file_handle.flush()


class DataFrameOptimizer:
    """
    Optimize DataFrame operations for the theme classifier
    """
    
    @staticmethod
    def deduplicate_efficiently(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
        """
        Efficiently deduplicate DataFrame
        """
        # Sort by the columns we care about for better cache locality
        df = df.sort_values(subset)
        
        # Use numpy for faster deduplication on large datasets
        if len(df) > 10000:
            # Convert to numpy for faster operations
            arr = df[subset].values
            _, unique_indices = np.unique(arr, return_index=True, axis=0)
            return df.iloc[unique_indices]
        else:
            return df.drop_duplicates(subset=subset)
    
    @staticmethod
    def merge_efficiently(df1: pd.DataFrame, df2: pd.DataFrame, 
                         on: str, how: str = 'left') -> pd.DataFrame:
        """
        Efficiently merge DataFrames
        """
        # Set index for faster merging
        if on in df1.columns and on in df2.columns:
            df1 = df1.set_index(on)
            df2 = df2.set_index(on)
            result = df1.merge(df2, left_index=True, right_index=True, how=how)
            return result.reset_index()
        else:
            return pd.merge(df1, df2, on=on, how=how)