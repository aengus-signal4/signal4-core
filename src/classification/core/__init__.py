"""
Core components for theme classification pipelines.

This module provides shared utilities for theme classification:
- Data structures (configs, candidates, results)
- FAISS index loading and searching
- LLM-based classification
- Database query utilities
"""

from .data_structures import (
    SearchCandidate,
    ClassificationResult,
    UnifiedConfig,
    SemanticConfig
)
from .faiss_loader import FAISSIndexLoader
from .llm_classifier import LLMClassifier
from .database_utils import (
    enrich_segments_with_metadata,
    fetch_segment_metadata
)
from .database_writer import DatabaseWriter

__all__ = [
    'SearchCandidate',
    'ClassificationResult',
    'UnifiedConfig',
    'SemanticConfig',
    'FAISSIndexLoader',
    'LLMClassifier',
    'enrich_segments_with_metadata',
    'fetch_segment_metadata',
    'DatabaseWriter',
]
