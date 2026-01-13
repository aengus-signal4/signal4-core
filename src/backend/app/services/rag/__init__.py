"""
RAG Summary Services
====================

Infrastructure for building RAG-based content summaries through:
1. Time-windowed FAISS index building
2. Clustering-based topic discovery
3. LLM-powered topic labeling and summarization
4. Hierarchical summarization with citation tracking
"""

from .topic_discovery import TopicDiscovery
from .topic_labeler import TopicLabeler
from .citation_manager import CitationManager, Citation
from .dynamic_grouper import DynamicGrouper, GroupFilter, GroupResult
from .segment_selector import SegmentSelector, SamplingWeights
from .theme_summarizer import ThemeSummarizer, ThemeSummary
from .meta_summarizer import MetaSummarizer, MetaSummary
from .text_generator import TextGenerator, PromptTemplateManager, PromptTemplate
from .quantitative_analyzer import QuantitativeAnalyzer

__all__ = [
    'TopicDiscovery',
    'TopicLabeler',
    'CitationManager',
    'Citation',
    'DynamicGrouper',
    'GroupFilter',
    'GroupResult',
    'SegmentSelector',
    'SamplingWeights',
    'ThemeSummarizer',
    'ThemeSummary',
    'MetaSummarizer',
    'MetaSummary',
    'TextGenerator',
    'PromptTemplateManager',
    'PromptTemplate',
    'QuantitativeAnalyzer'
]
