"""
RAG Workflows
=============

Pre-built analysis workflows using AnalysisPipeline.

Workflow classes:
    - SimpleRAGWorkflow: Direct Q&A with optional query expansion
    - HierarchicalSummaryWorkflow: Multi-group, multi-theme analysis

Protocol interfaces (from interfaces.py):
    - WorkflowProtocol: Base contract for workflow classes
    - StreamingWorkflowProtocol: Workflows with streaming support
    - Component protocols: SegmentRetrieverProtocol, ThemeExtractorProtocol, etc.
    - TypedDicts: WorkflowEvent, SimpleRAGResult, HierarchicalResult
"""

from .simple_rag_workflow import SimpleRAGWorkflow
from .hierarchical_summary_workflow import HierarchicalSummaryWorkflow

# Re-export protocols from interfaces for external consumers
from ..interfaces import (
    # Workflow protocols
    WorkflowProtocol,
    StreamingWorkflowProtocol,
    # Component protocols
    SegmentRetrieverProtocol,
    ThemeExtractorProtocol,
    SegmentSelectorProtocol,
    TextGeneratorProtocol,
    QuantitativeAnalyzerProtocol,
    # Service protocols
    LLMServiceProtocol,
    EmbeddingServiceProtocol,
    # Pipeline protocols
    PipelineStepProtocol,
    AnalysisPipelineProtocol,
    # TypedDicts for result types
    WorkflowEvent,
    SimpleRAGResult,
    HierarchicalResult,
    ThemeData,
    SegmentData,
    QuantitativeMetrics,
)

__all__ = [
    # Workflow classes
    'SimpleRAGWorkflow',
    'HierarchicalSummaryWorkflow',
    # Workflow protocols
    'WorkflowProtocol',
    'StreamingWorkflowProtocol',
    # Component protocols
    'SegmentRetrieverProtocol',
    'ThemeExtractorProtocol',
    'SegmentSelectorProtocol',
    'TextGeneratorProtocol',
    'QuantitativeAnalyzerProtocol',
    # Service protocols
    'LLMServiceProtocol',
    'EmbeddingServiceProtocol',
    # Pipeline protocols
    'PipelineStepProtocol',
    'AnalysisPipelineProtocol',
    # TypedDicts
    'WorkflowEvent',
    'SimpleRAGResult',
    'HierarchicalResult',
    'ThemeData',
    'SegmentData',
    'QuantitativeMetrics',
]
