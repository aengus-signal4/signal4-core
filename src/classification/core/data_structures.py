"""
Data structures for theme classification pipelines.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.classification.speaker_assignment import SpeakerSegment


@dataclass
class UnifiedConfig:
    """Configuration for unified theme classification pipeline (keyword + semantic)"""
    subthemes_csv: str
    output_csv: str
    keywords_file: Optional[str] = None
    similarity_threshold: float = 0.6
    keyword_boost: float = 0.1
    keyword_threshold_reduction: float = 0.04
    model_name: str = "tier_2"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    save_intermediate: bool = True
    use_gpu: bool = True
    project: Optional[str] = None
    force_language: Optional[str] = None

    # Model server configuration
    model_server_url: Optional[str] = None
    use_model_server: bool = True

    # Test mode
    test_mode: Optional[int] = None

    # Checkpoint configuration
    checkpoint_file: Optional[str] = None
    resume: bool = True

    # FAISS configuration
    use_faiss: bool = False
    faiss_index_path: Optional[str] = None

    # Cache configuration
    skip_stage1_cache: bool = False

    # Cleanup configuration
    cleanup_stale_segments: bool = False


@dataclass
class SemanticConfig:
    """Configuration for semantic-only theme classification pipeline"""
    subthemes_csv: str
    output_csv: str
    similarity_threshold: float = 0.38
    model_name: str = "tier_1"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    save_intermediate: bool = True
    use_gpu: bool = True
    project: Optional[str] = None
    force_language: Optional[str] = None

    # Model server configuration
    model_server_url: Optional[str] = None
    use_model_server: bool = True

    # Test mode
    test_mode: Optional[int] = None

    # Checkpoint configuration
    checkpoint_file: Optional[str] = None
    resume: bool = True

    # FAISS configuration (required for semantic-only)
    use_faiss: bool = True
    faiss_index_path: Optional[str] = None

    # Cache configuration
    skip_stage1_cache: bool = False

    # Cleanup configuration
    cleanup_stale_segments: bool = False


@dataclass
class SearchCandidate:
    """Unified candidate structure for search results"""
    segment_id: int
    content_id: str
    episode_title: str
    episode_channel: str
    segment_text: str
    start_time: float
    end_time: float
    segment_index: int

    # Search metadata
    similarity_score: float = 0.0
    matched_via: str = 'semantic'  # 'keyword', 'semantic', or 'both'
    matched_keywords: List[str] = None
    matching_themes: List[str] = None
    original_url: str = None

    # Semantic search metadata (for semantic classifier)
    theme_similarities: Dict[int, float] = None  # theme_id -> similarity score
    subtheme_similarities: Dict[int, float] = None  # subtheme_id -> similarity score

    # Speaker attribution fields
    speaker_attributed_text: str = None
    speaker_segments: List[SpeakerSegment] = None
    speaker_info: Dict[str, Any] = None
    source_sentence_ids: List[int] = None  # Sentence indices from sentences table
    sentence_texts: List[str] = None  # Sentence texts from sentences table
    speaker_ids: List[int] = None

    # LLM classification results (filled later)
    theme_ids: List[int] = None
    theme_names: List[str] = None
    theme_confidence: float = 0.0
    theme_reasoning: str = ""
    subtheme_results: Dict[int, Dict[str, Any]] = None

    # Validation results (filled in stage 4)
    validation_results: Dict[int, Dict[str, Any]] = None


@dataclass
class ClassificationResult:
    """Result from LLM classification"""
    theme_id: int = 0
    theme_name: str = ""
    theme_ids: List[int] = None
    theme_names: List[str] = None
    subtheme_ids: List[int] = None
    subtheme_names: List[str] = None
    confidence: float = 0.0
    reasoning: str = ""
