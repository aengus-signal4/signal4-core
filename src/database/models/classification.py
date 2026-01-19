"""
Classification models for theme and content analysis.

Contains:
- ClassificationSchema: Theme/subtheme definitions
- ThemeClassification: Detailed classification results
- CprmvAnalysis: CPRMV-specific theme classification
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Index, ARRAY,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime

from .base import Base


class ClassificationSchema(Base):
    """
    Classification schemas for theme/subtheme definitions.

    Stores versioned theme classification schemas loaded from CSV files.
    Enables multiple classification runs with different schema versions.

    Attributes:
        id: Primary key
        name: Schema name (e.g., 'CPRMV', 'LGBTQ_Education')
        version: Schema version string (e.g., 'v1.0', '2025-01-15')
        description: Optional description of schema
        themes_json: JSONB with theme definitions {theme_id: {name, description_en, description_fr}}
        subthemes_json: JSONB with subtheme definitions {subtheme_id: {theme_id, name, description_en, description_fr}}
        created_at: When schema was loaded

    Indexes:
        - Unique constraint on (name, version)
        - Index on name for lookups
    """
    __tablename__ = 'classification_schemas'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)

    # Schema definitions
    themes_json = Column(JSONB, nullable=False)
    subthemes_json = Column(JSONB, nullable=False)

    # Cached query embeddings for semantic search
    # Format: {subtheme_id: {lang: [embedding_vector]}}
    # Example: {"Q1": {"en": [0.1, 0.2, ...], "fr": [0.3, 0.4, ...]}}
    query_embeddings = Column(JSONB, nullable=True)

    # Metadata
    source_file = Column(String(500), nullable=True)  # Path to original CSV
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    classifications = relationship("ThemeClassification", back_populates="schema")

    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_schema_name_version'),
        Index('idx_schema_name', 'name'),
    )


class ThemeClassification(Base):
    """
    Detailed theme classification results with multi-stage tracking.

    Stores comprehensive classification results from semantic theme classifiers,
    tracking similarity scores, LLM classifications, and validation results
    across all pipeline stages.

    Attributes:
        id: Primary key
        segment_id: Foreign key to embedding_segments
        schema_id: Foreign key to classification_schemas

        # Final results (for fast queries)
        theme_ids: Array of theme IDs (integers or strings depending on schema)
        subtheme_ids: Array of subtheme IDs
        high_confidence_themes: Array of themes with confidence >= 0.75

        # Stage-by-stage results
        stage1_similarities: JSONB with {theme_id: score, subtheme_id: score}
        stage3_results: JSONB with LLM subtheme classifications per theme
        stage4_validations: JSONB with Likert scale validation scores

        # Aggregate confidence
        final_confidence_scores: JSONB with {theme_id: confidence, subtheme_id: confidence}

        # Match metadata
        matched_via: 'semantic', 'keyword', 'both'
        max_similarity_score: Highest similarity score from stage 1

        # Embedding for semantic search
        embedding: 2000-dim embedding copied from embedding_segments

        # Timestamps
        created_at: When classification was created
        updated_at: Last update

    Indexes:
        - GIN on theme_ids, subtheme_ids, high_confidence_themes
        - IVFFlat on embedding for semantic search
        - Unique constraint on (segment_id, schema_id)
    """
    __tablename__ = 'theme_classifications'

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey('embedding_segments.id'), nullable=False, index=True)
    schema_id = Column(Integer, ForeignKey('classification_schemas.id'), nullable=False, index=True)

    # Final classification results
    theme_ids = Column(ARRAY(String), nullable=False, server_default='{}')
    subtheme_ids = Column(ARRAY(String), nullable=False, server_default='{}')
    high_confidence_themes = Column(ARRAY(String), nullable=False, server_default='{}')

    # Stage-by-stage results (JSONB for flexibility)
    stage1_similarities = Column(JSONB, nullable=True)  # {theme_id: score, subtheme_id: score}
    stage3_results = Column(JSONB, nullable=True)  # {theme_id: {subtheme_ids, confidence, reasoning}}
    stage4_validations = Column(JSONB, nullable=True)  # {theme_id: {subtheme_id: {confidence, category}}}

    # Final aggregated confidence scores
    final_confidence_scores = Column(JSONB, nullable=False, server_default='{}')  # {theme_id: conf, subtheme_id: conf}

    # Match metadata
    matched_via = Column(String(20), nullable=True)
    max_similarity_score = Column(Float, nullable=True)

    # Embedding for semantic search (copied from embedding_segments.embedding_alt)
    embedding = Column(Vector(2000), nullable=True)

    # Stage 5: Final relevance check (automated LLM)
    # {is_relevant: bool, reasoning: str, relevance: str, model: str, checked_at: str}
    stage5_final_check = Column(JSONB, nullable=True)

    # Stage 6: LLM false positive detection
    # Specialized LLM check to identify content that Stage 5 incorrectly marked as relevant:
    # - pro_progressive: Defends feminist/LGBTQ+ positions (not attacking them)
    # - documenting_harm: Reports/documents prejudice without promoting it
    # - quote_without_endorsement: Quotes someone else's position without endorsing it
    # {
    #   is_false_positive: bool,
    #   false_positive_type: str|null ('pro_progressive'|'documenting_harm'|'quote_without_endorsement'),
    #   reasoning: str,
    #   confidence: str ('definitely'|'probably'|'possibly'|'probably_not'|'definitely_not'),
    #   model: str,
    #   checked_at: str
    # }
    stage6_false_positive_check = Column(JSONB, nullable=True)

    # Stage 7: Expanded context re-check for Stage 6 false positives
    # Re-evaluates segments with ±20 second context window
    # {
    #   is_false_positive: bool,
    #   speaker_stance: str ('strongly_holds'|'holds'|'leans_holds'|'neutral'|'leans_rejects'|'rejects'|'strongly_rejects'),
    #   reasoning: str,
    #   original_stance: str (from stage6),
    #   context_window_seconds: int,
    #   checked_at: str
    # }
    stage7_expanded_context = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    segment = relationship("EmbeddingSegment", foreign_keys=[segment_id])
    schema = relationship("ClassificationSchema", back_populates="classifications")

    __table_args__ = (
        # Unique constraint for idempotency
        UniqueConstraint('segment_id', 'schema_id', name='uq_segment_schema'),

        # GIN indexes for fast array containment queries
        Index('idx_theme_class_theme_ids', 'theme_ids', postgresql_using='gin'),
        Index('idx_theme_class_subtheme_ids', 'subtheme_ids', postgresql_using='gin'),
        Index('idx_theme_class_high_conf', 'high_confidence_themes', postgresql_using='gin'),

        # IVFFlat index for semantic search
        Index('idx_theme_class_embedding', 'embedding',
              postgresql_using='ivfflat',
              postgresql_with={'lists': 100},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),

        # Standard indexes
        Index('idx_theme_class_segment_id', 'segment_id'),
        Index('idx_theme_class_schema_id', 'schema_id'),
        Index('idx_theme_class_matched_via', 'matched_via'),
    )


class CprmvAnalysis(Base):
    """
    CPRMV theme classification results with embeddings for fast semantic search.

    This table stores theme classification results from the CPRMV project,
    enabling fast filtering by themes and direct semantic search within
    theme-filtered segments without requiring joins.

    Attributes:
        id: Primary key
        segment_id: Foreign key to embedding_segments
        themes: Array of all theme IDs (e.g., ['2B', '2C', '9A'])
        confidence_scores: JSONB with theme->confidence mapping (e.g., {"2B": 1.0, "2C": 0.75})
        high_confidence_themes: Array of themes with confidence >= 0.75 for fast filtering
        matched_via: How the segment was matched ('semantic', 'keyword', 'both')
        similarity_score: FAISS similarity score if matched via semantic search
        matched_keywords: Keywords that triggered the match (if applicable)
        embedding: 2000-dim embedding copied from embedding_segments.embedding_alt for fast search
        created_at: When this classification was added
        updated_at: Last update timestamp

    Indexes:
        - GIN on themes for fast array containment queries
        - GIN on high_confidence_themes for filtering >= 0.75 confidence
        - IVFFlat on embedding for fast semantic search within filtered results
        - Standard index on segment_id for joins

    Usage:
        # Find high-confidence anti-trans education segments
        SELECT * FROM cprmv_analysis
        WHERE high_confidence_themes @> ARRAY['2B']
        ORDER BY (confidence_scores->>'2B')::float DESC;

        # Semantic search within theme-filtered results
        SELECT * FROM cprmv_analysis
        WHERE high_confidence_themes @> ARRAY['2B', '2C']
        ORDER BY embedding <-> '[query_vector]'
        LIMIT 20;
    """
    __tablename__ = 'cprmv_analysis'

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey('embedding_segments.id'), nullable=False, unique=True, index=True)

    # Theme classifications
    themes = Column(ARRAY(String), nullable=False)
    confidence_scores = Column(JSONB, nullable=False)
    high_confidence_themes = Column(ARRAY(String), nullable=False)

    # Match metadata
    matched_via = Column(String(20), nullable=True)  # 'semantic', 'keyword', 'both'
    similarity_score = Column(Float, nullable=True)
    matched_keywords = Column(Text, nullable=True)

    # Embedding for direct semantic search (copied from embedding_segments.embedding_alt)
    embedding = Column(Vector(2000), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationship
    segment = relationship("EmbeddingSegment", foreign_keys=[segment_id])

    __table_args__ = (
        # GIN indexes for fast array containment queries
        Index('idx_cprmv_analysis_themes', 'themes', postgresql_using='gin'),
        Index('idx_cprmv_analysis_high_conf_themes', 'high_confidence_themes', postgresql_using='gin'),

        # IVFFlat index for fast semantic search (lists=100 for ~80k segments)
        Index('idx_cprmv_analysis_embedding', 'embedding',
              postgresql_using='ivfflat',
              postgresql_with={'lists': 100},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),

        # Standard indexes
        Index('idx_cprmv_analysis_segment_id', 'segment_id'),
        Index('idx_cprmv_analysis_matched_via', 'matched_via'),
    )
