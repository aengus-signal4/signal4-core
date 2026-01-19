"""
Query Cache Models
==================

SQLAlchemy models for the query variation caching system.

See src/backend/app/config/workflows.py for full caching architecture docs.

Tables:
-------
- query_variations: Normalized library of unique query texts with embeddings
- query_expansions: Maps original queries to their expanded variations

Design Rationale:
-----------------
The system uses a normalized design to maximize embedding reuse:

1. **query_variations** stores each unique text exactly once with its embedding.
   Lookup is O(1) via MD5 hash index on text_hash.

2. **query_expansions** maps original queries to their variations via FK.
   An original query typically has 5-10 variations.

3. If the same variation appears in multiple query expansions (common for
   entity-related queries), it's embedded only once.

Example:
--------
    Original: "What is Mark Carney saying?"
    Variations: ["Mark Carney statements", "Carney policy positions", ...]

    Original: "Mark Carney tariffs"
    Variations: ["Mark Carney trade policy", "Carney tariffs discussion", ...]

    If both generate "Mark Carney policy" as a variation, the embedding
    is stored once in query_variations and referenced by both expansions.

Performance:
------------
- Embedding lookup: <10ms for 10 variations (direct hash lookup)
- Embedding generation: ~200ms per query (0.6B model)
- Full cache hit saves: ~2-4 seconds per request

Service Layer:
--------------
Use QueryVariationService for all operations (not direct model access):
    from src.backend.app.services.query_variation_service import QueryVariationService
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime

from pgvector.sqlalchemy import Vector

from src.database.models.base import Base


class QueryVariation(Base):
    """
    Normalized library of query texts with embeddings.

    Each unique query text is stored once with its embedding.
    Multiple original queries can reference the same variation.

    Attributes:
        id: Primary key
        text_hash: MD5 hash of normalized text for fast dedup lookup
        text: Original query text
        embedding: 1024-dim vector from Qwen 0.6B model
        created_at: When first created
        last_used_at: Last time this variation was used in a search
        usage_count: Number of times used (for analytics)
    """
    __tablename__ = 'query_variations'

    id = Column(Integer, primary_key=True)
    text_hash = Column(String(32), nullable=False, unique=True, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # 0.6B model dimension
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, default=datetime.utcnow)
    usage_count = Column(Integer, default=1)

    # Relationships
    expansions = relationship("QueryExpansion", back_populates="variation")

    def __repr__(self):
        return f"<QueryVariation(id={self.id}, text='{self.text[:30]}...', usage={self.usage_count})>"


class QueryExpansion(Base):
    """
    Maps original queries to their expanded variations.

    Each record links an original query to one variation.
    An original query typically has 5-10 variations.

    Attributes:
        id: Primary key
        original_query_hash: MD5 hash of original query for fast lookup
        original_query: Full original query text
        variation_id: FK to query_variations
        position: Order in the expansion list (0-indexed)
        created_at: When this mapping was created
    """
    __tablename__ = 'query_expansions'

    id = Column(Integer, primary_key=True)
    original_query_hash = Column(String(32), nullable=False, index=True)
    original_query = Column(Text, nullable=False)
    variation_id = Column(Integer, ForeignKey('query_variations.id', ondelete='CASCADE'), nullable=False)
    position = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Unique constraint: same original query can't have duplicate variations
    __table_args__ = (
        UniqueConstraint('original_query_hash', 'variation_id', name='uq_expansion_variation'),
    )

    # Relationships
    variation = relationship("QueryVariation", back_populates="expansions")

    def __repr__(self):
        return f"<QueryExpansion(original='{self.original_query[:20]}...', variation_id={self.variation_id})>"
