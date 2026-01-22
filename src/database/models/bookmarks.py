"""
User Bookmarks Model
====================

Stores user bookmarks for episodes, channels, speakers, and segments.
Each bookmark can have an optional note for personal annotations.

Key Design:
- client_id: String from frontend config (not a database user ID)
- entity_type + entity_id: Polymorphic reference to bookmarked entity
- note: Searchable text field for user annotations
- Unique constraint ensures one bookmark per (user, entity_type, entity_id)
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Index, UniqueConstraint
)
from sqlalchemy.sql import func
from datetime import datetime

from .base import Base


class Bookmark(Base):
    """
    User bookmarks for episodes, channels, speakers, and segments.

    Entity types:
    - 'episode': entity_id = Content.id (integer)
    - 'channel': entity_id = Channel.id (integer)
    - 'speaker': entity_id = SpeakerIdentity.id (integer)
    - 'segment': entity_id = EmbeddingSegment.id (integer)

    Note: client_id comes from frontend config file (e.g., 'cprmv_dg'),
    not from a database users table. This allows user management to
    remain config-driven while persisting bookmarks in the database.
    """
    __tablename__ = 'bookmarks'

    id = Column(Integer, primary_key=True)

    # User identification (from config, not DB)
    client_id = Column(String(100), nullable=False, index=True)

    # Entity reference (polymorphic)
    entity_type = Column(String(20), nullable=False)  # 'episode', 'channel', 'speaker', 'segment'
    entity_id = Column(Integer, nullable=False)       # ID of the bookmarked entity

    # User note (searchable)
    note = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        # Unique constraint: one bookmark per user per entity
        UniqueConstraint('client_id', 'entity_type', 'entity_id', name='uq_bookmark_user_entity'),

        # Primary lookup: user's bookmarks
        Index('idx_bookmarks_client_id', 'client_id'),

        # Lookup by entity (for checking if bookmarked)
        Index('idx_bookmarks_entity', 'entity_type', 'entity_id'),

        # User's bookmarks sorted by creation date
        Index('idx_bookmarks_client_created', 'client_id', 'created_at'),

        # Full-text search on notes using trigram similarity
        # Requires pg_trgm extension: CREATE EXTENSION IF NOT EXISTS pg_trgm;
        Index(
            'idx_bookmarks_note_trgm',
            'note',
            postgresql_using='gin',
            postgresql_ops={'note': 'gin_trgm_ops'}
        ),
    )

    def __repr__(self):
        return f"<Bookmark(id={self.id}, client_id='{self.client_id}', type='{self.entity_type}', entity_id={self.entity_id})>"
