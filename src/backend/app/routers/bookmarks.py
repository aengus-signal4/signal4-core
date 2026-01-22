"""
Bookmarks Router
================

CRUD endpoints for user bookmarks with notes.
User is identified by client_id from NextAuth session (passed via X-Client-Id header).

Endpoints:
- POST /api/bookmarks - Create a bookmark (upsert)
- GET /api/bookmarks - List user's bookmarks with filtering and search
- GET /api/bookmarks/{id} - Get a specific bookmark
- PUT /api/bookmarks/{id} - Update a bookmark's note
- DELETE /api/bookmarks/{id} - Delete a bookmark
- GET /api/bookmarks/check/{entity_type}/{entity_id} - Check if entity is bookmarked
"""

import os
from datetime import datetime
from typing import Optional, List, Literal
from fastapi import APIRouter, HTTPException, Depends, Query, Header
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, desc, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert

from src.database.models import Bookmark, Content, Channel, SpeakerIdentity, EmbeddingSegment

from ..utils.backend_logger import get_logger
logger = get_logger("bookmarks_router")

router = APIRouter(prefix="/api/bookmarks", tags=["bookmarks"])

# Database connection
_engine = None
_SessionLocal = None


def get_db() -> Session:
    """Get database session"""
    global _engine, _SessionLocal

    if _SessionLocal is None:
        from src.backend.app.config.database import get_database_url
        _engine = create_engine(get_database_url(), pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine)

    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ====================
# Request/Response Models
# ====================

EntityType = Literal['episode', 'channel', 'speaker', 'segment']


class CreateBookmarkRequest(BaseModel):
    entity_type: EntityType
    entity_id: int
    note: Optional[str] = Field(None, max_length=2000)


class UpdateBookmarkRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=2000)


class BookmarkResponse(BaseModel):
    id: int
    entity_type: str
    entity_id: int
    note: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    # Populated entity data
    entity_title: Optional[str] = None
    entity_subtitle: Optional[str] = None
    # Segment-specific fields (only populated for entity_type='segment')
    segment_content_id: Optional[str] = None  # String content_id for media lookup
    segment_start_time: Optional[float] = None
    segment_end_time: Optional[float] = None

    class Config:
        from_attributes = True


class BookmarksListResponse(BaseModel):
    success: bool = True
    bookmarks: List[BookmarkResponse]
    total: int


class BookmarkCheckResponse(BaseModel):
    is_bookmarked: bool
    bookmark_id: Optional[int] = None


# ====================
# Helper Functions
# ====================

def get_entity_info(db: Session, entity_type: str, entity_id: int) -> dict:
    """
    Get display info for an entity.

    Returns dict with:
    - title: Display title
    - subtitle: Secondary info
    - For segments: content_id, start_time, end_time
    """
    result = {'title': None, 'subtitle': None}
    try:
        if entity_type == 'episode':
            content = db.query(Content).filter(Content.id == entity_id).first()
            if content:
                result['title'] = content.title
                result['subtitle'] = content.channel_name
        elif entity_type == 'channel':
            channel = db.query(Channel).filter(Channel.id == entity_id).first()
            if channel:
                result['title'] = channel.name
                result['subtitle'] = channel.platform
        elif entity_type == 'speaker':
            speaker = db.query(SpeakerIdentity).filter(SpeakerIdentity.id == entity_id).first()
            if speaker:
                result['title'] = speaker.display_name
                result['subtitle'] = speaker.role
        elif entity_type == 'segment':
            segment = db.query(EmbeddingSegment).filter(EmbeddingSegment.id == entity_id).first()
            if segment:
                # Get the parent content for context
                content = db.query(Content).filter(Content.id == segment.content_id).first()
                text_preview = segment.text[:100] + "..." if segment.text and len(segment.text) > 100 else segment.text
                result['title'] = text_preview
                result['subtitle'] = content.title if content else None
                # Include segment-specific data for media playback
                result['content_id'] = content.content_id if content else None
                result['start_time'] = segment.start_time
                result['end_time'] = segment.end_time
    except Exception as e:
        logger.warning(f"Error getting entity info for {entity_type}/{entity_id}: {e}")
    return result


def bookmark_to_response(db: Session, bookmark: Bookmark) -> BookmarkResponse:
    """Convert a Bookmark model to a response with entity info."""
    info = get_entity_info(db, bookmark.entity_type, bookmark.entity_id)
    return BookmarkResponse(
        id=bookmark.id,
        entity_type=bookmark.entity_type,
        entity_id=bookmark.entity_id,
        note=bookmark.note,
        created_at=bookmark.created_at,
        updated_at=bookmark.updated_at,
        entity_title=info.get('title'),
        entity_subtitle=info.get('subtitle'),
        # Segment-specific fields
        segment_content_id=info.get('content_id'),
        segment_start_time=info.get('start_time'),
        segment_end_time=info.get('end_time'),
    )


# ====================
# Endpoints
# ====================

@router.post("", response_model=BookmarkResponse)
async def create_bookmark(
    request: CreateBookmarkRequest,
    x_client_id: str = Header(..., description="Client ID from session"),
    db: Session = Depends(get_db)
):
    """
    Create or update a bookmark for an entity.
    If the bookmark already exists, updates the note.
    """
    # Check if bookmark already exists
    existing = db.query(Bookmark).filter(
        Bookmark.client_id == x_client_id,
        Bookmark.entity_type == request.entity_type,
        Bookmark.entity_id == request.entity_id
    ).first()

    if existing:
        # Update existing bookmark's note
        existing.note = request.note
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        logger.info(f"Updated bookmark {existing.id} for {x_client_id}")
        return bookmark_to_response(db, existing)

    # Create new bookmark
    bookmark = Bookmark(
        client_id=x_client_id,
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        note=request.note,
    )
    db.add(bookmark)
    db.commit()
    db.refresh(bookmark)

    logger.info(f"Created bookmark {bookmark.id} for {x_client_id}: {request.entity_type}/{request.entity_id}")
    return bookmark_to_response(db, bookmark)


@router.get("", response_model=BookmarksListResponse)
async def list_bookmarks(
    x_client_id: str = Header(..., description="Client ID from session"),
    entity_type: Optional[EntityType] = Query(None, description="Filter by entity type"),
    search: Optional[str] = Query(None, description="Search notes (keyword)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    List user's bookmarks with optional filtering.
    """
    query = db.query(Bookmark).filter(Bookmark.client_id == x_client_id)

    # Filter by entity type
    if entity_type:
        query = query.filter(Bookmark.entity_type == entity_type)

    # Keyword search on notes
    if search:
        # Use ILIKE for case-insensitive search
        # The trigram index will accelerate this
        query = query.filter(Bookmark.note.ilike(f'%{search}%'))

    # Get total count before pagination
    total = query.count()

    # Order by creation date (newest first) and paginate
    bookmarks = query.order_by(desc(Bookmark.created_at)).offset(offset).limit(limit).all()

    return BookmarksListResponse(
        bookmarks=[bookmark_to_response(db, b) for b in bookmarks],
        total=total,
    )


@router.get("/check/{entity_type}/{entity_id}", response_model=BookmarkCheckResponse)
async def check_bookmark(
    entity_type: EntityType,
    entity_id: int,
    x_client_id: str = Header(..., description="Client ID from session"),
    db: Session = Depends(get_db)
):
    """
    Check if user has bookmarked an entity.
    Returns bookmark_id if exists, null otherwise.
    """
    bookmark = db.query(Bookmark).filter(
        Bookmark.client_id == x_client_id,
        Bookmark.entity_type == entity_type,
        Bookmark.entity_id == entity_id
    ).first()

    return BookmarkCheckResponse(
        is_bookmarked=bookmark is not None,
        bookmark_id=bookmark.id if bookmark else None,
    )


@router.get("/{bookmark_id}", response_model=BookmarkResponse)
async def get_bookmark(
    bookmark_id: int,
    x_client_id: str = Header(..., description="Client ID from session"),
    db: Session = Depends(get_db)
):
    """Get a specific bookmark."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.id == bookmark_id,
        Bookmark.client_id == x_client_id  # Ensure user owns this bookmark
    ).first()

    if not bookmark:
        raise HTTPException(status_code=404, detail="Bookmark not found")

    return bookmark_to_response(db, bookmark)


@router.put("/{bookmark_id}", response_model=BookmarkResponse)
async def update_bookmark(
    bookmark_id: int,
    request: UpdateBookmarkRequest,
    x_client_id: str = Header(..., description="Client ID from session"),
    db: Session = Depends(get_db)
):
    """Update a bookmark's note."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.id == bookmark_id,
        Bookmark.client_id == x_client_id  # Ensure user owns this bookmark
    ).first()

    if not bookmark:
        raise HTTPException(status_code=404, detail="Bookmark not found")

    bookmark.note = request.note
    bookmark.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(bookmark)

    logger.info(f"Updated bookmark {bookmark_id} note for {x_client_id}")
    return bookmark_to_response(db, bookmark)


@router.delete("/{bookmark_id}")
async def delete_bookmark(
    bookmark_id: int,
    x_client_id: str = Header(..., description="Client ID from session"),
    db: Session = Depends(get_db)
):
    """Delete a bookmark."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.id == bookmark_id,
        Bookmark.client_id == x_client_id  # Ensure user owns this bookmark
    ).first()

    if not bookmark:
        raise HTTPException(status_code=404, detail="Bookmark not found")

    db.delete(bookmark)
    db.commit()

    logger.info(f"Deleted bookmark {bookmark_id} for {x_client_id}")
    return {"success": True, "message": "Bookmark deleted"}
