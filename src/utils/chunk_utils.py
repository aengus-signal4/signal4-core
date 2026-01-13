"""Utilities for planning and managing content chunks."""

from typing import List, Dict
import logging
from sqlalchemy.orm import Session
from ..database.models import Content, ContentChunk
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def create_chunk_plan(duration: float, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Create a plan for chunking content based on duration and configuration.
    
    Args:
        duration: Total duration in seconds
        chunk_size: Size of each chunk in seconds
        chunk_overlap: Overlap between chunks in seconds
        
    Returns:
        List of chunk specifications with index, start time, end time, and duration
    """
    chunks = []
    current_pos = 0
    chunk_index = 0
    
    # Calculate chunks with fixed step size
    step_size = chunk_size - chunk_overlap
    
    while current_pos < duration:
        chunk_end = min(current_pos + chunk_size, duration)
        chunk_duration = chunk_end - current_pos
        
        # Skip tiny chunks at the end
        if chunk_duration < 1.0:
            logger.info("Skipping final tiny chunk < 1.0s")
            break
            
        chunks.append({
            'index': chunk_index,
            'start_time': current_pos,
            'end_time': chunk_end,
            'duration': chunk_duration
        })
        
        # Move to next chunk with fixed step size
        current_pos += step_size
        chunk_index += 1
        
        # Log progress for very long content
        if chunk_index % 100 == 0:
            logger.info(f"Planned {chunk_index} chunks so far...")
    
    return chunks

def store_chunk_plan(session: Session, content: Content, chunks: List[Dict]) -> bool:
    """Store chunk plan in the database.
    
    Args:
        session: SQLAlchemy session
        content: Content object
        chunks: List of chunk specifications from create_chunk_plan()
        
    Returns:
        bool: True if successful
    """
    try:
        # Clear any existing chunks
        session.query(ContentChunk).filter_by(content_id=content.id).delete()
        
        # Create new chunk records
        for chunk in chunks:
            content_chunk = ContentChunk(
                content_id=content.id,
                chunk_index=chunk['index'],
                start_time=chunk['start_time'],
                end_time=chunk['end_time'],
                duration=chunk['duration'],
                extraction_status='pending',
                transcription_status='pending',
                created_at=datetime.now(timezone.utc)
            )
            session.add(content_chunk)
        
        # Update content record
        content.total_chunks = len(chunks)
        content.chunks_processed = 0
        content.chunks_status = {}
        content.last_updated = datetime.now(timezone.utc)
        
        session.commit()
        return True
        
    except Exception as e:
        logger.error(f"Error storing chunk plan: {str(e)}")
        session.rollback()
        return False

def get_chunk_plan(session: Session, content: Content) -> List[Dict]:
    """Retrieve chunk plan from database.
    
    Args:
        session: SQLAlchemy session
        content: Content object
        
    Returns:
        List of chunk specifications
    """
    try:
        chunks = session.query(ContentChunk)\
            .filter_by(content_id=content.id)\
            .order_by(ContentChunk.chunk_index)\
            .all()
            
        return [{
            'index': chunk.chunk_index,
            'start_time': chunk.start_time,
            'end_time': chunk.end_time,
            'duration': chunk.duration,
            'extraction_status': chunk.extraction_status,
            'transcription_status': chunk.transcription_status
        } for chunk in chunks]
        
    except Exception as e:
        logger.error(f"Error getting chunk plan: {str(e)}")
        return [] 