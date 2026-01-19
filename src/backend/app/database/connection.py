"""
Database Connection
===================

SQLAlchemy database session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

from ..config import settings

logger = logging.getLogger(__name__)

# Create engine with optimized connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,      # Verify connections before using (detects stale connections)
    pool_size=20,            # Maintain 20 connections in the pool
    max_overflow=40,         # Allow up to 40 additional connections under load
    pool_recycle=3600,       # Recycle connections after 1 hour (prevents stale connections)
    pool_timeout=30,         # Wait up to 30s for available connection
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db_session() -> Session:
    """
    Context manager for database sessions.

    Usage:
        with get_db_session() as session:
            results = session.query(Model).all()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        session.close()

def get_db():
    """
    Dependency for FastAPI endpoints.

    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            results = db.query(Model).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
