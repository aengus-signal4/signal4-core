#!/usr/bin/env python3
"""
Add rebase tracking fields to SpeakerEmbedding table
===================================================

This migration adds the new fields needed for the speaker clustering rebase system:
- rebase_status: Tracks the processing status of each embedding
- temporary_speaker_id: Temporary speaker ID used during processing
- rebase_batch_id: Batch ID for tracking rebase operations
- rebase_processed_at: When this embedding was processed by rebase

Author: Signal4 AI
Date: 2024-07-11
"""

import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

# Add the project root to Python path
sys.path.append(str(get_project_root()))

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from sqlalchemy import text

logger = setup_worker_logger('migrations')

def upgrade():
    """Add rebase tracking fields to SpeakerEmbedding table"""
    
    logger.info("Adding rebase tracking fields to speaker_embeddings table...")
    
    with get_session() as session:
        # Create the enum type first (if it doesn't exist)
        try:
            session.execute(text("""
                CREATE TYPE rebase_status AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');
            """))
            session.commit()
        except Exception as e:
            if "already exists" in str(e):
                logger.info("rebase_status enum already exists, updating values")
                session.rollback()
                # Update enum values to uppercase
                session.execute(text("""
                    ALTER TYPE rebase_status ADD VALUE 'PENDING' IF NOT EXISTS;
                    ALTER TYPE rebase_status ADD VALUE 'PROCESSING' IF NOT EXISTS;
                    ALTER TYPE rebase_status ADD VALUE 'COMPLETED' IF NOT EXISTS;
                    ALTER TYPE rebase_status ADD VALUE 'FAILED' IF NOT EXISTS;
                """))
                session.commit()
            else:
                session.rollback()
                raise
        
        # Add the new columns (if they don't exist)
        session.execute(text("""
            ALTER TABLE speaker_embeddings 
            ADD COLUMN IF NOT EXISTS rebase_status rebase_status DEFAULT 'COMPLETED' NOT NULL,
            ADD COLUMN IF NOT EXISTS temporary_speaker_id VARCHAR,
            ADD COLUMN IF NOT EXISTS rebase_batch_id VARCHAR,
            ADD COLUMN IF NOT EXISTS rebase_processed_at TIMESTAMP;
        """))
        
        # Create indexes for efficient querying (if they don't exist)
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_rebase_status ON speaker_embeddings(rebase_status);
            CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_rebase_batch ON speaker_embeddings(rebase_batch_id);
            CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_temp_speaker ON speaker_embeddings(temporary_speaker_id);
            CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_rebase_pending ON speaker_embeddings(rebase_status, created_at);
        """))
        
        # Mark existing embeddings as completed (they were processed by the old system)
        session.execute(text("""
            UPDATE speaker_embeddings 
            SET rebase_status = 'COMPLETED'
            WHERE rebase_status = 'PENDING' OR rebase_status = 'pending';
        """))
        
        try:
            session.commit()
            logger.info("Successfully added rebase tracking fields")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add rebase tracking fields: {e}")
            raise

def downgrade():
    """Remove rebase tracking fields from SpeakerEmbedding table"""
    
    logger.info("Removing rebase tracking fields from speaker_embeddings table...")
    
    with get_session() as session:
        # Drop indexes first
        session.execute(text("""
            DROP INDEX IF EXISTS idx_speaker_embeddings_rebase_status;
            DROP INDEX IF EXISTS idx_speaker_embeddings_rebase_batch;
            DROP INDEX IF EXISTS idx_speaker_embeddings_temp_speaker;
            DROP INDEX IF EXISTS idx_speaker_embeddings_rebase_pending;
        """))
        
        # Remove the columns
        session.execute(text("""
            ALTER TABLE speaker_embeddings 
            DROP COLUMN IF EXISTS rebase_status,
            DROP COLUMN IF EXISTS temporary_speaker_id,
            DROP COLUMN IF EXISTS rebase_batch_id,
            DROP COLUMN IF EXISTS rebase_processed_at;
        """))
        
        # Drop the enum type
        session.execute(text("""
            DROP TYPE IF EXISTS rebase_status;
        """))
        
        session.commit()
        logger.info("Successfully removed rebase tracking fields")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add rebase tracking fields migration")
    parser.add_argument("--downgrade", action="store_true", help="Downgrade instead of upgrade")
    args = parser.parse_args()
    
    if args.downgrade:
        downgrade()
    else:
        upgrade()