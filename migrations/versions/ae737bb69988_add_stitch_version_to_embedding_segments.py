"""add_stitch_version_to_embedding_segments

Revision ID: ae737bb69988
Revises: 7ee56bdff4b0
Create Date: 2025-06-02 09:13:17.328503

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ae737bb69988'
down_revision: Union[str, None] = '7ee56bdff4b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add stitch_version column to embedding_segments table if it doesn't exist
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='embedding_segments' 
                AND column_name='stitch_version'
            ) THEN
                ALTER TABLE embedding_segments 
                ADD COLUMN stitch_version VARCHAR(50);
            END IF;
        END $$;
    """)
    
    # Add embedding_version column to embedding_segments table if it doesn't exist
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='embedding_segments' 
                AND column_name='embedding_version'
            ) THEN
                ALTER TABLE embedding_segments 
                ADD COLUMN embedding_version VARCHAR(50);
            END IF;
        END $$;
    """)
    
    # Add segment_version column to content table if it doesn't exist 
    # (for tracking which version was used to create segments)
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='content' 
                AND column_name='segment_version'
            ) THEN
                ALTER TABLE content 
                ADD COLUMN segment_version VARCHAR(50);
                
                -- Add index on segment_version for efficient queries
                CREATE INDEX idx_content_segment_version ON content(segment_version);
            END IF;
        END $$;
    """)


def downgrade() -> None:
    # Remove the columns if they exist
    op.execute("""
        ALTER TABLE embedding_segments 
        DROP COLUMN IF EXISTS stitch_version;
    """)
    
    op.execute("""
        ALTER TABLE embedding_segments 
        DROP COLUMN IF EXISTS embedding_version;
    """)
    
    op.execute("""
        DROP INDEX IF EXISTS idx_content_segment_version;
        
        ALTER TABLE content 
        DROP COLUMN IF EXISTS segment_version;
    """)