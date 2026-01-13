"""fix speaker migrations

Revision ID: fix_speaker_migrations
Revises: add_is_diarized_column
Create Date: 2024-03-30 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = 'fix_speaker_migrations'
down_revision = 'add_is_diarized_column'
branch_labels = None
depends_on = None

def upgrade():
    # Create speakers table if it doesn't exist
    op.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            id SERIAL PRIMARY KEY,
            global_id VARCHAR(255) NOT NULL UNIQUE,
            universal_name VARCHAR(255) NOT NULL UNIQUE,
            display_name VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            total_segments INTEGER DEFAULT 0,
            total_duration FLOAT DEFAULT 0.0,
            last_seen TIMESTAMP WITH TIME ZONE,
            last_content_id VARCHAR(255),
            appearance_count INTEGER DEFAULT 0,
            meta_data JSONB DEFAULT '{}'::jsonb NOT NULL
        )
    """)

    # Create speaker_embeddings table if it doesn't exist
    op.execute("""
        CREATE TABLE IF NOT EXISTS speaker_embeddings (
            id SERIAL PRIMARY KEY,
            speaker_id INTEGER NOT NULL REFERENCES speakers(id),
            content_id VARCHAR(255) NOT NULL,
            embedding BYTEA NOT NULL,
            segment_count INTEGER NOT NULL DEFAULT 1,
            total_duration FLOAT NOT NULL DEFAULT 0.0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create speaker_transcriptions table if it doesn't exist
    op.execute("""
        CREATE TABLE IF NOT EXISTS speaker_transcriptions (
            id SERIAL PRIMARY KEY,
            content_id INTEGER NOT NULL REFERENCES content(id),
            speaker_id INTEGER NOT NULL REFERENCES speakers(id),
            start_time FLOAT NOT NULL,
            end_time FLOAT NOT NULL,
            text TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes if they don't exist
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_speaker_transcription_content ON speaker_transcriptions(content_id);
        CREATE INDEX IF NOT EXISTS idx_speaker_transcription_speaker ON speaker_transcriptions(speaker_id);
        CREATE INDEX IF NOT EXISTS idx_speaker_transcription_time ON speaker_transcriptions(start_time, end_time);
        CREATE INDEX IF NOT EXISTS idx_speaker_transcription_turn_index ON speaker_transcriptions(content_id, turn_index);
        CREATE INDEX IF NOT EXISTS idx_speaker_transcription_text ON speaker_transcriptions USING gin (to_tsvector('english', text));
    """)

def downgrade():
    # Drop indexes first
    op.execute("""
        DROP INDEX IF EXISTS idx_speaker_transcription_text;
        DROP INDEX IF EXISTS idx_speaker_transcription_turn_index;
        DROP INDEX IF EXISTS idx_speaker_transcription_time;
        DROP INDEX IF EXISTS idx_speaker_transcription_speaker;
        DROP INDEX IF EXISTS idx_speaker_transcription_content;
    """)

    # Drop tables
    op.execute("""
        DROP TABLE IF EXISTS speaker_transcriptions;
        DROP TABLE IF EXISTS speaker_embeddings;
        DROP TABLE IF EXISTS speakers;
    """) 