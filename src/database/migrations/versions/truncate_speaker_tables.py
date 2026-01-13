"""Truncate speaker-related tables to start fresh

Revision ID: truncate_speaker_tables
Revises: fix_speaker_embeddings
Create Date: 2024-04-02 06:15:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session
from src.database.models import SpeakerTranscription, SpeakerEmbedding, Speaker

# revision identifiers, used by Alembic.
revision = 'truncate_speaker_tables'
down_revision = 'fix_speaker_embeddings'
branch_labels = None
depends_on = None

def upgrade():
    # Get database connection
    connection = op.get_bind()
    session = Session(bind=connection)
    
    try:
        # Truncate tables in correct order (respecting foreign key constraints)
        session.execute(sa.text('TRUNCATE TABLE speaker_transcriptions CASCADE'))
        session.execute(sa.text('TRUNCATE TABLE speaker_embeddings CASCADE'))
        session.execute(sa.text('TRUNCATE TABLE speakers CASCADE'))
        
        # Reset the sequences
        session.execute(sa.text('ALTER SEQUENCE speaker_transcriptions_id_seq RESTART WITH 1'))
        session.execute(sa.text('ALTER SEQUENCE speaker_embeddings_id_seq RESTART WITH 1'))
        session.execute(sa.text('ALTER SEQUENCE speakers_id_seq RESTART WITH 1'))
        
        session.commit()
        print("Successfully truncated speaker-related tables")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()

def downgrade():
    # Cannot restore truncated data
    pass 