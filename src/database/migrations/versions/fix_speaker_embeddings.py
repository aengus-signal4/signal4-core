"""Fix speaker embeddings binary data

Revision ID: fix_speaker_embeddings
Revises: update_speaker_embedding_model
Create Date: 2024-04-01 06:50:00.000000

"""
from alembic import op
import sqlalchemy as sa
import numpy as np
from sqlalchemy.orm import Session
from src.database.models import SpeakerEmbedding

# revision identifiers, used by Alembic.
revision = 'fix_speaker_embeddings'
down_revision = 'update_speaker_embedding_model'
branch_labels = None
depends_on = None

def upgrade():
    # Get database connection
    connection = op.get_bind()
    session = Session(bind=connection)
    
    try:
        # Get all embeddings
        embeddings = session.query(SpeakerEmbedding).all()
        
        for emb in embeddings:
            try:
                # Try to load the embedding
                raw_data = np.frombuffer(emb.embedding, dtype=np.float32)
                
                # Check if data is valid
                if np.isnan(raw_data).any() or np.isinf(raw_data).any():
                    # If invalid, try to fix it
                    # First try to normalize the raw data
                    norm = np.linalg.norm(raw_data)
                    if norm > 0:
                        fixed_data = raw_data / norm
                        # Store the fixed data
                        emb.embedding = fixed_data.astype(np.float32).tobytes()
                    else:
                        # If normalization fails, skip this embedding
                        print(f"Skipping invalid embedding for speaker {emb.speaker_id}")
                        continue
            except Exception as e:
                print(f"Error processing embedding for speaker {emb.speaker_id}: {str(e)}")
                continue
        
        session.commit()
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()

def downgrade():
    # No downgrade needed as this is a data fix
    pass 