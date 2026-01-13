"""Update SpeakerEmbedding model with numpy_embedding property

Revision ID: update_speaker_embedding_model
Revises: 20240320_add_speaker_transcription
Create Date: 2024-04-01 06:45:00.000000

"""
from alembic import op
import sqlalchemy as sa
import numpy as np

# revision identifiers, used by Alembic.
revision = 'update_speaker_embedding_model'
down_revision = '20240320_add_speaker_transcription'
branch_labels = None
depends_on = None

def upgrade():
    # No schema changes needed, just updating the model
    pass

def downgrade():
    # No schema changes to revert
    pass 