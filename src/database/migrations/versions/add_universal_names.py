"""add universal names

Revision ID: add_universal_names
Revises: create_speaker_tables
Create Date: 2024-03-30 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'add_universal_names'
down_revision = 'create_speaker_tables'
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns to speakers table
    op.add_column('speakers', sa.Column('universal_name', sa.String(), nullable=True))
    op.add_column('speakers', sa.Column('last_content_id', sa.String(), nullable=True))
    op.add_column('speakers', sa.Column('appearance_count', sa.Integer(), nullable=False, server_default='0'))
    
    # Make universal_name not nullable after adding it
    op.execute("UPDATE speakers SET universal_name = CONCAT('SPEAKER_', SUBSTRING(global_id FROM 5)) WHERE universal_name IS NULL")
    op.alter_column('speakers', 'universal_name', nullable=False)
    
    # Add unique constraint on universal_name
    op.create_unique_constraint('uq_speakers_universal_name', 'speakers', ['universal_name'])

def downgrade():
    # Remove unique constraint
    op.drop_constraint('uq_speakers_universal_name', 'speakers', type_='unique')
    
    # Remove columns
    op.drop_column('speakers', 'appearance_count')
    op.drop_column('speakers', 'last_content_id')
    op.drop_column('speakers', 'universal_name') 