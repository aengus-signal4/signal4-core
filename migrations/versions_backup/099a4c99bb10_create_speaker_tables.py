"""create_speaker_tables

Revision ID: 099a4c99bb10
Revises: 
Create Date: 2025-03-29 07:14:54.653122

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from src.database.types import Vector

# revision identifiers, used by Alembic.
revision: str = '099a4c99bb10'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create speakers table
    op.create_table('speakers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('global_id', sa.String(length=255), nullable=False),
        sa.Column('universal_name', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('total_segments', sa.Integer(), server_default='0', nullable=False),
        sa.Column('total_duration', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_content_id', sa.String(length=255), nullable=True),
        sa.Column('appearance_count', sa.Integer(), server_default='0', nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('global_id'),
        sa.UniqueConstraint('universal_name'),
        schema='public'
    )
    
    # Create speaker_embeddings table
    op.create_table('speaker_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('speaker_id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.String(length=255), nullable=False),
        sa.Column('embedding', sa.LargeBinary(), nullable=False),
        sa.Column('segment_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('total_duration', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['speaker_id'], ['public.speakers.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='public'
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('speaker_embeddings', schema='public')
    op.drop_table('speakers', schema='public')
