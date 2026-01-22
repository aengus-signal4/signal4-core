"""Add bookmarks table

Revision ID: a1b2c3d4e5f6
Revises: upgrade_api_key_hash_to_bcrypt
Create Date: 2026-01-21 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'upgrade_api_key_hash_to_bcrypt'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure pg_trgm extension exists for trigram search
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')

    # Create bookmarks table
    op.create_table('bookmarks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('client_id', sa.String(length=100), nullable=False),
        sa.Column('entity_type', sa.String(length=20), nullable=False),
        sa.Column('entity_id', sa.Integer(), nullable=False),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Unique constraint: one bookmark per user per entity
    op.create_unique_constraint('uq_bookmark_user_entity', 'bookmarks', ['client_id', 'entity_type', 'entity_id'])

    # Indexes
    op.create_index('idx_bookmarks_client_id', 'bookmarks', ['client_id'], unique=False)
    op.create_index('idx_bookmarks_entity', 'bookmarks', ['entity_type', 'entity_id'], unique=False)
    op.create_index('idx_bookmarks_client_created', 'bookmarks', ['client_id', 'created_at'], unique=False)

    # Trigram index for note search
    op.execute('''
        CREATE INDEX idx_bookmarks_note_trgm
        ON bookmarks USING gin (note gin_trgm_ops)
    ''')


def downgrade() -> None:
    op.drop_index('idx_bookmarks_note_trgm', table_name='bookmarks')
    op.drop_index('idx_bookmarks_client_created', table_name='bookmarks')
    op.drop_index('idx_bookmarks_entity', table_name='bookmarks')
    op.drop_index('idx_bookmarks_client_id', table_name='bookmarks')
    op.drop_constraint('uq_bookmark_user_entity', 'bookmarks', type_='unique')
    op.drop_table('bookmarks')
