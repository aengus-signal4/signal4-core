"""Upgrade API key hash column for bcrypt support.

Revision ID: upgrade_api_key_hash_to_bcrypt
Revises: add_partial_hnsw_indexes
Create Date: 2026-01-19

This migration:
1. Increases key_hash column from 64 to 128 characters to support bcrypt hashes
   (bcrypt hashes are 60 chars, SHA-256 are 64 chars)
2. Adds index on key_prefix for efficient bcrypt key lookup
   (bcrypt requires prefix-based lookup then verification, unlike SHA-256 direct lookup)

Note: Existing SHA-256 hashed keys will continue to work. New keys will use bcrypt.
Keys can be migrated to bcrypt by regenerating them.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'upgrade_api_key_hash_to_bcrypt'
down_revision = 'add_partial_hnsw_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Increase key_hash column size to support bcrypt hashes
    op.alter_column(
        'api_keys',
        'key_hash',
        existing_type=sa.String(length=64),
        type_=sa.String(length=128),
        existing_nullable=False
    )

    # Add index on key_prefix for efficient bcrypt key lookup
    # bcrypt hashes have unique salts, so we can't do direct hash lookup
    # Instead we lookup by prefix, then verify with bcrypt
    op.create_index(
        'ix_api_keys_key_prefix',
        'api_keys',
        ['key_prefix'],
        unique=False
    )


def downgrade() -> None:
    # Remove key_prefix index
    op.drop_index('ix_api_keys_key_prefix', table_name='api_keys')

    # Revert key_hash column size
    # WARNING: This will truncate any bcrypt hashes (60 chars) but they fit in 64
    # However, this is a breaking change if there are bcrypt hashes stored
    op.alter_column(
        'api_keys',
        'key_hash',
        existing_type=sa.String(length=128),
        type_=sa.String(length=64),
        existing_nullable=False
    )
