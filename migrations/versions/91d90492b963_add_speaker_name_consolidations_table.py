"""add_hosts_consolidated_flag_to_content

Revision ID: 91d90492b963
Revises: f6b552dca8bd
Create Date: 2024-11-30

Adds a flag to track whether content.hosts has been consolidated (name variations resolved).
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '91d90492b963'
down_revision = 'f6b552dca8bd'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('content', sa.Column('hosts_consolidated', sa.Boolean(), nullable=True, server_default='false'))
    op.create_index('idx_content_hosts_consolidated', 'content', ['hosts_consolidated'])


def downgrade() -> None:
    op.drop_index('idx_content_hosts_consolidated', 'content')
    op.drop_column('content', 'hosts_consolidated')
