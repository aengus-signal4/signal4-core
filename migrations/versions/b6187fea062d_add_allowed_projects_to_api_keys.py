"""add allowed_projects to api_keys

Revision ID: b6187fea062d
Revises: d16ba030a3a4
Create Date: 2026-01-19 11:50:11.642753

Adds the allowed_projects JSONB column to api_keys table for project-level RBAC.
Keys with null or empty allowed_projects have access to all projects.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b6187fea062d'
down_revision: Union[str, None] = 'd16ba030a3a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add allowed_projects column to api_keys table
    # Nullable JSONB array - null or empty means full access to all projects
    op.add_column(
        'api_keys',
        sa.Column('allowed_projects', postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )


def downgrade() -> None:
    # Remove allowed_projects column
    op.drop_column('api_keys', 'allowed_projects')
