"""platform to source

Revision ID: platform_to_source
Revises: project_to_projects
Create Date: 2024-02-04 06:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'platform_to_source'
down_revision = 'project_to_projects'
branch_labels = None
depends_on = None

def upgrade():
    # Rename platform column to source
    op.alter_column('content', 'platform', new_column_name='source')
    
    # Update indexes
    op.drop_index('idx_content_platform')
    op.create_index('idx_content_source', 'content', ['source'])

def downgrade():
    # Rename source column back to platform
    op.alter_column('content', 'source', new_column_name='platform')
    
    # Update indexes
    op.drop_index('idx_content_source')
    op.create_index('idx_content_platform', 'content', ['platform']) 