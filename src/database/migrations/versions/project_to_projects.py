"""Rename project column to projects

Revision ID: project_to_projects_merge
Revises: 0131bc93677a
Create Date: 2024-02-04 03:33:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'project_to_projects_merge'
down_revision = '0131bc93677a'  # This is the current head from the history
branch_labels = None
depends_on = None

def upgrade():
    # Rename column in content table
    op.alter_column('content', 'project', new_column_name='projects',
                    existing_type=sa.String())
    
    # Drop old index if it exists
    op.drop_index('idx_content_project', table_name='content')
    
    # Create new index
    op.create_index('idx_content_projects', 'content', ['projects'])

def downgrade():
    # Rename column back in content table
    op.alter_column('content', 'projects', new_column_name='project',
                    existing_type=sa.String())
    
    # Drop new index
    op.drop_index('idx_content_projects', table_name='content')
    
    # Recreate old index
    op.create_index('idx_content_project', 'content', ['project']) 