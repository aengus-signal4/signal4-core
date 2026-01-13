"""clean content ids

Revision ID: clean_content_ids
Revises: ad3d068a3baa
Create Date: 2024-02-04 09:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision = 'clean_content_ids'
down_revision = 'ad3d068a3baa'
branch_labels = None
depends_on = None

def upgrade():
    # Create a temporary table to store old and new content IDs
    op.execute("""
        CREATE TEMPORARY TABLE content_id_map AS
        SELECT id, content_id as old_content_id,
               CASE 
                   WHEN position(':' in content_id) > 0 
                   THEN substring(content_id from position(':' in content_id) + 1)
                   ELSE content_id 
               END as new_content_id
        FROM content
        WHERE content_id LIKE '%:%';
    """)
    
    # Update content IDs in the content table
    op.execute("""
        UPDATE content
        SET content_id = CASE 
            WHEN position(':' in content_id) > 0 
            THEN substring(content_id from position(':' in content_id) + 1)
            ELSE content_id 
        END
        WHERE content_id LIKE '%:%';
    """)
    
    # Log the changes for verification
    op.execute("""
        INSERT INTO alembic_version_history (version_num, operation, details)
        SELECT 
            'clean_content_ids',
            'update_content_ids',
            json_build_object(
                'old_id', old_content_id,
                'new_id', new_content_id
            )::text
        FROM content_id_map;
    """)
    
    # Drop temporary table
    op.execute("DROP TABLE content_id_map;")

def downgrade():
    # Note: Since we're removing information, we can't reliably restore the original channel IDs
    # The best we can do is note that this is irreversible
    op.execute("""
        INSERT INTO alembic_version_history (version_num, operation, details)
        VALUES (
            'clean_content_ids',
            'downgrade_skipped',
            'Content ID cleanup cannot be reversed as channel IDs were removed'
        );
    """) 