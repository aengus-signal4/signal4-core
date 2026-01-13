-- Migration 001e: Add content_id_string to speaker_transcriptions for direct access

BEGIN;

-- Add content_id_string field to speaker_transcriptions
ALTER TABLE speaker_transcriptions
ADD COLUMN content_id_string VARCHAR;

-- Populate it from the existing content_id FK
UPDATE speaker_transcriptions st
SET content_id_string = c.content_id
FROM content c
WHERE st.content_id = c.id;

-- Add NOT NULL constraint
ALTER TABLE speaker_transcriptions
ALTER COLUMN content_id_string SET NOT NULL;

-- Add index for performance
CREATE INDEX idx_speaker_transcriptions_content_string ON speaker_transcriptions(content_id_string);

-- Report results
DO $$
DECLARE
    updated_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO updated_count
    FROM speaker_transcriptions 
    WHERE content_id_string IS NOT NULL;
    
    RAISE NOTICE 'Added content_id_string to speaker_transcriptions:';
    RAISE NOTICE '  - % transcriptions updated with content_id_string', updated_count;
END $$;

COMMIT;