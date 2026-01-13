-- Migration: Enhance SpeakerEmbedding to be the unified Speaker table
-- Date: 2025-01-11
-- Description: Add missing fields to speaker_embeddings and rename to speakers

BEGIN;

-- Step 1: Add the missing fields to speaker_embeddings table
ALTER TABLE speaker_embeddings 
ADD COLUMN global_id VARCHAR(255),
ADD COLUMN universal_name VARCHAR(255),
ADD COLUMN display_name VARCHAR(255),
ADD COLUMN canonical_speaker_id INTEGER,
ADD COLUMN is_canonical BOOLEAN DEFAULT TRUE NOT NULL,
ADD COLUMN merge_history JSON DEFAULT '[]',
ADD COLUMN merge_confidence FLOAT,
ADD COLUMN meta_data JSON DEFAULT '{}',
ADD COLUMN notes TEXT,
ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Step 2: Generate global_id and universal_name for all records
UPDATE speaker_embeddings 
SET 
    global_id = 'SPEAKER_' || substr(md5(content_id || '_' || local_speaker_id), 1, 8),
    universal_name = 'Speaker ' || id;

-- Step 3: Make the new fields NOT NULL now that they have values
ALTER TABLE speaker_embeddings 
ALTER COLUMN global_id SET NOT NULL,
ALTER COLUMN universal_name SET NOT NULL;

-- Step 4: Create indexes for the new fields
CREATE INDEX idx_speaker_embeddings_global_id ON speaker_embeddings(global_id);
CREATE INDEX idx_speaker_embeddings_universal_name ON speaker_embeddings(universal_name);
CREATE INDEX idx_speaker_embeddings_canonical ON speaker_embeddings(canonical_speaker_id);

-- Step 5: Add self-referential foreign key constraint for canonical relationships
ALTER TABLE speaker_embeddings 
ADD CONSTRAINT fk_speaker_embeddings_canonical 
FOREIGN KEY (canonical_speaker_id) REFERENCES speaker_embeddings(id);

-- Step 6: Update speaker_transcriptions to use the new speaker IDs
-- Create a mapping from old speaker IDs to speaker_embeddings IDs
CREATE TEMPORARY TABLE speaker_mapping AS
SELECT DISTINCT
    st.speaker_id as old_speaker_id,
    se.id as new_speaker_id
FROM speaker_transcriptions st
JOIN speaker_embeddings se ON true
WHERE se.content_id = (SELECT content_id FROM content WHERE id = st.content_id)
  AND se.local_speaker_id = 'SPEAKER_' || lpad(st.speaker_id::text, 2, '0')
LIMIT 1000000; -- Ensure we get a reasonable number

-- Update the speaker_transcriptions table
UPDATE speaker_transcriptions st
SET speaker_id = sm.new_speaker_id
FROM speaker_mapping sm
WHERE st.speaker_id = sm.old_speaker_id;

-- Step 7: Drop the old speakers table
DROP TABLE IF EXISTS speakers CASCADE;

-- Step 8: Rename speaker_embeddings to speakers
ALTER TABLE speaker_embeddings RENAME TO speakers;

-- Step 9: Rename the sequence
ALTER SEQUENCE speaker_embeddings_id_seq RENAME TO speakers_id_seq;

-- Step 10: Update the column default to use the new sequence name
ALTER TABLE speakers ALTER COLUMN id SET DEFAULT nextval('speakers_id_seq');

-- Step 11: Update foreign key constraint name in speaker_transcriptions
ALTER TABLE speaker_transcriptions 
DROP CONSTRAINT IF EXISTS speaker_transcriptions_speaker_id_fkey,
ADD CONSTRAINT fk_speaker_transcriptions_speaker_id 
FOREIGN KEY (speaker_id) REFERENCES speakers(id);

-- Step 12: Create speaker_number_seq if it doesn't exist for future use
CREATE SEQUENCE IF NOT EXISTS speaker_number_seq START WITH 1000000;

COMMIT;

-- Verification queries:
SELECT 'speakers' as table_name, COUNT(*) as count FROM speakers
UNION ALL
SELECT 'speaker_transcriptions', COUNT(*) FROM speaker_transcriptions;

-- Check the new constraint enforcement (should return 0 duplicates)
SELECT 
    content_id, 
    local_speaker_id, 
    COUNT(*) as count 
FROM speakers 
GROUP BY content_id, local_speaker_id 
HAVING COUNT(*) > 1;

-- Check that all speaker_transcriptions have valid speaker references
SELECT COUNT(*) as orphaned_transcriptions
FROM speaker_transcriptions st
LEFT JOIN speakers s ON st.speaker_id = s.id
WHERE s.id IS NULL;