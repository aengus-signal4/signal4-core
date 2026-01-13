-- Migration: Enhance SpeakerEmbedding to be the unified Speaker table
-- Date: 2025-01-11
-- Description: Add missing fields to speaker_embeddings to make it the unified speaker table

BEGIN;

-- Step 1: Add the missing fields to speaker_embeddings table
ALTER TABLE speaker_embeddings 
ADD COLUMN IF NOT EXISTS global_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS universal_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS display_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS canonical_speaker_id INTEGER,
ADD COLUMN IF NOT EXISTS is_canonical BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS merge_history JSON DEFAULT '[]',
ADD COLUMN IF NOT EXISTS merge_confidence FLOAT,
ADD COLUMN IF NOT EXISTS meta_data JSON DEFAULT '{}',
ADD COLUMN IF NOT EXISTS notes TEXT,
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Step 2: Generate global_id and universal_name for all records that don't have them
UPDATE speaker_embeddings 
SET 
    global_id = 'SPEAKER_' || substr(md5(content_id || '_' || local_speaker_id), 1, 8),
    universal_name = 'Speaker ' || id
WHERE global_id IS NULL;

-- Step 3: Create indexes for the new fields
CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_global_id ON speaker_embeddings(global_id);
CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_universal_name ON speaker_embeddings(universal_name);
CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_canonical ON speaker_embeddings(canonical_speaker_id);

-- Step 4: Add self-referential foreign key constraint for canonical relationships
ALTER TABLE speaker_embeddings 
ADD CONSTRAINT fk_speaker_embeddings_canonical 
FOREIGN KEY (canonical_speaker_id) REFERENCES speaker_embeddings(id) DEFERRABLE INITIALLY DEFERRED;

-- Step 5: Drop the old speakers table (we're keeping speaker_embeddings as the new speakers table)
DROP TABLE IF EXISTS speakers CASCADE;

-- Step 6: Rename speaker_embeddings to speakers
ALTER TABLE speaker_embeddings RENAME TO speakers;

-- Step 7: Rename the sequence
ALTER SEQUENCE speaker_embeddings_id_seq RENAME TO speakers_id_seq;

-- Step 8: Update the column default to use the new sequence name
ALTER TABLE speakers ALTER COLUMN id SET DEFAULT nextval('speakers_id_seq');

-- Step 9: Create speaker_number_seq for create_with_sequential_name method
CREATE SEQUENCE IF NOT EXISTS speaker_number_seq START WITH 1000000;

-- Step 10: Update any references in other tables will be handled by the application
-- The speaker_transcriptions table will need to be updated by the application logic

COMMIT;

-- Verification queries:
SELECT 'speakers' as table_name, COUNT(*) as count FROM speakers;

-- Check the constraint enforcement (should return 0 duplicates)
SELECT 
    content_id, 
    local_speaker_id, 
    COUNT(*) as count 
FROM speakers 
GROUP BY content_id, local_speaker_id 
HAVING COUNT(*) > 1;

-- Show sample of the new unified table
SELECT id, content_id, local_speaker_id, global_id, universal_name, 
       embedding_quality_score, rebase_status 
FROM speakers 
ORDER BY id 
LIMIT 5;