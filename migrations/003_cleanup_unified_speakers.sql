-- Migration: Cleanup unified speakers table structure
-- Date: 2025-01-11
-- Description: Remove old speaker_id foreign key and fix column names

BEGIN;

-- Step 1: Drop the old speaker_id column (was foreign key to old speakers table)
ALTER TABLE speakers DROP COLUMN IF EXISTS speaker_id CASCADE;

-- Step 2: Rename quality_score to embedding_quality_score for consistency
ALTER TABLE speakers RENAME COLUMN quality_score TO embedding_quality_score;

-- Step 3: Fix the unique constraint name
ALTER TABLE speakers DROP CONSTRAINT IF EXISTS uq_speaker_embedding_content;
ALTER TABLE speakers ADD CONSTRAINT uq_speakers_content_local UNIQUE (content_id, local_speaker_id);

-- Step 4: Update any null global_id values
UPDATE speakers 
SET global_id = 'SPEAKER_' || substr(md5(content_id::text || '_' || local_speaker_id), 1, 8)
WHERE global_id IS NULL;

-- Step 5: Make global_id NOT NULL now that all values are set
ALTER TABLE speakers ALTER COLUMN global_id SET NOT NULL;

-- Step 6: Create sequence for speaker names if it doesn't exist
CREATE SEQUENCE IF NOT EXISTS speaker_number_seq START WITH 1000000;

-- Step 7: Rename all indexes to match new table name
ALTER INDEX IF EXISTS idx_speaker_embeddings_canonical RENAME TO idx_speakers_canonical;
ALTER INDEX IF EXISTS idx_speaker_embeddings_global_id RENAME TO idx_speakers_global_id;
ALTER INDEX IF EXISTS idx_speaker_embeddings_universal_name RENAME TO idx_speakers_universal_name;
ALTER INDEX IF EXISTS idx_speaker_embeddings_content_id RENAME TO idx_speakers_content_id;
ALTER INDEX IF EXISTS idx_speaker_embeddings_created_at RENAME TO idx_speakers_created_at;
ALTER INDEX IF EXISTS idx_speaker_embeddings_duration RENAME TO idx_speakers_duration;
ALTER INDEX IF EXISTS idx_speaker_embeddings_quality RENAME TO idx_speakers_quality;
ALTER INDEX IF EXISTS idx_speaker_embeddings_rebase_batch RENAME TO idx_speakers_rebase_batch;
ALTER INDEX IF EXISTS idx_speaker_embeddings_rebase_pending RENAME TO idx_speakers_rebase_pending;
ALTER INDEX IF EXISTS idx_speaker_embeddings_rebase_status RENAME TO idx_speakers_rebase_status;
ALTER INDEX IF EXISTS idx_speaker_embeddings_speaker_quality RENAME TO idx_speakers_speaker_quality;
ALTER INDEX IF EXISTS idx_speaker_embeddings_temp_speaker RENAME TO idx_speakers_temp_speaker;
ALTER INDEX IF EXISTS idx_speaker_embeddings_embedding_hnsw RENAME TO idx_speakers_embedding_hnsw;
ALTER INDEX IF EXISTS idx_speaker_embeddings_clustering_query RENAME TO idx_speakers_clustering_query;

-- Step 8: Rename primary key constraint
ALTER INDEX IF EXISTS speaker_embeddings_pkey RENAME TO speakers_pkey;

-- Step 9: Update foreign key constraint name
ALTER TABLE speakers DROP CONSTRAINT IF EXISTS fk_speaker_embeddings_canonical;
ALTER TABLE speakers ADD CONSTRAINT fk_speakers_canonical 
FOREIGN KEY (canonical_speaker_id) REFERENCES speakers(id) 
DEFERRABLE INITIALLY DEFERRED;

COMMIT;

-- Verification
SELECT 'Table structure after cleanup:' as info;
\d speakers

-- Check for duplicates (should be 0)
SELECT content_id, local_speaker_id, COUNT(*) as count 
FROM speakers 
GROUP BY content_id, local_speaker_id 
HAVING COUNT(*) > 1;

-- Sample data
SELECT id, content_id, local_speaker_id, global_id, universal_name, 
       embedding IS NOT NULL as has_embedding, embedding_quality_score
FROM speakers 
ORDER BY id 
LIMIT 5;