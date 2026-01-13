-- Migration: Consolidate Speaker and SpeakerEmbedding tables into unified Speaker table
-- Date: 2025-01-11
-- Version: 2.0 - Fixed for PostgreSQL compatibility

BEGIN;

-- Step 1: Create the new unified speakers table
CREATE TABLE speakers_new (
    id SERIAL PRIMARY KEY,
    
    -- Content-Speaker Identity (1:1 constraint)
    content_id VARCHAR(255) NOT NULL,
    local_speaker_id VARCHAR(50) NOT NULL,
    
    -- Speaker Identity
    global_id VARCHAR(255) NOT NULL,
    universal_name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    
    -- Embedding Data
    embedding vector(512) NOT NULL,
    embedding_quality_score FLOAT DEFAULT 1.0,
    algorithm_version VARCHAR(50) DEFAULT 'stage6b_tight_clusters',
    
    -- Content Statistics
    duration FLOAT DEFAULT 0.0,
    segment_count INTEGER DEFAULT 0,
    
    -- Canonical Management
    canonical_speaker_id INTEGER,
    is_canonical BOOLEAN DEFAULT TRUE NOT NULL,
    merge_history JSON DEFAULT '[]',
    merge_confidence FLOAT,
    
    -- Rebase Status
    rebase_status VARCHAR(50) DEFAULT 'PENDING' NOT NULL,
    rebase_batch_id VARCHAR(255),
    rebase_processed_at TIMESTAMP,
    temporary_speaker_id VARCHAR(255),
    
    -- Metadata
    meta_data JSON DEFAULT '{}',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE (content_id, local_speaker_id)
);

-- Step 2: Create indexes for the new table
CREATE INDEX idx_speaker_new_global_id ON speakers_new(global_id);
CREATE INDEX idx_speaker_new_universal_name ON speakers_new(universal_name);
CREATE INDEX idx_speaker_new_content_id ON speakers_new(content_id);
CREATE INDEX idx_speaker_new_local_speaker_id ON speakers_new(local_speaker_id);
CREATE INDEX idx_speaker_new_duration ON speakers_new(duration);
CREATE INDEX idx_speaker_new_quality ON speakers_new(embedding_quality_score);
CREATE INDEX idx_speaker_new_canonical ON speakers_new(canonical_speaker_id);
CREATE INDEX idx_speaker_new_rebase_status ON speakers_new(rebase_status);
CREATE INDEX idx_speaker_new_rebase_batch ON speakers_new(rebase_batch_id);
CREATE INDEX idx_speaker_new_temp_speaker ON speakers_new(temporary_speaker_id);
CREATE INDEX idx_speaker_new_quality_duration ON speakers_new(embedding_quality_score, duration);
CREATE INDEX idx_speaker_new_rebase_pending ON speakers_new(rebase_status, created_at);

-- Step 3: Create pgvector index for similarity search
CREATE INDEX idx_speaker_new_embedding ON speakers_new USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Step 4: Migrate data from SpeakerEmbedding table (primary source)
-- Each SpeakerEmbedding record becomes a unified Speaker record
INSERT INTO speakers_new (
    content_id,
    local_speaker_id,
    global_id,
    universal_name,
    display_name,
    embedding,
    embedding_quality_score,
    algorithm_version,
    duration,
    segment_count,
    canonical_speaker_id,
    is_canonical,
    merge_history,
    merge_confidence,
    rebase_status,
    rebase_batch_id,
    rebase_processed_at,
    temporary_speaker_id,
    meta_data,
    notes,
    created_at,
    updated_at
)
SELECT 
    se.content_id,
    se.local_speaker_id,
    COALESCE(s.global_id, 'SPEAKER_' || substr(md5(se.content_id || '_' || se.local_speaker_id), 1, 8)) as global_id,
    COALESCE(s.universal_name, 'Speaker ' || se.speaker_id) as universal_name,
    s.display_name,
    se.embedding,
    se.quality_score,
    se.algorithm_version,
    se.duration,
    se.segment_count,
    s.canonical_speaker_id,
    COALESCE(s.is_canonical, true),
    COALESCE(s.merge_history::json, '[]'::json),
    s.merge_confidence,
    se.rebase_status::text,
    se.rebase_batch_id,
    se.rebase_processed_at,
    se.temporary_speaker_id,
    COALESCE(s.meta_data::json, '{}'::json),
    s.notes,
    se.created_at,
    CURRENT_TIMESTAMP
FROM speaker_embeddings se
LEFT JOIN speakers s ON se.speaker_id = s.id
ORDER BY se.created_at;

-- Step 5: Create a mapping from old speaker IDs to new speaker IDs
CREATE TEMPORARY TABLE speaker_id_mapping AS
SELECT 
    se.speaker_id as old_speaker_id,
    sn.id as new_speaker_id,
    se.content_id,
    se.local_speaker_id
FROM speaker_embeddings se
JOIN speakers_new sn ON se.content_id = sn.content_id AND se.local_speaker_id = sn.local_speaker_id;

-- Step 6: Update speaker_transcriptions to reference the new speaker IDs
UPDATE speaker_transcriptions st
SET speaker_id = sim.new_speaker_id
FROM speaker_id_mapping sim
WHERE st.speaker_id = sim.old_speaker_id;

-- Step 7: Handle any SpeakerTranscription records that don't have corresponding embeddings
-- Get a sample embedding for placeholder records
DO $$
DECLARE
    sample_embedding vector(512);
BEGIN
    SELECT embedding INTO sample_embedding FROM speakers_new LIMIT 1;
    
    -- Insert placeholder records for speakers without embeddings
    INSERT INTO speakers_new (
        content_id,
        local_speaker_id,
        global_id,
        universal_name,
        embedding,
        embedding_quality_score,
        algorithm_version,
        duration,
        segment_count,
        rebase_status,
        meta_data,
        notes,
        created_at,
        updated_at
    )
    SELECT DISTINCT
        c.content_id as content_id,
        'SPEAKER_' || lpad(st.speaker_id::text, 2, '0') as local_speaker_id,
        'SPEAKER_' || substr(md5(c.content_id || '_' || st.speaker_id::text), 1, 8) as global_id,
        COALESCE(s.universal_name, 'Speaker ' || st.speaker_id) as universal_name,
        sample_embedding,
        0.0 as embedding_quality_score,
        'legacy_migration' as algorithm_version,
        0.0 as duration,
        0 as segment_count,
        'PENDING' as rebase_status,
        COALESCE(s.meta_data::json, '{}'::json) as meta_data,
        'Migrated from legacy speaker without embedding' as notes,
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    FROM speaker_transcriptions st
    LEFT JOIN content c ON st.content_id = c.id
    LEFT JOIN speakers s ON st.speaker_id = s.id
    LEFT JOIN speakers_new sn ON c.content_id = sn.content_id 
        AND 'SPEAKER_' || lpad(st.speaker_id::text, 2, '0') = sn.local_speaker_id
    WHERE sn.id IS NULL AND c.content_id IS NOT NULL;
END $$;

-- Step 8: Update any remaining speaker_transcriptions references
UPDATE speaker_transcriptions st
SET speaker_id = sn.id
FROM speakers_new sn, content c
WHERE st.content_id = c.id
    AND c.content_id = sn.content_id 
    AND 'SPEAKER_' || lpad(st.speaker_id::text, 2, '0') = sn.local_speaker_id
    AND st.speaker_id NOT IN (SELECT new_speaker_id FROM speaker_id_mapping);

-- Step 9: Add self-referential foreign key constraint
ALTER TABLE speakers_new ADD CONSTRAINT fk_speakers_new_canonical 
    FOREIGN KEY (canonical_speaker_id) REFERENCES speakers_new(id);

-- Step 10: Drop old tables and rename new table
DROP TABLE IF EXISTS speaker_embeddings CASCADE;
DROP TABLE IF EXISTS speakers CASCADE;
ALTER TABLE speakers_new RENAME TO speakers;

-- Step 11: Update sequence to match new table
-- Reset the sequence to continue from current max
DO $$
DECLARE
    max_speaker_num INTEGER;
BEGIN
    SELECT COALESCE(MAX(CAST(SUBSTRING(universal_name FROM 'Speaker (\d+)') AS INTEGER)), 0) + 1
    INTO max_speaker_num
    FROM speakers
    WHERE universal_name ~ 'Speaker \d+';
    
    PERFORM setval('speaker_number_seq', max_speaker_num);
END $$;

-- Step 12: Update foreign key constraints
ALTER TABLE speaker_transcriptions 
ADD CONSTRAINT fk_speaker_transcriptions_speaker_id 
FOREIGN KEY (speaker_id) REFERENCES speakers(id);

COMMIT;

-- Verification queries:
SELECT 'speakers' as table_name, COUNT(*) as count FROM speakers
UNION ALL
SELECT 'speaker_transcriptions', COUNT(*) FROM speaker_transcriptions;

-- Check constraint enforcement
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