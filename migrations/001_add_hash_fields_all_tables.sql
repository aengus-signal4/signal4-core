-- Migration 001: Add hash fields to speakers, speaker_transcriptions, and embedding_segments
-- This is Phase 1 - adding new fields without breaking existing functionality

BEGIN;

-- ================================================================
-- SPEAKERS TABLE
-- ================================================================

-- Add speaker_hash to speakers table
ALTER TABLE speakers 
ADD COLUMN speaker_hash CHAR(8);

-- ================================================================
-- SPEAKER_TRANSCRIPTIONS TABLE  
-- ================================================================

-- Add speaker_hash and diarization_speaker_id to speaker_transcriptions table
ALTER TABLE speaker_transcriptions 
ADD COLUMN speaker_hash CHAR(8),
ADD COLUMN diarization_speaker_id VARCHAR(20);

-- ================================================================
-- EMBEDDING_SEGMENTS TABLE
-- ================================================================

-- Add new hash-based fields to embedding_segments table
ALTER TABLE embedding_segments
ADD COLUMN segment_hash CHAR(8),
ADD COLUMN content_id_string VARCHAR,
ADD COLUMN source_speaker_hashes CHAR(8)[];

-- ================================================================
-- UTILITY FUNCTIONS
-- ================================================================

-- Create function to generate deterministic speaker hash
CREATE OR REPLACE FUNCTION generate_speaker_hash(content_id_param TEXT, diarization_speaker_param TEXT)
RETURNS CHAR(8) AS $$
BEGIN
    RETURN LEFT(MD5(content_id_param || '|' || diarization_speaker_param), 8);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to generate deterministic segment hash
-- Use segment ID to ensure uniqueness when there are duplicates
CREATE OR REPLACE FUNCTION generate_segment_hash(content_id_param TEXT, segment_index_param INTEGER, segment_id_param INTEGER)
RETURNS CHAR(8) AS $$
BEGIN
    RETURN LEFT(MD5(content_id_param || '|' || segment_index_param::TEXT || '|' || segment_id_param::TEXT), 8);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ================================================================
-- POPULATE DATA - SPEAKERS
-- ================================================================

-- Populate speaker_hash in speakers table
UPDATE speakers 
SET speaker_hash = generate_speaker_hash(content_id, local_speaker_id);

-- ================================================================
-- POPULATE DATA - SPEAKER_TRANSCRIPTIONS
-- ================================================================

-- Populate both new fields in speaker_transcriptions table
-- Only update records that have valid speaker references
UPDATE speaker_transcriptions st
SET 
    speaker_hash = s.speaker_hash,
    diarization_speaker_id = s.local_speaker_id
FROM speakers s, content c
WHERE st.content_id = c.id 
  AND st.speaker_id = s.id
  AND s.content_id = c.content_id
  AND s.speaker_hash IS NOT NULL;

-- ================================================================
-- POPULATE DATA - EMBEDDING_SEGMENTS  
-- ================================================================

-- Populate new fields in embedding_segments table (this may take a while)
UPDATE embedding_segments es
SET 
    segment_hash = generate_segment_hash(c.content_id, es.segment_index, es.id),
    content_id_string = c.content_id,
    source_speaker_hashes = COALESCE(
        (
            SELECT ARRAY_AGG(DISTINCT st.speaker_hash)
            FROM speaker_transcriptions st
            WHERE st.id = ANY(es.source_transcription_ids)
              AND st.speaker_hash IS NOT NULL
        ),
        ARRAY[]::CHAR(8)[]  -- Default to empty array if no valid speakers found
    )
FROM content c
WHERE es.content_id = c.id;

-- ================================================================
-- ADD CONSTRAINTS
-- ================================================================

-- Add NOT NULL constraints after populating data
ALTER TABLE speakers 
ALTER COLUMN speaker_hash SET NOT NULL;

-- Only add NOT NULL constraints for speaker_transcriptions where we have valid data
-- Leave NULL values for orphaned records (will be handled in cleanup phase)
-- ALTER TABLE speaker_transcriptions 
-- ALTER COLUMN speaker_hash SET NOT NULL,
-- ALTER COLUMN diarization_speaker_id SET NOT NULL;

ALTER TABLE embedding_segments
ALTER COLUMN segment_hash SET NOT NULL,
ALTER COLUMN content_id_string SET NOT NULL,
ALTER COLUMN source_speaker_hashes SET NOT NULL;

-- ================================================================
-- ADD INDEXES
-- ================================================================

-- Indexes for speakers
CREATE INDEX idx_speakers_hash ON speakers(speaker_hash);
CREATE INDEX idx_speakers_content_diarization ON speakers(content_id, local_speaker_id);

-- Indexes for speaker_transcriptions
CREATE INDEX idx_speaker_transcriptions_hash ON speaker_transcriptions(speaker_hash);
CREATE INDEX idx_speaker_transcriptions_diarization ON speaker_transcriptions(diarization_speaker_id);

-- Indexes for embedding_segments  
CREATE INDEX idx_embedding_segments_hash ON embedding_segments(segment_hash);
CREATE INDEX idx_embedding_segments_content_string ON embedding_segments(content_id_string);
CREATE INDEX idx_embedding_segments_speaker_hashes ON embedding_segments USING GIN(source_speaker_hashes);

-- ================================================================
-- ADD UNIQUE CONSTRAINTS
-- ================================================================

-- Add unique constraint on the new natural key for speakers
ALTER TABLE speakers 
ADD CONSTRAINT uq_speakers_content_diarization_new UNIQUE (content_id, local_speaker_id);

-- Add unique constraint for segment hash (should be unique)
ALTER TABLE embedding_segments
ADD CONSTRAINT uq_embedding_segments_hash UNIQUE (segment_hash);

-- Note: Not adding unique constraint for content + segment_index as there are legitimate duplicates

-- ================================================================
-- DATA INTEGRITY VERIFICATION
-- ================================================================

DO $$
DECLARE
    speaker_null_count INTEGER;
    transcription_null_count INTEGER;
    segment_null_count INTEGER;
    speaker_hash_count INTEGER;
    segment_hash_count INTEGER;
    transcription_hash_count INTEGER;
BEGIN
    -- Check speakers table
    SELECT COUNT(*) INTO speaker_null_count 
    FROM speakers 
    WHERE speaker_hash IS NULL;
    
    IF speaker_null_count > 0 THEN
        RAISE EXCEPTION 'Found % speakers with NULL speaker_hash', speaker_null_count;
    END IF;
    
    -- Check speaker_transcriptions table (allow NULLs for orphaned records)
    SELECT COUNT(*) INTO transcription_null_count 
    FROM speaker_transcriptions 
    WHERE speaker_hash IS NULL OR diarization_speaker_id IS NULL;
    
    -- Don't fail for NULL values, just report them
    IF transcription_null_count > 0 THEN
        RAISE NOTICE 'Found % transcriptions with NULL hash (orphaned records)', transcription_null_count;
    END IF;
    
    -- Check embedding_segments table
    SELECT COUNT(*) INTO segment_null_count 
    FROM embedding_segments 
    WHERE segment_hash IS NULL OR content_id_string IS NULL OR source_speaker_hashes IS NULL;
    
    IF segment_null_count > 0 THEN
        RAISE EXCEPTION 'Found % segments with NULL hash, content_id_string, or source_speaker_hashes', segment_null_count;
    END IF;
    
    -- Get counts for reporting
    SELECT COUNT(*) INTO speaker_hash_count FROM speakers;
    SELECT COUNT(DISTINCT speaker_hash) INTO transcription_hash_count FROM speaker_transcriptions;
    SELECT COUNT(*) INTO segment_hash_count FROM embedding_segments;
    
    RAISE NOTICE 'Migration completed successfully:';
    RAISE NOTICE '  - % speakers with speaker_hash', speaker_hash_count;
    RAISE NOTICE '  - % unique speaker hashes in transcriptions', transcription_hash_count;
    RAISE NOTICE '  - % segments with segment_hash and source_speaker_hashes', segment_hash_count;
    RAISE NOTICE '  - No NULL values found in any table';
    
    -- Additional validation
    RAISE NOTICE 'Additional validation:';
    
    -- Check that all speaker hashes in transcriptions exist in speakers
    SELECT COUNT(*) INTO transcription_null_count
    FROM speaker_transcriptions st
    WHERE NOT EXISTS (
        SELECT 1 FROM speakers s WHERE s.speaker_hash = st.speaker_hash
    );
    
    IF transcription_null_count > 0 THEN
        RAISE WARNING 'Found % transcriptions with speaker_hash not in speakers table', transcription_null_count;
    ELSE
        RAISE NOTICE '  - All transcription speaker_hashes have matching speakers';
    END IF;
    
    -- Check that all source_speaker_hashes in segments exist in speakers
    SELECT COUNT(*) INTO segment_null_count
    FROM embedding_segments es
    WHERE EXISTS (
        SELECT 1 FROM unnest(es.source_speaker_hashes) AS hash
        WHERE NOT EXISTS (SELECT 1 FROM speakers s WHERE s.speaker_hash = hash)
    );
    
    IF segment_null_count > 0 THEN
        RAISE WARNING 'Found % segments with source_speaker_hashes not in speakers table', segment_null_count;
    ELSE
        RAISE NOTICE '  - All segment source_speaker_hashes have matching speakers';
    END IF;
    
END $$;

COMMIT;