-- Migration 001: Add speaker_hash fields to both tables
-- This is Phase 1 - adding new fields without breaking existing functionality

BEGIN;

-- Add speaker_hash to speakers table
ALTER TABLE speakers 
ADD COLUMN speaker_hash CHAR(8);

-- Add speaker_hash and diarization_speaker_id to speaker_transcriptions table
ALTER TABLE speaker_transcriptions 
ADD COLUMN speaker_hash CHAR(8),
ADD COLUMN diarization_speaker_id VARCHAR(20);

-- Create function to generate deterministic speaker hash
CREATE OR REPLACE FUNCTION generate_speaker_hash(content_id_param TEXT, diarization_speaker_param TEXT)
RETURNS CHAR(8) AS $$
BEGIN
    RETURN LEFT(MD5(content_id_param || '|' || diarization_speaker_param), 8);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Populate speaker_hash in speakers table
UPDATE speakers 
SET speaker_hash = generate_speaker_hash(content_id, local_speaker_id);

-- Populate both new fields in speaker_transcriptions table
-- This requires joining with content and speakers tables to get the source data
UPDATE speaker_transcriptions st
SET 
    speaker_hash = s.speaker_hash,
    diarization_speaker_id = s.local_speaker_id
FROM speakers s, content c
WHERE st.content_id = c.id 
  AND st.speaker_id = s.id
  AND s.content_id = c.content_id;

-- Add NOT NULL constraints after populating data
ALTER TABLE speakers 
ALTER COLUMN speaker_hash SET NOT NULL;

ALTER TABLE speaker_transcriptions 
ALTER COLUMN speaker_hash SET NOT NULL,
ALTER COLUMN diarization_speaker_id SET NOT NULL;

-- Add indexes for performance
CREATE INDEX idx_speakers_hash ON speakers(speaker_hash);
CREATE INDEX idx_speakers_content_diarization ON speakers(content_id, local_speaker_id);
CREATE INDEX idx_speaker_transcriptions_hash ON speaker_transcriptions(speaker_hash);
CREATE INDEX idx_speaker_transcriptions_content_diarization ON speaker_transcriptions(content_id, diarization_speaker_id);

-- Add unique constraint on the new natural key for speakers
ALTER TABLE speakers 
ADD CONSTRAINT uq_speakers_content_diarization_new UNIQUE (content_id, local_speaker_id);

-- Verify data integrity
DO $$
DECLARE
    speaker_hash_count INTEGER;
    transcription_hash_count INTEGER;
    speaker_null_count INTEGER;
    transcription_null_count INTEGER;
BEGIN
    -- Check that all speakers have speaker_hash
    SELECT COUNT(*) INTO speaker_null_count 
    FROM speakers 
    WHERE speaker_hash IS NULL;
    
    IF speaker_null_count > 0 THEN
        RAISE EXCEPTION 'Found % speakers with NULL speaker_hash', speaker_null_count;
    END IF;
    
    -- Check that all transcriptions have speaker_hash and diarization_speaker_id
    SELECT COUNT(*) INTO transcription_null_count 
    FROM speaker_transcriptions 
    WHERE speaker_hash IS NULL OR diarization_speaker_id IS NULL;
    
    IF transcription_null_count > 0 THEN
        RAISE EXCEPTION 'Found % transcriptions with NULL hash or diarization_speaker_id', transcription_null_count;
    END IF;
    
    -- Verify hash consistency between tables
    SELECT COUNT(*) INTO speaker_hash_count FROM speakers;
    SELECT COUNT(DISTINCT speaker_hash) INTO transcription_hash_count FROM speaker_transcriptions;
    
    RAISE NOTICE 'Migration completed successfully:';
    RAISE NOTICE '  - % speakers with speaker_hash', speaker_hash_count;
    RAISE NOTICE '  - % unique speaker hashes in transcriptions', transcription_hash_count;
    RAISE NOTICE '  - No NULL values found';
END $$;

COMMIT;