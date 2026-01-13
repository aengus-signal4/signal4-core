-- Migration 001c: Populate speaker_transcriptions hash fields

BEGIN;

-- Populate both new fields in speaker_transcriptions table
-- Only update records that have valid speaker references (speaker_id != -1)
UPDATE speaker_transcriptions st
SET 
    speaker_hash = s.speaker_hash,
    diarization_speaker_id = s.local_speaker_id
FROM speakers s, content c
WHERE st.content_id = c.id 
  AND st.speaker_id = s.id
  AND st.speaker_id != -1
  AND s.content_id = c.content_id
  AND s.speaker_hash IS NOT NULL;

-- Add indexes (but no NOT NULL constraints since we have unprocessed content)
CREATE INDEX idx_speaker_transcriptions_hash ON speaker_transcriptions(speaker_hash);
CREATE INDEX idx_speaker_transcriptions_diarization ON speaker_transcriptions(diarization_speaker_id);

-- Report on what was updated
DO $$
DECLARE
    updated_count INTEGER;
    null_count INTEGER;
    total_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO updated_count
    FROM speaker_transcriptions 
    WHERE speaker_hash IS NOT NULL;
    
    SELECT COUNT(*) INTO null_count
    FROM speaker_transcriptions 
    WHERE speaker_hash IS NULL;
    
    SELECT COUNT(*) INTO total_count
    FROM speaker_transcriptions;
    
    RAISE NOTICE 'Speaker transcriptions update completed:';
    RAISE NOTICE '  - % transcriptions updated with speaker_hash', updated_count;
    RAISE NOTICE '  - % transcriptions still need processing (speaker_id = -1)', null_count;
    RAISE NOTICE '  - % total transcriptions', total_count;
END $$;

COMMIT;