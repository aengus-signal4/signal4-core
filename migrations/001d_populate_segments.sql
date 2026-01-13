-- Migration 001d: Populate embedding_segments hash fields

BEGIN;

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

-- Add NOT NULL constraints
ALTER TABLE embedding_segments
ALTER COLUMN segment_hash SET NOT NULL,
ALTER COLUMN content_id_string SET NOT NULL,
ALTER COLUMN source_speaker_hashes SET NOT NULL;

-- Add indexes and constraints
CREATE INDEX idx_embedding_segments_hash ON embedding_segments(segment_hash);
CREATE INDEX idx_embedding_segments_content_string ON embedding_segments(content_id_string);
CREATE INDEX idx_embedding_segments_speaker_hashes ON embedding_segments USING GIN(source_speaker_hashes);

ALTER TABLE embedding_segments
ADD CONSTRAINT uq_embedding_segments_hash UNIQUE (segment_hash);

-- Report on what was updated
DO $$
DECLARE
    total_segments INTEGER;
    segments_with_speakers INTEGER;
    segments_without_speakers INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_segments FROM embedding_segments;
    
    SELECT COUNT(*) INTO segments_with_speakers
    FROM embedding_segments 
    WHERE array_length(source_speaker_hashes, 1) > 0;
    
    SELECT COUNT(*) INTO segments_without_speakers  
    FROM embedding_segments
    WHERE array_length(source_speaker_hashes, 1) IS NULL OR array_length(source_speaker_hashes, 1) = 0;
    
    RAISE NOTICE 'Embedding segments update completed:';
    RAISE NOTICE '  - % total segments processed', total_segments;
    RAISE NOTICE '  - % segments with speaker hashes', segments_with_speakers;
    RAISE NOTICE '  - % segments without speaker hashes (unprocessed content)', segments_without_speakers;
END $$;

COMMIT;