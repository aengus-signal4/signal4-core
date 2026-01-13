-- Migration 001a: Just add the new columns (no data population)

BEGIN;

-- Add speaker_hash to speakers table
ALTER TABLE speakers 
ADD COLUMN speaker_hash CHAR(8);

-- Add speaker_hash and diarization_speaker_id to speaker_transcriptions table
ALTER TABLE speaker_transcriptions 
ADD COLUMN speaker_hash CHAR(8),
ADD COLUMN diarization_speaker_id VARCHAR(20);

-- Add new hash-based fields to embedding_segments table
ALTER TABLE embedding_segments
ADD COLUMN segment_hash CHAR(8),
ADD COLUMN content_id_string VARCHAR,
ADD COLUMN source_speaker_hashes CHAR(8)[];

-- Create utility functions
CREATE OR REPLACE FUNCTION generate_speaker_hash(content_id_param TEXT, diarization_speaker_param TEXT)
RETURNS CHAR(8) AS $$
BEGIN
    RETURN LEFT(MD5(content_id_param || '|' || diarization_speaker_param), 8);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION generate_segment_hash(content_id_param TEXT, segment_index_param INTEGER, segment_id_param INTEGER)
RETURNS CHAR(8) AS $$
BEGIN
    RETURN LEFT(MD5(content_id_param || '|' || segment_index_param::TEXT || '|' || segment_id_param::TEXT), 8);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMIT;