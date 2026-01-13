-- Migration 001b: Populate speaker_hash in speakers table

BEGIN;

UPDATE speakers 
SET speaker_hash = generate_speaker_hash(content_id, local_speaker_id);

-- Add NOT NULL constraint
ALTER TABLE speakers 
ALTER COLUMN speaker_hash SET NOT NULL;

-- Add unique constraint
ALTER TABLE speakers 
ADD CONSTRAINT uq_speakers_content_diarization_new UNIQUE (content_id, local_speaker_id);

-- Add indexes
CREATE INDEX idx_speakers_hash ON speakers(speaker_hash);
CREATE INDEX idx_speakers_content_diarization ON speakers(content_id, local_speaker_id);

COMMIT;