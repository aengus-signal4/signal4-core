-- First, create a temporary column with the new dimension
ALTER TABLE speakers 
    ADD COLUMN embedding_new vector(256);

ALTER TABLE speaker_embeddings 
    ADD COLUMN embedding_new vector(256);

-- Convert existing embeddings by taking the first 256 dimensions
UPDATE speakers 
SET embedding_new = embedding[1:256];

UPDATE speaker_embeddings 
SET embedding_new = embedding[1:256];

-- Drop the old columns
ALTER TABLE speakers 
    DROP COLUMN embedding;

ALTER TABLE speaker_embeddings 
    DROP COLUMN embedding;

-- Rename the new columns to the original names
ALTER TABLE speakers 
    RENAME COLUMN embedding_new TO embedding;

ALTER TABLE speaker_embeddings 
    RENAME COLUMN embedding_new TO embedding;

-- Add comments to document the change
COMMENT ON COLUMN speakers.embedding IS '256-dimensional speaker embedding vector from pyannote/wespeaker-voxceleb-resnet34-LM model (truncated from 512 dimensions)';
COMMENT ON COLUMN speaker_embeddings.embedding IS '256-dimensional speaker embedding vector from pyannote/wespeaker-voxceleb-resnet34-LM model (truncated from 512 dimensions)'; 