-- Fix Speaker Embedding Dimensions from 256 to 512
-- WARNING: This will DELETE ALL EXISTING SPEAKER DATA!

-- Drop dependent objects first
DROP TABLE IF EXISTS speaker_embeddings CASCADE;
DROP TABLE IF EXISTS speaker_transcriptions CASCADE;

-- Drop and recreate speakers table with correct dimensions
DROP TABLE IF EXISTS speakers CASCADE;

CREATE TABLE speakers (
    id SERIAL PRIMARY KEY,
    global_id VARCHAR UNIQUE NOT NULL,
    universal_name VARCHAR UNIQUE NOT NULL,
    display_name VARCHAR,
    embedding vector(512) NOT NULL,  -- Changed from 256 to 512
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_segments INTEGER DEFAULT 0,
    total_duration FLOAT DEFAULT 0.0,
    last_seen TIMESTAMP,
    last_content_id VARCHAR,
    appearance_count INTEGER DEFAULT 0,
    meta_data JSONB DEFAULT '{}'::jsonb
);

-- Recreate indexes for speakers
CREATE INDEX idx_speaker_global_id ON speakers(global_id);
CREATE INDEX idx_speaker_universal_name ON speakers(universal_name);
CREATE INDEX idx_speaker_total_segments ON speakers(total_segments);
CREATE INDEX idx_speaker_total_duration ON speakers(total_duration);
CREATE INDEX idx_speaker_last_seen ON speakers(last_seen);
CREATE INDEX idx_speaker_appearance_count ON speakers(appearance_count);
CREATE INDEX idx_speaker_stats ON speakers(total_segments, total_duration, appearance_count);
CREATE INDEX idx_speaker_embedding ON speakers USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Recreate speaker_embeddings table
CREATE TABLE speaker_embeddings (
    id SERIAL PRIMARY KEY,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id),
    content_id VARCHAR NOT NULL,
    local_speaker_id VARCHAR NOT NULL,
    embedding vector(512) NOT NULL,  -- Changed from 256 to 512
    duration FLOAT NOT NULL,
    segment_count INTEGER NOT NULL,
    quality_score FLOAT DEFAULT 1.0,
    algorithm_version VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_speaker_embedding_content UNIQUE(speaker_id, content_id, local_speaker_id)
);

-- Recreate indexes for speaker_embeddings
CREATE INDEX idx_speaker_embeddings_speaker_id ON speaker_embeddings(speaker_id);
CREATE INDEX idx_speaker_embeddings_content_id ON speaker_embeddings(content_id);
CREATE INDEX idx_speaker_embeddings_created_at ON speaker_embeddings(created_at);
CREATE INDEX idx_speaker_embeddings_quality ON speaker_embeddings(quality_score);
CREATE INDEX idx_speaker_embeddings_duration ON speaker_embeddings(duration);
CREATE INDEX idx_speaker_embeddings_speaker_quality ON speaker_embeddings(speaker_id, quality_score, duration);

-- Recreate speaker_transcriptions table
CREATE TABLE speaker_transcriptions (
    id SERIAL PRIMARY KEY,
    content_id INTEGER NOT NULL REFERENCES content(id),
    speaker_id INTEGER NOT NULL REFERENCES speakers(id),
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    text TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    stitch_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Recreate indexes for speaker_transcriptions
CREATE INDEX idx_speaker_transcriptions_content_id ON speaker_transcriptions(content_id);
CREATE INDEX idx_speaker_transcriptions_speaker_id ON speaker_transcriptions(speaker_id);
CREATE INDEX idx_speaker_transcriptions_start_time ON speaker_transcriptions(start_time);
CREATE INDEX idx_speaker_transcriptions_end_time ON speaker_transcriptions(end_time);
CREATE INDEX idx_speaker_transcriptions_text ON speaker_transcriptions USING gin (text gin_trgm_ops);
CREATE INDEX idx_speaker_transcriptions_turn_index ON speaker_transcriptions(turn_index);

-- Ensure the speaker sequence exists
CREATE SEQUENCE IF NOT EXISTS speaker_number_seq START 1 INCREMENT 1;

-- Grant necessary permissions (adjust as needed)
GRANT ALL ON speakers TO av_content_user;
GRANT ALL ON speaker_embeddings TO av_content_user;
GRANT ALL ON speaker_transcriptions TO av_content_user;
GRANT USAGE ON SEQUENCE speaker_number_seq TO av_content_user;

-- Verify the changes
SELECT 
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns 
WHERE table_name IN ('speakers', 'speaker_embeddings') 
    AND column_name = 'embedding'
ORDER BY table_name;