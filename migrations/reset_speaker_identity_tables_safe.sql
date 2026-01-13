-- Safe reset of speaker identity tables with proper locking and error handling
-- This script handles existing tables and prevents deadlocks

BEGIN;

-- Set lock timeout to prevent long waits
SET lock_timeout = '10s';

-- Drop dependent tables first (in reverse dependency order)
DROP TABLE IF EXISTS speaker_identity_verifications CASCADE;
DROP TABLE IF EXISTS speaker_identity_merges CASCADE;
DROP TABLE IF EXISTS speaker_canonicals CASCADE;

-- Drop the main identity table and its cascade dependencies
DROP TABLE IF EXISTS speaker_identities CASCADE;

-- Drop the view if it exists
DROP VIEW IF EXISTS canonical_speakers CASCADE;

-- Reset the speakers table
UPDATE speakers SET 
    speaker_identity_id = NULL,
    canonical_speaker_id = NULL,
    canonical_confidence = NULL,
    canonical_reason = NULL,
    canonical_established_at = NULL,
    last_canonical_check = NULL;

-- Drop old columns if they exist
ALTER TABLE speakers DROP COLUMN IF EXISTS canonical_speaker_id CASCADE;
ALTER TABLE speakers DROP COLUMN IF EXISTS canonical_confidence CASCADE;
ALTER TABLE speakers DROP COLUMN IF EXISTS canonical_reason CASCADE;
ALTER TABLE speakers DROP COLUMN IF EXISTS canonical_established_at CASCADE;
ALTER TABLE speakers DROP COLUMN IF EXISTS last_canonical_check CASCADE;

-- Create speaker_identities table (master identity records)
CREATE TABLE IF NOT EXISTS speaker_identities (
    id SERIAL PRIMARY KEY,
    primary_name VARCHAR(255) NOT NULL,
    aliases TEXT[],  -- Array of known aliases
    description TEXT,
    tags TEXT[],  -- Tags like 'politician', 'journalist', etc.
    confidence_score FLOAT DEFAULT 0.0,
    is_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for speaker_identities
CREATE INDEX IF NOT EXISTS idx_speaker_identity_primary_name ON speaker_identities(primary_name);
CREATE INDEX IF NOT EXISTS idx_speaker_identity_aliases ON speaker_identities USING gin(aliases);
CREATE INDEX IF NOT EXISTS idx_speaker_identity_verification ON speaker_identities(is_verified);
CREATE INDEX IF NOT EXISTS idx_speaker_identity_tags ON speaker_identities USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_speaker_identity_active ON speaker_identities(is_active);
CREATE INDEX IF NOT EXISTS idx_speaker_identity_confidence ON speaker_identities(confidence_score);

-- Create speaker_canonicals table (maps speakers to their canonical representations)
CREATE TABLE IF NOT EXISTS speaker_canonicals (
    id SERIAL PRIMARY KEY,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    canonical_speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    confidence FLOAT NOT NULL,
    reason VARCHAR(50),  -- 'embedding_similarity', 'manual_merge', 'ai_inference'
    metadata JSONB,  -- Store additional info like similarity scores
    established_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(speaker_id)  -- Each speaker can only have one canonical
);

-- Create indexes for speaker_canonicals
CREATE INDEX IF NOT EXISTS idx_speaker_canonical_speaker ON speaker_canonicals(speaker_id);
CREATE INDEX IF NOT EXISTS idx_speaker_canonical_canonical ON speaker_canonicals(canonical_speaker_id);
CREATE INDEX IF NOT EXISTS idx_speaker_canonical_confidence ON speaker_canonicals(confidence);

-- Create speaker_identity_merges table (track merge history)
CREATE TABLE IF NOT EXISTS speaker_identity_merges (
    id SERIAL PRIMARY KEY,
    from_identity_id INTEGER NOT NULL,
    to_identity_id INTEGER NOT NULL REFERENCES speaker_identities(id) ON DELETE CASCADE,
    merged_by VARCHAR(100),  -- 'system', 'manual', username
    merge_reason TEXT,
    confidence FLOAT,
    metadata JSONB,
    merged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for speaker_identity_merges
CREATE INDEX IF NOT EXISTS idx_speaker_merge_from ON speaker_identity_merges(from_identity_id);
CREATE INDEX IF NOT EXISTS idx_speaker_merge_to ON speaker_identity_merges(to_identity_id);
CREATE INDEX IF NOT EXISTS idx_speaker_merge_date ON speaker_identity_merges(merged_at);

-- Create speaker_identity_verifications table (track verification history)
CREATE TABLE IF NOT EXISTS speaker_identity_verifications (
    id SERIAL PRIMARY KEY,
    speaker_identity_id INTEGER NOT NULL REFERENCES speaker_identities(id) ON DELETE CASCADE,
    verified_by VARCHAR(100) NOT NULL,
    verification_method VARCHAR(50),  -- 'manual', 'ai_assisted', 'bulk_import'
    verification_notes TEXT,
    metadata JSONB,
    verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for speaker_identity_verifications
CREATE INDEX IF NOT EXISTS idx_speaker_verification_identity ON speaker_identity_verifications(speaker_identity_id);
CREATE INDEX IF NOT EXISTS idx_speaker_verification_date ON speaker_identity_verifications(verified_at);
CREATE INDEX IF NOT EXISTS idx_speaker_verification_method ON speaker_identity_verifications(verification_method);
CREATE INDEX IF NOT EXISTS idx_speaker_verification_by ON speaker_identity_verifications(verified_by);

-- Add speaker_identity_id column to speakers table if it doesn't exist
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS speaker_identity_id INTEGER REFERENCES speaker_identities(id) ON DELETE SET NULL;

-- Add canonical columns to speakers table
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS canonical_speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL;
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS canonical_confidence FLOAT;
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS canonical_reason VARCHAR(50);
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS canonical_established_at TIMESTAMP;
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS last_canonical_check TIMESTAMP;

-- Create indexes for the new columns
CREATE INDEX IF NOT EXISTS idx_speakers_identity ON speakers(speaker_identity_id);
CREATE INDEX IF NOT EXISTS idx_speakers_canonical ON speakers(canonical_speaker_id);

-- Reset sequence if needed (only if table was just created)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM speaker_identities LIMIT 1) THEN
        ALTER SEQUENCE speaker_identities_id_seq RESTART WITH 1;
    END IF;
END $$;

-- Add helpful comments
COMMENT ON TABLE speaker_identities IS 'Master identity records for speakers across all content';
COMMENT ON TABLE speaker_canonicals IS 'Maps individual speaker instances to their canonical representation';
COMMENT ON TABLE speaker_identity_merges IS 'Audit trail of identity merges';
COMMENT ON TABLE speaker_identity_verifications IS 'Audit trail of identity verifications';
COMMENT ON COLUMN speakers.speaker_identity_id IS 'Link to verified speaker identity';
COMMENT ON COLUMN speakers.canonical_speaker_id IS 'The canonical speaker this speaker maps to';
COMMENT ON COLUMN speakers.canonical_confidence IS 'Confidence score for the canonical mapping';
COMMENT ON COLUMN speakers.canonical_reason IS 'How the canonical relationship was established';
COMMENT ON COLUMN speakers.canonical_established_at IS 'When the canonical relationship was created';

-- Create a view for canonical speakers
CREATE OR REPLACE VIEW canonical_speakers AS
SELECT 
    s.id,
    s.content_id,
    s.speaker_name,
    s.canonical_speaker_id,
    s.canonical_confidence,
    s.canonical_reason,
    cs.speaker_name as canonical_name,
    cs.importance_score as canonical_importance,
    COUNT(sc.speaker_id) as linked_speaker_count
FROM speakers s
LEFT JOIN speakers cs ON s.canonical_speaker_id = cs.id
LEFT JOIN speaker_canonicals sc ON cs.id = sc.canonical_speaker_id
WHERE s.canonical_speaker_id IS NOT NULL
GROUP BY s.id, s.content_id, s.speaker_name, s.canonical_speaker_id, 
         s.canonical_confidence, s.canonical_reason, cs.speaker_name, cs.importance_score;

-- Report the final state
SELECT 'Speaker Identities' as table_name, COUNT(*) as count FROM speaker_identities
UNION ALL
SELECT 'Speakers with Identity', COUNT(*) FROM speakers WHERE speaker_identity_id IS NOT NULL
UNION ALL
SELECT 'Total Speakers', COUNT(*) FROM speakers;

COMMIT;