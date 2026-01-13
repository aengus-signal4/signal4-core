-- Cleanup old canonical system columns
-- Removes deprecated canonical columns and simplifies the speaker table

BEGIN;

-- Set lock timeout to prevent long waits
SET lock_timeout = '10s';

-- ============================================================================
-- STEP 1: Drop the canonical_speakers view that depends on these columns
-- ============================================================================

DROP VIEW IF EXISTS canonical_speakers CASCADE;

-- ============================================================================
-- STEP 2: Drop deprecated columns from speakers table
-- ============================================================================

ALTER TABLE speakers 
    DROP COLUMN IF EXISTS canonical_speaker_id CASCADE,
    DROP COLUMN IF EXISTS is_canonical CASCADE,
    DROP COLUMN IF EXISTS is_cluster_representative CASCADE,
    DROP COLUMN IF EXISTS merge_history CASCADE,
    DROP COLUMN IF EXISTS merge_confidence CASCADE;

-- ============================================================================
-- STEP 3: Create a simpler view for speaker identities
-- ============================================================================

CREATE OR REPLACE VIEW speaker_identity_summary AS
SELECT 
    si.id as identity_id,
    si.primary_name,
    si.verification_status,
    si.confidence_score,
    COUNT(DISTINCT s.id) as speaker_count,
    COUNT(DISTINCT s.content_id) as content_count,
    SUM(s.duration) as total_duration,
    AVG(s.embedding_quality_score) as avg_quality
FROM speaker_identities si
LEFT JOIN speakers s ON s.speaker_identity_id = si.id
GROUP BY si.id, si.primary_name, si.verification_status, si.confidence_score;

-- ============================================================================
-- STEP 4: Report what was cleaned up
-- ============================================================================

SELECT 
    'Cleanup Complete' as status,
    COUNT(*) as total_speakers,
    COUNT(speaker_identity_id) as speakers_with_identity,
    COUNT(DISTINCT speaker_identity_id) as unique_identities,
    COUNT(cluster_id) as speakers_in_clusters,
    COUNT(DISTINCT cluster_id) as unique_clusters
FROM speakers;

COMMIT;

-- ============================================================================
-- To check remaining columns:
-- \d speakers
-- ============================================================================