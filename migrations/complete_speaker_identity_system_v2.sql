-- Complete Speaker Identity System Migration (v2)
-- This migration completes the speaker identity system by adding missing tables
-- and ensuring all relationships are properly established

BEGIN;

-- Set lock timeout to prevent long waits
SET lock_timeout = '10s';

-- ============================================================================
-- STEP 1: Create missing identity_merge_history table
-- ============================================================================

CREATE TABLE IF NOT EXISTS identity_merge_history (
    id SERIAL PRIMARY KEY,
    
    -- Operation type
    operation VARCHAR(20) NOT NULL,
    
    -- For merges: multiple sources -> one target
    -- For splits: one source -> multiple targets
    source_identity_ids INTEGER[] NOT NULL,
    target_identity_ids INTEGER[] NOT NULL,
    
    -- Operation details
    confidence FLOAT,
    reason TEXT,
    evidence JSONB DEFAULT '{}',
    
    -- Who/what performed it
    performed_by VARCHAR(100),
    clustering_run_id VARCHAR(64),
    
    -- When
    performed_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for identity_merge_history
CREATE INDEX IF NOT EXISTS idx_merge_history_operation ON identity_merge_history(operation);
CREATE INDEX IF NOT EXISTS idx_merge_history_source ON identity_merge_history USING gin(source_identity_ids);
CREATE INDEX IF NOT EXISTS idx_merge_history_target ON identity_merge_history USING gin(target_identity_ids);
CREATE INDEX IF NOT EXISTS idx_merge_history_performed ON identity_merge_history(performed_at);

-- ============================================================================
-- STEP 2: Ensure all columns exist on speakers table
-- ============================================================================

-- Add columns if they don't exist (safe to run multiple times)
ALTER TABLE speakers 
    ADD COLUMN IF NOT EXISTS speaker_identity_id INTEGER REFERENCES speaker_identities(id),
    ADD COLUMN IF NOT EXISTS assignment_confidence FLOAT,
    ADD COLUMN IF NOT EXISTS assignment_method VARCHAR(50),
    ADD COLUMN IF NOT EXISTS cluster_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS is_cluster_representative BOOLEAN DEFAULT FALSE;

-- Add indexes for new columns if they don't exist
CREATE INDEX IF NOT EXISTS idx_speakers_identity ON speakers(speaker_identity_id);
CREATE INDEX IF NOT EXISTS idx_speakers_cluster ON speakers(cluster_id);

-- ============================================================================
-- STEP 3: Ensure speaker_assignments has proper foreign key
-- ============================================================================

-- Add foreign key if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_assignment_embedding'
        AND table_name = 'speaker_assignments'
    ) THEN
        ALTER TABLE speaker_assignments 
            ADD CONSTRAINT fk_assignment_embedding 
            FOREIGN KEY (speaker_embedding_id) 
            REFERENCES speakers(id);
    END IF;
END $$;

-- ============================================================================
-- STEP 4: Create Unknown Speaker identity if it doesn't exist
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM speaker_identities 
        WHERE primary_name = 'Unknown Speaker'
    ) THEN
        INSERT INTO speaker_identities (
            primary_name,
            verification_status,
            confidence_score,
            notes
        ) VALUES (
            'Unknown Speaker',
            'unknown',
            0.0,
            'Placeholder for unidentified speakers'
        );
    END IF;
END $$;

-- ============================================================================
-- STEP 5: Clean up any orphaned canonical relationships
-- ============================================================================

-- Reset canonical columns that reference non-existent speakers
UPDATE speakers 
SET canonical_speaker_id = NULL 
WHERE canonical_speaker_id IS NOT NULL 
AND NOT EXISTS (
    SELECT 1 FROM speakers s2 
    WHERE s2.id = speakers.canonical_speaker_id
);

-- ============================================================================
-- STEP 6: Create backward compatibility view
-- ============================================================================

DROP VIEW IF EXISTS canonical_speakers CASCADE;

CREATE OR REPLACE VIEW canonical_speakers AS
SELECT 
    s.*,
    si.primary_name as identity_name,
    si.confidence_score as identity_confidence,
    si.verification_status,
    si.tags,
    si.bio,
    si.occupation,
    si.organization
FROM speakers s
LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
WHERE s.is_cluster_representative = true 
   OR s.is_canonical = true;

-- ============================================================================
-- STEP 7: Add helpful comments
-- ============================================================================

COMMENT ON TABLE speaker_identities IS 'Permanent speaker profiles that persist across re-clustering';
COMMENT ON TABLE speaker_assignments IS 'Tracks how speaker embeddings are assigned to identities';
COMMENT ON TABLE clustering_runs IS 'Audit trail of clustering operations';
COMMENT ON TABLE identity_merge_history IS 'History of speaker identity merges and splits';

COMMENT ON COLUMN speakers.speaker_identity_id IS 'Foreign key to permanent speaker identity';
COMMENT ON COLUMN speakers.assignment_confidence IS 'Confidence in the identity assignment';
COMMENT ON COLUMN speakers.assignment_method IS 'How the assignment was made (clustering, manual, etc)';
COMMENT ON COLUMN speakers.cluster_id IS 'Current cluster assignment';
COMMENT ON COLUMN speakers.is_cluster_representative IS 'Whether this embedding is the cluster centroid';

-- ============================================================================
-- STEP 8: Report the final state
-- ============================================================================

SELECT 'Database Tables' as category, COUNT(*) as count FROM information_schema.tables 
WHERE table_name IN ('speaker_identities', 'speaker_assignments', 'clustering_runs', 'identity_merge_history')
UNION ALL
SELECT 'Speaker Identities', COUNT(*) FROM speaker_identities
UNION ALL
SELECT 'Speakers Total', COUNT(*) FROM speakers
UNION ALL
SELECT 'Speakers with Identity', COUNT(*) FROM speakers WHERE speaker_identity_id IS NOT NULL
UNION ALL
SELECT 'Canonical Speakers', COUNT(*) FROM speakers WHERE is_canonical = true
UNION ALL
SELECT 'Cluster Representatives', COUNT(*) FROM speakers WHERE is_cluster_representative = true
UNION ALL
SELECT 'Speaker Assignments', COUNT(*) FROM speaker_assignments
ORDER BY category;

COMMIT;

-- ============================================================================
-- Migration complete!
-- ============================================================================

-- To verify the setup:
-- SELECT table_name FROM information_schema.tables 
-- WHERE table_name LIKE 'speaker%' OR table_name LIKE '%identity%' 
-- ORDER BY table_name;