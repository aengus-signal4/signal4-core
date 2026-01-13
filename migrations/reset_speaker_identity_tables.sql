-- Migration: Reset Speaker Identity Tables
-- Description: Clean setup of speaker identity structure for fresh clustering
-- Author: Claude
-- Date: 2024

-- ============================================================================
-- STEP 1: Clean up any existing identity tables (if partial migration ran)
-- ============================================================================

DROP TABLE IF EXISTS identity_merge_history CASCADE;
DROP TABLE IF EXISTS clustering_runs CASCADE;
DROP TABLE IF EXISTS speaker_assignments CASCADE;
DROP TABLE IF EXISTS speaker_identities CASCADE;
DROP VIEW IF EXISTS canonical_speakers;

-- ============================================================================
-- STEP 2: Reset speaker columns
-- ============================================================================

-- Clear old canonical relationships since we're starting fresh
UPDATE speakers SET 
    canonical_speaker_id = NULL,
    is_canonical = FALSE,
    merge_confidence = NULL,
    merge_history = '[]'::json;

-- Drop columns if they exist from previous migration
ALTER TABLE speakers 
    DROP COLUMN IF EXISTS speaker_identity_id,
    DROP COLUMN IF EXISTS assignment_confidence,
    DROP COLUMN IF EXISTS assignment_method,
    DROP COLUMN IF EXISTS cluster_id,
    DROP COLUMN IF EXISTS is_cluster_representative;

-- ============================================================================
-- STEP 3: Create fresh identity tables
-- ============================================================================

-- Speaker Identities table (permanent profiles)
CREATE TABLE speaker_identities (
    id SERIAL PRIMARY KEY,
    
    -- Primary identification
    primary_name VARCHAR(255),
    aliases TEXT[] DEFAULT '{}',
    
    -- Identity confidence and verification
    confidence_score FLOAT DEFAULT 0.5,
    verification_status VARCHAR(50) DEFAULT 'unverified', -- verified, unverified, disputed, unknown
    verification_metadata JSONB DEFAULT '{}',
    
    -- Rich metadata
    bio TEXT,
    occupation VARCHAR(255),
    organization VARCHAR(255),
    location VARCHAR(255),
    country VARCHAR(10),
    
    -- External profiles and IDs
    social_profiles JSONB DEFAULT '{}',
    external_ids JSONB DEFAULT '{}',
    website VARCHAR(500),
    
    -- Categorization
    tags TEXT[] DEFAULT '{}',
    speaker_type VARCHAR(50), -- host, guest, politician, expert, etc.
    
    -- Activity statistics (will be updated by clustering)
    first_appearance TIMESTAMP,
    last_appearance TIMESTAMP,
    total_episodes INTEGER DEFAULT 0,
    total_duration FLOAT DEFAULT 0.0,
    primary_channels TEXT[] DEFAULT '{}',
    
    -- Management
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);

-- Indexes for speaker_identities
CREATE INDEX idx_speaker_identity_primary_name ON speaker_identities(primary_name);
CREATE INDEX idx_speaker_identity_aliases ON speaker_identities USING gin(aliases);
CREATE INDEX idx_speaker_identity_verification ON speaker_identities(verification_status);
CREATE INDEX idx_speaker_identity_tags ON speaker_identities USING gin(tags);
CREATE INDEX idx_speaker_identity_active ON speaker_identities(is_active);
CREATE INDEX idx_speaker_identity_confidence ON speaker_identities(confidence_score);

-- Speaker Assignments table
CREATE TABLE speaker_assignments (
    id SERIAL PRIMARY KEY,
    
    -- The assignment
    speaker_embedding_id INTEGER NOT NULL REFERENCES speakers(id),
    speaker_identity_id INTEGER NOT NULL REFERENCES speaker_identities(id),
    
    -- Assignment details
    method VARCHAR(50) NOT NULL, -- clustering, direct_similarity, manual, llm
    confidence FLOAT NOT NULL,
    clustering_run_id VARCHAR(64),
    assigned_by VARCHAR(100),
    
    -- Evidence for the assignment
    evidence JSONB DEFAULT '{}',
    
    -- Temporal validity
    valid_from TIMESTAMP DEFAULT NOW() NOT NULL,
    valid_to TIMESTAMP, -- NULL means current
    
    -- Ensure only one current assignment per embedding
    CONSTRAINT unique_current_assignment UNIQUE (speaker_embedding_id, valid_to)
);

-- Indexes for speaker_assignments
CREATE INDEX idx_assignment_embedding ON speaker_assignments(speaker_embedding_id);
CREATE INDEX idx_assignment_identity ON speaker_assignments(speaker_identity_id);
CREATE INDEX idx_assignment_current ON speaker_assignments(speaker_embedding_id, valid_to);
CREATE INDEX idx_assignment_method ON speaker_assignments(method);
CREATE INDEX idx_assignment_run ON speaker_assignments(clustering_run_id);

-- Clustering Runs table
CREATE TABLE clustering_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE NOT NULL,
    
    -- Run metadata
    run_type VARCHAR(50) NOT NULL, -- full, incremental, orphan_assignment
    method VARCHAR(50) NOT NULL, -- anchor_canopy, dbscan, etc.
    parameters JSONB NOT NULL,
    
    -- Statistics
    embeddings_processed INTEGER DEFAULT 0,
    clusters_created INTEGER DEFAULT 0,
    assignments_made INTEGER DEFAULT 0,
    identities_created INTEGER DEFAULT 0,
    identities_merged INTEGER DEFAULT 0,
    
    -- Timing
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    processing_time_seconds FLOAT,
    
    -- Status
    status VARCHAR(20) DEFAULT 'running', -- running, completed, failed
    error_message TEXT
);

-- Indexes for clustering_runs
CREATE INDEX idx_clustering_run_id ON clustering_runs(run_id);
CREATE INDEX idx_clustering_run_status ON clustering_runs(status);
CREATE INDEX idx_clustering_run_started ON clustering_runs(started_at);

-- Identity Merge History table
CREATE TABLE identity_merge_history (
    id SERIAL PRIMARY KEY,
    
    -- Operation type
    operation VARCHAR(20) NOT NULL, -- merge, split
    
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
CREATE INDEX idx_merge_history_operation ON identity_merge_history(operation);
CREATE INDEX idx_merge_history_source ON identity_merge_history USING gin(source_identity_ids);
CREATE INDEX idx_merge_history_target ON identity_merge_history USING gin(target_identity_ids);
CREATE INDEX idx_merge_history_performed ON identity_merge_history(performed_at);

-- ============================================================================
-- STEP 4: Add new columns to speakers table
-- ============================================================================

ALTER TABLE speakers 
    ADD COLUMN speaker_identity_id INTEGER REFERENCES speaker_identities(id),
    ADD COLUMN assignment_confidence FLOAT,
    ADD COLUMN assignment_method VARCHAR(50),
    ADD COLUMN cluster_id VARCHAR(64),
    ADD COLUMN is_cluster_representative BOOLEAN DEFAULT FALSE;

-- Add indexes for new columns
CREATE INDEX idx_speakers_identity ON speakers(speaker_identity_id);
CREATE INDEX idx_speakers_cluster ON speakers(cluster_id);

-- ============================================================================
-- STEP 5: Create default Unknown Speaker identity
-- ============================================================================

INSERT INTO speaker_identities (
    id,
    primary_name,
    verification_status,
    confidence_score,
    notes,
    created_by
) VALUES (
    1, -- Use ID 1 for Unknown Speaker
    'Unknown Speaker',
    'unknown',
    0.0,
    'Default identity for unassigned speakers',
    'system'
);

-- Update sequence to start after our manual ID
SELECT setval('speaker_identities_id_seq', 1, true);

-- ============================================================================
-- STEP 6: Add helpful comments
-- ============================================================================

COMMENT ON TABLE speaker_identities IS 'Permanent speaker profiles that persist across re-clustering';
COMMENT ON TABLE speaker_assignments IS 'Tracks how speaker embeddings are assigned to identities with full history';
COMMENT ON TABLE clustering_runs IS 'Audit trail of all clustering operations';
COMMENT ON TABLE identity_merge_history IS 'History of speaker identity merges and splits';

COMMENT ON COLUMN speakers.speaker_identity_id IS 'Foreign key to permanent speaker identity';
COMMENT ON COLUMN speakers.assignment_confidence IS 'Confidence score for the identity assignment';
COMMENT ON COLUMN speakers.assignment_method IS 'How the assignment was made (clustering, manual, etc)';
COMMENT ON COLUMN speakers.cluster_id IS 'Current cluster assignment from clustering';
COMMENT ON COLUMN speakers.is_cluster_representative IS 'Whether this embedding is the cluster centroid';

-- ============================================================================
-- Done! Ready for fresh clustering
-- ============================================================================

-- Summary of what we've created:
SELECT 'Speaker Identities' as table_name, COUNT(*) as count FROM speaker_identities
UNION ALL
SELECT 'Speakers with Identity', COUNT(*) FROM speakers WHERE speaker_identity_id IS NOT NULL
UNION ALL
SELECT 'Total Speakers', COUNT(*) FROM speakers;