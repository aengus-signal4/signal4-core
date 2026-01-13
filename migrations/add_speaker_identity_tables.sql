-- Migration: Add Speaker Identity Tables
-- Description: Separates speaker identities from embeddings for better data management
-- Author: Claude
-- Date: 2024

-- ============================================================================
-- STEP 1: Create new tables
-- ============================================================================

-- Speaker Identities table (permanent profiles)
CREATE TABLE IF NOT EXISTS speaker_identities (
    id SERIAL PRIMARY KEY,
    
    -- Primary identification
    primary_name VARCHAR(255),
    aliases TEXT[] DEFAULT '{}',
    
    -- Identity confidence and verification
    confidence_score FLOAT DEFAULT 0.5,
    verification_status VARCHAR(50) DEFAULT 'unverified',
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
    speaker_type VARCHAR(50),
    
    -- Activity statistics (denormalized for performance)
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

-- Speaker Assignments table (tracks how embeddings map to identities)
CREATE TABLE IF NOT EXISTS speaker_assignments (
    id SERIAL PRIMARY KEY,
    
    -- The assignment (will add FKs after altering speakers table)
    speaker_embedding_id INTEGER NOT NULL,
    speaker_identity_id INTEGER NOT NULL REFERENCES speaker_identities(id),
    
    -- Assignment details
    method VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    clustering_run_id VARCHAR(64),
    assigned_by VARCHAR(100),
    
    -- Evidence for the assignment
    evidence JSONB DEFAULT '{}',
    
    -- Temporal validity
    valid_from TIMESTAMP DEFAULT NOW() NOT NULL,
    valid_to TIMESTAMP,
    
    -- Ensure only one current assignment per embedding
    CONSTRAINT unique_current_assignment UNIQUE (speaker_embedding_id, valid_to)
);

-- Indexes for speaker_assignments
CREATE INDEX idx_assignment_embedding ON speaker_assignments(speaker_embedding_id);
CREATE INDEX idx_assignment_identity ON speaker_assignments(speaker_identity_id);
CREATE INDEX idx_assignment_current ON speaker_assignments(speaker_embedding_id, valid_to);
CREATE INDEX idx_assignment_method ON speaker_assignments(method);
CREATE INDEX idx_assignment_run ON speaker_assignments(clustering_run_id);

-- Clustering Runs table (track clustering operations)
CREATE TABLE IF NOT EXISTS clustering_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE NOT NULL,
    
    -- Run metadata
    run_type VARCHAR(50) NOT NULL,
    method VARCHAR(50) NOT NULL,
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
    status VARCHAR(20) DEFAULT 'running',
    error_message TEXT
);

-- Indexes for clustering_runs
CREATE INDEX idx_clustering_run_id ON clustering_runs(run_id);
CREATE INDEX idx_clustering_run_status ON clustering_runs(status);
CREATE INDEX idx_clustering_run_started ON clustering_runs(started_at);

-- Identity Merge History table
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
CREATE INDEX idx_merge_history_operation ON identity_merge_history(operation);
CREATE INDEX idx_merge_history_source ON identity_merge_history USING gin(source_identity_ids);
CREATE INDEX idx_merge_history_target ON identity_merge_history USING gin(target_identity_ids);
CREATE INDEX idx_merge_history_performed ON identity_merge_history(performed_at);

-- ============================================================================
-- STEP 2: Add new columns to speakers table
-- ============================================================================

ALTER TABLE speakers 
    ADD COLUMN IF NOT EXISTS speaker_identity_id INTEGER REFERENCES speaker_identities(id),
    ADD COLUMN IF NOT EXISTS assignment_confidence FLOAT,
    ADD COLUMN IF NOT EXISTS assignment_method VARCHAR(50),
    ADD COLUMN IF NOT EXISTS cluster_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS is_cluster_representative BOOLEAN DEFAULT FALSE;

-- Add indexes for new columns
CREATE INDEX IF NOT EXISTS idx_speakers_identity ON speakers(speaker_identity_id);
CREATE INDEX IF NOT EXISTS idx_speakers_cluster ON speakers(cluster_id);

-- ============================================================================
-- STEP 3: Create Unknown Speaker identity
-- ============================================================================

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
) ON CONFLICT DO NOTHING;

-- ============================================================================
-- STEP 4: Migrate existing canonical speakers to identities
-- ============================================================================

-- Create identities for canonical speakers
WITH canonical_speakers AS (
    SELECT 
        s.id,
        s.speaker_hash,
        s.display_name,
        s.merge_confidence,
        s.created_at,
        s.updated_at,
        s.duration,
        s.meta_data
    FROM speakers s
    WHERE s.is_canonical = true
)
INSERT INTO speaker_identities (
    primary_name,
    confidence_score,
    first_appearance,
    last_appearance,
    created_at,
    updated_at,
    notes
)
SELECT 
    COALESCE(cs.display_name, 'Speaker_' || cs.speaker_hash) as primary_name,
    COALESCE(cs.merge_confidence, 0.7) as confidence_score,
    cs.created_at as first_appearance,
    cs.updated_at as last_appearance,
    cs.created_at,
    cs.updated_at,
    'Migrated from canonical speaker ' || cs.speaker_hash as notes
FROM canonical_speakers cs;

-- ============================================================================
-- STEP 5: Link existing speakers to new identities
-- ============================================================================

-- Update canonical speakers with their identity IDs
UPDATE speakers s
SET 
    speaker_identity_id = si.id,
    assignment_confidence = COALESCE(s.merge_confidence, 0.7),
    assignment_method = 'migration',
    cluster_id = s.speaker_hash
FROM speaker_identities si
WHERE s.is_canonical = true
AND (si.primary_name = COALESCE(s.display_name, 'Speaker_' || s.speaker_hash));

-- Update non-canonical speakers based on canonical relationships
UPDATE speakers s
SET 
    speaker_identity_id = canonical.speaker_identity_id,
    assignment_confidence = COALESCE(s.merge_confidence, 0.6),
    assignment_method = 'migration',
    cluster_id = canonical.cluster_id
FROM speakers canonical
WHERE s.canonical_speaker_id = canonical.id
AND canonical.speaker_identity_id IS NOT NULL;

-- Assign orphaned speakers to Unknown Speaker
UPDATE speakers s
SET 
    speaker_identity_id = (SELECT id FROM speaker_identities WHERE primary_name = 'Unknown Speaker'),
    assignment_confidence = 0.0,
    assignment_method = 'orphaned'
WHERE s.speaker_identity_id IS NULL;

-- ============================================================================
-- STEP 6: Create assignment history from migration
-- ============================================================================

INSERT INTO speaker_assignments (
    speaker_embedding_id,
    speaker_identity_id,
    method,
    confidence,
    assigned_by,
    evidence,
    valid_from
)
SELECT 
    s.id,
    s.speaker_identity_id,
    COALESCE(s.assignment_method, 'migration'),
    COALESCE(s.assignment_confidence, 0.5),
    'migration_script',
    jsonb_build_object(
        'canonical_speaker_id', s.canonical_speaker_id,
        'is_canonical', s.is_canonical,
        'merge_confidence', s.merge_confidence
    ),
    NOW()
FROM speakers s
WHERE s.speaker_identity_id IS NOT NULL;

-- ============================================================================
-- STEP 7: Update speaker identity statistics
-- ============================================================================

-- Update episode counts and duration for each identity
WITH identity_stats AS (
    SELECT 
        s.speaker_identity_id,
        COUNT(DISTINCT s.content_id) as episode_count,
        SUM(s.duration) as total_duration,
        MIN(s.created_at) as first_appearance,
        MAX(s.created_at) as last_appearance,
        ARRAY_AGG(DISTINCT c.channel_name) FILTER (WHERE c.channel_name IS NOT NULL) as channels
    FROM speakers s
    LEFT JOIN content c ON s.content_id::text = c.content_id
    WHERE s.speaker_identity_id IS NOT NULL
    GROUP BY s.speaker_identity_id
)
UPDATE speaker_identities si
SET 
    total_episodes = COALESCE(stats.episode_count, 0),
    total_duration = COALESCE(stats.total_duration, 0.0),
    first_appearance = stats.first_appearance,
    last_appearance = stats.last_appearance,
    primary_channels = COALESCE(stats.channels, '{}')
FROM identity_stats stats
WHERE si.id = stats.speaker_identity_id;

-- ============================================================================
-- STEP 8: Add foreign key for speaker_assignments
-- ============================================================================

ALTER TABLE speaker_assignments 
    ADD CONSTRAINT fk_assignment_embedding 
    FOREIGN KEY (speaker_embedding_id) 
    REFERENCES speakers(id);

-- ============================================================================
-- STEP 9: Create views for backward compatibility (optional)
-- ============================================================================

-- View that mimics old canonical speaker structure
CREATE OR REPLACE VIEW canonical_speakers AS
SELECT 
    s.*,
    si.primary_name as canonical_name,
    si.confidence_score as identity_confidence
FROM speakers s
JOIN speaker_identities si ON s.speaker_identity_id = si.id
WHERE s.is_cluster_representative = true;

-- ============================================================================
-- STEP 10: Add comments for documentation
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
-- Migration complete!
-- ============================================================================

-- To rollback:
-- DROP TABLE identity_merge_history CASCADE;
-- DROP TABLE clustering_runs CASCADE;
-- DROP TABLE speaker_assignments CASCADE;
-- DROP TABLE speaker_identities CASCADE;
-- ALTER TABLE speakers 
--     DROP COLUMN speaker_identity_id,
--     DROP COLUMN assignment_confidence,
--     DROP COLUMN assignment_method,
--     DROP COLUMN cluster_id,
--     DROP COLUMN is_cluster_representative;
-- DROP VIEW IF EXISTS canonical_speakers;