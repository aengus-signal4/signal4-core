-- ============================================================================
-- Embedding Cache & Clustering Infrastructure
-- ============================================================================
--
-- Complete infrastructure for semantic search and theme discovery:
--
-- PART 1-2: Embedding Cache Tables (with HNSW indexes)
--   - embedding_cache_30d (main + alt): 30-day rolling window, 6-hour refresh
--   - embedding_cache_7d (main + alt): 7-day hot cache, hourly refresh
--
-- INCREMENTAL REFRESH STRATEGY:
--   Previous approach (TRUNCATE + INSERT) took 50+ minutes due to HNSW rebuild.
--   New approach uses incremental updates:
--   1. DELETE rows where publish_date aged out of the window
--   2. INSERT new rows from recently updated content (last 12 hours)
--   3. ON CONFLICT (id) DO NOTHING to skip existing rows
--
--   Performance improvement:
--   - Old method: 50+ minutes (full HNSW rebuild)
--   - New method with inserts: ~3-4 minutes (incremental HNSW updates)
--   - New method no changes: ~50ms (just scans, nothing to do)
--
-- DUAL EMBEDDING MODEL ARCHITECTURE:
--   The system maintains TWO independent embedding models for each segment:
--
--   PRIMARY (embedding column):
--     - Source: embedding_segments.embedding
--     - Model: qwen3:0.6b (1024 dimensions)
--     - Optimized for: Fast, multilingual semantic search
--
--   ALTERNATE (embedding_alt column):
--     - Source: embedding_segments.embedding_alt
--     - Model: qwen3:4b (2000 dimensions, truncated from 2560)
--     - Optimized for: High-quality, nuanced semantic understanding
--
--   Both embeddings are stored in the SAME cache table to avoid duplicating metadata.
--   Each cache table has TWO HNSW indexes (one for each embedding column).
--   Clustering can be performed on either model for different analytical perspectives.
--
-- PART 3-6: Pre-computed Clustering Infrastructure
--   - embedding_clusters: Stores cluster assignments per segment
--   - cluster_metadata: Aggregate cluster statistics
--   - Helper functions: get_latest_clusters(), cleanup_old_clusters()
--   - pg_cron jobs: Automated clustering after cache refreshes
--
-- INDEX CHOICE: HNSW vs IVFFlat
--   We use HNSW indexes because:
--   - Better performance with filtered queries (WHERE + ORDER BY vector)
--   - No need to tune 'lists' parameter based on table size
--   - More consistent query times
--   - Better recall at same speed
--
--   HNSW parameters:
--   - m=16: connections per node (higher = better recall, more memory)
--   - ef_construction=64: build-time search width (higher = better index quality)
--   - ef_search=100: query-time search width (set via SET hnsw.ef_search)
--
-- Benefits:
-- - 10-100x faster semantic search via HNSW indexes
-- - Works well with filtered queries (project, date filters)
-- - Instant landing page loads (<500ms vs 90s) via pre-computed clusters
-- - Always fresh data via automated refresh schedules
-- - Dual models enable both fast queries (main) and quality results (alt)
-- - Incremental refresh means sub-second updates for typical runs
--
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- ============================================================================
-- PART 1: Embedding Cache Tables (30-day and 7-day views)
-- ============================================================================

-- Drop existing tables if rebuilding from scratch
-- DROP TABLE IF EXISTS embedding_cache_30d CASCADE;
-- DROP TABLE IF EXISTS embedding_cache_7d CASCADE;

-- 30-day table (built directly from source tables)
CREATE TABLE IF NOT EXISTS embedding_cache_30d AS
SELECT
  es.id,
  es.content_id,
  es.embedding,
  es.embedding_alt,
  c.projects,
  c.publish_date,
  c.channel_url,
  c.channel_name,
  c.title,
  c.main_language,
  es.source_speaker_hashes,
  es.text,
  es.start_time,
  es.end_time,
  es.segment_index,
  es.content_id_string,
  es.stitch_version
FROM embedding_segments es
JOIN content c ON es.content_id = c.id
WHERE c.publish_date >= NOW() - INTERVAL '30 days'
  AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL);

-- 7-day table (hot cache, built directly from source tables)
CREATE TABLE IF NOT EXISTS embedding_cache_7d AS
SELECT
  es.id,
  es.content_id,
  es.embedding,
  es.embedding_alt,
  c.projects,
  c.publish_date,
  c.channel_url,
  c.channel_name,
  c.title,
  c.main_language,
  es.source_speaker_hashes,
  es.text,
  es.start_time,
  es.end_time,
  es.segment_index,
  es.content_id_string,
  es.stitch_version
FROM embedding_segments es
JOIN content c ON es.content_id = c.id
WHERE c.publish_date >= NOW() - INTERVAL '7 days'
  AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL);

-- ============================================================================
-- PART 2: HNSW Vector Indexes on Cache Tables
-- ============================================================================
-- HNSW indexes provide better performance for filtered queries compared to IVFFlat.
-- Parameters: m=16 (connections), ef_construction=64 (build quality)
-- At query time, set hnsw.ef_search=100 for good recall/speed balance.

-- 30-day cache (~500K rows)
CREATE INDEX IF NOT EXISTS embedding_cache_30d_hnsw_main
  ON embedding_cache_30d
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS embedding_cache_30d_hnsw_alt
  ON embedding_cache_30d
  USING hnsw (embedding_alt vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 7-day cache (~100K rows)
CREATE INDEX IF NOT EXISTS embedding_cache_7d_hnsw_main
  ON embedding_cache_7d
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS embedding_cache_7d_hnsw_alt
  ON embedding_cache_7d
  USING hnsw (embedding_alt vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Supporting indexes (GIN for projects, B-tree for dates/content_id/language)
-- 30d
CREATE INDEX IF NOT EXISTS embedding_cache_30d_projects ON embedding_cache_30d USING gin (projects);
CREATE INDEX IF NOT EXISTS embedding_cache_30d_publish_date ON embedding_cache_30d (publish_date);
CREATE INDEX IF NOT EXISTS embedding_cache_30d_content_id ON embedding_cache_30d (content_id);
CREATE INDEX IF NOT EXISTS embedding_cache_30d_language ON embedding_cache_30d (main_language);

-- 7d
CREATE INDEX IF NOT EXISTS embedding_cache_7d_projects ON embedding_cache_7d USING gin (projects);
CREATE INDEX IF NOT EXISTS embedding_cache_7d_publish_date ON embedding_cache_7d (publish_date);
CREATE INDEX IF NOT EXISTS embedding_cache_7d_content_id ON embedding_cache_7d (content_id);
CREATE INDEX IF NOT EXISTS embedding_cache_7d_language ON embedding_cache_7d (main_language);

-- UNIQUE index on id for incremental refresh (ON CONFLICT support)
-- This is CRITICAL for the incremental refresh strategy
CREATE UNIQUE INDEX IF NOT EXISTS embedding_cache_30d_id ON embedding_cache_30d(id);
CREATE UNIQUE INDEX IF NOT EXISTS embedding_cache_7d_id ON embedding_cache_7d(id);

-- ============================================================================
-- PART 3: Cache Refresh Functions (INCREMENTAL)
-- ============================================================================
-- These functions use incremental updates instead of TRUNCATE + full rebuild.
-- This avoids the expensive HNSW index rebuild that takes 50+ minutes.
--
-- Strategy:
-- 1. DELETE rows where publish_date has aged out of the time window
-- 2. INSERT new rows from content updated in the last 12 hours
-- 3. ON CONFLICT (id) DO NOTHING to skip existing rows
--
-- The 12-hour lookback provides buffer for the 6-8 hour refresh cycle.

-- Incremental refresh for 30d cache
CREATE OR REPLACE FUNCTION refresh_embedding_cache_30d()
RETURNS void AS $$
DECLARE
    deleted_count INTEGER;
    inserted_count INTEGER;
BEGIN
    -- Delete rows that have aged out (publish_date now older than 30 days)
    DELETE FROM embedding_cache_30d
    WHERE publish_date < NOW() - INTERVAL '30 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Insert new rows from content updated in the last 12 hours
    -- (covers 6-8 hour refresh cycle with buffer)
    INSERT INTO embedding_cache_30d
    SELECT
      es.id, es.content_id, es.embedding, es.embedding_alt,
      c.projects, c.publish_date, c.channel_url, c.channel_name, c.title, c.main_language,
      es.source_speaker_hashes, es.text, es.start_time, es.end_time,
      es.segment_index, es.content_id_string, es.stitch_version
    FROM embedding_segments es
    JOIN content c ON es.content_id = c.id
    WHERE c.publish_date >= NOW() - INTERVAL '30 days'
      AND c.last_updated >= NOW() - INTERVAL '12 hours'
      AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL)
    ON CONFLICT (id) DO NOTHING;
    GET DIAGNOSTICS inserted_count = ROW_COUNT;

    RAISE NOTICE '30d cache: deleted % aged rows, inserted % new rows', deleted_count, inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Incremental refresh for 7d cache
CREATE OR REPLACE FUNCTION refresh_embedding_cache_7d()
RETURNS void AS $$
DECLARE
    deleted_count INTEGER;
    inserted_count INTEGER;
BEGIN
    -- Delete rows that have aged out (publish_date now older than 7 days)
    DELETE FROM embedding_cache_7d
    WHERE publish_date < NOW() - INTERVAL '7 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Insert new rows from content updated in the last 12 hours
    INSERT INTO embedding_cache_7d
    SELECT
      es.id, es.content_id, es.embedding, es.embedding_alt,
      c.projects, c.publish_date, c.channel_url, c.channel_name, c.title, c.main_language,
      es.source_speaker_hashes, es.text, es.start_time, es.end_time,
      es.segment_index, es.content_id_string, es.stitch_version
    FROM embedding_segments es
    JOIN content c ON es.content_id = c.id
    WHERE c.publish_date >= NOW() - INTERVAL '7 days'
      AND c.last_updated >= NOW() - INTERVAL '12 hours'
      AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL)
    ON CONFLICT (id) DO NOTHING;
    GET DIAGNOSTICS inserted_count = ROW_COUNT;

    RAISE NOTICE '7d cache: deleted % aged rows, inserted % new rows', deleted_count, inserted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 4: Cache Refresh Schedule (pg_cron)
-- ============================================================================

-- Remove existing cache refresh jobs
DO $$
BEGIN
    -- Clean up any old job names
    PERFORM cron.unschedule(jobname) FROM cron.job
    WHERE jobname IN (
        'refresh-180d', 'refresh-180d-main', 'refresh-180d-alt',
        'refresh-30d', 'refresh-30d-main', 'refresh-30d-alt',
        'refresh-7d', 'refresh-7d-main', 'refresh-7d-alt',
        'reconcile-180d-main-weekly', 'reconcile-180d-alt-weekly'
    );
EXCEPTION WHEN OTHERS THEN
    -- Ignore errors if jobs don't exist
    NULL;
END $$;

-- Schedule 30-day cache refresh (every 6 hours at 2:30, 8:30, 14:30, 20:30)
SELECT cron.schedule(
  'refresh-30d',
  '30 2,8,14,20 * * *',
  'SELECT refresh_embedding_cache_30d()'
);

-- Schedule 7-day cache refresh (hourly at :50)
SELECT cron.schedule(
  'refresh-7d',
  '50 * * * *',
  'SELECT refresh_embedding_cache_7d()'
);

-- ============================================================================
-- PART 5: Clustering Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS embedding_clusters (
    id SERIAL PRIMARY KEY,

    -- Segment reference
    segment_id INTEGER NOT NULL,
    content_id INTEGER,
    content_id_string VARCHAR,

    -- Clustering metadata
    time_window VARCHAR(10) NOT NULL,  -- '30d', '7d'
    model_version VARCHAR(20) NOT NULL,  -- 'main' (0.6b) or 'alt' (4b)
    cluster_id INTEGER NOT NULL,  -- -1 = noise/unclustered
    cluster_name VARCHAR(255),
    cluster_size INTEGER,  -- number of segments in this cluster

    -- Cluster statistics
    cluster_centroid VECTOR(1024),  -- mean embedding of cluster
    distance_to_centroid FLOAT,  -- distance of this segment to cluster center
    is_representative BOOLEAN DEFAULT FALSE,  -- is this segment a good exemplar?

    -- Filters (denormalized for fast querying)
    projects VARCHAR[],
    main_language VARCHAR(10),
    publish_date TIMESTAMP WITH TIME ZONE,

    -- Clustering run metadata
    clustering_method VARCHAR(50) DEFAULT 'hdbscan',
    clustering_params JSONB,  -- stores min_cluster_size, silhouette_score, etc.
    clustered_at TIMESTAMP DEFAULT NOW(),

    -- Unique constraint: one cluster assignment per segment per time window
    UNIQUE(segment_id, time_window, model_version, clustered_at)
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_clusters_time_window ON embedding_clusters(time_window);
CREATE INDEX IF NOT EXISTS idx_clusters_model ON embedding_clusters(model_version);
CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id ON embedding_clusters(cluster_id);
CREATE INDEX IF NOT EXISTS idx_clusters_projects ON embedding_clusters USING gin(projects);
CREATE INDEX IF NOT EXISTS idx_clusters_language ON embedding_clusters(main_language);
CREATE INDEX IF NOT EXISTS idx_clusters_publish_date ON embedding_clusters(publish_date);
CREATE INDEX IF NOT EXISTS idx_clusters_segment ON embedding_clusters(segment_id);
CREATE INDEX IF NOT EXISTS idx_clusters_clustered_at ON embedding_clusters(clustered_at DESC);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_clusters_window_model_cluster ON embedding_clusters(time_window, model_version, cluster_id);
CREATE INDEX IF NOT EXISTS idx_clusters_latest ON embedding_clusters(time_window, model_version, clustered_at DESC);

-- ============================================================================
-- PART 6: Cluster Metadata Summary Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS cluster_metadata (
    id SERIAL PRIMARY KEY,

    -- Clustering run identifier
    time_window VARCHAR(10) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    clustered_at TIMESTAMP DEFAULT NOW(),

    -- Cluster info
    cluster_id INTEGER NOT NULL,
    cluster_name VARCHAR(255),
    cluster_size INTEGER,

    -- Statistics
    avg_distance_to_centroid FLOAT,
    silhouette_score FLOAT,
    cluster_centroid VECTOR(1024),

    -- Representative segments (for naming/display)
    representative_segment_ids INTEGER[],

    -- Filters applied during clustering
    projects_filter VARCHAR[],
    languages_filter VARCHAR[],

    -- Clustering parameters
    clustering_method VARCHAR(50) DEFAULT 'hdbscan',
    clustering_params JSONB,

    UNIQUE(time_window, model_version, clustered_at, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_cluster_meta_window ON cluster_metadata(time_window);
CREATE INDEX IF NOT EXISTS idx_cluster_meta_latest ON cluster_metadata(time_window, model_version, clustered_at DESC);

-- ============================================================================
-- PART 7: Helper Functions
-- ============================================================================

-- Function to get latest clusters for a time window
CREATE OR REPLACE FUNCTION get_latest_clusters(
    p_time_window VARCHAR,
    p_model VARCHAR DEFAULT 'main',
    p_projects VARCHAR[] DEFAULT NULL,
    p_languages VARCHAR[] DEFAULT NULL
)
RETURNS TABLE(
    cluster_id INTEGER,
    cluster_name VARCHAR,
    cluster_size INTEGER,
    segment_ids INTEGER[],
    cluster_centroid VECTOR,
    clustered_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    WITH latest_run AS (
        SELECT MAX(ec.clustered_at) as max_time
        FROM embedding_clusters ec
        WHERE ec.time_window = p_time_window
        AND ec.model_version = p_model
    )
    SELECT
        ec.cluster_id,
        ec.cluster_name,
        ec.cluster_size,
        array_agg(ec.segment_id ORDER BY ec.is_representative DESC, ec.distance_to_centroid ASC) as segment_ids,
        ec.cluster_centroid,
        ec.clustered_at
    FROM embedding_clusters ec
    CROSS JOIN latest_run
    WHERE ec.time_window = p_time_window
    AND ec.model_version = p_model
    AND ec.clustered_at = latest_run.max_time
    AND ec.cluster_id >= 0  -- exclude noise points
    AND (p_projects IS NULL OR ec.projects && p_projects)
    AND (p_languages IS NULL OR ec.main_language = ANY(p_languages))
    GROUP BY ec.cluster_id, ec.cluster_name, ec.cluster_size, ec.cluster_centroid, ec.clustered_at
    ORDER BY ec.cluster_size DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old clustering runs (keep last 7 days)
CREATE OR REPLACE FUNCTION cleanup_old_clusters()
RETURNS TABLE(
    deleted_count INTEGER,
    oldest_kept TIMESTAMP
) AS $$
DECLARE
    v_deleted_count INTEGER;
    v_oldest_kept TIMESTAMP;
BEGIN
    -- Delete clusters older than 7 days
    DELETE FROM embedding_clusters
    WHERE clustered_at < NOW() - INTERVAL '7 days';

    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;

    -- Get oldest remaining timestamp
    SELECT MIN(clustered_at) INTO v_oldest_kept
    FROM embedding_clusters;

    -- Delete old metadata
    DELETE FROM cluster_metadata
    WHERE clustered_at < NOW() - INTERVAL '7 days';

    RETURN QUERY SELECT v_deleted_count, v_oldest_kept;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 8: pg_cron Clustering Schedule
-- ============================================================================

-- Remove existing clustering jobs if they exist
DO $$
BEGIN
    PERFORM cron.unschedule(jobname) FROM cron.job
    WHERE jobname IN (
        'cluster-30d-main', 'cluster-30d-alt',
        'cluster-7d-main', 'cluster-7d-alt',
        'cleanup-old-clusters'
    );
EXCEPTION WHEN OTHERS THEN
    NULL;
END $$;

-- Schedule 30d clustering (daily at 3:00 AM, after cache refresh at 2:30 AM)
SELECT cron.schedule(
    'cluster-30d-main',
    '0 3 * * *',
    $$python3 /Users/signal4/content_processing/src/backend/scripts/generate_embedding_clusters.py --time-window 30d --model main$$
);

SELECT cron.schedule(
    'cluster-30d-alt',
    '10 3 * * *',
    $$python3 /Users/signal4/content_processing/src/backend/scripts/generate_embedding_clusters.py --time-window 30d --model alt$$
);

-- Schedule 7d clustering (every 2 hours at :00, after cache refresh at :50)
SELECT cron.schedule(
    'cluster-7d-main',
    '0 */2 * * *',
    $$python3 /Users/signal4/content_processing/src/backend/scripts/generate_embedding_clusters.py --time-window 7d --model main$$
);

SELECT cron.schedule(
    'cluster-7d-alt',
    '10 */2 * * *',
    $$python3 /Users/signal4/content_processing/src/backend/scripts/generate_embedding_clusters.py --time-window 7d --model alt$$
);

-- Schedule cleanup (daily at 4 AM)
SELECT cron.schedule(
    'cleanup-old-clusters',
    '0 4 * * *',
    'SELECT cleanup_old_clusters()'
);

-- ============================================================================
-- PART 9: Monitoring Views
-- ============================================================================

CREATE OR REPLACE VIEW cluster_status AS
SELECT
    time_window,
    model_version,
    MAX(clustered_at) as last_clustered,
    NOW() - MAX(clustered_at) as age,
    COUNT(DISTINCT cluster_id) as num_clusters,
    COUNT(*) as total_segments,
    COUNT(*) FILTER (WHERE cluster_id >= 0) as clustered_segments,
    COUNT(*) FILTER (WHERE cluster_id = -1) as noise_segments,
    ROUND(100.0 * COUNT(*) FILTER (WHERE cluster_id >= 0) / NULLIF(COUNT(*), 0), 1) as clustered_percentage
FROM embedding_clusters
GROUP BY time_window, model_version
ORDER BY time_window, model_version;

-- ============================================================================
-- PART 10: Verification
-- ============================================================================

-- Check cache tables
SELECT 'embedding_cache_30d' as table_name, COUNT(*) as row_count FROM embedding_cache_30d
UNION ALL
SELECT 'embedding_cache_7d', COUNT(*) FROM embedding_cache_7d
UNION ALL
SELECT 'embedding_clusters', COUNT(*) FROM embedding_clusters
UNION ALL
SELECT 'cluster_metadata', COUNT(*) FROM cluster_metadata;

-- Check all scheduled jobs
SELECT
    jobid,
    jobname,
    schedule,
    active,
    command
FROM cron.job
WHERE jobname LIKE '%refresh%' OR jobname LIKE '%cluster%' OR jobname LIKE '%cleanup%'
ORDER BY jobname;

-- Show cluster status (will be empty until first clustering run)
SELECT * FROM cluster_status;

-- Show helper functions
SELECT
    proname as function_name,
    pg_get_function_identity_arguments(oid) as arguments
FROM pg_proc
WHERE proname IN ('get_latest_clusters', 'cleanup_old_clusters', 'refresh_embedding_cache_30d', 'refresh_embedding_cache_7d')
ORDER BY proname;

-- ============================================================================
-- Migration Complete
-- ============================================================================
--
-- Created:
-- - 2 embedding cache tables (30d, 7d) with BOTH embeddings (main + alt)
-- - 4 HNSW vector indexes (2 per cache table, one for each embedding model)
-- - 2 UNIQUE indexes on id for incremental refresh support
-- - Supporting indexes (GIN on projects, B-tree on dates/content_id/language)
-- - 2 pg_cron jobs for cache refresh (incremental)
-- - 2 clustering tables (embedding_clusters, cluster_metadata)
-- - 4 pg_cron jobs for automated clustering
-- - 1 pg_cron job for cleanup
-- - Helper functions and monitoring views
--
-- HNSW Index Parameters:
-- - m=16: connections per node
-- - ef_construction=64: build-time quality
-- - hnsw.ef_search=100: query-time search width (set at query time)
--
-- INCREMENTAL REFRESH STRATEGY:
-- - DELETE aged rows (uses publish_date index, fast)
-- - INSERT new rows from last 12 hours (uses last_updated index)
-- - ON CONFLICT (id) DO NOTHING (uses unique id index)
-- - Typical run: ~50ms when no changes, 3-4 min when inserting new content
-- - Old truncate method: 50+ minutes (full HNSW rebuild)
--
-- Schedule Summary:
-- - 30d cache: Every 6 hours at 2:30, 8:30, 14:30, 20:30
-- - 7d cache: Hourly at :50
-- - 30d clustering: Daily at 3:00 AM (main) / 3:10 AM (alt)
-- - 7d clustering: Every 2 hours (main) / +10min (alt)
-- - Cleanup: Daily at 4:00 AM
--
-- Next steps:
-- 1. Generate initial clusters manually:
--    python3 /Users/signal4/content_processing/src/backend/scripts/generate_embedding_clusters.py --time-window 30d --model main
--    python3 /Users/signal4/content_processing/src/backend/scripts/generate_embedding_clusters.py --time-window 7d --model main
--
-- 2. Verify setup:
--    SELECT * FROM cluster_status;
--
-- 3. Query clusters:
--    SELECT * FROM get_latest_clusters('30d', 'main');
--
-- 4. Test HNSW performance:
--    SET hnsw.ef_search = 100;
--    EXPLAIN ANALYZE SELECT ... ORDER BY embedding <=> query_vector LIMIT 200;
--
-- ============================================================================
