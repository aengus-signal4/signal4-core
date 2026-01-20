-- =============================================================================
-- Embedding Cache Refresh Functions
-- =============================================================================
--
-- These functions maintain the embedding_cache_7d and embedding_cache_30d tables,
-- which are materialized views of embedding_segments joined with content metadata.
-- The caches enable fast vector similarity search with HNSW indexes.
--
-- Called by the orchestrator's scheduled tasks:
--   - cache_refresh_7d: Hourly via scheduled_tasks.cache_refresh_7d
--   - cache_refresh_30d: 4x daily via scheduled_tasks.cache_refresh_30d
--
-- Key features:
--   1. Aging: Removes rows older than the cache window (7 or 30 days)
--   2. Smart backfill: Detects dates with >10% missing data and backfills them
--   3. Incremental: Adds recently updated content (last 2-12 hours)
--
-- The backfill logic ensures the cache self-heals if the orchestrator is down
-- for an extended period - gaps are automatically detected and filled on the
-- next scheduled run.
--
-- Source tables:
--   - embedding_segments: Text segments with vector embeddings
--   - content: Metadata (projects, publish_date, channel, title, language)
--
-- Cache tables have HNSW indexes per project for fast similarity search.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- refresh_embedding_cache_30d()
-- -----------------------------------------------------------------------------
-- Maintains the 30-day embedding cache.
--
-- Schedule: 4x daily (02:30, 08:30, 14:30, 20:30) via orchestrator
-- Timeout: 1800 seconds (30 minutes)
--
-- Steps:
--   1. Delete rows with publish_date older than 30 days
--   2. Detect gaps: dates where cache has <90% of source rows
--   3. Backfill any detected gaps
--   4. Incremental insert for content updated in last 12 hours
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.refresh_embedding_cache_30d()
RETURNS void
LANGUAGE plpgsql
AS $function$
DECLARE
    deleted_count INTEGER;
    inserted_count INTEGER;
    backfill_count INTEGER := 0;
    gap_rec RECORD;
BEGIN
    -- Step 1: Delete rows that have aged out
    DELETE FROM embedding_cache_30d
    WHERE publish_date < NOW() - INTERVAL '30 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Step 2: Detect gaps - dates with >10% missing segments
    FOR gap_rec IN
        WITH source_counts AS (
            SELECT
                DATE(c.publish_date) as dt,
                COUNT(*) as cnt
            FROM embedding_segments es
            JOIN content c ON es.content_id = c.id
            WHERE c.publish_date >= NOW() - INTERVAL '30 days'
              AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL)
            GROUP BY DATE(c.publish_date)
        ),
        cache_counts AS (
            SELECT DATE(publish_date) as dt, COUNT(*) as cnt
            FROM embedding_cache_30d
            GROUP BY DATE(publish_date)
        )
        SELECT s.dt as gap_date, s.cnt as source_cnt, COALESCE(c.cnt, 0) as cache_cnt
        FROM source_counts s
        LEFT JOIN cache_counts c ON s.dt = c.dt
        WHERE COALESCE(c.cnt, 0) < s.cnt * 0.9
    LOOP
        -- Backfill this date
        INSERT INTO embedding_cache_30d
        SELECT
            es.id, es.content_id, es.embedding, es.embedding_alt,
            c.projects, c.publish_date, c.channel_url, c.channel_name, c.title, c.main_language,
            es.source_speaker_hashes, es.text, es.start_time, es.end_time,
            es.segment_index, es.content_id_string, es.stitch_version
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE DATE(c.publish_date) = gap_rec.gap_date
          AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL)
        ON CONFLICT (id) DO NOTHING;

        GET DIAGNOSTICS inserted_count = ROW_COUNT;
        backfill_count := backfill_count + inserted_count;

        IF inserted_count > 0 THEN
            RAISE NOTICE '30d cache: backfilled % rows for %', inserted_count, gap_rec.gap_date;
        END IF;
    END LOOP;

    -- Step 3: Normal incremental update for recently updated content
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

    RAISE NOTICE '30d cache: deleted % aged, backfilled %, incremental %', deleted_count, backfill_count, inserted_count;
END;
$function$;


-- -----------------------------------------------------------------------------
-- refresh_embedding_cache_7d()
-- -----------------------------------------------------------------------------
-- Maintains the 7-day embedding cache.
--
-- Schedule: Hourly via orchestrator
-- Timeout: 600 seconds (10 minutes)
--
-- Steps:
--   1. Delete rows with publish_date older than 7 days
--   2. Detect gaps: dates where cache has <90% of source rows
--   3. Backfill any detected gaps
--   4. Incremental insert for content updated in last 2 hours
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.refresh_embedding_cache_7d()
RETURNS void
LANGUAGE plpgsql
AS $function$
DECLARE
    deleted_count INTEGER;
    inserted_count INTEGER;
    backfill_count INTEGER := 0;
    gap_rec RECORD;
BEGIN
    -- Step 1: Delete rows that have aged out
    DELETE FROM embedding_cache_7d
    WHERE publish_date < NOW() - INTERVAL '7 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Step 2: Detect gaps - dates with >10% missing segments
    FOR gap_rec IN
        WITH source_counts AS (
            SELECT
                DATE(c.publish_date) as dt,
                COUNT(*) as cnt
            FROM embedding_segments es
            JOIN content c ON es.content_id = c.id
            WHERE c.publish_date >= NOW() - INTERVAL '7 days'
              AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL)
            GROUP BY DATE(c.publish_date)
        ),
        cache_counts AS (
            SELECT DATE(publish_date) as dt, COUNT(*) as cnt
            FROM embedding_cache_7d
            GROUP BY DATE(publish_date)
        )
        SELECT s.dt as gap_date, s.cnt as source_cnt, COALESCE(c.cnt, 0) as cache_cnt
        FROM source_counts s
        LEFT JOIN cache_counts c ON s.dt = c.dt
        WHERE COALESCE(c.cnt, 0) < s.cnt * 0.9
    LOOP
        -- Backfill this date
        INSERT INTO embedding_cache_7d
        SELECT
            es.id, es.content_id, es.embedding, es.embedding_alt,
            c.projects, c.publish_date, c.channel_url, c.channel_name, c.title, c.main_language,
            es.source_speaker_hashes, es.text, es.start_time, es.end_time,
            es.segment_index, es.content_id_string, es.stitch_version
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE DATE(c.publish_date) = gap_rec.gap_date
          AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL)
        ON CONFLICT (id) DO NOTHING;

        GET DIAGNOSTICS inserted_count = ROW_COUNT;
        backfill_count := backfill_count + inserted_count;

        IF inserted_count > 0 THEN
            RAISE NOTICE '7d cache: backfilled % rows for %', inserted_count, gap_rec.gap_date;
        END IF;
    END LOOP;

    -- Step 3: Normal incremental update for recently updated content
    INSERT INTO embedding_cache_7d
    SELECT
        es.id, es.content_id, es.embedding, es.embedding_alt,
        c.projects, c.publish_date, c.channel_url, c.channel_name, c.title, c.main_language,
        es.source_speaker_hashes, es.text, es.start_time, es.end_time,
        es.segment_index, es.content_id_string, es.stitch_version
    FROM embedding_segments es
    JOIN content c ON es.content_id = c.id
    WHERE c.publish_date >= NOW() - INTERVAL '7 days'
      AND c.last_updated >= NOW() - INTERVAL '2 hours'
      AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL)
    ON CONFLICT (id) DO NOTHING;
    GET DIAGNOSTICS inserted_count = ROW_COUNT;

    RAISE NOTICE '7d cache: deleted % aged, backfilled %, incremental %', deleted_count, backfill_count, inserted_count;
END;
$function$;


-- =============================================================================
-- Cache Table Schema (for reference)
-- =============================================================================
-- The cache tables mirror embedding_segments with denormalized content fields:
--
-- CREATE TABLE embedding_cache_30d (
--     id INTEGER PRIMARY KEY,              -- embedding_segments.id
--     content_id INTEGER,                  -- FK to content
--     embedding vector(1024),              -- Main embedding (nomic-embed-text)
--     embedding_alt vector(1024),          -- Alt embedding (future use)
--     projects VARCHAR[],                  -- From content.projects
--     publish_date TIMESTAMPTZ,            -- From content.publish_date
--     channel_url VARCHAR,                 -- From content.channel_url
--     channel_name VARCHAR,                -- From content.channel_name
--     title VARCHAR,                       -- From content.title
--     main_language VARCHAR,               -- From content.main_language
--     source_speaker_hashes TEXT[],        -- Speaker hashes for filtering
--     text TEXT,                           -- Segment text
--     start_time FLOAT,                    -- Segment start time
--     end_time FLOAT,                      -- Segment end time
--     segment_index INTEGER,               -- Segment index within content
--     content_id_string VARCHAR,           -- content.content_id (string form)
--     stitch_version VARCHAR               -- Stitching algorithm version
-- );
--
-- Indexes (per-project HNSW for fast similarity search):
--   - embedding_cache_30d_hnsw_canadian (WHERE projects && ARRAY['Canadian'])
--   - embedding_cache_30d_hnsw_health (WHERE projects && ARRAY['Health'])
--   - embedding_cache_30d_hnsw_finance (WHERE projects && ARRAY['Finance'])
--   - embedding_cache_30d_hnsw_europe (WHERE projects && ARRAY['Europe'])
--   - embedding_cache_30d_hnsw_cprmv (WHERE projects && ARRAY['CPRMV'])
--   - embedding_cache_30d_hnsw_anglosphere (WHERE projects && ARRAY['Anglosphere'])
--   - embedding_cache_30d_hnsw_big_channels (WHERE projects && ARRAY['Big_Channels'])
--
-- Same structure for embedding_cache_7d.
-- =============================================================================


-- =============================================================================
-- Manual Full Rebuild (if needed)
-- =============================================================================
-- If the cache gets severely out of sync, use this procedure:
--
-- 1. Drop HNSW indexes (they slow bulk inserts):
--    DROP INDEX IF EXISTS embedding_cache_30d_hnsw_canadian;
--    DROP INDEX IF EXISTS embedding_cache_30d_hnsw_health;
--    ... (all HNSW indexes)
--
-- 2. Truncate and reload:
--    TRUNCATE embedding_cache_30d;
--    INSERT INTO embedding_cache_30d
--    SELECT es.id, es.content_id, es.embedding, es.embedding_alt,
--           c.projects, c.publish_date, c.channel_url, c.channel_name,
--           c.title, c.main_language, es.source_speaker_hashes, es.text,
--           es.start_time, es.end_time, es.segment_index, es.content_id_string,
--           es.stitch_version
--    FROM embedding_segments es
--    JOIN content c ON es.content_id = c.id
--    WHERE c.publish_date >= NOW() - INTERVAL '30 days'
--      AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL);
--
-- 3. Recreate HNSW indexes:
--    CREATE INDEX CONCURRENTLY embedding_cache_30d_hnsw_canadian
--        ON embedding_cache_30d USING hnsw (embedding vector_cosine_ops)
--        WITH (m='16', ef_construction='64')
--        WHERE (projects && ARRAY['Canadian']::varchar[]);
--    ... (all HNSW indexes)
-- =============================================================================
