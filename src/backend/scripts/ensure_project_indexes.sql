-- ============================================================================
-- Per-Project Partial HNSW Indexes
-- ============================================================================
--
-- Creates partial HNSW indexes for each project on the embedding_cache_30d
-- and embedding_cache_7d tables. These indexes enable fast (~8ms) semantic
-- search with project filters, compared to ~500ms with the full index + filter.
--
-- The search service queries each project's index separately, then combines
-- and reranks the results.
--
-- Uses only the 0.6B embedding (embedding column) to simplify.
--
-- ============================================================================

-- Function to ensure partial HNSW indexes exist for all active projects
CREATE OR REPLACE FUNCTION ensure_project_hnsw_indexes()
RETURNS TABLE(
    index_name TEXT,
    table_name TEXT,
    project TEXT,
    status TEXT
) AS $$
DECLARE
    proj TEXT;
    idx_name_30d TEXT;
    idx_name_7d TEXT;
    projects_list TEXT[] := ARRAY[
        'Canadian', 'Big_Channels', 'Health', 'Finance',
        'Europe', 'CPRMV', 'Anglosphere'
    ];
BEGIN
    FOREACH proj IN ARRAY projects_list
    LOOP
        -- 30d index
        idx_name_30d := 'embedding_cache_30d_hnsw_' || lower(proj);

        -- Check if 30d index exists
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes WHERE indexname = idx_name_30d
        ) THEN
            -- Create index (non-concurrent for function context)
            EXECUTE format(
                'CREATE INDEX %I ON embedding_cache_30d
                 USING hnsw (embedding vector_cosine_ops)
                 WITH (m = 16, ef_construction = 64)
                 WHERE (projects && ARRAY[%L]::varchar[])',
                idx_name_30d, proj
            );
            RETURN QUERY SELECT idx_name_30d, 'embedding_cache_30d'::TEXT, proj, 'CREATED'::TEXT;
        ELSE
            RETURN QUERY SELECT idx_name_30d, 'embedding_cache_30d'::TEXT, proj, 'EXISTS'::TEXT;
        END IF;

        -- 7d index
        idx_name_7d := 'embedding_cache_7d_hnsw_' || lower(proj);

        -- Check if 7d index exists
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes WHERE indexname = idx_name_7d
        ) THEN
            EXECUTE format(
                'CREATE INDEX %I ON embedding_cache_7d
                 USING hnsw (embedding vector_cosine_ops)
                 WITH (m = 16, ef_construction = 64)
                 WHERE (projects && ARRAY[%L]::varchar[])',
                idx_name_7d, proj
            );
            RETURN QUERY SELECT idx_name_7d, 'embedding_cache_7d'::TEXT, proj, 'CREATED'::TEXT;
        ELSE
            RETURN QUERY SELECT idx_name_7d, 'embedding_cache_7d'::TEXT, proj, 'EXISTS'::TEXT;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to list all per-project indexes with sizes
CREATE OR REPLACE FUNCTION list_project_hnsw_indexes()
RETURNS TABLE(
    index_name TEXT,
    table_name TEXT,
    project TEXT,
    index_size TEXT,
    row_estimate BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.indexrelid::regclass::TEXT as index_name,
        i.indrelid::regclass::TEXT as table_name,
        -- Extract project from WHERE clause
        CASE
            WHEN pg_get_indexdef(i.indexrelid) LIKE '%Canadian%' THEN 'Canadian'
            WHEN pg_get_indexdef(i.indexrelid) LIKE '%Big_Channels%' THEN 'Big_Channels'
            WHEN pg_get_indexdef(i.indexrelid) LIKE '%Health%' THEN 'Health'
            WHEN pg_get_indexdef(i.indexrelid) LIKE '%Finance%' THEN 'Finance'
            WHEN pg_get_indexdef(i.indexrelid) LIKE '%Europe%' THEN 'Europe'
            WHEN pg_get_indexdef(i.indexrelid) LIKE '%CPRMV%' THEN 'CPRMV'
            WHEN pg_get_indexdef(i.indexrelid) LIKE '%Anglosphere%' THEN 'Anglosphere'
            ELSE 'FULL'
        END as project,
        pg_size_pretty(pg_relation_size(i.indexrelid)) as index_size,
        c.reltuples::BIGINT as row_estimate
    FROM pg_index i
    JOIN pg_class c ON c.oid = i.indexrelid
    WHERE i.indrelid::regclass::TEXT LIKE 'embedding_cache%'
    AND c.relname LIKE '%hnsw%'
    AND c.relname NOT LIKE '%alt%'  -- Only 0.6B indexes
    ORDER BY i.indrelid::regclass, project;
END;
$$ LANGUAGE plpgsql;

-- Run the function to ensure indexes exist
SELECT * FROM ensure_project_hnsw_indexes();

-- Show all indexes
SELECT * FROM list_project_hnsw_indexes();
