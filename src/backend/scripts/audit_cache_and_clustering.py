#!/usr/bin/env python3
"""
Comprehensive Audit: Cache & Clustering Infrastructure
=======================================================

Complete audit of the entire semantic search infrastructure:
- Embedding cache tables (180d, 30d, 7d)
- IVFFlat vector indexes
- Clustering tables and metadata
- pg_cron jobs for refresh and clustering
- Helper functions
- Data integrity checks
- Performance benchmarks

Usage:
    python audit_cache_and_clustering.py [--performance] [--detailed]

Options:
    --performance   Run performance benchmarks (adds ~30 seconds)
    --detailed      Show detailed SQL and index definitions
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Database connection - use centralized config
def _get_db_config():
    """Get database config from environment variables."""
    from dotenv import load_dotenv
    from src.utils.paths import get_env_path
    env_path = get_env_path()
    if env_path.exists():
        load_dotenv(env_path)

    password = os.getenv('POSTGRES_PASSWORD')
    if not password:
        raise ValueError("POSTGRES_PASSWORD environment variable is required")

    return {
        'host': os.getenv('POSTGRES_HOST', '10.0.0.4'),
        'database': os.getenv('POSTGRES_DB', 'av_content'),
        'user': os.getenv('POSTGRES_USER', 'signal4'),
        'password': password,
    }

DB_CONFIG = _get_db_config()

CACHE_TABLES = [
    'embedding_cache_180d',
    'embedding_cache_30d',
    'embedding_cache_7d'
]

CLUSTERING_TABLES = [
    'embedding_clusters',
    'cluster_metadata'
]


def get_connection():
    """Get database connection."""
    return psycopg2.connect(**DB_CONFIG)


def print_header(title):
    """Print formatted section header."""
    logger.info("\n" + "=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def check_parent_tables(conn):
    """Verify parent tables exist and have data."""
    print_header("PARENT TABLES (Source Data)")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check embedding_segments
        cur.execute("""
            SELECT
                COUNT(*) as total_segments,
                COUNT(DISTINCT content_id) as unique_content,
                COUNT(embedding) as with_main_embedding,
                COUNT(embedding_alt) as with_alt_embedding,
                MIN(id) as min_id,
                MAX(id) as max_id
            FROM embedding_segments
        """)
        es_stats = cur.fetchone()

        logger.info("\nembedding_segments:")
        logger.info(f"  Total segments:      {es_stats['total_segments']:>12,}")
        logger.info(f"  Unique content:      {es_stats['unique_content']:>12,}")
        logger.info(f"  With main embedding: {es_stats['with_main_embedding']:>12,} ({100*es_stats['with_main_embedding']/es_stats['total_segments']:.1f}%)")
        logger.info(f"  With alt embedding:  {es_stats['with_alt_embedding']:>12,} ({100*es_stats['with_alt_embedding']/es_stats['total_segments']:.1f}%)")
        logger.info(f"  ID range:            {es_stats['min_id']:>12,} to {es_stats['max_id']:,}")

        # Check content
        cur.execute("""
            SELECT
                COUNT(*) as total_content,
                MIN(publish_date) as oldest_date,
                MAX(publish_date) as newest_date,
                COUNT(DISTINCT projects[1]) as unique_projects
            FROM content
            WHERE publish_date IS NOT NULL
        """)
        content_stats = cur.fetchone()

        age_days = (datetime.now().date() - content_stats['oldest_date'].date()).days

        logger.info("\ncontent:")
        logger.info(f"  Total content:  {content_stats['total_content']:>12,}")
        logger.info(f"  Date range:     {content_stats['oldest_date'].date()} to {content_stats['newest_date'].date()} ({age_days} days)")
        logger.info(f"  Projects:       {content_stats['unique_projects']:>12,}")


def check_cache_tables(conn):
    """Check cache table status."""
    print_header("CACHE TABLES (Derived Data)")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for table in CACHE_TABLES:
            # Check if table exists
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{table}'
                )
            """)

            if not cur.fetchone()['exists']:
                logger.error(f"\n{table}: ✗ TABLE DOES NOT EXIST")
                continue

            # Get table stats including both embeddings
            cur.execute(f"""
                SELECT
                    COUNT(*) as row_count,
                    COUNT(embedding) as with_main_embedding,
                    COUNT(embedding_alt) as with_alt_embedding,
                    COUNT(DISTINCT content_id) as unique_content,
                    COUNT(DISTINCT projects[1]) as unique_projects,
                    COUNT(DISTINCT main_language) as unique_languages,
                    MIN(publish_date) as oldest_date,
                    MAX(publish_date) as newest_date,
                    pg_size_pretty(pg_total_relation_size('{table}')) as total_size
                FROM {table}
            """)
            stats = cur.fetchone()

            if stats['row_count'] == 0:
                logger.warning(f"\n{table}:")
                logger.warning("  ⚠ EMPTY TABLE")
                continue

            age_days = (datetime.now().date() - stats['oldest_date'].date()).days
            expected_days = 180 if '180d' in table else (30 if '30d' in table else 7)

            main_pct = 100 * stats['with_main_embedding'] / stats['row_count'] if stats['row_count'] > 0 else 0
            alt_pct = 100 * stats['with_alt_embedding'] / stats['row_count'] if stats['row_count'] > 0 else 0

            logger.info(f"\n{table}:")
            logger.info(f"  ✓ Rows:         {stats['row_count']:>12,}")
            logger.info(f"    Main embeddings:  {stats['with_main_embedding']:>12,} ({main_pct:.1f}%)")
            logger.info(f"    Alt embeddings:   {stats['with_alt_embedding']:>12,} ({alt_pct:.1f}%)")
            logger.info(f"    Content:      {stats['unique_content']:>12,} unique")
            logger.info(f"    Projects:     {stats['unique_projects']:>12,} unique")
            logger.info(f"    Languages:    {stats['unique_languages']:>12,} unique")
            logger.info(f"    Date Range:   {stats['oldest_date'].date()} to {stats['newest_date'].date()} ({age_days} days)")
            logger.info(f"    Size:         {stats['total_size']:>12s}")

            # Warn if date range doesn't match expected window
            if age_days > expected_days + 2:
                logger.warning(f"  ⚠ Date range ({age_days} days) exceeds expected {expected_days} days - may need refresh")


def check_indexes(conn, detailed=False):
    """Check vector and supporting indexes."""
    print_header("INDEXES")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for table in CACHE_TABLES:
            # Get all indexes for this table
            cur.execute("""
                SELECT
                    i.indexname,
                    i.indexdef,
                    pg_size_pretty(pg_relation_size(c.oid)) as index_size
                FROM pg_indexes i
                JOIN pg_class c ON c.relname = i.indexname
                WHERE i.schemaname = 'public'
                AND i.tablename = %s
                ORDER BY i.indexname
            """, (table,))

            indexes = cur.fetchall()

            logger.info(f"\n{table}:")

            # Check for IVFFlat vector indexes (should have TWO: main and alt)
            vector_indexes = [idx for idx in indexes if 'ivfflat' in idx['indexname'].lower()]
            if not vector_indexes:
                logger.error("  ✗ NO IVFFLAT VECTOR INDEXES")
            elif len(vector_indexes) < 2:
                logger.warning(f"  ⚠ Only {len(vector_indexes)} IVFFlat index (expected 2: main + alt)")
                for idx in vector_indexes:
                    logger.info(f"    {idx['indexname']:48s} | {idx['index_size']:>10s}")
            else:
                logger.info(f"  ✓ Vector Indexes ({len(vector_indexes)}):")
                for idx in vector_indexes:
                    model_type = "main" if "main" in idx['indexname'] else ("alt" if "alt" in idx['indexname'] else "unknown")
                    logger.info(f"    {idx['indexname']:48s} | {idx['index_size']:>10s} ({model_type})")
                    if detailed:
                        logger.info(f"      {idx['indexdef']}")

            # Check for supporting indexes
            supporting = [idx for idx in indexes if 'ivfflat' not in idx['indexname'].lower()]
            expected_supporting = ['projects', 'publish_date', 'content_id', 'language']

            logger.info("  Supporting Indexes:")
            for expected in expected_supporting:
                found = any(expected in idx['indexname'] for idx in supporting)
                if found:
                    matching = [idx for idx in supporting if expected in idx['indexname']][0]
                    logger.info(f"    {matching['indexname']:48s} | {matching['index_size']:>10s}")
                else:
                    logger.warning(f"    ⚠ Missing index on {expected}")


def check_clustering_tables(conn):
    """Check clustering tables status."""
    print_header("CLUSTERING TABLES")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check embedding_clusters
        cur.execute("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT time_window) as time_windows,
                COUNT(DISTINCT model_version) as models,
                COUNT(DISTINCT cluster_id) as clusters,
                MAX(clustered_at) as last_clustered,
                NOW() - MAX(clustered_at) as age
            FROM embedding_clusters
        """)
        ec_stats = cur.fetchone()

        logger.info("\nembedding_clusters:")
        if ec_stats['total_rows'] == 0:
            logger.warning("  ⚠ EMPTY - No clusters generated yet")
            logger.info("  → Run: python3 generate_embedding_clusters.py --time-window 30d --model main")
        else:
            logger.info(f"  ✓ Total rows:      {ec_stats['total_rows']:>12,}")
            logger.info(f"    Time windows:    {ec_stats['time_windows']:>12,}")
            logger.info(f"    Models:          {ec_stats['models']:>12,}")
            logger.info(f"    Clusters:        {ec_stats['clusters']:>12,}")
            logger.info(f"    Last clustered:  {ec_stats['last_clustered']}")
            logger.info(f"    Age:             {ec_stats['age']}")

            # Check per time window
            cur.execute("""
                SELECT
                    time_window,
                    model_version,
                    MAX(clustered_at) as last_run,
                    COUNT(DISTINCT cluster_id) as num_clusters,
                    COUNT(*) as segments
                FROM embedding_clusters
                GROUP BY time_window, model_version
                ORDER BY time_window, model_version
            """)

            for row in cur.fetchall():
                age = datetime.now() - row['last_run']
                logger.info(f"\n  {row['time_window']}/{row['model_version']}:")
                logger.info(f"    Clusters:   {row['num_clusters']:>6,}")
                logger.info(f"    Segments:   {row['segments']:>6,}")
                logger.info(f"    Last run:   {row['last_run']} ({age})")

        # Check cluster_metadata
        cur.execute("""
            SELECT
                COUNT(*) as total_rows,
                MAX(clustered_at) as last_clustered
            FROM cluster_metadata
        """)
        cm_stats = cur.fetchone()

        logger.info("\ncluster_metadata:")
        if cm_stats['total_rows'] == 0:
            logger.warning("  ⚠ EMPTY")
        else:
            logger.info(f"  ✓ Total rows:      {cm_stats['total_rows']:>12,}")
            logger.info(f"    Last clustered:  {cm_stats['last_clustered']}")


def check_functions(conn):
    """Check helper functions exist."""
    print_header("HELPER FUNCTIONS")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        expected_functions = [
            'refresh_embedding_cache_180d',
            'refresh_embedding_cache_30d',
            'refresh_embedding_cache_7d',
            'get_latest_clusters',
            'cleanup_old_clusters'
        ]

        for func_name in expected_functions:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_proc
                    WHERE proname = %s
                )
            """, (func_name,))

            exists = cur.fetchone()['exists']
            if exists:
                logger.info(f"  ✓ {func_name}")
            else:
                logger.error(f"  ✗ {func_name} NOT FOUND")


def check_pg_cron_jobs(conn):
    """Check pg_cron scheduled jobs."""
    print_header("PG_CRON JOBS")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                jobid,
                jobname,
                schedule,
                active,
                command
            FROM cron.job
            WHERE jobname LIKE '%refresh%'
               OR jobname LIKE '%cluster%'
               OR jobname LIKE '%cleanup%'
            ORDER BY jobname
        """)

        jobs = cur.fetchall()

        if not jobs:
            logger.error("  ✗ NO CRON JOBS FOUND")
            return

        # Expected jobs
        expected_cache_jobs = [
            'refresh-180d',
            'refresh-30d',
            'refresh-7d'
        ]
        expected_cluster_jobs = [
            'cluster-30d-main', 'cluster-30d-alt',
            'cluster-7d-main', 'cluster-7d-alt',
            'cleanup-old-clusters'
        ]

        logger.info("\nCache Refresh Jobs (unified - both embeddings):")
        for job_name in expected_cache_jobs:
            job = next((j for j in jobs if j['jobname'] == job_name), None)
            if job:
                status = "✓ ACTIVE" if job['active'] else "✗ INACTIVE"
                logger.info(f"  {status:12s} | {job['jobname']:25s} | {job['schedule']:20s}")
            else:
                logger.error(f"  ✗ MISSING  | {job_name}")

        logger.info("\nClustering Jobs (separate for main/alt models):")
        for job_name in expected_cluster_jobs:
            job = next((j for j in jobs if j['jobname'] == job_name), None)
            if job:
                status = "✓ ACTIVE" if job['active'] else "✗ INACTIVE"
                logger.info(f"  {status:12s} | {job['jobname']:25s} | {job['schedule']:20s}")
            else:
                logger.error(f"  ✗ MISSING  | {job_name}")


def check_data_integrity(conn):
    """Verify data consistency between parent and cache tables."""
    print_header("DATA INTEGRITY CHECKS")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check 180d cache vs embedding_segments (should match rows with either embedding)
        cur.execute("""
            WITH source_count AS (
                SELECT COUNT(*) as expected
                FROM embedding_segments es
                JOIN content c ON es.content_id = c.id
                WHERE c.publish_date >= NOW() - INTERVAL '180 days'
                  AND (es.embedding IS NOT NULL OR es.embedding_alt IS NOT NULL)
            ),
            cache_count AS (
                SELECT COUNT(*) as actual
                FROM embedding_cache_180d
            )
            SELECT
                source_count.expected,
                cache_count.actual,
                cache_count.actual - source_count.expected as diff
            FROM source_count, cache_count
        """)

        result = cur.fetchone()
        logger.info(f"\n180d cache integrity:")
        logger.info(f"  Expected (from embedding_segments): {result['expected']:>12,}")
        logger.info(f"  Actual (in cache):                  {result['actual']:>12,}")
        logger.info(f"  Difference:                         {result['diff']:>12,}")

        if result['diff'] != 0:
            logger.warning(f"  ⚠ Cache may need refresh (difference: {result['diff']:,})")
        else:
            logger.info("  ✓ Cache is in sync with source data")

        # Check both embedding columns are populated correctly
        cur.execute("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(embedding) as has_main,
                COUNT(embedding_alt) as has_alt,
                COUNT(CASE WHEN embedding IS NULL AND embedding_alt IS NULL THEN 1 END) as has_neither
            FROM embedding_cache_180d
        """)

        result = cur.fetchone()
        logger.info(f"\n180d cache embedding coverage:")
        logger.info(f"  Rows with main embedding:  {result['has_main']:>12,} ({100*result['has_main']/result['total_rows']:.1f}%)")
        logger.info(f"  Rows with alt embedding:   {result['has_alt']:>12,} ({100*result['has_alt']/result['total_rows']:.1f}%)")
        if result['has_neither'] > 0:
            logger.warning(f"  ⚠ Rows with NO embeddings: {result['has_neither']:>12,}")
        else:
            logger.info(f"  ✓ All rows have at least one embedding")

        # Check 30d and 7d are subsets of 180d
        cur.execute("""
            SELECT
                (SELECT COUNT(*) FROM embedding_cache_30d) as cache_30d,
                (SELECT COUNT(*) FROM embedding_cache_7d) as cache_7d,
                (SELECT COUNT(*) FROM embedding_cache_180d
                 WHERE publish_date >= NOW() - INTERVAL '30 days') as expected_30d,
                (SELECT COUNT(*) FROM embedding_cache_180d
                 WHERE publish_date >= NOW() - INTERVAL '7 days') as expected_7d
        """)

        result = cur.fetchone()

        logger.info(f"\n30d cache subset check:")
        logger.info(f"  30d cache rows:        {result['cache_30d']:>12,}")
        logger.info(f"  Expected from 180d:    {result['expected_30d']:>12,}")
        if result['cache_30d'] == result['expected_30d']:
            logger.info("  ✓ 30d cache is correct subset")
        else:
            logger.warning(f"  ⚠ Mismatch: {result['cache_30d'] - result['expected_30d']:,}")

        logger.info(f"\n7d cache subset check:")
        logger.info(f"  7d cache rows:         {result['cache_7d']:>12,}")
        logger.info(f"  Expected from 180d:    {result['expected_7d']:>12,}")
        if result['cache_7d'] == result['expected_7d']:
            logger.info("  ✓ 7d cache is correct subset")
        else:
            logger.warning(f"  ⚠ Mismatch: {result['cache_7d'] - result['expected_7d']:,}")


def run_performance_tests(conn):
    """Run performance benchmarks on cache tables."""
    print_header("PERFORMANCE TESTS")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for table in ['embedding_cache_7d', 'embedding_cache_30d', 'embedding_cache_180d']:
            logger.info(f"\n{table}:")

            # Get sample embeddings for both models
            cur.execute(f"""
                SELECT embedding, embedding_alt
                FROM {table}
                WHERE embedding IS NOT NULL AND embedding_alt IS NOT NULL
                LIMIT 1
            """)

            result = cur.fetchone()
            if not result:
                logger.warning("  ⚠ No embeddings found")
                continue

            sample_main = result['embedding']
            sample_alt = result['embedding_alt']

            # Test MAIN embedding model
            logger.info("  Main Model (qwen3:0.6b, 1024-dim):")

            # Test 1: Top-10 similarity search
            start = time.time()
            cur.execute(f"""
                SELECT id, content_id
                FROM {table}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 10
            """, (sample_main,))
            cur.fetchall()
            top10_time = (time.time() - start) * 1000

            # Test 2: Top-200 similarity search
            start = time.time()
            cur.execute(f"""
                SELECT id, content_id
                FROM {table}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 200
            """, (sample_main,))
            cur.fetchall()
            top200_time = (time.time() - start) * 1000

            # Test 3: Filtered search
            start = time.time()
            cur.execute(f"""
                SELECT id, content_id
                FROM {table}
                WHERE projects && ARRAY['CPRMV']::varchar[]
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 100
            """, (sample_main,))
            cur.fetchall()
            filtered_time = (time.time() - start) * 1000

            logger.info(f"    Top-10 search:     {top10_time:>8.1f} ms")
            logger.info(f"    Top-200 search:    {top200_time:>8.1f} ms")
            logger.info(f"    Filtered search:   {filtered_time:>8.1f} ms")

            if top200_time > 200:
                logger.warning(f"    ⚠ Slow performance (should be <200ms)")

            # Test ALT embedding model
            logger.info("  Alt Model (qwen3:4b, 2000-dim):")

            # Test 1: Top-10 similarity search
            start = time.time()
            cur.execute(f"""
                SELECT id, content_id
                FROM {table}
                WHERE embedding_alt IS NOT NULL
                ORDER BY embedding_alt <=> %s::vector
                LIMIT 10
            """, (sample_alt,))
            cur.fetchall()
            top10_time_alt = (time.time() - start) * 1000

            # Test 2: Top-200 similarity search
            start = time.time()
            cur.execute(f"""
                SELECT id, content_id
                FROM {table}
                WHERE embedding_alt IS NOT NULL
                ORDER BY embedding_alt <=> %s::vector
                LIMIT 200
            """, (sample_alt,))
            cur.fetchall()
            top200_time_alt = (time.time() - start) * 1000

            # Test 3: Filtered search
            start = time.time()
            cur.execute(f"""
                SELECT id, content_id
                FROM {table}
                WHERE projects && ARRAY['CPRMV']::varchar[]
                  AND embedding_alt IS NOT NULL
                ORDER BY embedding_alt <=> %s::vector
                LIMIT 100
            """, (sample_alt,))
            cur.fetchall()
            filtered_time_alt = (time.time() - start) * 1000

            logger.info(f"    Top-10 search:     {top10_time_alt:>8.1f} ms")
            logger.info(f"    Top-200 search:    {top200_time_alt:>8.1f} ms")
            logger.info(f"    Filtered search:   {filtered_time_alt:>8.1f} ms")

            if top200_time_alt > 200:
                logger.warning(f"    ⚠ Slow performance (should be <200ms)")


def print_summary(conn):
    """Print final summary."""
    print_header("SUMMARY")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Overall status
        cur.execute("""
            SELECT
                (SELECT COUNT(*) FROM embedding_cache_180d) as cache_180d,
                (SELECT COUNT(*) FROM embedding_cache_30d) as cache_30d,
                (SELECT COUNT(*) FROM embedding_cache_7d) as cache_7d,
                (SELECT COUNT(*) FROM embedding_clusters) as clusters,
                (SELECT COUNT(*) FROM cluster_metadata) as cluster_meta
        """)

        stats = cur.fetchone()

        logger.info("\nInfrastructure Status:")
        logger.info(f"  Cache tables:         {'✓ Populated' if stats['cache_180d'] > 0 else '✗ Empty'}")
        logger.info(f"  Vector indexes:       {'✓ Check above' if stats['cache_180d'] > 0 else '✗ Not created'}")
        logger.info(f"  Clustering tables:    {'✓ Populated' if stats['clusters'] > 0 else '⚠ Empty (run clustering)'}")
        logger.info(f"  pg_cron jobs:         {'✓ Check above'}")

        logger.info("\nNext Steps:")
        if stats['cache_180d'] == 0:
            logger.info("  1. Tables are empty - this is expected after first migration")
            logger.info("  2. Wait for pg_cron to run, or manually call refresh functions")

        if stats['clusters'] == 0:
            logger.info("  1. Run clustering manually:")
            logger.info("     python3 generate_embedding_clusters.py --time-window 30d --model main")
            logger.info("  2. After that, pg_cron will handle automatic updates")


def main():
    parser = argparse.ArgumentParser(description='Audit cache and clustering infrastructure')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CACHE & CLUSTERING INFRASTRUCTURE AUDIT")
    logger.info("=" * 80)
    logger.info(f"\nConnecting to: {DB_CONFIG['host']}:{DB_CONFIG['database']}")

    conn = get_connection()

    try:
        check_parent_tables(conn)
        check_cache_tables(conn)
        check_indexes(conn, detailed=args.detailed)
        check_clustering_tables(conn)
        check_functions(conn)
        check_pg_cron_jobs(conn)
        check_data_integrity(conn)

        if args.performance:
            run_performance_tests(conn)

        print_summary(conn)

        logger.info("\n" + "=" * 80)
        logger.info("AUDIT COMPLETE")
        logger.info("=" * 80)

    finally:
        conn.close()


if __name__ == '__main__':
    main()
