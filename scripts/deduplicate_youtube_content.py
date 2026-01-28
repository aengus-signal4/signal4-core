#!/usr/bin/env python3
"""
Deduplicate YouTube Content Records
====================================

Merges duplicate YouTube content records where both yt_xxxxx and xxxxx versions exist.
The base version (without prefix) is kept as the canonical record since it has the
processing history and downloaded files.

Usage:
    python -m scripts.deduplicate_youtube_content --dry-run
    python -m scripts.deduplicate_youtube_content --dry-run --verbose
    python -m scripts.deduplicate_youtube_content --execute

Safety features:
- Dry-run mode by default (must explicitly pass --execute)
- Validates that base record has more pipeline progress before merging
- Handles edge cases where yt_ record has more progress (skips or warns)
- Preserves all project associations
- Updates foreign key references in related tables
- Detailed logging of all changes
"""

import argparse
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.session import get_session
from src.database.models import Content

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deduplicate_youtube')


def get_pipeline_progress_score(content: Content) -> int:
    """
    Calculate a score representing how far content has progressed in the pipeline.
    Higher score = more progress.
    """
    score = 0
    if content.is_downloaded:
        score += 1
    if content.is_converted:
        score += 2
    if content.is_transcribed:
        score += 4
    if content.is_diarized:
        score += 8
    if content.is_stitched:
        score += 16
    if content.is_embedded:
        score += 32
    if content.is_compressed:
        score += 64
    return score


def get_progress_flags(content: Content) -> Dict[str, bool]:
    """Get all pipeline progress flags as a dict."""
    return {
        'is_downloaded': content.is_downloaded,
        'is_converted': content.is_converted,
        'is_transcribed': content.is_transcribed,
        'is_diarized': content.is_diarized,
        'is_stitched': content.is_stitched,
        'is_embedded': content.is_embedded,
        'is_compressed': content.is_compressed,
        'blocked_download': content.blocked_download,
    }


def format_progress(content: Content) -> str:
    """Format pipeline progress as a string for logging."""
    flags = []
    if content.is_downloaded:
        flags.append('DL')
    if content.is_converted:
        flags.append('CV')
    if content.is_transcribed:
        flags.append('TR')
    if content.is_diarized:
        flags.append('DI')
    if content.is_stitched:
        flags.append('ST')
    if content.is_embedded:
        flags.append('EM')
    if content.is_compressed:
        flags.append('CP')
    if content.blocked_download:
        flags.append('BLOCKED')
    return '[' + ','.join(flags) + ']' if flags else '[none]'


def find_duplicates(session: Session) -> List[Tuple[Content, Content]]:
    """
    Find all duplicate pairs where yt_xxxxx and xxxxx both exist.
    Returns list of (yt_prefixed, base) tuples.
    """
    query = text("""
        SELECT
            c_yt.id as yt_id,
            c_base.id as base_id
        FROM content c_yt
        JOIN content c_base ON c_base.content_id = SUBSTRING(c_yt.content_id FROM 4)
        WHERE c_yt.content_id LIKE 'yt_%'
        AND c_yt.platform = 'youtube'
        AND c_base.platform = 'youtube'
        ORDER BY c_base.id
    """)

    results = session.execute(query).fetchall()

    duplicates = []
    for row in results:
        yt_content = session.get(Content, row.yt_id)
        base_content = session.get(Content, row.base_id)
        if yt_content and base_content:
            duplicates.append((yt_content, base_content))

    return duplicates


def check_related_records(session: Session, content_id: int, content_id_str: str) -> Dict[str, int]:
    """
    Check for related records in other tables that reference this content.
    Returns dict of table_name -> count.
    """
    related = {}

    # Tables with content_id (integer FK to content.id)
    fk_tables = [
        ('sentences', 'content_id'),
        ('embedding_segments', 'content_id'),
        ('speakers', 'content_id'),
        ('content_chunks', 'content_id'),
        ('speaker_assignments', 'content_id'),
        ('hierarchical_summaries', 'content_id'),
        ('theme_classifications', 'content_id'),
        ('bookmarks', 'content_id'),
    ]

    for table, column in fk_tables:
        try:
            result = session.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE {column} = :content_id"),
                {'content_id': content_id}
            ).scalar()
            if result and result > 0:
                related[table] = result
        except Exception:
            # Table might not exist or other error - rollback and continue
            session.rollback()

    # Tables with content_id as string (task_queue uses string content_id)
    str_tables = [
        ('tasks.task_queue', 'content_id'),
    ]

    for table, column in str_tables:
        try:
            result = session.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE {column} = :content_id"),
                {'content_id': content_id_str}
            ).scalar()
            if result and result > 0:
                related[f"{table} (str)"] = result
        except Exception:
            session.rollback()

    return related


def merge_projects(base_projects: List[str], yt_projects: List[str]) -> List[str]:
    """Merge project lists, preserving unique values."""
    base_set = set(base_projects) if base_projects else set()
    yt_set = set(yt_projects) if yt_projects else set()
    return list(base_set | yt_set)


def merge_metadata(base_meta: Optional[Dict], yt_meta: Optional[Dict]) -> Optional[Dict]:
    """Merge metadata dicts, preferring base values for conflicts."""
    if not yt_meta:
        return base_meta
    if not base_meta:
        return yt_meta

    # Start with yt metadata, overlay base metadata (base takes precedence)
    merged = {**yt_meta, **base_meta}
    return merged


def update_task_queue_references(
    session: Session,
    old_content_id: str,
    new_content_id: str,
    dry_run: bool
) -> int:
    """Update task_queue records to point to the canonical content_id."""
    count_query = text("""
        SELECT COUNT(*) FROM tasks.task_queue
        WHERE content_id = :old_id
    """)
    count = session.execute(count_query, {'old_id': old_content_id}).scalar()

    if count > 0 and not dry_run:
        update_query = text("""
            UPDATE tasks.task_queue
            SET content_id = :new_id
            WHERE content_id = :old_id
        """)
        session.execute(update_query, {'old_id': old_content_id, 'new_id': new_content_id})

    return count


def delete_fk_records(session: Session, table_name: str, content_id: int) -> int:
    """Delete FK records that reference the given content.id (integer)."""
    # These tables use content_id as integer FK to content.id
    delete_query = text(f"DELETE FROM {table_name} WHERE content_id = :content_id")
    result = session.execute(delete_query, {'content_id': content_id})
    return result.rowcount


def deduplicate(
    session: Session,
    duplicates: List[Tuple[Content, Content]],
    dry_run: bool = True,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Process duplicate pairs and merge them.

    Strategy:
    - Keep the base record (xxxxx) as canonical since it has processing history
    - Merge projects from yt_ record into base record
    - Merge metadata from yt_ record into base record
    - Update any task_queue references from yt_ to base content_id
    - Delete the yt_ record

    Returns stats dict.
    """
    stats = defaultdict(int)
    total = len(duplicates)
    batch_size = 500  # Commit every N records

    for idx, (yt_content, base_content) in enumerate(duplicates):
        if idx > 0 and idx % 1000 == 0:
            logger.info(f"  Progress: {idx}/{total} ({100*idx//total}%)")

        # Periodic commit to avoid large transactions
        if not dry_run and idx > 0 and idx % batch_size == 0:
            try:
                session.commit()
            except Exception as e:
                logger.error(f"Commit failed at idx {idx}: {e}")
                session.rollback()
                raise

        yt_progress = get_pipeline_progress_score(yt_content)
        base_progress = get_pipeline_progress_score(base_content)

        # Determine merge strategy
        if yt_progress > base_progress:
            # Edge case: yt_ record has MORE progress than base
            # This shouldn't happen but we need to handle it
            stats['skipped_yt_more_progress'] += 1
            yt_related = check_related_records(session, yt_content.id, yt_content.content_id)
            logger.warning(
                f"SKIP: {yt_content.content_id} has MORE progress than {base_content.content_id}! "
                f"yt={format_progress(yt_content)} base={format_progress(base_content)}"
            )
            if yt_related:
                logger.warning(f"  yt_ has related records: {yt_related}")
            continue

        # Check FK records if yt_ record has some progress (could have related data)
        yt_related = {}
        if yt_progress > 0:
            yt_related = check_related_records(session, yt_content.id, yt_content.content_id)

        # If yt_ has FK records but base has MORE progress, we can safely delete them
        # (they're redundant/incomplete data that will be orphaned)
        fk_records_to_delete = {k: v for k, v in yt_related.items() if k not in ['tasks.task_queue (str)']}

        # Safe to merge - log what we're doing
        if verbose:
            logger.info(
                f"MERGE: {yt_content.content_id} -> {base_content.content_id} "
                f"(yt={format_progress(yt_content)} base={format_progress(base_content)})"
            )
            if fk_records_to_delete:
                logger.info(f"  Will DELETE orphaned FK records: {fk_records_to_delete}")
            if yt_related.get('tasks.task_queue (str)'):
                logger.info(f"  Will UPDATE task_queue references: {yt_related.get('tasks.task_queue (str)', 0)}")

        # Merge projects
        merged_projects = merge_projects(base_content.projects, yt_content.projects)
        projects_added = set(merged_projects) - set(base_content.projects or [])
        if projects_added:
            stats['projects_merged'] += len(projects_added)
            if verbose:
                logger.info(f"  Adding projects to base: {projects_added}")

        # Merge metadata
        merged_metadata = merge_metadata(base_content.meta_data, yt_content.meta_data)

        if not dry_run:
            # Update base record
            base_content.projects = merged_projects
            base_content.meta_data = merged_metadata

            # Delete FK records from yt_ record (they're redundant since base has full processing)
            if fk_records_to_delete:
                for table_name, count in fk_records_to_delete.items():
                    delete_fk_records(session, table_name, yt_content.id)
                    stats[f'deleted_{table_name}'] += count

            # Update task_queue references
            task_updates = update_task_queue_references(
                session,
                yt_content.content_id,
                base_content.content_id,
                dry_run=False
            )
            if task_updates > 0:
                stats['task_queue_updated'] += task_updates

            # Delete yt_ record using raw SQL to avoid ORM cascade issues
            delete_content_query = text("DELETE FROM content WHERE id = :content_id")
            session.execute(delete_content_query, {'content_id': yt_content.id})
            session.flush()
        else:
            # Dry run - count what would be deleted
            if fk_records_to_delete:
                for table_name, count in fk_records_to_delete.items():
                    stats[f'would_delete_{table_name}'] += count

        stats['merged'] += 1

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description='Deduplicate YouTube content records (yt_xxxxx vs xxxxx)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview what would be done
    python -m scripts.deduplicate_youtube_content --dry-run

    # Preview with detailed output
    python -m scripts.deduplicate_youtube_content --dry-run --verbose

    # Actually perform the deduplication
    python -m scripts.deduplicate_youtube_content --execute
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Preview changes without making them (default)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform the deduplication (required to make changes)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output for each merge'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of duplicates to process (for testing)'
    )

    args = parser.parse_args()

    # Require explicit --execute to make changes
    dry_run = not args.execute

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("Use --execute to actually perform deduplication")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("EXECUTE MODE - Changes will be committed")
        logger.info("=" * 60)

    with get_session() as session:
        # Find all duplicates
        logger.info("Finding duplicate content records...")
        duplicates = find_duplicates(session)
        logger.info(f"Found {len(duplicates)} duplicate pairs")

        if not duplicates:
            logger.info("No duplicates to process")
            return

        # Apply limit if specified
        if args.limit:
            duplicates = duplicates[:args.limit]
            logger.info(f"Processing first {args.limit} duplicates only")

        # Analyze duplicates before processing (quick pass without FK checks)
        logger.info("\nAnalyzing duplicates (quick pass)...")

        yt_more_progress = 0
        base_more_progress = 0
        equal_progress = 0

        for yt_content, base_content in duplicates:
            yt_score = get_pipeline_progress_score(yt_content)
            base_score = get_pipeline_progress_score(base_content)

            if yt_score > base_score:
                yt_more_progress += 1
            elif base_score > yt_score:
                base_more_progress += 1
            else:
                equal_progress += 1

        logger.info(f"  Base has more progress: {base_more_progress}")
        logger.info(f"  Equal progress: {equal_progress}")
        logger.info(f"  yt_ has more progress (will skip): {yt_more_progress}")
        logger.info(f"  Note: FK checks will be done during processing")

        # Process duplicates
        logger.info("\nProcessing duplicates...")
        stats = deduplicate(session, duplicates, dry_run=dry_run, verbose=args.verbose)

        # Commit if not dry run
        if not dry_run:
            session.commit()
            logger.info("Changes committed to database")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for key, value in sorted(stats.items()):
            logger.info(f"  {key}: {value}")

        if dry_run:
            logger.info("\nThis was a DRY RUN. Use --execute to apply changes.")


if __name__ == '__main__':
    main()
