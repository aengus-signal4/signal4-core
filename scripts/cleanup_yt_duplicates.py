#!/usr/bin/env python3
"""
Cleanup script for duplicate yt_ prefixed content records.

This script removes yt_ prefixed content records where a non-prefixed version
exists with equal or more complete processing. It also removes associated
S3 files and related database records.

Safety:
- Only deletes yt_ records where the non-prefixed version has >= progress
- Skips the 4 edge cases where yt_ has MORE progress
- Dry run mode by default
"""

import argparse
import sys
from typing import List, Tuple
from sqlalchemy import text

sys.path.insert(0, "/Users/signal4/signal4/core")

from src.database.session import get_session
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger("cleanup_yt_duplicates")


def get_safe_yt_duplicates() -> List[Tuple[int, str]]:
    """
    Get yt_ content records that are safe to delete.

    Safe = the non-prefixed version has equal or more progress.
    Excludes the 4 edge cases where yt_ has more progress.

    Returns:
        List of (id, content_id) tuples for yt_ records to delete
    """
    with get_session() as session:
        result = session.execute(text("""
            SELECT c1.id, c1.content_id
            FROM content c1
            JOIN content c2 ON c2.content_id = SUBSTRING(c1.content_id FROM 4)
            WHERE c1.content_id LIKE 'yt_%%'
              -- Exclude cases where yt_ has more progress than non-prefixed
              AND NOT (
                (c1.is_transcribed AND NOT c2.is_transcribed)
                OR (c1.is_stitched AND NOT c2.is_stitched)
                OR (c1.is_embedded AND NOT c2.is_embedded)
              )
            ORDER BY c1.id
        """))
        return [(row[0], row[1]) for row in result.fetchall()]


def get_edge_cases() -> List[Tuple[str, str]]:
    """Get the edge cases where yt_ has more progress."""
    with get_session() as session:
        result = session.execute(text("""
            SELECT c1.content_id, c2.content_id
            FROM content c1
            JOIN content c2 ON c2.content_id = SUBSTRING(c1.content_id FROM 4)
            WHERE c1.content_id LIKE 'yt_%%'
              AND (
                (c1.is_transcribed AND NOT c2.is_transcribed)
                OR (c1.is_stitched AND NOT c2.is_stitched)
                OR (c1.is_embedded AND NOT c2.is_embedded)
              )
        """))
        return [(row[0], row[1]) for row in result.fetchall()]


def delete_related_records(content_id: int, content_id_str: str, dry_run: bool = True) -> dict:
    """
    Delete related records for a content item.

    Args:
        content_id: Integer primary key of content record
        content_id_str: String content_id (e.g., 'yt_abc123')
        dry_run: If True, don't actually delete

    Returns:
        Dict with counts of records deleted per table
    """
    counts = {}

    with get_session() as session:
        # 1. Delete speaker_assignments that reference speakers for this content
        result = session.execute(text("""
            SELECT COUNT(*) FROM speaker_assignments sa
            JOIN speakers sp ON sa.speaker_embedding_id = sp.id
            WHERE sp.content_id = :content_id_str
        """), {"content_id_str": content_id_str})
        counts["speaker_assignments"] = result.scalar()

        if not dry_run and counts["speaker_assignments"] > 0:
            session.execute(text("""
                DELETE FROM speaker_assignments sa
                USING speakers sp
                WHERE sa.speaker_embedding_id = sp.id
                  AND sp.content_id = :content_id_str
            """), {"content_id_str": content_id_str})

        # 2. Delete sentences (has FK to content.id and speakers)
        result = session.execute(text("""
            SELECT COUNT(*) FROM sentences WHERE content_id = :content_id
        """), {"content_id": content_id})
        counts["sentences"] = result.scalar()

        if not dry_run and counts["sentences"] > 0:
            session.execute(text("""
                DELETE FROM sentences WHERE content_id = :content_id
            """), {"content_id": content_id})

        # 3. Delete speakers (uses string content_id)
        result = session.execute(text("""
            SELECT COUNT(*) FROM speakers WHERE content_id = :content_id_str
        """), {"content_id_str": content_id_str})
        counts["speakers"] = result.scalar()

        if not dry_run and counts["speakers"] > 0:
            session.execute(text("""
                DELETE FROM speakers WHERE content_id = :content_id_str
            """), {"content_id_str": content_id_str})

        # 4. Delete embedding_segments (has FK to content.id)
        result = session.execute(text("""
            SELECT COUNT(*) FROM embedding_segments WHERE content_id = :content_id
        """), {"content_id": content_id})
        counts["embedding_segments"] = result.scalar()

        if not dry_run and counts["embedding_segments"] > 0:
            session.execute(text("""
                DELETE FROM embedding_segments WHERE content_id = :content_id
            """), {"content_id": content_id})

        # 5. Delete content_chunks (has FK to content.id)
        result = session.execute(text("""
            SELECT COUNT(*) FROM content_chunks WHERE content_id = :content_id
        """), {"content_id": content_id})
        counts["content_chunks"] = result.scalar()

        if not dry_run and counts["content_chunks"] > 0:
            session.execute(text("""
                DELETE FROM content_chunks WHERE content_id = :content_id
            """), {"content_id": content_id})

        # 6. Delete the content record itself
        counts["content"] = 1

        if not dry_run:
            session.execute(text("""
                DELETE FROM content WHERE id = :content_id
            """), {"content_id": content_id})
            session.commit()

    return counts


def delete_s3_folder(s3: S3Storage, content_id_str: str, dry_run: bool = True) -> int:
    """
    Delete S3 folder for a content item.

    Args:
        s3: S3Storage instance
        content_id_str: String content_id (e.g., 'yt_abc123')
        dry_run: If True, don't actually delete

    Returns:
        Number of files deleted (or would be deleted in dry run)
    """
    prefix = f"content/{content_id_str}/"
    files = s3.list_s3_objects(prefix=prefix)

    if not files:
        return 0

    if not dry_run:
        for file_key in files:
            s3.delete_file(file_key)

    return len(files)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up duplicate yt_ prefixed content records"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete records (default is dry run)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (for testing)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print details for each record"
    )
    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("Use --execute to actually delete records")
        print("=" * 60)
        print()
    else:
        print("=" * 60)
        print("EXECUTE MODE - Records will be deleted!")
        print("=" * 60)
        print()

    # Show edge cases that will be skipped
    edge_cases = get_edge_cases()
    print(f"Edge cases (yt_ has more progress, will be SKIPPED): {len(edge_cases)}")
    for yt_id, non_yt_id in edge_cases:
        print(f"  - {yt_id} (keeping, non-prefixed {non_yt_id} has less data)")
    print()

    # Get safe duplicates to delete
    duplicates = get_safe_yt_duplicates()
    total = len(duplicates)
    print(f"Safe yt_ duplicates to delete: {total}")

    if args.limit:
        duplicates = duplicates[:args.limit]
        print(f"Processing first {args.limit} records only")
    print()

    # Initialize S3
    s3 = S3Storage(S3StorageConfig())

    # Track totals
    totals = {
        "content": 0,
        "content_chunks": 0,
        "sentences": 0,
        "speakers": 0,
        "speaker_assignments": 0,
        "embedding_segments": 0,
        "s3_files": 0,
    }

    # Process each duplicate
    for i, (content_id, content_id_str) in enumerate(duplicates):
        if args.verbose or (i + 1) % 1000 == 0:
            print(f"[{i + 1}/{len(duplicates)}] Processing {content_id_str}...")

        # Delete related DB records
        counts = delete_related_records(content_id, content_id_str, dry_run=dry_run)
        for table, count in counts.items():
            totals[table] = totals.get(table, 0) + count

        # Delete S3 files
        s3_count = delete_s3_folder(s3, content_id_str, dry_run=dry_run)
        totals["s3_files"] += s3_count

        if args.verbose and s3_count > 0:
            print(f"  S3 files: {s3_count}")

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTED'}")
    print()
    print("Database records {'to delete' if dry_run else 'deleted'}:")
    print(f"  content:             {totals['content']:,}")
    print(f"  content_chunks:      {totals['content_chunks']:,}")
    print(f"  sentences:           {totals['sentences']:,}")
    print(f"  speakers:            {totals['speakers']:,}")
    print(f"  speaker_assignments: {totals['speaker_assignments']:,}")
    print(f"  embedding_segments:  {totals['embedding_segments']:,}")
    print()
    print(f"S3 files {'to delete' if dry_run else 'deleted'}: {totals['s3_files']:,}")

    if dry_run:
        print()
        print("To execute deletion, run with --execute flag")


if __name__ == "__main__":
    main()
