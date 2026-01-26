#!/usr/bin/env python3
"""
Find Phase 2 assignments that are likely ad contamination.

Pattern: A person is identified as a speaker with high duration (>30%)
on a channel that is NOT their own channel. This often happens when:
1. A pre-roll ad has someone self-introducing ("This is Dr. Jordan Peterson...")
2. The diarization groups this ad voice with the main host
3. Phase 2 LLM sees the self-intro and assigns the wrong name

Usage:
    uv run python scripts/find_phase2_ad_contamination.py
    uv run python scripts/find_phase2_ad_contamination.py --invalidate --apply
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import text
from tabulate import tabulate

sys.path.append('/Users/signal4/signal4/core')
from src.database.session import get_session


# Known mappings: person -> their channel patterns
PERSON_CHANNEL_MAP = {
    # Jordan Peterson variants
    'jordan peterson': ['jordan b peterson', 'jordan b. peterson podcast', 'mikhaila peterson'],
    'jordan b. peterson': ['jordan b peterson', 'jordan b. peterson podcast', 'mikhaila peterson'],
    'dr. jordan b. peterson': ['jordan b peterson', 'jordan b. peterson podcast', 'mikhaila peterson'],
    'dr. jordan peterson': ['jordan b peterson', 'jordan b. peterson podcast', 'mikhaila peterson'],
    'dr. jordan bean peterson': ['jordan b peterson', 'jordan b. peterson podcast', 'mikhaila peterson'],

    # Ben Shapiro
    'ben shapiro': ['ben shapiro'],

    # Matt Walsh
    'matt walsh': ['matt walsh'],

    # Michael Knowles
    'michael knowles': ['michael knowles'],

    # Add more as needed
}


def find_contaminated_assignments(min_duration_pct: float = 0.30) -> list:
    """
    Find Phase 2 assignments where:
    - Speaker has high duration (likely host)
    - But identified name doesn't match channel

    Returns list of (speaker_id, channel, identified_name, duration_pct, reasoning)
    """
    contaminated = []

    with get_session() as session:
        # Get all high-duration Phase 2 certain assignments
        query = text("""
            SELECT
                s.id as speaker_id,
                ch.display_name as channel,
                s.identification_details->'phase2'->>'identified_name' as identified_name,
                s.identification_details->'phase2'->>'evidence_type' as evidence_type,
                s.identification_details->'phase2'->>'reasoning' as reasoning,
                s.duration / c.duration as duration_pct
            FROM speakers s
            JOIN content c ON s.content_id = c.content_id
            JOIN channels ch ON c.channel_id = ch.id
            WHERE s.identification_details->'phase2'->>'status' = 'certain'
              AND s.duration / c.duration > :min_duration
              AND s.identification_details->'phase2'->>'evidence_type' = 'self_intro'
            ORDER BY s.identification_details->'phase2'->>'identified_name', ch.display_name
        """)

        results = session.execute(query, {'min_duration': min_duration_pct}).fetchall()

        for row in results:
            name_lower = row.identified_name.lower() if row.identified_name else ''
            channel_lower = row.channel.lower() if row.channel else ''

            # Check if this name has known channel associations
            if name_lower in PERSON_CHANNEL_MAP:
                expected_channels = PERSON_CHANNEL_MAP[name_lower]

                # Check if any expected channel pattern matches
                is_own_channel = any(
                    pattern in channel_lower
                    for pattern in expected_channels
                )

                if not is_own_channel:
                    contaminated.append({
                        'speaker_id': row.speaker_id,
                        'channel': row.channel,
                        'identified_name': row.identified_name,
                        'evidence_type': row.evidence_type,
                        'duration_pct': row.duration_pct,
                        'reasoning': row.reasoning[:200] if row.reasoning else ''
                    })

    return contaminated


def invalidate_assignments(speaker_ids: list, dry_run: bool = True):
    """
    Mark Phase 2 assignments as invalid.

    Sets identification_details['phase2']['status'] = 'invalidated_ad_contamination'
    """
    if dry_run:
        print(f"\n[DRY RUN] Would invalidate {len(speaker_ids)} speakers")
        return

    timestamp = datetime.now(timezone.utc).isoformat()

    with get_session() as session:
        for speaker_id in speaker_ids:
            # Update the phase2 status to invalidated
            query = text("""
                UPDATE speakers
                SET identification_details = jsonb_set(
                    identification_details,
                    '{phase2,status}',
                    '"invalidated_ad_contamination"'
                ),
                identification_details = jsonb_set(
                    identification_details,
                    '{phase2,invalidated_at}',
                    :timestamp::jsonb
                ),
                updated_at = NOW()
                WHERE id = :speaker_id
            """)

            session.execute(query, {
                'speaker_id': speaker_id,
                'timestamp': json.dumps(timestamp)
            })

        session.commit()
        print(f"\n✓ Invalidated {len(speaker_ids)} speakers")


def main():
    parser = argparse.ArgumentParser(description='Find Phase 2 ad contamination')
    parser.add_argument('--min-duration', type=float, default=0.30,
                       help='Minimum duration %% to consider (default: 0.30)')
    parser.add_argument('--invalidate', action='store_true',
                       help='Mark contaminated assignments as invalid')
    parser.add_argument('--apply', action='store_true',
                       help='Actually apply changes (default: dry run)')

    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 2 AD CONTAMINATION DETECTION")
    print("=" * 80)
    print(f"Min duration threshold: {args.min_duration * 100:.0f}%")
    print()

    # Find contaminated assignments
    contaminated = find_contaminated_assignments(args.min_duration)

    if not contaminated:
        print("No contaminated assignments found!")
        return

    # Group by identified_name
    by_name = defaultdict(list)
    for item in contaminated:
        by_name[item['identified_name']].append(item)

    print(f"Found {len(contaminated)} contaminated assignments across {len(by_name)} names:\n")

    # Summary table
    summary = []
    for name, items in sorted(by_name.items(), key=lambda x: len(x[1]), reverse=True):
        channels = set(i['channel'] for i in items)
        summary.append([
            name,
            len(items),
            ', '.join(list(channels)[:3]) + ('...' if len(channels) > 3 else '')
        ])

    print(tabulate(summary, headers=['Identified Name', 'Count', 'Channels'], tablefmt='simple'))

    # Show sample details
    print("\n" + "-" * 80)
    print("SAMPLE DETAILS (first 5 per name):")
    print("-" * 80)

    for name, items in sorted(by_name.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"\n'{name}' ({len(items)} total):")
        for item in items[:5]:
            print(f"  speaker {item['speaker_id']}: {item['channel']} ({item['duration_pct']*100:.0f}%)")
            print(f"    {item['reasoning'][:100]}...")

    # Invalidate if requested
    if args.invalidate:
        speaker_ids = [item['speaker_id'] for item in contaminated]
        invalidate_assignments(speaker_ids, dry_run=not args.apply)


if __name__ == '__main__':
    main()
