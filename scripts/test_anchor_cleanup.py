#!/usr/bin/env python3
"""
Test script for Phase 3 anchor cleanup (A+C approach)

Tests:
1. Intra-group outlier detection - find bad anchors within a name group
2. Cross-group variant merging - merge "Podhoritz" with "Podhoretz" based on
   centroid similarity + name similarity

Usage:
    uv run python scripts/test_anchor_cleanup.py --name "Podhor"
    uv run python scripts/test_anchor_cleanup.py --name "Jordan Peterson"
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import text
from tabulate import tabulate

sys.path.append('/Users/signal4/signal4/core')

from src.database.session import get_session


@dataclass
class AnchorGroup:
    """A group of anchors sharing the same Phase 2 name."""
    name: str
    speakers: List[Dict]  # speaker_id, embedding, duration, content_id, etc.
    centroid: Optional[np.ndarray] = None
    clean_centroid: Optional[np.ndarray] = None
    outlier_ids: Set[int] = None

    def __post_init__(self):
        self.outlier_ids = set()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def name_similarity(name1: str, name2: str) -> float:
    """
    Compute name similarity score (0-1).

    Handles:
    - Case insensitivity
    - "Dr." prefix removal
    - Middle name/initial differences
    - Levenshtein distance for transcription errors
    """
    # Normalize
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Remove common prefixes
    prefixes = ['dr. ', 'dr ', 'mr. ', 'mr ', 'ms. ', 'ms ', 'mrs. ', 'mrs ']
    for p in prefixes:
        if n1.startswith(p):
            n1 = n1[len(p):]
        if n2.startswith(p):
            n2 = n2[len(p):]

    # Exact match after normalization
    if n1 == n2:
        return 1.0

    # Levenshtein-based similarity
    max_len = max(len(n1), len(n2))
    if max_len == 0:
        return 1.0

    dist = levenshtein_distance(n1, n2)
    lev_sim = 1.0 - (dist / max_len)

    # Boost if same first/last name
    parts1 = n1.split()
    parts2 = n2.split()

    if parts1 and parts2:
        # First name match
        if parts1[0] == parts2[0]:
            lev_sim = max(lev_sim, 0.7)
        # Last name match
        if parts1[-1] == parts2[-1]:
            lev_sim = max(lev_sim, 0.7)
        # Both match
        if parts1[0] == parts2[0] and parts1[-1] == parts2[-1]:
            lev_sim = max(lev_sim, 0.9)

    return lev_sim


def load_anchor_groups(name_filter: str, min_anchors: int = 2) -> Dict[str, AnchorGroup]:
    """Load anchor groups matching the name filter."""
    with get_session() as session:
        query = text("""
            SELECT
                s.id as speaker_id,
                s.content_id,
                s.embedding,
                s.duration,
                s.embedding_quality_score as quality,
                s.identification_details->'phase2'->>'identified_name' as phase2_name,
                s.identification_details->'phase2'->>'evidence_type' as evidence_type,
                c.title as episode_title
            FROM speakers s
            JOIN content c ON s.content_id = c.content_id
            WHERE s.identification_details->'phase2'->>'status' = 'certain'
              AND s.embedding IS NOT NULL
              AND s.embedding_quality_score >= 0.5
              AND LOWER(s.identification_details->'phase2'->>'identified_name') LIKE :name_filter
            ORDER BY s.identification_details->'phase2'->>'identified_name', s.duration DESC
        """)

        results = session.execute(query, {'name_filter': f'%{name_filter.lower()}%'}).fetchall()

        # Group by name
        groups: Dict[str, AnchorGroup] = {}

        for row in results:
            name = row.phase2_name

            # Parse embedding
            emb_data = row.embedding
            if isinstance(emb_data, str):
                emb_data = json.loads(emb_data)
            embedding = np.array(emb_data, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            speaker = {
                'speaker_id': row.speaker_id,
                'content_id': row.content_id,
                'embedding': embedding,
                'duration': row.duration,
                'quality': row.quality,
                'evidence_type': row.evidence_type,
                'episode_title': row.episode_title[:60] if row.episode_title else 'N/A'
            }

            if name not in groups:
                groups[name] = AnchorGroup(name=name, speakers=[])
            groups[name].speakers.append(speaker)

        # Filter to groups with enough anchors
        groups = {k: v for k, v in groups.items() if len(v.speakers) >= min_anchors}

        return groups


def compute_intra_group_similarities(group: AnchorGroup) -> Dict[int, float]:
    """
    Compute average similarity of each anchor to others in the group.

    Returns dict of speaker_id -> avg_similarity_to_group
    """
    n = len(group.speakers)
    if n < 2:
        return {group.speakers[0]['speaker_id']: 1.0} if n == 1 else {}

    embeddings = np.array([s['embedding'] for s in group.speakers])

    # Pairwise cosine similarity (since normalized, just dot product)
    sim_matrix = embeddings @ embeddings.T

    # Average similarity for each speaker (excluding self)
    avg_sims = {}
    for i, speaker in enumerate(group.speakers):
        # Sum of similarities to others, divided by (n-1)
        other_sims = [sim_matrix[i, j] for j in range(n) if j != i]
        avg_sims[speaker['speaker_id']] = np.mean(other_sims)

    return avg_sims


def detect_outliers(group: AnchorGroup, threshold: float = 0.70) -> Tuple[List[int], Dict[int, float]]:
    """
    Detect outlier anchors whose similarity to group is below threshold.

    Returns:
        - List of outlier speaker_ids
        - Dict of all speaker_id -> avg_similarity
    """
    avg_sims = compute_intra_group_similarities(group)

    outliers = [sid for sid, sim in avg_sims.items() if sim < threshold]

    return outliers, avg_sims


def find_dense_core(group: AnchorGroup, min_core_similarity: float = 0.80) -> Tuple[List[int], List[int], Dict]:
    """
    Find the densest subcluster (core) within an anchor group.

    Instead of marking everything as outliers when the group is fragmented,
    this finds the largest set of speakers that are mutually similar.

    Algorithm:
    1. Build pairwise similarity matrix
    2. Find the speaker with highest average similarity to others (seed)
    3. Grow core by adding speakers similar to the current core centroid
    4. Stop when no more speakers meet threshold

    Returns:
        - core_ids: List of speaker_ids in the dense core
        - outlier_ids: List of speaker_ids NOT in the core
        - stats: Dict with core stats
    """
    n = len(group.speakers)
    if n < 2:
        return [group.speakers[0]['speaker_id']] if n == 1 else [], [], {}

    embeddings = np.array([s['embedding'] for s in group.speakers])
    speaker_ids = [s['speaker_id'] for s in group.speakers]

    # Pairwise similarity matrix
    sim_matrix = embeddings @ embeddings.T

    # Find seed: speaker with highest average similarity to others
    avg_sims = np.mean(sim_matrix, axis=1)
    seed_idx = np.argmax(avg_sims)

    # Grow core from seed
    core_indices = {seed_idx}
    remaining = set(range(n)) - core_indices

    while remaining:
        # Compute core centroid
        core_embeddings = embeddings[list(core_indices)]
        core_centroid = np.mean(core_embeddings, axis=0)
        core_centroid = core_centroid / np.linalg.norm(core_centroid)

        # Find best candidate to add
        best_idx = None
        best_sim = 0

        for idx in remaining:
            sim = float(np.dot(embeddings[idx], core_centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        # Add if meets threshold
        if best_sim >= min_core_similarity:
            core_indices.add(best_idx)
            remaining.remove(best_idx)
        else:
            break

    core_ids = [speaker_ids[i] for i in core_indices]
    outlier_ids = [speaker_ids[i] for i in remaining]

    # Compute core stats
    if len(core_indices) > 1:
        core_sims = []
        core_list = list(core_indices)
        for i, idx_i in enumerate(core_list):
            for idx_j in core_list[i+1:]:
                core_sims.append(sim_matrix[idx_i, idx_j])
        avg_core_sim = np.mean(core_sims)
    else:
        avg_core_sim = 1.0

    stats = {
        'core_size': len(core_ids),
        'outlier_size': len(outlier_ids),
        'avg_core_similarity': float(avg_core_sim),
        'seed_speaker_id': speaker_ids[seed_idx],
        'seed_avg_sim': float(avg_sims[seed_idx])
    }

    return core_ids, outlier_ids, stats


def build_centroid(group: AnchorGroup, exclude_ids: Set[int] = None) -> np.ndarray:
    """Build centroid from group embeddings, optionally excluding outliers."""
    exclude_ids = exclude_ids or set()

    embeddings = [
        s['embedding'] for s in group.speakers
        if s['speaker_id'] not in exclude_ids
    ]

    if not embeddings:
        # Fall back to all if exclusion leaves nothing
        embeddings = [s['embedding'] for s in group.speakers]

    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    return centroid


def find_mergeable_groups(
    groups: Dict[str, AnchorGroup],
    centroid_threshold: float = 0.80,
    name_threshold: float = 0.60
) -> List[Tuple[str, str, float, float]]:
    """
    Find pairs of groups that should be merged based on:
    - Centroid similarity >= centroid_threshold
    - Name similarity >= name_threshold

    Returns list of (name_a, name_b, centroid_sim, name_sim)
    """
    names = list(groups.keys())
    n = len(names)

    mergeable = []

    for i in range(n):
        for j in range(i + 1, n):
            name_a, name_b = names[i], names[j]

            # Check name similarity first (cheaper)
            n_sim = name_similarity(name_a, name_b)
            if n_sim < name_threshold:
                continue

            # Check centroid similarity
            centroid_a = groups[name_a].clean_centroid
            centroid_b = groups[name_b].clean_centroid

            if centroid_a is None or centroid_b is None:
                continue

            c_sim = float(np.dot(centroid_a, centroid_b))

            if c_sim >= centroid_threshold:
                mergeable.append((name_a, name_b, c_sim, n_sim))

    # Sort by centroid similarity descending
    mergeable.sort(key=lambda x: x[2], reverse=True)

    return mergeable


def main():
    parser = argparse.ArgumentParser(description='Test anchor cleanup (A+C approach)')
    parser.add_argument('--name', type=str, required=True, help='Name filter (e.g., "Podhor" or "Jordan Peterson")')
    parser.add_argument('--outlier-threshold', type=float, default=0.80, help='Min similarity to join core (default: 0.80)')
    parser.add_argument('--centroid-threshold', type=float, default=0.80, help='Centroid similarity for merge (default: 0.80)')
    parser.add_argument('--name-threshold', type=float, default=0.60, help='Name similarity for merge (default: 0.60)')
    parser.add_argument('--min-anchors', type=int, default=2, help='Minimum anchors per group (default: 2)')

    args = parser.parse_args()

    print("=" * 80)
    print(f"ANCHOR CLEANUP TEST: '{args.name}'")
    print("=" * 80)

    # Load anchor groups
    print(f"\n1. Loading anchor groups matching '{args.name}'...")
    groups = load_anchor_groups(args.name, min_anchors=args.min_anchors)

    if not groups:
        print("   No groups found!")
        return

    print(f"   Found {len(groups)} groups:")
    for name, group in sorted(groups.items(), key=lambda x: len(x[1].speakers), reverse=True):
        print(f"   - '{name}': {len(group.speakers)} anchors")

    # Step A: Find dense core in each group
    print(f"\n2. DENSE CORE DETECTION (min_core_similarity={args.outlier_threshold})")
    print("-" * 80)

    all_outliers = {}
    for name, group in sorted(groups.items(), key=lambda x: len(x[1].speakers), reverse=True):
        core_ids, outlier_ids, stats = find_dense_core(group, min_core_similarity=args.outlier_threshold)
        group.outlier_ids = set(outlier_ids)
        all_outliers[name] = outlier_ids

        core_pct = 100 * len(core_ids) / len(group.speakers) if group.speakers else 0
        print(f"\n   '{name}': {len(core_ids)}/{len(group.speakers)} in core ({core_pct:.0f}%)")
        print(f"      Core avg similarity: {stats.get('avg_core_similarity', 0):.3f}")

        if outlier_ids and len(outlier_ids) <= 10:
            print(f"      Outliers ({len(outlier_ids)}):")
            for speaker in group.speakers:
                if speaker['speaker_id'] in outlier_ids:
                    print(f"         speaker {speaker['speaker_id']}: {speaker['evidence_type']}, "
                          f"\"{speaker['episode_title']}\"")
        elif outlier_ids:
            print(f"      Outliers: {len(outlier_ids)} speakers (not listed)")

    # Build clean centroids
    print(f"\n3. BUILDING CLEAN CENTROIDS (excluding outliers)")
    print("-" * 80)

    for name, group in groups.items():
        # Raw centroid (all anchors)
        group.centroid = build_centroid(group, exclude_ids=set())
        # Clean centroid (excluding outliers)
        group.clean_centroid = build_centroid(group, exclude_ids=group.outlier_ids)

        n_clean = len(group.speakers) - len(group.outlier_ids)
        print(f"   '{name}': {n_clean}/{len(group.speakers)} anchors used for clean centroid")

    # Compare raw vs clean centroids
    print(f"\n4. RAW vs CLEAN CENTROID COMPARISON")
    print("-" * 80)

    for name, group in groups.items():
        if group.outlier_ids:
            raw_clean_sim = float(np.dot(group.centroid, group.clean_centroid))
            print(f"   '{name}': raw↔clean similarity = {raw_clean_sim:.4f}")

    # Step C: Cross-group variant merging
    print(f"\n5. CROSS-GROUP VARIANT DETECTION (centroid>={args.centroid_threshold}, name>={args.name_threshold})")
    print("-" * 80)

    mergeable = find_mergeable_groups(
        groups,
        centroid_threshold=args.centroid_threshold,
        name_threshold=args.name_threshold
    )

    if mergeable:
        print("\n   MERGEABLE PAIRS:")
        table_data = []
        for name_a, name_b, c_sim, n_sim in mergeable:
            count_a = len(groups[name_a].speakers)
            count_b = len(groups[name_b].speakers)
            table_data.append([name_a, count_a, name_b, count_b, f"{c_sim:.3f}", f"{n_sim:.3f}"])

        print(tabulate(
            table_data,
            headers=['Name A', '#A', 'Name B', '#B', 'Centroid Sim', 'Name Sim'],
            tablefmt='simple'
        ))

        # Show what the merge would look like
        print("\n   PROPOSED MERGES:")
        for name_a, name_b, c_sim, n_sim in mergeable:
            count_a = len(groups[name_a].speakers) - len(groups[name_a].outlier_ids)
            count_b = len(groups[name_b].speakers) - len(groups[name_b].outlier_ids)
            canonical = name_a if count_a >= count_b else name_b
            alias = name_b if count_a >= count_b else name_a
            print(f"      '{alias}' → '{canonical}' (combined: {count_a + count_b} clean anchors)")
    else:
        print("   No mergeable pairs found with current thresholds.")

    # Show ALL pairwise centroid similarities for debugging
    print(f"\n6. ALL PAIRWISE CENTROID SIMILARITIES")
    print("-" * 80)

    names = list(groups.keys())
    all_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            c_sim = float(np.dot(groups[name_a].clean_centroid, groups[name_b].clean_centroid))
            n_sim = name_similarity(name_a, name_b)
            all_pairs.append((name_a, name_b, c_sim, n_sim))

    all_pairs.sort(key=lambda x: x[2], reverse=True)
    for name_a, name_b, c_sim, n_sim in all_pairs[:15]:
        merge_status = "✓ MERGE" if c_sim >= args.centroid_threshold and n_sim >= args.name_threshold else ""
        print(f"   {name_a} <-> {name_b}")
        print(f"      centroid_sim={c_sim:.3f}, name_sim={n_sim:.3f} {merge_status}")

    # Check for pairs that have HIGH centroid sim but LOW name sim (potential co-hosts)
    print(f"\n7. HIGH CENTROID / LOW NAME SIMILARITY (potential different people)")
    print("-" * 80)

    suspicious = []
    for name_a, name_b, c_sim, n_sim in all_pairs:
        # High voice similarity but low name similarity
        if c_sim >= 0.75 and n_sim < args.name_threshold:
            suspicious.append((name_a, name_b, c_sim, n_sim))

    if suspicious:
        suspicious.sort(key=lambda x: x[2], reverse=True)
        print("   ⚠️ These pairs have similar voices but different names (DO NOT MERGE):")
        for name_a, name_b, c_sim, n_sim in suspicious[:10]:
            print(f"      '{name_a}' vs '{name_b}': centroid={c_sim:.3f}, name={n_sim:.3f}")
    else:
        print("   None found.")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_anchors = sum(len(g.speakers) for g in groups.values())
    total_outliers = sum(len(g.outlier_ids) for g in groups.values())
    print(f"Total anchors: {total_anchors}")
    print(f"Outliers detected: {total_outliers} ({100*total_outliers/total_anchors:.1f}%)")
    print(f"Mergeable pairs: {len(mergeable)}")
    print(f"Suspicious pairs (do not merge): {len(suspicious)}")


if __name__ == '__main__':
    main()
