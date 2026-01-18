#!/usr/bin/env python3
"""
Phase 3: Name Standardization for Speaker Identification
=========================================================

Standardizes speaker names by:
1. Applying verified aliases from config (idempotent)
2. Analyzing embeddings to discover potential merges
3. Generating suggestions for human/frontier review

This phase runs BEFORE label propagation to ensure consistent anchor names.

Key Features:
- Static alias application (from name_aliases.yaml)
- Embedding-based similarity detection
- Suggestions output for review (no auto-merge)
- Idempotent - processed items don't reappear

Output Files:
- config/name_suggestions.yaml - Human-readable suggestions
- config/name_suggestions.json - Machine-readable with audio samples

Review Dashboard:
    streamlit run dashboards/name_review.py --server.port 8502

Usage:
    # Generate suggestions
    python -m src.speaker_identification.strategies.name_standardization --project CPRMV

    # With custom threshold
    python -m src.speaker_identification.strategies.name_standardization \\
        --project CPRMV --threshold 0.75
"""

import argparse
import asyncio
import json
import sys
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from src.utils.paths import get_project_root
project_root = str(get_project_root())
sys.path.append(project_root)

from sqlalchemy import text

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from src.utils.config import get_project_date_range

# Path to name aliases config and suggestions output
NAME_ALIASES_PATH = Path(project_root) / 'config' / 'name_aliases.yaml'
NAME_SUGGESTIONS_PATH = Path(project_root) / 'config' / 'name_suggestions.yaml'
NAME_SUGGESTIONS_JSON_PATH = Path(project_root) / 'config' / 'name_suggestions.json'

logger = setup_worker_logger('speaker_identification.name_standardization')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

PHASE_KEY = "phase3_names"


@dataclass
class NameStandardizationConfig:
    """Configuration for name standardization."""
    # Similarity threshold for suggesting merges
    merge_threshold: float = 0.70

    # Minimum embedding quality to include
    min_embedding_quality: float = 0.50

    # Number of audio samples per name
    n_samples: int = 2


class NameStandardizationStrategy:
    """
    Phase 3: Name Standardization.

    Loads Phase 2 verified speakers, applies static aliases,
    analyzes embeddings for potential merges, and outputs
    suggestions for review.
    """

    def __init__(
        self,
        config: NameStandardizationConfig = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.config = config or NameStandardizationConfig()
        self.start_date = start_date
        self.end_date = end_date

        self.stats = {
            'total_verified_speakers': 0,
            'unique_names_before': 0,
            'unique_names_after': 0,
            'static_alias_merges': 0,
            'suggested_merges': 0,
            'flagged_handles': 0,
            'potential_do_not_merge': 0,
        }

    def _load_verified_speakers(self, project: str) -> Dict[str, List[Dict]]:
        """Load Phase 2 verified speakers grouped by name."""
        with get_session() as session:
            filters = [
                "s.embedding IS NOT NULL",
                "s.identification_details->'phase2'->>'status' = 'certain'",
            ]
            params = {'min_quality': self.config.min_embedding_quality}

            if project:
                filters.append(":project = ANY(c.projects)")
                params['project'] = project

            if self.start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = self.start_date
            if self.end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = self.end_date

            filter_clause = " AND ".join(filters)

            query = text(f"""
                SELECT
                    s.id as speaker_id,
                    s.content_id,
                    s.embedding,
                    s.duration,
                    COALESCE(s.embedding_quality_score, 0.5) as quality,
                    s.identification_details->'phase2'->>'identified_name' as name
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                WHERE {filter_clause}
                  AND COALESCE(s.embedding_quality_score, 0.5) >= :min_quality
                  AND c.is_stitched = true
                ORDER BY s.id
            """)

            results = session.execute(query, params).fetchall()

            speakers_by_name = defaultdict(list)
            for row in results:
                name = row.name
                if not name:
                    continue

                emb_data = row.embedding
                if isinstance(emb_data, str):
                    emb_data = json.loads(emb_data)
                embedding = np.array(emb_data, dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                speakers_by_name[name].append({
                    'speaker_id': row.speaker_id,
                    'content_id': row.content_id,
                    'embedding': embedding,
                    'duration': row.duration or 0,
                    'quality': row.quality,
                })

            return dict(speakers_by_name)

    def _load_name_aliases(self) -> Tuple[Dict[str, str], Set[str], Set[frozenset]]:
        """Load name aliases from config file."""
        alias_to_canonical = {}
        unresolved_handles = set()
        do_not_merge = set()

        if not NAME_ALIASES_PATH.exists():
            logger.info(f"No alias file found at {NAME_ALIASES_PATH}")
            return alias_to_canonical, unresolved_handles, do_not_merge

        try:
            with open(NAME_ALIASES_PATH, 'r') as f:
                config = yaml.safe_load(f)

            if not config:
                return alias_to_canonical, unresolved_handles, do_not_merge

            # Load aliases
            aliases = config.get('aliases', {})
            for canonical, alias_list in aliases.items():
                if alias_list:
                    for alias in alias_list:
                        alias_to_canonical[alias.lower()] = canonical

            # Load unresolved handles
            handles = config.get('unresolved_handles', [])
            if handles:
                unresolved_handles = set(h.lower() for h in handles)

            # Load do-not-merge pairs
            dnm_list = config.get('do_not_merge', [])
            if dnm_list:
                for pair in dnm_list:
                    if isinstance(pair, list) and len(pair) >= 2:
                        do_not_merge.add(frozenset(n.lower() for n in pair))

            logger.info(f"Loaded {len(alias_to_canonical)} aliases, "
                       f"{len(unresolved_handles)} unresolved handles, "
                       f"{len(do_not_merge)} do-not-merge pairs")

        except Exception as e:
            logger.warning(f"Error loading aliases: {e}")

        return alias_to_canonical, unresolved_handles, do_not_merge

    def _apply_static_aliases(
        self,
        speakers_by_name: Dict[str, List[Dict]],
        alias_to_canonical: Dict[str, str],
        unresolved_handles: Set[str]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, str], int]:
        """Apply static aliases from config file."""
        name_mappings = {}
        merged_speakers_by_name = {}
        unresolved_count = 0

        # Group names by their canonical form
        canonical_groups: Dict[str, List[str]] = defaultdict(list)

        for name in speakers_by_name.keys():
            name_lower = name.lower()

            if name_lower in alias_to_canonical:
                canonical = alias_to_canonical[name_lower]
                canonical_groups[canonical].append(name)
                if name != canonical:
                    name_mappings[name] = canonical
            else:
                canonical_groups[name].append(name)

            if name_lower in unresolved_handles:
                unresolved_count += 1
                logger.info(f"  [UNRESOLVED] '{name}' is a known handle")

        # Merge speakers for each canonical group
        for canonical, name_variants in canonical_groups.items():
            merged_speakers = []
            for variant in name_variants:
                if variant in speakers_by_name:
                    merged_speakers.extend(speakers_by_name[variant])

            if merged_speakers:
                merged_speakers_by_name[canonical] = merged_speakers

                if len(name_variants) > 1:
                    others = [v for v in name_variants if v != canonical]
                    logger.info(f"  [ALIAS] {others} -> '{canonical}'")

        return merged_speakers_by_name, name_mappings, unresolved_count

    def _score_name_quality(self, name: str) -> int:
        """Score a name's quality for canonicalization."""
        score = 0
        words = name.split()

        if len(words) >= 2:
            if all(w[0].isupper() for w in words if w):
                score += 3
        elif len(words) == 1:
            score -= 1

        if name == name.title():
            score += 1

        if re.match(r'^[a-z]+[A-Z]', name):
            score -= 3
        if len(words) == 1 and name.islower():
            score -= 2
        if re.search(r'\d', name):
            score -= 3
        if name.lower().startswith('the '):
            score -= 2

        return score

    def _is_likely_handle(self, name: str) -> bool:
        """Check if a name is likely a handle/pseudonym."""
        if re.match(r'^[a-z]+[A-Z]', name):
            return True
        if re.search(r'\d', name):
            return True
        words = name.split()
        if len(words) == 1 and name.islower():
            return True
        handle_indicators = ['mister', 'billboard', 'pleb', 'anonymous', 'the ']
        if any(ind in name.lower() for ind in handle_indicators):
            return True
        return False

    def _get_sample_segments(self, name: str, speakers: List[Dict], n_samples: int = 2) -> List[Dict]:
        """Get sample audio segments for a name."""
        sorted_speakers = sorted(speakers, key=lambda s: s.get('duration', 0) or 0, reverse=True)
        samples = []
        seen_content = set()

        with get_session() as session:
            for sp in sorted_speakers:
                if len(samples) >= n_samples:
                    break

                content_id = sp.get('content_id')
                speaker_id = sp.get('speaker_id')

                if content_id in seen_content and len(samples) > 0:
                    continue
                seen_content.add(content_id)

                result = session.execute(text("""
                    SELECT
                        st.start_time,
                        st.end_time,
                        st.text,
                        c.title as episode_title,
                        c.channel_id
                    FROM speaker_transcriptions st
                    JOIN content c ON st.content_id = c.id
                    WHERE c.content_id = :content_id
                      AND st.speaker_id = :speaker_id
                      AND st.end_time - st.start_time BETWEEN 5 AND 30
                      AND LENGTH(st.text) > 50
                    ORDER BY st.end_time - st.start_time DESC
                    LIMIT 1
                """), {
                    'content_id': content_id,
                    'speaker_id': speaker_id
                }).fetchone()

                if result:
                    samples.append({
                        'content_id': content_id,
                        'speaker_id': speaker_id,
                        'start_time': float(result.start_time),
                        'end_time': float(result.end_time),
                        'duration': round(float(result.end_time - result.start_time), 1),
                        'text': result.text[:200] + ('...' if len(result.text) > 200 else ''),
                        'episode_title': result.episode_title,
                        'channel_id': result.channel_id,
                    })

        return samples

    def _analyze_embeddings(
        self,
        speakers_by_name: Dict[str, List[Dict]],
        alias_to_canonical: Dict[str, str],
        unresolved_handles: Set[str],
        do_not_merge: Set[frozenset]
    ) -> Dict:
        """Analyze embeddings and generate suggestions."""
        # Build centroids for each name
        name_data = {}
        for name, speakers in speakers_by_name.items():
            if not speakers:
                continue

            embeddings = []
            total_duration = 0
            for sp in speakers:
                if sp.get('embedding') is not None:
                    embeddings.append(sp['embedding'])
                    total_duration += sp.get('duration', 0) or 0

            if not embeddings:
                continue

            emb_array = np.array(embeddings, dtype=np.float32)
            centroid = np.mean(emb_array, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            name_data[name] = {
                'centroid': centroid,
                'speakers': speakers,
                'total_hours': total_duration / 3600,
                'quality_score': self._score_name_quality(name),
                'is_handle': self._is_likely_handle(name),
                'n_episodes': len(set(sp.get('content_id') for sp in speakers))
            }

        if len(name_data) < 2:
            return {'suggested_merges': [], 'flagged_handles': [], 'potential_do_not_merge': []}

        # Build similarity matrix
        names = list(name_data.keys())
        n_names = len(names)
        centroids = np.array([name_data[n]['centroid'] for n in names], dtype=np.float32)
        sim_matrix = centroids @ centroids.T

        # Build lookup structures
        existing_aliases_lower = set(alias_to_canonical.keys())
        for canonical in alias_to_canonical.values():
            existing_aliases_lower.add(canonical.lower())

        canonical_to_variants: Dict[str, Set[str]] = defaultdict(set)
        for alias, canonical in alias_to_canonical.items():
            canonical_to_variants[canonical.lower()].add(alias)

        suggestions = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'threshold_used': self.config.merge_threshold,
            'suggested_merges': [],
            'flagged_handles': [],
            'potential_do_not_merge': [],
        }

        # Find merge candidates
        processed_pairs = set()
        for i in range(n_names):
            for j in range(i + 1, n_names):
                sim = float(sim_matrix[i, j])
                if sim < self.config.merge_threshold:
                    continue

                name_a, name_b = names[i], names[j]
                pair_key = frozenset([name_a.lower(), name_b.lower()])

                if pair_key in do_not_merge:
                    continue

                # Check if already merged
                a_canonical = alias_to_canonical.get(name_a.lower(), name_a.lower())
                b_canonical = alias_to_canonical.get(name_b.lower(), name_b.lower())
                if a_canonical == b_canonical:
                    continue

                a_variants = canonical_to_variants.get(a_canonical, set()) | {a_canonical}
                b_variants = canonical_to_variants.get(b_canonical, set()) | {b_canonical}
                if name_b.lower() in a_variants or name_a.lower() in b_variants:
                    continue

                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                # Determine canonical
                data_a, data_b = name_data[name_a], name_data[name_b]
                if data_a['quality_score'] >= data_b['quality_score']:
                    canonical, variant = name_a, name_b
                    canonical_data, variant_data = data_a, data_b
                else:
                    canonical, variant = name_b, name_a
                    canonical_data, variant_data = data_b, data_a

                canonical_samples = self._get_sample_segments(
                    canonical, canonical_data['speakers'], self.config.n_samples
                )
                variant_samples = self._get_sample_segments(
                    variant, variant_data['speakers'], self.config.n_samples
                )

                suggestions['suggested_merges'].append({
                    'canonical': canonical,
                    'variant': variant,
                    'similarity': round(sim, 3),
                    'evidence': {
                        'canonical_hours': round(canonical_data['total_hours'], 1),
                        'canonical_episodes': canonical_data['n_episodes'],
                        'canonical_quality_score': canonical_data['quality_score'],
                        'variant_hours': round(variant_data['total_hours'], 1),
                        'variant_episodes': variant_data['n_episodes'],
                        'variant_quality_score': variant_data['quality_score'],
                    },
                    'samples': {
                        'canonical': canonical_samples,
                        'variant': variant_samples,
                    }
                })

        # Flag handles
        for name, data in name_data.items():
            name_lower = name.lower()
            if data['is_handle']:
                if name_lower in existing_aliases_lower:
                    continue
                if name_lower in unresolved_handles:
                    continue

                handle_samples = self._get_sample_segments(
                    name, data['speakers'], self.config.n_samples
                )
                suggestions['flagged_handles'].append({
                    'handle': name,
                    'hours': round(data['total_hours'], 1),
                    'episodes': data['n_episodes'],
                    'samples': handle_samples,
                })

        # Find potential do-not-merge (co-hosts)
        for i in range(n_names):
            for j in range(i + 1, n_names):
                sim = float(sim_matrix[i, j])
                if sim < 0.60:
                    continue

                name_a, name_b = names[i], names[j]
                pair_key = frozenset([name_a.lower(), name_b.lower()])

                if pair_key in do_not_merge:
                    continue

                a_canonical = alias_to_canonical.get(name_a.lower(), name_a.lower())
                b_canonical = alias_to_canonical.get(name_b.lower(), name_b.lower())
                if a_canonical == b_canonical:
                    continue

                episodes_a = set(sp.get('content_id') for sp in name_data[name_a]['speakers'])
                episodes_b = set(sp.get('content_id') for sp in name_data[name_b]['speakers'])
                shared_episodes = episodes_a & episodes_b

                if shared_episodes:
                    samples_a = self._get_sample_segments(
                        name_a, name_data[name_a]['speakers'], self.config.n_samples
                    )
                    samples_b = self._get_sample_segments(
                        name_b, name_data[name_b]['speakers'], self.config.n_samples
                    )
                    suggestions['potential_do_not_merge'].append({
                        'names': [name_a, name_b],
                        'similarity': round(sim, 3),
                        'shared_episodes': len(shared_episodes),
                        'reason': 'Co-occur in same episodes (likely co-hosts)',
                        'samples': {
                            name_a: samples_a,
                            name_b: samples_b,
                        }
                    })

        return suggestions

    def _write_suggestions(self, suggestions: Dict) -> None:
        """Write suggestions to files."""
        # YAML output (human-readable)
        output = f"""# Name Standardization Suggestions
# Generated: {suggestions['generated_at']}
# Threshold: {suggestions['threshold_used']}
#
# INSTRUCTIONS:
# 1. Review each suggestion below
# 2. For approved merges: add to config/name_aliases.yaml under 'aliases'
# 3. For false positives: add to 'do_not_merge' in name_aliases.yaml
# 4. For handles: research real name and add to 'aliases', or add to 'unresolved_handles'
#
# This file is regenerated each Phase 3 run.
# Review dashboard: streamlit run dashboards/name_review.py --server.port 8502

"""
        output += "# " + "=" * 76 + "\n"
        output += "# SUGGESTED MERGES\n"
        output += "# " + "=" * 76 + "\n\n"

        if suggestions['suggested_merges']:
            output += "suggested_merges:\n"
            for merge in sorted(suggestions['suggested_merges'], key=lambda x: -x['similarity']):
                output += f"  - canonical: \"{merge['canonical']}\"\n"
                output += f"    variant: \"{merge['variant']}\"\n"
                output += f"    similarity: {merge['similarity']}\n"
                ev = merge['evidence']
                output += f"    # {ev['canonical_hours']}h vs {ev['variant_hours']}h\n\n"
        else:
            output += "suggested_merges: []\n\n"

        output += "# " + "=" * 76 + "\n"
        output += "# FLAGGED HANDLES\n"
        output += "# " + "=" * 76 + "\n\n"

        if suggestions['flagged_handles']:
            output += "flagged_handles:\n"
            for handle in sorted(suggestions['flagged_handles'], key=lambda x: -x['hours']):
                output += f"  - handle: \"{handle['handle']}\"\n"
                output += f"    hours: {handle['hours']}\n\n"
        else:
            output += "flagged_handles: []\n\n"

        output += "# " + "=" * 76 + "\n"
        output += "# POTENTIAL DO-NOT-MERGE\n"
        output += "# " + "=" * 76 + "\n\n"

        if suggestions['potential_do_not_merge']:
            output += "potential_do_not_merge:\n"
            for pair in sorted(suggestions['potential_do_not_merge'], key=lambda x: -x['similarity']):
                output += f"  - names: [\"{pair['names'][0]}\", \"{pair['names'][1]}\"]\n"
                output += f"    similarity: {pair['similarity']}\n"
                output += f"    shared_episodes: {pair['shared_episodes']}\n\n"
        else:
            output += "potential_do_not_merge: []\n"

        try:
            with open(NAME_SUGGESTIONS_PATH, 'w') as f:
                f.write(output)
            logger.info(f"YAML suggestions written to: {NAME_SUGGESTIONS_PATH}")
        except Exception as e:
            logger.warning(f"Failed to write YAML: {e}")

        # JSON output (for dashboard)
        try:
            with open(NAME_SUGGESTIONS_JSON_PATH, 'w') as f:
                json.dump(suggestions, f, indent=2, default=str)
            logger.info(f"JSON suggestions written to: {NAME_SUGGESTIONS_JSON_PATH}")
        except Exception as e:
            logger.warning(f"Failed to write JSON: {e}")

    async def run(self, project: str) -> Dict:
        """Run Phase 3 name standardization."""
        logger.info("=" * 80)
        logger.info("PHASE 3: NAME STANDARDIZATION")
        logger.info("=" * 80)
        logger.info(f"Project: {project}")
        logger.info(f"Threshold: {self.config.merge_threshold}")
        logger.info("-" * 80)

        # Load verified speakers
        logger.info("Loading Phase 2 verified speakers...")
        speakers_by_name = self._load_verified_speakers(project)
        self.stats['total_verified_speakers'] = sum(len(v) for v in speakers_by_name.values())
        self.stats['unique_names_before'] = len(speakers_by_name)
        logger.info(f"Loaded {self.stats['total_verified_speakers']} speakers with {self.stats['unique_names_before']} unique names")

        # Load aliases
        logger.info("-" * 80)
        logger.info("Loading name aliases...")
        alias_to_canonical, unresolved_handles, do_not_merge = self._load_name_aliases()

        # Apply static aliases
        logger.info("-" * 80)
        logger.info("Applying static aliases...")
        speakers_by_name, static_mappings, _ = self._apply_static_aliases(
            speakers_by_name, alias_to_canonical, unresolved_handles
        )
        self.stats['static_alias_merges'] = len(static_mappings)
        self.stats['unique_names_after'] = len(speakers_by_name)
        logger.info(f"Applied {len(static_mappings)} static merges: {self.stats['unique_names_before']} -> {self.stats['unique_names_after']} names")

        # Analyze embeddings
        logger.info("-" * 80)
        logger.info("Analyzing embeddings for suggestions...")
        suggestions = self._analyze_embeddings(
            speakers_by_name, alias_to_canonical, unresolved_handles, do_not_merge
        )

        self.stats['suggested_merges'] = len(suggestions['suggested_merges'])
        self.stats['flagged_handles'] = len(suggestions['flagged_handles'])
        self.stats['potential_do_not_merge'] = len(suggestions['potential_do_not_merge'])

        logger.info(f"Generated: {self.stats['suggested_merges']} merge suggestions, "
                   f"{self.stats['flagged_handles']} flagged handles, "
                   f"{self.stats['potential_do_not_merge']} potential do-not-merge")

        # Write suggestions
        self._write_suggestions(suggestions)

        logger.info("-" * 80)
        logger.info("PHASE 3 COMPLETE")
        logger.info("-" * 80)
        logger.info(f"Review suggestions: streamlit run dashboards/name_review.py --server.port 8502")

        return self.stats


async def main():
    parser = argparse.ArgumentParser(description='Phase 3: Name Standardization')
    parser.add_argument('--project', type=str, required=True, help='Project to process')
    parser.add_argument('--threshold', type=float, default=0.70, help='Similarity threshold for merges')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    # Get date range from config if not specified
    if not args.start_date or not args.end_date:
        start_date, end_date = get_project_date_range(args.project)
        args.start_date = args.start_date or start_date
        args.end_date = args.end_date or end_date

    config = NameStandardizationConfig(
        merge_threshold=args.threshold,
    )

    strategy = NameStandardizationStrategy(
        config=config,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    try:
        await strategy.run(project=args.project)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
