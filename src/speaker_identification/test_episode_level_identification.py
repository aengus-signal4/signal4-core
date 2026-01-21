"""
Episode-Level Speaker Identification Test
==========================================
Alternative approach: identify all speakers in an episode at once.
"""
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Optional

import aiohttp

sys.path.insert(0, '/Users/signal4/content_processing')

from sqlalchemy import text
from src.database.session import get_session
from src.speaker_identification.core.context_builder import ContextBuilder

LLM_ENDPOINT = "http://10.0.0.4:8002/llm-request"


@dataclass
class EpisodeSpeaker:
    speaker_id: int
    label: str
    duration_pct: float
    turn_count: int
    assigned_name: Optional[str]
    first_utterance: str
    last_utterance: str
    turn_before_first: Optional[str]
    turn_after_first: Optional[str]
    turn_before_last: Optional[str]
    turn_after_last: Optional[str]


@dataclass
class EpisodeData:
    content_id: str
    title: str
    channel_name: str
    description: str
    hosts: list[str]
    guests: list[str]
    speakers: list[EpisodeSpeaker]


def parse_json_field(value) -> list[str]:
    """Parse JSON field into list of names."""
    if not value:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return [item.get('name') for item in value if isinstance(item, dict) and item.get('name')]


def get_sample_episodes(project: str = 'CPRMV') -> list[dict]:
    """Get sample episodes with varying speaker counts."""
    samples = []

    with get_session() as session:
        for speaker_count in [1, 2, 3, 4]:
            operator = '=' if speaker_count < 4 else '>='
            # Use a CTE to pre-calculate significant speaker counts
            query = text(f"""
                WITH speaker_counts AS (
                    SELECT
                        s.content_id,
                        COUNT(*) as sig_speakers
                    FROM speakers s
                    JOIN content c ON s.content_id = c.content_id
                    WHERE :project = ANY(c.projects)
                    AND c.is_stitched = true
                    AND c.duration > 0
                    AND s.duration / NULLIF(c.duration, 0) >= 0.10
                    GROUP BY s.content_id
                    HAVING COUNT(*) {operator} :count
                )
                SELECT
                    c.content_id,
                    c.title,
                    ch.display_name as channel_name,
                    c.description,
                    c.hosts,
                    c.guests,
                    sc.sig_speakers
                FROM speaker_counts sc
                JOIN content c ON sc.content_id = c.content_id
                JOIN channels ch ON c.channel_id = ch.id
                ORDER BY RANDOM()
                LIMIT 13
            """)

            results = session.execute(query, {'project': project, 'count': speaker_count}).fetchall()
            for row in results:
                samples.append({
                    'content_id': row.content_id,
                    'title': row.title,
                    'channel_name': row.channel_name,
                    'description': row.description,
                    'hosts': parse_json_field(row.hosts),
                    'guests': parse_json_field(row.guests),
                    'sig_speakers': row.sig_speakers
                })

    return samples


def get_episode_speakers(content_id: str) -> list[EpisodeSpeaker]:
    """Get all speakers in an episode with >=10% duration."""
    cb = ContextBuilder()
    speakers = []

    with get_session() as session:
        query = text("""
            SELECT
                s.id as speaker_id,
                s.local_speaker_id as speaker_label,
                s.duration,
                c.duration as ep_duration,
                ROUND((100.0 * s.duration / NULLIF(c.duration, 0))::numeric, 1) as duration_pct,
                (SELECT COUNT(DISTINCT turn_index) FROM sentences sent
                 WHERE sent.speaker_id = s.id) as turn_count,
                si.primary_name as assigned_name
            FROM speakers s
            JOIN content c ON s.content_id = c.content_id
            LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
            WHERE c.content_id = :content_id
            AND s.duration / NULLIF(c.duration, 0) >= 0.10
            ORDER BY s.duration DESC
        """)

        results = session.execute(query, {'content_id': content_id}).fetchall()

        for row in results:
            # Get transcript context for this speaker
            context = cb.get_speaker_transcript_context(row.speaker_id)

            if not context:
                continue

            speakers.append(EpisodeSpeaker(
                speaker_id=row.speaker_id,
                label=row.speaker_label or f"SPEAKER_{row.speaker_id}",
                duration_pct=float(row.duration_pct or 0),
                turn_count=row.turn_count or 0,
                assigned_name=row.assigned_name,
                first_utterance=context.get('first_utterance', ''),
                last_utterance=context.get('last_utterance', ''),
                turn_before_first=context.get('turn_before_first'),
                turn_after_first=context.get('turn_after_first'),
                turn_before_last=context.get('turn_before_last'),
                turn_after_last=context.get('turn_after_last')
            ))

    return speakers


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text to max chars."""
    if not text:
        return "N/A"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_episode_prompt(episode: EpisodeData) -> str:
    """Build LLM prompt for episode-level identification."""

    # Build likely speakers list
    likely_speakers = []
    for h in episode.hosts:
        if h and h not in likely_speakers:
            likely_speakers.append(h)
    for g in episode.guests:
        if g and g not in likely_speakers:
            likely_speakers.append(g)

    likely_str = ", ".join(likely_speakers) if likely_speakers else "Unknown"

    # Build speaker labels list
    speaker_labels = [s.label for s in episode.speakers]

    # Build per-speaker sections
    speaker_sections = []
    for spk in episode.speakers:
        section = f"""
{spk.label} spoke for {spk.duration_pct:.1f}% of the episode ({spk.turn_count} turns).

First appearance:
[Before]: {truncate_text(spk.turn_before_first)}
[{spk.label}]: {truncate_text(spk.first_utterance)}
[After]: {truncate_text(spk.turn_after_first)}

Last appearance:
[Before]: {truncate_text(spk.turn_before_last)}
[{spk.label}]: {truncate_text(spk.last_utterance)}
[After]: {truncate_text(spk.turn_after_last)}
"""
        speaker_sections.append(section)

    prompt = f"""EPISODE SPEAKER IDENTIFICATION
==============================
Identify the speakers in this podcast episode.

EPISODE: {episode.title}
CHANNEL: {episode.channel_name}
DESCRIPTION: {truncate_text(episode.description, 500)}

LIKELY SPEAKERS: {likely_str}

We need to identify: {", ".join(speaker_labels)}

---
{"---".join(speaker_sections)}
---

INSTRUCTIONS:
1. Use transcript evidence to identify speakers (self-ID, introductions, being addressed)
2. Hosts typically have higher duration % and open/close episodes
3. If no evidence for a speaker, mark as "unknown"

Return ONLY valid JSON:
{{
  "identifications": [
    {{
      "speaker_label": "SPEAKER_00",
      "name": "Name or unknown",
      "confidence": "certain|very_likely|unlikely",
      "evidence": {{"quote": "exact quote if found", "type": "self_id|introduced_as|addressed_as|none"}}
    }}
  ]
}}"""

    return prompt


async def run_identification(prompt: str, session: aiohttp.ClientSession) -> dict:
    """Run LLM and get identifications."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a precise speaker identification assistant. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "model": "tier_1",
        "temperature": 0.1,
        "max_tokens": 2048
    }

    try:
        async with session.post(
            LLM_ENDPOINT,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            result = await response.json()
            raw = result.get('response', '')

            # Try to parse JSON
            try:
                parsed = json.loads(raw)
                return {
                    'success': True,
                    'parsed': parsed,
                    'raw': raw,
                    'endpoint': result.get('endpoint_used')
                }
            except json.JSONDecodeError:
                return {
                    'success': False,
                    'error': 'JSON parse error',
                    'raw': raw
                }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def validate_results(episode: EpisodeData, results: dict) -> dict:
    """Compare LLM results against existing assignments."""
    validation = {
        'total_speakers': len(episode.speakers),
        'identifications': [],
        'matches': 0,
        'mismatches': 0,
        'unknown': 0
    }

    if not results.get('success') or not results.get('parsed'):
        validation['error'] = results.get('error', 'No parsed results')
        return validation

    parsed = results['parsed']
    identifications = parsed.get('identifications', [])

    # Create lookup of existing assignments
    existing = {s.label: s.assigned_name for s in episode.speakers}

    for ident in identifications:
        label = ident.get('speaker_label')
        name = ident.get('name')
        confidence = ident.get('confidence')
        evidence = ident.get('evidence', {})

        existing_name = existing.get(label)

        result = {
            'label': label,
            'llm_name': name,
            'confidence': confidence,
            'evidence_type': evidence.get('type'),
            'existing_name': existing_name
        }

        if name and name.lower() != 'unknown':
            if existing_name:
                # Compare (case-insensitive, fuzzy)
                if existing_name.lower() == name.lower():
                    result['status'] = 'MATCH'
                    validation['matches'] += 1
                else:
                    result['status'] = 'MISMATCH'
                    validation['mismatches'] += 1
            else:
                result['status'] = 'NEW'
        else:
            result['status'] = 'UNKNOWN'
            validation['unknown'] += 1

        validation['identifications'].append(result)

    return validation


async def main():
    print("Episode-Level Speaker Identification Test")
    print("=" * 50)

    # Get sample episodes
    print("\nFinding sample episodes...")
    samples = get_sample_episodes('CPRMV')
    print(f"Found {len(samples)} sample episodes")

    results_all = []

    async with aiohttp.ClientSession() as http_session:
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] {sample['title'][:60]}...")
            print(f"  Channel: {sample['channel_name']}")
            print(f"  Significant speakers: {sample['sig_speakers']}")

            # Get speakers
            speakers = get_episode_speakers(sample['content_id'])
            if not speakers:
                print("  -> No speakers found, skipping")
                continue

            print(f"  Speakers found: {len(speakers)}")
            for spk in speakers:
                existing = f" (assigned: {spk.assigned_name})" if spk.assigned_name else ""
                print(f"    - {spk.label}: {spk.duration_pct:.1f}%, {spk.turn_count} turns{existing}")

            # Build episode data
            episode = EpisodeData(
                content_id=sample['content_id'],
                title=sample['title'],
                channel_name=sample['channel_name'],
                description=sample['description'] or '',
                hosts=sample['hosts'],
                guests=sample['guests'],
                speakers=speakers
            )

            # Build prompt
            prompt = build_episode_prompt(episode)

            # Run LLM
            print("  Running LLM identification...")
            llm_result = await run_identification(prompt, http_session)

            if not llm_result.get('success'):
                print(f"  -> LLM error: {llm_result.get('error')}")
                continue

            # Validate
            validation = validate_results(episode, llm_result)

            # Print results
            print(f"  Results: {validation['matches']} matches, {validation['mismatches']} mismatches, {validation['unknown']} unknown")
            for ident in validation.get('identifications', []):
                status_icon = {'MATCH': '+', 'MISMATCH': 'X', 'NEW': '?', 'UNKNOWN': '-'}.get(ident['status'], '?')
                existing_str = f" (was: {ident['existing_name']})" if ident['existing_name'] else ""
                print(f"    [{status_icon}] {ident['label']}: {ident['llm_name']} ({ident['confidence']}, {ident['evidence_type']}){existing_str}")

            results_all.append({
                'episode': sample,
                'speakers': len(speakers),
                'validation': validation,
                'prompt': prompt,
                'llm_response': llm_result
            })

            await asyncio.sleep(0.5)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    total_matches = sum(r['validation'].get('matches', 0) for r in results_all)
    total_mismatches = sum(r['validation'].get('mismatches', 0) for r in results_all)
    total_unknown = sum(r['validation'].get('unknown', 0) for r in results_all)

    print(f"Episodes processed: {len(results_all)}")
    print(f"Total identifications: {total_matches + total_mismatches + total_unknown}")
    print(f"  Matches: {total_matches}")
    print(f"  Mismatches: {total_mismatches}")
    print(f"  Unknown: {total_unknown}")

    # Save results
    output_file = 'episode_level_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
