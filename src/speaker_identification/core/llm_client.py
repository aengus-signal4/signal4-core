#!/usr/bin/env python3
"""
LLM Client for Speaker Identification
======================================

Client for communicating with the LLM balancer via the unified LLM client.
Uses tier_1 model (80B Qwen3) for high-quality speaker identification.

All requests go through the balancer which handles:
- Tier-aware routing (tier_1 to 80B endpoints, tier_2 to 30B, tier_3 to 4B)
- Load balancing across endpoints
- Health monitoring and failover
"""

import json
import logging
from typing import Dict, List, Optional

from src.utils.logger import setup_worker_logger
from src.utils.llm_client import LLMClient, create_speaker_id_client
from src.speaker_identification.prompts import PromptRegistry

logger = setup_worker_logger('speaker_identification.llm_client')


class MLXLLMClient:
    """Client for LLM API using unified client."""

    def __init__(
        self,
        host: str = "10.0.0.4",
        port: int = 8002,
        tier: str = "tier_1"
    ):
        """
        Initialize LLM client.

        Args:
            host: Model server host (ignored - uses unified client)
            port: Model server port (ignored - uses unified client)
            tier: Model tier to use (default: tier_1 = 80B model)

        Note: host/port are kept for API compatibility but the unified client
        handles endpoint discovery via get_llm_server_url().
        """
        self.tier = tier

        # Create unified LLM client for speaker identification (tier_1, priority 2)
        self._client = create_speaker_id_client(tier=tier)

    async def identify_channel_hosts(
        self,
        channel_name: str,
        channel_description: str,
        platform: str,
        priority: int = 2
    ) -> List[Dict]:
        """
        Identify hosts from channel description (Phase 1A).

        Args:
            channel_name: Name of the channel
            channel_description: Channel description text
            platform: Platform (youtube, podcast, etc.)
            priority: LLM request priority (1-99, lower = higher)

        Returns:
            List of identified hosts:
            [
                {
                    "name": "Full Name",
                    "role": "host",
                    "confidence": 0.95,
                    "reasoning": "Explicitly stated as host"
                }
            ]
        """
        prompt = PromptRegistry.phase1a_channel_hosts(
            channel_name,
            channel_description,
            platform
        )

        response = await self._call_llm(prompt, priority=priority)
        return self._parse_host_response(response)

    async def verify_host_identity(
        self,
        speaker_context: Dict,
        priority: int = 2
    ) -> Dict:
        """
        Verify if a speaker is the channel host using transcript evidence (Phase 2).

        Args:
            speaker_context: Context from get_speaker_transcript_context()
            priority: LLM request priority (1-99, lower = higher)

        Returns:
            {
                'is_host': bool,
                'confidence': 'certain' | 'very_likely' | 'unlikely',
                'reasoning': str,
                'name': str (if identifiable)
            }
        """
        # Build context sections for prompt
        context_parts = []
        if speaker_context.get('turn_before'):
            context_parts.append(f"[Previous speaker]: {speaker_context['turn_before']}")
        context_parts.append(f"[This speaker - FIRST]: {speaker_context['first_utterance']}")
        if speaker_context['total_turns'] > 2:
            context_parts.append(f"[This speaker speaks {speaker_context['total_turns']} times total, {speaker_context['duration_pct']:.1f}% of episode]")
        context_parts.append(f"[This speaker - LAST]: {speaker_context['last_utterance']}")
        if speaker_context.get('turn_after'):
            context_parts.append(f"[Next speaker]: {speaker_context['turn_after']}")
        context_text = "\n\n".join(context_parts)

        # Build cluster info if available
        cluster_info = ""
        target_host = "Unknown"
        if 'host_similarities' in speaker_context and speaker_context.get('host_similarities'):
            sims = speaker_context['host_similarities']
            target_host = speaker_context.get('best_host_name', 'Unknown')
            cluster_info = f"\n\nVOICE MATCH TARGET: '{target_host}'"
            cluster_info += f"\n\nThis speaker's voice similarity to each known host:"
            for host_name, sim in sorted(sims.items(), key=lambda x: -x[1]):
                marker = " <- BEST MATCH" if host_name == target_host else ""
                cluster_info += f"\n  - {host_name}: {sim:.3f}{marker}"
            cluster_info += f"\n\nQUESTION: Is this speaker '{target_host}'?"
        elif 'best_cluster_id' in speaker_context and 'cluster_similarities' in speaker_context:
            best_cluster = speaker_context['best_cluster_id']
            sims = speaker_context['cluster_similarities']
            cluster_info = f"\n\nMULTI-HOST CONTEXT: This channel has {len(sims)} potential host clusters."
            cluster_info += f"\nThis speaker's voice similarity scores:"
            for cid, sim in sorted(sims.items()):
                marker = " <- BEST MATCH" if cid == best_cluster else ""
                cluster_info += f"\n  - Host Cluster {cid + 1}: {sim:.3f}{marker}"
            cluster_info += f"\nBased on voice similarity, this speaker best matches Host Cluster {best_cluster + 1}."

        prompt = PromptRegistry.phase2_host_verification(
            target_host=target_host,
            episode_title=speaker_context['episode_title'],
            episode_description=speaker_context['episode_description'],
            context_text=context_text,
            cluster_info=cluster_info
        )

        logger.info("=" * 80)
        logger.info("LLM HOST VERIFICATION REQUEST")
        logger.info("=" * 80)
        logger.info(f"Episode: {speaker_context.get('episode_title', 'N/A')}")
        logger.info(f"Speaker ID: {speaker_context.get('speaker_id', 'N/A')}")
        logger.info(f"\nPROMPT:\n{prompt}")
        logger.info("=" * 80)

        response = await self._call_llm(prompt, priority=priority)

        logger.info("\nLLM RESPONSE:")
        logger.info(response)
        logger.info("=" * 80)

        parsed = self._parse_host_verification_response(response)

        logger.info(f"\nPARSED RESULT:")
        logger.info(f"  is_host: {parsed['is_host']}")
        logger.info(f"  confidence: {parsed['confidence']}")
        logger.info(f"  reasoning: {parsed['reasoning']}")
        logger.info(f"  name: {parsed.get('name', 'N/A')}")
        logger.info("=" * 80)

        return parsed

    async def verify_cluster_is_host(
        self,
        context: Dict,
        priority: int = 2
    ) -> Dict:
        """
        Identify which host a speaker cluster represents (Phase 2B).

        Used during centroid bootstrapping to identify the speaker from
        transcript evidence. Can identify ANY known host, not just the target.

        Uses different strategies based on channel type:
        - Single-host: Allows behavioral inference
        - Multi-host: Requires explicit name evidence, uses metadata context

        Args:
            context: Dict with keys:
                - all_host_names: List of all known host names for channel
                - cluster_size: Number of speakers in cluster
                - transcript_samples: List of sample transcripts from cluster
                - metadata_matches: (optional) Episode overlap data for multi-host

        Returns:
            {
                'identified_host': str (name from known hosts or 'unknown'),
                'confidence': 'certain' | 'very_likely' | 'unlikely',
                'reasoning': str
            }
        """
        all_hosts = context.get('all_host_names', [])
        qualified_hosts = context.get('qualified_hosts', [])
        target_host = context.get('target_host', all_hosts[0] if all_hosts else '')
        total_channel_episodes = context.get('total_channel_episodes', 0)

        # Determine prompt strategy based on:
        # 1. Number of qualified hosts - if multiple, we MUST distinguish between them
        # 2. Host dominance - a truly dominant host (80%+ of total episodes) can use behavioral
        #
        # Key insight: Even if one host appears more often, if there are multiple qualified hosts,
        # we need name evidence to tell them apart. Behavioral inference can identify "a host"
        # but not "which host".
        DOMINANCE_THRESHOLD = 0.80

        host_episode_count = next(
            (h['count'] for h in qualified_hosts if h['name'] == target_host),
            0
        )

        # Use total channel episodes for dominance calculation
        episode_ratio = host_episode_count / total_channel_episodes if total_channel_episodes > 0 else 0

        # Only use behavioral prompt if:
        # 1. There's only ONE qualified host (no ambiguity), OR
        # 2. This host appears in 80%+ of ALL episodes (truly dominant)
        num_qualified_hosts = len(qualified_hosts)
        use_behavioral_prompt = (
            num_qualified_hosts == 1 or
            episode_ratio >= DOMINANCE_THRESHOLD
        )

        strategy = "dominant-host (behavioral)" if use_behavioral_prompt else "multi-host (name-evidence)"

        # Build prompt using registry - different prompts based on dominance
        samples_text = PromptRegistry.build_transcript_samples_text(context.get('transcript_samples', []))

        if use_behavioral_prompt:
            # Dominant host (80%+ episodes): allow behavioral inference
            prompt = PromptRegistry.phase2b_single_host_identification(
                host_name=target_host,
                cluster_size=context.get('cluster_size', 0),
                samples_text=samples_text
            )
        else:
            # Non-dominant host: require name evidence
            prompt = PromptRegistry.phase2b_cluster_identification(
                all_host_names=all_hosts,
                cluster_size=context.get('cluster_size', 0),
                samples_text=samples_text,
                metadata_section=""  # Don't pass metadata - we want pure name evidence
            )

        logger.info("=" * 80)
        logger.info(f"LLM CLUSTER IDENTIFICATION REQUEST ({strategy})")
        logger.info("=" * 80)
        logger.info(f"Target host: {target_host} ({host_episode_count}/{total_channel_episodes} total episodes = {episode_ratio:.1%})")
        logger.info(f"Qualified hosts: {num_qualified_hosts}, threshold: {DOMINANCE_THRESHOLD:.0%}")
        logger.info(f"Known hosts: {all_hosts}")
        logger.info(f"Cluster size: {context.get('cluster_size', 'N/A')}")
        if context.get('metadata_matches'):
            logger.info(f"Metadata matches: {len(context['metadata_matches'])} hosts with overlap")
        logger.info(f"\nPROMPT:\n{prompt}")
        logger.info("=" * 80)

        response = await self._call_llm(prompt, priority=priority)

        logger.info("\nLLM RESPONSE:")
        logger.info(response)
        logger.info("=" * 80)

        parsed = self._parse_cluster_identification_response(response)

        logger.info(f"\nPARSED RESULT:")
        logger.info(f"  identified_host: {parsed['identified_host']}")
        logger.info(f"  confidence: {parsed['confidence']}")
        logger.info(f"  reasoning: {parsed['reasoning']}")
        logger.info("=" * 80)

        return parsed

    async def extract_episode_speakers(
        self,
        channel_hosts: List[str],
        episode_title: str,
        episode_description: str,
        publish_date: str,
        priority: int = 2
    ) -> List[Dict]:
        """
        Extract speaker names from episode metadata (Phase 1B).

        Args:
            channel_hosts: List of known host names from channel
            episode_title: Episode title
            episode_description: Episode description text
            publish_date: Publication date
            priority: LLM request priority (1-99, lower = higher)

        Returns:
            List of identified speakers:
            [
                {
                    "name": "Andrew Huberman",
                    "role": "host",
                    "confidence": "certain",
                    "reasoning": "Regular channel host"
                },
                {
                    "name": "Dr. Jack Feldman",
                    "role": "guest",
                    "confidence": "certain",
                    "reasoning": "Named in title and description"
                }
            ]
        """
        # Truncate extremely long descriptions to prevent LLM timeouts
        # Some podcast feeds include full transcripts in descriptions (150K+ chars)
        MAX_DESC_LENGTH = 10000  # ~2500 words, plenty for metadata extraction
        if len(episode_description) > MAX_DESC_LENGTH:
            logger.warning(f"Truncating description from {len(episode_description)} to {MAX_DESC_LENGTH} chars")
            episode_description = episode_description[:MAX_DESC_LENGTH] + "... [truncated]"

        prompt = PromptRegistry.phase1b_episode_speakers(
            channel_hosts,
            episode_title,
            episode_description,
            publish_date
        )

        response = await self._call_llm(prompt, priority=priority)
        return self._parse_episode_speakers_response(response)

    # =========================================================================
    # DEPRECATED: Old _build_* methods removed - now using PromptRegistry
    # See: src/speaker_identification/prompts/registry.py
    # =========================================================================

    async def close(self):
        """Close the persistent session. Call when done with client."""
        await self._client.close()

    async def _call_llm(
        self,
        prompt: str,
        priority: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> str:
        """Call LLM via unified client with slow backoff for Phase 2.

        Uses 5 retries with slower exponential backoff (base 3 = 3,9,27,81s)
        to handle temporary LLM server issues gracefully.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a precise speaker identification assistant. Always return valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Use 5 retries with slower backoff (3^n seconds: 3, 9, 27, 81)
        # for speaker ID which is LLM-heavy and can wait for server recovery
        return await self._client.call(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            priority=priority,
            retries=5,
            backoff_base=3.0,
        )

    async def _call_llm_batch(
        self,
        prompts: List[str],
        priority: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> List[str]:
        """Call LLM batch endpoint via unified client.

        Args:
            prompts: List of prompt strings
            priority: LLM request priority (1-99)
            temperature: Sampling temperature
            max_tokens: Max tokens per response

        Returns:
            List of response strings (same order as prompts)
        """
        logger.info(f"Calling batch endpoint with {len(prompts)} prompts")

        # Build message batches
        message_batches = [
            [
                {
                    "role": "system",
                    "content": "You are a precise speaker identification assistant. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            for prompt in prompts
        ]

        return await self._client.call_batch(
            message_batches=message_batches,
            temperature=temperature,
            max_tokens=max_tokens,
            priority=priority,
        )

    async def extract_episode_speakers_batch(
        self,
        episodes: List[Dict],
        priority: int = 2,
        batch_size: int = 4
    ) -> List[Dict]:
        """
        Extract speakers from multiple episodes in batches.

        Args:
            episodes: List of episode dicts with keys:
                - channel_hosts: List[str]
                - title: str
                - description: str
                - publish_date: str
            priority: LLM request priority (1-99)
            batch_size: Number of episodes per batch (default 4)

        Returns:
            List of parsed results (same order as input episodes):
            [
                {'speakers': [...], 'mentioned': [...]},
                ...
            ]
        """
        results = []
        total_batches = (len(episodes) + batch_size - 1) // batch_size

        logger.info(f"Processing {len(episodes)} episodes in {total_batches} batches of {batch_size}")

        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Build prompts for this batch
            prompts = [
                self._build_episode_extraction_prompt(
                    ep['channel_hosts'],
                    ep['title'],
                    ep['description'],
                    ep['publish_date']
                )
                for ep in batch
            ]

            logger.info(f"Sending batch {batch_num}/{total_batches} with {len(batch)} episodes to batch endpoint")

            # Call batch endpoint
            responses = await self._call_llm_batch(prompts, priority=priority)

            logger.info(f"Batch {batch_num} returned {len(responses)} responses")

            # Parse each response
            for response in responses:
                results.append(self._parse_episode_speakers_response(response))

        return results

    def _parse_host_verification_response(self, response: str) -> Dict:
        """Parse Phase 2 host verification response with categorical evidence fields."""
        try:
            # Clean markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response

            data = json.loads(response)

            # Validate required fields
            if "is_host" not in data:
                logger.warning(f"Invalid verification response format: {response[:200]}")
                return {
                    'is_host': False,
                    'confidence': 'unlikely',
                    'reasoning': 'Invalid response format',
                    'name': '',
                    'evidence_type': 'none',
                    'evidence_source': 'unknown',
                    'evidence_quote': ''
                }

            # Extract evidence fields (new structured format)
            evidence = data.get('evidence', {})

            return {
                'is_host': bool(data.get('is_host', False)),
                'confidence': data.get('confidence', 'unlikely'),
                'reasoning': evidence.get('interpretation', data.get('reasoning', '')),
                'name': data.get('name', ''),
                # New categorical evidence fields
                'evidence_type': evidence.get('type', 'none'),
                'evidence_source': evidence.get('speaker_source', 'unknown'),
                'evidence_quote': evidence.get('quote', '')
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse verification response: {e}\nResponse: {response[:500]}")
            return {
                'is_host': False,
                'confidence': 'unlikely',
                'reasoning': f'JSON parse error: {str(e)}',
                'name': '',
                'evidence_type': 'none',
                'evidence_source': 'unknown',
                'evidence_quote': ''
            }

    def _parse_cluster_identification_response(self, response: str) -> Dict:
        """Parse Phase 2B cluster identification response.

        Format (v2.1 - analysis first, then JSON):
        ANALYSIS: [reasoning]
        JSON: {"speaker_name": ..., "confidence": ...}

        Sets identified_host = speaker_name for backwards compatibility.
        """
        try:
            response = response.strip()

            # Extract reasoning from ANALYSIS section
            reasoning = ''
            if 'ANALYSIS:' in response:
                parts = response.split('ANALYSIS:', 1)
                if len(parts) > 1:
                    analysis_part = parts[1]
                    if 'JSON:' in analysis_part:
                        reasoning = analysis_part.split('JSON:', 1)[0].strip()
                    else:
                        reasoning = analysis_part.strip()

            # Extract JSON - look for { ... } pattern
            json_str = None
            if 'JSON:' in response:
                json_part = response.split('JSON:', 1)[1].strip()
                # Find the JSON object
                if json_part.startswith('{'):
                    # Find matching closing brace
                    brace_count = 0
                    for i, c in enumerate(json_part):
                        if c == '{':
                            brace_count += 1
                        elif c == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = json_part[:i+1]
                                break

            # Fallback: try to find any JSON object in response
            if not json_str:
                import re
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    json_str = json_match.group()

            if not json_str:
                raise json.JSONDecodeError("No JSON found", response, 0)

            # Clean markdown if present
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str

            data = json.loads(json_str)

            speaker_name = data.get('speaker_name') or data.get('identified_host', 'unknown')
            confidence = data.get('confidence', 'unlikely')

            # Normalize speaker_name
            if not speaker_name or speaker_name.lower() in ['unknown', 'none', '']:
                speaker_name = 'unknown'

            # For backwards compatibility
            identified_host = speaker_name if speaker_name != 'unknown' else 'unknown'

            # Extract evidence fields (new structured format)
            evidence = data.get('evidence', {})

            return {
                'identified_host': identified_host,
                'speaker_name': speaker_name,
                'role': 'host' if speaker_name != 'unknown' else 'unknown',
                'is_expected_host': speaker_name != 'unknown',
                'confidence': confidence,
                'reasoning': reasoning or data.get('reasoning', ''),
                # New categorical evidence fields
                'evidence_type': evidence.get('type', 'none'),
                'evidence_source': evidence.get('speaker_source', 'unknown'),
                'evidence_quote': evidence.get('quote', '')
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cluster identification response: {e}\nResponse: {response[:500]}")
            return {
                'identified_host': 'unknown',
                'speaker_name': 'unknown',
                'role': 'unknown',
                'is_expected_host': False,
                'confidence': 'unlikely',
                'reasoning': f'JSON parse error: {str(e)}',
                'evidence_type': 'none',
                'evidence_source': 'unknown',
                'evidence_quote': ''
            }

    def _parse_host_response(self, response: str) -> List[Dict]:
        """Parse Phase 1A host identification response."""
        try:
            # Clean markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response

            data = json.loads(response)

            if "hosts" not in data:
                logger.warning(f"Invalid host response format: {response[:200]}")
                return []

            return data["hosts"]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse host response: {e}\nResponse: {response[:500]}")
            return []

    def _parse_episode_speakers_response(self, response: str) -> Dict:
        """Parse Phase 1B episode speaker extraction response.

        Returns:
            Dict with 'speakers' and 'mentioned' lists
        """
        try:
            # Clean markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response

            data = json.loads(response)

            # Return both speakers and mentioned lists
            return {
                'speakers': data.get("speakers", []),
                'mentioned': data.get("mentioned", [])
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse speaker response: {e}\nResponse: {response[:500]}")
            return {'speakers': [], 'mentioned': []}

    async def consolidate_channel_names(
        self,
        channel_name: str,
        channel_description: str,
        host_distribution: Dict[str, int],
        priority: int = 3
    ) -> List[Dict]:
        """
        Consolidate name variations into canonical forms for a channel.

        Uses tier_1 (80B model) for high quality name resolution.

        Args:
            channel_name: Name of the channel
            channel_description: Channel description
            host_distribution: Dict mapping name -> episode count
            priority: LLM request priority (3 = background batch task)

        Returns:
            List of consolidated people:
            [
                {
                    "canonical_name": "Ian Sénéchal",
                    "variations": ["Ian", "Ian Senechal", "Ian Sénéchal"],
                    "total_episodes": 1085
                },
                ...
            ]
        """
        # Filter to names appearing in 5+ episodes to keep prompt manageable
        filtered_dist = {name: count for name, count in host_distribution.items() if count >= 5}

        if len(filtered_dist) < 2:
            logger.info("Not enough frequent names to consolidate")
            return []

        # Format distribution as text
        dist_lines = [f"{name}: {count}" for name, count in
                      sorted(filtered_dist.items(), key=lambda x: -x[1])]
        dist_text = "\n".join(dist_lines)

        prompt = f"""Identify the PRIMARY HOSTS of this podcast channel and consolidate name variations.

CHANNEL: {channel_name}
DESCRIPTION: {channel_description}

HOST NAME DISTRIBUTION (name: episode_count):
{dist_text}

YOUR TASK:
1. Consolidate name variations (same person with different spellings/formats)
2. Choose the best canonical name from the variations

RULES:
- Prefer full names with proper accents when available
- If only first name exists, use that as canonical
- Merge: accented/unaccented ("Sénéchal"/"Senechal"), titles ("Dr. X"/"X"), spelling variants

Return ONLY valid JSON with ALL people consolidated:
{{
  "people": [
    {{
      "canonical_name": "Best name from variations",
      "variations": ["all", "name", "forms"],
      "total_episodes": 123
    }}
  ]
}}"""

        response = await self._call_llm(prompt, priority=priority, max_tokens=4096)
        return self._parse_consolidation_response(response)

    def _parse_consolidation_response(self, response: str) -> List[Dict]:
        """Parse name consolidation response."""
        try:
            # Clean markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response

            data = json.loads(response)

            if "people" not in data:
                logger.warning(f"Invalid consolidation response format: {response[:200]}")
                return []

            return data["people"]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse consolidation response: {e}\nResponse: {response[:500]}")
            return []
