"""
Prompt Registry
===============

Central registry for all speaker identification prompts.
Each prompt is built from shared blocks plus phase-specific instructions.

Architecture:
    blocks.py     - Reusable instruction fragments
    registry.py   - Complete prompt templates (this file)
    __init__.py   - Public API

Usage:
    from src.speaker_identification.prompts import PromptRegistry

    # Phase 1A: Extract hosts from channel description
    prompt = PromptRegistry.phase1a_channel_hosts(
        channel_name="The Daily",
        description="Hosted by Michael Barbaro...",
        platform="podcast"
    )

    # Phase 3: Identify guest from transcript
    prompt = PromptRegistry.phase3_guest_identification(
        episode_title="Interview with Dr. Smith",
        description="...",
        host_name="Michael Barbaro",
        expected_guests=["Dr. Jane Smith"],
        transcript_section="...",
        duration_pct=35.5,
        total_turns=42
    )

Prompt Versioning:
    Each prompt method has a version number in its docstring.
    When modifying prompts, increment the version and document changes.
"""

from typing import List, Optional, Dict
from . import blocks


class PromptRegistry:
    """
    Central registry for all LLM prompts used in speaker identification.

    All prompts are class methods that return formatted prompt strings.
    This makes it easy to:
    - See all available prompts in one place
    - Track which blocks each prompt uses
    - Modify prompts centrally
    - A/B test different versions
    """

    # =========================================================================
    # PHASE 1A: Channel Host Extraction
    # =========================================================================

    @classmethod
    def phase1a_channel_hosts(
        cls,
        channel_name: str,
        description: str,
        platform: str
    ) -> str:
        """
        Build prompt for extracting hosts from channel description.

        Version: 1.0
        Used by: llm_client.py -> identify_channel_hosts()

        Args:
            channel_name: Name of the channel
            description: Channel description text
            platform: Platform (youtube, podcast, etc.)

        Returns:
            Formatted prompt string
        """
        return f"""You are identifying podcast/video channel hosts from channel descriptions.

CHANNEL: {channel_name} ({platform})
DESCRIPTION: {description}

Identify the host(s) of this channel. Look for:
- Explicit mentions: "hosted by X", "presented by X"
- Channel ownership: "X's podcast", "X's channel"
- First-person descriptions that name the host

{blocks.NO_HALLUCINATION_RULE}

Confidence levels:
- "certain": Name explicitly stated in description
- "very_likely": Name clearly implied (e.g., "John's Podcast" -> John)
- "somewhat_likely": Reasonable inference from description text only

{blocks.JSON_HOST_EXTRACTION}"""

    # =========================================================================
    # PHASE 1B: Episode Speaker Extraction
    # =========================================================================

    @classmethod
    def phase1b_episode_speakers(
        cls,
        channel_hosts: List[str],
        episode_title: str,
        episode_description: str,
        publish_date: str
    ) -> str:
        """
        Build prompt for extracting speakers from episode metadata.

        Version: 1.0
        Used by: llm_client.py -> extract_episode_speakers()

        Args:
            channel_hosts: List of known host names from channel
            episode_title: Episode title
            episode_description: Episode description text
            publish_date: Publication date

        Returns:
            Formatted prompt string
        """
        hosts_text = ", ".join(channel_hosts) if channel_hosts else "Unknown"

        return f"""Extract people from podcast/video metadata. Use FULL NAMES whenever possible.

KNOWN HOSTS: {hosts_text}
TITLE: {episode_title}
DESCRIPTION: {episode_description}

Categorize people into TWO groups:

1. SPEAKERS - People who actually appear and speak IN THIS SPECIFIC EPISODE:
   - Only include hosts who are EXPLICITLY MENTIONED in this episode's title/description
   - For multi-host channels: do NOT assume all hosts appear in every episode
   - If title/description mentions a specific host (e.g., "Jesse talks to...", "with Noor"), include them
   - If NO host is mentioned in the title/description, do NOT include any hosts
   - Guests who are interviewed/join the discussion
   - Patterns: "with Dr. X", "Dr. X joins", "featuring X"

2. MENTIONED - People only discussed/referenced (NOT organizations or groups):
   - Politicians/public figures being analyzed (e.g., "Justin Trudeau", "Pierre Poilievre")
   - People in news stories being discussed
   - Prefer full names (first + last) but recognizable single names are acceptable (e.g., "Trudeau")
   - Do NOT include: organizations, companies, groups, or generic roles without names
   - Patterns: "about X", "analyzes X's policy", "discusses X"

{blocks.JSON_EPISODE_SPEAKERS}"""

    # =========================================================================
    # PHASE 2 (NEW): Text Evidence Collection
    # =========================================================================

    @classmethod
    def phase2_text_evidence_collection(
        cls,
        episode_title: str,
        episode_description: Optional[str],
        possible_speakers: List[str],
        transcript_context: str,
        duration_pct: float,
        total_turns: int
    ) -> str:
        """
        Build prompt for finding text evidence proving speaker identity.

        Version: 1.0
        Used by: strategies/text_evidence_collection.py

        This is the STRICT text-evidence-only prompt. The LLM must find explicit
        transcript evidence (self-introduction, being addressed, being introduced).
        Metadata/title evidence alone is NOT sufficient.

        Args:
            episode_title: Episode title
            episode_description: Episode description (may be None)
            possible_speakers: List of likely speaker names from metadata
            transcript_context: 6-turn transcript context
            duration_pct: Speaker's % of episode duration
            total_turns: Number of turns this speaker has

        Returns:
            Formatted prompt string
        """
        # Format possible speakers
        speakers_list = ", ".join(possible_speakers) if possible_speakers else "Unknown"

        desc_text = episode_description[:400] if episode_description else "No description"

        return f"""Find TRANSCRIPT evidence proving who the UNKNOWN SPEAKER is.

EPISODE: {episode_title}
DESCRIPTION: {desc_text}

KNOWN SPEAKERS ON THIS CHANNEL (for spelling reference only):
{speakers_list}

TRANSCRIPT CONTEXT:
{transcript_context}

SPEAKER METRICS: {total_turns} speaking turns, {duration_pct:.1f}% of episode duration

---

YOUR TASK: Find TRANSCRIPT EVIDENCE that identifies the UNKNOWN SPEAKER.

**START WITH FIRST APPEARANCE** - Self-introductions almost always happen early:
- Look at the [UNKNOWN SPEAKER] text in "FIRST APPEARANCE" section FIRST
- Common patterns: "Welcome to [show], I'm [Name]", "This is [Name]", "I'm [Name]"
- If you find "I'm [Name]" in FIRST APPEARANCE, that IS the identity - stop looking

VALID EVIDENCE (you MUST find one of these to return evidence_found=true):
1. **SELF_INTRO**: The UNKNOWN SPEAKER says "I'm [Name]", "My name is [Name]", "This is [Name]"
   - The quote must come FROM the [UNKNOWN SPEAKER]'s utterances (NOT [Previous speaker] or [Next speaker])

2. **ADDRESSED**: A surrounding speaker says "[Name], ..." THEN the UNKNOWN SPEAKER responds

3. **INTRODUCED**: A speaker says "Welcome [Name]" or "Joining us is [Name]" BEFORE UNKNOWN SPEAKER talks

INVALID EVIDENCE (DO NOT USE):
- Name appears in episode title/description but NOT in transcript quotes
- Speaking style, topic expertise, or duration patterns
- The UNKNOWN SPEAKER says "Thanks [Someone]" or "I agree with [Someone]" - they are ADDRESSING/thanking others, NOT self-identifying!
- Voice/embedding similarity (you cannot hear audio)
- Assumptions about who "should" be speaking

{blocks.PHONETIC_MATCHING_SHORT}

{blocks.ADDRESSING_RULE}

{blocks.JSON_TEXT_EVIDENCE}"""

    # =========================================================================
    # PHASE 2 (LEGACY): Host Verification
    # =========================================================================

    @classmethod
    def phase2_host_verification(
        cls,
        target_host: str,
        episode_title: str,
        episode_description: str,
        context_text: str,
        cluster_info: str = ""
    ) -> str:
        """
        Build prompt for verifying if a speaker is a specific host.

        Version: 1.0
        Used by: llm_client.py -> verify_host_identity()

        Args:
            target_host: Name of the host we're verifying against
            episode_title: Episode title
            episode_description: Episode description (truncated to 300 chars)
            context_text: Transcript context with speaker turns
            cluster_info: Optional cluster/similarity information

        Returns:
            Formatted prompt string
        """
        return f"""Verify if this speaker is SPECIFICALLY the person named '{target_host}'.

EPISODE: {episode_title}
DESCRIPTION: {episode_description[:300]}

TRANSCRIPT EVIDENCE:
{context_text}
{cluster_info}

**CRITICAL - UNDERSTAND THE TRANSCRIPT STRUCTURE:**
- [Previous speaker] = A DIFFERENT PERSON who spoke before this speaker
- [This speaker - FIRST] = What THIS SPEAKER said at the start
- [This speaker - LAST] = What THIS SPEAKER said at the end
- [Next speaker] = A DIFFERENT PERSON who spoke after this speaker

Only [This speaker] sections contain words from the speaker we're identifying!
[Previous speaker] and [Next speaker] are DIFFERENT PEOPLE.

CRITICAL QUESTION: Is this speaker '{target_host}'?

You must find POSITIVE EVIDENCE that this speaker IS '{target_host}'.
Voice similarity alone is NOT sufficient - you need transcript evidence.

{blocks.PHONETIC_MATCHING}

{blocks.CONFIDENCE_LEVELS_HOST_VERIFICATION}

AUTOMATIC REJECTION (is_host=false):
- Speaker self-identifies as a clearly DIFFERENT person (not a transcription variant)
- Speaker is explicitly introduced as a guest/interviewee with a different name
- No clear name evidence linking speaker to '{target_host}'

**INVALID EVIDENCE - DO NOT USE:**
- Quotes or rhetoric "associated with" or "famous" from a person
- Speaking style, vocabulary, or content themes typical of someone
- Topics the speaker discusses that match someone's known interests
- Voice similarity scores alone (must have NAME evidence)
- Host behavior (speaking time, asking questions) alone
- Self-identification from [Previous speaker] or [Next speaker] - those are DIFFERENT PEOPLE!

NOTE: Content and style are NOT identification evidence. Many people quote others or discuss
similar topics. You need the speaker to SAY their name or BE CALLED by their name.

{blocks.JSON_HOST_VERIFICATION}"""

    # =========================================================================
    # PHASE 2B: Cluster Identification
    # =========================================================================

    @classmethod
    def phase2b_cluster_identification(
        cls,
        all_host_names: List[str],
        cluster_size: int,
        samples_text: str,
        metadata_section: str = ""
    ) -> str:
        """
        Build prompt for identifying which host a speaker cluster represents.

        Version: 2.0 - Simplified to pure textual name matching
        Used by: llm_client.py -> verify_cluster_is_host()

        Args:
            all_host_names: List of all known host names for channel
            cluster_size: Number of episodes this speaker appears in (not used in prompt)
            samples_text: Formatted transcript samples
            metadata_section: Optional metadata evidence section (ignored)

        Returns:
            Formatted prompt string
        """
        hosts_list = ", ".join(all_host_names)

        return f"""Find name evidence for this speaker in the transcript.

EXPECTED HOST NAMES: {hosts_list}

TRANSCRIPT (what THIS SPEAKER said):
{samples_text}

QUESTION: Is there textual evidence that this speaker IS one of the expected hosts?

LOOK FOR:
1. SELF-IDENTIFICATION: Speaker says "I'm [Name]", "This is [Name]", "My name is [Name]"
2. BEING ADDRESSED: Someone says "[Name], what do you think?" and then this speaker responds

DO NOT USE:
- Behavioral patterns (reading sponsors, hosting style)
- Episode metadata or overlap statistics
- Role inference (speaking time, asking questions)
- Third-person references the speaker makes about themselves

ADDRESSING vs BEING ADDRESSED:
- If THIS SPEAKER says "Thanks Jesse" → they are NOT Jesse (addressing someone else)
- If ANOTHER SPEAKER says "Thanks Jesse" then THIS SPEAKER responds → they ARE Jesse

{blocks.PHONETIC_MATCHING_SHORT}

CONFIDENCE:
- "certain": Speaker explicitly self-identifies with a name matching an expected host
- "very_likely": Speaker is addressed by a name matching an expected host
- "unlikely": No name evidence found in transcript

{blocks.JSON_CLUSTER_IDENTIFICATION}"""

    @classmethod
    def phase2b_single_host_identification(
        cls,
        host_name: str,
        cluster_size: int,
        samples_text: str
    ) -> str:
        """
        Build prompt for single-host channel verification.

        Version: 1.0
        Used by: llm_client.py -> verify_cluster_is_host() when single-host

        For single-host channels, we allow behavioral inference since there's
        only ONE host to identify. High speaking time + host behavior = host.

        Args:
            host_name: The single expected host name
            cluster_size: Number of episodes this speaker appears in
            samples_text: Formatted transcript samples with duration_pct

        Returns:
            Formatted prompt string
        """
        return f"""Determine if this speaker is the HOST of this single-host podcast/show.

EXPECTED HOST: {host_name}
CLUSTER INFO: This speaker appears in {cluster_size} different episodes with the same voice.

TRANSCRIPT SAMPLES:
{samples_text}

This is a SINGLE-HOST channel. There is only ONE regular host: {host_name}.
Anyone else speaking is a guest.

IS THIS SPEAKER THE HOST? Consider:

1. NAME EVIDENCE (strongest):
   - Self-identification: "I'm [Name]", "This is [Name]"
   - Being addressed: Someone says "[Name], ..." then this speaker responds
   - NICKNAMES COUNT: "Sam" matches "Samuel", "Mike" matches "Michael", etc.

2. HOST BEHAVIOR (valid for single-host):
   - Opens/closes episodes ("Welcome to...", "Thanks for listening...")
   - Reads sponsor messages, ads, credits
   - Introduces guests, asks interview questions
   - High speaking percentage (shown above) across multiple episodes

3. GUEST INDICATORS (means NOT the host):
   - Being introduced as a guest
   - Says "Thanks for having me"
   - Only appears in 1-2 episodes with low speaking time

For single-host channels, behavioral evidence IS sufficient if strong and consistent.
A speaker who opens/closes shows, reads ads, and appears in {cluster_size} episodes is almost certainly the host.

{blocks.PHONETIC_MATCHING_SHORT}

Think step by step, then return JSON:

ANALYSIS: [Is there name evidence? If not, is there strong host behavior across samples?]

JSON:
{{
  "speaker_name": "{host_name}" or "unknown",
  "confidence": "certain" | "very_likely" | "unlikely"
}}

Use "certain" if name evidence exists.
Use "very_likely" if strong, consistent host behavior across multiple samples.
Use "unlikely" if behavior suggests guest or insufficient evidence."""

    # =========================================================================
    # PHASE 3: Guest Identification (Transcript-based)
    # =========================================================================

    @classmethod
    def phase3_guest_identification(
        cls,
        episode_title: str,
        episode_description: Optional[str],
        possible_speakers: List[str],
        transcript_section: str,
        duration_pct: float,
        total_turns: int
    ) -> str:
        """
        Build prompt for identifying a guest speaker from transcript context.

        Version: 1.3 - Simplified to use combined possible_speakers list
        Used by: strategies/guest_identification.py -> _identify_guest()

        Args:
            episode_title: Episode title
            episode_description: Episode description (can be None)
            possible_speakers: Combined list of known hosts + expected guests from episode metadata
            transcript_section: 6-turn transcript context (with speaker names when known)
            duration_pct: Speaker's duration as percentage of episode
            total_turns: Total number of turns for this speaker

        Returns:
            Formatted prompt string
        """
        # Build possible speakers hint
        if possible_speakers:
            speakers_hint = f"The UNKNOWN SPEAKER may (but does not have to be) one of the following: {', '.join(possible_speakers)}"
        else:
            speakers_hint = ""

        return f"""Identify the UNKNOWN SPEAKER from a podcast episode.

EPISODE: {episode_title}
DESCRIPTION: {episode_description or 'N/A'}

{speakers_hint}

TRANSCRIPT ({total_turns} turns, {duration_pct:.1f}% of episode):
{transcript_section}

{blocks.PHONETIC_MATCHING_SHORT}

Look for name evidence in the transcript:
1. SELF-IDENTIFICATION: "I'm [Name]", "This is [Name]", "My name is [Name]"
2. INTRODUCTION: Another speaker says "Welcome [Name]" or "Joining us is [Name]" BEFORE the unknown speaker talks
3. BEING ADDRESSED: Another speaker says "[Name], what do you think?" then the unknown speaker responds

{blocks.ADDRESSING_RULE}

{blocks.CONFIDENCE_LEVELS_GUEST_STRICT}

{blocks.JSON_GUEST_IDENTIFICATION}"""

    # =========================================================================
    # PHASE 4: Guest Propagation Verification
    # =========================================================================

    @classmethod
    def phase4_match_verification(
        cls,
        candidate_name: str,
        similarity: float,
        episode_title: str,
        episode_description: Optional[str],
        possible_speakers: List[str],
        transcript_section: str,
        duration_pct: float,
        total_turns: int
    ) -> str:
        """
        Build prompt for verifying an embedding-based match with transcript evidence.

        Version: 1.3 - Simplified to use combined possible_speakers list
        Used by: strategies/guest_propagation.py -> _verify_with_llm()

        Args:
            candidate_name: Name of the candidate identity to verify
            similarity: Embedding similarity score (0-1)
            episode_title: Episode title
            episode_description: Episode description (can be None)
            possible_speakers: Combined list of known hosts + candidate from episode metadata
            transcript_section: 6-turn transcript context (with speaker names when known)
            duration_pct: Speaker's duration as percentage of episode
            total_turns: Total number of turns for this speaker

        Returns:
            Formatted prompt string
        """
        # Build possible speakers hint
        if possible_speakers:
            speakers_hint = f"The UNKNOWN SPEAKER may (but does not have to be) one of the following: {', '.join(possible_speakers)}"
        else:
            speakers_hint = ""

        return f"""Verify if the UNKNOWN SPEAKER is {candidate_name}.

EPISODE: {episode_title}
DESCRIPTION: {episode_description or 'N/A'}

{speakers_hint}

EMBEDDING SIMILARITY: {similarity:.3f} to {candidate_name}'s voice profile

TRANSCRIPT ({total_turns} turns, {duration_pct:.1f}% of episode):
{transcript_section}

{blocks.PHONETIC_MATCHING_SHORT}

Look for name evidence matching "{candidate_name}":
1. Self-identification: "I'm {candidate_name}" or similar
2. Being addressed: Another speaker says "{candidate_name}, ..." then the unknown speaker responds

{blocks.ADDRESSING_RULE}

{blocks.CONFIDENCE_LEVELS_IDENTIFICATION}

{blocks.JSON_MATCH_VERIFICATION}"""

    @classmethod
    def phase5_multi_candidate_verification(
        cls,
        candidates: List[Dict],
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: List[str],
        transcript_section: str,
        duration_pct: float,
        total_turns: int
    ) -> str:
        """
        Build prompt for verifying speaker identity against multiple embedding candidates.

        Version: 1.0 - Multi-candidate selection
        Used by: strategies/guest_propagation.py -> _verify_with_llm()

        Args:
            candidates: List of candidate dicts with 'identity_name' and 'similarity'
            episode_title: Episode title
            episode_description: Episode description (can be None)
            known_hosts: List of known host names from episode metadata
            transcript_section: 6-turn transcript context (with speaker names when known)
            duration_pct: Speaker's duration as percentage of episode
            total_turns: Total number of turns for this speaker

        Returns:
            Formatted prompt string
        """
        # Build candidates list with similarities
        candidates_text = "\n".join([
            f"  - {c['identity_name']} (similarity: {c['similarity']:.3f})"
            for c in candidates
        ])

        # Build known hosts hint
        if known_hosts:
            hosts_hint = f"Known hosts for this show: {', '.join(known_hosts)}"
        else:
            hosts_hint = ""

        candidate_names = [c['identity_name'] for c in candidates]

        return f"""Identify the UNKNOWN SPEAKER from the candidates below, or determine they are someone else.

EPISODE: {episode_title}
DESCRIPTION: {episode_description or 'N/A'}
{hosts_hint}

VOICE MATCH CANDIDATES (by embedding similarity):
{candidates_text}

TRANSCRIPT ({total_turns} turns, {duration_pct:.1f}% of episode):
{transcript_section}

{blocks.PHONETIC_MATCHING_SHORT}

TASK: Determine if the UNKNOWN SPEAKER is one of the candidates listed above, or someone else entirely.

Look for name evidence:
1. Self-identification: "I'm [Name]" or "This is [Name]"
2. Being addressed: Another speaker says "[Name], ..." then the unknown speaker responds

{blocks.ADDRESSING_RULE}

IMPORTANT: The candidates are ordered by voice similarity. Higher similarity means closer voice match.
- If you find clear name evidence matching a candidate, select that candidate
- If no clear evidence, but one candidate is very likely based on context, you may select them
- If the evidence points to someone NOT in the candidate list, set is_match=false

{blocks.CONFIDENCE_LEVELS_IDENTIFICATION}

Respond with JSON:
```json
{{
  "is_match": true/false,
  "identified_name": "Name from candidates or null if no match",
  "confidence": "certain|very_likely|probably|unlikely",
  "reasoning": "Brief explanation of evidence found",
  "evidence": {{
    "type": "self_intro|addressed|context|none",
    "speaker_source": "self|other|unknown",
    "quote": "relevant quote if any"
  }}
}}
```"""

    # =========================================================================
    # PHASE 5: Identity Merge Detection
    # =========================================================================

    @classmethod
    def phase5_identity_merge_verification(
        cls,
        name_a: str,
        count_a: int,
        episode_count_a: int,
        duration_a: str,
        channels_a: str,
        roles_a: str,
        first_appearance_a: str,
        samples_a: str,
        name_b: str,
        count_b: int,
        episode_count_b: int,
        duration_b: str,
        channels_b: str,
        roles_b: str,
        first_appearance_b: str,
        samples_b: str,
        similarity: float
    ) -> str:
        """
        Build prompt for verifying if two identities should be merged.

        Version: 1.0
        Used by: strategies/identity_merge_detection.py -> _verify_merge_with_llm()

        The LLM determines BOTH whether to merge AND which identity to keep.

        Args:
            name_a/name_b: Primary names of the identities
            count_a/count_b: Number of speaker samples
            episode_count_a/episode_count_b: Number of episodes
            duration_a/duration_b: Total speaking duration (formatted string)
            channels_a/channels_b: Channels where identity appears
            roles_a/roles_b: Role(s) of the identity
            first_appearance_a/first_appearance_b: First appearance date
            samples_a/samples_b: Formatted transcript samples
            similarity: Embedding similarity score

        Returns:
            Formatted prompt string
        """
        return f"""Two speaker identities have very similar voice embeddings. Determine if they are the same person.

IDENTITY A: "{name_a}"
- Speaker samples: {count_a} speakers across {episode_count_a} episodes
- Total duration: {duration_a}
- Channels: {channels_a}
- Role(s): {roles_a}
- First seen: {first_appearance_a}
- Transcript samples:
{samples_a}

IDENTITY B: "{name_b}"
- Speaker samples: {count_b} speakers across {episode_count_b} episodes
- Total duration: {duration_b}
- Channels: {channels_b}
- Role(s): {roles_b}
- First seen: {first_appearance_b}
- Transcript samples:
{samples_b}

EMBEDDING SIMILARITY: {similarity:.3f} (very high - voices are nearly identical)

TASK:
1. Are these the same person? Consider:
   - Do the names sound phonetically similar? (transcription errors are common)
   - Common variations: "Ian" vs "Yan", accented chars "Sénéchal" vs "Senechal"
   - Do the speaking styles/roles match?
   - Is there any evidence they are DIFFERENT people?

2. If same person, which identity should we KEEP? Consider:
   - Which has more data (more reliable centroid)?
   - Which name is more likely correct (proper spelling, accents)?
   - Which appeared first (historical continuity)?

{blocks.PHONETIC_MATCHING_SHORT}

Return JSON:
```json
{{
  "is_same_person": true/false,
  "confidence": "certain|very_likely|probably|unlikely",
  "keep_identity": "A" or "B" or null (if not same person),
  "canonical_name": "The correct name to use" or null,
  "reasoning": "Brief explanation",
  "evidence": {{
    "name_match": "exact|phonetic|different",
    "role_consistency": "same|different|unclear",
    "key_observation": "..."
  }}
}}
```

CONFIDENCE GUIDE:
- "certain": Names are exact/phonetic matches AND speaking styles match
- "very_likely": Names are very similar AND no contradictory evidence
- "probably": Some similarity but uncertain
- "unlikely": Names are clearly different or contradictory evidence found"""

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the standard system prompt for speaker identification."""
        return blocks.SYSTEM_SPEAKER_IDENTIFICATION

    @classmethod
    def build_transcript_samples_text(cls, samples: List[Dict]) -> str:
        """
        Build formatted transcript samples text for cluster identification.

        Args:
            samples: List of sample dicts with keys:
                - episode_title: str
                - duration_pct: float
                - total_turns: int
                - first_utterance: str
                - last_utterance: str

        Returns:
            Formatted samples text
        """
        samples_text = ""
        for i, sample in enumerate(samples, 1):
            samples_text += f"\n\n--- Sample {i} (from: {sample.get('episode_title', 'Unknown')}) ---\n"
            samples_text += f"Speaking time: {sample.get('duration_pct', 0):.1f}% of episode, {sample.get('total_turns', 0)} turns\n"
            samples_text += f"FIRST UTTERANCE: {sample.get('first_utterance', 'N/A')}\n"
            samples_text += f"LAST UTTERANCE: {sample.get('last_utterance', 'N/A')}"
        return samples_text

    @classmethod
    def build_metadata_matches_section(cls, metadata_matches: List[Dict]) -> str:
        """
        Build metadata evidence section for cluster identification.

        Args:
            metadata_matches: List of match dicts with keys:
                - host_name: str
                - overlap_ratio: float
                - overlap_count: int
                - cluster_episode_count: int

        Returns:
            Formatted metadata section or empty string
        """
        if not metadata_matches:
            return ""

        section = "\nEPISODE METADATA EVIDENCE:\nThis cluster's speakers appear in episodes labeled with these hosts:\n"
        for match in metadata_matches[:5]:
            pct = match.get('overlap_ratio', 0) * 100
            count = match.get('overlap_count', 0)
            total = match.get('cluster_episode_count', 0)
            section += f"  - {match['host_name']}: {pct:.0f}% overlap ({count}/{total} episodes)\n"
        return section
