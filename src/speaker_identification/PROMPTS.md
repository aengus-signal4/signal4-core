# Speaker Identification LLM Prompts

Central documentation of all LLM prompts used in the speaker identification pipeline.

## Overview

| Phase | Prompt Name | Location | Purpose |
|-------|-------------|----------|---------|
| 1A | `_build_channel_host_prompt` | `core/llm_client.py:415` | Extract hosts from channel description |
| 1B | `_build_episode_extraction_prompt` | `core/llm_client.py:456` | Extract speakers/mentioned from episode metadata |
| 2 | `_build_host_verification_prompt` | `core/llm_client.py:236` | Verify if speaker is a specific host |
| 2B | `_build_cluster_verification_prompt` | `core/llm_client.py:332` | Identify which host a speaker cluster represents |
| 3 | `_identify_speaker_by_transcript` | `strategies/guest_identification.py:230` | Identify guest from transcript context |
| 4 | `_verify_speaker_identity` | `strategies/guest_propagation.py:240` | Verify embedding match with transcript evidence |

---

## Phase 1A: Channel Host Extraction

**File:** `core/llm_client.py:415-454`

**Purpose:** Extract host names from channel descriptions. Only relies on explicit mentions in the description text.

```
You are identifying podcast/video channel hosts from channel descriptions.

CHANNEL: {channel_name} ({platform})
DESCRIPTION: {channel_description}

Identify the host(s) of this channel. Look for:
- Explicit mentions: "hosted by X", "presented by X"
- Channel ownership: "X's podcast", "X's channel"
- First-person descriptions that name the host

**CRITICAL: DO NOT HALLUCINATE OR GUESS HOSTS**
- ONLY return hosts whose names are EXPLICITLY mentioned in the description
- If no host name appears in the description, return an empty list
- DO NOT use external knowledge about who might host this channel
- DO NOT guess based on the channel name or topic

Confidence levels:
- "certain": Name explicitly stated in description
- "very_likely": Name clearly implied (e.g., "John's Podcast" → John)
- "somewhat_likely": Reasonable inference from description text only

Return ONLY valid JSON (no markdown, no explanation):
{
  "hosts": [
    {
      "name": "Full Name",
      "confidence": "certain",
      "reasoning": "Why you identified this person"
    }
  ]
}

If no clear host, return {"hosts": []}.
```

---

## Phase 1B: Episode Speaker Extraction

**File:** `core/llm_client.py:456-501`

**Purpose:** Extract speakers (hosts/guests who appear) and mentioned people from episode metadata.

```
Extract people from podcast/video metadata. Use FULL NAMES whenever possible.

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

Return ONLY valid JSON:
{
  "speakers": [
    {"name": "Full Host Name", "role": "host", "confidence": "certain", "reasoning": "Known channel host"},
    {"name": "Full Guest Name", "role": "guest", "confidence": "certain", "reasoning": "Named in title/description as appearing"}
  ],
  "mentioned": [
    {"name": "Full Person Name", "reasoning": "Discussed in description"},
    {"name": "Another Full Name", "reasoning": "Referenced in article being analyzed"}
  ]
}

Only include confidence "certain" or "very_likely".
```

---

## Phase 2: Host Verification

**File:** `core/llm_client.py:236-330`

**Purpose:** Verify if a speaker matches a specific known host using transcript evidence.

```
Verify if this speaker is SPECIFICALLY the person named '{target_host}'.

EPISODE: {episode_title}
DESCRIPTION: {episode_description[:300]}

TRANSCRIPT EVIDENCE:
{context_text}
{cluster_info}

CRITICAL QUESTION: Is this speaker '{target_host}'?

You must find POSITIVE EVIDENCE that this speaker IS '{target_host}'.
Voice similarity alone is NOT sufficient - you need transcript evidence.

**IMPORTANT - TRANSCRIPTION ERRORS ARE COMMON:**
Speech-to-text often misspells names phonetically. Accept names that SOUND similar:
- "Anise Haydari" / "Anissa Dari" / "Annie Said Are" = "Anis Haydari" (phonetic variations)
- "John Smith" / "Jon Smyth" = same person
- "Dr. Brown" / "Doctor Brown" / "Doc Brown" = same person
If a name in the transcript is phonetically similar to '{target_host}', treat it as a MATCH.

CONFIDENCE LEVELS:
- "certain": Speaker explicitly self-identifies with a name matching or phonetically similar to '{target_host}'
- "very_likely": Someone else addresses this speaker by a name matching or phonetically similar to '{target_host}'
- "unlikely": No name evidence, OR speaker clearly identifies as a completely different person (not a phonetic variation)

AUTOMATIC REJECTION (is_host=false):
- Speaker self-identifies as a clearly DIFFERENT person (not a transcription variant)
- Speaker is explicitly introduced as a guest/interviewee with a different name
- No clear name evidence linking speaker to '{target_host}'

NOTE: Host behavior (speaking time, asking questions) alone is NOT sufficient evidence.
But combined with a phonetically similar name, it strengthens the match.

Return ONLY valid JSON:
{
  "is_host": true/false,
  "confidence": "certain" | "very_likely" | "unlikely",
  "reasoning": "Brief explanation - cite specific name evidence or explain why rejected",
  "name": "Exact name if identifiable from transcript" (otherwise empty string)
}
```

---

## Phase 2B: Cluster Identification

**File:** `core/llm_client.py:332-403`

**Purpose:** Identify which host a speaker cluster represents (for multi-episode recurring speakers).

```
Identify WHO this speaker is based on transcript evidence.

{hosts_section}
{metadata_section}
CLUSTER INFO: This speaker appears in {cluster_size} DIFFERENT EPISODES with the same voice.

TRANSCRIPT SAMPLES (what THIS SPEAKER said):
{samples_text}

SIMPLE RULES - FOLLOW EXACTLY:

1. WHO IS THIS SPEAKER?
   - If speaker says "I'm [Name]" or "This is [Name]" → speaker IS that person
   - If speaker says "Thanks [Name]" or "I agree with [Name]" → speaker is NOT that person (they're addressing someone else)
   - If another person says "[Name], what do you think?" right before this speaker talks → this speaker IS [Name]

2. MATCH TO EXPECTED HOSTS:
   - If identified name matches an expected host (even phonetically) → is_expected_host = true
   - If identified name is someone ELSE → is_expected_host = false, use their actual name
   - If cannot identify → speaker_name = "unknown"

3. ROLE (based on {cluster_size} episodes):
   - {cluster_size}+ episodes = "host" or "co_host" (NOT "guest")
   - Opens/closes shows, introduces guests = "host"
   - Regular panelist/contributor = "co_host"

COMMON MISTAKE TO AVOID:
If the speaker says "I agree with Chantal" - the speaker is NOT Chantal!
If the speaker says "Thanks, Peter" - the speaker is NOT Peter!
These phrases mean the speaker is talking TO that person, not being them.

Keep reasoning brief (1-2 sentences). Return ONLY valid JSON:
{
  "speaker_name": "Name if identified, or 'unknown'",
  "role": "host" | "co_host" | "guest" | "unknown",
  "is_expected_host": true/false,
  "confidence": "certain" | "very_likely" | "probably" | "unlikely",
  "reasoning": "1-2 sentence explanation"
}
```

---

## Phase 3: Guest Identification (Transcript-based)

**File:** `strategies/guest_identification.py:230-302`

**Purpose:** Identify guest speakers using transcript context when hosts are already known.

```
Identify this speaker from a podcast episode.

EPISODE: {title}
DESCRIPTION: {description or 'N/A'}

KNOWN HOST: {host_name} (already identified - this is NOT {host_name}){expected_guests_text}

THIS SPEAKER's transcript ({total_turns} turns, {duration_pct}% of episode):
{transcript_section}

**IMPORTANT - TRANSCRIPTION ERRORS ARE COMMON:**
Names may be misspelled phonetically. Accept reasonable phonetic variations.
If an expected guest name sounds similar to a name in the transcript, it's likely a match.

YOUR TASK:
Identify this speaker by name. Look for:
1. EXPECTED GUESTS: If metadata lists expected guests, check if transcript evidence matches any of them
2. SELF-IDENTIFICATION: "I'm [Name]", "This is [Name]", "My name is [Name]"
3. HOST INTRODUCTION: The host may introduce them
4. TITLE/DESCRIPTION: Guest names often appear in episode metadata

**CRITICAL**: If THIS SPEAKER addresses someone by name (e.g., "Bruce?" or "Thanks, John"),
that person is NOT this speaker - they are addressing someone else. Only match names that
refer TO this speaker, not names this speaker uses to address others.

If no name can be determined from evidence, return "unknown".

Return ONLY valid JSON:
{
  "speaker_name": "Name if identified, or 'unknown'",
  "role": "guest" | "co_host" | "unknown",
  "confidence": "certain" | "very_likely" | "probably" | "unlikely",
  "reasoning": "Brief explanation of identification evidence"
}
```

---

## Phase 4: Guest Propagation Verification

**File:** `strategies/guest_propagation.py:240-310`

**Purpose:** Verify embedding-based matches with transcript evidence.

```
Verify if this speaker matches {candidate_name}.

EPISODE: {title}
DESCRIPTION: {description or 'N/A'}

KNOWN HOST: {host_name} (this is NOT the host)
EMBEDDING SIMILARITY: {similarity:.3f} to {candidate_name}'s voice profile

THIS SPEAKER's transcript ({total_turns} turns, {duration_pct}% of episode):
{transcript_section}

**IMPORTANT - TRANSCRIPTION ERRORS ARE COMMON:**
Names may be misspelled phonetically. Accept reasonable phonetic variations.

**CRITICAL**: If THIS SPEAKER addresses someone by name (e.g., "Bruce?" or "Thanks, John"),
that person is NOT this speaker - they are addressing someone else. Only match names that
refer TO this speaker, not names this speaker uses to address others.

YOUR TASK:
Does the transcript evidence support that this speaker is {candidate_name}?

Look for:
1. SELF-IDENTIFICATION: "I'm [Name]", "This is [Name]"
2. HOST INTRODUCTION: Host introduces them by name
3. CONTEXT CLUES: Professional background, topics discussed match known info about {candidate_name}

CONFIDENCE LEVELS:
- "certain": Explicit name match (self-identification or introduction)
- "very_likely": Strong contextual evidence (profession, expertise matches)
- "probably": Moderate evidence
- "unlikely": No supporting evidence or contradictory evidence

Return ONLY valid JSON:
{
  "is_match": true/false,
  "confidence": "certain" | "very_likely" | "probably" | "unlikely",
  "reasoning": "Brief explanation"
}
```

---

## Key Design Principles

### 1. Addressing vs Being Addressed
A critical error the LLM can make is confusing when a speaker ADDRESSES someone vs BEING that person:

- `"Bruce?"` or `"Thanks, John"` = speaker is NOT Bruce/John (they're addressing someone else)
- `"I'm Bruce"` or `"This is John speaking"` = speaker IS Bruce/John

This distinction is emphasized in Phases 2B, 3, and 4 prompts.

### 2. Phonetic Name Matching
Transcription often introduces spelling errors. All prompts emphasize:
- Accept phonetic variations (`"Anise Haydari"` ≈ `"Anis Haydari"`)
- Don't reject based on minor spelling differences

### 3. Evidence Requirements

| Phase | Evidence Required |
|-------|-------------------|
| 1A | Explicit mention in description |
| 1B | Explicit mention in title/description |
| 2 | Transcript name evidence (behavior alone insufficient) |
| 2B | Self-identification or being addressed |
| 3 | Self-ID, introduction, or expected guest match |
| 4 | Transcript support for embedding match |

### 4. Confidence Levels
Consistent across all prompts:
- **certain**: Explicit self-identification
- **very_likely**: Being addressed by name or strong contextual match
- **probably**: Moderate evidence
- **unlikely**: No evidence or contradictory evidence

---

## Modifying Prompts

When updating prompts:

1. Update the prompt in the source file
2. Update this documentation
3. Test with diverse examples to ensure no regressions
4. Pay special attention to the "addressing vs being addressed" distinction
