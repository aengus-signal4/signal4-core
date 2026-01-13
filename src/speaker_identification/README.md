# Speaker Identification System

Modular, incremental speaker identification system achieving 90%+ coverage through signal-rich strategies.

## Overview

**Current Coverage**: 4.7% (50K/1M speakers)
**Target Coverage**: 90%+
**Approach**: Signal-first (metadata, transcripts, embeddings) → clustering fallback

## Architecture

### Two-Phase Identification (Phase 1)

#### Phase 1A: Channel Host Identification
Extract hosts from channel descriptions using LLM.

**Example**:
```
Channel: Huberman Lab (podcast)
Description: "The Huberman Lab podcast is hosted by Andrew Huberman, Ph.D.,
a neuroscientist and tenured professor..."

LLM Output:
{
  "hosts": [
    {"name": "Andrew Huberman", "role": "host", "confidence": 0.95}
  ]
}
```

**Storage**: Results cached in `channel_host_cache` table for reuse.

#### Phase 1B: Episode Speaker Assignment
Match speakers to identities using channel hosts + episode descriptions.

**Example**:
```
Channel Hosts: ["Andrew Huberman"]
Episode: "How to Speak Clearly & With Confidence | Matt Abrahams"
Description: "My guest is Matt Abrahams, lecturer at Stanford..."

Speakers:
- SPEAKER_2: 66% of episode (1981s)
- SPEAKER_1: 25% of episode (732s)

LLM Output:
{
  "assignments": [
    {"speaker": "SPEAKER_2", "name": "Andrew Huberman", "role": "host", "confidence": 0.90},
    {"speaker": "SPEAKER_1", "name": "Matt Abrahams", "role": "guest", "confidence": 0.95}
  ]
}
```

**Result**: Creates `SpeakerIdentity` records and updates `speakers.speaker_identity_id`.

## Directory Structure

```
speaker_identification/
├── README.md                           # This file
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base_strategy.py                # Abstract base class
│   ├── llm_client.py                   # MLX client (10.0.0.4:8002)
│   ├── context_builder.py              # DB query utilities
│   └── identity_manager.py             # SpeakerIdentity CRUD
├── strategies/
│   ├── __init__.py
│   ├── 01_metadata_identification.py   # Phase 1A + 1B (metadata-only)
│   ├── 02_channel_hosts.py             # Future: Frequency-based hosts
│   ├── 03_prominent_guests.py          # Future: High-duration guests
│   ├── 04_embedding_similarity.py      # Future: Match to known identities
│   └── 05_cluster_fallback.py          # Future: Clustering remainder
├── orchestrator.py                     # Run strategies in sequence
└── run_daily.py                        # Cron entry point
```

## Database Schema

### channel_host_cache
Caches LLM-identified hosts per channel:
```sql
CREATE TABLE channel_host_cache (
    id SERIAL PRIMARY KEY,
    channel_id INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    host_name VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL,
    reasoning TEXT,
    method VARCHAR(50) DEFAULT 'llm_channel_description',
    identified_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    UNIQUE(channel_id, host_name)
);
```

### speaker_identities (existing)
Enhanced with new identification methods:
- `verification_status`: 'verified' | 'llm_identified' | 'low_confidence' | 'pending_review'
- `verification_metadata`: JSONB with identification details
- `confidence_score`: 0.0-1.0

### speakers (existing)
- `speaker_identity_id`: FK to speaker_identities (NULL if unassigned)
- `assignment_confidence`: Confidence score for this assignment
- `assignment_method`: e.g., 'metadata_identification'

## Configuration

Add to `config/config.yaml`:
```yaml
speaker_identification:
  enabled: true

  llm_server:
    endpoint: "10.0.0.4:8002"
    tier: "tier_1"  # Use best model (80B)

  metadata_strategy:
    enabled: true
    min_confidence: 0.60
    min_speaker_duration_pct: 0.10  # 10% of episode
    min_quality_score: 0.65
    batch_size: 10  # Parallel LLM calls
```

## Usage

### Run Phase 1 on Single Channel
```bash
python -m src.speaker_identification.strategies.01_metadata_identification \
    --channel-id 6109 \
    --apply
```

### Run Full Pipeline
```bash
python -m src.speaker_identification.orchestrator \
    --strategy metadata \
    --project CPRMV \
    --apply
```

### Dry Run (Preview)
```bash
python -m src.speaker_identification.orchestrator \
    --strategy metadata \
    --project CPRMV
```

## LLM Prompts

### Phase 1A: Channel Host Extraction
```
You are identifying podcast/video channel hosts from channel descriptions.

CHANNEL: {channel_name} ({platform})
DESCRIPTION: {channel_description}

Identify the host(s) of this channel. Look for:
- Explicit mentions: "hosted by X"
- Channel ownership: "X's podcast"
- Implied hosts from first-person descriptions

Return JSON:
{
  "hosts": [
    {
      "name": "Full Name",
      "role": "host",
      "confidence": 0.0-1.0,
      "reasoning": "Why you identified this person"
    }
  ]
}

If no clear host, return empty array.
```

### Phase 1B: Episode Speaker Assignment
```
You are assigning speaker labels to real people based on metadata.

CHANNEL HOSTS: {list of known hosts}
EPISODE TITLE: {title}
EPISODE DESCRIPTION: {description}
PUBLISHED: {date}

SPEAKERS (sorted by speaking time):
{for each speaker}
- Speaker {index}: {duration_pct}% of episode ({duration}s), {segment_count} segments
{endfor}

Based on the episode metadata, identify each speaker. Consider:
- Channel hosts typically have 40-70% speaking time
- Named guests in title/description
- Interview format (host asks questions, guest explains)

Return JSON:
{
  "assignments": [
    {
      "speaker_index": 0,
      "name": "Full Name" or null if unknown,
      "role": "host" or "guest" or "unknown",
      "confidence": 0.0-1.0,
      "reasoning": "Why you made this assignment"
    }
  ]
}
```

## Confidence Thresholds

| Score | Status | Action |
|-------|--------|--------|
| ≥ 0.85 | verified | Auto-assign, high confidence |
| 0.70-0.84 | llm_identified | Assign with medium confidence |
| 0.60-0.69 | low_confidence | Assign but flag for review |
| < 0.60 | pending_review | Do not assign automatically |

## Expected Outcomes (Phase 1)

### Coverage
- **Before**: 4.7% (50K/1M)
- **After Phase 1**: 45-55%
  - Identified hosts: ~30%
  - Named guests in descriptions: ~15-20%

### Speed
- ~50-100 channels/day with tier_1 model
- 2 LLM calls per channel (host identification)
- ~5-10 LLM calls per episode (speaker assignment)

### Quality
- High confidence (≥0.85): ~40% (explicitly named)
- Medium confidence (0.70-0.84): ~45% (inferred)
- Low confidence (0.60-0.69): ~15% (ambiguous)

## Future Phases

### Phase 2: Transcript Analysis
- Add self-introduction detection ("I'm X")
- Add conversational patterns
- Improve confidence scores

### Phase 3: Frequency & Similarity
- Channel-level frequency analysis
- Embedding similarity matching
- Cross-episode validation

### Phase 4: Clustering Fallback
- Anchor-canopy clustering for remainder
- Target remaining 5-10% coverage

## Development Workflow

1. **Create strategy**: Extend `BaseStrategy` in `strategies/`
2. **Add LLM prompts**: Define in strategy class
3. **Test on single channel**: Verify LLM outputs
4. **Test on project**: Run on 10-20 channels
5. **Measure coverage**: Compare before/after
6. **Add to orchestrator**: Include in daily pipeline

## Monitoring

Track metrics per strategy:
- Channels/episodes processed
- Speakers identified
- Average confidence scores
- Coverage improvement
- LLM call count and latency

Store in strategy run metadata for analysis.
