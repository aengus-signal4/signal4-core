# Speaker Identification System

## Overview

The speaker identification system assigns real-world identities to audio speakers extracted from podcast/video content. It runs as a sequential, idempotent pipeline that progressively builds up speaker information.

**Current Coverage Target**: 90%+ speaker identification through signal-rich strategies.

---

## NEW: Text-Evidence-First Pipeline (Recommended)

The new pipeline fixes a **centroid contamination bug** in the legacy system. The legacy Phase 2B would assign speakers to identities based on guest names from episode titles + embedding clustering WITHOUT transcript verification. This caused incorrect voice samples to contaminate identity centroids.

### New Pipeline Phases

| Phase | Name | Description |
|-------|------|-------------|
| 1 | Metadata Extraction | Same as legacy - extract hosts/guests from metadata |
| 2 | **Text Evidence Collection** | NEW - Find rock-solid transcript evidence (self-ID, addressed, introduced) |
| 3 | **Verified Centroid Generation** | NEW - Build centroids ONLY from text-verified speakers |
| 4 | **Embedding Propagation** | NEW - Match to verified centroids (0.90+ auto, 0.65-0.90 LLM) |
| 5 | Identity Merge Detection | Same as legacy - detect and merge duplicate identities |

### Key Differences from Legacy

1. **No assignment without evidence**: Text evidence OR high similarity to VERIFIED centroid required
2. **Binary evidence output**: "certain" or "none" (no middle ground)
3. **Higher auto-assign threshold**: 0.90 similarity (vs 0.80 in legacy) since verified centroids are trusted
4. **New database columns**:
   - `speakers.text_evidence_status`: 'certain', 'none', 'not_processed', 'short_utterance'
   - `speakers.evidence_type`: 'self_intro', 'addressed', 'introduced', 'none'
   - `speakers.evidence_quote`: Exact transcript quote proving identity
   - `speakers.assignment_source`: 'text_verified', 'embedding_propagation_auto', 'embedding_propagation_llm', 'legacy'
   - `speaker_identities.centroid_source`: 'text_verified', 'legacy'

### Running the New Pipeline

```bash
# Dry run (recommended first)
python -m src.speaker_identification.orchestrator --project CPRMV --new-pipeline

# Apply all phases
python -m src.speaker_identification.orchestrator --project CPRMV --new-pipeline --apply

# Run specific phases
python -m src.speaker_identification.orchestrator --project CPRMV --new-pipeline --phases 2,3,4 --apply

# Re-verify all legacy assignments (full migration)
python -m src.speaker_identification.orchestrator --project CPRMV --new-pipeline --include-assigned --apply
```

### Individual Strategy Commands

```bash
# Phase 2: Text Evidence Collection
python -m src.speaker_identification.strategies.text_evidence_collection --project CPRMV --apply

# Phase 3: Verified Centroid Generation
python -m src.speaker_identification.strategies.verified_centroid_generation --project CPRMV --apply

# Phase 4: Embedding Propagation
python -m src.speaker_identification.strategies.embedding_propagation --project CPRMV --apply
```

### Migration from Legacy

Before running the new pipeline on existing data:

1. **Apply database migration**: `alembic upgrade head` (adds new columns)
2. **Existing data is marked as 'legacy'**: `assignment_source` and `centroid_source` default to 'legacy'
3. **Re-verify with `--include-assigned`**: Process speakers already assigned to re-verify with text evidence

---

## LEGACY Pipeline

The legacy pipeline is preserved for backwards compatibility but has a known contamination bug in Phase 2B.

## Core Principles

### 1. Sequential, Idempotent Phases

Each phase runs independently and can be re-run safely:
- **Phase 1**: Metadata extraction (no embeddings required)
- **Phase 2**: Host embedding identification (embeddings + clustering + LLM)
- **Phase 3**: Guest identification (transcript + metadata + LLM, no embeddings)
- **Phase 4**: Identity centroid generation (clustering + auto-split for name collisions)
- **Phase 5**: Identity merge detection (detect and merge duplicate identities)
- **Phase 6**: Guest propagation (embedding similarity + LLM verification)

Each phase builds on previous phases but doesn't destroy their work. Re-running a phase should be safe and produce consistent results.

### 2. Confidence-Based Assignment

All assignments have confidence levels:
- **certain**: Direct evidence (self-identification, explicit name mention)
- **very_likely**: Strong circumstantial evidence
- **probably**: Moderate evidence (triggers retry with different samples)
- **unlikely**: Weak evidence (skip)
- **unknown**: No identification possible

Only `certain` and `very_likely` result in assignments.

### 3. Single Best Centroid per Identity

Speaker identities store ONE centroid (voice profile):
- Centroids are global, not channel-specific
- Compare quality before overwriting (keep best)
- Use quality-weighted averaging for centroid computation
- Channel roles are tracked separately in `channel_roles` dict

### 4. Channel-Specific Roles

A person can have different roles on different channels:
- `host`: Primary host
- `co_host`: Regular co-host
- `recurring_guest`: Appears frequently
- `guest`: One-time or occasional guest

Stored in `verification_metadata.channel_roles.{channel_id}`.

---

## Phase 1: Metadata Identification

**File**: `strategies/metadata_identification.py`

**Purpose**: Extract speaker information from text metadata only (no audio/embeddings).

### Phase 1A: Channel Host Extraction

1. Read channel description
2. LLM extracts host names and roles
3. Cache in `channels.hosts` column
4. Sync hosts from episode frequency data (hosts appearing in 10+ episodes promoted to channel level)

### Phase 1B: Episode Speaker Extraction

For each episode:
1. Read episode title/description
2. LLM extracts speakers (hosts appearing) and guests
3. Store in `content.hosts` and `content.guests` columns
4. Also extract `content.mentioned` (people referenced but not speaking)
5. Only save speakers with `certain` or `very_likely` confidence

### Phase 1C: Name Consolidation

1. Analyze host name distribution across channel episodes
2. LLM identifies name variations (e.g., "Dr. Andrew Huberman" vs "Andrew Huberman")
3. Update `content.hosts` with canonical names
4. Save aliases to `channel_host_cache` for future lookups

**Key Outputs**:
- `channels.hosts`: Array of host objects `[{name, role, confidence}]`
- `content.hosts`: Array of hosts appearing in this episode
- `content.guests`: Array of guests in this episode
- `content.mentioned`: Array of mentioned people

**Idempotency**: Skips episodes that already have `metadata_speakers_extracted` flag set.

**Usage**:
```bash
# Run on all active projects
python -m src.speaker_identification.strategies.metadata_identification --apply

# Run on specific project
python -m src.speaker_identification.strategies.metadata_identification \
    --project CPRMV --apply

# Run consolidation only
python -m src.speaker_identification.strategies.metadata_identification \
    --project CPRMV --consolidate-only --apply
```

---

## Phase 2: Host Embedding Identification

**File**: `strategies/host_embedding_identification.py`

**Purpose**: Use voice embeddings to verify and assign hosts to speakers.

### Phase 2A: Host Frequency Analysis

1. Query `content.hosts` to find qualified hosts (≥10 appearances)
2. Build list of hosts to bootstrap with their episode IDs

### Phase 2B: Bootstrap Host Centroids

For each qualified host:
1. Load TOP-3 speakers (by duration) from labeled episodes
2. Cluster at 0.78 similarity threshold
3. Use strategy pattern to verify clusters (see below)
4. Compute centroid from verified cluster (quality-weighted)
5. Assign cluster members to identity (1 per episode)

**Strategy Selection** (based on channel type):
- **SingleHostStrategy**: For channels with one primary host
  - First cluster: LLM verify
  - Subsequent clusters: Embedding similarity (≥0.80 merge, 0.80-0.95 merge with weighted average)
  - Efficient: minimizes LLM calls after first verification
- **MultiHostStrategy**: For channels with multiple co-hosts
  - Always LLM verify with metadata context
  - Uses episode overlap analysis to pre-filter candidates
  - Requires name evidence (self-ID, addressed by name)

**Discovered Hosts**: System can identify hosts not in expected list (substitute hosts, co-hosts) when LLM detects host behavior patterns.

### Phase 2C: Expand to Full Channel

1. Load long-speaking (≥15%) unassigned speakers from episodes where `content.hosts` contains known hosts
2. Skip episodes that already have a host assigned
3. Two-pass matching:
   - **Pass 1 (Fast)**: Auto-assign high-confidence matches (≥0.65 similarity)
   - **Pass 2 (LLM)**: Verify medium-confidence matches (0.50-0.65 similarity)
4. Only 1 speaker assigned per host per episode

**Key Outputs**:
- `speaker_identities.verification_metadata.centroid`: Voice embedding centroid
- `speaker_identities.verification_metadata.channel_roles`: Per-channel role data
- `speakers.speaker_identity_id`: Assignment to identity
- `speakers.assignment_confidence`: Confidence score
- `speakers.assignment_method`: How assignment was made

**Flags**:
- `--reset`: Clears speaker assignments and channel roles (preserves centroids)
- `--fresh`: Ignores existing centroids, re-bootstraps all hosts (updates DB if new centroid is better)

**Usage**:
```bash
# Single channel
python -m src.speaker_identification.strategies.host_embedding_identification \
    --channel-id 6109 --apply

# Reset and reprocess
python -m src.speaker_identification.strategies.host_embedding_identification \
    --channel-id 6109 --apply --reset

# Full project
python -m src.speaker_identification.strategies.host_embedding_identification \
    --project CPRMV --apply
```

---

## Phase 3: Guest Identification

**File**: `strategies/guest_identification.py`

**Purpose**: Identify guests in episodes where we know the host.

### Approach

For episodes with known hosts:
1. Find unassigned speakers (≥10% duration)
2. Build transcript context (6-turn pattern: turn before/after first and last appearance)
3. Extract expected guests from `content.guests` metadata
4. LLM identifies speaker using:
   - Episode title and description
   - Host name
   - Expected guests list
   - Transcript excerpts
5. Save LLM result to `speakers.llm_identification` (JSONB)
6. Create/match identity and assign speaker for `certain`/`very_likely` confidence

**Key Differences from Host Strategy**:
- Pure transcript + metadata based (no embedding comparison)
- Works on individual speakers, not clusters
- Leverages known host context for better identification

**Outputs**:
- `speakers.llm_identification`: JSONB with identified_name, role, confidence, reasoning
- `speaker_identities`: New guest identity records
- `speakers.speaker_identity_id`: Assignment to identity

**Usage**:
```bash
# Run on project
python -m src.speaker_identification.strategies.guest_identification \
    --project CPRMV --apply

# Only process episodes with named guests in metadata
python -m src.speaker_identification.strategies.guest_identification \
    --project CPRMV --require-named-guests --apply

# Include speakers with existing LLM results (re-run)
python -m src.speaker_identification.strategies.guest_identification \
    --project CPRMV --include-cached --apply
```

---

## Phase 4: Identity Centroid Generation

**File**: `strategies/centroid_generation.py`

**Purpose**: Build centroids for identities that don't have them (primarily guests from Phase 3).

### Approach

For each identity WITHOUT an existing centroid:
1. Load all assigned speakers with embeddings
2. If 1 speaker: Use that embedding as centroid
3. If 2+ speakers: Cluster at 0.78 threshold to verify same person
   - **Single cluster**: Build quality-weighted centroid from all speakers
   - **Multiple clusters**: Auto-split identity (name collision detected)

### Auto-Split on Name Collision

When speakers assigned to the same identity don't cluster together:
- Largest cluster keeps original identity
- Each outlier cluster becomes new identity: "Name (2)", "Name (3)", etc.
- Speakers are reassigned to their respective new identities
- All splits are logged for audit

### Scope

- Only processes identities WITHOUT existing centroids
- Hosts from Phase 2 keep their centroids unchanged
- Cross-channel guests will cluster if embeddings are similar

**Usage**:
```bash
# Run on all identities needing centroids
python -m src.speaker_identification.strategies.centroid_generation --apply

# Run on specific project
python -m src.speaker_identification.strategies.centroid_generation \
    --project CPRMV --apply

# Process single identity
python -m src.speaker_identification.strategies.centroid_generation \
    --identity-id 55921 --apply
```

---

## Phase 5: Identity Merge Detection

**File**: `strategies/identity_merge_detection.py`

**Purpose**: Detect and merge duplicate speaker identities that were created separately due to name variations, transcription errors, or independent identification across different episodes/channels.

### Problem

During Phase 3 and Phase 4, separate identities can be created for the same person due to:
- Name variations: "Ian Sénéchal" vs "Ian Senechal" vs "Yan Sénéchal"
- Transcription errors in different episodes
- Same guest identified independently across different channels
- Phase 4 auto-split creating false positives

### Approach

1. Build FAISS index of all identity centroids
2. Find high-similarity centroid pairs (≥ 0.85 threshold)
3. For each candidate pair:
   - **Co-appearance veto**: If both appear in same episode → NOT same person
   - Gather evidence: metadata, channels, transcript samples
   - LLM verification determines if same person AND which to keep
4. Execute confirmed merges:
   - Keep identity with better data quality/name correctness
   - Reassign speakers from merged identity
   - Rebuild centroid with combined speakers
   - Record audit trail

### Safety Mechanisms

- **Co-appearance veto**: Never merge identities appearing in same episode
- **Dry-run by default**: Preview merges before applying
- **Audit trail**: All merges recorded with LLM reasoning
- **Confidence threshold**: Only merge on "certain" or "very_likely"

**Usage**:
```bash
# Dry run on all identities
python -m src.speaker_identification.strategies.identity_merge_detection

# Apply merges on specific project
python -m src.speaker_identification.strategies.identity_merge_detection \
    --project CPRMV --apply

# Higher threshold (more conservative)
python -m src.speaker_identification.strategies.identity_merge_detection \
    --threshold 0.90 --apply
```

---

## Phase 6: Guest Propagation

**File**: `strategies/guest_propagation.py`

**Purpose**: Match unassigned speakers to existing identity centroids.

### Tiered Matching Approach

For each unassigned speaker with embedding:
1. Compare to all existing identity centroids using FAISS batch search
2. Apply tiered matching rules:
   - **≥ 0.80 (single match)**: Auto-match (embedding similarity alone sufficient)
   - **≥ 0.80 (multiple matches)**: LLM disambiguates between candidates
   - **≥ 0.65 + LLM "very_likely"**: Match
   - **< 0.65**: Skip for now (retry next run when new centroids may help)

### Process

1. Load all identity centroids from `speaker_identities`
2. Build FAISS index for efficient batch similarity search
3. Get unassigned speakers with embeddings
4. Batch similarity search - buckets speakers into:
   - Auto-match (single high-confidence)
   - Needs LLM (multiple candidates or 0.65-0.80)
   - Skip for retry (below 0.65)
5. Process each bucket accordingly
6. Cache LLM results in per-phase tracking

**Key Differences from Guest Identification**:
- Requires existing identity centroids
- Uses embedding similarity as primary signal
- Good for returning guests who have appeared before
- Uses FAISS for efficient batch processing

**Usage**:
```bash
# Run on project
python -m src.speaker_identification.strategies.guest_propagation \
    --project CPRMV --apply

# Include speakers with existing LLM results
python -m src.speaker_identification.strategies.guest_propagation \
    --project CPRMV --include-cached --apply
```

---

## Orchestrator

**File**: `orchestrator.py`

**Purpose**: Run the full pipeline or specific phases.

**Usage**:
```bash
# Run all phases on project (dry run)
python -m src.speaker_identification.orchestrator --project CPRMV

# Apply all phases
python -m src.speaker_identification.orchestrator --project CPRMV --apply

# Run specific phases
python -m src.speaker_identification.orchestrator --project CPRMV --phases 1,2 --apply

# Run on single channel (phases 3, 4, 5, 6)
python -m src.speaker_identification.orchestrator --channel-id 6569 --phases 3,4,5,6 --apply
```

---

## Data Model

### Tables

```
speaker_identities
├── id (PK)
├── primary_name (unique identifier)
├── alias_names (JSONB array)
├── role (deprecated - use channel_roles in verification_metadata)
├── confidence_score
├── verification_status ('verified', 'llm_identified', 'low_confidence', 'pending_review')
├── verification_metadata (JSONB)
│   ├── centroid (512-dim embedding)
│   ├── centroid_quality
│   ├── centroid_sample_count
│   ├── centroid_source_channel
│   └── channel_roles
│       └── {channel_id}
│           ├── role
│           ├── episode_count
│           └── total_duration
└── is_active

speakers
├── id (PK)
├── content_id (FK → content)
├── local_speaker_id
├── embedding (512-dim)
├── embedding_quality_score
├── duration
├── speaker_identity_id (FK → speaker_identities)
├── assignment_confidence
├── assignment_method
├── llm_identification (JSONB)  # Cached LLM results
│   ├── identified_name
│   ├── role
│   ├── confidence
│   ├── reasoning
│   ├── method
│   └── timestamp
└── updated_at

content
├── content_id (PK)
├── channel_id (FK → channels)
├── hosts (JSONB array)
├── guests (JSONB array)
├── mentioned (JSONB array)
├── metadata_speakers_extracted (boolean)
└── ...

channels
├── id (PK)
├── display_name
├── hosts (JSONB array of host objects)
└── ...

channel_host_cache
├── id (PK)
├── channel_id (FK → channels)
├── host_name
├── confidence
├── reasoning
├── method
├── host_aliases (JSONB array)  # Name variations
└── identified_at
```

### Verification Metadata Structure

```json
{
  "centroid": [0.123, 0.456, ...],  // 512-dim array
  "centroid_quality": 0.82,
  "centroid_sample_count": 45,
  "centroid_source_channel": 6109,
  "centroid_updated_at": "2024-01-15T...",
  "channel_roles": {
    "6109": {
      "role": "host",
      "episode_count": 234,
      "total_duration": 123456.7,
      "updated_at": "2024-01-15T..."
    },
    "7383": {
      "role": "guest",
      "episode_count": 3,
      "total_duration": 4567.8,
      "updated_at": "2024-01-15T..."
    }
  }
}
```

---

## Strategy Pattern Architecture

### Base Classes

```
src/speaker_identification/strategies/
├── base.py
│   ├── ClusterVerificationResult   # Result from cluster verification
│   ├── HostStrategyContext         # Context passed to strategies
│   └── HostVerificationStrategy    # ABC for host verification
```

### Host Strategies

```
├── hosts/
│   ├── single_host.py    # SingleHostStrategy
│   │   - First cluster: LLM verify
│   │   - Subsequent: Embedding similarity
│   │   - Stop after finding target host
│   │
│   └── multi_host.py     # MultiHostStrategy
│       - Always LLM verify with metadata
│       - Uses episode overlap analysis
│       - Continues until all hosts found or limit reached
```

### Core Components

```
├── core/
│   ├── llm_client.py        # MLXLLMClient - tiered LLM access
│   ├── context_builder.py   # DB queries for speaker/episode context
│   └── identity_manager.py  # SpeakerIdentity CRUD operations
```

---

## Thresholds and Parameters

### Embedding Similarity

| Threshold | Use |
|-----------|-----|
| 0.78 | Cluster formation (same speaker) |
| 0.85 | Identity merge detection threshold |
| 0.80 | Auto-match for guest propagation |
| 0.65 | High-confidence host auto-assign |
| 0.65 | Guest propagation + LLM "very_likely" |
| 0.50 | Medium-confidence (LLM verify) |

### Quality Scores

| Threshold | Use |
|-----------|-----|
| 0.65 | Centroid contribution (preferred) |
| 0.30-0.50 | Minimum for clustering/matching |

### Sample Counts

| Count | Use |
|-------|-----|
| 5 | Minimum samples for clustering |
| 5 | Minimum samples for centroid |
| 50 | Maximum samples for centroid |
| 10 | Minimum host occurrences in content.hosts |

### Duration Thresholds

| Threshold | Use |
|-----------|-----|
| 10% | Minimum speaker duration for guest identification |
| 15% | Minimum speaker duration for host expansion (Phase 2C) |

---

## Error Handling

### Graceful Degradation

1. **No hosts found in metadata**: Fall back to anonymous clustering
2. **No centroid bootstrapped**: Skip to next host
3. **LLM timeout/error**: Retry once, then skip speaker
4. **"Probably" confidence**: Retry with different transcript samples

### Reset Behavior

The `--reset` flag:
1. Clears `speakers.speaker_identity_id` for channel
2. Clears `channel_roles.{channel_id}` from identities
3. **Preserves** centroids (they're the best voice profile)
4. Preserves identity records (they're global)

---

## LLM Tiers

The system uses tiered LLM access via `MLXLLMClient`:
- **tier_1**: Best model (80B) - complex verification
- **tier_2**: Medium model - name consolidation
- **tier_3**: Fast model - metadata extraction

---

## Future Considerations

### Cross-Channel Matching

When a person appears on multiple channels:
1. Use existing centroid for matching
2. Add new channel to their `channel_roles`
3. Only update centroid if new one is better quality

### Identity Management

- **Merge** (Phase 5 - Implemented): Two identities are the same person
  - Uses FAISS to find high-similarity centroid pairs (≥0.85)
  - LLM verifies and decides which identity to keep
  - Reassigns speakers, rebuilds centroid
  - Co-appearance veto prevents false merges

- **Split** (Phase 4 - Implemented): One identity is actually two people
  - Detected during centroid generation
  - Speakers that don't cluster together are split
  - Creates new identities: "Name (2)", "Name (3)", etc.

### Quality Improvements

1. **Active learning**: Flag low-confidence for human review
2. **Cross-reference**: Use external databases (Wikidata, etc.)
3. **Transcript evidence**: Extract self-identification patterns ("I'm X", "as X mentioned")

---

## Review Dashboard

**File**: `dashboards/speaker_identification_review.py`

**Purpose**: Web-based dashboard for reviewing and validating speaker identification results.

### Features

#### Overview Tab
- **Key Metrics**: Total speakers, identification rate, speaking time coverage, unique identities
- **Progress Tracking**: Visual progress towards 90% coverage target
- **Top Speakers**: Ranked by total speaking duration
- **Role Breakdown**: Hosts vs guests distribution
- **Assignment Methods**: How speakers were identified (embedding, LLM, manual)
- **Channel Breakdown**: Per-channel identification statistics
- **Unidentified Speakers**: Speakers with ≥5% duration needing attention

#### Review Tab
- **Search**: Find speakers by name
- **Filters**: By verification status, role, minimum duration
- **Speaker Cards**: Summary with expandable details
- **Audio Playback**: Listen to transcript samples to validate identity
- **Edit Form**: Update speaker metadata, role, confidence
- **Pagination**: Browse through all identified speakers

### Usage

```bash
# Start the dashboard
streamlit run dashboards/speaker_identification_review.py

# Start on specific port
streamlit run dashboards/speaker_identification_review.py --server.port 8505
```

### Key Metrics Explained

| Metric | Description |
|--------|-------------|
| Identification Rate | % of speaker occurrences assigned to an identity |
| Coverage Rate | % of speaking time attributed to identified speakers |
| Unique Identities | Number of distinct people identified |
| Avg Confidence | Mean assignment confidence for identified speakers |

### Workflow

1. **Check Overview**: See current coverage and identify gaps
2. **Review Unidentified**: Focus on speakers with ≥5% duration
3. **Validate Identities**: Listen to audio samples, verify correct assignment
4. **Fix Errors**: Edit incorrectly identified speakers
5. **Track Progress**: Monitor coverage rate towards 90% target

---

## Quick Reference

### Run Full Pipeline

```bash
# Dry run
python -m src.speaker_identification.orchestrator --project CPRMV

# Apply changes
python -m src.speaker_identification.orchestrator --project CPRMV --apply
```

### Run Individual Phases

```bash
# Phase 1: Metadata extraction
python -m src.speaker_identification.strategies.metadata_identification --project CPRMV --apply

# Phase 2: Host embedding identification
python -m src.speaker_identification.strategies.host_embedding_identification --project CPRMV --apply

# Phase 3: Guest identification
python -m src.speaker_identification.strategies.guest_identification --project CPRMV --apply

# Phase 4: Centroid generation
python -m src.speaker_identification.strategies.centroid_generation --project CPRMV --apply

# Phase 5: Identity merge detection
python -m src.speaker_identification.strategies.identity_merge_detection --project CPRMV --apply

# Phase 6: Guest propagation
python -m src.speaker_identification.strategies.guest_propagation --project CPRMV --apply
```

### Check Progress

```sql
-- Overall speaker identification status
SELECT
    COUNT(*) as total_speakers,
    COUNT(speaker_identity_id) as identified,
    ROUND(COUNT(speaker_identity_id)::numeric / COUNT(*) * 100, 1) as id_rate,
    ROUND(SUM(CASE WHEN speaker_identity_id IS NOT NULL THEN duration ELSE 0 END) / SUM(duration) * 100, 1) as coverage_rate
FROM speakers s
JOIN content c ON s.content_id = c.content_id
WHERE 'CPRMV' = ANY(c.projects);

-- Top speakers by duration
SELECT
    si.primary_name,
    si.role,
    COUNT(DISTINCT s.id) as appearances,
    ROUND(SUM(s.duration) / 3600, 1) as hours
FROM speaker_identities si
JOIN speakers s ON s.speaker_identity_id = si.id
JOIN content c ON s.content_id = c.content_id
WHERE 'CPRMV' = ANY(c.projects)
GROUP BY si.id, si.primary_name, si.role
ORDER BY SUM(s.duration) DESC
LIMIT 20;
```

### Launch Dashboard

```bash
streamlit run dashboards/speaker_identification_review.py
```
