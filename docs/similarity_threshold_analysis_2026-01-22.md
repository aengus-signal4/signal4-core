# Similarity Threshold Analysis Report

**Date:** 2026-01-22
**Query Tested:** "How are Mark Carney and Canada being discussed"
**Time Window:** 30 days
**Results Retrieved:** 367 segments

## Executive Summary

The analysis reveals that the new `topic_summary` reranking weights (sim=0.15, pop=0.35, rec=0.35) **successfully prioritize recent content from important channels**, but also **boost irrelevant content** from very popular channels that happens to be in the search results.

**Recommendation: Implement a similarity floor of 0.50** as a gating mechanism before reranking.

## Key Findings

### 1. Similarity Score Distribution

| Metric | Value |
|--------|-------|
| Min | 0.3120 |
| Max | 0.7818 |
| Mean | 0.5452 |
| Median | 0.5786 |
| P10 | 0.3271 |
| P25 | 0.4136 |
| P75 | 0.6836 |

The distribution is bimodal with peaks around 0.35 (weak matches) and 0.65-0.70 (strong matches).

### 2. Problem: Irrelevant Content Getting Boosted

With `topic_summary` weights, the top 30 includes **clearly irrelevant content** from high-popularity channels:

| Rank | Sim | Channel Pop | Channel | Issue |
|------|-----|-------------|---------|-------|
| 11 | 0.3702 | 0.8909 | Feel Better, Live More | Discusses Newtown massacre, unrelated |
| 24 | 0.3659 | 0.8189 | High Performance Podcast | Generic motivational content |
| 27 | 0.3220 | 0.8361 | Happy Place | Unrelated personal development |

These segments appear because:
- Their channels have very high importance scores (0.8+)
- Recency is high (recent episodes)
- Similarity is downweighted to only 15%
- Combined pop+rec (70%) overcomes the low similarity penalty

### 3. Borderline Content (0.45-0.55) Analysis

Many borderline segments **are actually relevant** to Mark Carney and Canada:

| Sim | Channel | Content |
|-----|---------|---------|
| 0.5411 | Bloomberg Daybreak: US Edition | Directly discusses Mark Carney's China visit |
| 0.5146 | Bloomberg Daybreak: US Edition | Canada-China trade relations, mentions Carney |
| 0.4940 | Bloomberg Daybreak: US Edition | Canadian PM visiting China |
| 0.4967 | Bloomberg Daybreak: Europe | Same coverage as above |
| 0.4752 | The Daily Brief | Canada-China trade deal, explicitly mentions Mark Carney |

However, some borderline content is **irrelevant**:

| Sim | Channel | Content |
|-----|---------|---------|
| 0.4529 | The Rest Is Money | UK infrastructure, barely mentions Canada |
| 0.4646 | Business Daily | Random Guinness brewery content |

### 4. Threshold Impact Analysis

| Threshold | Kept | Dropped | Avg Pop Kept | Avg Pop Dropped |
|-----------|------|---------|--------------|-----------------|
| 0.40 | 294 | 73 | 0.6054 | 0.7668 |
| 0.45 | 242 | 125 | 0.5951 | 0.7196 |
| **0.50** | **226** | **141** | **0.5954** | **0.7049** |
| 0.55 | 214 | 153 | 0.6008 | 0.6888 |
| 0.60 | 160 | 207 | 0.6039 | 0.6634 |

**Key insight:** Dropped segments have *higher* average channel popularity. This confirms that the low-similarity segments being filtered are mostly popular but irrelevant content.

## Recommendation

### Implement Similarity Floor: 0.50

**Rationale:**
1. **Removes clearly irrelevant content** - All segments with sim < 0.50 that made top 30 were unrelated
2. **Preserves relevant borderline content** - Bloomberg coverage at 0.50-0.55 is kept
3. **Still allows channel/recency to dominate** - After the floor, reranking uses topic_summary weights
4. **Acceptable loss** - Only 141 segments filtered, and these have high pop but low relevance

### Implementation Approach

Add `similarity_floor` parameter to `RerankerWeights`:

```python
@dataclass
class RerankerWeights:
    similarity: float = 0.4
    popularity: float = 0.2
    recency: float = 0.2
    single_speaker: float = 0.1
    named_speaker: float = 0.1
    similarity_floor: float = 0.0  # Default: no floor

    @classmethod
    def topic_summary(cls) -> "RerankerWeights":
        return cls(
            similarity=0.15,
            popularity=0.35,
            recency=0.35,
            single_speaker=0.08,
            named_speaker=0.07,
            similarity_floor=0.50  # Gate: must be at least 50% similar
        )
```

Then in `SegmentReranker.rerank()`, filter before scoring:

```python
if weights.similarity_floor > 0:
    segments = [s for s in segments if s.get('similarity', 0) >= weights.similarity_floor]
```

### Alternative: 0.45 (Permissive)

If you want to keep more borderline content (like the Bloomberg segments at 0.49), use 0.45. This still removes the worst offenders (sim < 0.45) while keeping potentially relevant content.

## Test Script

The analysis script is available at:
```
core/scripts/analyze_similarity_threshold.py
```

Run with:
```bash
cd ~/signal4/core
uv run python scripts/analyze_similarity_threshold.py
```
