# Backend Utilities
**Purpose:** Analysis and debugging utilities for backend system evaluation

---

## Analysis Scripts

### `analyze_keyword_quality.py`
Evaluates keyword effectiveness for query expansion.

**Usage:**
```bash
python utilities/analyze_keyword_quality.py
```

**Metrics:**
- Keyword frequency distribution
- Semantic relevance scores
- Query expansion effectiveness

### `analyze_segment_quality.py`
Assesses segment quality metrics (length, coherence, speaker attribution).

**Usage:**
```bash
python utilities/analyze_segment_quality.py
```

**Metrics:**
- Segment length distribution
- Speaker label coverage
- Text coherence scores

### `compare_embedding_models.py`
Benchmarks different embedding models for performance and quality.

**Usage:**
```bash
python utilities/compare_embedding_models.py
```

**Comparison:**
- Qwen3-0.6B (1024-dim) vs Qwen3-4B (2000-dim)
- Retrieval quality metrics
- Inference speed benchmarks

### `evaluate_keyword_value.py`
Analyzes keyword value for search optimization.

**Usage:**
```bash
python utilities/evaluate_keyword_value.py
```

**Metrics:**
- Search result quality per keyword
- Keyword discriminatory power
- Optimization recommendations

---

## Debug Utilities

### `find_segment.py`
Find specific segments by ID, text, or content_id.

**Usage:**
```bash
# Find by segment ID
python utilities/find_segment.py --segment_id 12345

# Find by content ID
python utilities/find_segment.py --content_id VIDEO_ID

# Find by text search
python utilities/find_segment.py --text "climate change"
```

---

## Archived Test Scripts

Old test scripts that reference deprecated endpoints have been moved to:
```
archive/scripts/deprecated_tests/
```

These scripts tested:
- Old search endpoint (replaced by `/api/analysis`)
- FAISS hybrid search (replaced by pgvector)
- Direct endpoint testing (now covered by unit tests)

If you need to reference old testing patterns, see the archive.

---

## Development Notes

All utilities are standalone scripts that can be run directly. They use the production database and services, so:
- ⚠️ **Do not run in production** without understanding impact
- ✅ Run against staging/development database
- ✅ Results are logged to console (no file outputs by default)

### Database Connection

Scripts connect to:
- **Host:** 10.0.0.4
- **Database:** av_content
- **User:** signal4 (from .env)

Make sure `.env` is configured before running utilities.
