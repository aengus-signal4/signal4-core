# Testing Structure Consolidation Plan
**Date:** November 13, 2025

---

## Current Situation

**4 test-related locations:**
1. `tests/` - 8 unit tests (active) ✅
2. `testing_utils/` - SSE client, validators, test runner (active) ✅
3. `scripts/` - 13 test scripts (many deprecated) ⚠️
4. `test_results/` - Output directory (empty) ⚠️

**Total:** ~13 scripts need review

---

## Proposed Consolidation

### Keep (Active Testing)

**`tests/` directory** - Unit and integration tests ✅
```
tests/
├── test_analysis_pipeline.py       # Active
├── test_layer1.py                  # Active
├── test_query_parser.py            # Active
├── test_segment_selector.py        # Active
├── test_text_generator.py          # Active
├── test_theme_extractor.py         # Active
├── test_simple_rag_workflow.py     # Active
└── test_simple_rag_integration.py  # Active
```

**`testing_utils/` directory** - Test infrastructure ✅
```
testing_utils/
├── __init__.py
├── sse_client.py         # SSE streaming client
├── test_runner.py        # Test orchestration
├── validators.py         # Response validation
└── report_generator.py   # Output formatting
```

### Archive (Deprecated Scripts)

**Scripts to archive** (likely use deprecated endpoints):

1. **`test_climate_change_search.py`** (9KB)
   - **Uses:** Old search endpoint
   - **Status:** Needs update for `/api/analysis`
   - **Action:** Archive, create new version if needed

2. **`test_hybrid_search.py`** (8KB)
   - **Uses:** FAISS + pgvector (hybrid)
   - **Status:** FAISS deprecated
   - **Action:** Archive (obsolete)

3. **`test_query2doc_output.py`** (1KB)
   - **Uses:** Direct LLM service testing
   - **Status:** Unit test coverage exists
   - **Action:** Archive (covered by test_query_parser.py)

4. **`test_retrieval_strategies.py`** (6KB)
   - **Uses:** Old search patterns
   - **Status:** Unit tests cover this
   - **Action:** Archive (covered by test_segment_selector.py)

5. **`test_simple_queries.py`** (2KB)
   - **Uses:** Old search endpoint
   - **Status:** Needs update
   - **Action:** Archive or update to use `/api/analysis`

6. **`test_stance_variation_immigration.py`** (13KB)
   - **Uses:** Query expansion testing
   - **Status:** Covered by unit tests
   - **Action:** Archive (covered by test_query_parser.py)

**Analysis/utility scripts to keep:**

7. **`analyze_keyword_quality.py`** (6KB) ✅
   - **Purpose:** Keyword effectiveness analysis
   - **Status:** Utility, not a test
   - **Action:** Keep

8. **`analyze_segment_quality.py`** (3KB) ✅
   - **Purpose:** Segment quality metrics
   - **Status:** Utility, not a test
   - **Action:** Keep

9. **`compare_embedding_models.py`** (9KB) ✅
   - **Purpose:** Benchmark embedding models
   - **Status:** Utility, not a test
   - **Action:** Keep

10. **`evaluate_keyword_value.py`** (8KB) ✅
    - **Purpose:** Keyword value assessment
    - **Status:** Utility, not a test
    - **Action:** Keep

11. **`find_segment.py`** (1KB) ✅
    - **Purpose:** Debug utility for finding segments
    - **Status:** Utility, not a test
    - **Action:** Keep

### Remove (Empty Directories)

**`test_results/` directory** ⚠️
- **Status:** Empty, no files
- **Purpose:** Output directory for old tests
- **Action:** Remove (tests should output to `archive/test_results/` if needed)

---

## Consolidation Actions

### Phase 1: Archive Deprecated Test Scripts

```bash
# Create archive directory for old test scripts
mkdir -p archive/scripts/deprecated_tests

# Move deprecated test scripts
mv scripts/test_climate_change_search.py archive/scripts/deprecated_tests/
mv scripts/test_hybrid_search.py archive/scripts/deprecated_tests/
mv scripts/test_query2doc_output.py archive/scripts/deprecated_tests/
mv scripts/test_retrieval_strategies.py archive/scripts/deprecated_tests/
mv scripts/test_simple_queries.py archive/scripts/deprecated_tests/
mv scripts/test_stance_variation_immigration.py archive/scripts/deprecated_tests/
```

### Phase 2: Rename Scripts Directory

```bash
# Rename scripts/ to utilities/ for clarity
mv scripts/ utilities/

# Or keep as scripts/ but document purpose clearly
```

### Phase 3: Remove Empty Directories

```bash
# Remove empty test_results directory
rm -rf test_results/
```

### Phase 4: Create README for Utilities

Create `utilities/README.md`:
```markdown
# Backend Utilities

Analysis and debugging utilities for the backend system.

## Analysis Scripts
- `analyze_keyword_quality.py` - Keyword effectiveness metrics
- `analyze_segment_quality.py` - Segment quality assessment
- `compare_embedding_models.py` - Embedding model benchmarks
- `evaluate_keyword_value.py` - Keyword value analysis

## Debug Utilities
- `find_segment.py` - Find specific segments by ID or text

## Usage
All scripts are standalone and can be run directly:
```bash
python utilities/analyze_segment_quality.py
```

Note: Old test scripts have been archived to `archive/scripts/deprecated_tests/`
```

---

## Final Structure

```
src/backend/
├── tests/                           # Unit & integration tests (8 files)
│   ├── test_analysis_pipeline.py
│   ├── test_layer1.py
│   ├── test_query_parser.py
│   ├── test_segment_selector.py
│   ├── test_text_generator.py
│   ├── test_theme_extractor.py
│   ├── test_simple_rag_workflow.py
│   └── test_simple_rag_integration.py
│
├── testing_utils/                   # Test infrastructure (5 files)
│   ├── __init__.py
│   ├── sse_client.py
│   ├── test_runner.py
│   ├── validators.py
│   └── report_generator.py
│
├── utilities/                       # Analysis & debug scripts (5 files)
│   ├── README.md
│   ├── analyze_keyword_quality.py
│   ├── analyze_segment_quality.py
│   ├── compare_embedding_models.py
│   ├── evaluate_keyword_value.py
│   └── find_segment.py
│
└── archive/
    └── scripts/
        └── deprecated_tests/        # Old test scripts (6 files)
            ├── test_climate_change_search.py
            ├── test_hybrid_search.py
            ├── test_query2doc_output.py
            ├── test_retrieval_strategies.py
            ├── test_simple_queries.py
            └── test_stance_variation_immigration.py
```

---

## Benefits

1. **Clearer organization** - Testing vs utilities vs archived
2. **Reduced confusion** - No duplicate/overlapping tests
3. **Easier maintenance** - Clear what's active vs deprecated
4. **Better documentation** - Each directory has clear purpose

---

## Summary

**Current:**
- 4 test locations (tests/, testing_utils/, scripts/, test_results/)
- 13 scripts in scripts/ (mix of tests and utilities)
- Confusion about what's active

**After consolidation:**
- 3 organized locations (tests/, testing_utils/, utilities/)
- Clear separation: unit tests, infrastructure, analysis utilities
- All deprecated test scripts archived

**Files affected:**
- **Moved:** 6 deprecated test scripts
- **Kept:** 5 analysis utilities
- **Removed:** 1 empty directory (test_results/)
- **Created:** 1 README (utilities/README.md)
