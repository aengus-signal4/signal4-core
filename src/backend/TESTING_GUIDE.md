# Backend Testing Guide
**Date:** November 13, 2025
**Status:** Aligned with Simplified Architecture (3 endpoints)

---

## Overview

The backend testing infrastructure supports:
- **Unit tests** - Component-level testing (8 active tests in `tests/`)
- **Integration tests** - End-to-end workflow testing
- **SSE streaming tests** - Server-Sent Events simulation
- **Performance validation** - Response time and cache hit rate checks

---

## Testing Utilities (`testing_utils/`)

### SSE Client (`sse_client.py`)

Simulates frontend SSE consumption for testing streaming endpoints.

**Classes:**
- `SSEClient` - HTTP client for SSE endpoints
- `SSEEvent` - Single SSE event with timestamp
- `SSEStream` - Complete stream with metadata

**Usage:**
```python
from testing_utils import SSEClient

async with SSEClient("http://localhost:8002") as client:
    stream = await client.stream_post(
        "/api/analysis/stream",
        json={
            "query": "carbon tax",
            "dashboard_id": "cprmv-practitioner",
            "workflow": "simple_rag"
        }
    )

    # Access events
    for event in stream.events:
        print(f"{event.event_type}: {event.data}")

    # Get specific events
    complete_event = stream.get_event_by_type("complete")
    summary = complete_event.data.get("summary")
```

### Test Runner (`test_runner.py`)

Orchestrates test execution with progress reporting.

**Classes:**
- `TestRunner` - Main test orchestrator
- `TestResult` - Single test result with timing
- `TestSuiteResult` - Suite-level aggregation

**Usage:**
```python
from testing_utils import TestRunner

runner = TestRunner(verbose=True)

# Define tests
tests = [
    ("Test media endpoint", test_media_endpoint),
    ("Test analysis workflow", test_analysis_workflow),
]

# Run suite
await runner.run_suite("API Tests", tests)
runner.print_summary()
```

### Validators (`validators.py`)

Validates API responses for quality and correctness.

**Classes:**
- `ResultValidator` - Response structure validation
- `CacheValidator` - Cache hit rate validation
- `PerformanceValidator` - Response time validation

**Usage:**
```python
from testing_utils import ResultValidator, PerformanceValidator

# Validate search response
result = ResultValidator.validate_search_response(
    data=response_data,
    expected_min_results=10
)
assert result.passed, result.message

# Validate performance
perf_result = PerformanceValidator.validate_response_time(
    duration_ms=response_time,
    max_allowed_ms=2000
)
assert perf_result.passed, perf_result.message
```

### Report Generator (`report_generator.py`)

Formats test results for console output and HTML reports.

**Usage:**
```python
from testing_utils import print_test_header, print_sse_event, print_result

print_test_header("Media Endpoint Tests")

for event in stream.events:
    print_sse_event(event)

print_result("Test passed", success=True, duration_ms=1234)
```

---

## Active Unit Tests (`tests/`)

### Layer 1: Data Retrieval

**`test_layer1.py`**
- Tests: SegmentRetriever, QueryParser, SmartRetriever
- Focus: Database queries, filtering, embedding generation

**`test_query_parser.py`**
- Tests: Query expansion strategies (multi_query, query2doc, theme_queries)
- Focus: LLM-based query transformation

### Layer 2: Analysis Primitives

**`test_theme_extractor.py`**
- Tests: HDBSCAN clustering, theme labeling, sub-theme extraction
- Focus: Clustering quality, adaptive validation

**`test_segment_selector.py`**
- Tests: Weighted sampling strategies (diversity, centrality, recency)
- Focus: Sample quality, distribution

**`test_text_generator.py`**
- Tests: LLM generation, prompt templates, batch processing
- Focus: Text quality, citation formatting

### Layer 3: Workflows

**`test_analysis_pipeline.py`**
- Tests: Pipeline orchestration, step execution, streaming
- Focus: End-to-end pipeline flow

**`test_simple_rag_workflow.py`**
- Tests: SimpleRAG workflow (query → retrieve → summarize)
- Focus: Workflow correctness

**`test_simple_rag_integration.py`**
- Tests: Full integration with database and LLM
- Focus: End-to-end functionality

---

## Testing the New Architecture

### 1. Health Endpoint

```bash
# Basic health check
curl http://localhost:8002/health

# Check models loaded
curl http://localhost:8002/health/models

# Check database connection
curl http://localhost:8002/health/db
```

**Python test:**
```python
import requests

def test_health():
    response = requests.get("http://localhost:8002/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
```

### 2. Media Endpoint

**Basic media retrieval:**
```bash
curl "http://localhost:8002/api/media/content/VIDEO_ID?start_time=10&end_time=30"
```

**With transcription (batch):**
```bash
curl "http://localhost:8002/api/media/content/VIDEO_ID?transcribe=true&language=en"
```

**With transcription (streaming):**
```python
from testing_utils import SSEClient

async def test_media_with_transcription():
    async with SSEClient() as client:
        stream = await client.stream_get(
            "/api/media/content/VIDEO_ID",
            params={
                "transcribe": True,
                "stream": True,
                "language": "en"
            }
        )

        # Check for expected events
        assert stream.get_event_by_type("started")
        assert stream.get_event_by_type("media_ready")
        assert stream.get_event_by_type("transcript_ready")
        assert stream.get_event_by_type("complete")
```

### 3. Analysis Endpoint

**Simple RAG workflow:**
```python
from testing_utils import SSEClient, ResultValidator

async def test_simple_rag():
    async with SSEClient() as client:
        stream = await client.stream_post(
            "/api/analysis/stream",
            json={
                "query": "carbon tax policy",
                "dashboard_id": "cprmv-practitioner",
                "workflow": "simple_rag",
                "time_window_days": 30
            }
        )

        # Validate stream
        assert stream.status_code == 200
        assert len(stream.events) > 0

        # Get final result
        complete_event = stream.get_event_by_type("complete")
        assert complete_event is not None

        result_data = complete_event.data

        # Validate response structure
        validation = ResultValidator.validate_search_response(result_data)
        assert validation.passed, validation.message
```

**Custom pipeline:**
```python
async def test_custom_pipeline():
    async with SSEClient() as client:
        stream = await client.stream_post(
            "/api/analysis/stream",
            json={
                "query": "inflation rates",
                "dashboard_id": "cprmv-practitioner",
                "pipeline": [
                    {"step": "expand_query", "config": {"strategy": "multi_query"}},
                    {"step": "retrieve_segments", "config": {"k": 200}},
                    {"step": "quantitative_analysis", "config": {}},
                    {"step": "select_segments", "config": {"n": 20}},
                    {"step": "generate_summary", "config": {}}
                ]
            }
        )

        # Validate each step completed
        step_events = stream.get_events_by_type("step_complete")
        assert len(step_events) == 5  # 5 steps
```

**Hierarchical summary:**
```python
async def test_hierarchical_summary():
    async with SSEClient() as client:
        stream = await client.stream_post(
            "/api/analysis/stream",
            json={
                "query": "climate change",
                "dashboard_id": "cprmv-practitioner",
                "workflow": "hierarchical_summary",
                "time_window_days": 180
            }
        )

        # Validate themes extracted
        complete_event = stream.get_event_by_type("complete")
        result = complete_event.data

        assert "themes" in result
        assert len(result["themes"]) > 0

        # Check theme structure
        first_theme = result["themes"][0]
        assert "theme_name" in first_theme
        assert "summary" in first_theme
        assert "segment_count" in first_theme
```

---

## Running Tests

### Unit Tests (pytest)

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_analysis_pipeline.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test
pytest tests/test_query_parser.py::test_multi_query_expansion -v
```

### Integration Tests (with running server)

```bash
# Start backend server
uvicorn app.main:app --host 0.0.0.0 --port 8002

# Run integration tests
pytest tests/test_simple_rag_integration.py -v
```

### Manual Testing Scripts

Located in `scripts/`:
```bash
# Test simple queries
python scripts/test_simple_queries.py

# Test retrieval strategies
python scripts/test_retrieval_strategies.py

# Test stance variation
python scripts/test_stance_variation_immigration.py
```

**Note:** Some scripts in `scripts/` may reference deprecated endpoints. Review and update as needed.

---

## Test Coverage Goals

### Current Coverage (Estimated)

- **Layer 1 (Data Retrieval):** 85% coverage ✅
- **Layer 2 (Analysis):** 75% coverage ✅
- **Layer 3 (Workflows):** 70% coverage ✅
- **API Endpoints:** 40% coverage ⚠️

### Priority Areas for New Tests

1. **Media endpoint with transcription** (new functionality)
2. **Analysis endpoint streaming** (recently refactored)
3. **Custom pipeline validation** (declarative system)
4. **Error handling** (edge cases, invalid inputs)
5. **Cache effectiveness** (hit rates, performance)

---

## Writing New Tests

### Template for SSE Endpoint Test

```python
import pytest
from testing_utils import SSEClient, ResultValidator, PerformanceValidator

@pytest.mark.asyncio
async def test_my_workflow():
    """Test my custom workflow"""

    async with SSEClient("http://localhost:8002") as client:
        # Make request
        stream = await client.stream_post(
            "/api/analysis/stream",
            json={
                "query": "my test query",
                "dashboard_id": "cprmv-practitioner",
                "workflow": "simple_rag"
            }
        )

        # Basic checks
        assert stream.status_code == 200
        assert stream.error is None
        assert len(stream.events) > 0

        # Validate specific events
        started_event = stream.get_event_by_type("started")
        assert started_event is not None

        complete_event = stream.get_event_by_type("complete")
        assert complete_event is not None

        # Validate response content
        result_data = complete_event.data
        validation = ResultValidator.validate_search_response(result_data)
        assert validation.passed, validation.message

        # Validate performance
        perf = PerformanceValidator.validate_response_time(
            stream.total_duration_ms,
            max_allowed_ms=5000
        )
        assert perf.passed, perf.message
```

### Template for Unit Test

```python
import pytest
from app.services.rag.query_parser import QueryParser
from app.services.llm_service import LLMService

@pytest.fixture
def llm_service():
    """Create LLM service for testing"""
    from app.config.dashboard_config import load_dashboard_config
    config = load_dashboard_config("cprmv-practitioner")
    return LLMService(config, "cprmv-practitioner")

def test_query_parser_multi_query(llm_service):
    """Test multi-query expansion"""
    parser = QueryParser(llm_service)

    result = parser.expand_query(
        query="carbon tax",
        strategy="multi_query",
        n_variations=3
    )

    # Validate result structure
    assert "queries" in result
    assert len(result["queries"]) == 3
    assert all(isinstance(q, str) for q in result["queries"])

    # Validate query quality (basic checks)
    for query in result["queries"]:
        assert len(query) > 10  # Reasonable length
        assert "carbon" in query.lower() or "tax" in query.lower()
```

---

## Performance Benchmarks

### Expected Response Times (95th percentile)

| Endpoint | Operation | Expected Time | Notes |
|----------|-----------|---------------|-------|
| `/health` | Health check | < 50ms | Fast path |
| `/api/media` | Basic media | < 500ms | S3 + cache |
| `/api/media` | With transcription | 2-5s | AssemblyAI API |
| `/api/analysis` | Simple RAG | 3-8s | Query + retrieve + LLM |
| `/api/analysis` | Hierarchical | 15-60s | Multi-stage clustering |

### Cache Hit Rates (Expected)

| Cache Type | Expected Hit Rate | Notes |
|------------|------------------|-------|
| query2doc | > 60% | Common queries repeat |
| theme_summary | > 75% | Themes are stable |
| optimize_query | > 50% | Query patterns vary |
| search_results | > 25% | High query diversity |

---

## Continuous Integration

### GitHub Actions Workflow (Example)

```yaml
name: Backend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: signal4
          POSTGRES_DB: av_content
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          cd src/backend
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio

      - name: Run unit tests
        run: |
          cd src/backend
          pytest tests/ -v --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          files: ./src/backend/coverage.xml
```

---

## Troubleshooting

### Common Issues

**1. SSE Connection Timeouts**
```python
# Increase timeout for long-running workflows
async with SSEClient() as client:
    stream = await client.stream_post(
        "/api/analysis/stream",
        json={...},
        timeout=300  # 5 minutes
    )
```

**2. Database Connection Errors**
```bash
# Check database connectivity
psql -h 10.0.0.4 -U signal4 -d av_content -c "SELECT 1;"

# Check pg_cron jobs running
psql -h 10.0.0.4 -U signal4 -d av_content -c "SELECT * FROM cron.job WHERE active = true;"
```

**3. LLM Cache Misses**
```sql
-- Check cache statistics
SELECT cache_type, COUNT(*), AVG(access_count)
FROM llm_cache
WHERE dashboard_id = 'cprmv-practitioner'
GROUP BY cache_type;
```

**4. Embedding Model Not Loaded**
```bash
# Check if models are pre-loaded
curl http://localhost:8002/health/models

# Expected response:
# {"status": "healthy", "models_loaded": true, "model_dimensions": [1024, 2000]}
```

---

## Future Test Improvements

### Priority 1 (High Impact) 🔴
1. **Media endpoint tests** - New transcription integration
2. **Analysis endpoint streaming** - SSE validation
3. **Error handling tests** - Edge cases and invalid inputs

### Priority 2 (Medium Impact) 🟡
1. **Performance regression tests** - Catch slowdowns
2. **Cache effectiveness tests** - Monitor hit rates
3. **Custom pipeline validation** - Declarative system edge cases

### Priority 3 (Low Impact) 🟢
1. **Load testing** - Concurrent requests, rate limiting
2. **Security testing** - Input sanitization, SQL injection
3. **Browser compatibility** - SSE across different clients

---

## Summary

**Testing utilities are well-designed** and support the new 3-endpoint architecture with minimal updates needed.

**Key strengths:**
- ✅ SSE client ready for streaming tests
- ✅ Validators support all response types
- ✅ Test runner framework is flexible
- ✅ Report generation for CI/CD integration

**Action items:**
1. Write tests for new media+transcription functionality
2. Update `scripts/` tests to use new endpoints
3. Add integration tests for declarative pipelines
4. Set up CI/CD with coverage reporting

**Test coverage:** Ready to support production deployment with targeted additions for new features.
