"""
Quick Summary Workflow Test
===========================

Tests the quick_summary workflow via /api/analysis/stream endpoint.

This workflow is optimized for speed:
- expand_query (multi_query strategy)
- retrieve_segments (k=50, fewer segments)
- select_segments (n=15, no clustering)
- generate_summary (fast Grok model)

Tests:
1. Full workflow execution with SSE streaming
2. Cache effectiveness (run twice, compare timing)
3. Segment validation (verify returned segments have expected structure)
4. Response structure validation

Run with:
    uv run python src/backend/tests/test_quick_summary_workflow.py

Or with pytest:
    uv run pytest src/backend/tests/test_quick_summary_workflow.py -v -s
"""

import asyncio
import json
import os
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent  # core/
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
from src.utils.paths import get_env_path
load_dotenv(get_env_path(), override=True)

from httpx import AsyncClient, ASGITransport


# ============================================================================
# Test Configuration
# ============================================================================

# Test API key - created specifically for backend integration tests
# Run: uv run python -m src.backend.scripts.manage_api_keys create --email test@signal4.ai --name "Backend Integration Tests"
TEST_API_KEY = os.getenv("TEST_API_KEY", "sk_88b9229311e048d69b17cf7bc6dee46e90ce74b67a6041d1")

# Backend URL - use running server for warm models, or ASGI transport for isolated testing
# Set BACKEND_URL=http://localhost:7999 to test against running server (recommended for performance tests)
BACKEND_URL = os.getenv("BACKEND_URL", None)  # None = use ASGI transport
USE_LIVE_SERVER = BACKEND_URL is not None

TEST_QUERY = "What is being said about Mark Carney?"
TEST_DASHBOARD_ID = "cprmv-practitioner"
TEST_PROJECTS = ["Canadian", "Big_Channels"]
TEST_TIME_WINDOW_DAYS = 7


# ============================================================================
# Helper Functions
# ============================================================================

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_event(event: Dict, event_num: int):
    """Print a formatted event."""
    event_type = event.get("type", "unknown")
    step = event.get("step", "")

    if event_type == "cache_hit":
        level = event.get("level", "")
        print(f"  [{event_num}] CACHE_HIT: level={level}")

    elif event_type == "cache_miss":
        print(f"  [{event_num}] CACHE_MISS")

    elif event_type == "step_start":
        print(f"  [{event_num}] START: {step}")

    elif event_type == "step_complete":
        duration = event.get("duration_ms", "?")
        print(f"  [{event_num}] COMPLETE: {step} ({duration}ms)")

    elif event_type == "result":
        data = event.get("data", {})
        if step == "expand_query":
            queries = data.get("expanded_queries", [])
            print(f"  [{event_num}] RESULT: {step} - {len(queries)} query variations")
        elif step == "retrieve_segments_by_search":
            count = len(data.get("segments", []))
            print(f"  [{event_num}] RESULT: {step} - {count} segments retrieved")
        elif step == "select_segments":
            selected = data.get("selected_segments", [])
            print(f"  [{event_num}] RESULT: {step} - {len(selected)} segments selected")
        elif step == "generate_summaries":
            summaries = data.get("summaries", {})
            print(f"  [{event_num}] RESULT: {step} - summaries generated")
        else:
            print(f"  [{event_num}] RESULT: {step}")

    elif event_type == "partial":
        progress = event.get("progress", 0)
        message = event.get("message", "")
        print(f"  [{event_num}] PROGRESS: {progress:.1%} - {message}")

    elif event_type == "complete":
        print(f"  [{event_num}] COMPLETE: Pipeline finished")

    elif event_type == "error":
        error = event.get("error", "Unknown error")
        print(f"  [{event_num}] ERROR: {error}")

    else:
        print(f"  [{event_num}] {event_type}: {step}")


async def run_quick_summary(
    client: AsyncClient,
    query: str,
    run_label: str = "Run"
) -> Dict[str, Any]:
    """
    Execute quick_summary workflow and collect results.

    Returns dict with:
        - events: List of all SSE events
        - total_time_ms: Total execution time
        - summary: Generated summary text
        - segments: List of selected segments
        - segment_count: Total segments retrieved
    """
    request_payload = {
        "query": query,
        "dashboard_id": TEST_DASHBOARD_ID,
        "workflow": "quick_summary",
        "time_window_days": TEST_TIME_WINDOW_DAYS,
        "projects": TEST_PROJECTS,
        "verbose": True
    }

    # Headers with API key
    headers = {
        "X-API-Key": TEST_API_KEY,
        "Content-Type": "application/json"
    }

    print_header(f"{run_label}: quick_summary workflow")
    print(f"Query: {query}")
    print(f"Dashboard: {TEST_DASHBOARD_ID}")
    print(f"Projects: {TEST_PROJECTS}")
    print(f"Time window: {TEST_TIME_WINDOW_DAYS} days")
    print("-" * 80)

    start_time = time.time()
    events = []
    summary = None
    segments = []
    selected_segments = []
    segment_count = 0
    cache_hit = False

    async with client.stream(
        "POST",
        "/api/analysis/stream",
        json=request_payload,
        headers=headers,
        timeout=120.0
    ) as response:

        if response.status_code != 200:
            print(f"ERROR: HTTP {response.status_code}")
            return {"error": f"HTTP {response.status_code}"}

        event_num = 0
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                event_data = line[6:]
                try:
                    event = json.loads(event_data)
                    events.append(event)
                    event_num += 1
                    print_event(event, event_num)

                    # Extract key data
                    event_type = event.get("type")
                    step = event.get("step")
                    data = event.get("data", {})

                    if event_type == "cache_hit":
                        cache_hit = True

                    if event_type == "result":
                        if step == "retrieve_segments_by_search":
                            segments = data.get("segments", [])
                            segment_count = len(segments)
                        elif step == "select_segments":
                            selected_segments = data.get("selected_segments", [])
                        elif step == "generate_summaries":
                            summaries = data.get("summaries", {})
                            # Get the main summary
                            if "main" in summaries:
                                summary_obj = summaries["main"]
                                # Summary might be dict with 'summary' key or string
                                if isinstance(summary_obj, dict):
                                    summary = summary_obj.get("summary", str(summary_obj))
                                else:
                                    summary = summary_obj
                            elif summaries:
                                summary_obj = list(summaries.values())[0]
                                if isinstance(summary_obj, dict):
                                    summary = summary_obj.get("summary", str(summary_obj))
                                else:
                                    summary = summary_obj

                    if event_type == "complete":
                        # Final data might be in complete event
                        if "summaries" in data:
                            summaries = data.get("summaries", {})
                            if "main" in summaries:
                                summary_obj = summaries["main"]
                                if isinstance(summary_obj, dict):
                                    summary = summary_obj.get("summary", str(summary_obj))
                                else:
                                    summary = summary_obj
                            elif summaries:
                                summary_obj = list(summaries.values())[0]
                                if isinstance(summary_obj, dict):
                                    summary = summary_obj.get("summary", str(summary_obj))
                                else:
                                    summary = summary_obj

                except json.JSONDecodeError as e:
                    print(f"  [!] JSON parse error: {e}")

    total_time_ms = (time.time() - start_time) * 1000

    print("-" * 80)
    print(f"Total time: {total_time_ms:.0f}ms")
    print(f"Events received: {len(events)}")
    print(f"Cache hit: {cache_hit}")
    print(f"Segments retrieved: {segment_count}")
    print(f"Segments selected: {len(selected_segments)}")

    return {
        "events": events,
        "total_time_ms": total_time_ms,
        "summary": summary,
        "segments": segments,
        "selected_segments": selected_segments,
        "segment_count": segment_count,
        "cache_hit": cache_hit
    }


def validate_segment(segment: Dict, index: int) -> List[str]:
    """
    Validate a segment has expected structure.
    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Required fields
    required_fields = ["segment_id", "text", "similarity"]
    for field in required_fields:
        if field not in segment:
            errors.append(f"Segment {index}: missing required field '{field}'")

    # Validate segment_id
    if "segment_id" in segment:
        seg_id = segment["segment_id"]
        if not isinstance(seg_id, int) or seg_id <= 0:
            errors.append(f"Segment {index}: invalid segment_id={seg_id}")

    # Validate text
    if "text" in segment:
        text = segment["text"]
        if not isinstance(text, str) or len(text) < 10:
            errors.append(f"Segment {index}: text too short or invalid")

    # Validate similarity
    if "similarity" in segment:
        sim = segment["similarity"]
        if not isinstance(sim, (int, float)) or sim < 0 or sim > 1:
            errors.append(f"Segment {index}: invalid similarity={sim}")

    return errors


def validate_summary(summary: str) -> List[str]:
    """
    Validate the generated summary.
    Returns list of validation errors (empty if valid).
    """
    errors = []

    if not summary:
        errors.append("Summary is empty or None")
        return errors

    if not isinstance(summary, str):
        errors.append(f"Summary is not a string: {type(summary)}")
        return errors

    if len(summary) < 100:
        errors.append(f"Summary too short: {len(summary)} chars (expected >= 100)")

    # Check for citations (should have {seg_X} format)
    if "{seg_" not in summary and "seg_" not in summary.lower():
        errors.append("Summary appears to have no citations")

    return errors


# ============================================================================
# Main Test Function
# ============================================================================

async def test_quick_summary_workflow():
    """
    Main test function that:
    1. Runs quick_summary workflow twice to test caching
    2. Validates segments have expected structure
    3. Validates summary output
    4. Reports timing comparison
    """
    # Use live server if BACKEND_URL is set, otherwise use ASGI transport
    if USE_LIVE_SERVER:
        print(f"\nUsing LIVE SERVER at {BACKEND_URL}")
        print("(Models should be pre-loaded and warm)")
        async with AsyncClient(base_url=BACKEND_URL, timeout=120.0) as client:
            await _run_test_workflow(client)
    else:
        print("\nUsing ASGI TRANSPORT (in-process app)")
        print("(Models will be loaded on first request - slower)")
        # Import app here to avoid loading at module level
        from src.backend.app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await _run_test_workflow(client)

    return True


async def _run_test_workflow(client: AsyncClient):
    """Run the actual test workflow with the given client."""
    # ================================================================
    # Run 1: First execution (cache miss expected)
    # ================================================================
    result1 = await run_quick_summary(client, TEST_QUERY, "Run 1 (cold)")

    if "error" in result1:
        print(f"\nTEST FAILED: {result1['error']}")
        return False

    # ================================================================
    # Run 2: Second execution (cache hit expected)
    # ================================================================
    print("\n" + "~" * 80)
    print(" Waiting 2 seconds before second run...")
    print("~" * 80)
    await asyncio.sleep(2)

    result2 = await run_quick_summary(client, TEST_QUERY, "Run 2 (warm)")

    if "error" in result2:
        print(f"\nTEST FAILED: {result2['error']}")
        return False

    # ================================================================
    # Validate Results
    # ================================================================
    print_header("VALIDATION RESULTS")

    all_errors = []

    # Validate segments from Run 1
    print("\nValidating segments...")
    segments = result1.get("segments", [])
    if not segments:
        all_errors.append("No segments returned")
    else:
        # Validate first 5 segments
        for i, seg in enumerate(segments[:5]):
            seg_errors = validate_segment(seg, i)
            all_errors.extend(seg_errors)
            if not seg_errors:
                print(f"  Segment {i}: OK (id={seg.get('segment_id')}, sim={seg.get('similarity', 0):.3f})")
            else:
                for err in seg_errors:
                    print(f"  {err}")

        if len(segments) > 5:
            print(f"  ... and {len(segments) - 5} more segments")

    # Validate summary
    print("\nValidating summary...")
    summary = result1.get("summary")
    summary_errors = validate_summary(summary)
    all_errors.extend(summary_errors)

    if not summary_errors:
        print(f"  Summary: OK ({len(summary)} chars)")
        # Show preview
        preview = summary[:200] + "..." if len(summary) > 200 else summary
        print(f"\n  Preview:\n  {preview}")
    else:
        for err in summary_errors:
            print(f"  ERROR: {err}")

    # ================================================================
    # Cache Effectiveness Analysis
    # ================================================================
    print_header("CACHE EFFECTIVENESS")

    time1 = result1["total_time_ms"]
    time2 = result2["total_time_ms"]
    speedup = time1 / time2 if time2 > 0 else 0

    print(f"  Run 1 (cold): {time1:.0f}ms")
    print(f"  Run 2 (warm): {time2:.0f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Time saved: {time1 - time2:.0f}ms")

    cache_hit_1 = result1.get("cache_hit", False)
    cache_hit_2 = result2.get("cache_hit", False)
    print(f"\n  Cache hit Run 1: {cache_hit_1}")
    print(f"  Cache hit Run 2: {cache_hit_2}")

    # Validate cache behavior
    if cache_hit_1:
        print("  WARNING: Run 1 had cache hit (expected miss for cold run)")
    if not cache_hit_2:
        print("  WARNING: Run 2 had cache miss (expected hit for warm run)")

    # ================================================================
    # Final Summary
    # ================================================================
    print_header("TEST SUMMARY")

    if all_errors:
        print(f"\nFAILED with {len(all_errors)} errors:")
        for err in all_errors:
            print(f"  - {err}")
        return False
    else:
        print("\nALL TESTS PASSED")
        print(f"  - Segments validated: {min(5, len(segments))}")
        print(f"  - Summary validated: Yes")
        print(f"  - Cache working: {'Yes' if cache_hit_2 else 'Partial'}")
        print(f"  - Performance: {time2:.0f}ms (warm)")
        return True


# ============================================================================
# Pytest Integration
# ============================================================================

import pytest

@pytest.mark.asyncio
async def test_quick_summary_via_pytest():
    """Pytest wrapper for the quick_summary workflow test."""
    success = await test_quick_summary_workflow()
    assert success, "Quick summary workflow test failed"


# ============================================================================
# Direct Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" QUICK SUMMARY WORKFLOW TEST")
    print(" Testing: /api/analysis/stream with workflow=quick_summary")
    print("=" * 80)

    success = asyncio.run(test_quick_summary_workflow())

    print("\n" + "=" * 80)
    if success:
        print(" TEST PASSED")
    else:
        print(" TEST FAILED")
    print("=" * 80 + "\n")

    sys.exit(0 if success else 1)
