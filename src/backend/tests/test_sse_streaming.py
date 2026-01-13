"""
SSE Streaming Test for Analysis Pipeline
=========================================

Tests the /api/analysis/stream endpoint with real database and components.
Shows step-by-step SSE events for debugging and verification.

Run with: pytest tests/test_sse_streaming.py -v -s --log-cli-level=INFO
"""

import pytest
import os
import json
import asyncio
from httpx import AsyncClient, ASGITransport

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.insert(0, str(get_project_root()))

from app.main import app


# ============================================================================
# Test 1: Simple RAG Workflow - Full SSE Stream
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_sse_stream():
    """Test simple_rag workflow with SSE streaming - shows all events."""

    # Create async HTTP client for SSE
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        request_payload = {
            "query": "Pierre Poilievre carbon tax",
            "dashboard_id": "cprmv-practitioner",
            "workflow": "simple_rag",
            "time_window_days": 7,
            "projects": ["CPRMV"],
            "languages": ["en"]
        }

        print("\n" + "="*80)
        print("SSE STREAMING TEST: simple_rag workflow")
        print("="*80)
        print(f"Request: {json.dumps(request_payload, indent=2)}")
        print("="*80)

        # Send POST request with SSE streaming
        async with client.stream(
            "POST",
            "/api/analysis/stream",
            json=request_payload,
            timeout=120.0
        ) as response:

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            event_count = 0
            events = []

            # Process SSE stream
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_data = line[6:]  # Remove "data: " prefix

                    try:
                        event = json.loads(event_data)
                        events.append(event)
                        event_count += 1

                        event_type = event.get("type")
                        step_name = event.get("step")
                        message = event.get("message", "")
                        data = event.get("data", {})

                        print(f"\n[Event #{event_count}] Type: {event_type}")

                        if step_name:
                            print(f"  Step: {step_name}")

                        if message:
                            print(f"  Message: {message}")

                        # Show relevant data for each event type
                        if event_type == "step_start":
                            print(f"  → Starting step: {step_name}")

                        elif event_type == "step_progress":
                            progress = event.get("progress", 0)
                            total = event.get("total", 0)
                            print(f"  → Progress: {progress}/{total}")

                        elif event_type == "step_complete":
                            print(f"  → Step complete: {step_name}")

                            # Show key metrics from each step
                            if "expanded_queries" in data:
                                print(f"    ✓ Expanded queries: {len(data['expanded_queries'])}")

                            if "segment_count" in data:
                                print(f"    ✓ Segments retrieved: {data['segment_count']}")

                            if "total_segments" in data:
                                print(f"    ✓ Quantitative analysis: {data['total_segments']} segments")
                                if "discourse_centrality" in data:
                                    dc = data["discourse_centrality"]
                                    print(f"    ✓ Discourse centrality: {dc.get('prevalence_rating', 'N/A')}")

                            if "selected_count" in data:
                                print(f"    ✓ Segments selected: {data['selected_count']}")

                            if "summary" in data:
                                summary_preview = data["summary"][:100] + "..." if len(data["summary"]) > 100 else data["summary"]
                                print(f"    ✓ Summary generated: {len(data['summary'])} chars")
                                print(f"    Preview: {summary_preview}")

                        elif event_type == "complete":
                            print(f"  ✓ PIPELINE COMPLETE")

                            # Show final results
                            if "segments" in data:
                                print(f"    Final segments: {len(data['segments'])}")
                            if "summaries" in data:
                                print(f"    Final summaries: {list(data['summaries'].keys())}")
                            if "quantitative_metrics" in data:
                                print(f"    Quantitative metrics included: Yes")

                        elif event_type == "error":
                            error_msg = event.get("error", "Unknown error")
                            print(f"  ✗ ERROR: {error_msg}")
                            pytest.fail(f"Pipeline error: {error_msg}")

                    except json.JSONDecodeError as e:
                        print(f"  ✗ Failed to parse event data: {e}")
                        print(f"  Raw data: {event_data}")

            print("\n" + "="*80)
            print(f"STREAM COMPLETE - Processed {event_count} events")
            print("="*80)

            # Verify we got expected events
            assert event_count > 0, "Should receive at least one event"

            # Verify event sequence
            event_types = [e.get("type") for e in events]
            assert "complete" in event_types or "error" in event_types, "Should have final event"

            # Check for key steps (if no error)
            if "error" not in event_types:
                steps = [e.get("step") for e in events if e.get("step")]
                print(f"\nSteps executed: {steps}")

                assert "expand_query" in steps, "Should expand query"
                assert "retrieve_segments_by_search" in steps, "Should retrieve segments by search"
                assert "quantitative_analysis" in steps, "Should do quantitative analysis"
                assert "select_segments" in steps, "Should select segments"
                assert "generate_summary" in steps, "Should generate summary"

            return events


# ============================================================================
# Test 2: Search Only Workflow
# ============================================================================

@pytest.mark.asyncio
async def test_search_only_sse_stream():
    """Test search_only workflow - no summarization."""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        request_payload = {
            "query": "immigration policy",
            "dashboard_id": "cprmv-practitioner",
            "workflow": "search_only",
            "time_window_days": 7,
            "projects": ["CPRMV"],
            "languages": ["en"]
        }

        print("\n" + "="*80)
        print("SSE STREAMING TEST: search_only workflow")
        print("="*80)

        async with client.stream(
            "POST",
            "/api/analysis/stream",
            json=request_payload,
            timeout=60.0
        ) as response:

            assert response.status_code == 200

            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    events.append(event)
                    print(f"  [{event.get('type')}] {event.get('step', '')} - {event.get('message', '')}")

            # Verify no summarization steps
            steps = [e.get("step") for e in events if e.get("step")]
            assert "generate_summary" not in steps, "search_only should not generate summary"
            assert "expand_query" in steps
            assert "retrieve_segments_by_search" in steps

            print(f"✓ Search only completed with {len(events)} events")


# ============================================================================
# Test 3: Custom Pipeline with Config Overrides
# ============================================================================

@pytest.mark.asyncio
async def test_custom_pipeline_with_overrides():
    """Test custom pipeline definition with config overrides."""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        request_payload = {
            "query": "climate change",
            "dashboard_id": "cprmv-practitioner",
            "workflow": "simple_rag",
            "time_window_days": 30,  # Override to 30 days
            "config_overrides": {
                "retrieve_segments_by_search": {
                    "k": 50  # Override k from 200 to 50
                },
                "select_segments": {
                    "n": 10  # Override n from 20 to 10
                }
            },
            "projects": ["Europe"],
            "languages": ["en"]
        }

        print("\n" + "="*80)
        print("SSE STREAMING TEST: custom pipeline with overrides")
        print("="*80)
        print("Overrides: k=50, n=10, time_window=30d")

        async with client.stream(
            "POST",
            "/api/analysis/stream",
            json=request_payload,
            timeout=120.0
        ) as response:

            assert response.status_code == 200

            events = []
            segment_count = None
            selected_count = None

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    events.append(event)

                    # Track key metrics
                    if event.get("type") == "step_complete":
                        data = event.get("data", {})
                        if "segment_count" in data:
                            segment_count = data["segment_count"]
                            print(f"  Retrieved: {segment_count} segments")
                        if "selected_count" in data:
                            selected_count = data["selected_count"]
                            print(f"  Selected: {selected_count} segments")

            # Verify overrides were applied
            assert segment_count is not None, "Should retrieve segments"
            assert selected_count == 10, f"Should select 10 segments (got {selected_count})"

            print(f"✓ Custom pipeline completed successfully")


# ============================================================================
# Test 4: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_invalid_workflow_error():
    """Test error handling for invalid workflow."""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        request_payload = {
            "query": "test",
            "dashboard_id": "cprmv-practitioner",
            "workflow": "nonexistent_workflow"
        }

        print("\n" + "="*80)
        print("SSE STREAMING TEST: invalid workflow (error handling)")
        print("="*80)

        async with client.stream(
            "POST",
            "/api/analysis/stream",
            json=request_payload,
            timeout=30.0
        ) as response:

            assert response.status_code == 200  # SSE always returns 200

            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    events.append(event)

            # Should get error event
            event_types = [e.get("type") for e in events]
            assert "error" in event_types, "Should receive error event"

            error_event = [e for e in events if e.get("type") == "error"][0]
            assert "workflow" in error_event.get("error", "").lower()

            print(f"✓ Error handled correctly: {error_event.get('error')}")


# ============================================================================
# Test 5: Discourse Centrality with Baseline
# ============================================================================

@pytest.mark.asyncio
async def test_discourse_centrality_with_baseline():
    """Test quantitative analysis includes discourse centrality (prevalence rating)."""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        request_payload = {
            "query": "Pierre Poilievre",
            "dashboard_id": "cprmv-practitioner",
            "workflow": "simple_rag",
            "time_window_days": 7,  # 7-day window for baseline
            "projects": ["CPRMV"],
            "languages": ["en"]
        }

        print("\n" + "="*80)
        print("SSE STREAMING TEST: discourse centrality (prevalence rating)")
        print("="*80)

        async with client.stream(
            "POST",
            "/api/analysis/stream",
            json=request_payload,
            timeout=120.0
        ) as response:

            assert response.status_code == 200

            found_discourse_centrality = False

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])

                    if event.get("type") == "step_complete" and event.get("step") == "quantitative_analysis":
                        data = event.get("data", {})

                        print(f"\nQuantitative Analysis Results:")
                        print(f"  Total segments: {data.get('total_segments', 'N/A')}")

                        if "discourse_centrality" in data:
                            dc = data["discourse_centrality"]
                            found_discourse_centrality = True

                            print(f"\n  Discourse Centrality (Prevalence Rating):")
                            print(f"    Query segments: {dc.get('query_segment_count', 'N/A')}")
                            print(f"    Baseline segments: {dc.get('baseline_segment_count', 'N/A')}")
                            print(f"    Prevalence: {dc.get('prevalence_percentage', 'N/A')}%")
                            print(f"    Rating: {dc.get('prevalence_rating', 'N/A')}")
                            print(f"    Interpretation: {dc.get('interpretation', 'N/A')}")

                            # Verify structure
                            assert "query_segment_count" in dc
                            assert "baseline_segment_count" in dc
                            assert "prevalence_percentage" in dc
                            assert "prevalence_rating" in dc
                            assert "interpretation" in dc

            assert found_discourse_centrality, "Should include discourse centrality in quantitative analysis"
            print(f"\n✓ Discourse centrality verified")


# ============================================================================
# Helper: Run Interactive Test
# ============================================================================

async def run_interactive_test():
    """
    Run an interactive test that shows SSE events in real-time.
    Use this for manual debugging.

    Run: python tests/test_sse_streaming.py
    """
    print("\n" + "="*80)
    print("INTERACTIVE SSE STREAMING TEST")
    print("="*80)

    events = await test_simple_rag_sse_stream()

    print("\n" + "="*80)
    print("EVENT SUMMARY")
    print("="*80)

    for i, event in enumerate(events, 1):
        print(f"{i}. [{event.get('type')}] {event.get('step', '')} - {event.get('message', '')}")

    print(f"\nTotal events: {len(events)}")


if __name__ == "__main__":
    # Run interactive test when executed directly
    asyncio.run(run_interactive_test())
