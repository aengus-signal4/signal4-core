#!/usr/bin/env python3
"""
Direct SSE Streaming Test for SimpleRAG Workflow
=================================================

This script directly tests the /api/analysis/stream endpoint
and shows all SSE events step-by-step in real-time.

Usage:
    python test_sse_streaming_manual.py

Requirements:
    - Backend server running on port 8002
    - Database accessible at 10.0.0.4
"""

import requests
import json
import time
from datetime import datetime


def test_sse_streaming(
    query: str = "Pierre Poilievre carbon tax",
    workflow: str = "simple_rag",
    time_window_days: int = 7,
    projects: list = None,
    languages: list = None
):
    """Test SSE streaming with real-time event display."""

    if projects is None:
        projects = ["CPRMV"]
    if languages is None:
        languages = ["en"]

    url = "http://localhost:7999/api/analysis/stream"

    payload = {
        "query": query,
        "dashboard_id": "cprmv-practitioner",
        "workflow": workflow,
        "time_window_days": time_window_days,
        "projects": projects,
        "languages": languages
    }

    print("\n" + "="*80)
    print(f"SSE STREAMING TEST: {workflow}")
    print("="*80)
    print(f"Query: {query}")
    print(f"Time window: {time_window_days} days")
    print(f"Projects: {projects}")
    print(f"Languages: {languages}")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    start_time = time.time()

    try:
        # Send POST request with streaming
        response = requests.post(
            url,
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=120
        )

        if response.status_code != 200:
            print(f"\n✗ ERROR: HTTP {response.status_code}")
            print(response.text)
            return

        print("\n✓ Connected to SSE stream\n")

        event_count = 0
        events = []

        # Process SSE stream line by line
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

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

                    # Calculate elapsed time
                    elapsed = time.time() - start_time

                    print(f"[{elapsed:6.1f}s] Event #{event_count}: {event_type}")

                    if step_name:
                        print(f"          Step: {step_name}")

                    if message:
                        print(f"          Message: {message}")

                    # Show relevant data for each event type
                    if event_type == "step_start":
                        print(f"          → Starting: {step_name}")

                    elif event_type == "step_progress":
                        progress = event.get("progress", 0)
                        total = event.get("total", 0)
                        pct = (progress / total * 100) if total > 0 else 0
                        print(f"          → Progress: {progress}/{total} ({pct:.0f}%)")

                    elif event_type == "step_complete":
                        print(f"          ✓ Completed: {step_name}")

                        # Show key metrics from each step
                        if "expanded_queries" in data:
                            queries = data["expanded_queries"]
                            print(f"            • Expanded queries: {len(queries)}")
                            for i, q in enumerate(queries[:3], 1):  # Show first 3
                                print(f"              {i}. {q}")
                            if len(queries) > 3:
                                print(f"              ... and {len(queries) - 3} more")

                        if "segment_count" in data:
                            print(f"            • Segments retrieved: {data['segment_count']}")

                        if "total_segments" in data:
                            print(f"            • Quantitative analysis:")
                            print(f"              - Total segments: {data.get('total_segments')}")
                            print(f"              - Unique videos: {data.get('unique_videos')}")
                            print(f"              - Unique channels: {data.get('unique_channels')}")

                            if "discourse_centrality" in data:
                                dc = data["discourse_centrality"]
                                print(f"            • Discourse Centrality (Prevalence Rating):")
                                print(f"              - Query segments: {dc.get('query_segment_count')}")
                                print(f"              - Baseline segments: {dc.get('baseline_segment_count')}")
                                print(f"              - Prevalence: {dc.get('prevalence_percentage'):.2f}%")
                                print(f"              - Rating: {dc.get('prevalence_rating')}")
                                print(f"              - Interpretation: {dc.get('interpretation')}")

                        if "selected_count" in data:
                            print(f"            • Segments selected: {data['selected_count']}")
                            print(f"            • Selection strategy: {data.get('strategy', 'N/A')}")

                        if "summary" in data:
                            summary = data["summary"]
                            print(f"            • Summary generated: {len(summary)} chars")
                            preview = summary[:200] + "..." if len(summary) > 200 else summary
                            print(f"            • Preview: {preview}")

                    elif event_type == "complete":
                        print(f"          ✓✓✓ PIPELINE COMPLETE ✓✓✓")

                        # Show final results
                        if "segments" in data:
                            print(f"            • Final segments: {len(data['segments'])}")
                        if "summaries" in data:
                            summary_keys = list(data['summaries'].keys())
                            print(f"            • Final summaries: {summary_keys}")
                        if "quantitative_metrics" in data:
                            print(f"            • Quantitative metrics: ✓ Included")

                    elif event_type == "error":
                        error_msg = event.get("error", "Unknown error")
                        print(f"          ✗✗✗ ERROR: {error_msg}")

                    print()  # Blank line between events

                except json.JSONDecodeError as e:
                    print(f"✗ Failed to parse JSON: {e}")
                    print(f"  Raw data: {event_data}")

        # Summary
        total_time = time.time() - start_time
        print("="*80)
        print(f"STREAM COMPLETE")
        print("="*80)
        print(f"Total events: {event_count}")
        print(f"Total time: {total_time:.1f}s")
        print("="*80)

        # Show event summary
        print("\nEVENT SUMMARY:")
        event_types = {}
        for event in events:
            etype = event.get("type")
            event_types[etype] = event_types.get(etype, 0) + 1

        for etype, count in event_types.items():
            print(f"  {etype}: {count}")

        # Show step sequence
        steps = [e.get("step") for e in events if e.get("step")]
        if steps:
            print(f"\nSTEP SEQUENCE:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")

        return events

    except requests.exceptions.Timeout:
        print("\n✗ ERROR: Request timeout (>120s)")
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to backend server")
        print("  Make sure backend is running on port 8002")
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_workflows():
    """Test available workflows."""
    url = "http://localhost:7999/api/analysis/workflows"

    print("\n" + "="*80)
    print("AVAILABLE WORKFLOWS")
    print("="*80)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            workflows = response.json().get("workflows", {})
            for name, description in workflows.items():
                print(f"\n{name}:")
                print(f"  {description}")
        else:
            print(f"✗ ERROR: HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ ERROR: {e}")


def test_steps():
    """Test available steps."""
    url = "http://localhost:7999/api/analysis/steps"

    print("\n" + "="*80)
    print("AVAILABLE STEPS")
    print("="*80)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            steps = response.json().get("steps", [])
            for step in steps:
                print(f"\n{step['name']}:")
                print(f"  Description: {step.get('description', 'N/A')}")
                print(f"  Parameters: {step.get('parameters', {})}")
        else:
            print(f"✗ ERROR: HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ ERROR: {e}")


if __name__ == "__main__":
    import sys

    # Check if backend is running
    try:
        health_response = requests.get("http://localhost:7999/health", timeout=5)
        if health_response.status_code != 200:
            print("✗ Backend health check failed")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("✗ Backend not running on port 7999")
        print("  Start with: uvicorn src.backend.app.main:app --host 0.0.0.0 --port 7999")
        sys.exit(1)

    # Show available workflows
    test_workflows()

    # Run main SSE streaming test
    print("\n")
    input("Press Enter to start SSE streaming test...")

    test_sse_streaming(
        query="Pierre Poilievre carbon tax",
        workflow="simple_rag",
        time_window_days=7,
        projects=["CPRMV"],
        languages=["en"]
    )

    print("\n✓ Test complete")
