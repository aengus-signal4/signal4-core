"""
Quick test script for discourse_summary_local workflow.

Tests the local LLM backend integration with the discourse_summary workflow.
Uses tier_2 (30B Qwen3) instead of Grok API.

Usage:
    python tests/test_quick_summary_workflow.py

Or with custom parameters:
    BACKEND_URL=http://localhost:7999 python tests/test_quick_summary_workflow.py
"""

import os
import sys
import json
import asyncio
import aiohttp

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7999")
API_KEY = os.getenv("BACKEND_API_KEY", "sk_14b18617ec904382bc49de7ae19b950a8afd7c5b83854d35")

# Test parameters
TEST_CONFIG = {
    "workflow": "discourse_summary_local",
    "dashboard_id": "health",
    "config_overrides": {
        "retrieve_all_segments": {
            "time_window_days": 7,
        }
    },
    "global_filters": {
        "projects": ["health"],
        "time_window_days": 7
    }
}


async def stream_analysis():
    """Run the discourse_summary_local workflow with SSE streaming."""

    url = f"{BACKEND_URL}/api/analysis/stream"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }

    print(f"\n{'='*60}")
    print(f"Testing discourse_summary_local workflow")
    print(f"{'='*60}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Workflow: {TEST_CONFIG['workflow']}")
    print(f"Dashboard: {TEST_CONFIG['dashboard_id']}")
    print(f"Projects: {TEST_CONFIG['global_filters']['projects']}")
    print(f"Time window: {TEST_CONFIG['global_filters']['time_window_days']} days")
    print(f"{'='*60}\n")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url,
                json=TEST_CONFIG,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=600)  # 10 min timeout
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error: HTTP {response.status}")
                    print(f"Response: {error_text[:500]}")
                    return

                print("✓ Connected to SSE stream\n")

                # Process SSE events
                event_count = 0
                async for line in response.content:
                    line = line.decode('utf-8').strip()

                    if not line:
                        continue

                    if line.startswith('data: '):
                        data_str = line[6:]

                        if data_str == '[DONE]':
                            print("\n✓ Stream complete")
                            break

                        try:
                            event = json.loads(data_str)
                            event_count += 1

                            event_type = event.get('type', 'unknown')

                            if event_type == 'step_start':
                                step = event.get('step', 'unknown')
                                print(f"\n📍 Step: {step}")

                            elif event_type == 'partial':
                                data = event.get('data', {})
                                phase = data.get('phase', '')
                                message = event.get('message', '')
                                progress = event.get('progress', 0)

                                if message:
                                    print(f"   → {message} ({progress*100:.0f}%)")

                            elif event_type == 'result':
                                step = event.get('step', 'unknown')
                                data = event.get('data', {})

                                # Print key metrics based on step
                                if step == 'retrieve_all_segments':
                                    segments = data.get('segments', [])
                                    print(f"   ✓ Retrieved {len(segments):,} segments")

                                elif step == 'corpus_analysis':
                                    stats = data.get('corpus_stats', {})
                                    print(f"   ✓ Corpus: {stats.get('total_episodes', 0)} episodes, "
                                          f"{stats.get('total_duration_hours', 0):.1f} hours, "
                                          f"{stats.get('total_channels', 0)} channels")

                                elif step == 'extract_themes':
                                    themes = data.get('themes', [])
                                    print(f"   ✓ Extracted {len(themes)} themes")
                                    for i, theme in enumerate(themes[:5]):
                                        name = getattr(theme, 'theme_name', None) or theme.get('theme_name', f'Theme {i+1}')
                                        count = len(getattr(theme, 'segments', []) or theme.get('segments', []))
                                        print(f"      {i+1}. {name} ({count} segments)")

                                elif step == 'select_segments':
                                    selected = data.get('selected_segments', [])
                                    print(f"   ✓ Selected {len(selected)} segments for summarization")

                                elif step == 'generate_summaries':
                                    summaries = data.get('summaries', {})
                                    if isinstance(summaries, dict):
                                        for level, content in summaries.items():
                                            if isinstance(content, dict):
                                                overall = content.get('overall_summary', '')
                                                if overall:
                                                    print(f"\n   📝 {level.upper()} SUMMARY (first 500 chars):")
                                                    print(f"   {overall[:500]}...")
                                            elif isinstance(content, str):
                                                print(f"\n   📝 {level.upper()} SUMMARY (first 500 chars):")
                                                print(f"   {content[:500]}...")

                            elif event_type == 'error':
                                error = event.get('error', 'Unknown error')
                                print(f"\n❌ Error: {error}")

                        except json.JSONDecodeError:
                            # Not JSON, might be a heartbeat or other message
                            pass

                print(f"\n{'='*60}")
                print(f"Total events received: {event_count}")
                print(f"{'='*60}")

        except aiohttp.ClientError as e:
            print(f"❌ Connection error: {e}")
        except asyncio.TimeoutError:
            print("❌ Request timed out after 10 minutes")


def main():
    """Main entry point."""
    print("\nDiscourse Summary Local LLM Test")
    print("=" * 40)
    print("This tests the discourse_summary workflow using local tier_2 MLX model")
    print("instead of the Grok API.\n")

    # Check if LLM balancer is accessible
    import requests
    try:
        llm_health = requests.get("http://10.0.0.4:8002/health", timeout=5)
        if llm_health.status_code == 200:
            print("✓ LLM balancer is accessible at 10.0.0.4:8002")
        else:
            print(f"⚠ LLM balancer returned status {llm_health.status_code}")
    except Exception as e:
        print(f"⚠ Could not reach LLM balancer: {e}")
        print("  The test may fail if local LLM is not available.")

    # Run the async test
    asyncio.run(stream_analysis())


if __name__ == "__main__":
    main()
