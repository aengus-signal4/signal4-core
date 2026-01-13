"""
Test QueryParser and SmartRetriever
====================================

Test natural language query parsing and intelligent retrieval
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.rag.query_parser import QueryParser
from app.services.rag.smart_retriever import SmartRetriever
from app.services.llm_service import LLMService
from app.config.dashboard_config import load_dashboard_config


def get_llm_service():
    """Initialize LLMService for tests."""
    config = load_dashboard_config("cprmv-practitioner")
    return LLMService(config, "cprmv-practitioner")


def test_query_parser_llm():
    """Test LLM-based query parsing."""
    print("\n" + "="*60)
    print("Testing QueryParser (LLM-based)")
    print("="*60)

    llm_service = get_llm_service()
    parser = QueryParser(llm_service)

    # Test 1: Project and language extraction
    print("\n[Test 1] Parse: 'Recent French content from CPRMV'")
    filters = parser.parse("Recent French content from CPRMV")
    print(f"  ✓ Projects: {filters['projects']}")
    print(f"  ✓ Languages: {filters['languages']}")
    print(f"  ✓ Time window: {filters['time_window_days']} days")
    print(f"  ✓ Intent: {filters['intent']}")

    assert "CPRMV" in filters["projects"], "Should extract CPRMV project"
    assert "fr" in filters["languages"], "Should extract French language"
    assert filters["time_window_days"] == 30, "Should use default 'recent' = 30 days"

    # Test 2: Time period extraction
    print("\n[Test 2] Parse: 'English videos from the past week'")
    filters = parser.parse("English videos from the past week")
    print(f"  ✓ Languages: {filters['languages']}")
    print(f"  ✓ Time window: {filters['time_window_days']} days")

    # Test 3: Intent extraction
    print("\n[Test 3] Parse: 'Analyze Canadian content about immigration'")
    filters = parser.parse("Analyze Canadian content about immigration")
    print(f"  ✓ Projects: {filters['projects']}")
    print(f"  ✓ Intent: {filters['intent']}")
    print(f"  ✓ Keywords: {filters['keywords']}")

    # Test 4: Compare intent
    print("\n[Test 4] Parse: 'Compare English vs French content'")
    filters = parser.parse("Compare English vs French content")
    print(f"  ✓ Languages: {filters['languages']}")
    print(f"  ✓ Intent: {filters['intent']}")

    # Test 5: Complex query
    print("\n[Test 5] Parse: 'Show me Canadian political discussions from the past 2 months'")
    filters = parser.parse("Show me Canadian political discussions from the past 2 months")
    print(f"  ✓ Projects: {filters['projects']}")
    print(f"  ✓ Time window: {filters['time_window_days']} days")
    print(f"  ✓ Keywords: {filters['keywords']}")
    print(f"  ✓ Intent: {filters['intent']}")

    # Test 6: Multi-project query
    print("\n[Test 6] Parse: 'Compare Europe and Canadian content about climate change'")
    filters = parser.parse("Compare Europe and Canadian content about climate change")
    print(f"  ✓ Projects: {filters['projects']}")
    print(f"  ✓ Keywords: {filters['keywords']}")
    print(f"  ✓ Intent: {filters['intent']}")

    print("\n✅ QueryParser (LLM-based): All tests passed!")


def test_smart_retriever():
    """Test SmartRetriever with natural language queries."""
    print("\n" + "="*60)
    print("Testing SmartRetriever")
    print("="*60)

    llm_service = get_llm_service()
    retriever = SmartRetriever(llm_service)

    # Test 1: Natural language query
    print("\n[Test 1] Natural language: 'Recent French content from CPRMV'")
    result = retriever.retrieve(
        query="Recent French content from CPRMV",
        k=10,
        semantic_search=False  # No semantic search for this test
    )

    print(f"  ✓ Retrieved {len(result['segments'])} segments")
    print(f"  ✓ Filters applied: projects={result['filters']['projects']}")
    print(f"  ✓ Languages: {result['filters']['languages']}")
    print(f"  ✓ Time window: {result['filters']['time_window_days']} days")

    assert len(result["segments"]) > 0, "Should retrieve some segments"
    assert result["filters"]["languages"] == ["fr"], "Should filter to French"

    # Test 2: Count query
    print("\n[Test 2] Count: 'English content from the past week'")
    count = retriever.count(
        query="English content from the past week"
    )
    print(f"  ✓ Count: {count} segments match query")

    # Test 3: Retrieve and prepare index
    print("\n[Test 3] Retrieve and prepare index")
    segments, index, segment_ids, filters = retriever.retrieve_and_prepare_index(
        projects=["CPRMV"],
        languages=["en"],
        time_window_days=7
    )

    print(f"  ✓ Retrieved {len(segments)} segments")
    print(f"  ✓ Index size: {len(segment_ids)} segments")
    print(f"  ✓ Index dimension: {index.d}")

    assert len(segments) > 0, "Should retrieve segments"
    assert index is not None, "Should build index"
    assert len(segment_ids) > 0, "Should have segment IDs"

    # Test 4: Semantic search with natural language query
    print("\n[Test 4] Semantic search: 'immigration policy'")
    result = retriever.retrieve(
        query="immigration policy",
        projects=["CPRMV"],
        time_window_days=30,
        k=5,
        semantic_search=True,
        threshold=0.5
    )

    print(f"  ✓ Found {len(result['segments'])} semantically similar segments")
    if result["segments"]:
        print(f"  ✓ Top result: {result['segments'][0].text[:80]}...")

    print("\n✅ SmartRetriever: All tests passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("QUERY PARSER & SMART RETRIEVER - TEST SUITE (LLM-ONLY)")
    print("="*60)

    try:
        # Test LLM-based parsing
        print("\n[INFO] Testing LLM-based parsing (may take 10-30s)...")
        test_query_parser_llm()

        # Test SmartRetriever
        print("\n[INFO] Testing SmartRetriever...")
        test_smart_retriever()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
