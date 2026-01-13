"""
Example Usage: Hierarchical Summarization System
=================================================

Demonstrates how to use the hierarchical summarization API.
"""

import requests
import json
from pprint import pprint

# API Base URL
BASE_URL = "http://localhost:7999"  # Adjust to your backend URL


def example_1_single_group_clustering():
    """
    Example 1: Single group with clustering-based theme discovery

    Use case: Discover themes in French religious content over 30 days
    """
    print("=" * 80)
    print("Example 1: Single Group with Clustering")
    print("=" * 80)

    request = {
        "time_window_days": 30,
        "groupings": [
            {
                "group_id": "religious_fr",
                "group_name": "Religious French Content",
                "filter": {
                    "keywords": ["religieux", "chrétien"],
                    "language": "fr"
                }
            }
        ],
        "theme_discovery_method": "clustering",
        "clustering_params": {
            "min_cluster_size": 15,
            "umap_n_components": 50
        },
        "samples_per_theme": 20,
        "generate_meta_summary": True,
        "synthesis_type": "cross_theme"
    }

    response = requests.post(
        f"{BASE_URL}/api/summary/hierarchical/generate",
        json=request,
        timeout=300  # 5 minutes
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success! Generated {result['total_themes']} theme summaries")
        print(f"  Total citations: {result['total_citations']}")
        print(f"  Processing time: {result['processing_time_ms']/1000:.1f}s")
        print(f"  Summary ID: {result['summary_id']}")

        # Print first theme summary
        if result['theme_summaries']:
            theme = result['theme_summaries'][0]
            print(f"\n--- First Theme: {theme['theme_name']} ---")
            print(f"Summary: {theme['summary_text'][:300]}...")
            print(f"Citations: {len(theme['citations'])} segments cited")

        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None


def example_2_multi_group_comparison():
    """
    Example 2: Multiple groups with cross-group comparison

    Use case: Compare English vs French religious discourse
    """
    print("\n" + "=" * 80)
    print("Example 2: Multi-Group Cross-Language Comparison")
    print("=" * 80)

    request = {
        "time_window_days": 30,
        "groupings": [
            {
                "group_id": "religious_en",
                "group_name": "Religious English Content",
                "filter": {
                    "keywords": ["religious", "christian"],
                    "language": "en"
                }
            },
            {
                "group_id": "religious_fr",
                "group_name": "Religious French Content",
                "filter": {
                    "keywords": ["religieux", "chrétien"],
                    "language": "fr"
                }
            }
        ],
        "theme_discovery_method": "clustering",
        "samples_per_theme": 20,
        "generate_meta_summary": True,
        "synthesis_type": "cross_group"  # Compare groups
    }

    response = requests.post(
        f"{BASE_URL}/api/summary/hierarchical/generate",
        json=request,
        timeout=600  # 10 minutes for larger request
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success! Analyzed {result['total_groups']} groups")
        print(f"  Total themes: {result['total_themes']}")

        if result['meta_summary']:
            print(f"\n--- Cross-Group Meta-Summary ---")
            print(result['meta_summary']['synthesis_text'][:500] + "...")

        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None


def example_3_predefined_themes():
    """
    Example 3: Using predefined themes (CPRMV use case)

    Use case: Analyze specific themes in masculinist content
    """
    print("\n" + "=" * 80)
    print("Example 3: Predefined Themes (CPRMV)")
    print("=" * 80)

    request = {
        "time_window_days": 30,
        "groupings": [
            {
                "group_id": "masculinist",
                "group_name": "Masculinist Content",
                "filter": {
                    "keywords": ["masculinist", "men's rights", "antiféminisme"]
                }
            }
        ],
        "theme_discovery_method": "predefined",
        "predefined_themes": [
            {
                "theme_id": "A1",
                "theme_name": "Biological Essentialism",
                "query_variations": [
                    "Men and women are biologically different",
                    "Gender is determined by biology not society",
                    "Natural differences between sexes",
                    "Les hommes et les femmes sont biologiquement différents"
                ]
            },
            {
                "theme_id": "A2",
                "theme_name": "Anti-Feminism",
                "query_variations": [
                    "Feminism has gone too far",
                    "Men are oppressed by feminism",
                    "Feminist ideology is harmful",
                    "Le féminisme opprime les hommes"
                ]
            },
            {
                "theme_id": "B3",
                "theme_name": "Trans Rights Opposition",
                "query_variations": [
                    "Transgender ideology threatens children",
                    "Biological sex cannot be changed",
                    "Protecting children from gender theory",
                    "L'idéologie transgenre menace les enfants"
                ]
            }
        ],
        "samples_per_theme": 20,
        "generate_meta_summary": True,
        "synthesis_type": "cross_theme"
    }

    response = requests.post(
        f"{BASE_URL}/api/summary/hierarchical/generate",
        json=request,
        timeout=600
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success! Analyzed {len(request['predefined_themes'])} predefined themes")

        for theme in result['theme_summaries']:
            print(f"\n--- Theme: {theme['theme_name']} ---")
            print(f"Segments: {theme['segment_count']}")
            print(f"Citations: {len(theme['citations'])}")

        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None


def example_4_retrieve_citation():
    """
    Example 4: Retrieve full segment by citation ID

    Use case: Get source text for a citation found in summary
    """
    print("\n" + "=" * 80)
    print("Example 4: Retrieve Citation")
    print("=" * 80)

    # Example citation ID (replace with actual from your summary)
    citation_id = "[G_religious_fr-T5-S12847]"

    response = requests.get(
        f"{BASE_URL}/api/summary/hierarchical/citation/{citation_id}"
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Retrieved segment for: {citation_id}")
        print(f"  Channel: {result['channel_name']}")
        print(f"  Title: {result['title']}")
        print(f"  Date: {result['publish_date']}")
        print(f"  Text: {result['text'][:200]}...")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None


def example_5_channel_specific():
    """
    Example 5: Analyze specific channels

    Use case: Deep dive into specific content creators
    """
    print("\n" + "=" * 80)
    print("Example 5: Channel-Specific Analysis")
    print("=" * 80)

    request = {
        "time_window_days": 90,  # Longer window for specific channels
        "groupings": [
            {
                "group_id": "jordan_peterson",
                "group_name": "Jordan Peterson Content",
                "filter": {
                    "channel_urls": [
                        "https://www.youtube.com/@JordanBPeterson"
                    ]
                }
            }
        ],
        "theme_discovery_method": "clustering",
        "clustering_params": {
            "min_cluster_size": 10  # Smaller clusters for single channel
        },
        "samples_per_theme": 15,
        "generate_meta_summary": True,
        "synthesis_type": "temporal"  # Temporal evolution
    }

    response = requests.post(
        f"{BASE_URL}/api/summary/hierarchical/generate",
        json=request,
        timeout=300
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success! Analyzed Jordan Peterson content")
        print(f"  Themes discovered: {result['total_themes']}")
        print(f"  Time range: 90 days")

        if result['meta_summary']:
            print(f"\n--- Temporal Evolution Meta-Summary ---")
            print(result['meta_summary']['synthesis_text'][:400] + "...")

        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("HIERARCHICAL SUMMARIZATION SYSTEM - USAGE EXAMPLES")
    print("=" * 80)

    # Check if backend is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"⚠️  Backend not responding at {BASE_URL}")
            print("   Make sure the backend is running: python src/backend/app/main.py")
            return
    except requests.exceptions.RequestException:
        print(f"⚠️  Cannot connect to backend at {BASE_URL}")
        print("   Make sure the backend is running: python src/backend/app/main.py")
        return

    print(f"✓ Backend is running at {BASE_URL}\n")

    # Run examples (comment out as needed)

    # Example 1: Basic clustering
    # example_1_single_group_clustering()

    # Example 2: Multi-group comparison
    # example_2_multi_group_comparison()

    # Example 3: Predefined themes
    # example_3_predefined_themes()

    # Example 4: Retrieve citation
    # example_4_retrieve_citation()

    # Example 5: Channel-specific
    # example_5_channel_specific()

    print("\n" + "=" * 80)
    print("Uncomment examples in main() to run them")
    print("=" * 80)


if __name__ == "__main__":
    main()
