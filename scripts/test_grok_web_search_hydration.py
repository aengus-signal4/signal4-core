#!/usr/bin/env python3
"""
Test Grok Web Search for Speaker Hydration
===========================================

Tests using xAI Grok's Live Search API instead of just Wikipedia
for enriching speaker profiles.

This tests two prompts:
1. Biographical lookup (replaces Wikipedia)
2. Social media handle verification (your second prompt)

Usage:
    cd ~/signal4/core
    uv run python scripts/test_grok_web_search_hydration.py

    # Or test a specific speaker
    uv run python scripts/test_grok_web_search_hydration.py --name "Ezra Levant" --podcast "Rebel News"
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aiohttp
from dotenv import load_dotenv

# Load environment - check multiple locations
load_dotenv(project_root / "src/backend/.env")  # Backend has the API keys
load_dotenv(project_root / ".env")

XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    print("ERROR: XAI_API_KEY not found in environment")
    sys.exit(1)


def create_bio_prompt(name: str, podcast: str = None) -> str:
    """
    Create prompt for biographical lookup using web search.
    This replaces the Wikipedia lookup.
    """
    context = f" who appears on {podcast}" if podcast else ""

    return f"""Use web search to find biographical information about this person.

**Name**: {name}{context}

## SEARCH STRATEGY
1. Search for their Wikipedia page, official website, or LinkedIn profile
2. Look for news articles or interviews that describe who they are
3. Cross-reference multiple sources for accuracy

## OUTPUT FORMAT

Return only this JSON object:

```json
{{
    "full_name": "...",
    "bio": "1-2 sentence description of who this person is",
    "country": "Country of residence/origin",
    "occupation": "Primary role/profession",
    "organization": "Primary affiliation/employer or null",
    "website": "domain only (e.g., rebelnews.com) or null"
}}
```

IMPORTANT:
- Use only verified information from reliable sources
- If unsure about a field, use null
- Bio should be factual and neutral
- Do NOT include birth dates in the bio
"""


def create_social_media_prompt(short_name: str, full_name: str, podcast: str) -> str:
    """
    Create prompt for social media handle lookup.
    This is your second search prompt for verified social accounts.
    """
    return f"""Use web search to find the person's verified social media accounts. Focus on public accounts they use for their commentary work.

## INPUTS

**Name**: {short_name} / {full_name}

**Appears In**: {podcast}

## SEARCH STRATEGY
1. **Start with verification searches**: "[name] verified Twitter", "[name] official Instagram", "[name] social media accounts"
2. **Cross-reference with reliable sources**: Check if accounts are mentioned on their website, Wikipedia, news articles, or official bios
3. **Verify authenticity**: Ensure bio, content, and posting history align with the known person
4. **Use "not_found"** rather than guessing or including unverified accounts

## OUTPUT FORMAT

Return only this JSON object:

```json
{{
    "full_name": "...",
    "social_media": {{
        "twitter": "...",       // @username or "not_found"
        "instagram": "...",     // @username or "not_found"
        "youtube": "...",       // channel name/URL or "not_found"
        "facebook": "...",      // page name/URL or "not_found"
        "twitch": "...",        // @username or "not_found"
        "tiktok": "...",        // @username or "not_found"
        "linkedin": "...",      // profile URL or "not_found"
        "telegram": "...",      // @username or channel or "not_found"
        "bluesky": "...",       // @username or "not_found"
        "truth_social": "..."   // @username or "not_found"
    }}
}}
"""


async def call_grok_with_search(
    prompt: str,
    search_mode: str = "on",
    max_search_results: int = 10,
    model: str = "grok-4"
) -> dict:
    """
    Call Grok API with Live Search enabled.

    Args:
        prompt: User prompt
        search_mode: "on", "auto", or "off"
        max_search_results: Max sources to consider
        model: Model to use (grok-4 recommended for search)

    Returns:
        Dict with response content, citations, and usage
    """
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant helping to find verified biographical and social media information about public figures. Always cite your sources and only return information you can verify."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "search_parameters": {
                "mode": search_mode,
                "return_citations": True,
                "max_search_results": max_search_results,
                "sources": [
                    {"type": "web"},
                    {"type": "x"},  # X/Twitter posts for social verification
                    {"type": "news"}
                ]
            }
        }

        async with session.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return {
                    "error": f"API error {response.status}: {error_text}",
                    "content": None,
                    "citations": [],
                    "usage": {}
                }

            result = await response.json()

            # Extract response
            choices = result.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            citations = choices[0]["message"].get("citations", []) if choices else []
            usage = result.get("usage", {})

            return {
                "content": content,
                "citations": citations,
                "usage": usage,
                "num_sources_used": usage.get("num_sources_used", 0),
                "cost_estimate": f"${usage.get('num_sources_used', 0) * 0.025:.4f}"
            }


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not content:
        return {}

    text = content.strip()

    # Extract JSON from markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw content: {content[:500]}")
        return {}


async def test_speaker_hydration(name: str, podcast: str = None):
    """
    Test the full speaker hydration process with Grok web search.
    """
    print("=" * 80)
    print(f"TESTING SPEAKER HYDRATION WITH GROK WEB SEARCH")
    print("=" * 80)
    print(f"Speaker: {name}")
    if podcast:
        print(f"Podcast: {podcast}")
    print()

    # =========================================================================
    # STEP 1: Biographical Lookup
    # =========================================================================
    print("-" * 40)
    print("STEP 1: Biographical Lookup")
    print("-" * 40)

    bio_prompt = create_bio_prompt(name, podcast)
    print(f"Prompt:\n{bio_prompt[:300]}...")
    print()

    bio_result = await call_grok_with_search(bio_prompt, search_mode="on")

    if bio_result.get("error"):
        print(f"ERROR: {bio_result['error']}")
        return

    print(f"Sources used: {bio_result['num_sources_used']}")
    print(f"Estimated cost: {bio_result['cost_estimate']}")
    print()

    if bio_result.get("citations"):
        print("Citations:")
        for url in bio_result["citations"][:5]:
            print(f"  - {url}")
        print()

    print("Raw Response:")
    print(bio_result["content"][:800])
    print()

    bio_data = parse_json_response(bio_result["content"])
    if bio_data:
        print("Parsed Bio Data:")
        print(json.dumps(bio_data, indent=2))
    print()

    # =========================================================================
    # STEP 2: Social Media Lookup
    # =========================================================================
    print("-" * 40)
    print("STEP 2: Social Media Lookup")
    print("-" * 40)

    # Use full name from bio lookup if available
    full_name = bio_data.get("full_name", name) if bio_data else name
    short_name = name.split()[0] if " " in name else name
    podcast_name = podcast or "Unknown Show"

    social_prompt = create_social_media_prompt(short_name, full_name, podcast_name)
    print(f"Prompt:\n{social_prompt[:300]}...")
    print()

    social_result = await call_grok_with_search(social_prompt, search_mode="on")

    if social_result.get("error"):
        print(f"ERROR: {social_result['error']}")
        return

    print(f"Sources used: {social_result['num_sources_used']}")
    print(f"Estimated cost: {social_result['cost_estimate']}")
    print()

    if social_result.get("citations"):
        print("Citations:")
        for url in social_result["citations"][:5]:
            print(f"  - {url}")
        print()

    print("Raw Response:")
    print(social_result["content"][:800])
    print()

    social_data = parse_json_response(social_result["content"])
    if social_data:
        print("Parsed Social Media Data:")
        print(json.dumps(social_data, indent=2))
    print()

    # =========================================================================
    # COMBINED RESULT
    # =========================================================================
    print("=" * 80)
    print("COMBINED HYDRATION RESULT")
    print("=" * 80)

    combined = {
        "name": full_name,
        "bio": bio_data.get("bio") if bio_data else None,
        "country": bio_data.get("country") if bio_data else None,
        "occupation": bio_data.get("occupation") if bio_data else None,
        "organization": bio_data.get("organization") if bio_data else None,
        "website": bio_data.get("website") if bio_data else None,
        "social_profiles": {}
    }

    # Merge social media (filter out "not_found")
    if social_data and social_data.get("social_media"):
        for platform, handle in social_data["social_media"].items():
            if handle and handle.lower() != "not_found":
                combined["social_profiles"][platform] = handle

    # Add sources used
    total_sources = bio_result["num_sources_used"] + social_result["num_sources_used"]
    total_cost = total_sources * 0.025

    combined["_metadata"] = {
        "sources_used": total_sources,
        "estimated_cost": f"${total_cost:.4f}",
        "bio_citations": bio_result.get("citations", [])[:3],
        "social_citations": social_result.get("citations", [])[:3]
    }

    print(json.dumps(combined, indent=2))

    # =========================================================================
    # COMPARISON WITH WIKIPEDIA-ONLY
    # =========================================================================
    print()
    print("-" * 40)
    print("EVALUATION")
    print("-" * 40)
    print()
    print("Advantages of Web Search over Wikipedia-only:")
    print("  1. Can find social media handles (Wikipedia rarely has these)")
    print("  2. More recent information (Wikipedia can be outdated)")
    print("  3. Finds lesser-known figures (not everyone has Wikipedia pages)")
    print("  4. X/Twitter integration can verify active accounts")
    print()
    print("Disadvantages:")
    print(f"  1. Cost: ${total_cost:.4f} per speaker (vs $0 for Wikipedia)")
    print("  2. Slower (web search takes longer)")
    print("  3. May return unverified information")
    print()

    # Cost projection
    print("Cost Projection for 1000 speakers:")
    print(f"  At {total_sources} sources/speaker: ${1000 * total_cost:.2f}")


async def main():
    parser = argparse.ArgumentParser(description="Test Grok Web Search for Speaker Hydration")
    parser.add_argument("--name", type=str, default="Ezra Levant", help="Speaker name to test")
    parser.add_argument("--podcast", type=str, default="Rebel News", help="Podcast/show name")
    args = parser.parse_args()

    await test_speaker_hydration(args.name, args.podcast)


if __name__ == "__main__":
    asyncio.run(main())
