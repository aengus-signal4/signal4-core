#!/usr/bin/env python3
"""
Test Google/DuckDuckGo Search + LLM for Speaker Hydration
==========================================================

Uses free web search (DuckDuckGo) to find pages, then passes content to Grok
for synthesis. Much cheaper than Grok Live Search ($0.15/speaker).

Cost comparison:
- Grok Live Search: $0.15/speaker (6 sources @ $0.025 each)
- This approach: ~$0.002/speaker (just LLM token cost, search is free)

Usage:
    cd ~/signal4/core
    uv run python scripts/test_google_search_hydration.py --name "Ezra Levant" --podcast "Rebel News"
"""

import argparse
import asyncio
import json
import os
import sys
import re
from pathlib import Path
from urllib.parse import quote_plus

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment
load_dotenv(project_root / "src/backend/.env")
load_dotenv(project_root / ".env")

XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    print("ERROR: XAI_API_KEY not found in environment")
    sys.exit(1)


async def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo and return results.
    Free, no API key needed.

    Returns list of {title, url, snippet}
    """
    # Use DuckDuckGo HTML search (no API needed)
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    async with aiohttp.ClientSession() as session:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        try:
            async with session.get(search_url, headers=headers, timeout=15) as resp:
                if resp.status != 200:
                    print(f"  Search failed: {resp.status}")
                    return []

                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []
                for result in soup.select(".result")[:max_results]:
                    title_elem = result.select_one(".result__title")
                    link_elem = result.select_one(".result__url")
                    snippet_elem = result.select_one(".result__snippet")

                    if title_elem and link_elem:
                        # Extract actual URL from DuckDuckGo redirect
                        url = link_elem.get_text(strip=True)
                        if not url.startswith("http"):
                            url = "https://" + url

                        results.append({
                            "title": title_elem.get_text(strip=True),
                            "url": url,
                            "snippet": snippet_elem.get_text(strip=True) if snippet_elem else ""
                        })

                return results

        except Exception as e:
            print(f"  Search error: {e}")
            return []


async def fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and extract text content from a URL.
    """
    async with aiohttp.ClientSession() as session:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    return ""

                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    tag.decompose()

                # Get text
                text = soup.get_text(separator=" ", strip=True)

                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text)

                return text[:max_chars]

        except Exception as e:
            return ""


async def call_llm_balancer(prompt: str, tier: str = "tier_1", priority: int = 1) -> dict:
    """
    Call via LLM balancer - routes through your infrastructure.
    Uses tier_1 (80B models) with priority 1 for best quality.
    """
    from src.utils.llm_client import LLMClient

    client = LLMClient(tier=tier, priority=priority, task_type="text")

    try:
        messages = [
            {
                "role": "system",
                "content": "You are a research assistant extracting biographical and social media information from web content. Only use information explicitly stated in the provided content."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = await client.call(
            messages=messages,
            temperature=0.1,
            max_tokens=1024
        )

        return {
            "content": response,
            "usage": {},  # Balancer doesn't return usage stats
            "cost_estimate": "$0.00 (internal)"  # Using your own infrastructure
        }

    except Exception as e:
        return {"error": f"LLM balancer error: {e}"}
    finally:
        await client.close()


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response."""
    if not content:
        return {}

    text = content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def hydrate_speaker_with_search(name: str, podcast: str = None):
    """
    Full hydration pipeline using free search + LLM synthesis.
    """
    print("=" * 80)
    print("SPEAKER HYDRATION: Free Search + LLM Synthesis")
    print("=" * 80)
    print(f"Speaker: {name}")
    if podcast:
        print(f"Podcast: {podcast}")
    print()

    total_cost = 0.0

    # =========================================================================
    # STEP 1: Search for biographical info
    # =========================================================================
    print("-" * 40)
    print("STEP 1: Search for biographical info")
    print("-" * 40)

    bio_query = f"{name} biography wikipedia"
    if podcast:
        bio_query = f"{name} {podcast} biography"

    print(f"Query: {bio_query}")
    bio_results = await duckduckgo_search(bio_query, max_results=3)
    print(f"Found {len(bio_results)} results")

    # Fetch content from top results
    bio_content = []
    for r in bio_results[:2]:
        print(f"  Fetching: {r['url'][:60]}...")
        content = await fetch_page_content(r['url'])
        if content:
            bio_content.append(f"Source: {r['url']}\n{content[:1500]}")

    print()

    # =========================================================================
    # STEP 2: Search for social media
    # =========================================================================
    print("-" * 40)
    print("STEP 2: Search for social media accounts")
    print("-" * 40)

    social_query = f"{name} twitter instagram official social media"
    print(f"Query: {social_query}")
    social_results = await duckduckgo_search(social_query, max_results=3)
    print(f"Found {len(social_results)} results")

    social_content = []
    for r in social_results[:2]:
        print(f"  Fetching: {r['url'][:60]}...")
        content = await fetch_page_content(r['url'])
        if content:
            social_content.append(f"Source: {r['url']}\n{content[:1000]}")

    print()

    # =========================================================================
    # STEP 3: Synthesize with LLM
    # =========================================================================
    print("-" * 40)
    print("STEP 3: Synthesize with LLM (no live search)")
    print("-" * 40)

    # Combine all content
    all_content = "\n\n---\n\n".join(bio_content + social_content)

    if not all_content:
        print("No content fetched - cannot proceed")
        return

    prompt = f"""Based on the following web content, extract information about {name}.

## WEB CONTENT:
{all_content[:6000]}

## TASK:
Extract biographical and social media information. Only include information explicitly found in the content above.

Return JSON:
```json
{{
    "full_name": "...",
    "bio": "1-2 sentence description",
    "country": "...",
    "occupation": "...",
    "organization": "...",
    "website": "domain only or null",
    "social_media": {{
        "twitter": "@handle or null",
        "instagram": "@handle or null",
        "youtube": "channel or null",
        "facebook": "page or null"
    }}
}}
```

If information is not found in the content, use null. Do not guess."""

    print(f"Prompt length: {len(prompt)} chars")

    result = await call_llm_balancer(prompt, tier="tier_1", priority=1)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    print(f"LLM Cost: {result['cost_estimate']}")
    # Parse cost - handle "internal" case
    cost_str = result['cost_estimate'].replace('$', '').split()[0]
    try:
        total_cost += float(cost_str)
    except ValueError:
        total_cost += 0.0  # Internal = free
    print()

    print("Raw Response:")
    print(result["content"][:800])
    print()

    data = parse_json_response(result["content"])

    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if data:
        # Clean up social media (remove nulls)
        if data.get("social_media"):
            data["social_media"] = {
                k: v for k, v in data["social_media"].items()
                if v and v.lower() not in ["null", "not_found", "none"]
            }

        print(json.dumps(data, indent=2))

    print()
    print("-" * 40)
    print("COST COMPARISON")
    print("-" * 40)
    print(f"This approach:      ${total_cost:.6f}")
    print(f"Grok Live Search:   $0.150000")
    print(f"Savings:            {(0.15 - total_cost) / 0.15 * 100:.1f}%")
    print()
    print("Cost at scale:")
    print(f"  1,000 speakers: ${total_cost * 1000:.2f} (vs $150 with Live Search)")
    print(f"  10,000 speakers: ${total_cost * 10000:.2f} (vs $1,500 with Live Search)")


async def main():
    parser = argparse.ArgumentParser(description="Test Search + LLM for Speaker Hydration")
    parser.add_argument("--name", type=str, default="Ezra Levant", help="Speaker name")
    parser.add_argument("--podcast", type=str, default="Rebel News", help="Podcast name")
    args = parser.parse_args()

    await hydrate_speaker_with_search(args.name, args.podcast)


if __name__ == "__main__":
    asyncio.run(main())
