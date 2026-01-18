#!/usr/bin/env python3
"""
Targeted Search Hydration - Single bio search + per-platform social searches
=============================================================================

Strategy:
1. Single search for bio: "[name] [podcast] biography" → LLM extracts bio
2. Per-platform searches: "[name] instagram", "[name] twitter", etc.
   → Extract handles directly from URLs in search results (no page fetch needed)

This is more respectful of rate limits and more targeted.

Usage:
    cd ~/signal4/core
    uv run python scripts/test_targeted_search_hydration.py
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote_plus, urlparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(project_root / "src/backend/.env")
load_dotenv(project_root / ".env")


# Rate limiting - be respectful
SEARCH_DELAY = 1.5  # seconds between searches


async def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo with rate limiting."""
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    async with aiohttp.ClientSession() as session:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        try:
            async with session.get(search_url, headers=headers, timeout=15) as resp:
                if resp.status != 200:
                    return []

                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []
                for result in soup.select(".result")[:max_results]:
                    title_elem = result.select_one(".result__title")
                    link_elem = result.select_one(".result__url")
                    snippet_elem = result.select_one(".result__snippet")

                    if title_elem and link_elem:
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
            print(f"    Search error: {e}")
            return []


async def fetch_page_content(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract text content from a URL."""
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

                for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    tag.decompose()

                text = soup.get_text(separator=" ", strip=True)
                text = re.sub(r'\s+', ' ', text)

                return text[:max_chars]

        except Exception:
            return ""


def extract_social_handle_from_url(url: str, platform: str, search_name: str = None) -> str | None:
    """
    Extract social media handle directly from URL.
    Much more reliable than searching page content.

    Args:
        url: The URL to extract from
        platform: Platform name
        search_name: Original search name for validation
    """
    patterns = {
        "twitter": [
            r"(?:twitter\.com|x\.com)/(@?[\w_]+)(?:\?|$|/)",
        ],
        "instagram": [
            r"instagram\.com/([\w._]+)(?:\?|$|/)",
        ],
        "youtube": [
            r"youtube\.com/@([\w_-]+)",
            r"youtube\.com/c/([\w_-]+)",
            r"youtube\.com/channel/(UC[\w-]+)",
            r"youtube\.com/user/([\w_-]+)",
        ],
        "facebook": [
            r"facebook\.com/([\w.]+)(?:\?|$|/)",
        ],
        "tiktok": [
            r"tiktok\.com/@([\w._]+)",
        ],
        "linkedin": [
            r"linkedin\.com/in/([\w-]+)",
        ],
        "bluesky": [
            r"bsky\.app/profile/([\w.]+)",
        ],
        "truth_social": [
            r"truthsocial\.com/@([\w]+)",
        ],
        "rumble": [
            r"rumble\.com/c/([\w-]+)",
            r"rumble\.com/user/([\w-]+)",
        ],
        "substack": [
            r"([\w-]+)\.substack\.com",
        ],
    }

    if platform not in patterns:
        return None

    # Skip common non-profile pages
    skip_handles = {
        "home", "explore", "search", "login", "signup", "about", "help",
        "settings", "intent", "share", "hashtag", "watch", "results",
        "channel", "user", "c", "in", "p", "reel", "stories", "tv",
        "terms", "privacy", "contact", "press", "blog", "jobs", "status",
        "i", "x", "dr", "the", "official", "real", "news", "media"
    }

    for pattern in patterns[platform]:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            handle = match.group(1)

            # Skip too short or generic handles
            if len(handle) < 3:
                continue
            if handle.lower() in skip_handles:
                continue

            # For Twitter/Instagram, validate handle looks related to search name
            if search_name and platform in ["twitter", "instagram"]:
                name_parts = search_name.lower().replace("-", " ").replace("_", " ").split()
                handle_lower = handle.lower().replace("_", "").replace(".", "")

                # Check if any part of name appears in handle
                name_match = any(
                    part in handle_lower or handle_lower in part
                    for part in name_parts if len(part) > 2
                )

                # Also check for known patterns like "dr" + name, name + "official"
                if not name_match:
                    # Skip this result - doesn't look like the person
                    continue

            # Add @ prefix for platforms that use it
            if platform in ["twitter", "instagram", "tiktok", "bluesky", "truth_social"]:
                if not handle.startswith("@"):
                    handle = f"@{handle}"

            return handle

    return None


async def call_llm_balancer(prompt: str, tier: str = "tier_1", priority: int = 1) -> dict:
    """Call via LLM balancer."""
    from src.utils.llm_client import LLMClient

    client = LLMClient(tier=tier, priority=priority, task_type="text")

    try:
        messages = [
            {
                "role": "system",
                "content": "You are a research assistant extracting biographical information from web content. Only use information explicitly stated in the provided content. Be concise."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = await client.call(
            messages=messages,
            temperature=0.1,
            max_tokens=512
        )

        return {"content": response}

    except Exception as e:
        return {"error": f"LLM error: {e}"}
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


async def hydrate_speaker(name: str, context: str = None) -> dict:
    """
    Hydrate a single speaker with targeted searches.

    Args:
        name: Speaker name
        context: Additional context (podcast name, known role, etc.)
    """
    print(f"\n{'='*60}")
    print(f"HYDRATING: {name}")
    if context:
        print(f"Context: {context}")
    print('='*60)

    result = {
        "name": name,
        "full_name": None,
        "bio": None,
        "country": None,
        "occupation": None,
        "organization": None,
        "website": None,
        "social_profiles": {},
        "searches_made": 0,
    }

    # =========================================================================
    # STEP 1: Bio search
    # =========================================================================
    print("\n[1] Searching for biography...")

    bio_query = f"{name} biography"
    if context:
        bio_query = f"{name} {context} biography"

    print(f"    Query: {bio_query}")
    bio_results = await duckduckgo_search(bio_query, max_results=3)
    result["searches_made"] += 1

    # Fetch content from Wikipedia or first good result
    bio_content = ""
    for r in bio_results:
        if "wikipedia.org" in r["url"].lower():
            print(f"    → Found Wikipedia: {r['url'][:50]}...")
            bio_content = await fetch_page_content(r["url"], max_chars=3000)
            break

    if not bio_content and bio_results:
        # Fall back to first result
        print(f"    → Using first result: {bio_results[0]['url'][:50]}...")
        bio_content = await fetch_page_content(bio_results[0]["url"], max_chars=3000)

    # Also check for website in bio results
    for r in bio_results:
        url_lower = r["url"].lower()
        if name.lower().replace(" ", "") in url_lower.replace(".", "").replace("-", ""):
            # Looks like personal website
            domain = urlparse(r["url"]).netloc.replace("www.", "")
            if domain and "wikipedia" not in domain and "facebook" not in domain:
                result["website"] = domain
                print(f"    → Found website: {domain}")
                break

    await asyncio.sleep(SEARCH_DELAY)

    # =========================================================================
    # STEP 2: Extract bio with LLM
    # =========================================================================
    if bio_content:
        print("\n[2] Extracting bio with LLM...")

        prompt = f"""From this web content about {name}, extract:

CONTENT:
{bio_content[:2500]}

Return JSON only:
```json
{{
    "full_name": "full legal name or null",
    "bio": "1-2 sentence description",
    "country": "country or null",
    "occupation": "primary role or null",
    "organization": "employer/affiliation or null"
}}
```"""

        llm_result = await call_llm_balancer(prompt)
        if not llm_result.get("error"):
            data = parse_json_response(llm_result["content"])
            if data:
                result["full_name"] = data.get("full_name")
                result["bio"] = data.get("bio")
                result["country"] = data.get("country")
                result["occupation"] = data.get("occupation")
                result["organization"] = data.get("organization")
                print(f"    ✓ Bio extracted")
    else:
        print("    ✗ No bio content found")

    # =========================================================================
    # STEP 3: Per-platform social searches
    # =========================================================================
    platforms = [
        ("twitter", "twitter OR x.com"),
        ("instagram", "instagram"),
        ("youtube", "youtube channel"),
        ("facebook", "facebook"),
        ("linkedin", "linkedin"),
        ("tiktok", "tiktok"),
        ("substack", "substack"),
        ("rumble", "rumble"),
    ]

    print("\n[3] Searching for social profiles...")

    for platform, search_term in platforms:
        query = f"{name} {search_term}"
        print(f"    {platform}: ", end="", flush=True)

        results = await duckduckgo_search(query, max_results=3)
        result["searches_made"] += 1

        # Try to extract handle from URLs
        handle = None
        for r in results:
            handle = extract_social_handle_from_url(r["url"], platform, search_name=name)
            if handle:
                break

        if handle:
            result["social_profiles"][platform] = handle
            print(f"✓ {handle}")
        else:
            print("✗")

        await asyncio.sleep(SEARCH_DELAY)

    # =========================================================================
    # STEP 4: Final LLM verification pass
    # =========================================================================
    print("\n[4] Final LLM verification...")

    # Build verification prompt with all collected data
    social_list = "\n".join([
        f"  - {platform}: {handle}"
        for platform, handle in result["social_profiles"].items()
    ]) or "  (none found)"

    verify_prompt = f"""Verify and clean up this speaker profile data. Remove any handles that don't belong to this person.

SPEAKER: {name}
CONTEXT: {context or 'N/A'}

COLLECTED BIO DATA:
- Full name: {result.get('full_name') or 'unknown'}
- Bio: {result.get('bio') or 'unknown'}
- Country: {result.get('country') or 'unknown'}
- Occupation: {result.get('occupation') or 'unknown'}
- Organization: {result.get('organization') or 'unknown'}
- Website: {result.get('website') or 'unknown'}

COLLECTED SOCIAL HANDLES (may contain errors):
{social_list}

TASK:
1. Verify each social handle looks correct for this person
2. Remove any handles that are clearly wrong (different person, organization account, etc.)
3. Fix any obvious formatting issues (add @ where needed, etc.)
4. Keep "null" for handles you're unsure about

Return JSON only:
```json
{{
    "full_name": "verified full name",
    "bio": "1-2 sentence bio",
    "country": "country or null",
    "occupation": "occupation or null",
    "organization": "organization or null",
    "website": "domain only or null",
    "social_profiles": {{
        "twitter": "@handle or null",
        "instagram": "@handle or null",
        "youtube": "channel name or ID or null",
        "facebook": "page name or null",
        "tiktok": "@handle or null",
        "linkedin": "profile slug or null",
        "substack": "subdomain or null",
        "rumble": "channel or null"
    }}
}}
```"""

    verify_result = await call_llm_balancer(verify_prompt)

    if not verify_result.get("error"):
        verified_data = parse_json_response(verify_result["content"])
        if verified_data:
            # Update result with verified data
            result["full_name"] = verified_data.get("full_name") or result.get("full_name")
            result["bio"] = verified_data.get("bio") or result.get("bio")
            result["country"] = verified_data.get("country") or result.get("country")
            result["occupation"] = verified_data.get("occupation") or result.get("occupation")
            result["organization"] = verified_data.get("organization") or result.get("organization")
            result["website"] = verified_data.get("website") or result.get("website")

            # Replace social profiles with verified ones (filter nulls)
            if verified_data.get("social_profiles"):
                result["social_profiles"] = {
                    k: v for k, v in verified_data["social_profiles"].items()
                    if v and v.lower() not in ["null", "none", "unknown", "not_found"]
                }

            print(f"    ✓ Verified {len(result['social_profiles'])} social profiles")
    else:
        print(f"    ✗ Verification failed: {verify_result.get('error')}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'─'*40}")
    print(f"Searches made: {result['searches_made']}")
    print(f"Social profiles found: {len(result['social_profiles'])}")

    return result


async def main():
    """Test with 5 speakers of varying fame levels."""

    test_speakers = [
        # Very famous - should have everything
        ("Jordan Peterson", "Daily Wire podcast"),

        # Famous in conservative media
        ("Ezra Levant", "Rebel News"),

        # Well-known Canadian political commentator
        ("David Frum", "The Atlantic"),

        # Less famous - Canadian journalist
        ("Andrew Lawton", "True North"),

        # Lesser known - local podcaster (may have limited info)
        ("Viva Frei", "YouTube legal commentary"),
    ]

    all_results = []

    for name, context in test_speakers:
        result = await hydrate_speaker(name, context)
        all_results.append(result)

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_searches = sum(r["searches_made"] for r in all_results)
    total_socials = sum(len(r["social_profiles"]) for r in all_results)

    print(f"\nTotal searches: {total_searches}")
    print(f"Total social profiles found: {total_socials}")
    print(f"Avg per speaker: {total_socials / len(all_results):.1f} profiles")
    print(f"\nRate limit friendly: {total_searches * SEARCH_DELAY:.0f}s total delay")

    print("\n" + "-" * 70)
    print("RESULTS BY SPEAKER:")
    print("-" * 70)

    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  Full name: {r.get('full_name', 'N/A')}")
        print(f"  Bio: {(r.get('bio') or 'N/A')[:80]}...")
        print(f"  Country: {r.get('country', 'N/A')}")
        print(f"  Occupation: {r.get('occupation', 'N/A')}")
        print(f"  Organization: {r.get('organization', 'N/A')}")
        print(f"  Website: {r.get('website', 'N/A')}")
        print(f"  Social profiles: {r.get('social_profiles', {})}")

    # Cost analysis
    print("\n" + "-" * 70)
    print("COST ANALYSIS:")
    print("-" * 70)
    print(f"DuckDuckGo searches: FREE (but rate limited)")
    print(f"LLM calls: {len(all_results)} × tier_1 = ~$0 (internal)")
    print(f"Grok Live Search equivalent: {len(all_results)} × $0.15 = ${len(all_results) * 0.15:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
