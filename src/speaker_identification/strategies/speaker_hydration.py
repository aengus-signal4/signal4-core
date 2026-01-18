#!/usr/bin/env python3
"""
Speaker Hydration (Phase 6)
============================

Retrieves detailed biographical information about identified speakers
using external APIs to enrich speaker profiles.

Data Sources (configurable):
- Web Search (primary) - DuckDuckGo + LLM for bio and social profiles
- Wikipedia API - Public figures (fallback)
- LinkedIn API - Professional info (placeholder, API not yet available)
- Custom APIs - Pluggable data sources

Process:
1. Query speaker_identities needing hydration (no bio, or stale data)
2. For each identity, query configured data sources
3. Merge and deduplicate results
4. Update speaker_identities with enriched data

Hydration Fields:
- bio: Brief biography/description
- occupation: Job title / role
- organization: Company / employer
- location: City, State/Province
- country: Country
- social_profiles: {"linkedin": url, "twitter": url, ...}
- external_ids: {"linkedin_id": "...", "wikipedia_id": "..."}
- website: Personal/professional website

Usage:
    # Dry run on all identities needing hydration
    python -m src.speaker_identification.strategies.speaker_hydration

    # Run on specific project
    python -m src.speaker_identification.strategies.speaker_hydration \\
        --project CPRMV --apply

    # Force re-hydration of all identities
    python -m src.speaker_identification.strategies.speaker_hydration \\
        --force --apply
"""

import argparse
import asyncio
import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from sqlalchemy import text

# Add project root to path
from src.utils.paths import get_project_root
project_root = str(get_project_root())
sys.path.append(project_root)

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from src.utils.llm_client import LLMClient
from src.speaker_identification.prompts import PromptRegistry

logger = setup_worker_logger('speaker_identification.speaker_hydration')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Phase key for verification_metadata JSONB
PHASE_KEY = "phase6_hydration"


# =============================================================================
# DATA SOURCE INTERFACE
# =============================================================================

class HydrationDataSource(ABC):
    """
    Abstract base class for hydration data sources.

    Implement this interface to add new data sources (LinkedIn, Wikipedia, etc.)
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this data source."""
        pass

    @abstractmethod
    async def lookup(self, name: str, hints: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Look up a person by name.

        Args:
            name: Full name of the person to look up
            hints: Optional hints to improve matching:
                - organization: Known employer/company
                - occupation: Known job title
                - location: Known location
                - channels: Channels they appear on

        Returns:
            Dict with hydration data or None if not found:
            {
                'bio': str,
                'occupation': str,
                'organization': str,
                'location': str,
                'country': str,
                'website': str,
                'social_profiles': {'linkedin': url, ...},
                'external_ids': {'linkedin_id': '...', ...},
                'confidence': float (0-1),
                'raw_data': dict (original API response for debugging)
            }
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this data source is configured and available."""
        pass


class LinkedInDataSource(HydrationDataSource):
    """
    LinkedIn data source for speaker hydration.

    TODO: Implement with actual LinkedIn API once available.
    This is a placeholder skeleton.
    """

    def __init__(self, api_endpoint: str = None, api_key: str = None):
        """
        Initialize LinkedIn data source.

        Args:
            api_endpoint: LinkedIn API endpoint URL
            api_key: API authentication key
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self._client = None

    @property
    def source_name(self) -> str:
        return "linkedin"

    async def is_available(self) -> bool:
        """Check if LinkedIn API is configured."""
        return bool(self.api_endpoint and self.api_key)

    async def lookup(self, name: str, hints: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Look up a person on LinkedIn.

        TODO: Implement actual API call once endpoint is available.
        """
        if not await self.is_available():
            logger.debug("LinkedIn API not configured")
            return None

        hints = hints or {}

        # TODO: Implement actual LinkedIn API call
        # Placeholder structure for when API is ready:
        #
        # async with aiohttp.ClientSession() as session:
        #     params = {
        #         'name': name,
        #         'company': hints.get('organization'),
        #         'title': hints.get('occupation'),
        #         'location': hints.get('location'),
        #     }
        #     headers = {'Authorization': f'Bearer {self.api_key}'}
        #
        #     async with session.get(self.api_endpoint, params=params, headers=headers) as resp:
        #         if resp.status == 200:
        #             data = await resp.json()
        #             return self._parse_response(data)
        #         else:
        #             logger.warning(f"LinkedIn API error: {resp.status}")
        #             return None

        logger.debug(f"LinkedIn lookup for '{name}' - API not yet implemented")
        return None

    def _parse_response(self, data: Dict) -> Dict:
        """
        Parse LinkedIn API response into standard hydration format.

        TODO: Update based on actual API response structure.
        """
        return {
            'bio': data.get('summary', ''),
            'occupation': data.get('headline', ''),
            'organization': data.get('company', {}).get('name', ''),
            'location': data.get('location', ''),
            'country': data.get('country', ''),
            'website': data.get('website', ''),
            'social_profiles': {
                'linkedin': data.get('profile_url', '')
            },
            'external_ids': {
                'linkedin_id': data.get('id', '')
            },
            'confidence': 0.9,  # High confidence for direct LinkedIn match
            'raw_data': data
        }


class WikipediaDataSource(HydrationDataSource):
    """
    Wikipedia data source for public figures.

    Uses Wikipedia REST API to get biographical information.
    Good for politicians, celebrities, well-known figures.

    API docs: https://en.wikipedia.org/api/rest_v1/
    """

    def __init__(self):
        self.api_endpoint = "https://en.wikipedia.org/api/rest_v1/page/summary"
        self._session = None

    @property
    def source_name(self) -> str:
        return "wikipedia"

    async def is_available(self) -> bool:
        """Wikipedia API is always available."""
        return True

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={'User-Agent': 'SpeakerIdentification/1.0 (signal4.ca)'}
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def lookup(self, name: str, hints: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Look up a person on Wikipedia.

        Args:
            name: Person's name (e.g., "Ezra Levant")
            hints: Optional hints (not used for Wikipedia)

        Returns:
            Hydration data dict or None if not found
        """
        import aiohttp

        # Convert name to Wikipedia title format (spaces -> underscores)
        wiki_title = name.replace(' ', '_')
        url = f"{self.api_endpoint}/{wiki_title}"

        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 404:
                    logger.debug(f"Wikipedia: '{name}' not found")
                    return None

                if resp.status != 200:
                    logger.warning(f"Wikipedia API error for '{name}': {resp.status}")
                    return None

                data = await resp.json()
                return self._parse_response(data, name)

        except aiohttp.ClientError as e:
            logger.warning(f"Wikipedia request error for '{name}': {e}")
            return None
        except Exception as e:
            logger.warning(f"Wikipedia lookup error for '{name}': {e}")
            return None

    def _parse_response(self, data: Dict, original_name: str) -> Optional[Dict]:
        """
        Parse Wikipedia API response into standard hydration format.

        Args:
            data: Wikipedia API response
            original_name: Original search name for validation
        """
        # Check if this is a disambiguation or redirect
        page_type = data.get('type', '')
        if page_type == 'disambiguation':
            logger.debug(f"Wikipedia: '{original_name}' is a disambiguation page")
            return None

        # Get the extract (summary text)
        extract = data.get('extract', '')
        if not extract:
            return None

        # Parse description for occupation hint
        description = data.get('description', '')

        # Extract occupation from description (e.g., "Canadian media personality and conservative activist")
        occupation = description if description else None

        # Try to extract country from description
        country = None
        country_indicators = ['Canadian', 'American', 'British', 'Australian', 'French', 'German']
        for indicator in country_indicators:
            if indicator in description:
                country = indicator.rstrip('n')  # "Canadian" -> "Canada" (rough)
                if indicator == 'Canadian':
                    country = 'Canada'
                elif indicator == 'American':
                    country = 'United States'
                elif indicator == 'British':
                    country = 'United Kingdom'
                break

        # Get image URL if available
        thumbnail = data.get('thumbnail', {})
        image_url = thumbnail.get('source') if thumbnail else None

        # Get Wikipedia page URL
        content_urls = data.get('content_urls', {})
        desktop_urls = content_urls.get('desktop', {})
        wiki_url = desktop_urls.get('page', '')

        # Get Wikidata ID for external_ids
        wikidata_id = data.get('wikibase_item', '')

        return {
            'bio': extract,
            'occupation': occupation,
            'organization': None,  # Not typically in summary
            'location': None,
            'country': country,
            'website': None,
            'image_url': image_url,
            'social_profiles': {
                'wikipedia': wiki_url
            } if wiki_url else {},
            'external_ids': {
                'wikipedia_id': str(data.get('pageid', '')),
                'wikidata_id': wikidata_id
            },
            'confidence': 0.85,  # High confidence for direct Wikipedia match
            'raw_data': {
                'title': data.get('title'),
                'description': description,
                'type': page_type
            }
        }


# =============================================================================
# WEB SEARCH DATA SOURCE (DuckDuckGo + LLM)
# =============================================================================

class WebSearchDataSource(HydrationDataSource):
    """
    Web search data source using DuckDuckGo + LLM verification.

    Process:
    1. Search for biography: "[name] [context] biography"
    2. Fetch Wikipedia or top result content
    3. Per-platform social searches: "[name] twitter", "[name] instagram", etc.
    4. Extract handles from URLs
    5. Final LLM verification pass to clean up and validate

    Cost: Free searches + internal LLM calls
    Rate limit: 1.5s between searches (respectful of DuckDuckGo)
    """

    SEARCH_DELAY = 1.5  # seconds between searches

    # Platforms to search for social profiles
    SOCIAL_PLATFORMS = [
        ("twitter", "twitter OR x.com"),
        ("instagram", "instagram"),
        ("youtube", "youtube channel"),
        ("facebook", "facebook"),
        ("tiktok", "tiktok"),
        ("substack", "substack"),
        ("rumble", "rumble"),
    ]

    def __init__(self, llm_client: 'LLMClient' = None):
        """
        Initialize web search data source.

        Args:
            llm_client: LLM client for bio extraction and verification
        """
        self._session = None
        self._llm_client = llm_client

    @property
    def source_name(self) -> str:
        return "web_search"

    async def is_available(self) -> bool:
        """Web search is always available."""
        return True

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _duckduckgo_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search DuckDuckGo and return results."""
        import re
        from urllib.parse import quote_plus
        from bs4 import BeautifulSoup

        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        try:
            session = await self._get_session()
            async with session.get(search_url, timeout=15) as resp:
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
            logger.debug(f"Search error: {e}")
            return []

    async def _fetch_page_content(self, url: str, max_chars: int = 4000) -> str:
        """Fetch and extract text content from a URL."""
        import re
        from bs4 import BeautifulSoup

        try:
            session = await self._get_session()
            async with session.get(url, timeout=10) as resp:
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

    def _extract_social_handle(self, url: str, platform: str, search_name: str) -> Optional[str]:
        """Extract social media handle from URL with name validation."""
        import re

        patterns = {
            "twitter": [r"(?:twitter\.com|x\.com)/(@?[\w_]+)(?:\?|$|/)"],
            "instagram": [r"instagram\.com/([\w._]+)(?:\?|$|/)"],
            "youtube": [
                r"youtube\.com/@([\w_-]+)",
                r"youtube\.com/c/([\w_-]+)",
                r"youtube\.com/channel/(UC[\w-]+)",
                r"youtube\.com/user/([\w_-]+)",
            ],
            "facebook": [r"facebook\.com/([\w.]+)(?:\?|$|/)"],
            "tiktok": [r"tiktok\.com/@([\w._]+)"],
            "linkedin": [r"linkedin\.com/in/([\w-]+)"],
            "substack": [r"([\w-]+)\.substack\.com"],
            "rumble": [r"rumble\.com/c/([\w-]+)", r"rumble\.com/user/([\w-]+)"],
        }

        if platform not in patterns:
            return None

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

                if len(handle) < 3 or handle.lower() in skip_handles:
                    continue

                # Validate handle looks related to search name for Twitter/Instagram
                if platform in ["twitter", "instagram"]:
                    name_parts = search_name.lower().replace("-", " ").replace("_", " ").split()
                    handle_lower = handle.lower().replace("_", "").replace(".", "")

                    name_match = any(
                        part in handle_lower or handle_lower in part
                        for part in name_parts if len(part) > 2
                    )

                    if not name_match:
                        continue

                # Add @ prefix for platforms that use it
                if platform in ["twitter", "instagram", "tiktok"]:
                    if not handle.startswith("@"):
                        handle = f"@{handle}"

                return handle

        return None

    async def lookup(self, name: str, hints: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Look up a person using web search + LLM verification.

        Args:
            name: Person's name
            hints: Optional hints (channels, role, etc.)

        Returns:
            Hydration data dict or None if not found
        """
        import asyncio
        from urllib.parse import urlparse

        hints = hints or {}
        context = ", ".join(hints.get('channels', [])[:3]) if hints.get('channels') else None

        result = {
            'bio': None,
            'occupation': None,
            'organization': None,
            'location': None,
            'country': None,
            'website': None,
            'social_profiles': {},
            'external_ids': {},
            'confidence': 0.7,
            'raw_data': {'searches': 0}
        }

        # -----------------------------------------------------------------
        # STEP 1: Bio search
        # -----------------------------------------------------------------
        bio_query = f"{name} biography"
        if context:
            bio_query = f"{name} {context} biography"

        bio_results = await self._duckduckgo_search(bio_query, max_results=3)
        result['raw_data']['searches'] += 1

        # Fetch content from Wikipedia or first good result
        bio_content = ""
        for r in bio_results:
            if "wikipedia.org" in r["url"].lower():
                bio_content = await self._fetch_page_content(r["url"], max_chars=3000)
                result['external_ids']['wikipedia_url'] = r["url"]
                break

        if not bio_content and bio_results:
            bio_content = await self._fetch_page_content(bio_results[0]["url"], max_chars=3000)

        # Check for personal website in results
        for r in bio_results:
            url_lower = r["url"].lower()
            name_normalized = name.lower().replace(" ", "").replace("-", "")
            domain = urlparse(r["url"]).netloc.replace("www.", "")
            domain_normalized = domain.replace(".", "").replace("-", "")

            if name_normalized in domain_normalized or any(
                part in domain_normalized for part in name.lower().split() if len(part) > 3
            ):
                if "wikipedia" not in domain and "facebook" not in domain:
                    result['website'] = domain
                    break

        await asyncio.sleep(self.SEARCH_DELAY)

        # -----------------------------------------------------------------
        # STEP 2: Extract bio with LLM
        # -----------------------------------------------------------------
        if bio_content and self._llm_client:
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

            try:
                response = await self._llm_client.call(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512
                )

                # Parse JSON
                text = response.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                data = json.loads(text)
                result['bio'] = data.get('bio')
                result['country'] = data.get('country')
                result['occupation'] = data.get('occupation')
                result['organization'] = data.get('organization')
                if data.get('full_name'):
                    result['raw_data']['full_name'] = data['full_name']

            except Exception as e:
                logger.debug(f"Bio extraction error for {name}: {e}")

        # -----------------------------------------------------------------
        # STEP 3: Per-platform social searches
        # -----------------------------------------------------------------
        for platform, search_term in self.SOCIAL_PLATFORMS:
            query = f"{name} {search_term}"

            search_results = await self._duckduckgo_search(query, max_results=3)
            result['raw_data']['searches'] += 1

            # Extract handle from URLs
            for r in search_results:
                handle = self._extract_social_handle(r["url"], platform, name)
                if handle:
                    result['social_profiles'][platform] = handle
                    break

            await asyncio.sleep(self.SEARCH_DELAY)

        # -----------------------------------------------------------------
        # STEP 4: Final LLM verification
        # -----------------------------------------------------------------
        if self._llm_client and (result['bio'] or result['social_profiles']):
            social_list = "\n".join([
                f"  - {platform}: {handle}"
                for platform, handle in result['social_profiles'].items()
            ]) or "  (none found)"

            verify_prompt = f"""Verify and clean up this speaker profile data. Remove any handles that don't belong to this person.

SPEAKER: {name}
CONTEXT: {context or 'N/A'}

COLLECTED BIO DATA:
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
3. Fix any obvious formatting issues
4. If you know the correct handle from your knowledge, you may add it

Return JSON only:
```json
{{
    "bio": "1-2 sentence bio",
    "country": "country or null",
    "occupation": "occupation or null",
    "organization": "organization or null",
    "website": "domain only or null",
    "social_profiles": {{
        "twitter": "@handle or null",
        "instagram": "@handle or null",
        "youtube": "channel or null",
        "facebook": "page or null",
        "tiktok": "@handle or null",
        "substack": "subdomain or null",
        "rumble": "channel or null"
    }}
}}
```"""

            try:
                response = await self._llm_client.call(
                    messages=[{"role": "user", "content": verify_prompt}],
                    temperature=0.1,
                    max_tokens=512
                )

                text = response.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                verified = json.loads(text)

                # Update with verified data
                result['bio'] = verified.get('bio') or result.get('bio')
                result['country'] = verified.get('country') or result.get('country')
                result['occupation'] = verified.get('occupation') or result.get('occupation')
                result['organization'] = verified.get('organization') or result.get('organization')
                result['website'] = verified.get('website') or result.get('website')

                # Replace social profiles with verified ones
                if verified.get('social_profiles'):
                    result['social_profiles'] = {
                        k: v for k, v in verified['social_profiles'].items()
                        if v and v.lower() not in ["null", "none", "unknown", "not_found"]
                    }

                result['confidence'] = 0.85  # Higher confidence after verification

            except Exception as e:
                logger.debug(f"Verification error for {name}: {e}")

        # Return None if we didn't find anything useful
        if not result['bio'] and not result['social_profiles']:
            return None

        return result


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================

class SpeakerHydrationStrategy:
    """
    Phase 6: Hydrate speaker identities with external biographical data.

    This strategy queries configured data sources (LinkedIn, Wikipedia, etc.)
    to enrich speaker profiles with detailed biographical information.
    """

    def __init__(
        self,
        dry_run: bool = True,
        force: bool = False,
        max_identities: int = None,
        min_episodes: int = 2,
        data_sources: List[HydrationDataSource] = None
    ):
        """
        Initialize strategy.

        Args:
            dry_run: If True, don't make DB changes
            force: If True, re-hydrate identities that already have data
            max_identities: Maximum identities to process per run
            min_episodes: Minimum episodes for identity to be hydrated
            data_sources: List of data sources to query (default: LinkedIn, Wikipedia)
        """
        self.dry_run = dry_run
        self.force = force
        self.max_identities = max_identities
        self.min_episodes = min_episodes

        # Initialize LLM client for synthesis (tier_1 for quality)
        self.llm_client = LLMClient(
            tier="tier_1",
            task_type="text",
            priority=1,  # High priority for quality
        )

        # Initialize data sources
        if data_sources is None:
            self.data_sources = [
                # Web search is now primary - free and finds social profiles
                WebSearchDataSource(llm_client=self.llm_client),
                # Wikipedia as fallback for public figures
                WikipediaDataSource(),
                # LinkedIn placeholder (API not yet available)
                LinkedInDataSource(
                    api_endpoint=self._get_config('linkedin_api_endpoint'),
                    api_key=self._get_config('linkedin_api_key')
                ),
            ]
        else:
            self.data_sources = data_sources

        self.stats = {
            'identities_queried': 0,
            'identities_hydrated': 0,
            'identities_skipped': 0,
            'identities_no_data': 0,
            'api_calls': 0,
            'api_errors': 0,
            'sources_used': {},
            'errors': []
        }

    def _get_config(self, key: str) -> Optional[str]:
        """
        Get configuration value from environment or config file.

        TODO: Implement proper config loading.
        """
        import os
        return os.environ.get(key.upper())

    def _get_identities_for_hydration(self, project: str = None) -> List[Dict]:
        """
        Get speaker identities that need hydration.

        Returns identities where:
        - Has a name (primary_name IS NOT NULL)
        - Has at least min_episodes episodes
        - Either: no bio yet, or force=True
        """
        with get_session() as session:
            filters = [
                "si.primary_name IS NOT NULL",
                "si.is_active = TRUE"
            ]
            params = {'min_episodes': self.min_episodes}

            if project:
                # Filter to identities with speakers in this project
                filters.append("""
                    si.id IN (
                        SELECT DISTINCT s.speaker_identity_id
                        FROM speakers s
                        JOIN content c ON s.content_id = c.content_id
                        WHERE :project = ANY(c.projects)
                          AND s.speaker_identity_id IS NOT NULL
                    )
                """)
                params['project'] = project

            # Skip already hydrated unless force=True
            if not self.force:
                filters.append("(si.bio IS NULL OR si.bio = '')")

            filter_clause = " AND ".join(filters)

            if self.max_identities:
                params['limit'] = self.max_identities
                limit_clause = "LIMIT :limit"
            else:
                limit_clause = ""

            query = text(f"""
                SELECT
                    si.id as identity_id,
                    si.primary_name,
                    si.aliases,
                    si.role,
                    si.occupation,
                    si.organization,
                    si.location,
                    si.bio,
                    si.verification_metadata,
                    COUNT(DISTINCT s.content_id) as episode_count,
                    COALESCE(SUM(s.duration), 0) as total_duration,
                    ARRAY_AGG(DISTINCT ch.display_name) FILTER (WHERE ch.display_name IS NOT NULL) as channels
                FROM speaker_identities si
                LEFT JOIN speakers s ON s.speaker_identity_id = si.id
                LEFT JOIN content c ON s.content_id = c.content_id
                LEFT JOIN channels ch ON c.channel_id = ch.id
                WHERE {filter_clause}
                GROUP BY si.id
                HAVING COUNT(DISTINCT s.content_id) >= :min_episodes
                ORDER BY COUNT(DISTINCT s.content_id) DESC
                {limit_clause}
            """)

            results = session.execute(query, params).fetchall()
            return [dict(row._mapping) for row in results]

    async def _hydrate_identity(self, identity: Dict) -> Optional[Dict]:
        """
        Hydrate a single identity by querying data sources then synthesizing with LLM.

        Process:
        1. Query all available data sources (Wikipedia, LinkedIn, etc.)
        2. Pass raw data to LLM for synthesis into structured profile
        3. Return merged results with LLM-generated bio and extracted fields

        Args:
            identity: Identity dict from database

        Returns:
            Hydration data dict or None if no data found
        """
        name = identity['primary_name']

        # Build hints for data sources
        hints = {
            'organization': identity.get('organization'),
            'occupation': identity.get('occupation'),
            'location': identity.get('location'),
            'channels': identity.get('channels', []),
            'role': identity.get('role')
        }

        # Query all available data sources
        raw_results = []
        wikipedia_data = None
        linkedin_data = None

        for source in self.data_sources:
            if not await source.is_available():
                continue

            try:
                self.stats['api_calls'] += 1
                data = await source.lookup(name, hints)

                if data:
                    data['source'] = source.source_name
                    raw_results.append(data)

                    # Track source-specific data
                    if source.source_name == 'wikipedia':
                        wikipedia_data = data
                    elif source.source_name == 'linkedin':
                        linkedin_data = data

                    # Track source usage
                    source_name = source.source_name
                    self.stats['sources_used'][source_name] = \
                        self.stats['sources_used'].get(source_name, 0) + 1

            except Exception as e:
                logger.warning(f"Error querying {source.source_name} for '{name}': {e}")
                self.stats['api_errors'] += 1
                self.stats['errors'].append(f"{source.source_name}: {str(e)}")

        # If no data found from any source, return None
        if not raw_results:
            return None

        # Synthesize with LLM
        synthesized = await self._synthesize_with_llm(
            name=name,
            wikipedia_data=wikipedia_data,
            linkedin_data=linkedin_data,
            channels=identity.get('channels', []),
            role=identity.get('role')
        )

        if not synthesized:
            # Fall back to raw merge if LLM fails
            return self._merge_hydration_results(raw_results)

        # Merge LLM-synthesized data with external_ids from sources
        merged = {
            'bio': synthesized.get('bio'),
            'occupation': synthesized.get('occupation'),
            'organization': synthesized.get('organization'),
            'location': None,
            'country': synthesized.get('country'),
            'website': synthesized.get('website'),
            'social_profiles': {},
            'external_ids': {},
            'sources': [r['source'] for r in raw_results] + ['llm_synthesis'],
            'hydrated_at': datetime.now(timezone.utc).isoformat()
        }

        # Build social_profiles from LLM output
        if synthesized.get('twitter'):
            merged['social_profiles']['twitter'] = synthesized['twitter']
        if synthesized.get('youtube'):
            merged['social_profiles']['youtube'] = synthesized['youtube']
        if synthesized.get('linkedin'):
            merged['social_profiles']['linkedin'] = synthesized['linkedin']

        # Preserve Wikipedia URL in social_profiles
        if wikipedia_data and wikipedia_data.get('social_profiles', {}).get('wikipedia'):
            merged['social_profiles']['wikipedia'] = wikipedia_data['social_profiles']['wikipedia']

        # Merge external_ids from all sources
        for result in raw_results:
            if result.get('external_ids'):
                merged['external_ids'].update(result['external_ids'])

        return merged

    async def _synthesize_with_llm(
        self,
        name: str,
        wikipedia_data: Optional[Dict],
        linkedin_data: Optional[Dict],
        channels: List[str],
        role: Optional[str]
    ) -> Optional[Dict]:
        """
        Use LLM to synthesize a concise speaker profile from raw data.

        Args:
            name: Speaker name
            wikipedia_data: Raw Wikipedia data (if found)
            linkedin_data: Raw LinkedIn data (if found)
            channels: List of channels the speaker appears on
            role: Known role (host, guest, etc.)

        Returns:
            Dict with synthesized fields or None if LLM fails
        """
        # Build prompt
        prompt = PromptRegistry.phase5_speaker_hydration(
            name=name,
            wikipedia_bio=wikipedia_data.get('bio') if wikipedia_data else None,
            wikipedia_description=wikipedia_data.get('raw_data', {}).get('description') if wikipedia_data else None,
            linkedin_data=json.dumps(linkedin_data.get('raw_data', {}), indent=2) if linkedin_data else None,
            channels=channels,
            role=role
        )

        try:
            self.stats['api_calls'] += 1
            response = await self.llm_client.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )

            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)
            return data

        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON parse error for '{name}': {e}")
            self.stats['api_errors'] += 1
            return None
        except Exception as e:
            logger.warning(f"LLM synthesis error for '{name}': {e}")
            self.stats['api_errors'] += 1
            return None

    def _merge_hydration_results(self, results: List[Dict]) -> Dict:
        """
        Merge hydration results from multiple sources.

        Prefers higher confidence sources for each field.
        Combines social_profiles and external_ids from all sources.
        """
        # Sort by confidence descending
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        merged = {
            'bio': None,
            'occupation': None,
            'organization': None,
            'location': None,
            'country': None,
            'website': None,
            'social_profiles': {},
            'external_ids': {},
            'sources': [],
            'hydrated_at': datetime.now(timezone.utc).isoformat()
        }

        for result in results:
            source = result.get('source', 'unknown')
            merged['sources'].append(source)

            # Take first non-empty value for each field
            for field in ['bio', 'occupation', 'organization', 'location', 'country', 'website']:
                if not merged[field] and result.get(field):
                    merged[field] = result[field]

            # Merge social_profiles
            if result.get('social_profiles'):
                merged['social_profiles'].update(result['social_profiles'])

            # Merge external_ids
            if result.get('external_ids'):
                merged['external_ids'].update(result['external_ids'])

        return merged

    def _save_hydration_result(self, identity_id: int, data: Dict):
        """Save hydration results to speaker_identities table."""
        if self.dry_run:
            return

        with get_session() as session:
            # Build hydration metadata for verification_metadata
            hydration_meta = {
                PHASE_KEY: {
                    'sources': data.get('sources', []),
                    'hydrated_at': data.get('hydrated_at')
                }
            }

            # Prepare JSONB strings
            social_profiles_json = json.dumps(data.get('social_profiles', {}))
            external_ids_json = json.dumps(data.get('external_ids', {}))
            hydration_meta_json = json.dumps(hydration_meta)

            # Use raw SQL with psycopg2-style parameters to handle JSONB casts properly
            # The session.connection() gives us the raw connection
            conn = session.connection()
            conn.execute(
                text("""
                    UPDATE speaker_identities
                    SET
                        bio = COALESCE(:bio, bio),
                        occupation = COALESCE(:occupation, occupation),
                        organization = COALESCE(:organization, organization),
                        location = COALESCE(:location, location),
                        country = COALESCE(:country, country),
                        website = COALESCE(:website, website),
                        social_profiles = COALESCE(social_profiles, '{}'::jsonb) || (:social_profiles)::jsonb,
                        external_ids = COALESCE(external_ids, '{}'::jsonb) || (:external_ids)::jsonb,
                        verification_metadata = COALESCE(verification_metadata, '{}'::jsonb) || (:hydration_meta)::jsonb,
                        updated_at = NOW()
                    WHERE id = :identity_id
                """),
                {
                    'identity_id': identity_id,
                    'bio': data.get('bio'),
                    'occupation': data.get('occupation'),
                    'organization': data.get('organization'),
                    'location': data.get('location'),
                    'country': data.get('country'),
                    'website': data.get('website'),
                    'social_profiles': social_profiles_json,
                    'external_ids': external_ids_json,
                    'hydration_meta': hydration_meta_json
                }
            )
            session.commit()

    async def close(self):
        """Close all data source sessions and LLM client."""
        for source in self.data_sources:
            if hasattr(source, 'close'):
                await source.close()
        if self.llm_client:
            await self.llm_client.close()

    async def run(self, project: str = None) -> Dict:
        """
        Run speaker hydration.

        Args:
            project: Optional project filter

        Returns:
            Stats dict
        """
        logger.info("=" * 80)
        logger.info("SPEAKER HYDRATION (Phase 6)")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Force re-hydration: {self.force}")
        logger.info(f"Min episodes: {self.min_episodes}")
        if project:
            logger.info(f"Project: {project}")

        # Check available data sources
        available_sources = []
        for source in self.data_sources:
            if await source.is_available():
                available_sources.append(source.source_name)

        if not available_sources:
            logger.warning("No data sources available! Configure API credentials.")
            logger.info("Available source types: linkedin, wikipedia")
            return self.stats

        logger.info(f"Available data sources: {', '.join(available_sources)}")
        logger.info("-" * 80)

        # Get identities for hydration
        identities = self._get_identities_for_hydration(project)
        self.stats['identities_queried'] = len(identities)
        logger.info(f"Found {len(identities)} identities for hydration")

        if not identities:
            logger.info("No identities need hydration")
            return self.stats

        # Process each identity
        from tqdm import tqdm

        for identity in tqdm(identities, desc="Hydrating identities"):
            name = identity['primary_name']
            identity_id = identity['identity_id']
            episodes = identity['episode_count']

            try:
                data = await self._hydrate_identity(identity)

                if data:
                    self._save_hydration_result(identity_id, data)
                    self.stats['identities_hydrated'] += 1

                    sources = ", ".join(data.get('sources', []))
                    logger.debug(f"✓ {name} (ID:{identity_id}, {episodes} eps) - hydrated from {sources}")
                else:
                    self.stats['identities_no_data'] += 1
                    logger.debug(f"✗ {name} (ID:{identity_id}) - no data found")

            except Exception as e:
                logger.error(f"Error hydrating {name} (ID:{identity_id}): {e}")
                self.stats['errors'].append(f"{name}: {str(e)}")

        self._print_summary()
        return self.stats

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Identities queried: {self.stats['identities_queried']}")
        logger.info(f"Identities hydrated: {self.stats['identities_hydrated']}")
        logger.info(f"Identities no data: {self.stats['identities_no_data']}")
        logger.info(f"API calls made: {self.stats['api_calls']}")
        logger.info(f"API errors: {self.stats['api_errors']}")
        if self.stats['sources_used']:
            logger.info(f"Sources used: {self.stats['sources_used']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 6: Speaker Hydration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on all identities
  python -m src.speaker_identification.strategies.speaker_hydration

  # Apply hydration on specific project
  python -m src.speaker_identification.strategies.speaker_hydration \\
      --project CPRMV --apply

  # Force re-hydration of all identities
  python -m src.speaker_identification.strategies.speaker_hydration \\
      --force --apply

  # Limit to top 100 identities by episode count
  python -m src.speaker_identification.strategies.speaker_hydration \\
      --max-identities 100 --apply

Environment Variables:
  LINKEDIN_API_ENDPOINT - LinkedIn API endpoint URL
  LINKEDIN_API_KEY      - LinkedIn API authentication key
"""
    )

    parser.add_argument('--project', type=str, help='Filter to project')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--force', action='store_true',
                       help='Re-hydrate identities that already have data')
    parser.add_argument('--max-identities', type=int,
                       help='Maximum identities to process')
    parser.add_argument('--min-episodes', type=int, default=2,
                       help='Minimum episodes for identity to be hydrated (default: 2)')

    args = parser.parse_args()

    strategy = SpeakerHydrationStrategy(
        dry_run=not args.apply,
        force=args.force,
        max_identities=args.max_identities,
        min_episodes=args.min_episodes
    )

    try:
        await strategy.run(project=args.project)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await strategy.close()


if __name__ == '__main__':
    asyncio.run(main())
