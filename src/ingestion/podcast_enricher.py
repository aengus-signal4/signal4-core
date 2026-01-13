#!/usr/bin/env python3
"""
Podcast Enrichment Module

Enriches podcast charts with metadata from PodcastIndex API and maintains
a master database of all podcasts with historical rankings.
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import requests
from sqlalchemy import and_, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm.attributes import flag_modified
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.session import get_session
from src.database.models import Channel, PodcastChart
from src.utils.logger import setup_worker_logger
from src.utils.project_utils import normalize_language_code
from src.utils.llm_client import LLMClient, create_enrichment_client

logger = setup_worker_logger('podcast_enricher')


# Importance score weighting constants - Western-oriented tiers
# Tier 1: Primary Western markets (20x weight)
# Tier 2: Secondary strategic markets (5x weight)
# Tier 3: All others (1x weight)

COUNTRY_WEIGHTS = {
    # Tier 1 - Primary Western markets
    'us': 20.0,  # United States
    'ca': 20.0,  # Canada
    'gb': 20.0,  # United Kingdom
    'au': 20.0,  # Australia
    'fr': 20.0,  # France

    # Tier 2 - Secondary strategic markets
    'de': 5.0,   # Germany
    'ua': 5.0,   # Ukraine
    'ru': 5.0,   # Russia
    'ng': 5.0,   # Nigeria
    'za': 5.0,   # South Africa
    'nz': 5.0,   # New Zealand
    'ie': 5.0,   # Ireland
    'in': 5.0,   # India
    'ke': 5.0,   # Kenya
    'gh': 5.0,   # Ghana
    'pl': 5.0,   # Poland
    'nl': 5.0,   # Netherlands
    'se': 5.0,   # Sweden
    'mx': 5.0,   # Mexico
    'br': 5.0,   # Brazil
    'jp': 5.0,   # Japan

    # Tier 3 - All other countries default to 1.0
}

PLATFORM_WEIGHTS = {
    'apple': 1.0,   # Apple Podcasts
    'spotify': 0.9  # Slightly lower market share
}

# Categories that represent overall/top charts (most competitive)
OVERALL_CATEGORIES = {'all-podcasts', 'top-podcasts'}

# Category weights - only used for non-overall categories as a penalty
CATEGORY_WEIGHTS = {
    'news': 0.3,           # Major category but still niche vs overall
    'politics': 0.3,
    'news-commentary': 0.25,
    'society-culture': 0.2,
    'history': 0.2,
    'health-fitness': 0.2,
    'business': 0.25,
    # All other categories default to 0.15
}


def calculate_importance_score(monthly_rankings: Dict, current_month: str = None) -> float:
    """
    Calculate channel importance score from historical chart rankings.

    Formula: Best ranking per platform per country per month, summed across all.
    - Takes only ONE entry per (month, platform, country) - the best one
    - Overall charts (all-podcasts, top-podcasts) count at full weight
    - Category-specific charts get heavy penalty (0.15-0.3x) since #3 in health != #200 overall
    - English-speaking markets weighted higher (Tier 1: 20x, Tier 2: 5x, Tier 3: 1x)
    - Recent months weighted higher (0.9^months_ago)

    Args:
        monthly_rankings: Dict of month -> {chart_key: rank}
                         e.g., {"2025-12": {"apple_us_news": 5, "apple_ca_politics": 10}}
        current_month: Optional current month for recency calc (default: now)

    Returns:
        Importance score (higher = more important, no upper bound)
    """
    if not monthly_rankings:
        return 0.0

    if current_month is None:
        current_month = datetime.utcnow().strftime('%Y-%m')

    current_year, current_month_num = int(current_month[:4]), int(current_month[5:7])

    total_weighted_score = 0.0

    for month, charts in monthly_rankings.items():
        if not charts:
            continue

        # Calculate months ago for recency decay
        try:
            year, month_num = int(month[:4]), int(month[5:7])
            months_ago = (current_year - year) * 12 + (current_month_num - month_num)
            recency_factor = 0.9 ** max(0, months_ago)
        except (ValueError, IndexError):
            recency_factor = 0.5  # Old/malformed months get lower weight

        # Group by (platform, country) and find best weighted score for each
        # Key: (platform, country) -> best weighted score
        best_scores = {}

        for chart_key, rank in charts.items():
            # Parse chart_key format: "platform_country_category"
            # e.g., "apple_us_news", "spotify_ca_all-podcasts"
            parts = chart_key.split('_')
            if len(parts) < 3:
                continue

            platform = parts[0]
            country = parts[1]
            category = '_'.join(parts[2:])  # Handle categories like "all-podcasts"

            # Calculate rank score (1-100, higher is better)
            # #1 = 100, #100 = 50.25, #200 = 0.5
            rank_score = 100 * (1 - (rank - 1) / 199)

            # Apply weights
            country_weight = COUNTRY_WEIGHTS.get(country, 1.0)
            platform_weight = PLATFORM_WEIGHTS.get(platform, 1.0)

            # Overall charts (all-podcasts, top-podcasts) get full weight
            # Category-specific charts get heavy penalty
            if category in OVERALL_CATEGORIES:
                category_weight = 1.0
            else:
                category_weight = CATEGORY_WEIGHTS.get(category, 0.15)

            weighted_score = (
                rank_score *
                country_weight *
                platform_weight *
                category_weight *
                recency_factor
            )

            # Keep only the best score per (platform, country)
            key = (platform, country)
            if key not in best_scores or weighted_score > best_scores[key]:
                best_scores[key] = weighted_score

        # Sum the best scores for this month
        total_weighted_score += sum(best_scores.values())

    return round(total_weighted_score, 2)


def update_importance_scores(session, channel_ids: List[int] = None, current_month: str = None) -> int:
    """
    Update importance scores for channels.

    Args:
        session: Database session
        channel_ids: Optional list of channel IDs to update (default: all podcasts)
        current_month: Optional current month for recency calc

    Returns:
        Number of channels updated
    """
    query = session.query(Channel).filter(Channel.platform == 'podcast')

    if channel_ids:
        query = query.filter(Channel.id.in_(channel_ids))

    channels = query.all()
    updated = 0

    for channel in channels:
        monthly_rankings = (channel.platform_metadata or {}).get('monthly_rankings', {})
        new_score = calculate_importance_score(monthly_rankings, current_month)

        if channel.importance_score != new_score:
            channel.importance_score = new_score
            channel.updated_at = datetime.utcnow()
            updated += 1

    session.commit()
    return updated


class PodcastEnricher:
    """Enrich podcasts with metadata from PodcastIndex API"""

    # PodcastIndex API credentials
    API_KEY = "BMFU5QPJZTDVY2XVPCWC"
    API_SECRET = "QKGcr2$HZtGpefJQXNN8zEd#THfY7bkaSnSeKdUP"
    BASE_URL = "https://api.podcastindex.org/api/1.0"

    def __init__(self):
        """Initialize podcast enricher."""
        self.session = requests.Session()
        self._llm_client: Optional[LLMClient] = None
        logger.info("Initialized PodcastEnricher with PostgreSQL backend")

    async def _get_llm_client(self) -> LLMClient:
        """Get or create the LLM client for podcast matching."""
        if self._llm_client is None:
            self._llm_client = create_enrichment_client()  # tier_3, priority 5
        return self._llm_client

    async def close(self):
        """Close resources."""
        if self._llm_client is not None:
            await self._llm_client.close()
            self._llm_client = None

    def _get_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers for PodcastIndex API"""
        epoch_time = str(int(time.time()))
        data_to_hash = self.API_KEY + self.API_SECRET + epoch_time
        sha_1 = hashlib.sha1(data_to_hash.encode()).hexdigest()

        return {
            'X-Auth-Date': epoch_time,
            'X-Auth-Key': self.API_KEY,
            'Authorization': sha_1,
            'User-Agent': 'PodcastEnricher/1.0'
        }

    def _api_search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Raw API search - returns list of feeds.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of feed dicts from API
        """
        time.sleep(2.0)  # Rate limiting

        params = {
            'q': query,
            'max': max_results,
            'clean': 'true'
        }

        headers = self._get_auth_headers()
        response = self.session.get(
            f"{self.BASE_URL}/search/byterm",
            headers=headers,
            params=params,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        if data.get('status') and data.get('count', 0) > 0:
            return data.get('feeds', [])
        return []

    def _score_match(self, feed: Dict, podcast_name: str, creator: str = None) -> Tuple[float, str]:
        """
        Score how well a feed matches the search criteria.

        Returns:
            Tuple of (score, reason)
        """
        feed_title = feed.get('title', '').lower()
        feed_author = feed.get('author', '').lower()
        search_title = podcast_name.lower()
        search_creator = (creator or '').lower()

        # Exact title match
        if feed_title == search_title:
            if not creator or search_creator in feed_author or feed_author in search_creator:
                return (1.0, 'exact_title_and_author')
            return (0.9, 'exact_title')

        # Title contains search (feed title is longer - likely the same podcast with subtitle)
        if search_title in feed_title:
            base_score = len(search_title) / len(feed_title)
            if creator and (search_creator in feed_author or feed_author in search_creator):
                return (min(base_score + 0.3, 0.95), 'partial_title_author_match')
            return (base_score, 'search_in_feed')

        # Feed title contained in search - be more careful here
        # "Alejandro Dolina" matches "Alejandro Dolina LVST" but might be different podcast
        if feed_title in search_title:
            # Only high score if feed author matches search author/creator
            if creator and (search_creator in feed_author or feed_author in search_creator):
                return (0.8, 'feed_in_search_author_match')
            # Otherwise lower score - likely different podcast
            ratio = len(feed_title) / len(search_title)
            return (ratio * 0.5, 'feed_in_search_no_author')

        # Author in feed matches significant part of search
        # e.g., "Alejandro Dolina LVST" search, author "Alejandro Dolina"
        search_words = set(search_title.split()) - {'the', 'a', 'an', 'podcast', 'show', 'with', 'el', 'la', 'los', 'las'}
        author_words = set(feed_author.split())
        author_overlap = search_words & author_words

        if len(author_overlap) >= 2:
            # Author contains significant words from search title
            # Likely the podcast is by this author
            return (0.7, 'author_words_in_search')

        # Check word overlap in title
        feed_words = set(feed_title.split())
        title_overlap = search_words & feed_words
        if len(title_overlap) >= 2:
            return (0.4 * len(title_overlap) / len(search_words), 'word_overlap')

        return (0.0, 'no_match')

    def _extract_metadata(self, feed: Dict) -> Dict:
        """Extract standardized metadata from a feed dict."""
        categories = []
        raw_categories = feed.get('categories') or {}
        for key, value in raw_categories.items():
            if isinstance(value, str):
                categories.append(value)

        return {
            'description': feed.get('description', ''),
            'rss_url': feed.get('url', ''),
            'creator': feed.get('author', ''),
            'categories': categories,
            'episode_count': feed.get('episodeCount', 0),
            'podcast_index_id': str(feed.get('id', '')),
            'language': feed.get('language', ''),
            'last_updated': feed.get('lastUpdateTime', '')
        }

    async def _call_llm_for_match_async(self, podcast_name: str, creator: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Async version: Use LLM to select best match from ambiguous candidates.
        """
        if not candidates:
            return None

        # Filter candidates - must have recent activity (last 90 days) to be charting
        now = time.time()
        ninety_days_ago = now - (90 * 24 * 60 * 60)

        active_candidates = []
        for feed in candidates:
            last_update = feed.get('newestItemPubdate', 0) or feed.get('lastUpdateTime', 0)
            if last_update > ninety_days_ago:
                active_candidates.append(feed)

        if not active_candidates:
            logger.debug(f"No active candidates (with recent episodes) for '{podcast_name}'")
            return None

        # Format candidates for LLM
        candidate_list = []
        for i, feed in enumerate(active_candidates[:8]):
            candidate_list.append({
                'index': i,
                'title': feed.get('title', ''),
                'author': feed.get('author', ''),
                'episode_count': feed.get('episodeCount', 0),
                'description': (feed.get('description', '') or '')[:200]
            })

        candidates = active_candidates

        prompt = f"""Match this podcast from chart data to the correct PodcastIndex entry.

Chart podcast:
- Name: {podcast_name}
- Creator: {creator or 'Unknown'}

Candidates from PodcastIndex:
{json.dumps(candidate_list, indent=2)}

STRICT matching rules:
- Only match if the candidate is CLEARLY the same podcast
- The title OR author must have significant overlap with the chart name
- Abbreviations are OK (e.g., "LVST" = "La Venganza Será Terrible") but must be plausible
- Do NOT match just because topics seem similar
- Do NOT match if the names are completely different with no connection
- When in doubt, return -1

Return JSON with:
- "match_index": index of best match (0-{len(candidate_list)-1}), or -1 if NO CLEAR MATCH
- "confidence": "high" only if names clearly correspond, "medium" if abbreviation/nickname match, "low" if uncertain
- "reasoning": explain the name connection

If none of the candidates have a clear name/author connection to "{podcast_name}", return match_index: -1."""

        try:
            llm_client = await self._get_llm_client()
            llm_response = await llm_client.call_simple(
                prompt=prompt,
                system_prompt='You are a podcast matching expert. Identify which PodcastIndex entry corresponds to the chart podcast. Return only valid JSON.',
                max_tokens=300,
            )

            if llm_response:
                if '```json' in llm_response:
                    llm_response = llm_response.split('```json')[1].split('```')[0]
                elif '```' in llm_response:
                    llm_response = llm_response.split('```')[1].split('```')[0]

                match_data = json.loads(llm_response.strip())
                match_index = match_data.get('match_index', -1)
                confidence = match_data.get('confidence', 'low')

                if match_index >= 0 and match_index < len(candidates):
                    matched = candidates[match_index]
                    if confidence == 'high':
                        logger.info(f"✓ MATCHED '{podcast_name}' -> '{matched.get('title')}' by {matched.get('author')} ({matched.get('episodeCount', 0)} eps) - {match_data.get('reasoning', '')}")
                        return matched
                    else:
                        logger.info(f"✗ REJECTED ({confidence}) '{podcast_name}' -> '{matched.get('title')}' - {match_data.get('reasoning', '')}")
                else:
                    logger.info(f"✗ NO MATCH for '{podcast_name}' ({len(candidates)} candidates) - {match_data.get('reasoning', '')}")

        except Exception as e:
            logger.debug(f"LLM matching failed for '{podcast_name}': {e}")

        return None

    def _call_llm_for_match(self, podcast_name: str, creator: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Use LLM to select best match from ambiguous candidates.

        Args:
            podcast_name: Original podcast name from charts
            creator: Creator name if available
            candidates: List of candidate feeds from API

        Returns:
            Best matching feed or None
        """
        # Use async version via asyncio.run for sync compatibility
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self._call_llm_for_match_async(podcast_name, creator, candidates))
        except RuntimeError:
            # No running loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._call_llm_for_match_async(podcast_name, creator, candidates))

    def search_podcast(self, podcast_name: str, creator: str = None, max_retries: int = 3) -> Optional[Dict]:
        """
        Search for podcast metadata using multi-strategy candidate collection + LLM selection.

        Collects candidates from multiple search strategies, then uses LLM to pick the best match.

        Args:
            podcast_name: Podcast name from charts
            creator: Creator name if available
            max_retries: Max retries per API call

        Returns:
            Metadata dict or None
        """
        all_candidates = []
        seen_ids = set()

        def add_candidates(feeds: List[Dict]):
            """Add feeds to candidates, deduping by ID."""
            for feed in feeds:
                fid = feed.get('id')
                if fid and fid not in seen_ids:
                    seen_ids.add(fid)
                    all_candidates.append(feed)

        for attempt in range(max_retries):
            try:
                # Strategy 1: Full name search (with creator if available)
                search_query = f"{podcast_name} {creator}" if creator and creator not in podcast_name else podcast_name
                feeds1 = self._api_search(search_query, max_results=10)
                add_candidates(feeds1)

                # Check for exact title match - skip LLM if perfect match
                for feed in feeds1:
                    if feed.get('title', '').lower() == podcast_name.lower():
                        logger.debug(f"Exact match found: '{podcast_name}' -> '{feed.get('title')}'")
                        return self._extract_metadata(feed)

                # Strategy 2: Search without common suffixes
                suffixes_to_remove = ['LVST', 'Podcast', 'Show', 'Radio', 'Audio', 'Official', 'The']
                clean_name = podcast_name
                for suffix in suffixes_to_remove:
                    if clean_name.endswith(f' {suffix}'):
                        clean_name = clean_name[:-len(suffix)-1].strip()
                    if clean_name.startswith(f'{suffix} '):
                        clean_name = clean_name[len(suffix)+1:].strip()

                if clean_name != podcast_name and len(clean_name) >= 3:
                    feeds2 = self._api_search(clean_name, max_results=10)
                    add_candidates(feeds2)

                # Strategy 3: Search by creator/author only
                if creator and len(creator) >= 3:
                    feeds3 = self._api_search(creator, max_results=10)
                    add_candidates(feeds3)

                # Strategy 4: Search by individual significant words from podcast name
                name_words = [w for w in podcast_name.split() if len(w) >= 4 and w.lower() not in
                             {'podcast', 'show', 'radio', 'audio', 'the', 'with', 'official'}]
                if len(name_words) >= 2:
                    # Try first two significant words
                    word_query = ' '.join(name_words[:2])
                    if word_query != clean_name and word_query != podcast_name:
                        feeds4 = self._api_search(word_query, max_results=5)
                        add_candidates(feeds4)

                # Use LLM to select best match from all candidates
                if all_candidates:
                    llm_match = self._call_llm_for_match(podcast_name, creator, all_candidates)
                    if llm_match:
                        logger.debug(f"LLM matched '{podcast_name}' -> '{llm_match.get('title')}'")
                        return self._extract_metadata(llm_match)

                # No match found
                return None

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for '{podcast_name}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        return None

    async def search_podcast_async(self, podcast_name: str, creator: str = None) -> Optional[Dict]:
        """
        Async version: Search for podcast metadata using multi-strategy candidate collection + LLM selection.

        API searches are still sync (rate limited anyway), but LLM call is async.
        """
        all_candidates = []
        seen_ids = set()

        def add_candidates(feeds: List[Dict]):
            for feed in feeds:
                fid = feed.get('id')
                if fid and fid not in seen_ids:
                    seen_ids.add(fid)
                    all_candidates.append(feed)

        try:
            # Strategy 1: Full name search
            search_query = f"{podcast_name} {creator}" if creator and creator not in podcast_name else podcast_name
            feeds1 = self._api_search(search_query, max_results=10)
            add_candidates(feeds1)

            # Check for exact title match
            for feed in feeds1:
                if feed.get('title', '').lower() == podcast_name.lower():
                    logger.info(f"✓ EXACT '{podcast_name}' -> '{feed.get('title')}'")
                    return self._extract_metadata(feed)

            # Strategy 2: Search without common suffixes
            suffixes_to_remove = ['LVST', 'Podcast', 'Show', 'Radio', 'Audio', 'Official', 'The']
            clean_name = podcast_name
            for suffix in suffixes_to_remove:
                if clean_name.endswith(f' {suffix}'):
                    clean_name = clean_name[:-len(suffix)-1].strip()
                if clean_name.startswith(f'{suffix} '):
                    clean_name = clean_name[len(suffix)+1:].strip()

            if clean_name != podcast_name and len(clean_name) >= 3:
                feeds2 = self._api_search(clean_name, max_results=10)
                add_candidates(feeds2)

            # Strategy 3: Search by creator
            if creator and len(creator) >= 3:
                feeds3 = self._api_search(creator, max_results=10)
                add_candidates(feeds3)

            # Strategy 4: Search by significant words
            name_words = [w for w in podcast_name.split() if len(w) >= 4 and w.lower() not in
                         {'podcast', 'show', 'radio', 'audio', 'the', 'with', 'official'}]
            if len(name_words) >= 2:
                word_query = ' '.join(name_words[:2])
                if word_query != clean_name and word_query != podcast_name:
                    feeds4 = self._api_search(word_query, max_results=5)
                    add_candidates(feeds4)

            # Use async LLM to select best match
            if all_candidates:
                llm_match = await self._call_llm_for_match_async(podcast_name, creator, all_candidates)
                if llm_match:
                    return self._extract_metadata(llm_match)

            return None

        except Exception as e:
            logger.warning(f"Search failed for '{podcast_name}': {e}")
            return None

    async def enrich_monthly_charts_async(self, month: str, batch_size: int = 8, min_importance: float = 100.0) -> Dict[str, int]:
        """
        Async version: Enrich podcasts with concurrent LLM calls.

        Args:
            month: Month identifier (e.g., "2025-10")
            batch_size: Number of concurrent enrichment tasks
            min_importance: Minimum importance score to enrich (default: 100.0)

        Returns:
            Statistics dictionary
        """
        stats = {
            'unique_podcasts': 0,
            'existing_podcasts': 0,
            'new_podcasts': 0,
            'stale_podcasts': 0,
            'skipped_low_importance': 0,
            'enriched_from_api': 0,
            'failed_enrichments': 0,
            'rankings_updated': 0
        }

        logger.info("="*60)
        logger.info("Starting Podcast Enrichment (Async)")
        logger.info("="*60)
        logger.info(f"Month: {month}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Min importance: {min_importance}")

        with get_session() as session:
            # Step 1: Get all unique podcast channels from charts
            chart_podcasts = session.query(
                Channel.id,
                Channel.display_name,
                Channel.platform_metadata,
                Channel.importance_score
            ).join(
                PodcastChart,
                Channel.id == PodcastChart.channel_id
            ).filter(
                PodcastChart.month == month,
                Channel.platform == 'podcast'
            ).distinct().all()

            stats['unique_podcasts'] = len(chart_podcasts)
            logger.info(f"Found {stats['unique_podcasts']} unique podcasts in {month} charts")

            # Step 2: Identify podcasts needing enrichment
            podcasts_to_enrich = []
            for channel_id, display_name, platform_metadata, importance_score in chart_podcasts:
                platform_metadata = platform_metadata or {}
                last_enriched_str = platform_metadata.get('last_enriched')

                last_enriched = None
                if last_enriched_str:
                    try:
                        last_enriched = datetime.fromisoformat(last_enriched_str.replace('Z', '+00:00'))
                    except:
                        pass

                is_stale = (not last_enriched or
                           (datetime.utcnow() - last_enriched.replace(tzinfo=None)).days > 180)

                if is_stale:
                    # Check importance score threshold
                    if (importance_score or 0) < min_importance:
                        stats['skipped_low_importance'] += 1
                        continue

                    if last_enriched:
                        stats['stale_podcasts'] += 1
                    else:
                        stats['new_podcasts'] += 1
                    creator = platform_metadata.get('creator', '')
                    podcasts_to_enrich.append((channel_id, display_name, creator))
                else:
                    stats['existing_podcasts'] += 1

            logger.info(f"  Already enriched (fresh): {stats['existing_podcasts']}")
            logger.info(f"  New podcasts: {stats['new_podcasts']}")
            logger.info(f"  Stale podcasts: {stats['stale_podcasts']}")
            logger.info(f"  Skipped (importance < {min_importance}): {stats['skipped_low_importance']}")
            logger.info(f"  Total to enrich: {len(podcasts_to_enrich)}")

        # Step 3: Enrich podcasts with async batching
        if podcasts_to_enrich:
            logger.info("Enriching podcasts from PodcastIndex API...")

            semaphore = asyncio.Semaphore(batch_size)
            stats_lock = asyncio.Lock()
            pbar = tqdm(total=len(podcasts_to_enrich), desc="Enriching", unit="podcast")

            async def process_podcast(channel_id: int, display_name: str, creator: str):
                async with semaphore:
                    try:
                        metadata = await self.search_podcast_async(display_name, creator)

                        # Update database (sync - quick operation)
                        with get_session() as session:
                            channel = session.query(Channel).filter_by(id=channel_id).first()
                            if channel:
                                pm = dict(channel.platform_metadata or {})

                                if metadata:
                                    channel.description = metadata.get('description', '')
                                    channel.primary_url = metadata.get('rss_url', channel.primary_url)
                                    channel.language = normalize_language_code(metadata.get('language', '')) or channel.language
                                    pm.update({
                                        'creator': metadata.get('creator', creator),
                                        'categories': metadata.get('categories', []),
                                        'episode_count': metadata.get('episode_count', 0),
                                        'podcast_index_id': metadata.get('podcast_index_id', ''),
                                        'last_enriched': datetime.utcnow().isoformat()
                                    })
                                    async with stats_lock:
                                        stats['enriched_from_api'] += 1
                                else:
                                    pm['last_enriched'] = datetime.utcnow().isoformat()
                                    pm['enrichment_status'] = 'not_found'
                                    async with stats_lock:
                                        stats['failed_enrichments'] += 1

                                channel.platform_metadata = pm
                                flag_modified(channel, 'platform_metadata')
                                channel.updated_at = datetime.utcnow()
                                session.commit()

                    except Exception as e:
                        logger.error(f"Error enriching {display_name}: {e}")
                        async with stats_lock:
                            stats['failed_enrichments'] += 1
                    finally:
                        pbar.update(1)

            # Run all tasks with semaphore controlling concurrency
            tasks = [process_podcast(cid, name, creator) for cid, name, creator in podcasts_to_enrich]
            await asyncio.gather(*tasks)
            pbar.close()

        # Step 4: Update monthly rankings
        logger.info("Updating monthly rankings...")
        with get_session() as session:
            chart_entries = session.query(
                PodcastChart.channel_id,
                PodcastChart.chart_key,
                PodcastChart.rank
            ).filter(
                PodcastChart.month == month
            ).all()

            rankings_by_channel = {}
            for channel_id, chart_key, rank in chart_entries:
                if channel_id not in rankings_by_channel:
                    rankings_by_channel[channel_id] = {}
                rankings_by_channel[channel_id][chart_key] = rank

            for channel_id, chart_rankings in rankings_by_channel.items():
                channel = session.query(Channel).filter_by(id=channel_id).first()
                if channel:
                    pm = dict(channel.platform_metadata or {})
                    monthly_rankings = dict(pm.get('monthly_rankings', {}))
                    monthly_rankings[month] = chart_rankings
                    pm['monthly_rankings'] = monthly_rankings
                    channel.platform_metadata = pm
                    flag_modified(channel, 'platform_metadata')
                    channel.updated_at = datetime.utcnow()
                    stats['rankings_updated'] += 1

            session.commit()

        # Step 5: Update importance scores for all channels with rankings
        logger.info("Updating importance scores...")
        with get_session() as session:
            channel_ids = list(rankings_by_channel.keys())
            scores_updated = update_importance_scores(session, channel_ids, current_month=month)
            stats['scores_updated'] = scores_updated
            logger.info(f"Updated importance scores for {scores_updated} channels")

        # Log summary
        logger.info("="*60)
        logger.info("Enrichment Summary")
        logger.info("="*60)
        logger.info(f"Unique podcasts in charts: {stats['unique_podcasts']}")
        logger.info(f"Already enriched (fresh): {stats['existing_podcasts']}")
        logger.info(f"New podcasts: {stats['new_podcasts']}")
        logger.info(f"Stale podcasts: {stats['stale_podcasts']}")
        logger.info(f"  Enriched from API: {stats['enriched_from_api']}")
        logger.info(f"  Failed enrichments: {stats['failed_enrichments']}")
        logger.info(f"Rankings updated: {stats['rankings_updated']}")
        logger.info(f"Importance scores updated: {stats.get('scores_updated', 0)}")
        logger.info("="*60)

        return stats

    def enrich_monthly_charts(self, month: str) -> Dict[str, int]:
        """
        Enrich all podcasts from monthly charts using channels table.

        Args:
            month: Month identifier (e.g., "2025-10")

        Returns:
            Statistics dictionary
        """
        stats = {
            'unique_podcasts': 0,
            'existing_podcasts': 0,
            'new_podcasts': 0,
            'stale_podcasts': 0,
            'enriched_from_api': 0,
            'failed_enrichments': 0,
            'rankings_updated': 0
        }

        logger.info("="*60)
        logger.info("Starting Podcast Enrichment")
        logger.info("="*60)
        logger.info(f"Month: {month}")

        with get_session() as session:
            # Step 1: Get all unique podcast channels from charts for this month
            chart_podcasts = session.query(
                Channel.id,
                Channel.display_name,
                Channel.platform_metadata
            ).join(
                PodcastChart,
                Channel.id == PodcastChart.channel_id
            ).filter(
                PodcastChart.month == month,
                Channel.platform == 'podcast'
            ).distinct().all()

            stats['unique_podcasts'] = len(chart_podcasts)
            logger.info(f"Found {stats['unique_podcasts']} unique podcasts in {month} charts")

            # Step 2: Identify podcasts needing enrichment
            podcasts_to_enrich = []
            for channel_id, display_name, platform_metadata in chart_podcasts:
                platform_metadata = platform_metadata or {}
                last_enriched_str = platform_metadata.get('last_enriched')

                # Parse last_enriched from ISO string if present
                last_enriched = None
                if last_enriched_str:
                    try:
                        last_enriched = datetime.fromisoformat(last_enriched_str.replace('Z', '+00:00'))
                    except:
                        pass

                # Check if never enriched or stale (>180 days)
                is_stale = (not last_enriched or
                           (datetime.utcnow() - last_enriched.replace(tzinfo=None)).days > 180)

                if is_stale:
                    if last_enriched:
                        stats['stale_podcasts'] += 1
                        logger.debug(f"Stale enrichment for: {display_name}")
                    else:
                        stats['new_podcasts'] += 1

                    creator = platform_metadata.get('creator', '')
                    podcasts_to_enrich.append((channel_id, display_name, creator))
                else:
                    stats['existing_podcasts'] += 1

            logger.info(f"  Already enriched (fresh): {stats['existing_podcasts']}")
            logger.info(f"  New podcasts: {stats['new_podcasts']}")
            logger.info(f"  Stale podcasts: {stats['stale_podcasts']}")
            logger.info(f"  Total to enrich: {len(podcasts_to_enrich)}")

            # Step 3: Enrich podcasts
            if podcasts_to_enrich:
                logger.info("Enriching podcasts from PodcastIndex API...")

                with tqdm(total=len(podcasts_to_enrich), desc="Enriching", unit="podcast") as pbar:
                    for channel_id, display_name, creator in podcasts_to_enrich:
                        pbar.set_description(f"Enriching: {display_name[:30]}...")

                        # Fetch from PodcastIndex API
                        metadata = self.search_podcast(display_name, creator)

                        if metadata:
                            # Update channel metadata in database
                            channel = session.query(Channel).filter_by(id=channel_id).first()
                            if channel:
                                channel.description = metadata.get('description', '')
                                channel.primary_url = metadata.get('rss_url', channel.primary_url)
                                channel.language = normalize_language_code(metadata.get('language', '')) or channel.language

                                # Update platform_metadata with enriched data
                                # Make a copy to ensure SQLAlchemy detects the change
                                pm = dict(channel.platform_metadata or {})
                                pm.update({
                                    'creator': metadata.get('creator', creator),
                                    'categories': metadata.get('categories', []),
                                    'episode_count': metadata.get('episode_count', 0),
                                    'podcast_index_id': metadata.get('podcast_index_id', ''),
                                    'last_enriched': datetime.utcnow().isoformat()
                                })
                                channel.platform_metadata = pm
                                flag_modified(channel, 'platform_metadata')
                                channel.updated_at = datetime.utcnow()
                                session.commit()

                                stats['enriched_from_api'] += 1
                        else:
                            # Mark as checked even if not found, to avoid re-checking
                            channel = session.query(Channel).filter_by(id=channel_id).first()
                            if channel:
                                pm = dict(channel.platform_metadata or {})
                                pm['last_enriched'] = datetime.utcnow().isoformat()
                                pm['enrichment_status'] = 'not_found'
                                channel.platform_metadata = pm
                                flag_modified(channel, 'platform_metadata')
                                channel.updated_at = datetime.utcnow()
                                session.commit()
                            stats['failed_enrichments'] += 1
                            logger.debug(f"No metadata found for: {display_name}")

                        pbar.update(1)

            # Step 4: Update monthly_rankings for all podcasts in this month's charts
            logger.info("Updating monthly rankings...")

            # Get all chart entries for this month grouped by channel
            chart_entries = session.query(
                PodcastChart.channel_id,
                PodcastChart.chart_key,
                PodcastChart.rank
            ).filter(
                PodcastChart.month == month
            ).all()

            # Group by channel_id
            rankings_by_channel = {}
            for channel_id, chart_key, rank in chart_entries:
                if channel_id not in rankings_by_channel:
                    rankings_by_channel[channel_id] = {}
                rankings_by_channel[channel_id][chart_key] = rank

            # Update each channel's monthly_rankings in platform_metadata
            for channel_id, chart_rankings in rankings_by_channel.items():
                channel = session.query(Channel).filter_by(id=channel_id).first()
                if channel:
                    # Get existing platform_metadata - make a copy for SQLAlchemy change detection
                    pm = dict(channel.platform_metadata or {})
                    monthly_rankings = dict(pm.get('monthly_rankings', {}))

                    # Update with current month's rankings
                    monthly_rankings[month] = chart_rankings

                    # Save back to database
                    pm['monthly_rankings'] = monthly_rankings
                    channel.platform_metadata = pm
                    flag_modified(channel, 'platform_metadata')
                    channel.updated_at = datetime.utcnow()

                    stats['rankings_updated'] += 1

            session.commit()

            # Step 5: Update importance scores for all channels with rankings
            logger.info("Updating importance scores...")
            channel_ids = list(rankings_by_channel.keys())
            scores_updated = update_importance_scores(session, channel_ids, current_month=month)
            stats['scores_updated'] = scores_updated
            logger.info(f"Updated importance scores for {scores_updated} channels")

        # Log summary
        logger.info("="*60)
        logger.info("Enrichment Summary")
        logger.info("="*60)
        logger.info(f"Unique podcasts in charts: {stats['unique_podcasts']}")
        logger.info(f"Already enriched (fresh): {stats['existing_podcasts']}")
        logger.info(f"New podcasts: {stats['new_podcasts']}")
        logger.info(f"Stale podcasts: {stats['stale_podcasts']}")
        logger.info(f"  Enriched from API: {stats['enriched_from_api']}")
        logger.info(f"  Failed enrichments: {stats['failed_enrichments']}")
        logger.info(f"Rankings updated: {stats['rankings_updated']}")
        logger.info(f"Importance scores updated: {stats.get('scores_updated', 0)}")
        logger.info("="*60)

        return stats


def enrich_podcasts(month: str, batch_size: int = 8, min_importance: float = 100.0) -> Dict[str, int]:
    """
    Enrich podcasts from monthly charts (async with concurrent LLM calls).

    Args:
        month: Month identifier (e.g., "2025-10")
        batch_size: Number of concurrent enrichment tasks
        min_importance: Minimum importance score to enrich (default: 100.0)

    Returns:
        Statistics dictionary
    """
    enricher = PodcastEnricher()
    return asyncio.run(enricher.enrich_monthly_charts_async(month, batch_size=batch_size, min_importance=min_importance))


def enrich_podcasts_sync(month: str) -> Dict[str, int]:
    """
    Enrich podcasts from monthly charts (sequential, for debugging).

    Args:
        month: Month identifier (e.g., "2025-10")

    Returns:
        Statistics dictionary
    """
    enricher = PodcastEnricher()
    return enricher.enrich_monthly_charts(month)
