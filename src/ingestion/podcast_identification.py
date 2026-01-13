#!/usr/bin/env python3
"""
Generic podcast identification and collection tool.

Collects top podcasts from Podstatus charts, enriches with metadata from PodcastIndex,
classifies by type using LLM, and outputs to project sources.csv.

Usage:
    python src/ingestion/podcast_identification.py \
        --project Europe \
        --countries "fr,de,gb,es,it,ua" \
        --podcast-type "political, current affairs or news" \
        --top-n 200
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config

logger = setup_worker_logger('podcast_identification')


# Country code mapping
COUNTRY_CODES = {
    # Europe
    'fr': {'name': 'France', 'language': 'fr'},
    'de': {'name': 'Germany', 'language': 'de'},
    'gb': {'name': 'United Kingdom', 'language': 'en'},
    'es': {'name': 'Spain', 'language': 'es'},
    'it': {'name': 'Italy', 'language': 'it'},
    'ua': {'name': 'Ukraine', 'language': 'uk'},
    # North America
    'us': {'name': 'United States', 'language': 'en'},
    'ca': {'name': 'Canada', 'language': 'en'},
    'mx': {'name': 'Mexico', 'language': 'es'},
    # Additional European
    'pl': {'name': 'Poland', 'language': 'pl'},
    'nl': {'name': 'Netherlands', 'language': 'nl'},
    'se': {'name': 'Sweden', 'language': 'sv'},
    'no': {'name': 'Norway', 'language': 'no'},
    'dk': {'name': 'Denmark', 'language': 'da'},
    'fi': {'name': 'Finland', 'language': 'fi'},
    'pt': {'name': 'Portugal', 'language': 'pt'},
    'gr': {'name': 'Greece', 'language': 'el'},
    'cz': {'name': 'Czech Republic', 'language': 'cs'},
    'ro': {'name': 'Romania', 'language': 'ro'},
    'hu': {'name': 'Hungary', 'language': 'hu'},
    # Asia-Pacific
    'jp': {'name': 'Japan', 'language': 'ja'},
    'kr': {'name': 'South Korea', 'language': 'ko'},
    'au': {'name': 'Australia', 'language': 'en'},
    'nz': {'name': 'New Zealand', 'language': 'en'},
    'in': {'name': 'India', 'language': 'en'},
    'ph': {'name': 'Philippines', 'language': 'en'},
    # South America
    'br': {'name': 'Brazil', 'language': 'pt'},
    'ar': {'name': 'Argentina', 'language': 'es'},
    'cl': {'name': 'Chile', 'language': 'es'},
    # Africa
    'ng': {'name': 'Nigeria', 'language': 'en'},
    'za': {'name': 'South Africa', 'language': 'en'},
    'ke': {'name': 'Kenya', 'language': 'en'},
    # Additional Europe (Anglosphere)
    'ie': {'name': 'Ireland', 'language': 'en'},
}


@dataclass
class Podcast:
    """Podcast information"""
    rank: int
    name: str
    platform: str
    country_code: str
    country_name: str
    language: str
    description: Optional[str] = None
    rss_url: Optional[str] = None
    creator: Optional[str] = None
    episode_count: Optional[int] = None
    categories: Optional[List[str]] = None
    is_relevant: Optional[bool] = None
    relevance_score: Optional[float] = None
    relevance_reason: Optional[str] = None
    podcast_index_id: Optional[str] = None
    last_updated: Optional[str] = None


class PodstatusScraper:
    """Scrape podcast charts from podstatus.com"""
    
    BASE_URL = "https://podstatus.com/charts"
    PLATFORMS = ["spotify", "apple"]
    
    # URL patterns for different platforms
    PLATFORM_PATTERNS = {
        "spotify": "spotify/{country}/top-podcasts",
        "apple": "applepodcasts/{country}/all-podcasts"
    }
    
    def __init__(self, output_dir: Path = None):
        self.session = None
        self.output_dir = output_dir
        self.collection_dir = None
        if output_dir:
            self.collection_dir = output_dir / "collection"
            self.collection_dir.mkdir(parents=True, exist_ok=True)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_collection_filename(self, platform: str, country_code: str) -> str:
        """Generate collection filename"""
        month_year = datetime.now().strftime("%b%Y").lower()
        return f"{platform}_{country_code}_{month_year}.csv"
    
    def _check_cached_collection(self, platform: str, country_code: str) -> Optional[List[Podcast]]:
        """Check if we have a recent cached collection file"""
        if not self.collection_dir:
            return None
            
        filename = self._get_collection_filename(platform, country_code)
        filepath = self.collection_dir / filename
        
        if filepath.exists():
            logger.info(f"Found cached collection: {filepath}")
            return self._load_collection_csv(filepath, platform, country_code)
        
        return None
    
    def _load_collection_csv(self, filepath: Path, platform: str, country_code: str) -> List[Podcast]:
        """Load podcasts from a collection CSV"""
        podcasts = []
        country_info = COUNTRY_CODES.get(country_code, {
            'name': country_code.upper(),
            'language': 'en'
        })
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    podcast = Podcast(
                        rank=int(row['rank']),
                        name=row['name'],
                        platform=platform.capitalize(),
                        country_code=country_code,
                        country_name=country_info['name'],
                        language=country_info['language'],
                        creator=row.get('creator', None) or None
                    )
                    podcasts.append(podcast)
            
            logger.info(f"Loaded {len(podcasts)} podcasts from cache")
            return podcasts
            
        except Exception as e:
            logger.error(f"Failed to load collection CSV {filepath}: {e}")
            return []
    
    def _save_collection_csv(self, podcasts: List[Podcast], platform: str, country_code: str):
        """Save raw collection data to CSV"""
        if not self.collection_dir or not podcasts:
            return
            
        filename = self._get_collection_filename(platform, country_code)
        filepath = self.collection_dir / filename
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['rank', 'name', 'creator', 'platform', 'country']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for podcast in podcasts:
                    writer.writerow({
                        'rank': podcast.rank,
                        'name': podcast.name,
                        'creator': podcast.creator or '',
                        'platform': podcast.platform,
                        'country': podcast.country_name
                    })
            
            logger.info(f"Saved raw collection to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save collection CSV: {e}")
    
    async def fetch_chart(self, platform: str, country_code: str, top_n: int = 200, 
                         force_fetch: bool = False) -> List[Podcast]:
        """Fetch and parse a single chart"""
        
        # Check for cached collection first (unless force_fetch is True)
        if not force_fetch:
            cached_podcasts = self._check_cached_collection(platform, country_code)
            if cached_podcasts:
                # Return only the requested top_n from cache
                return cached_podcasts[:top_n]
        
        # Use the correct URL pattern for each platform
        if platform in self.PLATFORM_PATTERNS:
            url_pattern = self.PLATFORM_PATTERNS[platform]
            url = f"{self.BASE_URL}/{url_pattern.format(country=country_code)}"
        else:
            # Fallback to old pattern
            url = f"{self.BASE_URL}/{platform}/{country_code}/top-podcasts"
        
        try:
            logger.info(f"Fetching {platform} chart for {country_code} from {url}")
            
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch {url}: HTTP {response.status}")
                    return []
                    
                html = await response.text()
                podcasts = self.parse_chart_html(html, platform, country_code, top_n)
                
                # Save the raw collection data
                if podcasts:
                    self._save_collection_csv(podcasts, platform, country_code)
                
                return podcasts
                
        except Exception as e:
            logger.error(f"Error fetching {platform} {country_code}: {e}")
            return []
    
    def parse_chart_html(self, html: str, platform: str, country_code: str, top_n: int) -> List[Podcast]:
        """Parse Podstatus HTML to extract podcast names"""
        podcasts = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Podstatus uses h4 tags with class "fw-normal mb-1" for podcast titles
            # and small tags with class "text-muted" for creators
            title_elements = soup.find_all('h4', class_='fw-normal mb-1')
            
            if not title_elements:
                logger.warning(f"No podcast titles found in HTML for {platform} {country_code}")
                return []
            
            country_info = COUNTRY_CODES.get(country_code, {
                'name': country_code.upper(),
                'language': 'en'
            })
            
            for idx, title_elem in enumerate(title_elements[:top_n], 1):
                try:
                    # Get podcast name
                    podcast_name = title_elem.get_text(strip=True)
                    
                    # Try to find creator (usually in the next small tag)
                    creator = None
                    parent = title_elem.parent
                    if parent:
                        creator_elem = parent.find('small', class_='text-muted')
                        if creator_elem:
                            creator = creator_elem.get_text(strip=True)
                    
                    if podcast_name:
                        podcast = Podcast(
                            rank=idx,
                            name=podcast_name,
                            platform=platform.capitalize(),
                            country_code=country_code,
                            country_name=country_info['name'],
                            language=country_info['language'],
                            creator=creator
                        )
                        podcasts.append(podcast)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse podcast {idx}: {e}")
                    continue
            
            logger.info(f"Parsed {len(podcasts)} podcasts from {platform} {country_code}")
            
        except Exception as e:
            logger.error(f"Failed to parse HTML for {platform} {country_code}: {e}")
        
        return podcasts
    
    async def fetch_all_charts(self, country_codes: List[str], platforms: Optional[List[str]] = None, 
                              top_n: int = 200) -> Dict[str, List[Podcast]]:
        """Fetch charts for multiple countries and platforms"""
        if platforms is None:
            platforms = self.PLATFORMS
        
        # First, determine what we need to fetch
        current_month = datetime.now().strftime("%b%Y").lower()
        required_charts = []
        cached_charts = []
        
        for country_code in country_codes:
            for platform in platforms:
                chart_id = (platform, country_code)
                filename = self._get_collection_filename(platform, country_code)
                filepath = self.collection_dir / filename if self.collection_dir else None
                
                # Check if we have a cached file for the current month
                if filepath and filepath.exists():
                    # Check if it's from the current month
                    if current_month in filename:
                        cached_charts.append(chart_id)
                        logger.info(f"✓ Found current month cache for {platform} {country_code}: {filename}")
                    else:
                        # Old month, need to fetch fresh
                        required_charts.append(chart_id)
                        logger.info(f"⟳ Cache from old month for {platform} {country_code}, will fetch fresh")
                else:
                    required_charts.append(chart_id)
                    logger.info(f"✗ No cache for {platform} {country_code}, will fetch")
        
        # Summary
        total_charts = len(country_codes) * len(platforms)
        logger.info(f"Chart status: {len(cached_charts)} cached, {len(required_charts)} to fetch (total: {total_charts})")
        
        # Fetch all charts (both cached and new)
        all_podcasts = {}
        charts_fetched = 0
        
        # Create progress bar for collection
        with tqdm(total=total_charts, desc="Collecting charts", unit="chart") as pbar:
            for country_code in country_codes:
                country_podcasts = []
                
                for platform in platforms:
                    chart_id = (platform, country_code)
                    pbar.set_description(f"Collecting {platform} {country_code}")
                    
                    if chart_id in cached_charts:
                        # Load from cache
                        podcasts = self._check_cached_collection(platform, country_code)
                        if podcasts:
                            country_podcasts.extend(podcasts[:top_n])
                        pbar.update(1)
                    
                    elif chart_id in required_charts:
                        # Need to fetch from web
                        charts_fetched += 1
                        logger.info(f"Fetching {charts_fetched}/{len(required_charts)}: {platform} {country_code}")
                        
                        podcasts = await self.fetch_chart(platform, country_code, top_n, force_fetch=True)
                        country_podcasts.extend(podcasts)
                        
                        pbar.update(1)
                        
                        # Be respectful to the server - 4-6 second delay between fetches
                        if charts_fetched < len(required_charts):
                            delay = 4 + (charts_fetched % 3)  # Varies between 4-6 seconds
                            logger.info(f"Waiting {delay} seconds before next request...")
                            await asyncio.sleep(delay)
                
                all_podcasts[country_code] = country_podcasts
        
        logger.info(f"Collection complete: {len(cached_charts)} from cache, {charts_fetched} fetched fresh")
        return all_podcasts


class PodcastEnricher:
    """Enrich podcasts with metadata from PodcastIndex API"""
    
    # PodcastIndex API credentials (from existing script)
    API_KEY = "BMFU5QPJZTDVY2XVPCWC"
    API_SECRET = "QKGcr2$HZtGpefJQXNN8zEd#THfY7bkaSnSeKdUP"
    BASE_URL = "https://api.podcastindex.org/api/1.0"
    
    def __init__(self, output_dir: Path = None):
        self.session = requests.Session()
        self.output_dir = output_dir
        self.cache_dir = None
        self.enrichment_cache = {}
        if output_dir:
            self.cache_dir = output_dir / "cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_enrichment_cache()
        
    def _load_enrichment_cache(self):
        """Load cached enrichment data (persistent across months)"""
        cache_file = self.cache_dir / "enrichment_cache_persistent.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.enrichment_cache = json.load(f)
                logger.info(f"Loaded enrichment cache with {len(self.enrichment_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load enrichment cache: {e}")
                self.enrichment_cache = {}
    
    def _save_enrichment_cache(self):
        """Save enrichment cache to disk (persistent across months)"""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / "enrichment_cache_persistent.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.enrichment_cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved enrichment cache with {len(self.enrichment_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save enrichment cache: {e}")
    
    def _get_cache_key(self, podcast_name: str) -> str:
        """Generate unique cache key for a podcast (name only, works across platforms/countries)"""
        # Just use the podcast name as key since the same podcast has the same metadata everywhere
        return podcast_name.lower().strip()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers for PodcastIndex API"""
        epoch_time = str(int(time.time()))
        data_to_hash = self.API_KEY + self.API_SECRET + epoch_time
        sha_1 = hashlib.sha1(data_to_hash.encode()).hexdigest()
        
        return {
            'X-Auth-Date': epoch_time,
            'X-Auth-Key': self.API_KEY,
            'Authorization': sha_1,
            'User-Agent': 'PodcastIdentification/1.0'
        }
    
    def search_podcast(self, podcast_name: str, creator: str = None, max_retries: int = 3) -> Optional[Dict]:
        """Search for podcast metadata using PodcastIndex API"""
        
        # Build search query - include creator if available for better matching
        if creator and creator not in podcast_name:
            search_query = f"{podcast_name} {creator}"
        else:
            search_query = podcast_name
        
        for attempt in range(max_retries):
            try:
                # Rate limiting - be respectful to the API
                time.sleep(1.5)
                
                params = {
                    'q': search_query,
                    'max': 5,
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
                    feeds = data['feeds']
                    
                    # Find best match
                    best_match = None
                    best_score = 0
                    
                    for feed in feeds[:3]:
                        feed_title = feed.get('title', '').lower()
                        feed_author = feed.get('author', '').lower()
                        search_title = podcast_name.lower()
                        search_creator = (creator or '').lower()
                        
                        # Exact title match
                        if feed_title == search_title:
                            # If we have a creator, verify it matches too
                            if not creator or search_creator in feed_author or feed_author in search_creator:
                                best_match = feed
                                break
                            else:
                                # Title matches but creator doesn't - still consider it
                                score = 0.8
                        # Partial title match
                        elif search_title in feed_title or feed_title in search_title:
                            score = len(search_title) / max(len(feed_title), 1)
                            # Boost score if creator also matches
                            if creator and (search_creator in feed_author or feed_author in search_creator):
                                score += 0.3
                        else:
                            continue
                            
                        if score > best_score:
                            best_score = score
                            best_match = feed
                    
                    if best_match:
                        return {
                            'description': best_match.get('description', ''),
                            'rss_url': best_match.get('url', ''),
                            'creator': best_match.get('author', ''),
                            'categories': self._parse_categories(best_match.get('categories', {})),
                            'episode_count': best_match.get('episodeCount', 0),
                            'podcast_index_id': str(best_match.get('id', '')),
                            'language': best_match.get('language', ''),
                            'last_updated': best_match.get('lastUpdateTime', '')
                        }
                
                return None
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for '{podcast_name}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        return None
    
    def _parse_categories(self, categories_dict: Dict) -> List[str]:
        """Parse categories from PodcastIndex format"""
        categories = []
        for key, value in categories_dict.items():
            if isinstance(value, str):
                categories.append(value)
        return categories
    
    def enrich_podcasts(self, podcasts: List[Podcast]) -> List[Podcast]:
        """Enrich a list of podcasts with metadata"""
        total = len(podcasts)
        enriched_from_cache = 0
        enriched_from_api = 0
        failed_enrichments = 0
        
        logger.info(f"Enriching {total} podcasts with metadata")
        
        # Create progress bar for enrichment
        with tqdm(total=total, desc="Enriching podcasts", unit="podcast") as pbar:
            for idx, podcast in enumerate(podcasts, 1):
                # Update description less frequently to improve performance
                if idx % 5 == 0 or idx == 1:
                    pbar.set_description(f"Enriching: {podcast.name[:30]}... (cache:{enriched_from_cache} api:{enriched_from_api} failed:{failed_enrichments})")
                
                try:
                    # Check cache first
                    cache_key = self._get_cache_key(podcast.name)
                    
                    if cache_key in self.enrichment_cache:
                        metadata = self.enrichment_cache[cache_key]
                        enriched_from_cache += 1
                        logger.debug(f"Using cached enrichment for: {podcast.name}")
                    else:
                        metadata = self.search_podcast(podcast.name, podcast.creator)
                        if metadata:
                            # Save to cache
                            self.enrichment_cache[cache_key] = metadata
                            enriched_from_api += 1
                            # Save cache periodically
                            if enriched_from_api % 10 == 0:
                                self._save_enrichment_cache()
                        else:
                            logger.debug(f"No metadata found for: {podcast.name} by {podcast.creator or 'Unknown'}")
                    
                    if metadata:
                        podcast.description = metadata['description']
                        podcast.rss_url = metadata['rss_url']
                        podcast.creator = metadata['creator'] if not podcast.creator else podcast.creator
                        podcast.categories = metadata['categories']
                        podcast.episode_count = metadata['episode_count']
                        podcast.podcast_index_id = metadata['podcast_index_id']
                        podcast.last_updated = metadata['last_updated']
                        
                        # Update language if more specific
                        if metadata.get('language'):
                            podcast.language = metadata['language']
                
                except Exception as e:
                    failed_enrichments += 1
                    logger.warning(f"Failed to enrich podcast '{podcast.name}': {e}")
                    # Continue with next podcast instead of stopping
                
                pbar.update(1)
        
        # Final cache save
        self._save_enrichment_cache()
        
        logger.info(f"Enrichment complete - Cache: {enriched_from_cache}, API: {enriched_from_api}, Failed: {failed_enrichments}")
        return podcasts


class PodcastClassifier:
    """Classify podcasts using LLM"""
    
    def __init__(self, llm_server_url: str = None, output_dir: Path = None):
        """Initialize classifier with LLM server URL"""
        if llm_server_url is None:
            config = load_config()
            llm_server_url = config.get('processing', {}).get('llms', {}).get('server_url', 'http://10.0.0.4:8002')
        
        self.llm_server_url = llm_server_url
        self.session = requests.Session()
        self.output_dir = output_dir
        self.cache_dir = None
        self.classification_cache = {}
        if output_dir:
            self.cache_dir = output_dir / "cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_classification_cache()
    
    def _load_classification_cache(self):
        """Load cached classification data (persistent across months)"""
        cache_file = self.cache_dir / "classification_cache_persistent.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.classification_cache = json.load(f)
                logger.info(f"Loaded classification cache with {len(self.classification_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load classification cache: {e}")
                self.classification_cache = {}
    
    def _save_classification_cache(self):
        """Save classification cache to disk (persistent across months)"""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / "classification_cache_persistent.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.classification_cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved classification cache with {len(self.classification_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save classification cache: {e}")
    
    def _get_cache_key(self, podcast_name: str, podcast_type: str) -> str:
        """Generate unique cache key for classification (works across platforms/countries)"""
        # Include the classification criteria in the key
        type_hash = hashlib.md5(podcast_type.encode()).hexdigest()[:8]
        return f"{podcast_name.lower().strip()}|{type_hash}"
        
    def classify_podcast(self, podcast: Podcast, podcast_type: str) -> Tuple[bool, float, str]:
        """
        Classify a single podcast based on the specified type.
        
        Returns:
            Tuple of (is_relevant, confidence_score, reason)
        """
        
        # Build context from podcast metadata
        context = f"""
Podcast: {podcast.name}
Description: {podcast.description or 'No description available'}
Categories: {', '.join(podcast.categories) if podcast.categories else 'Unknown'}
Creator: {podcast.creator or 'Unknown'}
Country: {podcast.country_name}
Language: {podcast.language}
Platform: {podcast.platform} (Rank #{podcast.rank})
"""
        
        # Create classification prompt
        prompt = f"""You are a podcast classification expert. Analyze this podcast and determine if it matches the following criteria:

Criteria: {podcast_type}

{context}

Respond with a JSON object containing:
1. "is_relevant": true/false - whether this podcast matches the criteria
2. "confidence": 0.0 to 1.0 - your confidence in this classification
3. "reason": brief explanation of your decision (max 50 words)

Focus on:
- The podcast name and description
- The categories it belongs to
- The type of content it likely produces

Be strict - only mark as relevant if clearly matches the criteria.

JSON Response:"""
        
        try:
            # Prepare LLM request
            request_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a podcast classification expert. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": "qwen3:4b-instruct",
                "temperature": 0.1,
                "max_tokens": 150
            }
            
            # Call LLM server
            response = self.session.post(
                f"{self.llm_server_url}/llm-request",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            llm_response = result.get('response', '{}')
            
            # Extract JSON from response
            try:
                # Try to find JSON in the response
                json_match = re.search(r'\{[^}]+\}', llm_response)
                if json_match:
                    classification = json.loads(json_match.group())
                else:
                    classification = json.loads(llm_response)
                
                return (
                    classification.get('is_relevant', False),
                    classification.get('confidence', 0.0),
                    classification.get('reason', 'No reason provided')
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response for {podcast.name}: {llm_response}")
                return False, 0.0, "Failed to parse classification"
                
        except Exception as e:
            logger.error(f"Classification error for {podcast.name}: {e}")
            return False, 0.0, f"Classification error: {str(e)}"
    
    def classify_podcasts_batch(self, podcasts: List[Podcast], podcast_type: str, 
                                batch_size: int = 10) -> List[Podcast]:
        """Classify multiple podcasts in batches"""
        total = len(podcasts)
        classified_from_cache = 0
        classified_from_llm = 0
        
        logger.info(f"Classifying {total} podcasts for type: {podcast_type}")
        
        # Create progress bar for classification
        with tqdm(total=total, desc="Classifying podcasts", unit="podcast") as pbar:
            for idx, podcast in enumerate(podcasts, 1):
                # Update description less frequently to improve performance
                if idx % 5 == 0 or idx == 1:
                    pbar.set_description(f"Classifying: {podcast.name[:30]}... (cache:{classified_from_cache} llm:{classified_from_llm})")
                
                # Check cache first
                cache_key = self._get_cache_key(podcast.name, podcast_type)
                
                if cache_key in self.classification_cache:
                    cached_result = self.classification_cache[cache_key]
                    is_relevant = cached_result['is_relevant']
                    confidence = cached_result['confidence']
                    reason = cached_result['reason']
                    classified_from_cache += 1
                    logger.debug(f"Using cached classification for: {podcast.name}")
                else:
                    is_relevant, confidence, reason = self.classify_podcast(podcast, podcast_type)
                    
                    # Save to cache
                    self.classification_cache[cache_key] = {
                        'is_relevant': is_relevant,
                        'confidence': confidence,
                        'reason': reason
                    }
                    classified_from_llm += 1
                    
                    # Save cache periodically
                    if classified_from_llm % 10 == 0:
                        self._save_classification_cache()
                    
                    # Small delay to avoid overwhelming the server
                    time.sleep(0.5)
                
                podcast.is_relevant = is_relevant
                podcast.relevance_score = confidence
                podcast.relevance_reason = reason
                
                pbar.update(1)
        
        # Final cache save
        self._save_classification_cache()
        
        # Log summary
        relevant_count = sum(1 for p in podcasts if p.is_relevant)
        logger.info(f"Classification complete - Cache: {classified_from_cache}, LLM: {classified_from_llm}")
        logger.info(f"Relevant podcasts: {relevant_count}/{total}")
        
        return podcasts


class SourcesWriter:
    """Write podcasts to sources.csv format"""
    
    @staticmethod
    def write_sources_csv(podcasts: List[Podcast], output_path: Path, 
                         filter_relevant: bool = False):
        """Write podcasts to sources.csv file with classification column"""
        
        # Always write all podcasts, classification column shows relevance
        logger.info(f"Writing all {len(podcasts)} podcasts with classification data")
        
        # Group podcasts by name to combine platform/country info
        podcast_groups = {}
        for podcast in podcasts:
            name_key = podcast.name.lower().strip()
            if name_key not in podcast_groups:
                podcast_groups[name_key] = []
            podcast_groups[name_key].append(podcast)
        
        # Create final list with combined metadata
        unique_podcasts = []
        for name_key, group in podcast_groups.items():
            # Use the best-ranked podcast as the primary
            primary = min(group, key=lambda p: p.rank)
            
            # Combine platform and country info from all appearances
            platforms_countries = []
            for p in group:
                platforms_countries.append(f"{p.platform} {p.country_name} #{p.rank}")
            
            # Update primary podcast with combined info
            primary._combined_appearances = ', '.join(platforms_countries)
            unique_podcasts.append(primary)
        
        logger.info(f"After deduplication: {len(unique_podcasts)} unique podcasts")
        
        # Count relevant podcasts
        relevant_count = sum(1 for p in unique_podcasts if p.is_relevant)
        logger.info(f"Classified as relevant: {relevant_count}/{len(unique_podcasts)}")
        
        # Write to CSV with classification columns
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['channel_name', 'description', 'rss_url', 'language', 
                         'creator', 'video_count', 'is_political', 'confidence_score', 
                         'classification_reason', 'notes']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for podcast in unique_podcasts:
                # Use combined appearances if available
                if hasattr(podcast, '_combined_appearances'):
                    notes_parts = [podcast._combined_appearances]
                else:
                    notes_parts = [f"{podcast.platform} {podcast.country_name} Rank {podcast.rank}"]
                
                if podcast.categories:
                    notes_parts.append(f"Categories: {', '.join(podcast.categories[:3])}")
                
                writer.writerow({
                    'channel_name': podcast.name,
                    'description': (podcast.description or '')[:500],
                    'rss_url': podcast.rss_url or '',
                    'language': podcast.language,
                    'creator': podcast.creator or '',
                    'video_count': str(podcast.episode_count) if podcast.episode_count else '',
                    'is_political': 'Yes' if podcast.is_relevant else 'No',
                    'confidence_score': f"{podcast.relevance_score:.2f}" if podcast.relevance_score is not None else '',
                    'classification_reason': podcast.relevance_reason or '',
                    'notes': ' | '.join(notes_parts)
                })
        
        logger.info(f"Sources saved to {output_path}")
    
    @staticmethod
    def write_detailed_json(podcasts: List[Podcast], output_path: Path):
        """Write detailed podcast data to JSON for debugging/analysis"""
        output_path = output_path.with_suffix('.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(p) for p in podcasts],
                f,
                indent=2,
                ensure_ascii=False
            )
        
        logger.info(f"Detailed data saved to {output_path}")


async def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Identify and collect podcasts from various countries and platforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect political podcasts for Europe project
  %(prog)s --project Europe --countries "fr,de,gb,es,it,ua" --podcast-type "political, current affairs or news"
  
  # Collect all top podcasts from US and Canada
  %(prog)s --project NorthAmerica --countries "us,ca" --top-n 100 --no-filter
  
  # Collect tech podcasts from specific countries
  %(prog)s --project Tech --countries "us,gb,de" --podcast-type "technology, software, or startups" --platforms spotify
        """
    )
    
    parser.add_argument('--project', required=True,
                       help='Project name (e.g., Europe, NorthAmerica)')
    
    parser.add_argument('--countries', required=True,
                       help='Comma-separated country codes (e.g., "fr,de,gb,es,it,ua")')
    
    parser.add_argument('--podcast-type', 
                       default="political, current affairs or news",
                       help='Type of podcasts to identify (used for LLM classification)')
    
    parser.add_argument('--platforms', 
                       default="spotify,apple",
                       help='Comma-separated platforms to scrape (default: spotify,apple)')
    
    parser.add_argument('--top-n', type=int, default=200,
                       help='Number of top podcasts to collect per platform (default: 200)')
    
    parser.add_argument('--no-filter', action='store_true',
                       help='Save all podcasts without filtering by relevance')
    
    parser.add_argument('--skip-enrichment', action='store_true',
                       help='Skip PodcastIndex enrichment step')
    
    parser.add_argument('--skip-classification', action='store_true',
                       help='Skip LLM classification step')
    
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory (default: projects/{project})')
    
    parser.add_argument('--llm-server',
                       help='LLM server URL (default: from config)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Parse arguments
    country_codes = [c.strip().lower() for c in args.countries.split(',')]
    platforms = [p.strip().lower() for p in args.platforms.split(',')]
    
    # Validate country codes
    invalid_codes = [c for c in country_codes if c not in COUNTRY_CODES]
    if invalid_codes:
        logger.error(f"Invalid country codes: {', '.join(invalid_codes)}")
        logger.info(f"Valid codes: {', '.join(sorted(COUNTRY_CODES.keys()))}")
        return 1
    
    # Set output directory
    output_dir = args.output_dir or Path(f"projects/{args.project}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting podcast identification for project: {args.project}")
    logger.info(f"Countries: {', '.join(country_codes)}")
    logger.info(f"Platforms: {', '.join(platforms)}")
    logger.info(f"Podcast type: {args.podcast_type}")
    logger.info(f"Output directory: {output_dir}")
    
    all_podcasts = []
    
    # Step 1: Scrape charts
    logger.info("=" * 60)
    logger.info("Step 1: Scraping podcast charts")
    logger.info("=" * 60)
    
    async with PodstatusScraper(output_dir) as scraper:
        charts = await scraper.fetch_all_charts(country_codes, platforms, args.top_n)
        
        for country_code, podcasts in charts.items():
            all_podcasts.extend(podcasts)
            logger.info(f"{COUNTRY_CODES[country_code]['name']}: {len(podcasts)} podcasts")
    
    if not all_podcasts:
        logger.error("No podcasts collected")
        return 1
    
    logger.info(f"Total podcasts collected: {len(all_podcasts)}")
    
    # Deduplicate podcasts before enrichment/classification
    seen_names = {}
    unique_podcasts = []
    duplicate_count = 0
    
    for podcast in all_podcasts:
        name_key = podcast.name.lower().strip()
        if name_key not in seen_names:
            seen_names[name_key] = podcast
            unique_podcasts.append(podcast)
        else:
            duplicate_count += 1
            # Keep the higher-ranked one (lower rank number is better)
            existing = seen_names[name_key]
            if podcast.rank < existing.rank:
                # Replace with better-ranked version
                unique_podcasts.remove(existing)
                unique_podcasts.append(podcast)
                seen_names[name_key] = podcast
                logger.debug(f"Replaced {existing.platform} rank {existing.rank} with {podcast.platform} rank {podcast.rank} for: {podcast.name}")
    
    if duplicate_count > 0:
        logger.info(f"Deduplication: {len(all_podcasts)} → {len(unique_podcasts)} podcasts ({duplicate_count} duplicates removed)")
        all_podcasts = unique_podcasts
    else:
        logger.info(f"No duplicates found among {len(all_podcasts)} podcasts")
    
    # Step 2: Enrich with metadata
    if not args.skip_enrichment:
        logger.info("=" * 60)
        logger.info("Step 2: Enriching with PodcastIndex metadata")
        logger.info("=" * 60)
        
        enricher = PodcastEnricher(output_dir)
        all_podcasts = enricher.enrich_podcasts(all_podcasts)
    else:
        logger.info("Skipping enrichment step")
    
    # Step 3: Classify podcasts
    if not args.skip_classification:
        logger.info("=" * 60)
        logger.info("Step 3: Classifying podcasts with LLM")
        logger.info("=" * 60)
        
        classifier = PodcastClassifier(args.llm_server, output_dir)
        all_podcasts = classifier.classify_podcasts_batch(all_podcasts, args.podcast_type)
        
        # Log classification results
        relevant = [p for p in all_podcasts if p.is_relevant]
        logger.info(f"Classification results: {len(relevant)}/{len(all_podcasts)} marked as relevant")
    else:
        logger.info("Skipping classification step")
        # Mark all as relevant if skipping classification
        for podcast in all_podcasts:
            podcast.is_relevant = True
            podcast.relevance_score = 1.0
            podcast.relevance_reason = "Classification skipped"
    
    # Step 4: Write output
    logger.info("=" * 60)
    logger.info("Step 4: Writing output files")
    logger.info("=" * 60)
    
    writer = SourcesWriter()
    
    # Write sources.csv (always includes all podcasts with classification data)
    sources_path = output_dir / "sources.csv"
    writer.write_sources_csv(all_podcasts, sources_path)
    
    # Write detailed JSON
    json_path = output_dir / f"podcasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer.write_detailed_json(all_podcasts, json_path)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    
    if not args.skip_classification:
        relevant = [p for p in all_podcasts if p.is_relevant]
        by_country = {}
        for p in relevant:
            if p.country_code not in by_country:
                by_country[p.country_code] = 0
            by_country[p.country_code] += 1
        
        logger.info(f"Total relevant podcasts: {len(relevant)}")
        for country_code, count in sorted(by_country.items()):
            logger.info(f"  {COUNTRY_CODES[country_code]['name']}: {count}")
    else:
        logger.info(f"Total podcasts: {len(all_podcasts)}")
    
    logger.info(f"Output saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))