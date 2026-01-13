#!/usr/bin/env python3
"""
Podcast Chart Collection Module

Collects top podcast charts from Podstatus.com across multiple countries,
platforms, and categories. Designed to be run monthly to capture rankings.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from src.database.models import Channel, PodcastChart

logger = setup_worker_logger('chart_collector')


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
    """Podcast information from charts"""
    rank: int
    name: str
    platform: str
    country_code: str
    country_name: str
    category: str
    creator: Optional[str] = None


class ChartCollector:
    """Scrape podcast charts from podstatus.com"""

    BASE_URL = "https://podstatus.com/charts"
    PLATFORMS = ["spotify", "apple"]

    # Categories we're interested in (subset of all available)
    # If not specified, will try to discover from platform pages
    DEFAULT_CATEGORIES = [
        "all-podcasts",      # Apple default
        "top-podcasts",      # Spotify default
        "news-commentary",
        "news",
        "politics",
        "history",
        "society-culture",
        # Health categories
        "health-fitness",
        "alternative-health",
        "mental-health",
        # Business/Finance categories
        "business",
        "business-news",
    ]

    # URL patterns for different platforms
    PLATFORM_PATTERNS = {
        "spotify": "spotify/{country}/{category}",
        "apple": "applepodcasts/{country}/{category}"
    }

    # Platform index pages for discovery
    PLATFORM_INDEX = {
        "spotify": "spotify",
        "apple": "applepodcasts"
    }

    def __init__(self, month: str, delay_range: tuple = (6, 8), output_dir: str = None):
        """
        Initialize chart collector.

        Args:
            month: Month identifier (e.g., "2025-10")
            delay_range: (min, max) seconds to delay between requests
            output_dir: Directory to save CSV files (e.g., "projects/podcast_charts/2025-11")
        """
        self.month = month
        self.delay_min, self.delay_max = delay_range
        self.session = None

        # Setup output directory for CSV files
        if output_dir:
            self.output_dir = Path(output_dir) / "raw_charts"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

        logger.info(f"Initialized ChartCollector for {month}")
        logger.info(f"Storing charts in PostgreSQL database")
        if self.output_dir:
            logger.info(f"Saving CSV files to: {self.output_dir}")

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

    def _check_existing_chart(self, platform: str, country_code: str, category: str) -> bool:
        """Check if chart already exists in database AND CSV file for this month"""
        chart_key = PodcastChart.get_chart_key(platform, country_code, category)

        # Check database
        db_exists = False
        try:
            with get_session() as session:
                count = session.query(PodcastChart).filter_by(
                    month=self.month,
                    chart_key=chart_key
                ).count()
                db_exists = count > 0
        except Exception as e:
            logger.error(f"Error checking existing chart in DB {chart_key}: {e}")

        # Check CSV file
        csv_exists = False
        if self.output_dir:
            filename = f"{platform}_{country_code}_{category}.csv"
            filepath = self.output_dir / filename
            csv_exists = filepath.exists()

        # Only skip if both exist
        return db_exists and csv_exists

    async def discover_countries(self, platform: str) -> List[str]:
        """
        Discover available countries for a platform by scraping the index page.

        Args:
            platform: Platform name (spotify, apple)

        Returns:
            List of country codes
        """
        if platform not in self.PLATFORM_INDEX:
            logger.warning(f"Unknown platform: {platform}")
            return []

        url = f"{self.BASE_URL}/{self.PLATFORM_INDEX[platform]}"

        try:
            logger.info(f"Discovering countries for {platform} from {url}")
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch platform index: HTTP {response.status}")
                    return []

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Find all country links (they're in the format /charts/{platform}/{country})
                countries = set()
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if f'/charts/{self.PLATFORM_INDEX[platform]}/' in href:
                        # Extract country code from URL (handle both absolute and relative URLs)
                        # URL can be: /charts/spotify/us OR https://podstatus.com/charts/spotify/us
                        parts = href.split('/')
                        # Find the position of 'charts' in the path
                        try:
                            charts_idx = parts.index('charts')
                            # Platform should be at charts_idx + 1, country at charts_idx + 2
                            if len(parts) > charts_idx + 2:
                                country_code = parts[charts_idx + 2]
                                # Only include if it looks like a country code (2-3 letters)
                                if len(country_code) <= 3 and country_code.isalpha():
                                    countries.add(country_code.lower())
                        except (ValueError, IndexError):
                            continue

                countries_list = sorted(list(countries))
                logger.info(f"Discovered {len(countries_list)} countries for {platform}")

                # Polite delay after discovery request
                await asyncio.sleep(2)

                return countries_list

        except Exception as e:
            logger.error(f"Error discovering countries for {platform}: {e}")
            return []

    async def discover_categories(self, platform: str, country_code: str) -> List[str]:
        """
        Discover available categories for a platform/country by scraping the country page.

        Args:
            platform: Platform name (spotify, apple)
            country_code: Country code

        Returns:
            List of category slugs
        """
        url = f"{self.BASE_URL}/{self.PLATFORM_INDEX[platform]}/{country_code}"

        try:
            logger.debug(f"Discovering categories for {platform} {country_code}")
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch country page: HTTP {response.status}")
                    return []

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Find all category links
                categories = set()
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if f'/charts/{self.PLATFORM_INDEX[platform]}/{country_code}/' in href:
                        # Extract category from URL (handle both absolute and relative URLs)
                        parts = href.split('/')
                        # Find the position of 'charts' in the path
                        try:
                            charts_idx = parts.index('charts')
                            # Platform at charts_idx + 1, country at charts_idx + 2, category at charts_idx + 3
                            if len(parts) > charts_idx + 3:
                                category = parts[charts_idx + 3]
                                if category:  # Non-empty category
                                    categories.add(category.lower())
                        except (ValueError, IndexError):
                            continue

                categories_list = sorted(list(categories))
                logger.debug(f"Discovered {len(categories_list)} categories for {platform} {country_code}")

                # Polite delay after category discovery request
                await asyncio.sleep(1)

                return categories_list

        except Exception as e:
            logger.warning(f"Error discovering categories for {platform} {country_code}: {e}")
            return []

    def _save_chart_to_db(self, podcasts: List[Podcast], platform: str, country_code: str, category: str):
        """Save chart data to PostgreSQL database using channels table"""
        if not podcasts:
            return

        chart_key = PodcastChart.get_chart_key(platform, country_code, category)

        try:
            with get_session() as session:
                # Process each podcast in the chart
                for podcast in podcasts:
                    channel_key = Channel.get_channel_key(podcast.name)

                    # Find or create channel (podcast)
                    channel = session.query(Channel).filter_by(
                        channel_key=channel_key,
                        platform='podcast'
                    ).first()

                    if not channel:
                        # Create new channel (will be enriched later)
                        channel = Channel(
                            channel_key=channel_key,
                            display_name=podcast.name,
                            platform='podcast',
                            primary_url='',  # Will be filled during enrichment
                            status='discovered',
                            platform_metadata={
                                'creator': podcast.creator or '',
                                'first_seen': datetime.utcnow().isoformat(),
                                'monthly_rankings': {}
                            },
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        session.add(channel)
                        session.flush()  # Get the ID

                    # Check if chart entry already exists
                    existing_chart = session.query(PodcastChart).filter_by(
                        channel_id=channel.id,
                        month=self.month,
                        chart_key=chart_key
                    ).first()

                    if existing_chart:
                        # Update rank if changed
                        if existing_chart.rank != podcast.rank:
                            existing_chart.rank = podcast.rank
                            existing_chart.collected_at = datetime.utcnow()
                    else:
                        # Create new chart entry
                        chart_entry = PodcastChart(
                            channel_id=channel.id,
                            month=self.month,
                            platform=platform.lower(),
                            country=country_code.lower(),
                            category=category.lower(),
                            rank=podcast.rank,
                            chart_key=chart_key,
                            collected_at=datetime.utcnow()
                        )
                        session.add(chart_entry)

                session.commit()
                logger.debug(f"Saved {len(podcasts)} podcasts to database for {chart_key}")

        except Exception as e:
            logger.error(f"Failed to save chart to database {chart_key}: {e}", exc_info=True)

    def _save_chart_to_csv(self, podcasts: List[Podcast], platform: str, country_code: str, category: str):
        """Save chart data to local CSV file"""
        if not podcasts or not self.output_dir:
            return

        try:
            import csv

            # Create filename: platform_country_category.csv
            filename = f"{platform}_{country_code}_{category}.csv"
            filepath = self.output_dir / filename

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['rank', 'name', 'creator', 'platform', 'country', 'category'])

                # Write podcasts
                for podcast in podcasts:
                    writer.writerow([
                        podcast.rank,
                        podcast.name,
                        podcast.creator or '',
                        podcast.platform,
                        podcast.country_name,
                        podcast.category
                    ])

            logger.debug(f"Saved {len(podcasts)} podcasts to CSV: {filename}")

        except Exception as e:
            logger.error(f"Failed to save chart to CSV {platform}_{country_code}_{category}: {e}")

    async def fetch_chart(self, platform: str, country_code: str, category: str,
                         top_n: int = 200) -> Optional[List[Podcast]]:
        """
        Fetch and parse a single chart.

        Returns:
            List of Podcast objects, or None if chart not available
        """
        # Build URL based on platform pattern
        if platform in self.PLATFORM_PATTERNS:
            url_pattern = self.PLATFORM_PATTERNS[platform]
            url = f"{self.BASE_URL}/{url_pattern.format(country=country_code, category=category)}"
        else:
            logger.warning(f"Unknown platform: {platform}")
            return None

        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 404:
                    logger.debug(f"Chart not available: {platform} × {country_code} × {category}")
                    return None

                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None

                html = await response.text()
                podcasts = self.parse_chart_html(html, platform, country_code, category, top_n)

                if not podcasts:
                    logger.debug(f"No podcasts found: {platform} × {country_code} × {category}")
                    return None

                return podcasts

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching: {platform} × {country_code} × {category}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {platform} × {country_code} × {category}: {e}")
            return None

    def parse_chart_html(self, html: str, platform: str, country_code: str,
                        category: str, top_n: int) -> List[Podcast]:
        """Parse Podstatus HTML to extract podcast names"""
        podcasts = []

        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Podstatus uses h4 tags with class "fw-normal mb-1" for podcast titles
            title_elements = soup.find_all('h4', class_='fw-normal mb-1')

            if not title_elements:
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
                            category=category,
                            creator=creator
                        )
                        podcasts.append(podcast)

                except Exception as e:
                    logger.warning(f"Failed to parse podcast {idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to parse HTML for {platform} {country_code} {category}: {e}")

        return podcasts

    async def collect_all_charts(
        self,
        countries: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        top_n: int = 200,
        skip_existing: bool = True,
        discover_mode: bool = True
    ) -> Dict[str, int]:
        """
        Collect charts for multiple countries, platforms, and categories.

        Args:
            countries: List of country codes (None = discover all)
            platforms: List of platforms (default: all)
            categories: List of categories (None = discover per country)
            top_n: Number of top podcasts to collect per chart
            skip_existing: Skip charts that already exist for this month
            discover_mode: If True, intelligently discover countries/categories

        Returns:
            Statistics dictionary
        """
        if platforms is None:
            platforms = self.PLATFORMS

        # Statistics
        stats = {
            'total_attempts': 0,
            'successful': 0,
            'not_available': 0,
            'failed': 0,
            'skipped_existing': 0,
            'total_podcasts': 0,
            'discovery_requests': 0
        }

        logger.info(f"Starting chart collection")
        logger.info(f"  Discovery mode: {discover_mode}")
        logger.info(f"  Platforms: {len(platforms)}")
        logger.info(f"  Delay range: {self.delay_min}-{self.delay_max} seconds")

        # Build collection plan (platform -> country -> categories)
        collection_plan = {}

        for platform in platforms:
            # Discover or use provided countries
            if discover_mode and countries is None:
                platform_countries = await self.discover_countries(platform)
                stats['discovery_requests'] += 1
                await asyncio.sleep(2)  # Be polite after discovery
            elif countries:
                platform_countries = countries
            else:
                logger.error("Must specify countries or enable discover_mode")
                return stats

            logger.info(f"{platform}: {len(platform_countries)} countries")

            collection_plan[platform] = {}

            for country_code in platform_countries:
                # Discover or use provided categories
                if discover_mode and categories is None:
                    # Apple has same categories for all countries, so discover once
                    # Spotify varies by country, so discover for each country
                    if platform == 'apple' and not collection_plan[platform]:
                        # First Apple country - discover categories
                        country_categories = await self.discover_categories(platform, country_code)
                        stats['discovery_requests'] += 1
                        # Filter to only our interested categories
                        country_categories = [c for c in country_categories if c in self.DEFAULT_CATEGORIES]
                        await asyncio.sleep(1)  # Be polite after discovery
                    elif platform == 'apple':
                        # Reuse categories from first Apple country
                        first_country = list(collection_plan[platform].keys())[0]
                        country_categories = collection_plan[platform][first_country]
                    elif platform == 'spotify':
                        # Spotify varies by country - discover each time
                        country_categories = await self.discover_categories(platform, country_code)
                        stats['discovery_requests'] += 1
                        # Filter to only our interested categories
                        country_categories = [c for c in country_categories if c in self.DEFAULT_CATEGORIES]
                        await asyncio.sleep(1)  # Be polite after discovery
                    else:
                        country_categories = self.DEFAULT_CATEGORIES
                elif categories:
                    country_categories = categories
                else:
                    country_categories = self.DEFAULT_CATEGORIES

                if country_categories:
                    collection_plan[platform][country_code] = country_categories

        # Calculate total charts to collect
        total_charts = sum(
            len(categories)
            for platform_data in collection_plan.values()
            for categories in platform_data.values()
        )
        stats['total_attempts'] = total_charts

        logger.info(f"Collection plan: {total_charts} charts to collect")
        logger.info(f"  Discovery requests: {stats['discovery_requests']}")

        # Collect charts with progress bar
        with tqdm(total=total_charts, desc="Collecting charts", unit="chart") as pbar:
            chart_num = 0

            for platform, countries_data in collection_plan.items():
                for country_code, categories_list in countries_data.items():
                    for category in categories_list:
                        chart_num += 1
                        pbar.set_description(f"Collecting {platform} {country_code} {category}")

                        # Check if already exists
                        if skip_existing and self._check_existing_chart(platform, country_code, category):
                            logger.debug(f"Skipping existing: {platform} × {country_code} × {category}")
                            stats['skipped_existing'] += 1
                            pbar.update(1)
                            continue

                        # Fetch chart
                        podcasts = await self.fetch_chart(platform, country_code, category, top_n)

                        if podcasts is None:
                            stats['not_available'] += 1
                        elif len(podcasts) == 0:
                            stats['failed'] += 1
                        else:
                            stats['successful'] += 1
                            stats['total_podcasts'] += len(podcasts)
                            self._save_chart_to_db(podcasts, platform, country_code, category)
                            self._save_chart_to_csv(podcasts, platform, country_code, category)

                        pbar.update(1)

                        # Delay before next request (be polite)
                        if chart_num < total_charts:
                            # Vary delay slightly to be more human-like
                            import random
                            delay = random.uniform(self.delay_min, self.delay_max)
                            await asyncio.sleep(delay)

        # Log summary
        logger.info("="*60)
        logger.info("Chart Collection Summary")
        logger.info("="*60)
        logger.info(f"Total attempts: {stats['total_attempts']}")
        logger.info(f"Successful: {stats['successful']}")
        logger.info(f"Not available (404): {stats['not_available']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped (existing): {stats['skipped_existing']}")
        logger.info(f"Total podcasts collected: {stats['total_podcasts']}")
        logger.info("="*60)

        return stats


async def main_async(
    month: str,
    countries: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    top_n: int = 200,
    delay_range: tuple = (6, 8),
    skip_existing: bool = True,
    discover_mode: bool = True,
    output_dir: Optional[str] = None
):
    """Async main function for chart collection"""
    async with ChartCollector(month, delay_range, output_dir) as collector:
        stats = await collector.collect_all_charts(
            countries=countries,
            platforms=platforms,
            categories=categories,
            top_n=top_n,
            skip_existing=skip_existing,
            discover_mode=discover_mode
        )
        return stats


def collect_charts(
    month: str,
    countries: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    top_n: int = 200,
    delay_range: tuple = (6, 8),
    skip_existing: bool = True,
    discover_mode: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, int]:
    """
    Synchronous wrapper for chart collection.

    Args:
        month: Month identifier (e.g., "2025-10")
        countries: List of country codes (None = discover all)
        platforms: List of platforms (default: ["spotify", "apple"])
        categories: List of categories (None = discover per country)
        top_n: Number of top podcasts per chart
        delay_range: (min, max) seconds between requests
        skip_existing: Skip charts that already exist
        discover_mode: If True, intelligently discover countries/categories
        output_dir: Directory to save CSV files (e.g., "projects/podcast_charts/2025-11")

    Returns:
        Statistics dictionary
    """
    return asyncio.run(main_async(
        month=month,
        countries=countries,
        platforms=platforms,
        categories=categories,
        top_n=top_n,
        delay_range=delay_range,
        skip_existing=skip_existing,
        discover_mode=discover_mode,
        output_dir=output_dir
    ))
