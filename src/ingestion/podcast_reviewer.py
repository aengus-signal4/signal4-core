#!/usr/bin/env python3
"""
Podcast Review and Merge Module (Phase 4)

DEPRECATED: This CLI-based reviewer is deprecated.
Use the Streamlit dashboard instead:

    streamlit run dashboards/podcast_review.py

The dashboard provides:
- Web-based UI for reviewing podcasts
- Multi-project assignment
- Rejection tracking (per-project)
- Writes to channel_projects table (not sources.csv)

This file is kept for reference only.
------------------------------------------------------------

Reviews classified political podcasts from the database and allows
interactive approval to merge them into a project's sources.csv file.
"""

import csv
import os
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.session import get_session
from src.database.models import PodcastMetadata, PodcastChart
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('podcast_reviewer')


def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def load_existing_sources(sources_file: Path) -> Tuple[List[Dict], Set[str], Set[str], Dict[str, Dict]]:
    """
    Load existing sources from CSV file.

    Returns:
        Tuple of (list of source dicts, set of RSS URLs, set of lowercase names, dict mapping RSS URL to source)
    """
    sources = []
    rss_urls = set()
    names = set()
    url_to_source = {}

    if not sources_file.exists():
        logger.warning(f"Sources file not found: {sources_file}")
        return sources, rss_urls, names, url_to_source

    with open(sources_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row)
            if row.get('podcast'):
                rss_url = row['podcast'].strip()
                rss_urls.add(rss_url)
                url_to_source[rss_url] = row
            if row.get('channel_name'):
                names.add(row['channel_name'].strip().lower())

    logger.info(f"Loaded {len(sources)} existing sources")
    return sources, rss_urls, names, url_to_source


def find_duplicate(
    podcast_name: str,
    rss_url: str,
    existing_urls: Set[str],
    existing_names: Set[str],
    similarity_threshold: float = 0.9
) -> Optional[str]:
    """
    Check if podcast is a duplicate based on URL or name similarity.

    Returns:
        Reason for duplicate, or None if not a duplicate
    """
    # Check exact URL match
    if rss_url and rss_url.strip() in existing_urls:
        return "Exact RSS URL match"

    # Check exact name match (case-insensitive)
    if podcast_name.strip().lower() in existing_names:
        return "Exact name match"

    # Check similar names
    podcast_lower = podcast_name.strip().lower()
    for existing_name in existing_names:
        ratio = similarity_ratio(podcast_lower, existing_name)
        if ratio >= similarity_threshold:
            return f"Similar name match ({ratio:.0%} similar to '{existing_name}')"

    return None


def get_best_chart_rank(monthly_rankings: dict, countries: Optional[List[str]] = None) -> Tuple[Optional[int], Optional[str]]:
    """
    Get the best (lowest) chart rank from monthly rankings, optionally filtered by countries.

    Args:
        monthly_rankings: Monthly rankings dictionary
        countries: Optional list of country codes to filter by

    Returns:
        Tuple of (best_rank, chart_info)
    """
    if not monthly_rankings:
        return None, None

    best_rank = float('inf')
    best_chart = None

    for month, charts in monthly_rankings.items():
        for chart_key, rank in charts.items():
            # Filter by country if specified
            if countries:
                # Chart key format: platform_country_category
                parts = chart_key.split('_')
                if len(parts) >= 2:
                    chart_country = parts[1].lower()
                    if chart_country not in countries:
                        continue

            if rank < best_rank:
                best_rank = rank
                best_chart = chart_key

    if best_rank == float('inf'):
        return None, None

    return int(best_rank), best_chart


def format_chart_info(chart_key: str) -> str:
    """Format chart key into readable string"""
    parts = chart_key.split('_')
    if len(parts) >= 3:
        platform = parts[0].capitalize()
        country = parts[1].upper()
        category = ' '.join(parts[2:]).title()
        return f"{platform} {country} {category}"
    return chart_key


def save_sources_csv(sources_file: Path, sources: List[Dict]):
    """Save sources to CSV file"""
    # Ensure directory exists
    sources_file.parent.mkdir(parents=True, exist_ok=True)

    with open(sources_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['channel_name', 'description', 'podcast', 'language',
                     'author', 'category', 'added', 'reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sources)


def review_and_merge_podcasts(
    project_dir: Path,
    countries: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Dict[str, int]:
    """
    Review classified political podcasts and merge approved ones into sources.csv.

    Args:
        project_dir: Project directory (e.g., projects/Big_Channels)
        countries: Filter by country codes (e.g., ["us"])
        limit: Maximum number of podcasts to review

    Returns:
        Statistics dictionary
    """
    stats = {
        'reviewed': 0,
        'approved': 0,
        'skipped': 0,
        'duplicates': 0
    }

    sources_file = project_dir / "sources.csv"

    # Load existing sources
    existing_sources, existing_urls, existing_names, url_to_source = load_existing_sources(sources_file)

    # Track if we need to update existing sources with missing descriptions
    sources_updated = False

    logger.info("="*70)
    logger.info("REVIEWING CLASSIFIED PODCASTS")
    logger.info("="*70)
    if countries:
        logger.info(f"Filtering by countries: {countries}")
        logger.info(f"Countries type: {type(countries)}")
    if limit:
        logger.info(f"Limiting to {limit} podcasts")

    # Query database for relevant political podcasts
    with get_session() as session:
        query = session.query(
            PodcastMetadata.podcast_name,
            PodcastMetadata.description,
            PodcastMetadata.rss_url,
            PodcastMetadata.language,
            PodcastMetadata.creator,
            PodcastMetadata.categories,
            PodcastMetadata.monthly_rankings,
            PodcastMetadata.meta_data
        ).filter(
            PodcastMetadata.meta_data['classification']['is_relevant'].astext == 'true'
        )

        # Apply country filter if specified
        if countries:
            # We need to filter based on monthly_rankings content, not just chart existence
            # First get all relevant podcasts, then filter in Python
            pass

        podcasts = query.all()

    if not podcasts:
        logger.warning("No relevant podcasts found")
        return stats

    logger.info(f"Found {len(podcasts)} relevant podcasts")

    # First pass: Update existing sources with missing descriptions and normalize language codes
    logger.info("Checking for existing sources with missing descriptions...")
    for podcast in podcasts:
        if podcast.rss_url and podcast.rss_url in url_to_source:
            existing = url_to_source[podcast.rss_url]

            # Update description if missing
            if not existing.get('description') or existing['description'].strip() == '':
                if podcast.description:
                    logger.info(f"Updating description for: {podcast.podcast_name}")
                    existing['description'] = podcast.description
                    sources_updated = True

            # Normalize language code if needed (remove regional variants)
            if existing.get('language') and '-' in existing['language']:
                old_lang = existing['language']
                existing['language'] = old_lang.split('-')[0].lower()
                logger.info(f"Normalizing language code for {podcast.podcast_name}: {old_lang} → {existing['language']}")
                sources_updated = True

    # Save if we updated any descriptions or language codes
    if sources_updated:
        save_sources_csv(sources_file, existing_sources)
        logger.info(f"✓ Updated sources.csv with descriptions and normalized language codes")

    # Sort by best chart rank (ascending), filtered by countries
    podcasts_with_ranks = []
    for podcast in podcasts:
        best_rank, chart_info = get_best_chart_rank(podcast.monthly_rankings, countries)
        # Skip podcasts that don't have any ranks in the specified countries
        if countries and (best_rank is None or best_rank == 999999):
            continue
        podcasts_with_ranks.append((podcast, best_rank or 999999, chart_info))

    podcasts_with_ranks.sort(key=lambda x: x[1])

    # Apply limit if specified
    if limit:
        podcasts_with_ranks = podcasts_with_ranks[:limit]

    logger.info("\n" + "="*70)
    logger.info("INTERACTIVE REVIEW")
    logger.info("="*70)
    logger.info("Commands: y=approve, n=skip, q=quit\n")

    # Review each podcast
    for podcast, best_rank, chart_info in podcasts_with_ranks:
        stats['reviewed'] += 1

        print("\n" + "="*70)
        print(f"Podcast #{stats['reviewed']} of {len(podcasts_with_ranks)}")
        print("="*70)
        print(f"Name: {podcast.podcast_name}")
        print(f"Creator: {podcast.creator or 'Unknown'}")
        print(f"Language: {podcast.language or 'Unknown'}")
        print(f"RSS URL: {podcast.rss_url or 'Unknown'}")

        if best_rank and best_rank < 999999:
            print(f"Best Rank: #{best_rank}")
            if chart_info:
                print(f"Chart: {format_chart_info(chart_info)}")

        if podcast.categories:
            print(f"Categories: {', '.join(podcast.categories)}")

        # Show classification info
        classification = podcast.meta_data.get('classification', {})
        confidence = classification.get('confidence', 0)
        reason = classification.get('reason', '')
        print(f"\nClassification Confidence: {confidence:.2f}")
        print(f"Reason: {reason}")

        # Show description (truncated to 800 chars)
        if podcast.description:
            desc = podcast.description[:800]
            if len(podcast.description) > 800:
                desc += "..."
            print(f"\nDescription: {desc}")

        # Check for duplicates
        duplicate_reason = find_duplicate(
            podcast.podcast_name,
            podcast.rss_url or '',
            existing_urls,
            existing_names
        )

        if duplicate_reason:
            print(f"\n⚠️  DUPLICATE DETECTED: {duplicate_reason}")
            stats['duplicates'] += 1
            print("Skipping...")
            stats['skipped'] += 1
            continue

        # Ask for approval
        print("\n" + "-"*70)
        while True:
            response = input("Add to sources.csv? [y/n/q]: ").strip().lower()

            if response == 'q':
                print("\nReview cancelled by user")
                logger.info("Review cancelled by user")
                # Save before quitting if we approved any
                if stats['approved'] > 0:
                    save_sources_csv(sources_file, existing_sources)
                return stats

            if response in ['y', 'n']:
                break

            print("Invalid input. Please enter 'y', 'n', or 'q'")

        if response == 'y':
            # Get today's date in YYYY-MM-DD format
            today = datetime.now().strftime('%Y-%m-%d')

            # Build full reason with classification details
            full_reason = f"{reason} [Confidence: {confidence:.2f}, Classification: {classification.get('type', 'unknown')}]"

            # Normalize language code (remove regional variants like en-us, en-gb)
            language = podcast.language or ''
            if language and '-' in language:
                language = language.split('-')[0].lower()

            # Add to sources
            new_source = {
                'channel_name': podcast.podcast_name,
                'description': podcast.description or '',
                'podcast': podcast.rss_url or '',
                'language': language,
                'author': podcast.creator or '',
                'category': ', '.join(podcast.categories) if podcast.categories else '',
                'added': today,
                'reason': full_reason
            }

            existing_sources.append(new_source)
            existing_urls.add(podcast.rss_url or '')
            existing_names.add(podcast.podcast_name.lower())

            stats['approved'] += 1

            # Save immediately after each approval
            save_sources_csv(sources_file, existing_sources)
            print("✓ Added to sources and saved")
        else:
            stats['skipped'] += 1
            print("✗ Skipped")

    # Final confirmation
    if stats['approved'] > 0:
        logger.info(f"✓ All changes saved to {sources_file}")

    return stats
