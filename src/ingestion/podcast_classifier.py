#!/usr/bin/env python3
"""
Podcast Classification Module (Phase 3)

Classifies podcasts from the database using LLM based on project-specific criteria.
Stores classification results in channels.platform_metadata JSONB field.
"""

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.session import get_session
from src.database.models import Channel, PodcastChart
from src.utils.config import load_config
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('podcast_classifier')


class PodcastClassifier:
    """Classify podcasts using LLM"""

    def __init__(self, project_dir: Path, llm_server_url: str = None):
        """
        Initialize classifier.

        Args:
            project_dir: Project directory for cache (e.g., projects/UnitedStates)
            llm_server_url: LLM server URL (default from config)
        """
        if llm_server_url is None:
            config = load_config()
            llm_server_url = config.get('processing', {}).get('llms', {}).get('server_url', 'http://10.0.0.4:8002')

        self.llm_server_url = llm_server_url
        self.session = requests.Session()
        self.project_dir = project_dir
        self.cache_dir = project_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.classification_cache = {}
        self._load_classification_cache()

        logger.info(f"Initialized PodcastClassifier for {project_dir}")
        logger.info(f"LLM Server: {llm_server_url}")

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
        cache_file = self.cache_dir / "classification_cache_persistent.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.classification_cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved classification cache with {len(self.classification_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save classification cache: {e}")

    def _get_cache_key(self, podcast_name: str, classification_type: str) -> str:
        """Generate unique cache key for classification"""
        type_hash = hashlib.md5(classification_type.encode()).hexdigest()[:8]
        return f"{podcast_name.lower().strip()}|{type_hash}"

    def classify_podcast(
        self,
        podcast_name: str,
        description: str,
        creator: str,
        categories: List[str],
        classification_type: str
    ) -> Tuple[bool, float, str]:
        """
        Classify a single podcast based on the specified type.

        Args:
            podcast_name: Name of the podcast
            description: Podcast description
            creator: Podcast creator/author
            categories: List of podcast categories
            classification_type: What to classify for (e.g., "political, current affairs or news")

        Returns:
            Tuple of (is_relevant, confidence_score, reason)
        """

        # Build context from podcast metadata
        context = f"""
Podcast: {podcast_name}
Description: {description or 'No description available'}
Categories: {', '.join(categories) if categories else 'Unknown'}
Creator: {creator or 'Unknown'}
"""

        # Create classification prompt
        prompt = f"""You are a podcast classification expert. Analyze this podcast and determine if it matches the following criteria:

Criteria: {classification_type}

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
                # Try to parse the full response first
                try:
                    classification = json.loads(llm_response)
                except json.JSONDecodeError:
                    # Try to find JSON object in the response (handles markdown code blocks, etc.)
                    # Use a more robust regex that handles nested content
                    json_match = re.search(r'\{[\s\S]*?"is_relevant"[\s\S]*?\}', llm_response)
                    if json_match:
                        # Clean up common issues
                        json_str = json_match.group()
                        # Fix unescaped quotes in reason field by finding and fixing them
                        classification = json.loads(json_str)
                    else:
                        raise json.JSONDecodeError("No JSON found", llm_response, 0)

                return (
                    classification.get('is_relevant', False),
                    float(classification.get('confidence', 0.0)),
                    classification.get('reason', 'No reason provided')
                )
            except json.JSONDecodeError:
                # Last resort: try to extract values using regex
                is_relevant = '"is_relevant": true' in llm_response.lower() or '"is_relevant":true' in llm_response.lower()
                conf_match = re.search(r'"confidence":\s*([\d.]+)', llm_response)
                confidence = float(conf_match.group(1)) if conf_match else 0.5
                reason_match = re.search(r'"reason":\s*"([^"]*(?:"[^"]*"[^"]*)*)"', llm_response)
                reason = reason_match.group(1) if reason_match else "Parsed from malformed response"

                logger.debug(f"Fallback parsing for {podcast_name}: relevant={is_relevant}")
                return is_relevant, confidence, reason

        except Exception as e:
            logger.error(f"Classification error for {podcast_name}: {e}")
            return False, 0.0, f"Classification error: {str(e)}"

    def classify_all_podcasts(
        self,
        classification_type: str,
        require_description: bool = True
    ) -> Dict[str, int]:
        """
        Classify ALL podcasts in the database (not filtered by month/charts).

        Args:
            classification_type: What to classify for
            require_description: Only classify podcasts with descriptions (default: True)

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_podcasts': 0,
            'classified_from_cache': 0,
            'classified_from_llm': 0,
            'relevant_podcasts': 0,
            'updated_in_db': 0
        }

        logger.info("="*60)
        logger.info("Starting Podcast Classification (ALL PODCASTS)")
        logger.info("="*60)
        logger.info(f"Classification Type: {classification_type}")

        with get_session() as session:
            # Query all podcast channels (optionally requiring description)
            query = session.query(
                Channel.id,
                Channel.display_name,
                Channel.description,
                Channel.platform_metadata
            ).filter(
                Channel.platform == 'podcast'
            )

            if require_description:
                query = query.filter(
                    Channel.description.isnot(None),
                    Channel.description != ''
                )

            podcasts = query.all()
            stats['total_podcasts'] = len(podcasts)

            logger.info(f"Found {stats['total_podcasts']} podcasts to classify")

            if not podcasts:
                logger.warning("No podcasts found")
                return stats

            # Classify each podcast
            with tqdm(total=len(podcasts), desc="Classifying", unit="podcast") as pbar:
                for channel_id, display_name, description, platform_metadata in podcasts:
                    pbar.set_description(f"Classifying: {display_name[:30]}...")

                    # Extract metadata fields
                    platform_metadata = platform_metadata or {}
                    creator = platform_metadata.get('creator', '')
                    categories = platform_metadata.get('categories', [])

                    # Layer 1: Check database platform_metadata first (most reliable)
                    existing_classification = platform_metadata.get('classification', {})
                    if (existing_classification.get('type') == classification_type and
                        existing_classification.get('is_relevant') is not None):
                        # Already classified with this criteria - skip entirely
                        is_relevant = existing_classification['is_relevant']
                        confidence = existing_classification['confidence']
                        reason = existing_classification['reason']
                        stats['classified_from_cache'] += 1
                        stats['updated_in_db'] += 1  # Already in DB
                        logger.debug(f"Using database classification for: {podcast_name}")

                        if is_relevant:
                            stats['relevant_podcasts'] += 1

                        pbar.update(1)
                        continue

                    # Layer 2: Check persistent cache
                    cache_key = self._get_cache_key(display_name, classification_type)

                    if cache_key in self.classification_cache:
                        cached_result = self.classification_cache[cache_key]
                        is_relevant = cached_result['is_relevant']
                        confidence = cached_result['confidence']
                        reason = cached_result['reason']
                        stats['classified_from_cache'] += 1
                        logger.debug(f"Using cached classification for: {display_name}")
                    else:
                        # Layer 3: Classify using LLM (only if not in DB or cache)
                        is_relevant, confidence, reason = self.classify_podcast(
                            podcast_name=display_name,
                            description=description or '',
                            creator=creator,
                            categories=categories,
                            classification_type=classification_type
                        )

                        # Save to cache
                        self.classification_cache[cache_key] = {
                            'is_relevant': is_relevant,
                            'confidence': confidence,
                            'reason': reason
                        }
                        stats['classified_from_llm'] += 1

                        # Save cache periodically
                        if stats['classified_from_llm'] % 10 == 0:
                            self._save_classification_cache()

                        # Small delay to avoid overwhelming the server
                        time.sleep(0.5)

                    # Update platform_metadata in database
                    channel = session.query(Channel).filter_by(id=channel_id).first()
                    if channel:
                        # Get existing platform_metadata or create new
                        pm = channel.platform_metadata or {}

                        # Store classification result
                        pm['classification'] = {
                            'type': classification_type,
                            'is_relevant': is_relevant,
                            'confidence': confidence,
                            'reason': reason,
                            'classified_at': 'all_podcasts'
                        }

                        channel.platform_metadata = pm
                        channel.updated_at = datetime.utcnow()
                        session.commit()
                        stats['updated_in_db'] += 1

                        if is_relevant:
                            stats['relevant_podcasts'] += 1

                    pbar.update(1)

            # Final cache save
            self._save_classification_cache()

        # Log summary
        logger.info("="*60)
        logger.info("Classification Summary")
        logger.info("="*60)
        logger.info(f"Total podcasts: {stats['total_podcasts']}")
        logger.info(f"Classified from cache: {stats['classified_from_cache']}")
        logger.info(f"Classified from LLM: {stats['classified_from_llm']}")
        logger.info(f"Relevant podcasts: {stats['relevant_podcasts']}")
        logger.info(f"Updated in database: {stats['updated_in_db']}")
        logger.info("="*60)

        return stats

    def classify_monthly_podcasts(
        self,
        month: str,
        classification_type: str,
        countries: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_rank: Optional[int] = None,
        max_rank: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Classify podcasts from monthly charts and update database.

        Args:
            month: Month identifier (e.g., "2025-10")
            classification_type: What to classify for
            countries: Filter by country codes (e.g., ["us", "ca"])
            platforms: Filter by platforms (e.g., ["spotify", "apple"])
            categories: Filter by categories (e.g., ["news", "politics"])
            min_rank: Minimum rank (inclusive)
            max_rank: Maximum rank (inclusive)

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_podcasts': 0,
            'classified_from_cache': 0,
            'classified_from_llm': 0,
            'relevant_podcasts': 0,
            'updated_in_db': 0
        }

        logger.info("="*60)
        logger.info("Starting Podcast Classification")
        logger.info("="*60)
        logger.info(f"Month: {month}")
        logger.info(f"Classification Type: {classification_type}")

        with get_session() as session:
            # Build subquery for filtered chart entries
            chart_subquery = session.query(PodcastChart.channel_id).filter(
                PodcastChart.month == month
            )

            # Apply filters to chart subquery
            if countries:
                chart_subquery = chart_subquery.filter(PodcastChart.country.in_(countries))
            if platforms:
                chart_subquery = chart_subquery.filter(PodcastChart.platform.in_(platforms))
            if categories:
                chart_subquery = chart_subquery.filter(PodcastChart.category.in_(categories))
            if min_rank is not None:
                chart_subquery = chart_subquery.filter(PodcastChart.rank >= min_rank)
            if max_rank is not None:
                chart_subquery = chart_subquery.filter(PodcastChart.rank <= max_rank)

            # Get distinct channel IDs from filtered charts
            chart_subquery = chart_subquery.distinct().subquery()

            # Query channels that match the filtered chart entries
            podcasts = session.query(
                Channel.id,
                Channel.display_name,
                Channel.description,
                Channel.platform_metadata
            ).join(
                chart_subquery,
                Channel.id == chart_subquery.c.channel_id
            ).filter(
                Channel.platform == 'podcast'
            ).all()
            stats['total_podcasts'] = len(podcasts)

            logger.info(f"Found {stats['total_podcasts']} unique podcasts to classify")

            if not podcasts:
                logger.warning("No podcasts found matching filters")
                return stats

            # Classify each podcast with improved progress display
            # Custom progress bar format showing real-time stats
            pbar_format = "{desc} | {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt} | ETA: {remaining}"

            with tqdm(total=len(podcasts), desc="Starting", unit="podcast",
                      bar_format=pbar_format, ncols=120) as pbar:
                for channel_id, display_name, description, platform_metadata in podcasts:
                    source = "?"  # Track where classification came from

                    # Extract fields from platform_metadata
                    platform_metadata = platform_metadata or {}
                    creator = platform_metadata.get('creator', '')
                    categories = platform_metadata.get('categories', [])

                    # Layer 1: Check database platform_metadata first
                    existing_classification = platform_metadata.get('classification', {})
                    if (existing_classification.get('type') == classification_type and
                        existing_classification.get('is_relevant') is not None):
                        # Already classified - skip
                        is_relevant = existing_classification['is_relevant']
                        confidence = existing_classification['confidence']
                        reason = existing_classification['reason']
                        stats['classified_from_cache'] += 1
                        stats['updated_in_db'] += 1
                        source = "DB"

                        if is_relevant:
                            stats['relevant_podcasts'] += 1

                        # Update progress with stats
                        rel = stats['relevant_podcasts']
                        llm = stats['classified_from_llm']
                        cache = stats['classified_from_cache']
                        pbar.set_description(f"✓{rel} | LLM:{llm} Cache:{cache} | {display_name[:25]}")
                        pbar.update(1)
                        continue

                    # Layer 2: Check persistent cache
                    cache_key = self._get_cache_key(display_name, classification_type)

                    if cache_key in self.classification_cache:
                        cached_result = self.classification_cache[cache_key]
                        is_relevant = cached_result['is_relevant']
                        confidence = cached_result['confidence']
                        reason = cached_result['reason']
                        stats['classified_from_cache'] += 1
                        source = "CACHE"
                    else:
                        # Layer 3: Classify using LLM (only if not in DB or cache)
                        is_relevant, confidence, reason = self.classify_podcast(
                            podcast_name=display_name,
                            description=description or '',
                            creator=creator or '',
                            categories=categories or [],
                            classification_type=classification_type
                        )

                        # Save to cache
                        self.classification_cache[cache_key] = {
                            'is_relevant': is_relevant,
                            'confidence': confidence,
                            'reason': reason
                        }
                        stats['classified_from_llm'] += 1
                        source = "LLM"

                        # Save cache periodically
                        if stats['classified_from_llm'] % 10 == 0:
                            self._save_classification_cache()

                        # Small delay to avoid overwhelming the server
                        time.sleep(0.5)

                    # Update platform_metadata in database
                    channel = session.query(Channel).filter_by(id=channel_id).first()
                    if channel:
                        # Get existing platform_metadata or create new - must create a NEW dict for SQLAlchemy to detect change
                        pm = dict(channel.platform_metadata) if channel.platform_metadata else {}

                        # Store classification result
                        pm['classification'] = {
                            'type': classification_type,
                            'is_relevant': is_relevant,
                            'confidence': confidence,
                            'reason': reason,
                            'classified_at': month
                        }

                        # Assign the new dict to trigger SQLAlchemy change detection
                        channel.platform_metadata = pm
                        channel.updated_at = datetime.utcnow()
                        session.commit()
                        stats['updated_in_db'] += 1

                        if is_relevant:
                            stats['relevant_podcasts'] += 1

                    # Log each classification with details (use tqdm.write to avoid progress bar collision)
                    icon = "✓ RELEVANT" if is_relevant else "✗ SKIP    "
                    log_msg = f"{icon} [{source:5}] {display_name[:45]:<45} | conf:{confidence:.2f} | {reason[:80]}"
                    tqdm.write(log_msg)
                    logger.info(f"{icon} [{source}] {display_name} | conf:{confidence:.2f} | {reason}")

                    # Update progress bar with live stats
                    rel = stats['relevant_podcasts']
                    llm = stats['classified_from_llm']
                    cache = stats['classified_from_cache']
                    pbar.set_description(f"Relevant:{rel} | LLM:{llm} Cache:{cache}")
                    pbar.update(1)

            # Final cache save
            self._save_classification_cache()

        # Log summary
        logger.info("="*60)
        logger.info("Classification Summary")
        logger.info("="*60)
        logger.info(f"Total podcasts: {stats['total_podcasts']}")
        logger.info(f"Classified from cache: {stats['classified_from_cache']}")
        logger.info(f"Classified from LLM: {stats['classified_from_llm']}")
        logger.info(f"Relevant podcasts: {stats['relevant_podcasts']}")
        logger.info(f"Updated in database: {stats['updated_in_db']}")
        logger.info("="*60)

        return stats


def export_relevant_podcasts(
    month: str,
    project_dir: Path,
    output_file: Path,
    countries: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    min_rank: Optional[int] = None,
    max_rank: Optional[int] = None,
    min_confidence: float = 0.5
) -> int:
    """
    Export classified podcasts marked as relevant to CSV for manual review.

    Args:
        month: Month identifier (e.g., "2025-10")
        project_dir: Project directory
        output_file: Output CSV file path
        countries: Filter by country codes
        platforms: Filter by platforms
        categories: Filter by categories
        min_rank: Minimum rank filter
        max_rank: Maximum rank filter
        min_confidence: Minimum confidence score (default: 0.5)

    Returns:
        Number of podcasts exported
    """
    import csv

    logger.info(f"Exporting relevant podcasts to {output_file}")

    with get_session() as session:
        # Build query for podcasts in this month's charts
        query = session.query(
            Channel.display_name,
            Channel.description,
            Channel.platform_metadata,
            PodcastChart.platform,
            PodcastChart.country,
            PodcastChart.category,
            PodcastChart.rank
        ).join(
            PodcastChart,
            Channel.id == PodcastChart.channel_id
        ).filter(
            Channel.platform == 'podcast',
            PodcastChart.month == month
        )

        # Apply filters
        if countries:
            query = query.filter(PodcastChart.country.in_(countries))
        if platforms:
            query = query.filter(PodcastChart.platform.in_(platforms))
        if categories:
            query = query.filter(PodcastChart.category.in_(categories))
        if min_rank is not None:
            query = query.filter(PodcastChart.rank >= min_rank)
        if max_rank is not None:
            query = query.filter(PodcastChart.rank <= max_rank)

        # Get all matching podcasts
        results = query.all()

        # Filter for relevant podcasts and prepare data
        export_data = []
        seen_podcasts = set()

        for row in results:
            (display_name, description, platform_metadata, platform, country, category, rank) = row

            # Check if already processed (avoid duplicates from multiple charts)
            if display_name in seen_podcasts:
                continue

            # Extract fields from platform_metadata
            platform_metadata = platform_metadata or {}
            rss_url = platform_metadata.get('rss_url', '')
            language = platform_metadata.get('language', '')
            creator = platform_metadata.get('creator', '')
            episode_count = platform_metadata.get('episode_count', 0)
            monthly_rankings = platform_metadata.get('monthly_rankings', {})

            # Check if classified and relevant
            classification = platform_metadata.get('classification', {})
            if not classification.get('is_relevant', False):
                continue

            confidence = classification.get('confidence', 0.0)
            if confidence < min_confidence:
                continue

            seen_podcasts.add(display_name)

            # Build notes with chart positions
            chart_notes = []
            if monthly_rankings and month in monthly_rankings:
                month_charts = monthly_rankings[month]
                for chart_key, chart_rank in sorted(month_charts.items(), key=lambda x: x[1]):
                    parts = chart_key.split('_')
                    if len(parts) >= 3:
                        plat = parts[0].capitalize()
                        ctry = parts[1].upper()
                        cat = ' '.join(parts[2:]).title()
                        chart_notes.append(f"{plat} {ctry} #{chart_rank} ({cat})")

            export_data.append({
                'channel_name': display_name,
                'description': description or '',
                'rss_url': rss_url,
                'language': language,
                'creator': creator,
                'video_count': episode_count,
                'is_political': 'Yes' if classification.get('is_relevant') else 'No',
                'confidence_score': confidence,
                'classification_reason': classification.get('reason', ''),
                'notes': ' | '.join(chart_notes) if chart_notes else ''
            })

        # Sort by confidence (highest first)
        export_data.sort(key=lambda x: x['confidence_score'], reverse=True)

        # Write to CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'channel_name', 'description', 'rss_url', 'language', 'creator',
                'video_count', 'is_political', 'confidence_score',
                'classification_reason', 'notes'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_data)

        logger.info(f"Exported {len(export_data)} relevant podcasts to {output_file}")
        return len(export_data)


def classify_podcasts(
    month: str,
    project_dir: Path,
    classification_type: str,
    countries: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    min_rank: Optional[int] = None,
    max_rank: Optional[int] = None,
    llm_server_url: str = None,
    export_csv: bool = True
) -> Dict[str, int]:
    """
    Classify podcasts from monthly charts.

    Args:
        month: Month identifier (e.g., "2025-10")
        project_dir: Project directory for cache
        classification_type: What to classify for
        countries: Filter by country codes
        platforms: Filter by platforms
        categories: Filter by categories
        min_rank: Minimum rank filter
        max_rank: Maximum rank filter
        llm_server_url: LLM server URL (optional)
        export_csv: Export relevant podcasts to CSV (default: True)

    Returns:
        Statistics dictionary
    """
    classifier = PodcastClassifier(project_dir, llm_server_url)
    stats = classifier.classify_monthly_podcasts(
        month=month,
        classification_type=classification_type,
        countries=countries,
        platforms=platforms,
        categories=categories,
        min_rank=min_rank,
        max_rank=max_rank
    )

    # Export relevant podcasts to CSV for manual review
    if export_csv:
        output_file = project_dir / f"classified_podcasts_{month}.csv"
        exported_count = export_relevant_podcasts(
            month=month,
            project_dir=project_dir,
            output_file=output_file,
            countries=countries,
            platforms=platforms,
            categories=categories,
            min_rank=min_rank,
            max_rank=max_rank
        )
        stats['exported_to_csv'] = exported_count
        logger.info(f"Review classified podcasts at: {output_file}")

    return stats
