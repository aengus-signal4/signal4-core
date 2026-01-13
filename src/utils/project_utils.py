#!/usr/bin/env python3
"""
Project-related utility functions.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_language_code(language: str) -> str:
    """Normalize language code to simple 2-letter ISO 639-1 format.

    Examples:
        'en-US' -> 'en'
        'fr-CA' -> 'fr'
        'EN' -> 'en'
        'English' -> 'en'

    Args:
        language: Language code in any format

    Returns:
        Lowercase 2-letter language code
    """
    if not language:
        return 'en'  # Default to English

    # Lowercase first
    lang = language.lower().strip()

    # Handle full language names
    name_to_code = {
        'english': 'en',
        'french': 'fr',
        'spanish': 'es',
        'german': 'de',
        'italian': 'it',
        'portuguese': 'pt',
        'russian': 'ru',
        'chinese': 'zh',
        'japanese': 'ja',
        'korean': 'ko',
        'arabic': 'ar',
        'burmese': 'my',
        'norwegia': 'no',
    }
    if lang in name_to_code:
        return name_to_code[lang]

    # Handle legacy codes
    legacy_codes = {
        'iw': 'he',  # Hebrew
        'ji': 'he',  # Yiddish -> Hebrew
        'in': 'id',  # Indonesian
        'jw': 'jv',  # Javanese
        'fil': 'tl',  # Filipino -> Tagalog
        'cmn': 'zh',  # Mandarin -> Chinese
        'yue': 'zh',  # Cantonese -> Chinese
    }

    # Split on hyphen or underscore to get base code
    base = re.split(r'[-_]', lang)[0]

    # Apply legacy code mapping
    if base in legacy_codes:
        return legacy_codes[base]

    # Return base code (first 2-3 chars)
    return base[:3] if len(base) == 3 else base[:2]


def get_language_from_db(channel_url: str) -> Optional[str]:
    """Get language for a channel from the channels table (DB-first approach).

    Checks both primary_url and channel_sources for matches.

    Args:
        channel_url: The channel URL to look up

    Returns:
        Language code if found, None otherwise
    """
    try:
        from src.database.session import get_session
        from sqlalchemy import text

        with get_session() as session:
            # Query channels table - check primary_url and channel_sources
            query = text("""
                SELECT c.language
                FROM channels c
                LEFT JOIN channel_sources cs ON c.id = cs.channel_id
                WHERE c.language IS NOT NULL
                  AND (c.primary_url = :url OR cs.source_url = :url)
                LIMIT 1
            """)

            result = session.execute(query, {'url': channel_url}).fetchone()
            if result and result.language:
                return result.language

            # Try normalized URL matching for YouTube
            if 'youtube.com' in channel_url:
                normalized = normalize_youtube_url(channel_url)
                if normalized:
                    # Check if any channel URL contains this normalized form
                    query = text("""
                        SELECT c.language, c.primary_url
                        FROM channels c
                        LEFT JOIN channel_sources cs ON c.id = cs.channel_id
                        WHERE c.language IS NOT NULL
                          AND (c.primary_url LIKE :pattern OR cs.source_url LIKE :pattern)
                        LIMIT 1
                    """)
                    result = session.execute(query, {'pattern': f'%{normalized}%'}).fetchone()
                    if result and result.language:
                        return result.language

            return None

    except Exception as e:
        logger.warning(f"Error querying channels table for language: {e}")
        return None


def normalize_youtube_url(url: str) -> str:
    """Normalize YouTube URL to channel ID format for comparison.

    Handles:
    - https://www.youtube.com/@handle -> channel ID (requires lookup)
    - https://www.youtube.com/channel/CHANNEL_ID -> CHANNEL_ID
    - https://www.youtube.com/c/CustomName -> channel ID (requires lookup)
    - https://www.youtube.com/user/Username -> channel ID (requires lookup)

    For now, we extract the identifying part and compare those.
    """
    if not url:
        return ""

    # Extract channel ID if present
    channel_match = re.search(r'/channel/([^/\?]+)', url)
    if channel_match:
        return channel_match.group(1)

    # Extract handle if present
    handle_match = re.search(r'/@([^/\?]+)', url)
    if handle_match:
        return f"@{handle_match.group(1)}"

    # Extract custom name if present
    custom_match = re.search(r'/c/([^/\?]+)', url)
    if custom_match:
        return f"c/{custom_match.group(1)}"

    # Extract user name if present
    user_match = re.search(r'/user/([^/\?]+)', url)
    if user_match:
        return f"user/{user_match.group(1)}"

    # Return the URL as-is if we can't parse it
    return url

def get_language_for_channel(channel_url: str, project_sources: dict = None) -> str:
    """Get language for a channel - DB-first, with CSV fallback.

    Priority:
    1. Query channels table in database (source of truth)
    2. Fall back to project_sources dict (from CSV) if provided
    3. Raise ValueError if not found anywhere

    Args:
        channel_url: The channel URL to look up
        project_sources: Optional dict with 'url_to_language' mapping (legacy CSV support)

    Returns:
        Language code (e.g., 'en', 'fr')

    Raises:
        ValueError if channel not found in DB or CSV
    """
    # 1. Try database first (source of truth)
    db_language = get_language_from_db(channel_url)
    if db_language:
        normalized = normalize_language_code(db_language)
        logger.debug(f"Language for {channel_url}: {normalized} (from DB)")
        return normalized

    # 2. Fall back to CSV-based project_sources if provided
    if project_sources and 'url_to_language' in project_sources:
        # Try exact match first
        if channel_url in project_sources['url_to_language']:
            language = normalize_language_code(project_sources['url_to_language'][channel_url])
            logger.debug(f"Language for {channel_url}: {language} (from CSV exact match)")
            return language

        # Try to find a match by comparing base URLs (in case of slight differences)
        for url, language in project_sources['url_to_language'].items():
            if url in channel_url or channel_url in url:
                normalized = normalize_language_code(language)
                logger.debug(f"Language for {channel_url}: {normalized} (from CSV partial match)")
                return normalized

        # Try normalized lookup (fast O(1) lookup using pre-computed normalized mapping)
        if 'normalized_to_language' in project_sources:
            normalized_input = normalize_youtube_url(channel_url)
            if normalized_input and normalized_input in project_sources['normalized_to_language']:
                language = normalize_language_code(project_sources['normalized_to_language'][normalized_input])
                logger.debug(f"Language for {channel_url}: {language} (from CSV normalized)")
                return language

        # Fallback: Try normalized comparison for YouTube URLs (slower O(n) scan)
        normalized_input = normalize_youtube_url(channel_url)
        for url, language in project_sources['url_to_language'].items():
            normalized_source = normalize_youtube_url(url)
            if normalized_input and normalized_source and normalized_input == normalized_source:
                normalized = normalize_language_code(language)
                logger.debug(f"Language for {channel_url}: {normalized} (from CSV normalized scan)")
                return normalized

    # No match found - fail explicitly instead of defaulting to English
    raise ValueError(
        f"Channel URL not found in database or sources.csv: {channel_url}\n"
        f"Please add this channel to the channels table with the correct language."
    )