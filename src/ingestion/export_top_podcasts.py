#!/usr/bin/env python3
"""
Export Top Podcasts for Consideration

Exports top podcasts from chart data that aren't already in project sources,
with LLM classification data if available. Provides flexible filtering options.

Usage:
    # Basic - top 200 from US/CA in October 2025
    python src/ingestion/export_top_podcasts.py

    # Custom month and limit
    python src/ingestion/export_top_podcasts.py --month 2025-11 --limit 500

    # Specific countries
    python src/ingestion/export_top_podcasts.py --countries us gb de

    # Filter by LLM relevance
    python src/ingestion/export_top_podcasts.py --relevant-only

    # Minimum rank threshold
    python src/ingestion/export_top_podcasts.py --max-rank 50

    # Exclude specific sources
    python src/ingestion/export_top_podcasts.py --exclude-sources Big_channels Canadian

    # Custom output location
    python src/ingestion/export_top_podcasts.py --output /path/to/output.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import List, Set, Dict, Optional

# Add parent directory to path
sys.path.append(str(get_project_root()))

from src.database.session import get_session
from src.database.models import PodcastMetadata, PodcastChart
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('export_top_podcasts')


def load_existing_rss_urls(
    source_projects: List[str],
    projects_dir: Path
) -> tuple[Set[str], Set[str]]:
    """
    Load RSS URLs and channel names from existing source files to filter duplicates.

    Args:
        source_projects: List of project names (e.g., ['Big_channels', 'Canadian'])
        projects_dir: Path to projects directory

    Returns:
        Tuple of (RSS URLs set, channel names set) already in sources
    """
    existing_rss_urls = set()
    existing_names = set()

    for project in source_projects:
        sources_file = projects_dir / project / 'sources.csv'
        if not sources_file.exists():
            logger.warning(f"Sources file not found: {sources_file}")
            continue

        try:
            with open(sources_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rss = row.get('podcast', '').strip()
                    name = row.get('channel_name', '').strip()
                    if rss:
                        existing_rss_urls.add(rss)
                    if name:
                        existing_names.add(name.lower())

            logger.info(f"Loaded {len(existing_rss_urls)} RSS URLs and {len(existing_names)} names from {project}")
        except Exception as e:
            logger.error(f"Error reading {sources_file}: {e}")

    return existing_rss_urls, existing_names


def export_top_podcasts(
    month: str,
    countries: List[str],
    limit: int,
    max_rank: Optional[int],
    min_chart_appearances: int,
    relevant_only: bool,
    exclude_sources: List[str],
    output_path: Path,
    projects_dir: Path,
    include_no_rss: bool = False
) -> int:
    """
    Export top podcasts from charts to CSV.

    Args:
        month: Month identifier (e.g., '2025-10')
        countries: List of country codes (e.g., ['us', 'ca'])
        limit: Maximum number of podcasts to export
        max_rank: Only include podcasts with best rank <= this value
        min_chart_appearances: Minimum number of chart appearances
        relevant_only: Only include podcasts marked as relevant by LLM
        exclude_sources: List of project names to exclude RSS URLs from
        output_path: Output CSV file path
        projects_dir: Path to projects directory
        include_no_rss: Include podcasts without RSS URLs

    Returns:
        Number of podcasts exported
    """
    logger.info(f"Exporting top podcasts for {month} from {', '.join(countries)}")
    logger.info(f"Limit: {limit}, Max rank: {max_rank}, Min appearances: {min_chart_appearances}")

    # Load existing RSS URLs and names to filter
    existing_rss_urls, existing_names = load_existing_rss_urls(exclude_sources, projects_dir)
    logger.info(f"Filtering out {len(existing_rss_urls)} existing RSS URLs and {len(existing_names)} channel names")

    # Query database
    try:
        with get_session() as session:
            # Build query
            query = session.query(
                PodcastMetadata.podcast_name,
                PodcastMetadata.creator,
                PodcastMetadata.description,
                PodcastMetadata.rss_url,
                PodcastMetadata.language,
                PodcastMetadata.categories,
                PodcastMetadata.episode_count,
                PodcastMetadata.meta_data
            ).join(
                PodcastChart,
                PodcastMetadata.id == PodcastChart.podcast_id
            ).filter(
                PodcastChart.month == month,
                PodcastChart.country.in_(countries)
            )

            # Add rank filter if specified
            if max_rank is not None:
                query = query.filter(PodcastChart.rank <= max_rank)

            # Group and aggregate
            from sqlalchemy import func
            query = query.group_by(
                PodcastMetadata.id,
                PodcastMetadata.podcast_name,
                PodcastMetadata.creator,
                PodcastMetadata.description,
                PodcastMetadata.rss_url,
                PodcastMetadata.language,
                PodcastMetadata.categories,
                PodcastMetadata.episode_count,
                PodcastMetadata.meta_data
            ).having(
                func.count(func.distinct(PodcastChart.chart_key)) >= min_chart_appearances
            )

            # Add aggregated columns
            from sqlalchemy import literal_column
            query = query.add_columns(
                func.min(PodcastChart.rank).label('best_rank'),
                func.string_agg(
                    func.distinct(PodcastChart.country),
                    literal_column("', '")
                ).label('countries'),
                func.count(func.distinct(PodcastChart.chart_key)).label('chart_appearances')
            )

            # Order by best rank
            query = query.order_by(func.min(PodcastChart.rank).asc())

            # Execute query
            results = query.all()
            logger.info(f"Found {len(results)} podcasts matching criteria")

            # Process results
            podcasts = []
            for row in results:
                podcast_name = row.podcast_name or ''
                rss_url = row.rss_url or ''

                # Skip if already in sources (by RSS URL)
                if rss_url and rss_url in existing_rss_urls:
                    continue

                # Skip if already in sources (by channel name)
                if podcast_name and podcast_name.lower() in existing_names:
                    continue

                # Skip if no RSS URL and not including those
                if not rss_url and not include_no_rss:
                    continue

                # Parse LLM classification
                llm_is_relevant = ''
                llm_type = ''
                llm_reason = ''
                llm_confidence = ''

                if row.meta_data and 'classification' in row.meta_data:
                    classification = row.meta_data['classification']
                    llm_is_relevant = str(classification.get('is_relevant', ''))
                    llm_type = classification.get('type', '')
                    llm_reason = classification.get('reason', '')
                    llm_confidence = str(classification.get('confidence', ''))

                # Filter by relevance if requested
                if relevant_only and llm_is_relevant != 'True':
                    continue

                # Parse categories
                categories_str = ''
                if row.categories:
                    if isinstance(row.categories, list):
                        categories_str = ', '.join(row.categories)
                    elif isinstance(row.categories, dict):
                        categories_str = ', '.join(row.categories.keys())

                # Build category string: month-rank-category
                # e.g., "2025-10-rank1-news" or "2025-10-rank5-politics,news"
                category = f"{month}-rank{row.best_rank}"
                if categories_str:
                    # Take first 2-3 categories to keep it concise
                    cats = [c.strip() for c in categories_str.split(',')[:3]]
                    category += f"-{','.join(cats)}"

                podcasts.append({
                    'channel_name': row.podcast_name or '',
                    'description': row.description or '',
                    'podcast': rss_url,
                    'language': row.language or '',
                    'author': row.creator or '',
                    'category': category,
                    'llm_classification': llm_is_relevant,
                    'llm_justification': llm_reason
                })

                # Stop if we've hit the limit
                if len(podcasts) >= limit:
                    break

            logger.info(f"Filtered to {len(podcasts)} new podcasts")

            # Write to CSV
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'channel_name', 'description', 'podcast', 'language', 'author', 'category',
                    'llm_classification', 'llm_justification'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(podcasts)

            logger.info(f"Wrote {len(podcasts)} podcasts to {output_path}")

            # Print summary
            if podcasts:
                print(f"\n{'='*70}")
                print(f"EXPORT SUMMARY")
                print(f"{'='*70}")
                print(f"Total podcasts exported: {len(podcasts)}")
                print(f"Output file: {output_path}")
                print(f"\nTop 10 podcasts:")
                for i, p in enumerate(podcasts[:10], 1):
                    name = p['channel_name'][:50]
                    category = p['category'][:30]
                    print(f"  {i:2}. {name:50} | {category}")
                print(f"{'='*70}\n")

            return len(podcasts)

    except Exception as e:
        logger.error(f"Error exporting podcasts: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Export top podcasts from charts for consideration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export - top 200 from US/CA in October 2025
  %(prog)s

  # Custom month and limit
  %(prog)s --month 2025-11 --limit 500

  # Specific countries
  %(prog)s --countries us gb de fr

  # Only LLM-marked relevant podcasts
  %(prog)s --relevant-only

  # Top 100 ranked podcasts only
  %(prog)s --max-rank 100 --limit 100

  # Exclude specific source projects
  %(prog)s --exclude-sources Big_channels Canadian CPRMV

  # Include podcasts without RSS URLs (not enriched yet)
  %(prog)s --include-no-rss

  # Custom output location
  %(prog)s --output /path/to/custom_output.csv

  # Multiple filters combined
  %(prog)s --month 2025-11 --countries us --max-rank 50 --relevant-only --limit 100
        """
    )

    # Required arguments
    parser.add_argument(
        '--month',
        type=str,
        default='2025-10',
        help='Month identifier (default: 2025-10)'
    )

    # Country selection
    parser.add_argument(
        '--countries',
        nargs='+',
        default=['us', 'ca'],
        help='Country codes to include (default: us ca)'
    )

    # Filtering options
    parser.add_argument(
        '--limit',
        type=int,
        default=200,
        help='Maximum number of podcasts to export (default: 200)'
    )

    parser.add_argument(
        '--max-rank',
        type=int,
        default=None,
        help='Only include podcasts with best rank <= this value (default: no limit)'
    )

    parser.add_argument(
        '--min-chart-appearances',
        type=int,
        default=1,
        help='Minimum number of different charts podcast must appear in (default: 1)'
    )

    parser.add_argument(
        '--relevant-only',
        action='store_true',
        help='Only include podcasts marked as relevant by LLM classification'
    )

    parser.add_argument(
        '--include-no-rss',
        action='store_true',
        help='Include podcasts without RSS URLs (not enriched yet)'
    )

    # Source exclusion
    parser.add_argument(
        '--exclude-sources',
        nargs='+',
        default=['Big_channels', 'Canadian'],
        help='Project source files to exclude RSS URLs from (default: Big_channels Canadian)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: projects/for_consideration.csv)'
    )

    parser.add_argument(
        '--projects-dir',
        type=str,
        default=None,
        help='Projects directory path (default: auto-detect)'
    )

    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set up paths
    if args.projects_dir:
        projects_dir = Path(args.projects_dir)
    else:
        # Auto-detect based on script location
        script_dir = get_project_root()
        projects_dir = script_dir / 'projects'

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = projects_dir / 'for_consideration.csv'

    # Validate
    if not projects_dir.exists():
        logger.error(f"Projects directory not found: {projects_dir}")
        sys.exit(1)

    # Log configuration
    if args.verbose:
        logger.info("Configuration:")
        logger.info(f"  Month: {args.month}")
        logger.info(f"  Countries: {', '.join(args.countries)}")
        logger.info(f"  Limit: {args.limit}")
        logger.info(f"  Max rank: {args.max_rank}")
        logger.info(f"  Min chart appearances: {args.min_chart_appearances}")
        logger.info(f"  Relevant only: {args.relevant_only}")
        logger.info(f"  Include no RSS: {args.include_no_rss}")
        logger.info(f"  Exclude sources: {', '.join(args.exclude_sources)}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Projects dir: {projects_dir}")

    # Export
    try:
        count = export_top_podcasts(
            month=args.month,
            countries=args.countries,
            limit=args.limit,
            max_rank=args.max_rank,
            min_chart_appearances=args.min_chart_appearances,
            relevant_only=args.relevant_only,
            exclude_sources=args.exclude_sources,
            output_path=output_path,
            projects_dir=projects_dir,
            include_no_rss=args.include_no_rss
        )

        if count == 0:
            logger.warning("No podcasts exported. Try adjusting filters.")
            sys.exit(1)
        else:
            logger.info(f"Successfully exported {count} podcasts")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
