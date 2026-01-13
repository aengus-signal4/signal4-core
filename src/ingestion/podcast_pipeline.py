#!/usr/bin/env python3
"""
Podcast Collection Pipeline Orchestrator

Orchestrates the podcast collection and enrichment process:
- Phase 1 (collect): Scrape charts from Podstatus
- Phase 2 (enrich): Enrich with PodcastIndex metadata
- Phase 3 (classify): LLM classification
- Phase 4 (review): Streamlit dashboard for review/assignment
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(get_project_root()))

from src.ingestion.chart_collector import collect_charts, COUNTRY_CODES, ChartCollector
from src.ingestion.podcast_enricher import enrich_podcasts
from src.ingestion.podcast_classifier import classify_podcasts
# Note: podcast_reviewer.py is deprecated - use dashboards/podcast_review.py instead
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('podcast_pipeline')


class PodcastPipeline:
    """Orchestrator for podcast collection pipeline"""

    def __init__(
        self,
        base_dir: Path = None,
        delay_range: tuple = (6, 8)
    ):
        """
        Initialize pipeline.

        Args:
            base_dir: Base directory (default: projects/podcast_charts)
            delay_range: (min, max) seconds between chart requests
        """
        if base_dir is None:
            base_dir = Path("projects/podcast_charts")

        self.base_dir = base_dir
        self.delay_range = delay_range

        logger.info(f"Initialized PodcastPipeline")
        logger.info(f"Base directory: {base_dir}")

    def run_phase1_collect(
        self,
        month: str,
        countries: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        top_n: int = 200,
        skip_existing: bool = True,
        discover_mode: bool = True
    ) -> dict:
        """
        Run Phase 1: Chart Collection

        Args:
            month: Month identifier (e.g., "2025-10")
            countries: List of country codes (None = discover all)
            platforms: List of platforms (default: ["spotify", "apple"])
            categories: List of categories (None = discover per country)
            top_n: Number of top podcasts per chart
            skip_existing: Skip charts that already exist
            discover_mode: If True, intelligently discover countries/categories

        Returns:
            Statistics dictionary
        """
        logger.info("="*70)
        logger.info("PHASE 1: CHART COLLECTION")
        logger.info("="*70)

        # Create output directory for this month
        output_dir = self.base_dir / month
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = collect_charts(
            month=month,
            countries=countries,
            platforms=platforms,
            categories=categories,
            top_n=top_n,
            delay_range=self.delay_range,
            skip_existing=skip_existing,
            discover_mode=discover_mode,
            output_dir=str(output_dir)
        )

        return stats

    def run_phase2_enrich(
        self,
        month: str,
        min_importance: float = 100.0
    ) -> dict:
        """
        Run Phase 2: Podcast Enrichment

        Args:
            month: Month identifier (e.g., "2025-10")
            min_importance: Minimum importance score to enrich (default: 100.0)

        Returns:
            Statistics dictionary
        """
        logger.info("="*70)
        logger.info("PHASE 2: PODCAST ENRICHMENT")
        logger.info("="*70)

        stats = enrich_podcasts(month=month, min_importance=min_importance)

        return stats

    def run_phase3_classify(
        self,
        month: str,
        project_name: str,
        classification_type: str,
        countries: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_rank: Optional[int] = None,
        max_rank: Optional[int] = None
    ) -> dict:
        """
        Run Phase 3: Podcast Classification

        Args:
            month: Month identifier (e.g., "2025-10")
            project_name: Project name (e.g., "UnitedStates")
            classification_type: Classification criteria (e.g., "political, current affairs or news")
            countries: Filter by country codes
            platforms: Filter by platforms
            categories: Filter by categories
            min_rank: Minimum rank filter
            max_rank: Maximum rank filter

        Returns:
            Statistics dictionary
        """
        logger.info("="*70)
        logger.info("PHASE 3: PODCAST CLASSIFICATION")
        logger.info("="*70)

        project_dir = Path("projects") / project_name

        stats = classify_podcasts(
            month=month,
            project_dir=project_dir,
            classification_type=classification_type,
            countries=countries,
            platforms=platforms,
            categories=categories,
            min_rank=min_rank,
            max_rank=max_rank
        )

        return stats

    def run_phase4_review(
        self,
        project_name: str,
        countries: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> dict:
        """
        Run Phase 4: Review and Assign Podcasts

        NOTE: This phase now uses a Streamlit dashboard instead of CLI.
        Run: streamlit run dashboards/podcast_review.py

        Args:
            project_name: Project name (e.g., "CPRMV")
            countries: Filter by country codes (set in dashboard)
            limit: Not used (dashboard has its own pagination)

        Returns:
            Instructions for launching dashboard
        """
        logger.info("="*70)
        logger.info("PHASE 4: REVIEW AND ASSIGN PODCASTS")
        logger.info("="*70)
        logger.info("")
        logger.info("Phase 4 now uses a Streamlit dashboard for review.")
        logger.info("")
        logger.info("To review podcasts, run:")
        logger.info("    streamlit run dashboards/podcast_review.py")
        logger.info("")
        logger.info(f"Then select project: {project_name}")
        if countries:
            logger.info(f"And filter by countries: {', '.join(countries)}")
        logger.info("")
        logger.info("The dashboard allows you to:")
        logger.info("  - Review candidates sorted by chart rank")
        logger.info("  - Approve podcasts to multiple projects at once")
        logger.info("  - Reject podcasts (tracked per-project)")
        logger.info("  - View assigned and rejected podcasts")
        logger.info("="*70)

        return {
            'status': 'dashboard_required',
            'command': 'streamlit run dashboards/podcast_review.py',
            'project': project_name
        }

    def run_all_phases(
        self,
        month: str,
        countries: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        top_n: int = 200,
        skip_existing: bool = True,
        discover_mode: bool = True
    ) -> dict:
        """
        Run all phases in sequence.

        Args:
            month: Month identifier (e.g., "2025-10")
            countries: List of country codes (None = discover all)
            platforms: List of platforms
            categories: List of categories (None = discover per country)
            top_n: Number of top podcasts per chart
            skip_existing: Skip existing charts
            discover_mode: If True, intelligently discover countries/categories

        Returns:
            Combined statistics dictionary
        """
        logger.info("="*70)
        logger.info("RUNNING ALL PHASES")
        logger.info("="*70)

        # Phase 1: Collect
        collect_stats = self.run_phase1_collect(
            month=month,
            countries=countries,
            platforms=platforms,
            categories=categories,
            top_n=top_n,
            skip_existing=skip_existing,
            discover_mode=discover_mode
        )

        # Phase 2: Enrich
        enrich_stats = self.run_phase2_enrich(month=month)

        # Combine stats
        return {
            'collect': collect_stats,
            'enrich': enrich_stats
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Podcast Collection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect October 2025 charts for US
  python src/ingestion/podcast_pipeline.py \\
    --phase collect \\
    --month 2025-10 \\
    --countries us

  # Enrich October 2025 charts
  python src/ingestion/podcast_pipeline.py \\
    --phase enrich \\
    --month 2025-10

  # Classify podcasts for a project
  python src/ingestion/podcast_pipeline.py \\
    --phase classify \\
    --month 2025-10 \\
    --project UnitedStates \\
    --countries us \\
    --max-rank 50

  # Review and merge classified podcasts into sources.csv
  python src/ingestion/podcast_pipeline.py \\
    --phase review \\
    --project Big_Channels \\
    --countries us \\
    -n 50

  # Run all phases
  python src/ingestion/podcast_pipeline.py \\
    --phase all \\
    --month 2025-10 \\
    --countries us,ca,gb,fr,de
        """
    )

    # Phase selection
    parser.add_argument(
        '--phase',
        required=True,
        choices=['collect', 'enrich', 'classify', 'review', 'all'],
        help='Which phase to run'
    )

    # Common arguments
    parser.add_argument(
        '--month',
        help='Month identifier (e.g., 2025-10). Default: current month'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('projects/podcast_collection'),
        help='Base directory (default: projects/podcast_collection)'
    )

    # Phase 1 (collect) arguments
    parser.add_argument(
        '--countries',
        help='Comma-separated country codes (e.g., us,ca,gb). Omit to discover all.'
    )
    parser.add_argument(
        '--no-discover',
        action='store_true',
        help='Disable intelligent discovery (requires explicit countries/categories)'
    )
    parser.add_argument(
        '--platforms',
        help='Comma-separated platforms (default: spotify,apple)'
    )
    parser.add_argument(
        '--categories',
        help='Comma-separated categories (default: all 6 categories)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=200,
        help='Number of top podcasts per chart (default: 200)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-fetch charts that already exist'
    )
    parser.add_argument(
        '--delay-min',
        type=int,
        default=6,
        help='Minimum delay between requests in seconds (default: 6)'
    )
    parser.add_argument(
        '--delay-max',
        type=int,
        default=8,
        help='Maximum delay between requests in seconds (default: 8)'
    )

    # Phase 2 (enrich) arguments
    parser.add_argument(
        '--min-importance',
        type=float,
        default=100.0,
        help='Minimum importance score to enrich (default: 100.0). Set to 0 to enrich all.'
    )

    # Phase 3 (classify) arguments
    parser.add_argument(
        '--project',
        help='Project name for classification (e.g., UnitedStates, Europe)'
    )
    parser.add_argument(
        '--classification-type',
        default='political, current affairs or news',
        help='Classification criteria (default: "political, current affairs or news")'
    )
    parser.add_argument(
        '--min-rank',
        type=int,
        help='Minimum rank to classify (e.g., 1)'
    )
    parser.add_argument(
        '--max-rank',
        type=int,
        help='Maximum rank to classify (e.g., 50)'
    )

    # Phase 4 (review) arguments
    parser.add_argument(
        '-n',
        type=int,
        help='Number of podcasts to review (default: all)'
    )

    args = parser.parse_args()

    # Determine month
    if not args.month:
        args.month = datetime.now().strftime('%Y-%m')
        logger.info(f"No month specified, using current: {args.month}")

    # Parse countries and discovery mode
    countries = None
    discover_mode = not args.no_discover

    if args.phase in ['collect', 'all']:
        if args.countries:
            countries = [c.strip().lower() for c in args.countries.split(',')]
            logger.info(f"Using specified countries: {', '.join(countries)}")
        else:
            if discover_mode:
                logger.info("No countries specified - will discover all available")
            else:
                logger.error("Must specify --countries when --no-discover is set")
                return 1
    elif args.phase == 'classify':
        # Parse countries for classify phase
        if args.countries:
            countries = [c.strip().lower() for c in args.countries.split(',')]
            logger.info(f"Filtering classification to countries: {', '.join(countries)}")
    elif args.phase == 'review':
        # Parse countries for review phase
        if args.countries:
            countries = [c.strip().lower() for c in args.countries.split(',')]
            logger.info(f"Filtering review to countries: {', '.join(countries)}")

    # Parse platforms
    platforms = None
    if args.platforms:
        platforms = [p.strip().lower() for p in args.platforms.split(',')]

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip().lower() for c in args.categories.split(',')]

    # Initialize pipeline
    pipeline = PodcastPipeline(
        base_dir=args.base_dir,
        delay_range=(args.delay_min, args.delay_max)
    )

    try:
        # Run requested phase
        if args.phase == 'collect':
            stats = pipeline.run_phase1_collect(
                month=args.month,
                countries=countries,
                platforms=platforms,
                categories=categories,
                top_n=args.top_n,
                skip_existing=not args.no_skip_existing,
                discover_mode=discover_mode
            )

            logger.info("="*70)
            logger.info("PHASE 1 COMPLETE")
            logger.info("="*70)
            logger.info(f"Discovery requests: {stats.get('discovery_requests', 0)}")
            logger.info(f"Successful charts: {stats['successful']}")
            logger.info(f"Total podcasts collected: {stats['total_podcasts']}")

        elif args.phase == 'enrich':
            stats = pipeline.run_phase2_enrich(month=args.month, min_importance=args.min_importance)

            logger.info("="*70)
            logger.info("PHASE 2 COMPLETE")
            logger.info("="*70)
            logger.info(f"New podcasts enriched: {stats['new_podcasts']}")
            logger.info(f"Skipped (low importance): {stats.get('skipped_low_importance', 0)}")
            logger.info(f"Rankings updated: {stats['rankings_updated']}")

        elif args.phase == 'classify':
            # Validate required arguments
            if not args.project:
                logger.error("--project is required for classification phase")
                return 1

            stats = pipeline.run_phase3_classify(
                month=args.month,
                project_name=args.project,
                classification_type=args.classification_type,
                countries=countries if countries else None,
                platforms=platforms if platforms else None,
                categories=categories if categories else None,
                min_rank=args.min_rank,
                max_rank=args.max_rank
            )

            logger.info("="*70)
            logger.info("PHASE 3 COMPLETE")
            logger.info("="*70)
            logger.info(f"Total podcasts: {stats['total_podcasts']}")
            logger.info(f"Classified from cache: {stats['classified_from_cache']}")
            logger.info(f"Classified from LLM: {stats['classified_from_llm']}")
            logger.info(f"Relevant podcasts: {stats['relevant_podcasts']}")
            if 'exported_to_csv' in stats:
                logger.info(f"Exported to CSV: {stats['exported_to_csv']} podcasts")
                logger.info(f"Review file: projects/{args.project}/classified_podcasts_{args.month}.csv")

        elif args.phase == 'review':
            # Validate required arguments
            if not args.project:
                logger.error("--project is required for review phase")
                return 1

            stats = pipeline.run_phase4_review(
                project_name=args.project,
                countries=countries if countries else None,
                limit=args.n
            )

            # Stats now contains dashboard launch instructions
            if stats.get('status') == 'dashboard_required':
                logger.info("")
                logger.info("Run the command above to launch the review dashboard.")

        elif args.phase == 'all':
            stats = pipeline.run_all_phases(
                month=args.month,
                countries=countries,
                platforms=platforms,
                categories=categories,
                top_n=args.top_n,
                skip_existing=not args.no_skip_existing,
                discover_mode=discover_mode
            )

            logger.info("="*70)
            logger.info("ALL PHASES COMPLETE")
            logger.info("="*70)
            logger.info("Phase 1 (Collect):")
            logger.info(f"  Discovery requests: {stats['collect'].get('discovery_requests', 0)}")
            logger.info(f"  Successful charts: {stats['collect']['successful']}")
            logger.info(f"  Total podcasts: {stats['collect']['total_podcasts']}")
            logger.info("Phase 2 (Enrich):")
            logger.info(f"  New podcasts: {stats['enrich']['new_podcasts']}")
            logger.info(f"  Rankings updated: {stats['enrich']['rankings_updated']}")

        logger.info("="*70)
        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
