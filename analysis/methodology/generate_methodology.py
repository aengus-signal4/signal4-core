#!/usr/bin/env python3
"""
Generate LaTeX methodology section for Nature paper.

This script queries the PostgreSQL database to extract corpus statistics
and generates LaTeX files for the methodology section and appendix.

Usage:
    python -m scripts.methodology.generate_methodology --output-dir output/methodology

    # Full options
    python -m scripts.methodology.generate_methodology \
        --output-dir output/methodology \
        --start-date 2015-01-01 \
        --end-date 2025-12-31 \
        --projects Big_Channels Canadian Europe \
        --exclude-languages ru uk ua ca \
        --include-speaker-id \
        --include-classification \
        --chart-month 2025-12
"""

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Set

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_PROJECTS,
    EXCLUDED_LANGUAGES, LANGUAGE_NAMES
)
from db_queries import (
    get_corpus_statistics,
    get_channel_list,
    get_language_distribution,
    get_year_distribution,
    get_processing_statistics,
    get_chart_rankings_summary,
    get_total_word_count_estimate
)
from latex_templates import (
    generate_document_header,
    generate_corpus_overview_section,
    generate_corpus_table_rows,
    generate_selection_criteria_section,
    generate_pipeline_section,
    generate_quality_section,
    generate_language_table,
    generate_year_table,
    generate_models_table,
    generate_appendix_header,
    generate_podcast_table,
    generate_summary_tables,
    generate_document_footer
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX methodology section for Nature paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/methodology",
        help="Output directory for generated files (default: output/methodology)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE.isoformat(),
        help=f"Start date for filtering (default: {DEFAULT_START_DATE.isoformat()})"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE.isoformat(),
        help=f"End date for filtering (default: {DEFAULT_END_DATE.isoformat()})"
    )

    parser.add_argument(
        "--projects",
        nargs="+",
        default=DEFAULT_PROJECTS,
        help=f"Projects to include (default: {' '.join(DEFAULT_PROJECTS)})"
    )

    parser.add_argument(
        "--exclude-languages",
        nargs="+",
        default=list(EXCLUDED_LANGUAGES),
        help=f"Languages to exclude (default: {' '.join(EXCLUDED_LANGUAGES)})"
    )

    parser.add_argument(
        "--chart-month",
        type=str,
        default="2025-12",
        help="Month for chart rankings (YYYY-MM format, default: 2025-12)"
    )

    parser.add_argument(
        "--include-speaker-id",
        action="store_true",
        help="Include speaker identification methodology section"
    )

    parser.add_argument(
        "--include-classification",
        action="store_true",
        help="Include theme classification methodology section"
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Methodology: Podcast Corpus Collection and Processing",
        help="Document title"
    )

    parser.add_argument(
        "--authors",
        type=str,
        default="",
        help="Author names for document"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def log(message: str, verbose: bool = True):
    """Print message if verbose mode is enabled."""
    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def generate_methodology_main(
    output_dir: str,
    start_date: date,
    end_date: date,
    projects: List[str],
    exclude_languages: Set[str],
    chart_month: str,
    include_speaker_id: bool,
    include_classification: bool,
    title: str,
    authors: str,
    verbose: bool
) -> str:
    """Generate the main methodology LaTeX document."""

    log("Querying corpus statistics...", verbose)
    stats = get_corpus_statistics(projects, start_date, end_date, exclude_languages)

    log("Querying language distribution...", verbose)
    lang_stats = get_language_distribution(projects, start_date, end_date, exclude_languages)

    log("Querying temporal distribution...", verbose)
    year_stats = get_year_distribution(projects, start_date, end_date, exclude_languages)

    generation_date = datetime.now().strftime('%Y-%m-%d')

    # Build document
    sections = []

    # Header
    sections.append(generate_document_header(title, authors))

    # Abstract/overview
    sections.append(generate_corpus_overview_section(stats, generation_date))
    sections.append(generate_corpus_table_rows(stats))

    # Selection criteria
    sections.append(generate_selection_criteria_section())

    # Pipeline documentation
    sections.append(generate_pipeline_section(include_speaker_id, include_classification))

    # Quality assurance
    sections.append(generate_quality_section())

    # Supplementary tables
    sections.append("\n\\section{Supplementary Tables}\n")
    sections.append(generate_language_table(lang_stats))
    sections.append(generate_year_table(year_stats))
    sections.append(generate_models_table())

    # Reference to appendix
    sections.append("""
\\section{Data Availability}

The complete list of podcasts included in this corpus is provided in
Appendix~\\ref{sec:appendix_podcasts}. This includes podcast names,
languages, chart rankings as of December 2025, and brief descriptions.

The transcription corpus and associated embeddings are available upon
reasonable request for research purposes.
""")

    # Footer
    sections.append(generate_document_footer())

    return "\n\n".join(sections)


def generate_appendix(
    output_dir: str,
    projects: List[str],
    exclude_languages: Set[str],
    chart_month: str,
    verbose: bool
) -> str:
    """Generate the appendix LaTeX document with podcast listings."""

    log("Querying channel list with chart rankings...", verbose)
    channels = get_channel_list(projects, exclude_languages, chart_month)
    log(f"  Found {len(channels)} channels", verbose)

    log("Querying language distribution for summary...", verbose)
    lang_stats = get_language_distribution(
        projects,
        date(2015, 1, 1),
        date(2025, 12, 31),
        exclude_languages
    )

    # Build document
    sections = []

    # Header (standalone document)
    sections.append(generate_document_header(
        "Appendix: Podcast Corpus Details",
        ""
    ))

    # Appendix content
    sections.append(generate_appendix_header())

    # Summary tables first
    sections.append(generate_summary_tables(channels, lang_stats))

    # Full listings by project
    for project in sorted(set(ch.project for ch in channels)):
        log(f"  Generating table for {project}...", verbose)
        sections.append(generate_podcast_table(channels, project))

    # Footer
    sections.append(generate_document_footer())

    return "\n\n".join(sections)


def generate_statistics_summary(
    projects: List[str],
    exclude_languages: Set[str],
    start_date: date,
    end_date: date,
    verbose: bool
) -> str:
    """Generate a plain text summary of statistics for quick reference."""

    log("Generating statistics summary...", verbose)

    stats = get_corpus_statistics(
        projects,
        start_date,
        end_date,
        exclude_languages
    )

    processing = get_processing_statistics(projects, exclude_languages)
    word_counts = get_total_word_count_estimate(projects, exclude_languages)
    chart_summary = get_chart_rankings_summary("2025-12")

    lines = [
        "=" * 60,
        "CORPUS STATISTICS SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Date Range: {start_date} to {end_date}",
        "=" * 60,
        "",
        "CORPUS COMPOSITION",
        "-" * 40
    ]

    total_episodes = 0
    total_transcribed = 0
    total_hours = 0
    total_channels = 0

    for project, s in sorted(stats.items()):
        total_episodes += s.total_episodes
        total_transcribed += s.transcribed_episodes
        total_hours += s.total_hours
        total_channels += s.channel_count

        pct = s.transcribed_episodes / s.total_episodes * 100 if s.total_episodes > 0 else 0
        lines.append(f"\n{project}:")
        lines.append(f"  Channels: {s.channel_count:,}")
        lines.append(f"  Episodes: {s.total_episodes:,} (indexed)")
        lines.append(f"  Transcribed: {s.transcribed_episodes:,} ({pct:.1f}%)")
        lines.append(f"  Hours: {s.total_hours:,.1f}")
        lines.append(f"  Avg duration: {s.avg_duration_minutes:.1f} min")
        lines.append(f"  Date range: {s.earliest_date} to {s.latest_date}")

    pct_total = total_transcribed / total_episodes * 100 if total_episodes > 0 else 0
    lines.extend([
        "",
        "-" * 40,
        "TOTALS:",
        f"  Channels: {total_channels:,}",
        f"  Episodes: {total_episodes:,}",
        f"  Transcribed: {total_transcribed:,} ({pct_total:.1f}%)",
        f"  Hours: {total_hours:,.1f}",
        "",
        "PROCESSING STATISTICS",
        "-" * 40
    ])

    # Segments
    total_segments = sum(
        p.get('count', 0)
        for p in processing.get('segments', {}).values()
    )
    lines.append(f"Total embedding segments: {total_segments:,}")

    # Estimated word count
    total_words = sum(word_counts.values())
    lines.append(f"Estimated word count: {total_words:,}")

    # Speaker turns
    total_turns = sum(
        p.get('turn_count', 0)
        for p in processing.get('speakers', {}).values()
    )
    lines.append(f"Total speaker turns: {total_turns:,}")

    # Chart coverage
    if chart_summary.get('platforms'):
        lines.extend([
            "",
            "CHART RANKINGS (December 2025)",
            "-" * 40
        ])
        for platform, data in chart_summary['platforms'].items():
            lines.append(
                f"  {platform.title()}: {data['unique_channels']:,} channels, "
                f"{data['countries']} countries"
            )

    lines.extend(["", "=" * 60])

    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    exclude_languages = set(args.exclude_languages)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Output directory: {output_dir}", args.verbose)
    log(f"Date range: {start_date} to {end_date}", args.verbose)
    log(f"Projects: {args.projects}", args.verbose)
    log(f"Excluded languages: {exclude_languages}", args.verbose)

    # Generate main methodology document
    log("\n--- Generating main methodology document ---", args.verbose)
    methodology_tex = generate_methodology_main(
        str(output_dir),
        start_date,
        end_date,
        args.projects,
        exclude_languages,
        args.chart_month,
        args.include_speaker_id,
        args.include_classification,
        args.title,
        args.authors,
        args.verbose
    )

    methodology_path = output_dir / "methodology.tex"
    with open(methodology_path, "w", encoding="utf-8") as f:
        f.write(methodology_tex)
    log(f"Written: {methodology_path}", args.verbose)

    # Generate appendix document
    log("\n--- Generating appendix document ---", args.verbose)
    appendix_tex = generate_appendix(
        str(output_dir),
        args.projects,
        exclude_languages,
        args.chart_month,
        args.verbose
    )

    appendix_path = output_dir / "appendix_podcasts.tex"
    with open(appendix_path, "w", encoding="utf-8") as f:
        f.write(appendix_tex)
    log(f"Written: {appendix_path}", args.verbose)

    # Generate statistics summary (plain text)
    log("\n--- Generating statistics summary ---", args.verbose)
    stats_summary = generate_statistics_summary(
        args.projects,
        exclude_languages,
        start_date,
        end_date,
        args.verbose
    )

    stats_path = output_dir / "statistics_summary.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_summary)
    log(f"Written: {stats_path}", args.verbose)

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {methodology_path}")
    print(f"  - {appendix_path}")
    print(f"  - {stats_path}")
    print(f"\nTo compile LaTeX:")
    print(f"  pdflatex {methodology_path}")
    print(f"  pdflatex {appendix_path}")


if __name__ == "__main__":
    main()
