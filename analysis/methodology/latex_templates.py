"""
LaTeX templates and generation functions for methodology section.

This module provides templates and functions to generate Nature-style
LaTeX documentation for the methodology section.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from textwrap import dedent

try:
    from .config import (
        LANGUAGE_NAMES, MODELS, PIPELINE_STAGES, LATEX_CONFIG,
        PROJECT_COUNTRIES
    )
    from .db_queries import CorpusStats, ChannelInfo, LanguageStats, YearStats
except ImportError:
    from config import (
        LANGUAGE_NAMES, MODELS, PIPELINE_STAGES, LATEX_CONFIG,
        PROJECT_COUNTRIES
    )
    from db_queries import CorpusStats, ChannelInfo, LanguageStats, YearStats


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not text:
        return ""
    replacements = [
        ('\\', '\\textbackslash{}'),
        ('&', '\\&'),
        ('%', '\\%'),
        ('$', '\\$'),
        ('#', '\\#'),
        ('_', '\\_'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('~', '\\textasciitilde{}'),
        ('^', '\\textasciicircum{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def truncate_description(desc: Optional[str], max_length: int = 150) -> str:
    """Truncate description to max length with ellipsis."""
    if not desc:
        return ""
    desc = desc.replace('\n', ' ').strip()
    if len(desc) <= max_length:
        return escape_latex(desc)
    return escape_latex(desc[:max_length-3].rsplit(' ', 1)[0] + '...')


def format_number(n: int) -> str:
    """Format number with thousands separator."""
    return f"{n:,}"


def format_hours(hours: float) -> str:
    """Format hours with one decimal place."""
    return f"{hours:,.1f}"


def generate_document_header(title: str, authors: str = "") -> str:
    """Generate LaTeX document header with necessary packages."""
    packages = "\n".join([f"\\usepackage{{{pkg}}}" for pkg in LATEX_CONFIG["packages"]])

    return dedent(f"""
    \\documentclass[11pt]{{article}}

    % Packages
    {packages}

    % Custom colors
    \\definecolor{{tableheader}}{{RGB}}{{240, 240, 240}}

    % Title
    \\title{{{title}}}
    \\author{{{authors}}}
    \\date{{Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}}

    \\begin{{document}}

    \\maketitle
    """).strip()


def generate_corpus_overview_section(
    stats: Dict[str, CorpusStats],
    generation_date: str
) -> str:
    """Generate the corpus overview section."""
    # Calculate totals
    total_episodes = sum(s.total_episodes for s in stats.values())
    total_transcribed = sum(s.transcribed_episodes for s in stats.values())
    total_hours = sum(s.total_hours for s in stats.values())
    total_channels = sum(s.channel_count for s in stats.values())

    # Find date range
    earliest = min((s.earliest_date for s in stats.values() if s.earliest_date), default=None)
    latest = max((s.latest_date for s in stats.values() if s.latest_date), default=None)

    return dedent(f"""
    \\section{{Data Collection and Processing}}

    \\subsection{{Corpus Overview}}

    This study analyzes a large-scale corpus of political podcast transcriptions
    collected from {format_number(total_channels)} podcast channels across North America
    and Europe. The corpus spans from {earliest.strftime('%B %Y') if earliest else 'N/A'}
    to {latest.strftime('%B %Y') if latest else 'N/A'}, comprising
    {format_number(total_episodes)} episodes with {format_number(total_transcribed)}
    ({total_transcribed/total_episodes*100:.1f}\\%) fully transcribed, representing
    approximately {format_hours(total_hours)} hours of audio content.

    Table~\\ref{{tab:corpus_summary}} presents the corpus composition by geographic region.

    \\begin{{table}}[H]
    \\centering
    \\caption{{Corpus composition by region. Episodes indexed as of {generation_date}.}}
    \\label{{tab:corpus_summary}}
    {LATEX_CONFIG['table_font_size']}
    \\begin{{tabular}}{{lrrrr}}
    \\toprule
    \\textbf{{Region}} & \\textbf{{Channels}} & \\textbf{{Episodes}} & \\textbf{{Transcribed}} & \\textbf{{Hours}} \\\\
    \\midrule
    """).strip()


def generate_corpus_table_rows(stats: Dict[str, CorpusStats]) -> str:
    """Generate table rows for corpus summary."""
    rows = []

    # Project display names
    display_names = {
        "Big_Channels": "United States",
        "Canadian": "Canada",
        "Europe": "European Union"
    }

    for project, s in sorted(stats.items()):
        name = display_names.get(project, project)
        pct = s.transcribed_episodes / s.total_episodes * 100 if s.total_episodes > 0 else 0
        rows.append(
            f"{name} & {format_number(s.channel_count)} & "
            f"{format_number(s.total_episodes)} & "
            f"{format_number(s.transcribed_episodes)} ({pct:.1f}\\%) & "
            f"{format_hours(s.total_hours)} \\\\"
        )

    # Total row
    total_channels = sum(s.channel_count for s in stats.values())
    total_episodes = sum(s.total_episodes for s in stats.values())
    total_transcribed = sum(s.transcribed_episodes for s in stats.values())
    total_hours = sum(s.total_hours for s in stats.values())
    pct_total = total_transcribed / total_episodes * 100 if total_episodes > 0 else 0

    rows.append("\\midrule")
    rows.append(
        f"\\textbf{{Total}} & \\textbf{{{format_number(total_channels)}}} & "
        f"\\textbf{{{format_number(total_episodes)}}} & "
        f"\\textbf{{{format_number(total_transcribed)}}} ({pct_total:.1f}\\%) & "
        f"\\textbf{{{format_hours(total_hours)}}} \\\\"
    )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")

    return "\n".join(rows)


def generate_selection_criteria_section() -> str:
    """Generate the podcast selection criteria section."""
    return dedent("""
    \\subsection{Podcast Selection Criteria}

    Podcasts were selected based on systematic chart rankings from major
    podcast platforms (Spotify and Apple Podcasts) to ensure representation
    of widely-consumed political content. The selection process followed
    a three-stage pipeline:

    \\begin{enumerate}
        \\item \\textbf{Chart Collection}: We collected top-200 podcast rankings
        from Podstatus.com, which aggregates chart data from Spotify and Apple
        Podcasts across 175 countries and multiple content categories including
        news, politics, news-commentary, and society-culture.

        \\item \\textbf{Metadata Enrichment}: RSS feed URLs and detailed podcast
        metadata were retrieved from the PodcastIndex API. For ambiguous matches,
        we employed LLM-assisted disambiguation to ensure accurate feed identification.

        \\item \\textbf{Relevance Classification}: Each podcast was classified for
        political relevance using a combination of category tags, description
        analysis, and LLM-based content classification with confidence scoring.
    \\end{enumerate}

    This approach prioritizes podcasts with demonstrated audience reach while
    maintaining systematic, reproducible selection criteria.
    """).strip()


def generate_pipeline_section(
    include_speaker_id: bool = False,
    include_classification: bool = False
) -> str:
    """Generate the data processing pipeline section."""

    core_pipeline = dedent("""
    \\subsection{Data Collection Pipeline}

    The data collection and processing pipeline consists of the following stages:

    \\subsubsection{Episode Acquisition}

    Episode metadata is extracted from RSS feeds using the feedparser library.
    For each episode, we capture: title, description, publication date, duration,
    and audio file URL. Episodes are deduplicated using unique episode identifiers
    (GUID or URL hash). Audio files are downloaded via HTTP/HTTPS with retry logic,
    supporting multiple formats including MP3, M4A, AAC, OGG, and WAV.

    \\subsubsection{Audio Standardization}

    All audio files are converted to a standardized format for consistent downstream
    processing: 16 kHz sample rate, mono channel, 16-bit PCM WAV. Audio normalization
    follows the ITU-R BS.1770 loudness standard using FFmpeg's loudnorm filter.
    Long episodes are processed in 5-minute chunks with 2-second overlap to manage
    memory constraints while maintaining transcription continuity.

    \\subsection{Audio Processing and Transcription}

    \\subsubsection{Speech-to-Text}

    Transcription is performed using the Whisper Large v3 Turbo model
    (mlx-community/whisper-large-v3-turbo), optimized for Apple Silicon via
    the MLX framework. Key parameters:

    \\begin{itemize}
        \\item Word-level timestamps enabled for precise alignment
        \\item Automatic language detection with manual override from metadata
        \\item Beam search decoding (beam size = 5)
        \\item Temperature = 0.0 for deterministic output
    \\end{itemize}

    For English content, an alternative single-pass Parakeet model is used
    for improved performance on English-specific content.

    \\subsection{Speaker Diarization and Attribution}

    \\subsubsection{Diarization}

    Speaker diarization employs the PyAnnote speaker-diarization-3.1 pipeline
    to identify speaker segments within each episode. The model detects speaker
    changes and assigns temporary speaker labels (SPEAKER\\_0, SPEAKER\\_1, etc.)
    with start and end timestamps for each speaking segment.

    \\subsubsection{Speaker Attribution (Stitching)}

    A 14-stage ``stitching'' pipeline aligns word-level transcription with
    speaker diarization to produce attributed transcripts:

    \\begin{enumerate}
        \\item Data loading and validation
        \\item Boundary deduplication and overlap removal
        \\item Word-level table construction with grammar classification
        \\item Single-speaker segment assignment (high confidence)
        \\item Multi-speaker segment resolution using speaker embeddings
        \\item Speaker centroid calculation from high-quality segments
        \\item Embedding-based speaker assignment for ambiguous regions
        \\item Conflict resolution and cleanup
        \\item Output generation with turn-level speaker attribution
    \\end{enumerate}

    Speaker embeddings are extracted using the PyAnnote embedding model,
    with quality tiering based on segment duration (high quality: $\\geq$2.0s,
    medium: 1.5--2.0s, low: $<$1.5s).

    \\subsubsection{Semantic Segmentation}

    For downstream analysis, speaker-attributed transcripts are segmented into
    semantically coherent units using a beam-search algorithm that optimizes
    for both semantic coherence and segment length:

    \\begin{itemize}
        \\item Target segment length: 250 tokens (range: 50--400)
        \\item Coherence threshold: 0.7 (cosine similarity)
        \\item Similarity model: XLM-RoBERTa (sentence-transformers/paraphrase-xlm-r-multilingual-v1)
        \\item Beam width: 5 candidates
        \\item Lookahead: 3 sentences
    \\end{itemize}

    Each segment is embedded using the Qwen3-Embedding-0.6B model (1024 dimensions)
    for semantic search and analysis applications.
    """).strip()

    optional_sections = ""

    if include_speaker_id:
        optional_sections += dedent("""

    \\subsubsection{Cross-Episode Speaker Identification}

    For longitudinal analysis, speakers are identified across episodes using
    a multi-phase pipeline:

    \\begin{enumerate}
        \\item \\textbf{Text Evidence Collection}: LLM analysis of speaker turns
        to identify explicit name mentions and self-introductions
        \\item \\textbf{Anchor-Verified Clustering}: Speakers with text-verified
        identities serve as anchors; unidentified speakers are clustered based
        on embedding similarity and role consistency
        \\item \\textbf{Identity Merge Detection}: Cross-episode deduplication
        using centroid similarity ($>$0.85 threshold) with LLM verification
    \\end{enumerate}
        """).strip()

    if include_classification:
        optional_sections += dedent("""

    \\subsubsection{Theme Classification}

    Transcript segments are classified into thematic categories using a
    semantic search approach:

    \\begin{enumerate}
        \\item FAISS index of theme/subtheme embeddings (pre-computed from descriptions)
        \\item Semantic similarity search (threshold: 0.38)
        \\item LLM validation for subtheme assignment
        \\item Likert scale confidence scoring (1--5)
    \\end{enumerate}
        """).strip()

    return core_pipeline + optional_sections


def generate_quality_section() -> str:
    """Generate quality assurance section."""
    return dedent("""
    \\subsection{Quality Assurance}

    Multiple quality control measures are implemented throughout the pipeline:

    \\begin{itemize}
        \\item \\textbf{Audio Validation}: FFprobe integrity checks, minimum
        file size (50KB), bitrate validation ($\\geq$10 kbps)
        \\item \\textbf{Transcription Validation}: Segment sanitization (empty
        text removal, timestamp validation, speaking-rate heuristics)
        \\item \\textbf{Speaker Attribution}: Coverage metrics (percentage of
        words assigned), confidence distribution tracking
        \\item \\textbf{Timing Precision}: Word-level timestamp verification
        with fallback to linear interpolation ($\\pm$50ms precision target)
    \\end{itemize}

    Corrupted or incomplete downloads are automatically detected and
    blocked from reprocessing. Processing errors are logged with
    structured metadata for audit and debugging.
    """).strip()


def generate_language_table(lang_stats: List[LanguageStats]) -> str:
    """Generate language distribution table."""
    rows = ["\\begin{table}[H]",
            "\\centering",
            "\\caption{Corpus composition by language.}",
            "\\label{tab:language_distribution}",
            LATEX_CONFIG['table_font_size'],
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "\\textbf{Language} & \\textbf{Channels} & \\textbf{Episodes} & "
            "\\textbf{Transcribed} & \\textbf{Hours} \\\\",
            "\\midrule"]

    for stat in lang_stats[:15]:  # Top 15 languages
        lang_name = LANGUAGE_NAMES.get(stat.language, stat.language.upper())
        pct = stat.transcribed_count / stat.episode_count * 100 if stat.episode_count > 0 else 0
        rows.append(
            f"{lang_name} & {format_number(stat.channel_count)} & "
            f"{format_number(stat.episode_count)} & "
            f"{format_number(stat.transcribed_count)} ({pct:.0f}\\%) & "
            f"{format_hours(stat.hours)} \\\\"
        )

    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(rows)


def generate_year_table(year_stats: List[YearStats]) -> str:
    """Generate temporal distribution table."""
    rows = ["\\begin{table}[H]",
            "\\centering",
            "\\caption{Temporal distribution of episodes by publication year.}",
            "\\label{tab:year_distribution}",
            LATEX_CONFIG['table_font_size'],
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "\\textbf{Year} & \\textbf{Episodes} & \\textbf{Transcribed} & "
            "\\textbf{\\% Complete} \\\\",
            "\\midrule"]

    for stat in year_stats:
        pct = stat.transcribed_count / stat.episode_count * 100 if stat.episode_count > 0 else 0
        rows.append(
            f"{stat.year} & {format_number(stat.episode_count)} & "
            f"{format_number(stat.transcribed_count)} & {pct:.1f}\\% \\\\"
        )

    # Total
    total_episodes = sum(s.episode_count for s in year_stats)
    total_transcribed = sum(s.transcribed_count for s in year_stats)
    pct_total = total_transcribed / total_episodes * 100 if total_episodes > 0 else 0

    rows.append("\\midrule")
    rows.append(
        f"\\textbf{{Total}} & \\textbf{{{format_number(total_episodes)}}} & "
        f"\\textbf{{{format_number(total_transcribed)}}} & "
        f"\\textbf{{{pct_total:.1f}\\%}} \\\\"
    )
    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(rows)


def generate_models_table() -> str:
    """Generate table of models used."""
    rows = ["\\begin{table}[H]",
            "\\centering",
            "\\caption{Machine learning models used in the processing pipeline.}",
            "\\label{tab:models}",
            LATEX_CONFIG['table_font_size'],
            "\\begin{tabular}{llll}",
            "\\toprule",
            "\\textbf{Task} & \\textbf{Model} & \\textbf{Version} & "
            "\\textbf{Details} \\\\",
            "\\midrule"]

    for key, model in MODELS.items():
        details = f"{model.get('dimension', '')} dim" if 'dimension' in model else model['type']
        rows.append(
            f"{model['type']} & {escape_latex(model['name'])} & "
            f"\\texttt{{{escape_latex(model['version'])}}} & {details} \\\\"
        )

    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(rows)


def generate_appendix_header() -> str:
    """Generate appendix document header."""
    return dedent("""
    \\appendix
    \\section{Podcast Corpus Details}
    \\label{sec:appendix_podcasts}

    This appendix provides the complete list of podcasts included in the corpus,
    along with their chart rankings as of December 2025.
    """).strip()


def generate_podcast_table(
    channels: List[ChannelInfo],
    project_filter: Optional[str] = None
) -> str:
    """Generate the full podcast listing table."""
    filtered = [c for c in channels if not project_filter or c.project == project_filter]

    # Project display names
    display_names = {
        "Big_Channels": "United States",
        "Canadian": "Canada",
        "Europe": "European Union"
    }

    project_name = display_names.get(project_filter, project_filter) if project_filter else "All Regions"

    rows = [
        f"\\subsection{{{project_name} Podcasts}}",
        "",
        "\\begin{longtable}{p{4cm}llcp{6cm}}",
        "\\caption{Podcasts included from " + project_name + ".}",
        f"\\label{{tab:podcasts_{project_filter or 'all'}}} \\\\",
        "\\toprule",
        "\\textbf{Podcast} & \\textbf{Lang} & \\textbf{Rank} & "
        "\\textbf{Episodes} & \\textbf{Description} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{5}{c}{\\textit{Continued from previous page}} \\\\",
        "\\toprule",
        "\\textbf{Podcast} & \\textbf{Lang} & \\textbf{Rank} & "
        "\\textbf{Episodes} & \\textbf{Description} \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{5}{r}{\\textit{Continued on next page}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot"
    ]

    for ch in filtered:
        name = escape_latex(ch.name[:40] + '...' if len(ch.name) > 40 else ch.name)
        lang = ch.language.upper() if ch.language else 'EN'
        rank = str(ch.chart_rank) if ch.chart_rank else '--'
        episodes = format_number(ch.episode_count)
        desc = truncate_description(ch.description, 100)

        rows.append(f"{name} & {lang} & {rank} & {episodes} & {desc} \\\\")

    rows.append("\\end{longtable}")
    return "\n".join(rows)


def generate_summary_tables(
    channels: List[ChannelInfo],
    lang_stats: List[LanguageStats]
) -> str:
    """Generate summary tables for appendix."""

    # Summary by project
    project_counts = {}
    for ch in channels:
        project_counts[ch.project] = project_counts.get(ch.project, 0) + 1

    display_names = {
        "Big_Channels": "United States",
        "Canadian": "Canada",
        "Europe": "European Union"
    }

    rows = [
        "\\subsection{Summary Statistics}",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Number of podcasts by region.}",
        "\\label{tab:summary_by_region}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "\\textbf{Region} & \\textbf{Podcasts} \\\\",
        "\\midrule"
    ]

    for project, count in sorted(project_counts.items()):
        name = display_names.get(project, project)
        rows.append(f"{name} & {count} \\\\")

    rows.append("\\midrule")
    rows.append(f"\\textbf{{Total}} & \\textbf{{{sum(project_counts.values())}}} \\\\")
    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    # Summary by language
    rows.extend([
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Number of channels and episodes by language.}",
        "\\label{tab:summary_by_language}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "\\textbf{Language} & \\textbf{Channels} & \\textbf{Episodes} & "
        "\\textbf{Hours} \\\\",
        "\\midrule"
    ])

    for stat in lang_stats[:10]:
        lang_name = LANGUAGE_NAMES.get(stat.language, stat.language.upper())
        rows.append(
            f"{lang_name} & {stat.channel_count} & "
            f"{format_number(stat.episode_count)} & {format_hours(stat.hours)} \\\\"
        )

    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    return "\n".join(rows)


def generate_document_footer() -> str:
    """Generate document footer."""
    return dedent("""
    \\end{document}
    """).strip()
