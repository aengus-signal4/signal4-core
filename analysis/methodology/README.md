# Methodology Section Generator

This package generates LaTeX methodology documentation for the Nature paper on podcast transcription corpus analysis.

## Overview

The generator queries the PostgreSQL database to extract live corpus statistics and produces:

1. **`methodology.tex`** - Main methodology section with:
   - Corpus overview and statistics tables
   - Podcast selection criteria
   - Data collection pipeline documentation
   - Audio processing and transcription methodology
   - Speaker diarization and attribution
   - Quality assurance procedures
   - Model version documentation

2. **`appendix_podcasts.tex`** - Supplementary appendix with:
   - Summary tables by region and language
   - Complete podcast listing (700+ channels)
   - Chart rankings as of December 2025

3. **`statistics_summary.txt`** - Plain text summary for quick reference

## Prerequisites

### Python Dependencies

This module uses the shared `src.utils.config` for database configuration and `.env` loading. No additional dependencies beyond the core project requirements.

```bash
# Required packages (already in core requirements)
pip install psycopg2-binary pyyaml python-dotenv
```

### LaTeX Distribution (for PDF compilation)

**macOS:**
```bash
brew install --cask mactex
# Or minimal version:
brew install --cask basictex
sudo tlmgr install booktabs longtable multirow caption float
```

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
# Or minimal:
sudo apt-get install texlive-latex-base texlive-latex-extra
```

**Windows:**
- Install MiKTeX from https://miktex.org/

## Quick Start

### Option 1: Shell Script (Recommended)

```bash
# From methodology directory
cd core/analysis/methodology
./generate_pdf.sh

# With optional sections
./generate_pdf.sh --include-speaker-id --include-classification
```

This will:
1. Generate LaTeX files from database
2. Compile to PDF using pdflatex
3. Clean up auxiliary files

### Option 2: Python Only (LaTeX files without PDF)

```bash
# From methodology directory
cd core/analysis/methodology
python3 generate_methodology.py \
    --output-dir output/methodology \
    --verbose
```

Then compile manually:
```bash
cd output/methodology
pdflatex methodology.tex
pdflatex methodology.tex  # Run twice for references
pdflatex appendix_podcasts.tex
pdflatex appendix_podcasts.tex
```

## Command Line Options

```
usage: generate_methodology.py [-h] [--output-dir OUTPUT_DIR]
                               [--start-date START_DATE] [--end-date END_DATE]
                               [--projects PROJECTS [PROJECTS ...]]
                               [--exclude-languages EXCLUDE_LANGUAGES [EXCLUDE_LANGUAGES ...]]
                               [--chart-month CHART_MONTH]
                               [--include-speaker-id] [--include-classification]
                               [--title TITLE] [--authors AUTHORS] [--verbose]

Options:
  --output-dir          Output directory (default: output/methodology)
  --start-date          Start date YYYY-MM-DD (default: 2018-01-01)
  --end-date            End date YYYY-MM-DD (default: 2025-12-31)
  --projects            Projects to include (default: Big_Channels Canadian Europe)
  --exclude-languages   Languages to exclude (default: ru uk ua ca)
  --chart-month         Chart rankings month (default: 2025-12)
  --include-speaker-id  Include speaker identification section
  --include-classification  Include theme classification section
  --title               Document title
  --authors             Author names
  --verbose, -v         Verbose output
```

## Examples

### Default Generation (2018-2025, EU languages only)

```bash
./generate_pdf.sh
```

### Full Pipeline Documentation

```bash
./generate_pdf.sh \
    --include-speaker-id \
    --include-classification
```

### Custom Date Range

```bash
python3 generate_methodology.py \
    --output-dir output/methodology \
    --start-date 2020-01-01 \
    --end-date 2025-12-31 \
    --verbose
```

### US and Canada Only

```bash
python3 generate_methodology.py \
    --output-dir output/methodology \
    --projects Big_Channels Canadian \
    --verbose
```

## Output Structure

```
output/methodology/
├── methodology.tex          # Main document LaTeX source
├── methodology.pdf          # Compiled PDF (if pdflatex available)
├── appendix_podcasts.tex    # Appendix LaTeX source
├── appendix_podcasts.pdf    # Compiled PDF (if pdflatex available)
└── statistics_summary.txt   # Plain text statistics
```

## Current Dataset Statistics

As of January 2026 (2018-01-01 to 2025-12-31, EU languages only):

| Region | Channels | Episodes | Transcribed | Hours |
|--------|----------|----------|-------------|-------|
| United States | 311 | 226,458 | 205,919 (90.9%) | 164,041 |
| Canada | 138 | 82,301 | 72,981 (88.7%) | 37,292 |
| European Union | 366 | 140,880 | 26,093 (18.5%)* | 10,961 |
| **Total** | **815** | **449,639** | **304,993 (67.8%)** | **212,294** |

*Europe transcription is ongoing

## Language Filtering

By default, the generator excludes non-EU languages:
- `ru` - Russian
- `uk` - Ukrainian
- `ca` - Catalan (regional language)

Included EU languages: English, French, German, Spanish, Italian, Portuguese, Dutch, Polish, Romanian, Greek, Czech, Hungarian, Swedish, Danish, Finnish, Slovak, Bulgarian, Croatian, Lithuanian, Latvian, Estonian, Slovenian, Maltese, Irish.

## Customization

### Adding New Sections

Edit `latex_templates.py` to add new section generators. The modular design allows toggling sections via command-line flags.

### Updating Model Information

Edit `config.py` to update the `MODELS` dictionary with new model versions.

### Changing Table Formatting

Modify `LATEX_CONFIG` in `config.py` to adjust table styling, font sizes, and description truncation length.

## Troubleshooting

### Database Connection Error

The module loads database credentials from `core/.env` via `src.utils.config`. Ensure:
1. The `.env` file exists with `POSTGRES_PASSWORD` set
2. PostgreSQL is running and accessible

Test connection:
```bash
# Uses credentials from .env
cd core/analysis/methodology
python3 -c "from config import DB_CONFIG; print('Host:', DB_CONFIG['host'])"
```

### LaTeX Compilation Errors

Run pdflatex manually to see detailed errors:
```bash
cd output/methodology
pdflatex methodology.tex
```

Common issues:
- Missing packages: Install via `tlmgr install <package>`
- Special characters: Check for unescaped `&`, `%`, `$`, `#`, `_` in podcast descriptions

### Memory Issues with Large Tables

The appendix uses `longtable` for pagination. If compilation is slow, consider:
- Reducing the number of podcasts per table
- Using `\small` or `\footnotesize` for table content

## File Structure

```
core/analysis/methodology/
├── __init__.py              # Package initialization
├── config.py                # Configuration constants
├── db_queries.py            # Database query functions
├── latex_templates.py       # LaTeX generation templates
├── generate_methodology.py  # Main CLI script
├── generate_pdf.sh          # Shell script for PDF generation
└── README.md                # This file
```
