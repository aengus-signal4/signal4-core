# Podcast Ingestion System

This directory contains the podcast collection, classification, enrichment, and export pipeline for discovering and evaluating new podcast sources.

## Overview

The podcast ingestion system collects top podcasts from chart data (Podstatus.com), enriches them with metadata, classifies them using LLM, and exports curated lists for manual review and addition to project sources.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Collection                          │
│  chart_collector.py: Scrape charts → podcast_charts table       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Data Enrichment                           │
│  podcast_pipeline.py --phase enrich: Add RSS URLs, metadata     │
│  → podcast_metadata table (rss_url, description, etc.)          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Classification                           │
│  classify_all_podcasts.py: Analyze relevance for politics/news  │
│  → podcast_metadata.meta_data->classification                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Export & Review                          │
│  export_top_podcasts.py: Generate curated lists for review      │
│  → projects/for_consideration.csv                               │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

### `podcast_metadata`
Stores core podcast information:
- `podcast_name`, `creator`, `description`
- `rss_url` (added during enrichment)
- `language`, `categories`, `episode_count`
- `meta_data` (JSONB) - contains LLM classification:
  ```json
  {
    "classification": {
      "is_relevant": true,
      "type": "politics, news, current events...",
      "reason": "Justification for classification",
      "confidence": 0.95
    }
  }
  ```

### `podcast_charts`
Tracks podcast rankings across platforms:
- `podcast_id` → references `podcast_metadata`
- `month`, `platform`, `country`, `category`
- `rank`, `chart_key`
- Enables tracking: "Top 50 in US News category for Oct 2025"

## Pipeline Scripts

### 1. Chart Collection

**Script:** `chart_collector.py`
**Usage:** Via `podcast_pipeline.py`

```bash
# Collect all charts for October 2025
python src/ingestion/podcast_pipeline.py --phase collect --month 2025-10

# Collect specific countries
python src/ingestion/podcast_pipeline.py --phase collect --month 2025-10 --countries us ca uk
```

**What it does:**
- Scrapes podcast charts from Podstatus.com
- Supports: Apple Podcasts, Spotify
- Collects: US, CA, UK, and 20+ other countries
- Stores rankings in `podcast_charts` table
- Creates `podcast_metadata` entries (without enrichment)

**Output:**
- Raw podcast names and rankings
- Chart appearances tracked per podcast
- No RSS URLs yet (collected during enrichment)

---

### 2. Enrichment

**Script:** `podcast_enricher.py`
**Usage:** Via `podcast_pipeline.py`

```bash
# Enrich all podcasts from October 2025 charts
python src/ingestion/podcast_pipeline.py --phase enrich --month 2025-10

# Batch size and concurrency control
python src/ingestion/podcast_pipeline.py --phase enrich --month 2025-10 --batch-size 100 --max-concurrent 5

# Force re-enrichment
python src/ingestion/podcast_pipeline.py --phase enrich --month 2025-10 --force-reenrich
```

**What it does:**
- Searches for podcasts on Podcast Index API
- Retrieves RSS feed URLs
- Fetches additional metadata (description, episode count, etc.)
- Updates `podcast_metadata` with enriched data
- Marks `last_enriched` timestamp

**Requirements:**
- Podcast Index API key (set in environment)
- Network access to podcast RSS feeds

**Output:**
- RSS URLs added to `podcast_metadata.rss_url`
- Descriptions, episode counts, categories populated
- Enables downstream classification

---

### 3. Classification

**Script:** `classify_all_podcasts.py`
**Usage:** Direct execution

```bash
# Classify all unenriched podcasts
python scripts/classify_all_podcasts.py

# Classify specific month
python scripts/classify_all_podcasts.py --month 2025-10

# Re-classify all podcasts
python scripts/classify_all_podcasts.py --force-reclassify

# Dry run (preview classifications)
python scripts/classify_all_podcasts.py --dry-run
```

**What it does:**
- Uses LLM (OpenAI/Claude) to analyze podcast descriptions
- Determines relevance for: "politics, news, current events, social issues, policy"
- Generates classification with confidence score and reasoning
- Stores in `podcast_metadata.meta_data->classification`

**Classification Output:**
```json
{
  "is_relevant": true,
  "type": "politics, news, current events, social issues, or policy discussion",
  "reason": "Detailed explanation of why this podcast is/isn't relevant",
  "confidence": 0.95,
  "classified_at": "2025-10-06T12:34:56"
}
```

**Use Cases:**
- Filter relevant podcasts for politics-focused projects
- Prioritize high-confidence matches
- Understand reasoning behind classifications

---

### 4. Export for Review

**Script:** `export_top_podcasts.py`
**Usage:** Direct execution with flexible CLI

```bash
# Basic: Export top 200 from US/CA for October 2025
python src/ingestion/export_top_podcasts.py

# Export only LLM-relevant podcasts
python src/ingestion/export_top_podcasts.py --relevant-only

# Top 50 ranked podcasts only
python src/ingestion/export_top_podcasts.py --max-rank 50 --limit 50

# Specific countries
python src/ingestion/export_top_podcasts.py --countries us uk de fr

# Custom output location
python src/ingestion/export_top_podcasts.py --output /path/to/output.csv

# Different month
python src/ingestion/export_top_podcasts.py --month 2025-11

# Exclude different source projects
python src/ingestion/export_top_podcasts.py --exclude-sources Big_channels Canadian CPRMV

# Include podcasts without RSS (not yet enriched)
python src/ingestion/export_top_podcasts.py --include-no-rss

# Verbose logging
python src/ingestion/export_top_podcasts.py --verbose
```

**What it does:**
- Queries top podcasts from `podcast_charts` joined with `podcast_metadata`
- Filters out podcasts already in specified source projects (by RSS URL)
- Includes LLM classification data if available
- Exports to CSV in sources.csv-compatible format

**Output CSV Format:**
```csv
channel_name,description,podcast,language,author,category,llm_classification,llm_justification
The Daily,The biggest stories...,https://feeds...,en,NYT,2025-10-rank1-News,True,"Explicit focus on news..."
```

**Category Format:** `{month}-rank{rank}-{categories}`
- Example: `2025-10-rank1-News`
- Example: `2025-10-rank15-Politics,News,Commentary`
- Quickly shows: month collected, best rank achieved, topics

**CLI Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--month` | Month to query (YYYY-MM) | `2025-10` |
| `--countries` | Country codes (space-separated) | `us ca` |
| `--limit` | Maximum podcasts to export | `200` |
| `--max-rank` | Only include podcasts ranked ≤ this | None (all ranks) |
| `--min-chart-appearances` | Minimum chart appearances | `1` |
| `--relevant-only` | Only LLM-marked relevant podcasts | `False` |
| `--include-no-rss` | Include unenriched podcasts | `False` |
| `--exclude-sources` | Projects to exclude RSS URLs from | `Big_channels Canadian` |
| `--output` | Output CSV path | `projects/for_consideration.csv` |
| `--projects-dir` | Projects directory location | Auto-detect |
| `--verbose` | Enable verbose logging | `False` |

---

## Complete Workflow Example

### Monthly Podcast Discovery Workflow

```bash
# 1. Collect charts for current month
python src/ingestion/podcast_pipeline.py --phase collect --month 2025-10

# 2. Enrich with RSS URLs and metadata
python src/ingestion/podcast_pipeline.py --phase enrich --month 2025-10

# 3. Classify for relevance (politics/news focus)
python scripts/classify_all_podcasts.py --month 2025-10

# 4. Export top relevant podcasts for review
python src/ingestion/export_top_podcasts.py --relevant-only --max-rank 100 --limit 50

# 5. Review projects/for_consideration.csv
# 6. Manually add selected podcasts to project sources
```

### Quarterly Refresh Workflow

```bash
# Collect last 3 months
for month in 2025-10 2025-11 2025-12; do
    python src/ingestion/podcast_pipeline.py --phase collect --month $month
    python src/ingestion/podcast_pipeline.py --phase enrich --month $month
done

# Classify all
python scripts/classify_all_podcasts.py

# Export best performers across 3 months
python src/ingestion/export_top_podcasts.py --limit 500 --max-rank 25
```

### Expansion to New Countries

```bash
# Collect charts from European countries
python src/ingestion/podcast_pipeline.py \
    --phase collect \
    --month 2025-10 \
    --countries de fr es it nl

# Enrich and classify
python src/ingestion/podcast_pipeline.py --phase enrich --month 2025-10
python scripts/classify_all_podcasts.py

# Export German news podcasts
python src/ingestion/export_top_podcasts.py \
    --countries de \
    --relevant-only \
    --output projects/german_podcasts.csv
```

---

## Output Files

### for_consideration.csv

**Location:** `projects/for_consideration.csv`

**Purpose:** Curated list of top podcasts for manual review before adding to project sources.

**Format:** Compatible with `projects/{ProjectName}/sources.csv`

**Columns:**
1. `channel_name` - Podcast name
2. `description` - Full description for context
3. `podcast` - RSS feed URL (required for ingestion)
4. `language` - Language code (en, fr, de, etc.)
5. `author` - Creator/publisher
6. `category` - Format: `{month}-rank{best_rank}-{topics}`
   - Shows when collected, best ranking, and relevant topics
   - Example: `2025-10-rank3-Politics,News`
7. `llm_classification` - `True`/`False`/empty
   - `True` = LLM identified as relevant for politics/news
   - `False` = LLM identified as not relevant
   - Empty = Not yet classified
8. `llm_justification` - LLM reasoning for classification
   - Explains why podcast is/isn't relevant
   - Useful for manual review decisions

**Usage:**
1. Review CSV, sort by `llm_classification=True` or low `rank`
2. Read `llm_justification` for context
3. Add selected podcasts to appropriate project sources:
   ```bash
   # Manually copy selected rows to:
   projects/Big_channels/sources.csv
   projects/Canadian/sources.csv
   projects/{YourProject}/sources.csv
   ```

---

## Configuration

### Environment Variables

```bash
# Podcast Index API (for enrichment)
PODCAST_INDEX_API_KEY=your_key_here
PODCAST_INDEX_API_SECRET=your_secret_here

# LLM API (for classification)
OPENAI_API_KEY=your_key_here
# OR
ANTHROPIC_API_KEY=your_key_here
```

### Database Connection

Uses standard database connection from `src/database/session.py`:
- Configured in `config/config.yaml`
- PostgreSQL with `podcast_metadata` and `podcast_charts` tables
- Connection managed via SQLAlchemy

### Logging

All scripts use `src/utils/logger.py`:
- Worker logger setup for consistent formatting
- Log level controlled via script arguments (`--verbose`)
- Logs to console and file (if configured)

---

## Troubleshooting

### Chart Collection Issues

**Problem:** No charts collected
```bash
# Check if Podstatus.com is accessible
curl -I https://podstatus.com

# Try specific country
python src/ingestion/podcast_pipeline.py --phase collect --month 2025-10 --countries us

# Check database
psql -d av_content -c "SELECT COUNT(*) FROM podcast_charts WHERE month='2025-10';"
```

**Problem:** Rate limiting / timeouts
```bash
# Increase delay between requests
# Edit chart_collector.py: delay_range=(10, 15)  # Slower but safer
```

### Enrichment Issues

**Problem:** No RSS URLs found
```bash
# Check Podcast Index API credentials
echo $PODCAST_INDEX_API_KEY

# Test with smaller batch
python src/ingestion/podcast_pipeline.py --phase enrich --month 2025-10 --batch-size 10

# Check API status
curl -H "X-Auth-Key: $PODCAST_INDEX_API_KEY" https://api.podcastindex.org/api/1.0/search/byterm?q=test
```

**Problem:** Some podcasts not enriched
- Not all podcasts are in Podcast Index database
- Use `--include-no-rss` flag in export to see unenriched podcasts
- May need manual research for RSS URLs

### Classification Issues

**Problem:** LLM classification failing
```bash
# Check API key
echo $OPENAI_API_KEY

# Test with dry run
python scripts/classify_all_podcasts.py --dry-run

# Reduce batch size
python scripts/classify_all_podcasts.py --batch-size 5
```

**Problem:** Classifications seem incorrect
- Review `llm_justification` column for reasoning
- Adjust classification prompt in `classify_all_podcasts.py`
- Use `--force-reclassify` to re-run with updated prompt

### Export Issues

**Problem:** Fewer podcasts than expected
```bash
# Check what's being filtered
python src/ingestion/export_top_podcasts.py --verbose

# Include unenriched
python src/ingestion/export_top_podcasts.py --include-no-rss

# Increase rank threshold
python src/ingestion/export_top_podcasts.py --max-rank 200
```

**Problem:** Duplicate RSS URLs
- RSS URL filtering is working correctly
- Podcast is already in your sources
- Check: `projects/Big_channels/sources.csv` or `projects/Canadian/sources.csv`

---

## Database Queries

Useful queries for monitoring the pipeline:

```sql
-- Chart collection status
SELECT month, COUNT(*) as total_charts,
       COUNT(DISTINCT podcast_id) as unique_podcasts
FROM podcast_charts
GROUP BY month
ORDER BY month DESC;

-- Enrichment status
SELECT
    COUNT(*) as total,
    COUNT(rss_url) as enriched,
    COUNT(last_enriched) as enriched_timestamp
FROM podcast_metadata;

-- Classification status
SELECT
    meta_data->>'classification'->'is_relevant' as relevant,
    COUNT(*) as count
FROM podcast_metadata
WHERE meta_data IS NOT NULL
GROUP BY relevant;

-- Top podcasts by chart appearances
SELECT
    pm.podcast_name,
    COUNT(DISTINCT pc.chart_key) as appearances,
    MIN(pc.rank) as best_rank
FROM podcast_metadata pm
JOIN podcast_charts pc ON pm.id = pc.podcast_id
WHERE pc.month = '2025-10'
GROUP BY pm.id, pm.podcast_name
ORDER BY appearances DESC
LIMIT 20;

-- LLM-relevant podcasts from recent charts
SELECT
    pm.podcast_name,
    pm.creator,
    MIN(pc.rank) as best_rank,
    pm.meta_data->'classification'->>'reason' as reason
FROM podcast_metadata pm
JOIN podcast_charts pc ON pm.id = pc.podcast_id
WHERE pc.month = '2025-10'
  AND pm.meta_data->'classification'->>'is_relevant' = 'true'
GROUP BY pm.id
ORDER BY best_rank;
```

---

## Best Practices

### Monthly Collection Cadence

Run collection on the **1st-5th** of each month:
```bash
# Collect previous month's charts
LAST_MONTH=$(date -d "last month" +%Y-%m)
python src/ingestion/podcast_pipeline.py --phase collect --month $LAST_MONTH
```

### Incremental Enrichment

Only enrich new podcasts to save API calls:
```bash
# Enrichment skips already-enriched podcasts by default
python src/ingestion/podcast_pipeline.py --phase enrich --month 2025-10
```

### Classification Efficiency

Run classification **after** enrichment completes:
```bash
# Don't classify until RSS URLs are available
# Bad: Classify immediately after collection
# Good: Collect → Enrich → Classify
```

### Review Workflow

1. Export with `--relevant-only` first for high-priority candidates
2. Review LLM justifications for accuracy
3. Export without `--relevant-only` for broader discovery
4. Cross-reference with existing sources to avoid duplicates

### Source Management

When adding podcasts to sources:
```bash
# 1. Copy selected rows from for_consideration.csv
# 2. Remove llm_classification and llm_justification columns
# 3. Add to appropriate project sources.csv
# 4. Commit to git with descriptive message

git add projects/Big_channels/sources.csv
git commit -m "Add 10 new podcasts from Oct 2025 charts"
```

---

## Future Enhancements

Potential improvements to the pipeline:

1. **Automated Source Addition**
   - Flag high-confidence podcasts (rank ≤ 10, LLM confidence ≥ 0.95)
   - Auto-add to sources with review flag

2. **Multi-Language Classification**
   - Separate LLM prompts for different languages
   - Language-specific relevance criteria

3. **Historical Trending**
   - Track podcast rank changes month-over-month
   - Identify rising/falling podcasts
   - Alert on sudden popularity spikes

4. **Category Expansion**
   - Classify for multiple project types (tech, business, culture)
   - Multi-label classification
   - Project-specific relevance scoring

5. **Quality Metrics**
   - Track episode frequency and consistency
   - Analyze description quality
   - Evaluate creator credibility

6. **Automated Testing**
   - Validate RSS feed accessibility
   - Check episode availability
   - Test download success rates

---

## Support

For issues or questions:
1. Check logs: `logs/worker_{worker_name}.log`
2. Review this documentation
3. Check database state with SQL queries above
4. Consult main project documentation: `CLAUDE.md`
