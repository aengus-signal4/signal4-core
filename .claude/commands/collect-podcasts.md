# Podcast Chart Collection Pipeline

Collect, enrich, and classify podcast charts from Podstatus.com for active projects.

## Instructions

Run through the podcast collection pipeline interactively. Before starting, gather required information from the user.

### Step 1: Determine Target Month

Ask the user which month to collect (default: current month). Format: YYYY-MM

### Step 2: Determine Scope

Ask the user which project(s) to collect for. The active projects and their relevant countries are:

| Project | Countries | Classification Focus |
|---------|-----------|---------------------|
| **CPRMV** | `us` | Political, news, current affairs (Canadian/US extremism focus) |
| **Big_Channels** | `us, ca` | Political, news, current affairs |
| **Canadian** | `ca` | Political, news, current affairs |
| **Health** | `us, ca, gb, au` | Health, wellness, alternative health |
| **Finance** | `us, ca, gb` | Finance, business, economics |
| **Europe** | `fr, de, gb, es, it, nl, se, no, dk, fi, pt, pl, cz, ro, hu, gr, ua` | Political, news, current affairs |
| **Anglosphere** | `us, gb, ca, au, nz, ie, in, ng, za, ph, ke` | Political, news, current affairs (English-speaking) |

Options:
1. **All active projects** - Collect for all enabled projects (comprehensive)
2. **Specific project(s)** - User selects which project(s)
3. **Custom countries** - User specifies country codes directly

### Step 3: Determine Phases to Run

Ask which phases to execute:

1. **Phase 1: Collect** - Scrape charts from Podstatus.com (required for new data)
2. **Phase 2: Enrich** - Add RSS URLs and metadata via PodcastIndex API
3. **Phase 3: Classify** - LLM classification for relevance (requires project selection)
4. **All phases** - Run collect + enrich (classify requires manual project selection)

### Step 4: Execute Pipeline

Based on user selections, construct and run the appropriate commands:

```bash
# Phase 1: Collect charts
python -m src.ingestion.podcast_pipeline --phase collect --month {MONTH} --countries {COUNTRIES}

# Phase 2: Enrich with metadata
python -m src.ingestion.podcast_pipeline --phase enrich --month {MONTH}

# Phase 3: Classify for a specific project
python -m src.ingestion.podcast_pipeline --phase classify --month {MONTH} --project {PROJECT} --countries {COUNTRIES} --max-rank 100
```

Or run collect + enrich together:
```bash
python -m src.ingestion.podcast_pipeline --phase all --month {MONTH} --countries {COUNTRIES}
```

### Step 5: Summary and Next Steps

After execution, summarize:
- Number of charts collected
- Number of podcasts enriched
- Number classified as relevant (if classify phase ran)

Suggest next steps:
- Review classified podcasts: `streamlit run dashboards/podcast_review.py`
- Export for consideration: `python src/ingestion/export_top_podcasts.py --relevant-only --countries {COUNTRIES}`

## Country Code Reference

Common country codes for reference:
- **North America**: `us` (USA), `ca` (Canada), `mx` (Mexico)
- **Europe**: `gb` (UK), `fr` (France), `de` (Germany), `es` (Spain), `it` (Italy), `nl` (Netherlands), `se` (Sweden), `no` (Norway), `dk` (Denmark), `fi` (Finland), `pt` (Portugal), `pl` (Poland), `cz` (Czech), `ro` (Romania), `hu` (Hungary), `gr` (Greece), `ua` (Ukraine), `ie` (Ireland)
- **Asia-Pacific**: `au` (Australia), `nz` (New Zealand), `jp` (Japan), `kr` (S. Korea), `in` (India), `ph` (Philippines)
- **South America**: `br` (Brazil), `ar` (Argentina), `cl` (Chile)
- **Africa**: `ng` (Nigeria), `za` (South Africa), `ke` (Kenya)

## Classification Types by Project

When running Phase 3 (classify), use these classification types:

| Project | Classification Type |
|---------|---------------------|
| CPRMV, Big_Channels, Canadian, Europe, Anglosphere | `political, current affairs or news` |
| Health | `health, wellness, medicine, or alternative health` |
| Finance | `finance, business, economics, or investing` |

## Example Session

User: `/collect-podcasts`

Claude: "Which month should I collect charts for?"
- Current month (2025-01) [Recommended]
- Previous month (2024-12)
- Other (specify)

User: "Current month"

Claude: "Which project scope?"
- All active projects
- Big_Channels (US, CA)
- Canadian (CA only)
- Europe (17 European countries)
- Anglosphere (11 English-speaking countries)
- Custom countries

User: "Big_Channels"

Claude: "Which phases to run?"
- Collect only (Phase 1)
- Collect + Enrich (Phases 1-2) [Recommended]
- Full pipeline with classification (Phases 1-3)

User: "Full pipeline"

Claude: *Runs commands and reports results*
