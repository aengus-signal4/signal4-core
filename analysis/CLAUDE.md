# Analysis Repository

Ad-hoc analyses, client requests, and exploratory data work for Signal4.

## Purpose

This repository contains:
- **Client deliverables** - Custom analyses requested by clients
- **Ad-hoc investigations** - One-off data explorations
- **Research notebooks** - Experimental analyses and prototypes
- **Reports** - Generated reports and visualizations

## Structure

```
analysis/
├── projects/           # Individual analysis projects
│   └── YYYY-MM-name/   # Project folders by date
├── shared/             # Shared utilities and helpers
├── templates/          # Notebook and report templates
└── outputs/            # Generated reports (gitignored)
```

## Project Organization

Each project should be in its own folder under `projects/`:

```
projects/
└── 2024-01-client-sentiment/
    ├── README.md           # Project description and findings
    ├── notebooks/          # Jupyter notebooks
    ├── scripts/            # Python scripts
    ├── data/               # Local data files (gitignored)
    └── outputs/            # Generated outputs
```

## Data Access

Analysis scripts can read from:
- **PostgreSQL** (av_content database) - read-only access
- **MinIO/S3** (av-content bucket) - read-only access
- **Backend API** - for processed data and search

```python
# Example database connection
import os
from sqlalchemy import create_engine

engine = create_engine(
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
```

## Key Principles

- **Read-only** - Never modify production data
- **Reproducible** - Document steps and use version control
- **Self-contained** - Each project should be independent
- **Clean outputs** - Don't commit large data files or outputs

## Environment Setup

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
# Edit .env with actual credentials
```

## Security

- **Never commit credentials** - Use `.env` files only
- **Never commit client data** - Keep in gitignored `data/` folders
- **Anonymize outputs** - Remove PII from shared reports
