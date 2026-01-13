"""
Configuration constants for the methodology generator.
"""

import sys
from datetime import date
from pathlib import Path
from typing import List, Dict, Set

# Add core/src to path for imports
_script_dir = Path(__file__).parent
_core_src = _script_dir.parent.parent / "src"
if str(_core_src) not in sys.path:
    sys.path.insert(0, str(_core_src.parent))

from src.utils.config import load_config as load_yaml_config, get_credential

# Load config from YAML (with .env loading)
_config = load_yaml_config()

# Database connection (from config.yaml with env substitution)
DB_CONFIG = {
    "host": _config.get("database", {}).get("host", "10.0.0.4"),
    "port": _config.get("database", {}).get("port", 5432),
    "database": _config.get("database", {}).get("database", "av_content"),
    "user": _config.get("database", {}).get("user", "signal4"),
    "password": _config.get("database", {}).get("password") or get_credential("POSTGRES_PASSWORD", "")
}

# Default date range for paper
DEFAULT_START_DATE = date(2018, 1, 1)
DEFAULT_END_DATE = date(2025, 12, 31)

# Projects to include
DEFAULT_PROJECTS = ["Big_Channels", "Canadian", "Europe"]

# EU Member State language codes (ISO 639-1)
# These are the languages we INCLUDE for the EU focus
EU_LANGUAGES: Set[str] = {
    "en",  # English (Ireland, Malta)
    "fr",  # French (France, Belgium, Luxembourg)
    "de",  # German (Germany, Austria, Belgium, Luxembourg)
    "es",  # Spanish (Spain)
    "it",  # Italian (Italy)
    "pt",  # Portuguese (Portugal)
    "nl",  # Dutch (Netherlands, Belgium)
    "pl",  # Polish (Poland)
    "ro",  # Romanian (Romania)
    "el",  # Greek (Greece, Cyprus)
    "cs",  # Czech (Czech Republic)
    "hu",  # Hungarian (Hungary)
    "sv",  # Swedish (Sweden, Finland)
    "da",  # Danish (Denmark)
    "fi",  # Finnish (Finland)
    "sk",  # Slovak (Slovakia)
    "bg",  # Bulgarian (Bulgaria)
    "hr",  # Croatian (Croatia)
    "lt",  # Lithuanian (Lithuania)
    "lv",  # Latvian (Latvia)
    "et",  # Estonian (Estonia)
    "sl",  # Slovenian (Slovenia)
    "mt",  # Maltese (Malta)
    "ga",  # Irish (Ireland)
}

# Languages to EXCLUDE (non-EU or regional)
EXCLUDED_LANGUAGES: Set[str] = {
    "ru",  # Russian (not EU)
    "uk",  # Ukrainian (not EU)
    "ca",  # Catalan (regional language, not separate country)
}

# Country mappings for projects
PROJECT_COUNTRIES: Dict[str, List[str]] = {
    "Big_Channels": ["US"],
    "Canadian": ["CA"],
    "Europe": ["FR", "DE", "ES", "IT", "PT", "NL", "PL", "RO", "GR", "CZ",
               "HU", "SE", "DK", "FI", "SK", "BG", "HR", "LT", "LV", "EE",
               "SI", "MT", "IE", "AT", "BE", "LU", "CY"]
}

# Language display names
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ro": "Romanian",
    "el": "Greek",
    "cs": "Czech",
    "hu": "Hungarian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "sl": "Slovenian",
    "mt": "Maltese",
    "ga": "Irish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ca": "Catalan",
}

# Processing pipeline stages (for configurable documentation)
PIPELINE_STAGES = {
    "discovery": {
        "name": "Podcast Discovery",
        "file": "src/ingestion/chart_collector.py",
        "required": True,
    },
    "enrichment": {
        "name": "Metadata Enrichment",
        "file": "src/ingestion/podcast_enricher.py",
        "required": True,
    },
    "indexing": {
        "name": "Episode Indexing",
        "file": "src/ingestion/podcast_indexer.py",
        "required": True,
    },
    "download": {
        "name": "Audio Download",
        "file": "src/processing_steps/download_podcast.py",
        "required": True,
    },
    "convert": {
        "name": "Audio Conversion",
        "file": "src/processing_steps/convert.py",
        "required": True,
    },
    "transcribe": {
        "name": "Transcription",
        "file": "src/processing_steps/transcribe_whisper.py",
        "required": True,
    },
    "diarize": {
        "name": "Speaker Diarization",
        "file": "src/processing_steps/diarize_pyannote.py",
        "required": True,
    },
    "stitch": {
        "name": "Speaker Attribution",
        "file": "src/processing_steps/stitch.py",
        "required": True,
    },
    "segment": {
        "name": "Semantic Segmentation",
        "file": "src/processing_steps/segment_embeddings.py",
        "required": True,
    },
    "speaker_id": {
        "name": "Speaker Identification",
        "file": "src/speaker_identification/orchestrator.py",
        "required": False,  # Optional section
    },
    "classification": {
        "name": "Theme Classification",
        "file": "src/classification/semantic_theme_classifier.py",
        "required": False,  # Optional section
    },
}

# Model versions for reproducibility
MODELS = {
    "transcription": {
        "name": "Whisper Large v3 Turbo",
        "version": "mlx-community/whisper-large-v3-turbo",
        "type": "Speech-to-Text",
    },
    "diarization": {
        "name": "PyAnnote Speaker Diarization",
        "version": "pyannote/speaker-diarization-3.1",
        "type": "Speaker Diarization",
    },
    "embedding_primary": {
        "name": "Qwen3 Embedding",
        "version": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 1024,
        "type": "Text Embedding",
    },
    "embedding_similarity": {
        "name": "XLM-RoBERTa",
        "version": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        "dimension": 768,
        "type": "Semantic Similarity",
    },
    "speaker_embedding": {
        "name": "PyAnnote Embedding",
        "version": "pyannote/embedding",
        "type": "Speaker Embedding",
    },
}

# LaTeX formatting
LATEX_CONFIG = {
    "document_class": "article",
    "packages": [
        "booktabs",
        "longtable",
        "multirow",
        "caption",
        "hyperref",
        "xcolor",
        "graphicx",
        "float",
    ],
    "table_font_size": "\\small",
    "max_description_length": 150,  # Characters to truncate descriptions
}
