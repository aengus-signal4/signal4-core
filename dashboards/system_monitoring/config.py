"""
Configuration constants and loaders for the system monitoring dashboard.
"""

from pathlib import Path
import yaml

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('system_monitoring')

# API URLs
ORCHESTRATOR_API_URL = "http://localhost:8001"
LLM_BALANCER_URL = "http://localhost:8002"

# Model servers configuration (from config.yaml)
MODEL_SERVERS = {
    'worker0': {'host': '10.0.0.34', 'port': 8004, 'tiers': ['tier_1', 'tier_2']},
    'head': {'host': '10.0.0.4', 'port': 8004, 'tiers': ['tier_2']},
    'worker5': {'host': '10.0.0.209', 'port': 8004, 'tiers': ['tier_3']},
}

# Worker processors configuration
WORKER_PROCESSORS = {
    'worker0': {'host': '10.0.0.34', 'port': 8000},
    'worker1': {'host': '10.0.0.101', 'port': 8000},
    'worker2': {'host': '10.0.0.181', 'port': 8000},
    'worker3': {'host': '10.0.0.159', 'port': 8000},
    'worker4': {'host': '10.0.0.213', 'port': 8000},
    'worker5': {'host': '10.0.0.209', 'port': 8000},
    'worker6': {'host': '10.0.0.189', 'port': 8000},
}

# Log sources configuration for the Logs tab
LOG_SOURCES = {
    'Orchestrator': {
        'orchestrator': {
            'name': 'Orchestrator V2',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_orchestrator_v2.log',
            'local': True,
        },
        'task_orchestrator': {
            'name': 'Task Orchestrator',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_task_orchestrator.log',
            'local': True,
        },
        'transcribe_orchestrator': {
            'name': 'Transcribe Orchestrator',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_transcribe_orchestrator.log',
            'local': True,
        },
    },
    'Model Servers': {
        'worker6_model_server': {
            'name': 'Worker6 Model Server',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_model_server.log',
            'local': True,
        },
        'worker0_model_server': {
            'name': 'Worker0 Model Server',
            'command': 'ssh signal4@10.0.0.34 "tail -f /Users/signal4/logs/content_processing/worker0/worker0_model_server.log"',
            'local': False,
            'host': '10.0.0.34',
        },
        'worker5_model_server': {
            'name': 'Worker5 Model Server',
            'command': 'ssh signal4@10.0.0.209 "tail -f /Users/signal4/logs/content_processing/worker5/worker5_model_server.log"',
            'local': False,
            'host': '10.0.0.209',
        },
    },
    'Task Processors': {
        'worker6_processor': {
            'name': 'Worker6 Processor',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_task_processor.log',
            'local': True,
        },
        'worker0_processor': {
            'name': 'Worker0 Processor',
            'command': 'ssh signal4@10.0.0.34 "tail -f /Users/signal4/logs/content_processing/worker0/worker0_task_processor.log"',
            'local': False,
            'host': '10.0.0.34',
        },
        'worker1_processor': {
            'name': 'Worker1 Processor',
            'command': 'ssh signal4@10.0.0.101 "tail -f /Users/signal4/logs/content_processing/worker1/worker1_task_processor.log"',
            'local': False,
            'host': '10.0.0.101',
        },
        'worker2_processor': {
            'name': 'Worker2 Processor',
            'command': 'ssh signal4@10.0.0.181 "tail -f /Users/signal4/logs/content_processing/worker2/worker2_task_processor.log"',
            'local': False,
            'host': '10.0.0.181',
        },
        'worker3_processor': {
            'name': 'Worker3 Processor',
            'command': 'ssh signal4@10.0.0.159 "tail -f /Users/signal4/logs/content_processing/worker3/worker3_task_processor.log"',
            'local': False,
            'host': '10.0.0.159',
        },
        'worker4_processor': {
            'name': 'Worker4 Processor',
            'command': 'ssh signal4@10.0.0.213 "tail -f /Users/signal4/logs/content_processing/worker4/worker4_task_processor.log"',
            'local': False,
            'host': '10.0.0.213',
        },
        'worker5_processor': {
            'name': 'Worker5 Processor',
            'command': 'ssh signal4@10.0.0.209 "tail -f /Users/signal4/logs/content_processing/worker5/worker5_task_processor.log"',
            'local': False,
            'host': '10.0.0.209',
        },
    },
    'Other': {
        'embedding_hydrator': {
            'name': 'Embedding Hydrator',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_hydrate_embeddings.log',
            'local': True,
        },
        'speaker_id_orchestrator': {
            'name': 'Speaker ID Orchestrator',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_speaker_identification.orchestrator.log',
            'local': True,
        },
    },
}

# Quick status grid configuration - organized by category
QUICK_STATUS_GROUPS = {
    'Orchestrator': [
        {'name': 'Orchestrator', 'url': 'http://localhost:8001', 'endpoint': '/api/health',
         'log_cmd': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_orchestrator_v2.log'},
    ],
    'LLMs': [
        {'name': 'Worker6', 'url': 'http://10.0.0.189:8004', 'endpoint': '/health',
         'log_cmd': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_model_server.log'},
        {'name': 'Worker0', 'url': 'http://10.0.0.34:8004', 'endpoint': '/health',
         'log_cmd': 'ssh signal4@10.0.0.34 "tail -f /Users/signal4/logs/content_processing/worker0/worker0_model_server.log"'},
        {'name': 'Worker5', 'url': 'http://10.0.0.209:8004', 'endpoint': '/health',
         'log_cmd': 'ssh signal4@10.0.0.209 "tail -f /Users/signal4/logs/content_processing/worker5/worker5_model_server.log"'},
    ],
    'Workers': [
        {'name': 'W0', 'url': 'http://10.0.0.34:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.34 "tail -f /Users/signal4/logs/content_processing/worker0/worker0_task_processor.log"'},
        {'name': 'W1', 'url': 'http://10.0.0.101:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.101 "tail -f /Users/signal4/logs/content_processing/worker1/worker1_task_processor.log"'},
        {'name': 'W2', 'url': 'http://10.0.0.181:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.181 "tail -f /Users/signal4/logs/content_processing/worker2/worker2_task_processor.log"'},
        {'name': 'W3', 'url': 'http://10.0.0.159:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.159 "tail -f /Users/signal4/logs/content_processing/worker3/worker3_task_processor.log"'},
        {'name': 'W4', 'url': 'http://10.0.0.213:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.213 "tail -f /Users/signal4/logs/content_processing/worker4/worker4_task_processor.log"'},
        {'name': 'W5', 'url': 'http://10.0.0.209:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.209 "tail -f /Users/signal4/logs/content_processing/worker5/worker5_task_processor.log"'},
        {'name': 'W6', 'url': 'http://10.0.0.189:8000', 'endpoint': '/tasks',
         'log_cmd': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_task_processor.log'},
    ],
    'Scheduled': [
        {'name': 'Podcast Index', 'task_id': 'podcast_index_download'},
        {'name': 'YouTube Index', 'task_id': 'youtube_index_download'},
        {'name': 'Embeddings', 'task_id': 'embedding_hydration'},
        {'name': 'Speaker ID 1', 'task_id': 'speaker_id_phase1'},
        {'name': 'Speaker ID 2', 'task_id': 'speaker_id_phase2'},
        {'name': 'Tone', 'task_id': 'tone_hydration'},
        {'name': 'Cache 30d', 'task_id': 'cache_refresh_30d'},
        {'name': 'Cache 7d', 'task_id': 'cache_refresh_7d'},
    ],
}

# Pipeline stage colors
PIPELINE_COLORS = {
    'Pending Download': '#e74c3c',
    'Downloaded': '#95a5a6',
    'Audio Extracted': '#e67e22',
    'Transcribed': '#3498db',
    'Diarized & Transcribed': '#9b59b6',
    'Stitched': '#2ecc71',
    'Segment Embeddings': '#155724',
    'Blocked': '#c0392b',
    'Skipped (Short)': '#7f8c8d',
}

# Task type colors
TASK_COLORS = {
    'download': '#1f77b4',
    'convert': '#ff7f0e',
    'diarize': '#d62728',
    'transcribe': '#2ca02c',
    'stitch': '#9467bd',
    'cleanup_and_compress': '#e377c2',
    'segment_embeddings': '#17becf',
}


def load_config() -> dict:
    """Load configuration from yaml file"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}
