"""
Model Server Configuration
Defines model requirements and configurations for different task types
"""
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    type: str
    memory_required_mb: int
    load_time_seconds: int
    gpu_required: bool = True
    endpoint_path: str = ""

# Define which models are needed for each task type
MODEL_REQUIREMENTS: Dict[str, List[str]] = {
    'transcribe': ['whisper-large-v3-turbo'],
    'diarize': ['pyannote-diarization-3.1'],
    'stitch': ['pyannote-speaker-embedding', 'sentence-transformer'],  # For speaker embeddings and LLM coherence
    'segment_embeddings': ['sentence-transformer'],
    'convert': [],  # No models needed
    'download_youtube': [],  # No models needed
    'download_podcast': [],  # No models needed
}

# Define model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    'whisper-large-v3-turbo': ModelConfig(
        name='whisper-large-v3-turbo',
        type='mlx_whisper',
        memory_required_mb=5000,
        load_time_seconds=30,
        gpu_required=True,
        endpoint_path='/transcribe'
    ),
    'pyannote-diarization-3.1': ModelConfig(
        name='pyannote-diarization-3.1',
        type='pyannote_diarization',
        memory_required_mb=3000,
        load_time_seconds=45,
        gpu_required=True,
        endpoint_path='/diarize'
    ),
    'pyannote-speaker-embedding': ModelConfig(
        name='pyannote-speaker-embedding',
        type='pyannote_embedding',
        memory_required_mb=2000,
        load_time_seconds=20,
        gpu_required=True,
        endpoint_path='/embed_speaker'
    ),
    'sentence-transformer': ModelConfig(
        name='sentence-transformer',
        type='sentence_transformer',
        memory_required_mb=1500,
        load_time_seconds=15,
        gpu_required=False,  # Can run on CPU
        endpoint_path='/embed_text'
    )
}

def get_required_models_for_worker(task_types: List[str]) -> Set[str]:
    """Get all models required for a worker based on its task types"""
    required_models = set()
    for task_type in task_types:
        models = MODEL_REQUIREMENTS.get(task_type, [])
        required_models.update(models)
    return required_models

def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name)

def estimate_memory_requirements(models: List[str]) -> int:
    """Estimate total memory requirements for a list of models in MB"""
    total_memory = 0
    for model_name in models:
        config = get_model_config(model_name)
        if config:
            total_memory += config.memory_required_mb
    return total_memory

def has_gpu_models(models: List[str]) -> bool:
    """Check if any of the models require GPU"""
    for model_name in models:
        config = get_model_config(model_name)
        if config and config.gpu_required:
            return True
    return False