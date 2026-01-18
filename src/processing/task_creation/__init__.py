"""
Task creation module.

Provides task factory and strategies for creating pipeline tasks.
"""

from src.processing.task_creation.factory import TaskFactory
from src.processing.task_creation.strategies import (
    TaskCreationStrategy,
    DownloadTaskStrategy,
    ConvertTaskStrategy,
    TranscribeTaskStrategy,
    DiarizeTaskStrategy,
    StitchTaskStrategy,
    CleanupTaskStrategy,
)

__all__ = [
    'TaskFactory',
    'TaskCreationStrategy',
    'DownloadTaskStrategy',
    'ConvertTaskStrategy',
    'TranscribeTaskStrategy',
    'DiarizeTaskStrategy',
    'StitchTaskStrategy',
    'CleanupTaskStrategy',
]
