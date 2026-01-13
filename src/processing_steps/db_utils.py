#!/usr/bin/env python3
import sys
from pathlib import Path

from src.utils.paths import get_project_root
import asyncio
import logging
import argparse
from typing import Dict, Optional, List
from datetime import datetime
import time

# Add project root to Python path
sys.path.append(str(get_project_root()))

from src.database.session import get_session
from src.database.models import TaskQueue, Content, ContentChunk
from src.database.state_manager import StateManager
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('db_utils')

class ProcessingStepDB:
    """Database utilities for processing steps"""
    
    def __init__(self):
        """Initialize database utilities"""
        self.state_manager = StateManager()
        
    async def update_task_status(self, task_id: str, status: str, result: Dict, error: Optional[str] = None) -> bool:
        """Update task status in database"""
        try:
            with get_session() as session:
                task = session.query(TaskQueue).filter_by(id=int(task_id)).first()
                if not task:
                    logger.error(f"Task {task_id} not found")
                    return False
                    
                task.status = status
                task.result = result
                if error:
                    task.error = error
                task.completed_at = datetime.now()
                session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error updating task status: {str(e)}")
            return False
            
    async def get_next_task(self, task_type: str) -> Optional[Dict]:
        """Get next pending task of specified type"""
        try:
            with get_session() as session:
                task = session.query(TaskQueue).filter_by(
                    status='pending',
                    task_type=task_type
                ).order_by(
                    TaskQueue.priority.desc(),
                    TaskQueue.created_at.asc()
                ).first()
                
                if task:
                    # Mark as processing
                    task.status = 'processing'
                    task.started_at = datetime.now()
                    session.commit()
                    
                    return {
                        'id': task.id,
                        'content_id': task.content_id,
                        'task_type': task.task_type,
                        'input_data': task.input_data or {}
                    }
                    
                return None
                
        except Exception as e:
            logger.error(f"Error getting next task: {str(e)}")
            return None
            
    # Wrapper methods for StateManager
    
    async def update_download_status(self, content_id: str, success: bool, error: Optional[str] = None) -> bool:
        """Update content download status"""
        return await self.state_manager.update_download_status(content_id, success, error)
        
    async def update_conversion_status(self, content_id: str, total_chunks: int, chunk_data: List[Dict]) -> bool:
        """Update content conversion status"""
        return await self.state_manager.update_conversion_status(content_id, total_chunks, chunk_data)
        
    async def update_transcription_status(self, content_id: str, success: bool) -> bool:
        """Update content transcription status"""
        return await self.state_manager.update_transcription_status(content_id, success)
        
    async def update_diarization_status(self, content_id: str, success: bool, result_data: Optional[Dict] = None) -> bool:
        """Update content diarization status"""
        return await self.state_manager.update_diarization_complete_status(content_id, success, result_data)
        
    async def update_content_status(self, content_id: str, **kwargs) -> bool:
        """Update general content status fields"""
        return await self.state_manager.update_content_status(content_id, **kwargs)

def add_db_args(parser: argparse.ArgumentParser):
    """Add database-related arguments to an argument parser"""
    parser.add_argument('--database', action='store_true',
                       help='Update task and content status in database')
    parser.add_argument('--next', action='store_true',
                       help='Get and process next pending task from queue')
    return parser 