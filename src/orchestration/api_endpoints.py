"""
API Endpoints - FastAPI endpoints for the orchestrator
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime, timezone
import logging
from sqlalchemy import text as sa_text

logger = logging.getLogger(__name__)

# Request/Response models
class TaskCallbackRequest(BaseModel):
    task_id: str
    original_task_id: str
    content_id: str
    task_type: str
    status: str
    duration: float
    result: Dict[str, Any]
    worker_id: str
    chunk_index: Optional[int] = None

class TaskCallbackResponse(BaseModel):
    status: str
    message: str

class GlobalPauseRequest(BaseModel):
    duration_seconds: int

class GlobalPauseResponse(BaseModel):
    status: str
    message: str
    paused_until: str

class WorkerStatusResponse(BaseModel):
    workers: Dict[str, Dict[str, Any]]
    active_workers: int
    total_workers: int

def create_api_endpoints(orchestrator) -> FastAPI:
    """Create FastAPI app with orchestrator endpoints"""
    
    app = FastAPI(
        title="Task Orchestrator API",
        description="Content processing task orchestration",
        version="2.0.0"
    )
    
    @app.post("/api/task_callback", response_model=TaskCallbackResponse)
    async def task_callback(request: TaskCallbackRequest, background_tasks: BackgroundTasks):
        """
        Handle task completion callbacks from workers (fire-and-forget).

        Returns immediately with 200 OK while processing continues in background.
        This eliminates callback latency and allows workers to immediately request next task.
        """
        try:
            logger.debug(f"Received callback for task {request.task_id} (original: {request.original_task_id}) "
                       f"from worker {request.worker_id}: {request.status}")

            # Convert to dict for V1 compatibility
            callback_data = request.dict()

            # Schedule background processing (non-blocking)
            background_tasks.add_task(
                orchestrator.handle_task_callback,
                callback_data
            )

            # Return immediately (< 5ms response time)
            return TaskCallbackResponse(
                status='accepted',
                message=f"Callback accepted for task {request.task_id}"
            )

        except Exception as e:
            logger.error(f"Error accepting callback for task {request.task_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/global_pause", response_model=GlobalPauseResponse)
    async def global_pause(request: GlobalPauseRequest):
        """Pause all task assignment globally"""
        try:
            paused_until = orchestrator.set_global_pause(request.duration_seconds)
            
            return GlobalPauseResponse(
                status="success",
                message=f"Task assignment paused for {request.duration_seconds} seconds",
                paused_until=paused_until.isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error setting global pause: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/workers", response_model=WorkerStatusResponse)
    async def get_workers():
        """Get status of all workers with enhanced retry information"""
        try:
            workers_status = {}
            active_count = 0
            
            for worker_id, worker in orchestrator.worker_pool.workers.items():
                # Use the orchestrator's enhanced worker status method
                if hasattr(orchestrator, 'get_worker_status'):
                    status_info = orchestrator.get_worker_status(worker_id)
                else:
                    # Fallback to basic status
                    status_info = {
                        'worker_id': worker_id,
                        'status': getattr(worker, 'status', 'unknown'),
                        'active_tasks': getattr(worker, 'active_tasks', 0),
                        'max_concurrent_tasks': getattr(worker, 'max_concurrent_tasks', 0),
                        'task_types': getattr(worker, 'task_types', []),
                        'task_counts_by_type': getattr(worker, 'get_task_counts_by_type', lambda: {})(),
                    }
                
                # Add network monitoring stats
                if hasattr(orchestrator, 'network_monitor'):
                    network_stats = orchestrator.network_monitor.get_stats()
                    worker_details = network_stats.get('worker_details', {}).get(worker_id, {})
                    if worker_details:
                        status_info['network_monitoring'] = worker_details
                
                # Add failure tracking info
                if hasattr(orchestrator, 'failure_tracker'):
                    for task_type in status_info.get('task_types', []):
                        failure_info = orchestrator.failure_tracker.get_failure_info(worker_id, task_type)
                        if failure_info['count'] > 0:
                            if 'failure_info' not in status_info:
                                status_info['failure_info'] = {}
                            status_info['failure_info'][task_type] = failure_info
                
                # Add service status if available
                if hasattr(orchestrator, 'service_manager'):
                    try:
                        services_status = await orchestrator.service_manager.get_all_services_status(worker_id)
                        if services_status:
                            status_info['services'] = {
                                name: {
                                    'status': service.status.value,
                                    'port': service.port,
                                    'models': service.models if hasattr(service, 'models') else []
                                }
                                for name, service in services_status.items()
                            }
                    except Exception as e:
                        logger.debug(f"Could not get service status for worker {worker_id}: {str(e)}")
                
                workers_status[worker_id] = status_info
                
                if status_info['status'] == 'active':
                    active_count += 1
            
            return WorkerStatusResponse(
                workers=workers_status,
                active_workers=active_count,
                total_workers=len(workers_status)
            )
            
        except Exception as e:
            logger.error(f"Error getting workers status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/workers/{worker_id}")
    async def get_worker_status(worker_id: str):
        """Get detailed status of a specific worker including retry information"""
        try:
            if hasattr(orchestrator, 'get_worker_status'):
                status = orchestrator.get_worker_status(worker_id)
                if not status:
                    raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
                
                # Add additional monitoring information
                if hasattr(orchestrator, 'network_monitor'):
                    network_stats = orchestrator.network_monitor.get_stats()
                    worker_details = network_stats.get('worker_details', {}).get(worker_id, {})
                    if worker_details:
                        status['network_monitoring'] = worker_details
                
                # Add failure tracking information for all task types
                if hasattr(orchestrator, 'failure_tracker'):
                    status['failure_tracking'] = {}
                    for task_type in status.get('task_types', []):
                        failure_info = orchestrator.failure_tracker.get_failure_info(worker_id, task_type)
                        status['failure_tracking'][task_type] = failure_info
                
                # Add service status
                if hasattr(orchestrator, 'service_manager'):
                    try:
                        services_status = await orchestrator.service_manager.get_all_services_status(worker_id)
                        if services_status:
                            status['services'] = {
                                name: {
                                    'status': service.status.value,
                                    'port': service.port,
                                    'models': service.models if hasattr(service, 'models') else []
                                }
                                for name, service in services_status.items()
                            }
                    except Exception as e:
                        logger.debug(f"Could not get service status for worker {worker_id}: {str(e)}")
                        status['services'] = {"error": str(e)}
                
                return status
            else:
                # Fallback for basic orchestrator
                if worker_id not in orchestrator.worker_pool.workers:
                    raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

                worker = orchestrator.worker_pool.workers[worker_id]
                return {
                    'worker_id': worker_id,
                    'status': getattr(worker, 'status', 'unknown'),
                    'active_tasks': getattr(worker, 'active_tasks', 0),
                    'max_concurrent_tasks': getattr(worker, 'max_concurrent_tasks', 0),
                    'task_types': getattr(worker, 'task_types', []),
                    'task_counts_by_type': getattr(worker, 'get_task_counts_by_type', lambda: {})(),
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting worker {worker_id} status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/tasks/stats")
    async def get_task_stats():
        """Get task queue statistics"""
        try:
            if hasattr(orchestrator, 'task_manager'):
                stats = await orchestrator.task_manager.get_task_stats()
                return stats
            else:
                return {"error": "Task manager not available"}
                
        except Exception as e:
            logger.error(f"Error getting task stats: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0"
        }
    
    # === Unified Scheduled Task API ===

    @app.get("/api/scheduled_tasks/status")
    async def get_scheduled_tasks_status():
        """Get status of all scheduled tasks"""
        try:
            if hasattr(orchestrator, 'scheduled_task_manager'):
                return orchestrator.scheduled_task_manager.get_status()
            else:
                return {"error": "Scheduled task manager not available"}
        except Exception as e:
            logger.error(f"Error getting scheduled tasks status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/scheduled_tasks/{task_id}/status")
    async def get_scheduled_task_status(task_id: str):
        """Get status of a specific scheduled task"""
        try:
            if hasattr(orchestrator, 'scheduled_task_manager'):
                status = orchestrator.scheduled_task_manager.get_task_status(task_id)
                if status:
                    return status
                else:
                    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            else:
                return {"error": "Scheduled task manager not available"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting task {task_id} status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/scheduled_tasks/{task_id}/trigger")
    async def trigger_scheduled_task(task_id: str):
        """Manually trigger a specific scheduled task"""
        try:
            if hasattr(orchestrator, 'scheduled_task_manager'):
                result = await orchestrator.scheduled_task_manager.trigger_task(task_id)
                return result
            else:
                return {"error": "Scheduled task manager not available"}
        except Exception as e:
            logger.error(f"Error triggering task {task_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/scheduled_tasks/{task_id}/enable")
    async def enable_scheduled_task(task_id: str, enabled: bool = True):
        """Enable or disable a specific scheduled task"""
        try:
            if hasattr(orchestrator, 'scheduled_task_manager'):
                result = orchestrator.scheduled_task_manager.set_enabled(task_id, enabled)
                return result
            else:
                return {"error": "Scheduled task manager not available"}
        except Exception as e:
            logger.error(f"Error setting task {task_id} enabled={enabled}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/services/status")
    async def get_services_status():
        """Get status of LLM server and dashboards"""
        try:
            if hasattr(orchestrator, 'service_startup'):
                status = orchestrator.service_startup.get_status()
                return status
            else:
                return {"error": "Service startup manager not available"}
        except Exception as e:
            logger.error(f"Error getting services status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/services/{service_name}/restart")
    async def restart_service(service_name: str):
        """Restart a specific service (llm_server, worker_monitoring, project_monitoring)"""
        try:
            if hasattr(orchestrator, 'service_startup'):
                success = await orchestrator.service_startup.restart_service(service_name)
                if success:
                    return {"status": "success", "message": f"Service {service_name} restarted"}
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to restart {service_name}")
            else:
                return {"error": "Service startup manager not available"}
        except Exception as e:
            logger.error(f"Error restarting service {service_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/monitoring/stats")
    async def get_monitoring_stats():
        """Get monitoring component statistics including pipeline progress"""
        try:
            from src.database.session import get_session

            # Get pipeline progress summary
            progress_summary = {}
            try:
                with get_session() as session:
                    # Quick embedding count
                    emb_result = session.execute(sa_text("""
                        SELECT
                            COUNT(*) as total,
                            COUNT(embedding) FILTER (WHERE embedding IS NOT NULL) as with_primary,
                            COUNT(embedding_alt) FILTER (WHERE embedding_alt IS NOT NULL) as with_alt
                        FROM embedding_segments
                    """)).fetchone()

                    # Quick speaker count
                    spk_result = session.execute(sa_text("""
                        SELECT
                            COUNT(*) as total,
                            COUNT(speaker_identity_id) FILTER (WHERE speaker_identity_id IS NOT NULL) as identified
                        FROM speakers
                    """)).fetchone()

                progress_summary = {
                    "embedding_segments": {
                        "total": emb_result.total,
                        "with_primary": emb_result.with_primary,
                        "with_alternative": emb_result.with_alt,
                        "primary_percent": round(100 * emb_result.with_primary / (emb_result.total or 1), 1)
                    },
                    "speakers": {
                        "total": spk_result.total,
                        "identified": spk_result.identified,
                        "percent": round(100 * spk_result.identified / (spk_result.total or 1), 1)
                    }
                }
            except Exception as pe:
                logger.warning(f"Could not get progress summary: {pe}")
                progress_summary = {"error": str(pe)}

            stats = {
                "timeout_manager": orchestrator.timeout_manager.get_stats(),
                "network_monitor": orchestrator.network_monitor.get_stats(),
                "failure_tracker": orchestrator.failure_tracker.get_stats(),
                "s3_monitor": orchestrator.s3_monitor.get_stats(),
                "reactive_assignment": orchestrator.reactive_assignment.get_stats(),
                "scheduled_tasks": orchestrator.scheduled_task_manager.get_status() if hasattr(orchestrator, 'scheduled_task_manager') else {},
                "services": orchestrator.service_startup.get_status(),
                "progress": progress_summary,
                "global_pause": {
                    "is_paused": orchestrator.global_pause_until is not None,
                    "pause_until": orchestrator.global_pause_until.isoformat() if orchestrator.global_pause_until else None
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting monitoring stats: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/failure_tracker/reset")
    async def reset_failure_tracker(worker_id: Optional[str] = None, task_type: Optional[str] = None):
        """Reset failure tracking for worker/task combinations"""
        try:
            orchestrator.failure_tracker.reset_failures(worker_id, task_type)
            return {
                "status": "success", 
                "message": f"Reset failures for worker={worker_id}, task_type={task_type}"
            }
        except Exception as e:
            logger.error(f"Error resetting failure tracker: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/timeout_manager/update")
    async def update_timeout(task_type: str, timeout_seconds: int):
        """Update timeout for a task type"""
        try:
            orchestrator.timeout_manager.update_timeout(task_type, timeout_seconds)
            return {
                "status": "success",
                "message": f"Updated timeout for {task_type} to {timeout_seconds} seconds"
            }
        except Exception as e:
            logger.error(f"Error updating timeout: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/workers/{worker_id}/restart")
    async def restart_worker(worker_id: str):
        """Restart a specific worker with enhanced retry status reporting"""
        try:
            if worker_id not in orchestrator.worker_pool.workers:
                raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

            worker = orchestrator.worker_pool.workers[worker_id]

            # Get current retry status
            retry_info = {
                'attempts_before_restart': getattr(worker, 'restart_attempts', 0),
                'max_attempts': getattr(worker, 'max_restart_attempts', 5),
                'can_restart': getattr(worker, 'should_attempt_restart', lambda: True)(),
                'backoff_remaining': getattr(worker, 'get_restart_backoff_remaining', lambda: 0)()
            }
            
            success = await orchestrator.restart_worker(worker_id)
            
            # Get updated retry status
            retry_info_after = {
                'attempts_after_restart': getattr(worker, 'restart_attempts', 0),
                'next_backoff': getattr(worker, 'get_restart_backoff_remaining', lambda: 0)()
            }
            
            if success:
                return {
                    "status": "success", 
                    "message": f"Worker {worker_id} restart succeeded",
                    "retry_info": {**retry_info, **retry_info_after}
                }
            else:
                return {
                    "status": "error", 
                    "message": f"Worker {worker_id} restart failed",
                    "retry_info": {**retry_info, **retry_info_after}
                }
                
        except Exception as e:
            logger.error(f"Error restarting worker {worker_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/workers/{worker_id}/retry/reset")
    async def reset_worker_retry_count(worker_id: str):
        """Reset retry count for a worker (useful for manual intervention)"""
        try:
            if worker_id not in orchestrator.worker_pool.workers:
                raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

            worker = orchestrator.worker_pool.workers[worker_id]

            # Store old values for response
            old_attempts = getattr(worker, 'restart_attempts', 0)
            old_backoff = getattr(worker, 'get_restart_backoff_remaining', lambda: 0)()
            
            # Reset retry tracking
            if hasattr(worker, 'restart_attempts'):
                worker.restart_attempts = 0
            if hasattr(worker, 'last_restart_attempt'):
                worker.last_restart_attempt = None
            
            return {
                "status": "success",
                "message": f"Reset retry count for worker {worker_id}",
                "before": {
                    "restart_attempts": old_attempts,
                    "backoff_remaining": old_backoff
                },
                "after": {
                    "restart_attempts": 0,
                    "backoff_remaining": 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error resetting retry count for worker {worker_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/workers/{worker_id}/retry")
    async def get_worker_retry_status(worker_id: str):
        """Get detailed retry status for a worker"""
        try:
            if worker_id not in orchestrator.worker_pool.workers:
                raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

            worker = orchestrator.worker_pool.workers[worker_id]

            retry_status = {
                'worker_id': worker_id,
                'restart_attempts': getattr(worker, 'restart_attempts', 0),
                'max_restart_attempts': getattr(worker, 'max_restart_attempts', 5),
                'last_restart_attempt': getattr(worker, 'last_restart_attempt', None),
                'can_restart': getattr(worker, 'should_attempt_restart', lambda: True)(),
                'backoff_remaining': getattr(worker, 'get_restart_backoff_remaining', lambda: 0)(),
                'restart_backoff_base': getattr(worker, 'restart_backoff_base', 60),
                'restart_backoff_max': getattr(worker, 'restart_backoff_max', 900)
            }
            
            # Format timestamp
            if retry_status['last_restart_attempt']:
                retry_status['last_restart_attempt'] = retry_status['last_restart_attempt'].isoformat()
            
            # Calculate next retry time if applicable
            if not retry_status['can_restart'] and retry_status['backoff_remaining'] > 0:
                from datetime import datetime, timezone, timedelta
                next_retry = datetime.now(timezone.utc) + timedelta(seconds=retry_status['backoff_remaining'])
                retry_status['next_retry_at'] = next_retry.isoformat()
            
            return retry_status
            
        except Exception as e:
            logger.error(f"Error getting retry status for worker {worker_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/workers/{worker_id}/services/{service_name}/restart")
    async def restart_worker_service(worker_id: str, service_name: str):
        """Restart a specific service on a worker"""
        try:
            if not hasattr(orchestrator, 'service_manager'):
                raise HTTPException(status_code=501, detail="Service management not available")

            if worker_id not in orchestrator.worker_pool.workers:
                raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
            
            if service_name == 'model_server':
                # Get task types for this worker
                worker_config = orchestrator.config['processing']['workers'].get(worker_id, {})
                task_types = worker_config.get('task_types', [])
                
                # Stop and start model server
                await orchestrator.service_manager.stop_model_server(worker_id)
                success = await orchestrator.service_manager.start_model_server(worker_id, task_types)
                
            elif service_name == 'llm_server':
                # LLM server management is disabled
                return {"status": "disabled", "message": f"LLM server management is disabled - managed independently"}
                
            elif service_name == 'task_processor':
                # Stop and start task processor
                await orchestrator.service_manager.stop_task_processor(worker_id)
                success = await orchestrator.service_manager.start_task_processor(worker_id)
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown service: {service_name}")
            
            if success:
                return {"status": "success", "message": f"Service {service_name} restarted on {worker_id}"}
            else:
                return {"status": "error", "message": f"Failed to restart {service_name} on {worker_id}"}
                
        except Exception as e:
            logger.error(f"Error restarting service {service_name} on {worker_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/daily_indexing/status")
    async def get_daily_indexing_status():
        """Get daily indexing status"""
        try:
            if not hasattr(orchestrator, 'daily_indexing'):
                raise HTTPException(status_code=501, detail="Daily indexing not available")
            
            status = orchestrator.daily_indexing.get_status()
            return status
        except Exception as e:
            logger.error(f"Error getting daily indexing status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/daily_indexing/trigger")
    async def trigger_daily_indexing():
        """Manually trigger daily indexing"""
        try:
            if not hasattr(orchestrator, 'daily_indexing'):
                raise HTTPException(status_code=501, detail="Daily indexing not available")

            # Check if already running
            if orchestrator.daily_indexing.is_running:
                return {
                    "status": "error",
                    "message": "Daily indexing is already running"
                }

            # Trigger the indexing in background
            asyncio.create_task(orchestrator.daily_indexing.run_indexing())

            return {
                "status": "success",
                "message": "Daily indexing triggered successfully"
            }
        except Exception as e:
            logger.error(f"Error triggering daily indexing: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/speaker_clustering/status")
    async def get_speaker_clustering_status():
        """Get speaker clustering status"""
        try:
            if hasattr(orchestrator, 'speaker_clustering'):
                status = await orchestrator.speaker_clustering.get_status()
                return status
            else:
                return {"error": "Speaker clustering manager not available"}
        except Exception as e:
            logger.error(f"Error getting speaker clustering status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/podcast_collection/status")
    async def get_podcast_collection_status():
        """Get podcast collection status"""
        try:
            if hasattr(orchestrator, 'podcast_collection'):
                status = await orchestrator.podcast_collection.get_status()
                return status
            else:
                return {"error": "Podcast collection manager not available"}
        except Exception as e:
            logger.error(f"Error getting podcast collection status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/progress")
    async def get_pipeline_progress():
        """
        Get progress metrics for embedding hydration and speaker labeling pipelines.

        Returns counts and percentages for:
        - Embedding segments (primary and alternative embeddings)
        - Speaker identification (by phase and overall)
        - Content processing state
        """
        try:
            from src.database.session import get_session

            with get_session() as session:
                # Embedding progress query
                embedding_result = session.execute(sa_text("""
                    SELECT
                        COUNT(*) as total_segments,
                        COUNT(embedding) FILTER (WHERE embedding IS NOT NULL) as with_primary,
                        COUNT(embedding_alt) FILTER (WHERE embedding_alt IS NOT NULL) as with_alternative,
                        COUNT(*) FILTER (WHERE embedding IS NULL) as missing_primary,
                        COUNT(*) FILTER (WHERE embedding_alt IS NULL) as missing_alternative
                    FROM embedding_segments
                """)).fetchone()

                # Speaker identification progress query
                speaker_result = session.execute(sa_text("""
                    SELECT
                        COUNT(*) as total_speakers,
                        COUNT(speaker_identity_id) FILTER (WHERE speaker_identity_id IS NOT NULL) as with_identity,
                        COUNT(*) FILTER (WHERE speaker_identity_id IS NULL) as without_identity,
                        COUNT(*) FILTER (WHERE text_evidence_status IS NOT NULL AND text_evidence_status != 'not_processed') as phase2_processed,
                        COUNT(*) FILTER (WHERE text_evidence_status = 'certain') as phase2_certain,
                        COUNT(*) FILTER (WHERE text_evidence_status = 'none') as phase2_no_evidence,
                        COUNT(*) FILTER (WHERE duration > 60) as significant_duration
                    FROM speakers
                """)).fetchone()

                # Speaker identities count
                identity_result = session.execute(sa_text("""
                    SELECT
                        COUNT(*) as total_identities,
                        COUNT(CASE WHEN primary_name IS NOT NULL AND primary_name != '' THEN 1 END) as named_identities
                    FROM speaker_identities
                """)).fetchone()

                # Content processing progress
                content_result = session.execute(sa_text("""
                    SELECT
                        COUNT(*) as total_content,
                        COUNT(*) FILTER (WHERE is_stitched = true) as stitched,
                        COUNT(*) FILTER (WHERE is_embedded = true) as embedded,
                        COUNT(*) FILTER (WHERE is_stitched = true AND is_embedded = false) as needs_embedding
                    FROM content
                    WHERE blocked_download = false AND is_duplicate = false AND is_short = false
                """)).fetchone()

            # Calculate percentages
            total_segments = embedding_result.total_segments or 1
            total_speakers = speaker_result.total_speakers or 1
            significant_speakers = speaker_result.significant_duration or 1
            total_content = content_result.total_content or 1

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "embedding": {
                    "total_segments": embedding_result.total_segments,
                    "primary": {
                        "completed": embedding_result.with_primary,
                        "missing": embedding_result.missing_primary,
                        "percent": round(100 * embedding_result.with_primary / total_segments, 2)
                    },
                    "alternative": {
                        "completed": embedding_result.with_alternative,
                        "missing": embedding_result.missing_alternative,
                        "percent": round(100 * embedding_result.with_alternative / total_segments, 2)
                    }
                },
                "speaker_identification": {
                    "total_speakers": speaker_result.total_speakers,
                    "significant_duration_speakers": speaker_result.significant_duration,
                    "with_identity": {
                        "count": speaker_result.with_identity,
                        "percent": round(100 * speaker_result.with_identity / total_speakers, 2)
                    },
                    "phase2_text_evidence": {
                        "processed": speaker_result.phase2_processed,
                        "certain": speaker_result.phase2_certain,
                        "no_evidence": speaker_result.phase2_no_evidence,
                        "percent_of_significant": round(100 * speaker_result.phase2_processed / significant_speakers, 2) if speaker_result.phase2_processed else 0
                    },
                    "identities": {
                        "total": identity_result.total_identities,
                        "named": identity_result.named_identities
                    }
                },
                "content": {
                    "total": content_result.total_content,
                    "stitched": content_result.stitched,
                    "embedded": content_result.embedded,
                    "needs_embedding": content_result.needs_embedding,
                    "percent_embedded": round(100 * content_result.embedded / total_content, 2)
                }
            }

        except Exception as e:
            logger.error(f"Error getting pipeline progress: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/progress/embedding")
    async def get_embedding_progress(project: Optional[str] = None):
        """
        Get detailed embedding progress, optionally filtered by project.

        Shows breakdown by embedding version and model.
        """
        try:
            from src.database.session import get_session

            with get_session() as session:
                if project:
                    # Filter by project
                    result = session.execute(sa_text("""
                        SELECT
                            COUNT(*) as total_segments,
                            COUNT(es.embedding) FILTER (WHERE es.embedding IS NOT NULL) as with_primary,
                            COUNT(es.embedding_alt) FILTER (WHERE es.embedding_alt IS NOT NULL) as with_alternative,
                            es.embedding_version,
                            COUNT(*) as version_count
                        FROM embedding_segments es
                        JOIN content c ON es.content_id = c.id
                        WHERE :project = ANY(c.projects)
                        GROUP BY es.embedding_version
                        ORDER BY version_count DESC
                    """), {'project': project}).fetchall()

                    total_result = session.execute(sa_text("""
                        SELECT
                            COUNT(*) as total_segments,
                            COUNT(es.embedding) FILTER (WHERE es.embedding IS NOT NULL) as with_primary,
                            COUNT(es.embedding_alt) FILTER (WHERE es.embedding_alt IS NOT NULL) as with_alternative
                        FROM embedding_segments es
                        JOIN content c ON es.content_id = c.id
                        WHERE :project = ANY(c.projects)
                    """), {'project': project}).fetchone()
                else:
                    # Global stats
                    result = session.execute(sa_text("""
                        SELECT
                            embedding_version,
                            COUNT(*) as version_count,
                            COUNT(embedding) FILTER (WHERE embedding IS NOT NULL) as with_embedding,
                            COUNT(embedding_alt) FILTER (WHERE embedding_alt IS NOT NULL) as with_alt
                        FROM embedding_segments
                        GROUP BY embedding_version
                        ORDER BY version_count DESC
                        LIMIT 20
                    """)).fetchall()

                    total_result = session.execute(sa_text("""
                        SELECT
                            COUNT(*) as total_segments,
                            COUNT(embedding) FILTER (WHERE embedding IS NOT NULL) as with_primary,
                            COUNT(embedding_alt) FILTER (WHERE embedding_alt IS NOT NULL) as with_alternative
                        FROM embedding_segments
                    """)).fetchone()

                total = total_result.total_segments or 1

                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "project": project,
                    "total_segments": total_result.total_segments,
                    "primary_embeddings": total_result.with_primary,
                    "alternative_embeddings": total_result.with_alternative,
                    "primary_percent": round(100 * total_result.with_primary / total, 2),
                    "alternative_percent": round(100 * total_result.with_alternative / total, 2),
                    "by_version": [
                        {
                            "version": row.embedding_version,
                            "count": row.version_count if hasattr(row, 'version_count') else row[1],
                        }
                        for row in result[:10]
                    ]
                }

        except Exception as e:
            logger.error(f"Error getting embedding progress: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/progress/speaker")
    async def get_speaker_progress(project: Optional[str] = None):
        """
        Get detailed speaker identification progress, optionally filtered by project.

        Shows breakdown by identification phase and status.
        """
        try:
            from src.database.session import get_session

            with get_session() as session:
                if project:
                    # Filter by project
                    result = session.execute(sa_text("""
                        SELECT
                            COUNT(*) as total_speakers,
                            COUNT(s.speaker_identity_id) FILTER (WHERE s.speaker_identity_id IS NOT NULL) as with_identity,
                            COUNT(*) FILTER (WHERE s.text_evidence_status IS NOT NULL AND s.text_evidence_status != 'not_processed') as phase2_processed,
                            COUNT(*) FILTER (WHERE s.text_evidence_status = 'certain') as phase2_certain,
                            COUNT(*) FILTER (WHERE s.text_evidence_status = 'none') as phase2_no_evidence,
                            COUNT(*) FILTER (WHERE s.duration > 60) as significant_duration,
                            COUNT(DISTINCT s.speaker_identity_id) FILTER (WHERE s.speaker_identity_id IS NOT NULL) as unique_identities
                        FROM speakers s
                        JOIN content c ON s.content_id = c.id
                        WHERE :project = ANY(c.projects)
                    """), {'project': project}).fetchone()

                    # Get status breakdown
                    status_result = session.execute(sa_text("""
                        SELECT
                            COALESCE(s.text_evidence_status, 'not_processed') as status,
                            COUNT(*) as count
                        FROM speakers s
                        JOIN content c ON s.content_id = c.id
                        WHERE :project = ANY(c.projects) AND s.duration > 60
                        GROUP BY s.text_evidence_status
                        ORDER BY count DESC
                    """), {'project': project}).fetchall()
                else:
                    # Global stats
                    result = session.execute(sa_text("""
                        SELECT
                            COUNT(*) as total_speakers,
                            COUNT(speaker_identity_id) FILTER (WHERE speaker_identity_id IS NOT NULL) as with_identity,
                            COUNT(*) FILTER (WHERE text_evidence_status IS NOT NULL AND text_evidence_status != 'not_processed') as phase2_processed,
                            COUNT(*) FILTER (WHERE text_evidence_status = 'certain') as phase2_certain,
                            COUNT(*) FILTER (WHERE text_evidence_status = 'none') as phase2_no_evidence,
                            COUNT(*) FILTER (WHERE duration > 60) as significant_duration,
                            COUNT(DISTINCT speaker_identity_id) FILTER (WHERE speaker_identity_id IS NOT NULL) as unique_identities
                        FROM speakers
                    """)).fetchone()

                    # Get status breakdown
                    status_result = session.execute(sa_text("""
                        SELECT
                            COALESCE(text_evidence_status, 'not_processed') as status,
                            COUNT(*) as count
                        FROM speakers
                        WHERE duration > 60
                        GROUP BY text_evidence_status
                        ORDER BY count DESC
                    """)).fetchall()

                total = result.total_speakers or 1
                significant = result.significant_duration or 1

                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "project": project,
                    "total_speakers": result.total_speakers,
                    "significant_duration": result.significant_duration,
                    "with_identity": {
                        "count": result.with_identity,
                        "percent": round(100 * result.with_identity / total, 2),
                        "unique_identities": result.unique_identities
                    },
                    "phase2_text_evidence": {
                        "processed": result.phase2_processed,
                        "certain": result.phase2_certain,
                        "no_evidence": result.phase2_no_evidence,
                        "percent_of_significant": round(100 * result.phase2_processed / significant, 2) if result.phase2_processed else 0
                    },
                    "by_status": [
                        {"status": row.status, "count": row.count}
                        for row in status_result
                    ]
                }

        except Exception as e:
            logger.error(f"Error getting speaker progress: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # ============================================================
    # Scheduled Tasks API (Unified Scheduler)
    # ============================================================

    @app.get("/api/scheduled_tasks")
    async def get_all_scheduled_tasks():
        """Get status of all scheduled tasks"""
        try:
            if hasattr(orchestrator, 'scheduled_task_manager'):
                return orchestrator.scheduled_task_manager.get_status()
            else:
                return {"error": "Scheduled task manager not available"}
        except Exception as e:
            logger.error(f"Error getting scheduled tasks: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/scheduled_tasks/{task_id}")
    async def get_scheduled_task(task_id: str):
        """Get status of a specific scheduled task"""
        try:
            if hasattr(orchestrator, 'scheduled_task_manager'):
                status = orchestrator.scheduled_task_manager.get_task_status(task_id)
                if status is None:
                    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
                return status
            else:
                raise HTTPException(status_code=501, detail="Scheduled task manager not available")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting scheduled task {task_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/scheduled_tasks/{task_id}/trigger")
    async def trigger_scheduled_task(task_id: str):
        """Manually trigger a scheduled task"""
        try:
            if hasattr(orchestrator, 'scheduled_task_manager'):
                result = await orchestrator.scheduled_task_manager.trigger_task(task_id)
                return result
            else:
                raise HTTPException(status_code=501, detail="Scheduled task manager not available")
        except Exception as e:
            logger.error(f"Error triggering scheduled task {task_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/scheduled_tasks/{task_id}/enable")
    async def set_scheduled_task_enabled(task_id: str, enabled: bool = True):
        """Enable or disable a scheduled task"""
        try:
            if hasattr(orchestrator, 'scheduled_task_manager'):
                result = orchestrator.scheduled_task_manager.set_enabled(task_id, enabled)
                return result
            else:
                raise HTTPException(status_code=501, detail="Scheduled task manager not available")
        except Exception as e:
            logger.error(f"Error setting scheduled task {task_id} enabled={enabled}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # === Code Deployment Endpoints ===

    @app.post("/api/deployment/deploy")
    async def deploy_code(force_restart: bool = False):
        """Deploy code to all workers"""
        try:
            if hasattr(orchestrator, 'code_deployment') and hasattr(orchestrator, 'worker_pool'):
                logger.info(f"Starting code deployment (force_restart={force_restart})")
                deployment_status = await orchestrator.code_deployment.deploy_to_all_workers(
                    orchestrator.worker_pool, force_restart=force_restart
                )
                return {
                    'status': 'completed' if deployment_status.completed_at else 'in_progress',
                    'total_workers': deployment_status.total_workers,
                    'successful_workers': deployment_status.successful_workers,
                    'failed_workers': deployment_status.failed_workers,
                    'success_rate': len(deployment_status.successful_workers) / max(1, deployment_status.total_workers),
                    'started_at': deployment_status.started_at.isoformat(),
                    'completed_at': deployment_status.completed_at.isoformat() if deployment_status.completed_at else None
                }
            else:
                raise HTTPException(status_code=501, detail="Code deployment manager not available")
        except Exception as e:
            logger.error(f"Error deploying code: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/deployment/status")
    async def get_deployment_status():
        """Get current deployment status"""
        try:
            if hasattr(orchestrator, 'code_deployment'):
                return orchestrator.code_deployment.get_deployment_status()
            else:
                raise HTTPException(status_code=501, detail="Code deployment manager not available")
        except Exception as e:
            logger.error(f"Error getting deployment status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/deployment/history")
    async def get_deployment_history(limit: int = 10):
        """Get deployment history"""
        try:
            if hasattr(orchestrator, 'code_deployment'):
                return orchestrator.code_deployment.get_deployment_history(limit)
            else:
                raise HTTPException(status_code=501, detail="Code deployment manager not available")
        except Exception as e:
            logger.error(f"Error getting deployment history: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return app