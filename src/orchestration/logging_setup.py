"""
Logging Setup - Centralized logging configuration for the orchestrator
"""
import logging
from pathlib import Path
from typing import Tuple, Optional

def setup_orchestrator_logging(log_dir: Optional[Path] = None) -> Tuple[logging.Logger, logging.Logger, logging.Logger]:
    """
    Set up dedicated loggers for orchestrator operations

    Returns:
        Tuple of (completion_logger, error_logger, run_logger)
    """
    if log_dir is None:
        log_dir = Path("/Users/signal4/logs/content_processing")

    log_dir.mkdir(parents=True, exist_ok=True)

    # Console formatter for task events
    console_formatter = logging.Formatter('%(asctime)s [%(name)s] %(message)s')

    # Error logger - file + console
    error_logger = logging.getLogger('orchestrator_errors')
    error_logger.setLevel(logging.ERROR)
    error_logger.propagate = False
    error_log_file = log_dir / "orchestrator_errors.log"
    error_file_handler = logging.FileHandler(error_log_file)
    error_file_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter('%(asctime)s %(message)s')
    error_file_handler.setFormatter(error_formatter)
    error_logger.addHandler(error_file_handler)
    # Add console handler for errors (with immediate flush)
    error_console_handler = logging.StreamHandler()
    error_console_handler.setLevel(logging.ERROR)
    error_console_handler.setFormatter(console_formatter)
    error_console_handler.stream.reconfigure(line_buffering=True)  # Flush after each line
    error_logger.addHandler(error_console_handler)

    # Completion logger - file + console
    completion_logger = logging.getLogger('task_completion')
    completion_logger.setLevel(logging.INFO)
    completion_logger.propagate = False
    completion_log_file = log_dir / "task_completions.log"
    completion_file_handler = logging.FileHandler(completion_log_file)
    completion_file_handler.setLevel(logging.INFO)
    completion_formatter = logging.Formatter('%(asctime)s [%(message)s')
    completion_file_handler.setFormatter(completion_formatter)
    completion_logger.addHandler(completion_file_handler)
    # Add console handler for completions (with immediate flush)
    completion_console_handler = logging.StreamHandler()
    completion_console_handler.setLevel(logging.INFO)
    completion_console_handler.setFormatter(console_formatter)
    completion_console_handler.stream.reconfigure(line_buffering=True)  # Flush after each line
    completion_logger.addHandler(completion_console_handler)

    # Hourly run logger (file only - not needed on console)
    run_log_file = log_dir / "orchestrator_runs.log"
    run_logger = logging.getLogger('orchestrator_runs')
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = False
    run_file_handler = logging.FileHandler(run_log_file)
    run_file_handler.setLevel(logging.INFO)
    run_formatter = logging.Formatter('%(asctime)s - %(message)s')
    run_file_handler.setFormatter(run_formatter)
    run_logger.addHandler(run_file_handler)

    return completion_logger, error_logger, run_logger


def configure_noise_suppression():
    """
    Suppress noisy loggers while keeping important messages visible.
    Call this after setting up orchestrator logging.
    """
    # Suppress background loop noise (task assignment chatter)
    logging.getLogger('src.automation.background_loops').setLevel(logging.WARNING)

    # Suppress reactive assignment info logs
    logging.getLogger('src.orchestration.reactive_assignment').setLevel(logging.WARNING)

    # Suppress timeout manager info logs (keep warnings for actual timeouts)
    logging.getLogger('src.orchestration.timeout_manager').setLevel(logging.WARNING)

    # Suppress failure tracker info logs (keep warnings)
    logging.getLogger('src.orchestration.failure_tracker').setLevel(logging.WARNING)

    # Suppress uvicorn access logs (HTTP request noise)
    logging.getLogger('uvicorn.access').setLevel(logging.ERROR)

    # Suppress task queue cache debug/info
    logging.getLogger('src.orchestration.task_queue_cache').setLevel(logging.WARNING)

    # Suppress task manager routine operations
    logging.getLogger('src.orchestration.task_manager').setLevel(logging.WARNING)

    # Suppress asyncssh connection chatter
    logging.getLogger('asyncssh').setLevel(logging.WARNING)


class TaskLogger:
    """Handles task-specific logging operations"""
    
    def __init__(self, completion_logger: logging.Logger, error_logger: logging.Logger, main_logger: logging.Logger):
        self.completion_logger = completion_logger
        self.error_logger = error_logger
        self.main_logger = main_logger
    
    def log_completion(self, worker_id: str, task_type: str, content_id: str,
                      chunk_index: Optional[int] = None, time_taken: float = 0.0):
        """Log task completion to dedicated file and console"""
        try:
            chunk_str = f" (chunk {chunk_index})" if chunk_index is not None else ""
            message = f"[{worker_id}] [{task_type}] Task completed: {content_id}{chunk_str} ({time_taken:.1f}s)"
            # Only log to completion_logger (has both file and console handlers)
            # Don't also log to main_logger to avoid duplicate console output
            self.completion_logger.info(message)
        except Exception as e:
            self.main_logger.error(f"Error logging task completion: {str(e)}", exc_info=True)
    
    def log_error(self, worker_id: str, task_type: str, content_id: str, error_details: str,
                  chunk_index: Optional[int] = None, time_taken: float = 0.0):
        """Log task error to dedicated file and console"""
        try:
            chunk_str = f" (chunk {chunk_index})" if chunk_index is not None else ""
            message = f"[{worker_id}] [{task_type}] Task error: {content_id}{chunk_str} Error: {error_details} ({time_taken:.1f}s)"
            # Only log to error_logger (has both file and console handlers)
            # Don't also log to main_logger to avoid duplicate console output
            self.error_logger.error(message)
        except Exception as e:
            self.main_logger.error(f"Error logging task error to file: {str(e)}", exc_info=True)