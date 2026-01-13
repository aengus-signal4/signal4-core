import logging
from logging import handlers  # Add explicit import for handlers
from pathlib import Path
import socket
from typing import Optional, Dict
import os
from datetime import datetime, timezone
import sys
from .ip_utils import get_worker_ip

# Cache for loggers to avoid duplicate creation
_logger_cache: Dict[str, logging.Logger] = {}


def load_config():
    """Load config - uses centralized config module.

    Kept for backward compatibility with imports.
    """
    from .config import load_config as _load_config
    return _load_config()


def get_worker_name() -> str:
    """Get the worker name based on the IP address.

    Uses centralized node_utils for consistency across codebase.
    """
    try:
        # Use centralized implementation (lazy import to avoid circular deps)
        from .node_utils import get_worker_name as _get_worker_name
        name = _get_worker_name()
        return name if name else socket.gethostname()
    except Exception:
        # Fallback to hostname if anything fails
        try:
            return socket.gethostname()
        except:
            return "unknown-worker"

class LogCaptureHandler(logging.Handler):
    """Handler that captures logs and redirects them through our custom formatters"""
    def __init__(self, worker_logger):
        super().__init__()
        self.worker_logger = worker_logger
        
    def emit(self, record):
        try:
            # Create a new record with our format
            msg = self.format(record)
            # Log through our worker logger
            self.worker_logger.info(msg)
        except Exception:
            self.handleError(record)

def capture_third_party_logs(worker_logger):
    """Capture and redirect logs from third-party libraries"""
    # Create capture handler
    capture_handler = LogCaptureHandler(worker_logger)
    capture_handler.setFormatter(WorkerLogFormatter())
    
    # Capture logs from common third-party libraries
    for name in ['speechbrain', 's3_storage', 'whisper', 'torch', 'numpy', 'youtube_dl']:
        logger = logging.getLogger(name)
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add our capture handler
        logger.addHandler(capture_handler)
        logger.propagate = False

class ErrorHandlingFileHandler(logging.FileHandler):
    """File handler that suppresses errors when writing"""
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(filename)
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass  # Ignore directory creation errors
            
        try:
            super().__init__(filename, mode, encoding, delay)
        except Exception:
            # If we can't open the file, create a fallback handler to console
            self.stream = sys.stderr
            self.mode = mode
            self.encoding = encoding
            
    def emit(self, record):
        try:
            # Check if Python is shutting down
            if not sys or not sys.modules:
                return
                
            msg = self.format(record)
            try:
                self.stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                # If write fails, try to reopen the file
                try:
                    if self.stream and self.stream != sys.stderr:
                        self.stream.close()
                    self.stream = self._open()
                    self.stream.write(msg + self.terminator)
                    self.flush()
                except Exception:
                    # If reopening fails, write to stderr
                    if self.stream != sys.stderr:
                        self.stream = sys.stderr
                        self.stream.write(msg + self.terminator)
                        self.flush()
        except Exception:
            pass  # Silently continue on any other errors

class RotatingFileHandlerWithCompression(handlers.RotatingFileHandler):
    """Rotating file handler that compresses old log files"""
    def emit(self, record):
        try:
            # Check if Python is shutting down
            if not sys or not sys.modules:
                return
            super().emit(record)
        except Exception:
            pass  # Silently continue on any errors

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self.rotation_filename(self.baseFilename + ".1.gz")
            if os.path.exists(dfn):
                os.remove(dfn)
            # Compress the current log file
            import gzip
            with open(self.baseFilename, 'rb') as f_in:
                with gzip.open(dfn, 'wb') as f_out:
                    f_out.writelines(f_in)
        self.mode = 'w'
        self.stream = self._open()

class TaskLogFormatter(logging.Formatter):
    """Formatter for task-level logs (important events that should be centrally logged and shown)"""
    def format(self, record):
        try:
            # Get worker name for task logs
            worker_name = get_worker_name()
            
            # Format timestamp
            timestamp = datetime.fromtimestamp(record.created, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            
            # Format based on record type
            if hasattr(record, 'task_event'):
                # Task completion format
                task_id = getattr(record, 'task_id', 'unknown')
                content_id = getattr(record, 'content_id', 'none')
                duration = getattr(record, 'duration', 0.0)
                component = getattr(record, 'component', 'unknown')
                publish_date = getattr(record, 'publish_date', None)
                session_task_count = getattr(record, 'session_task_count', None)
                session_task_limit = getattr(record, 'session_task_limit', None)
                process_task_count = getattr(record, 'process_task_count', None)
                chunk_index = getattr(record, 'chunk_index', None)
                
                # Format date and counts
                date_str = f" [{publish_date:%Y-%m-%d}]" if publish_date else ""
                session_str = f" [Session: {session_task_count}/{session_task_limit}]" if session_task_count is not None and session_task_limit is not None else ""
                process_str = f" [Process: {process_task_count}]" if process_task_count is not None else ""
                chunk_str = f" [Chunk: {chunk_index}]" if chunk_index is not None else ""
                
                return f"{timestamp} [{worker_name}] [{component}] Task completed: {content_id}{date_str}{chunk_str} ({duration:.1f}s){session_str}{process_str}"
            else:
                # Other important task events
                component = getattr(record, 'component', 'unknown')
                return f"{timestamp} [{worker_name}] [{component}] {record.getMessage()}"
        except Exception:
            return record.getMessage()

class WorkerLogFormatter(logging.Formatter):
    """Formatter for detailed worker-level logs (debug/info messages for troubleshooting)"""
    def format(self, record):
        try:
            # Get worker name
            worker_name = get_worker_name()
            
            # Format timestamp
            timestamp = datetime.fromtimestamp(record.created, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get logger name without worker prefix
            logger_name = record.name.split('.')[-1] if '.' in record.name else record.name
            
            return f"{timestamp} [{worker_name}.{logger_name}] [{record.levelname}] {record.getMessage()}"
        except Exception:
            return record.getMessage()

def setup_task_logger(task_type: str) -> logging.Logger:
    """Set up task-level logger for important events that should be centrally logged and shown
    
    Args:
        task_type: Type of task (e.g. 'transcribe', 'extract_audio')
    """
    try:
        worker_name = get_worker_name()
        task_type = task_type.replace('worker.', '')
        logger_name = f"task.{worker_name}.{task_type}"
        
        # Return cached logger if it exists
        if logger_name in _logger_cache:
            return _logger_cache[logger_name]

        # Set up central log directory
        config = load_config()
        central_log_dir = Path(config['logging']['base_path'])
        
        # Create all parent directories if they don't exist
        try:
            central_log_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(central_log_dir, 0o755)  # Ensure directory is readable/writable
        except Exception as e:
            print(f"Warning: Could not create/chmod log directory {central_log_dir}: {e}")
            
        # Set up worker-specific directory
        worker_log_dir = central_log_dir / worker_name
        try:
            worker_log_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(worker_log_dir, 0o755)  # Ensure directory is readable/writable
        except Exception as e:
            print(f"Warning: Could not create/chmod worker log directory {worker_log_dir}: {e}")
        
        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
        # Determine the log file name with worker prefix
        log_file_name = f"{worker_name}_{task_type}.log"
        central_log_path = worker_log_dir / log_file_name
        
        # Add central file handler
        try:
            fh = RotatingFileHandlerWithCompression(
                str(central_log_path),
                maxBytes=50*1024*1024,
                backupCount=10
            )
            fh.setFormatter(TaskLogFormatter())
            logger.addHandler(fh)
        except Exception as e:
            print(f"Warning: Could not create file handler for {central_log_path}: {e}")
        
        # Add console handler for task events
        ch = logging.StreamHandler()
        ch.setFormatter(TaskLogFormatter())
        logger.addHandler(ch)
        
        # Cache and return
        _logger_cache[logger_name] = logger
        return logger
        
    except Exception as e:
        # Fallback to basic console logging
        fallback = logging.getLogger(f"fallback.task.{worker_name}.{task_type}")
        fallback.setLevel(logging.INFO)
        if not fallback.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(message)s'))
            fallback.addHandler(ch)
        return fallback

def setup_worker_logger(worker_type: str) -> logging.Logger:
    """Set up worker-level logger for detailed debug/info messages
    
    Args:
        worker_type: Type of worker (e.g. 'transcribe', 'extract_audio')
    """
    try:
        worker_name = get_worker_name()
        if not worker_type.startswith('worker.'):
            worker_type = f"worker.{worker_type}"
            
        logger_name = f"{worker_name}.{worker_type}"
        
        # Return cached logger if it exists
        if logger_name in _logger_cache:
            return _logger_cache[logger_name]
        
        # Set up local log directory
        config = load_config()
        log_dir = Path(config['logging']['base_path'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # Ensure the worker directory exists
        worker_log_dir = log_dir / worker_name
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine the log file name - prepend worker name to file
        log_file_name = f"{worker_name}_{worker_type.replace('worker.', '')}.log"
        log_path = worker_log_dir / log_file_name
        
        # Add rotating file handler for detailed logging
        fh = RotatingFileHandlerWithCompression(
            str(log_path),
            maxBytes=10*1024*1024,
            backupCount=5
        )
        fh.setFormatter(WorkerLogFormatter())
        logger.addHandler(fh)
        
        # Add console handler that only shows task completion events
        ch = logging.StreamHandler()
        ch.setFormatter(TaskLogFormatter())
        ch.addFilter(lambda record: hasattr(record, 'task_event'))
        logger.addHandler(ch)
        
        # Capture third-party logs
        capture_third_party_logs(logger)
        
        # Cache and return
        _logger_cache[logger_name] = logger
        return logger
        
    except Exception as e:
        # Fallback to basic console logging
        fallback = logging.getLogger(f"fallback.worker.{worker_type}")
        fallback.setLevel(logging.INFO)
        if not fallback.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(message)s'))
            fallback.addHandler(ch)
        return fallback

def log_task_completion(task_type: str, task_id: str, content_id: str, duration: float, 
                     publish_date: Optional[datetime] = None, 
                     session_task_count: Optional[int] = None, 
                     session_task_limit: Optional[int] = None,
                     process_task_count: Optional[int] = None,
                     chunk_index: Optional[int] = None):
    """Log task completion to central log and console
    
    Args:
        task_type: Type of task (e.g. 'transcribe', 'extract_audio')
        task_id: ID of the task
        content_id: ID of the content being processed
        duration: Duration of task execution in seconds
        publish_date: Optional publish date of the content
        session_task_count: Optional count of tasks completed in current session
        session_task_limit: Optional task limit for current session
        process_task_count: Optional count of total tasks completed in this process
        chunk_index: Optional index of the chunk being processed
    """
    try:
        logger = setup_task_logger(task_type)
        extra = {
            'task_event': True,
            'task_id': task_id,
            'content_id': content_id,
            'duration': duration,
            'component': task_type,
            'publish_date': publish_date,
            'session_task_count': session_task_count,
            'session_task_limit': session_task_limit,
            'process_task_count': process_task_count,
            'chunk_index': chunk_index
        }
        logger.info("", extra=extra)
    except Exception as e:
        print(f"Error logging task completion: {str(e)}")

# Update aliases
setup_completion_logger = setup_task_logger  # For backwards compatibility
setup_indexer_logger = lambda name: setup_worker_logger(f"indexer.{name}")
setup_process_logger = lambda name: setup_worker_logger(f"process.{name}")

# Create main logger instance
logger = setup_worker_logger('main') 