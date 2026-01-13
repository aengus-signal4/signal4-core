"""Database session management."""

import os
import socket
import subprocess
import psycopg2
import time
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySession
from sqlalchemy.exc import OperationalError
from contextlib import contextmanager
from src.utils.logger import logger, setup_worker_logger, get_worker_ip
from src.utils.config import load_config  # Centralized config with env loading

class Session:
    """Database session manager that handles connection configuration and pooling."""
    
    _instance = None
    _engine: Optional[Engine] = None
    _session_factory = None
    
    def __new__(cls):
        """Ensure singleton pattern for session manager."""
        if cls._instance is None:
            cls._instance = super(Session, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize session manager if not already initialized."""
        if self._initialized:
            return
            
        self.logger = setup_worker_logger('database')
        self.config = self._load_config()
        
        # Create engine with pooling if enabled
        if self.config['pool']['enabled']:
            self.logger.info("Initializing database connection pool")
            
            # Log connection details
            self.logger.info(f"Database host: {self.config['host']}")
            self.logger.info(f"Database port: {self.config['port']}")
            self.logger.info(f"Database name: {self.config['database']}")
            self.logger.info(f"Pool size: {self.config['pool']['size']}")
            self.logger.info(f"Max overflow: {self.config['pool']['max_overflow']}")
            
            # Test connectivity before creating engine
            if not self._test_connectivity(self.config['host'], self.config['port']):
                self.logger.error("Failed to establish database connectivity")
                raise ConnectionError("Could not establish database connection")
            
            connection_url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            self.logger.info(f"Creating engine with URL: {connection_url.replace(self.config['password'], '****')}")
            
            try:
                self._engine = create_engine(
                    connection_url,
                    pool_size=self.config['pool']['size'],
                    max_overflow=self.config['pool']['max_overflow'],
                    pool_timeout=self.config['pool']['timeout'],
                    pool_recycle=self.config['pool']['recycle'],
                    pool_pre_ping=self.config['pool'].get('pre_ping', True),
                    connect_args={
                        'connect_timeout': self.config['connection']['timeout'],
                        'application_name': self.config['connection']['application_name'],
                        'client_encoding': self.config['connection']['client_encoding'],
                        'options': f"-c statement_timeout={self.config['connection'].get('statement_timeout', 30000)}"
                    }
                )
                self.logger.info("Successfully created database engine with connection pool")
            except Exception as e:
                self.logger.error(f"Failed to create database engine: {str(e)}")
                raise
        else:
            # Create basic engine without pooling
            self.logger.info("Creating database engine without pooling")
            connection_url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            self.logger.info(f"Creating engine with URL: {connection_url.replace(self.config['password'], '****')}")
            
            try:
                self._engine = create_engine(
                    connection_url,
                    connect_args={
                        'connect_timeout': self.config['connection']['timeout'],
                        'application_name': self.config['connection']['application_name'],
                        'client_encoding': self.config['connection']['client_encoding']
                    }
                )
                self.logger.info("Successfully created basic database engine")
            except Exception as e:
                self.logger.error(f"Failed to create database engine: {str(e)}")
                raise
        
        # Test connection with retry
        max_retries = 2
        retry_delay = 5.0
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying database connection test (attempt {attempt + 1}/{max_retries + 1}) after {retry_delay}s delay...")
                    time.sleep(retry_delay)
                
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))  # Test basic connection
                    self.logger.info("Successfully tested database connection")
                    break  # Success, exit retry loop
            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(f"Failed to test database connection after {max_retries + 1} attempts: {str(e)}")
                    raise
                else:
                    self.logger.warning(f"Database connection test failed (attempt {attempt + 1}): {str(e)}")
        
        # Create session factory
        self._session_factory = sessionmaker(bind=self._engine)
        self._initialized = True
        self.logger.info("Database session manager initialization complete")
        
    def _get_local_ip(self) -> str:
        """Get local IP address using multiple methods."""
        return get_worker_ip()
    
    def _test_connectivity(self, host: str, port: int, max_retries: int = 2, retry_delay: float = 5.0) -> bool:
        """Test network connectivity to database host with retry mechanism."""
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying connectivity test to {host}:{port} (attempt {attempt + 1}/{max_retries + 1}) after {retry_delay}s delay...")
                    time.sleep(retry_delay)
                else:
                    self.logger.info(f"Testing network connectivity to {host}:{port}...")
                
                # Try netcat to test port
                nc_result = subprocess.run(
                    ['nc', '-zv', '-w', '2', host, str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if nc_result.returncode == 0:
                    self.logger.info(f"Successfully connected to port {port} on {host}")
                    return True
                else:
                    self.logger.warning(f"Could not connect to port {port} on {host} (nc returned {nc_result.returncode})")
                    
                    # Try a direct socket test as fallback
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(2)
                        result = sock.connect_ex((host, port))
                        sock.close()
                        if result == 0:
                            self.logger.info(f"Socket test succeeded for {host}:{port}")
                            return True
                        else:
                            self.logger.warning(f"Socket test failed for {host}:{port} (error {result})")
                    except Exception as socket_e:
                        self.logger.warning(f"Socket test exception for {host}:{port}: {socket_e}")
                    
                    # If this was the last attempt, log failure
                    if attempt == max_retries:
                        self.logger.error(f"All connectivity attempts failed for {host}:{port}")
                        
            except Exception as e:
                self.logger.warning(f"Error testing network connectivity (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return False
        
        # All attempts failed
        return False
    
    def _is_head_node(self) -> bool:
        """Determine if we are running on the head node by checking worker type in config."""
        try:
            # Get worker IP
            local_ip = self._get_local_ip()

            # Load worker information from config (with env substitution)
            full_config = load_config()
            workers_config = full_config.get('processing', {}).get('workers', {})

            # Find any worker with type "head" and check if our IP matches
            for worker_id, worker_config in workers_config.items():
                if worker_config.get('type') == 'head':
                    eth_ip = worker_config.get('eth', '')
                    wifi_ip = worker_config.get('wifi', '')
                    
                    # If our IP matches either head node IP, we're on the head node
                    if local_ip == eth_ip or local_ip == wifi_ip:
                        self.logger.info(f"Identified as head node (worker {worker_id}) with IP: {local_ip}")
                        return True
            
            self.logger.info(f"Not the head node (IP: {local_ip})")
            return False
        except Exception as e:
            self.logger.warning(f"Error determining if head node: {e}")
            return False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate database configuration, implementing fallback for worker nodes."""
        # Load full config with env variable substitution (handles ${POSTGRES_PASSWORD} etc.)
        full_config = load_config()

        # Get database specific config
        db_config = full_config['database']
        
        # Determine if we're on head node
        is_head_node = self._is_head_node()
        
        # For head node, always use localhost
        if is_head_node:
            db_config['host'] = '127.0.0.1'
            self.logger.info("Using localhost for database connection (we are on head node)")
            # Test connectivity for head node
            if not self._test_connectivity(db_config['host'], db_config['port']):
                 self.logger.error(f"Failed network connectivity test to localhost:{db_config['port']} on head node.")
                 raise ConnectionError(f"Could not establish network connectivity to localhost:{db_config['port']}")
        else:
            # Worker node: Try connecting to head node eth, then wifi
            self.logger.info("Attempting database connection from worker node.")
            
            # Find head node in workers config
            workers_config = full_config.get('processing', {}).get('workers', {})
            head_node_config = None
            for worker_id, worker_config in workers_config.items():
                if worker_config.get('type') == 'head':
                    head_node_config = worker_config
                    break
            
            if not head_node_config:
                raise ConnectionError("No head node found in worker configuration")
                
            primary_ip = head_node_config.get('eth')
            fallback_ip = head_node_config.get('wifi')
            db_port = db_config['port']
            
            connected = False
            last_error_host = None
            
            # Try primary IP (eth)
            if primary_ip:
                self.logger.info(f"Attempting connection to primary head node IP (eth): {primary_ip}:{db_port}")
                if self._test_connectivity(primary_ip, db_port):
                    db_config['host'] = primary_ip
                    self.logger.info(f"Successfully connected to primary head node IP: {primary_ip}")
                    connected = True
                else:
                    self.logger.warning(f"Failed to connect to primary head node IP: {primary_ip}")
                    last_error_host = primary_ip
            
            # Try fallback IP (wifi) if primary failed or doesn't exist
            if not connected and fallback_ip:
                self.logger.info(f"Attempting connection to fallback head node IP (wifi): {fallback_ip}:{db_port}")
                if self._test_connectivity(fallback_ip, db_port):
                    db_config['host'] = fallback_ip
                    self.logger.info(f"Successfully connected to fallback head node IP: {fallback_ip}")
                    connected = True
                else:
                    self.logger.warning(f"Failed to connect to fallback head node IP: {fallback_ip}")
                    if not last_error_host: # Store fallback IP if primary wasn't even tried
                         last_error_host = fallback_ip

            # Check if connection was successful
            if not connected:
                # For worker5, be more aggressive about using wifi fallback
                local_ip = self._get_local_ip()
                is_worker5 = local_ip in ['10.0.0.209', '10.0.0.99']
                
                if is_worker5 and fallback_ip:
                    self.logger.warning(f"Worker5 connectivity failed, forcing wifi IP fallback: {fallback_ip}")
                    db_config['host'] = fallback_ip
                    connected = True
                
                if not connected:
                    error_message = f"Could not establish network connectivity to head node database on port {db_port}"
                    if primary_ip and fallback_ip:
                        error_message += f" after trying eth ({primary_ip}) and wifi ({fallback_ip})."
                    elif primary_ip:
                        error_message += f" after trying eth ({primary_ip}). No fallback wifi IP configured."
                    elif fallback_ip:
                         error_message += f" after trying wifi ({fallback_ip}). No primary eth IP configured."
                    else:
                         error_message += ". No head node eth or wifi IP configured."
                    self.logger.error(error_message)
                    raise ConnectionError(error_message)
            
        # Ensure required fields are present
        required_fields = ['user', 'password', 'database']
        missing_fields = [field for field in required_fields if field not in db_config]
        if missing_fields:
            raise ValueError(f"Missing required database config fields: {missing_fields}")
        
        # Log safe configuration (using the final determined host)
        safe_config = db_config.copy()
        safe_config.pop('password', None)
        self.logger.info(f"Using final database config: {safe_config}")
        
        return db_config
    
    def get_session(self):
        """Get a database session from the pool."""
        if not self._session_factory:
            raise RuntimeError("Session factory not initialized")
        return self._session_factory()
    
    def dispose(self):
        """Dispose of the engine and all pooled connections."""
        if self._engine:
            self.logger.info("Disposing database engine and connection pool")
            self._engine.dispose()
            
    def __del__(self):
        """Ensure proper cleanup on deletion."""
        self.dispose()

    def get_connection(self):
        """Get raw database connection for special cases."""
        return self._engine.raw_connection()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for psycopg2."""
        return {
            'dbname': self.config['database'],
            'user': self.config['user'],
            'password': self.config['password'],
            'host': self.config['host'],
            'port': self.config['port']
        }

# Global session manager instance (lazy initialized)
session_manager = None

def _get_session_manager():
    """Get or create the global session manager instance."""
    global session_manager
    if session_manager is None:
        session_manager = Session()
    return session_manager

# Expose convenience functions that use the global instance
def get_engine() -> Engine:
    """Get the SQLAlchemy engine."""
    manager = _get_session_manager()
    if not manager._engine:
        raise RuntimeError("Engine not initialized")
    return manager._engine

@contextmanager
def get_session():
    """Context manager for database sessions."""
    session = _get_session_manager().get_session()
    try:
        yield session
    finally:
        session.close()

@contextmanager
def get_optimized_session():
    """Context manager for database sessions with clustering optimizations."""
    session = _get_session_manager().get_session()
    try:
        # Set optimization parameters for this session
        session.execute(text("SET work_mem = '256MB'"))
        # Try to set HNSW parameter (may fail if not in a vector operation context)
        try:
            session.execute(text("SET LOCAL hnsw.ef_search = 100"))
        except:
            # This is expected - hnsw.ef_search can only be set during vector operations
            pass
        yield session
    finally:
        session.close()

def get_connection():
    """Get a new psycopg2 connection."""
    return _get_session_manager().get_connection()

def get_connection_info() -> Dict[str, Any]:
    """Get connection information for psycopg2."""
    return _get_session_manager().get_connection_info()

def init_db():
    """Initialize the database schema."""
    try:
        from .models import Base
        Base.metadata.create_all(get_engine())
        logger.info("Successfully initialized database schema")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise 