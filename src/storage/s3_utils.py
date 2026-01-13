"""
S3 utilities for managing content storage in local S3-compatible storage (e.g. MinIO).
"""
from typing import Optional, Dict, List, BinaryIO
import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError
import logging
from pathlib import Path
import socket
import time
import json
import asyncio
import gzip
import tempfile
from datetime import datetime, timedelta

# Set up dedicated S3 logger that writes to worker-specific log files
def _setup_s3_logger():
    try:
        from src.utils.logger import setup_worker_logger
        return setup_worker_logger('s3_utils')
    except:
        # Fallback to basic logging if worker logger setup fails
        fallback = logging.getLogger('s3_utils')
        fallback.setLevel(logging.INFO)
        if not fallback.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s [S3] %(message)s'))
            fallback.addHandler(ch)
        return fallback

logger = _setup_s3_logger()

class S3ConnectionError(Exception):
    """Custom exception for S3 connection issues"""
    pass

class S3StorageConfig:
    """Configuration for S3 storage"""
    def __init__(
        self,
        endpoint_url: str = None,
        fallback_endpoint_url: str = None,
        access_key: str = None,
        secret_key: str = None,
        bucket_name: str = None,
        use_ssl: bool = False
    ):
        """Initialize S3 storage configuration"""
        self.endpoint_url = endpoint_url or os.environ.get('S3_ENDPOINT', 'http://10.0.0.147:9000')
        self.fallback_endpoint_url = fallback_endpoint_url or os.environ.get('S3_FALLBACK_ENDPOINT', 'http://10.0.0.251:9000')
        self.access_key = access_key or os.environ.get('S3_ACCESS_KEY')
        self.secret_key = secret_key or os.environ.get('S3_SECRET_KEY')
        self.bucket_name = bucket_name or os.environ.get('S3_BUCKET')
        self.use_ssl = use_ssl if use_ssl is not None else os.environ.get('S3_USE_SSL', 'false').lower() == 'true'

        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError("Missing required S3 configuration")

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'S3StorageConfig':
        """Create S3StorageConfig instance from dictionary
        
        Args:
            config_dict: Dictionary containing S3 configuration
            
        Returns:
            S3StorageConfig instance
        """
        return cls(
            endpoint_url=config_dict.get('endpoint_url'),
            fallback_endpoint_url=config_dict.get('fallback_endpoint_url'),
            access_key=config_dict.get('access_key'),
            secret_key=config_dict.get('secret_key'),
            bucket_name=config_dict.get('bucket_name'),
            use_ssl=config_dict.get('use_ssl', False)
        )

class S3Storage:
    """Handles interactions with S3-compatible storage"""
    
    def __init__(self, config: S3StorageConfig):
        """Initialize S3 storage with configuration"""
        self.config = config
        self.logger = logging.getLogger('s3_storage')
        self._client = None
        self._using_fallback = False
        self._last_health_check = None
        self._endpoint_is_healthy = True
        self._health_check_interval = 300  # 5 minutes
        self._initialize_client()

    def _initialize_client(self, use_fallback: bool = False) -> None:
        """Initialize S3 client with primary or fallback endpoint"""
        endpoint = self.config.fallback_endpoint_url if use_fallback else self.config.endpoint_url
        self.logger.info(f"Initializing S3 client with endpoint: {endpoint}")
        
        # Use shorter timeouts for faster failover
        config = Config(
            signature_version='s3v4',
            connect_timeout=5,  # 5 second connection timeout
            read_timeout=10,    # 10 second read timeout
            retries={'max_attempts': 1}  # Don't retry at boto3 level, we handle retries
        )
        
        self._client = boto3.client('s3',
            endpoint_url=endpoint,
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            config=config,
            verify=False
        )
        
        # Test connection with short timeout
        try:
            # Quick health check with minimal timeout
            start_time = time.time()
            self._client.list_buckets()
            elapsed = time.time() - start_time
            
            self._using_fallback = use_fallback
            self._endpoint_is_healthy = True
            self._last_health_check = datetime.now()
            
            self.logger.info(f"Successfully connected to S3 at {endpoint} ({elapsed:.2f}s)")
            
        except Exception as e:
            if not use_fallback:
                self.logger.warning(f"Failed to connect to primary S3 endpoint {endpoint} ({str(e)}), switching to fallback")
                self._initialize_client(use_fallback=True)
            else:
                self.logger.error(f"Failed to connect to S3 (both primary and fallback): {str(e)}")
                raise S3ConnectionError(f"Could not connect to S3 at either endpoint: {str(e)}")

    def _check_and_maybe_switch_endpoint(self) -> None:
        """Check endpoint health and switch if needed"""
        if not self._using_fallback and not self._endpoint_is_healthy:
            # If primary is unhealthy, periodically try to switch back
            if (self._last_health_check is None or 
                datetime.now() - self._last_health_check > timedelta(seconds=self._health_check_interval)):
                try:
                    self.logger.debug("Testing primary endpoint health...")
                    self._initialize_client(use_fallback=False)
                except Exception:
                    self.logger.debug("Primary endpoint still unhealthy, staying on fallback")
                    self._last_health_check = datetime.now()

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3 with retry logic and endpoint switching"""
        self._check_and_maybe_switch_endpoint()
        
        # Reduced retry configuration for faster response
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"File exists check attempt {attempt + 1}/{max_retries}")
                self._client.head_object(Bucket=self.config.bucket_name, Key=s3_key)
                return True
                
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return False
                if attempt == max_retries - 1:
                    self.logger.error(f"Error checking if {s3_key} exists after {max_retries} attempts: {str(e)}")
                    return False
                else:
                    self.logger.warning(f"File exists check attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)
                    continue
            except (ConnectionError, EndpointConnectionError) as e:
                self.logger.warning(f"Connection error on attempt {attempt + 1}: {str(e)}")
                if not self._using_fallback:
                    self.logger.info("Switching to fallback endpoint due to connection error")
                    try:
                        self._initialize_client(use_fallback=True)
                        self._endpoint_is_healthy = False
                        continue
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback endpoint also failed: {str(fallback_error)}")
                        return False
                elif attempt == max_retries - 1:
                    self.logger.error(f"Both endpoints failed: {str(e)}")
                    return False
                else:
                    time.sleep(1)
                    continue
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Error checking if {s3_key} exists after {max_retries} attempts: {str(e)}")
                    return False
                else:
                    self.logger.warning(f"File exists check attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)
                    continue

    def read_json(self, s3_key: str) -> Optional[Dict]:
        """Read a JSON file directly from S3 into memory with retry logic and endpoint switching"""
        self._check_and_maybe_switch_endpoint()
        
        # Reduced retry configuration for faster response
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Read JSON attempt {attempt + 1}/{max_retries}")
                self.logger.debug(f"Reading JSON from {s3_key}")
                response = self._client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key
                )
                content = response['Body'].read().decode('utf-8')
                return json.loads(content)
                
            except (ConnectionError, EndpointConnectionError) as e:
                self.logger.warning(f"Connection error on read JSON attempt {attempt + 1}: {str(e)}")
                if not self._using_fallback:
                    self.logger.info("Switching to fallback endpoint due to connection error")
                    try:
                        self._initialize_client(use_fallback=True)
                        self._endpoint_is_healthy = False
                        continue
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback endpoint also failed: {str(fallback_error)}")
                        return None
                elif attempt == max_retries - 1:
                    self.logger.error(f"Both endpoints failed: {str(e)}")
                    return None
                else:
                    time.sleep(1)
                    continue
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to read JSON from {s3_key} after {max_retries} attempts: {str(e)}")
                    return None
                else:
                    self.logger.warning(f"Read JSON attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)
                    continue

    def read_json_flexible(self, s3_key: str) -> Optional[Dict]:
        """Read a JSON file from S3, handling both compressed and uncompressed versions.
        
        This method first tries to read the compressed version (.json.gz), and if that
        doesn't exist, falls back to the uncompressed version (.json).
        
        Args:
            s3_key: S3 key for the JSON file (without .gz extension)
            
        Returns:
            Dictionary containing the JSON data, or None if file doesn't exist
        """
        # First try compressed version
        compressed_key = s3_key + '.gz' if not s3_key.endswith('.gz') else s3_key
        uncompressed_key = s3_key.replace('.gz', '') if s3_key.endswith('.gz') else s3_key
        
        self.logger.debug(f"Trying to read JSON: compressed={compressed_key}, uncompressed={uncompressed_key}")
        
        # Try compressed version first
        if self.file_exists(compressed_key):
            self.logger.debug(f"Found compressed version: {compressed_key}")
            return self._read_compressed_json(compressed_key)
        
        # Fall back to uncompressed version
        elif self.file_exists(uncompressed_key):
            self.logger.debug(f"Found uncompressed version: {uncompressed_key}")
            return self.read_json(uncompressed_key)
        
        else:
            self.logger.warning(f"Neither compressed nor uncompressed version found for: {s3_key}")
            return None

    def _read_compressed_json(self, s3_key: str) -> Optional[Dict]:
        """Read a compressed JSON file from S3."""
        # Retry configuration
        max_retries = 3
        base_timeout = 15  # Base timeout in seconds
        max_timeout = 60   # Maximum timeout in seconds
        
        for attempt in range(max_retries):
            try:
                # Calculate timeout with exponential backoff
                timeout = min(base_timeout * (2 ** attempt), max_timeout)
                self.logger.debug(f"Read compressed JSON attempt {attempt + 1}/{max_retries} with {timeout}s timeout")
                
                self.logger.debug(f"Reading compressed JSON from {s3_key}")
                response = self._client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key
                )
                
                # Decompress the content
                compressed_content = response['Body'].read()
                decompressed_content = gzip.decompress(compressed_content).decode('utf-8')
                return json.loads(decompressed_content)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to read compressed JSON from {s3_key} after {max_retries} attempts: {str(e)}")
                    return None
                else:
                    self.logger.warning(f"Read compressed JSON attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)
                    continue

    def download_json_flexible(self, s3_key: str, local_path: str) -> bool:
        """Download a JSON file from S3 to local disk, handling both compressed and uncompressed versions.
        
        This method first tries to download the compressed version (.json.gz), decompresses it,
        and saves it as uncompressed JSON. If compressed version doesn't exist, downloads
        the uncompressed version directly.
        
        Args:
            s3_key: S3 key for the JSON file (without .gz extension)
            local_path: Local path where to save the uncompressed JSON file
            
        Returns:
            True if successful, False otherwise
        """
        # First try compressed version
        compressed_key = s3_key + '.gz' if not s3_key.endswith('.gz') else s3_key
        uncompressed_key = s3_key.replace('.gz', '') if s3_key.endswith('.gz') else s3_key
        
        self.logger.debug(f"Trying to download JSON: compressed={compressed_key}, uncompressed={uncompressed_key}")
        
        # Try compressed version first
        if self.file_exists(compressed_key):
            self.logger.debug(f"Found compressed version: {compressed_key}")
            return self._download_and_decompress_json(compressed_key, local_path)
        
        # Fall back to uncompressed version
        elif self.file_exists(uncompressed_key):
            self.logger.debug(f"Found uncompressed version: {uncompressed_key}")
            return self.download_file(uncompressed_key, local_path)
        
        else:
            self.logger.warning(f"Neither compressed nor uncompressed version found for: {s3_key}")
            return False

    def _download_and_decompress_json(self, s3_key: str, local_path: str) -> bool:
        """Download and decompress a JSON file from S3."""
        try:
            # Download compressed file to temporary location
            with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as temp_file:
                temp_compressed_path = temp_file.name
            
            try:
                # Download compressed file
                if not self.download_file(s3_key, temp_compressed_path):
                    return False
                
                # Decompress to final location
                with gzip.open(temp_compressed_path, 'rb') as f_in:
                    with open(local_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                self.logger.info(f"Successfully downloaded and decompressed {s3_key} to {local_path}")
                return True
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_compressed_path):
                    os.unlink(temp_compressed_path)
                    
        except Exception as e:
            self.logger.error(f"Error downloading and decompressing {s3_key}: {str(e)}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a file from S3 with retry logic and endpoint switching"""
        self._check_and_maybe_switch_endpoint()
        
        # Check file size to determine download method
        try:
            response = self._client.head_object(Bucket=self.config.bucket_name, Key=s3_key)
            file_size = response.get('ContentLength', 0)
            self.logger.info(f"File size for {s3_key}: {file_size:,} bytes")
            
            # Use streaming download for files larger than 50MB
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                self.logger.info(f"Using streaming download for large file: {s3_key}")
                success = self._download_large_file_streaming(s3_key, local_path, file_size)
                if not success:
                    self.logger.warning(f"Streaming download failed, trying direct download for {s3_key}")
                    # Try direct download without streaming
                    success = self._download_direct(s3_key, local_path)
                    if not success:
                        self.logger.warning(f"Direct download failed, trying AWS CLI fallback for {s3_key}")
                        return self._download_with_aws_cli(s3_key, local_path)
                return success
        except Exception as e:
            self.logger.warning(f"Could not determine file size for {s3_key}: {e}")
            # Fall through to regular download
        
        # Reduced retry configuration for faster response
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Download attempt {attempt + 1}/{max_retries}")
                self.logger.info(f"Downloading {s3_key} to {local_path}")
                self._client.download_file(self.config.bucket_name, s3_key, local_path)
                return True
                
            except (ConnectionError, EndpointConnectionError) as e:
                self.logger.warning(f"Connection error on download attempt {attempt + 1}: {str(e)}")
                if not self._using_fallback:
                    self.logger.info("Switching to fallback endpoint due to connection error")
                    try:
                        self._initialize_client(use_fallback=True)
                        self._endpoint_is_healthy = False
                        continue
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback endpoint also failed: {str(fallback_error)}")
                        return False
                elif attempt == max_retries - 1:
                    self.logger.error(f"Both endpoints failed: {str(e)}")
                    return False
                else:
                    time.sleep(1)
                    continue
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to download {s3_key} after {max_retries} attempts: {str(e)}")
                    return False
                else:
                    self.logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                    # Clean up any partial download
                    if os.path.exists(local_path):
                        try:
                            os.unlink(local_path)
                        except Exception as cleanup_error:
                            self.logger.warning(f"Failed to clean up partial download: {str(cleanup_error)}")
                    time.sleep(1)
                    continue

    def _download_large_file_streaming(self, s3_key: str, local_path: str, file_size: int) -> bool:
        """Download a large file using resumable chunks with extended timeouts."""
        import boto3
        from botocore.client import Config
        
        # Create a client with extended timeouts for large files
        extended_config = Config(
            signature_version='s3v4',
            connect_timeout=30,     # 30 second connection timeout (increased)
            read_timeout=120,       # 120 second read timeout per chunk (increased)
            retries={'max_attempts': 1}  # Don't retry at boto3 level, we handle it
        )
        
        max_retries = 10  # Increased retries for better reliability
        chunk_size = 1024 * 1024 * 2   # 2MB chunks for better stability (reduced)
        
        # Start with fresh file
        if os.path.exists(local_path):
            os.unlink(local_path)
        
        bytes_downloaded = 0
        
        while bytes_downloaded < file_size:
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Calculate range for this chunk
                    start_byte = bytes_downloaded
                    end_byte = min(start_byte + chunk_size - 1, file_size - 1)
                    range_header = f"bytes={start_byte}-{end_byte}"
                    
                    # Prefer fallback endpoint for large file downloads (more stable)
                    endpoint = self.config.fallback_endpoint_url
                    extended_client = boto3.client('s3',
                        endpoint_url=endpoint,
                        aws_access_key_id=self.config.access_key,
                        aws_secret_access_key=self.config.secret_key,
                        config=extended_config,
                        verify=False
                    )
                    
                    self.logger.debug(f"Downloading chunk {start_byte}-{end_byte} (attempt {attempt + 1}/{max_retries})")
                    
                    # Get the object chunk
                    response = extended_client.get_object(
                        Bucket=self.config.bucket_name, 
                        Key=s3_key, 
                        Range=range_header
                    )
                    
                    # Write chunk to file with chunked reading to handle IncompleteRead
                    body = response['Body']
                    chunk_data = b''
                    
                    # Read in smaller sub-chunks to avoid IncompleteRead errors
                    sub_chunk_size = 1024 * 256  # 256KB sub-chunks
                    while True:
                        try:
                            data = body.read(sub_chunk_size)
                            if not data:
                                break
                            chunk_data += data
                        except Exception as read_error:
                            if 'IncompleteRead' in str(read_error):
                                # If we got some data before the error, consider it a partial success
                                if chunk_data:
                                    self.logger.warning(f"IncompleteRead at byte {start_byte + len(chunk_data)}, saving {len(chunk_data)} bytes")
                                    break
                                else:
                                    # No data received, this is a real error - re-raise to trigger retry
                                    self.logger.debug(f"IncompleteRead with no data at byte {start_byte}")
                                    raise
                            else:
                                raise
                    
                    # Only write if we got some data
                    if chunk_data:
                        mode = 'ab' if bytes_downloaded > 0 else 'wb'
                        with open(local_path, mode) as f:
                            f.write(chunk_data)
                        
                        bytes_downloaded += len(chunk_data)
                        progress = int((bytes_downloaded / file_size) * 100)
                        
                        if progress % 10 == 0 or bytes_downloaded == file_size:
                            self.logger.info(f"Download progress: {progress}% ({bytes_downloaded:,} / {file_size:,} bytes)")
                        
                        success = True  # Mark as success even for partial chunks
                        break
                    else:
                        # No data received, treat as failure
                        raise Exception(f"No data received for chunk at byte {start_byte}")
                    
                except (ConnectionError, EndpointConnectionError) as e:
                    self.logger.warning(f"Connection error downloading chunk at {start_byte} (attempt {attempt + 1}): {str(e)}")
                    if not self._using_fallback and attempt < max_retries - 1:
                        self.logger.info("Switching to fallback endpoint for chunk download")
                        try:
                            self._initialize_client(use_fallback=True)
                            self._endpoint_is_healthy = False
                            continue
                        except Exception as fallback_error:
                            self.logger.error(f"Fallback endpoint also failed: {str(fallback_error)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                except Exception as e:
                    self.logger.warning(f"Error downloading chunk at {start_byte} (attempt {attempt + 1}): {str(e)}")
                    
                    # If we've been getting IncompleteRead errors consistently, try switching endpoints
                    if 'IncompleteRead' in str(e) and attempt == 2:
                        self.logger.info("IncompleteRead errors detected, switching endpoint")
                        # Toggle between primary and fallback endpoint
                        if endpoint == self.config.fallback_endpoint_url:
                            endpoint = self.config.endpoint_url
                        else:
                            endpoint = self.config.fallback_endpoint_url
                    
                    # Use longer backoff for network issues
                    backoff_time = min(60, 2 ** attempt)  # Cap at 60 seconds
                    self.logger.info(f"Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                    continue
            
            if not success:
                self.logger.error(f"Failed to download chunk at byte {start_byte} after {max_retries} attempts")
                # Clean up partial file
                if os.path.exists(local_path):
                    os.unlink(local_path)
                return False
        
        self.logger.info(f"Successfully downloaded {s3_key} ({file_size:,} bytes) using resumable chunks")
        return True

    def _download_direct(self, s3_key: str, local_path: str) -> bool:
        """Direct download without streaming for files that fail with streaming."""
        import boto3
        from botocore.client import Config
        
        self.logger.info(f"Attempting direct download for {s3_key}")
        
        # Use extended timeouts for large files
        extended_config = Config(
            signature_version='s3v4',
            connect_timeout=60,      # 60 second connection timeout
            read_timeout=600,        # 10 minute read timeout for entire file
            retries={'max_attempts': 3}
        )
        
        # Try both endpoints
        endpoints = [self.config.fallback_endpoint_url, self.config.endpoint_url]
        
        for endpoint in endpoints:
            try:
                self.logger.info(f"Trying direct download from endpoint: {endpoint}")
                
                client = boto3.client('s3',
                    endpoint_url=endpoint,
                    aws_access_key_id=self.config.access_key,
                    aws_secret_access_key=self.config.secret_key,
                    config=extended_config,
                    verify=False
                )
                
                # Direct download entire file
                response = client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key
                )
                
                # Write entire file at once
                with open(local_path, 'wb') as f:
                    f.write(response['Body'].read())
                
                self.logger.info(f"Successfully downloaded {s3_key} using direct download")
                return True
                
            except Exception as e:
                self.logger.warning(f"Direct download failed from {endpoint}: {str(e)}")
                continue
        
        return False

    def _download_with_aws_cli(self, s3_key: str, local_path: str) -> bool:
        """Fallback download using AWS CLI for problematic large files."""
        import subprocess
        
        self.logger.info(f"Attempting AWS CLI download for {s3_key}")
        
        # Try both endpoints
        endpoints = [self.config.fallback_endpoint_url, self.config.endpoint_url]
        
        for endpoint in endpoints:
            try:
                # Set AWS CLI environment variables
                env = os.environ.copy()
                env['AWS_ACCESS_KEY_ID'] = self.config.access_key
                env['AWS_SECRET_ACCESS_KEY'] = self.config.secret_key
                
                # Construct AWS CLI command
                cmd = [
                    'aws', 's3', 'cp',
                    f's3://{self.config.bucket_name}/{s3_key}',
                    local_path,
                    '--endpoint-url', endpoint
                ]
                
                self.logger.info(f"Running AWS CLI command with endpoint {endpoint}")
                self.logger.debug(f"Command: {' '.join(cmd)}")
                
                # Run with timeout (10 minutes)
                result = subprocess.run(
                    cmd, 
                    env=env, 
                    timeout=600,  # 10 minute timeout
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    if os.path.exists(local_path):
                        file_size = os.path.getsize(local_path)
                        self.logger.info(f"AWS CLI download successful: {file_size:,} bytes")
                        return True
                    else:
                        self.logger.error("AWS CLI reported success but file doesn't exist")
                else:
                    self.logger.warning(f"AWS CLI download failed with code {result.returncode}")
                    self.logger.warning(f"stderr: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"AWS CLI download timed out after 10 minutes for endpoint {endpoint}")
            except Exception as e:
                self.logger.error(f"Error running AWS CLI download with endpoint {endpoint}: {e}")
        
        self.logger.error("All AWS CLI download attempts failed")
        return False

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload a file to S3 with retry logic and endpoint switching"""
        self._check_and_maybe_switch_endpoint()
        
        # Verify file exists and get size
        if not os.path.exists(local_path):
            self.logger.error(f"Local file does not exist: {local_path}")
            return False
            
        file_size = os.path.getsize(local_path)
        self.logger.info(f"Uploading {local_path} ({file_size} bytes) to {s3_key}")
        
        # Reduced retry configuration for faster response
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Upload attempt {attempt + 1}/{max_retries}")
                
                try:
                    # Try multipart upload first
                    self._client.upload_file(local_path, self.config.bucket_name, s3_key)
                    self.logger.info(f"Successfully uploaded {s3_key}")
                    return True
                except Exception as e:
                    self.logger.warning(f"Multipart upload failed, trying direct upload: {str(e)}")
                    # Try direct upload as fallback
                    with open(local_path, 'rb') as f:
                        self._client.put_object(
                            Bucket=self.config.bucket_name,
                            Key=s3_key,
                            Body=f
                        )
                    self.logger.info(f"Successfully uploaded {s3_key} using direct upload")
                    return True
                    
            except (ConnectionError, EndpointConnectionError) as e:
                self.logger.warning(f"Connection error on upload attempt {attempt + 1}: {str(e)}")
                if not self._using_fallback:
                    self.logger.info("Switching to fallback endpoint due to connection error")
                    try:
                        self._initialize_client(use_fallback=True)
                        self._endpoint_is_healthy = False
                        continue
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback endpoint also failed: {str(fallback_error)}")
                        return False
                elif attempt == max_retries - 1:
                    self.logger.error(f"Both endpoints failed: {str(e)}")
                    return False
                else:
                    time.sleep(1)
                    continue
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to upload {s3_key} after {max_retries} attempts: {str(e)}", exc_info=True)
                    return False
                else:
                    self.logger.warning(f"Upload attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)
                    continue

    def upload_json(self, s3_key: str, data: Dict) -> bool:
        """Upload JSON data directly to S3 with retry logic and endpoint switching.
        
        Args:
            s3_key: S3 key to upload to
            data: Dictionary to upload as JSON
            
        Returns:
            True if successful, False otherwise
        """
        self._check_and_maybe_switch_endpoint()
        
        # Reduced retry configuration for faster response
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Upload JSON attempt {attempt + 1}/{max_retries}")
                
                # Convert data to JSON string
                json_str = json.dumps(data, indent=2)
                
                # Upload directly using put_object
                self._client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key,
                    Body=json_str.encode('utf-8'),
                    ContentType='application/json'
                )
                
                self.logger.info(f"Successfully uploaded JSON to {s3_key}")
                return True
                
            except (ConnectionError, EndpointConnectionError) as e:
                self.logger.warning(f"Connection error on upload JSON attempt {attempt + 1}: {str(e)}")
                if not self._using_fallback:
                    self.logger.info("Switching to fallback endpoint due to connection error")
                    try:
                        self._initialize_client(use_fallback=True)
                        self._endpoint_is_healthy = False
                        continue
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback endpoint also failed: {str(fallback_error)}")
                        return False
                elif attempt == max_retries - 1:
                    self.logger.error(f"Both endpoints failed: {str(e)}")
                    return False
                else:
                    time.sleep(1)
                    continue
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to upload JSON to {s3_key} after {max_retries} attempts: {str(e)}")
                    return False
                else:
                    self.logger.warning(f"Upload JSON attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)
                    continue

    def list_files(self, prefix: str = '') -> list:
        """List files in S3 bucket with optional prefix"""
        try:
            response = self._client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            self.logger.error(f"Error listing files with prefix {prefix}: {str(e)}")
            return []

    def list_s3_objects(self, prefix: str = '', suffix: str = '') -> List[str]:
        """
        List S3 objects with specific prefix and suffix.
        
        Args:
            prefix: Prefix to filter objects (e.g. 'content/')
            suffix: Suffix to filter objects (e.g. '.json')
            
        Returns:
            List of matching S3 object keys
        """
        try:
            all_objects = self.list_files(prefix)
            if suffix:
                return [obj for obj in all_objects if obj.endswith(suffix)]
            return all_objects
        except Exception as e:
            logger.error(f"Error listing S3 objects with prefix {prefix} and suffix {suffix}: {e}")
            return []

    def delete_file(self, s3_key: str) -> bool:
        """Delete a file from S3 with retry logic"""
        # Retry configuration
        max_retries = 3
        base_timeout = 15  # Base timeout in seconds
        max_timeout = 60   # Maximum timeout in seconds
        
        for attempt in range(max_retries):
            try:
                # Calculate timeout with exponential backoff
                timeout = min(base_timeout * (2 ** attempt), max_timeout)
                self.logger.debug(f"Delete file attempt {attempt + 1}/{max_retries} with {timeout}s timeout")
                
                self.logger.info(f"Deleting {s3_key}")
                self._client.delete_object(Bucket=self.config.bucket_name, Key=s3_key)
                return True
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to delete {s3_key} after {max_retries} attempts: {str(e)}")
                    return False
                else:
                    self.logger.warning(f"Delete file attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)
                    continue

    def get_file_url(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """Get a pre-signed URL for a file with retry logic"""
        # Retry configuration
        max_retries = 3
        base_timeout = 15  # Base timeout in seconds
        max_timeout = 60   # Maximum timeout in seconds
        
        for attempt in range(max_retries):
            try:
                # Calculate timeout with exponential backoff
                timeout = min(base_timeout * (2 ** attempt), max_timeout)
                self.logger.debug(f"Get file URL attempt {attempt + 1}/{max_retries} with {timeout}s timeout")
                
                url = self._client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': self.config.bucket_name,
                        'Key': s3_key
                    },
                    ExpiresIn=expires_in
                )
                return url
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error generating URL for {s3_key} after {max_retries} attempts: {str(e)}")
                    return None
                else:
                    logger.warning(f"Get file URL attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)
                    continue

    def file_exists_flexible(self, s3_key: str) -> bool:
        """Check if a file exists in S3, handling both compressed and uncompressed versions.
        
        Args:
            s3_key: S3 key for the file (without .gz extension)
            
        Returns:
            True if either compressed or uncompressed version exists
        """
        # First try compressed version
        compressed_key = s3_key + '.gz' if not s3_key.endswith('.gz') else s3_key
        uncompressed_key = s3_key.replace('.gz', '') if s3_key.endswith('.gz') else s3_key
        
        # Check compressed version first (preferred)
        if self.file_exists(compressed_key):
            self.logger.debug(f"Found compressed version: {compressed_key}")
            return True
        
        # Fall back to uncompressed version
        if self.file_exists(uncompressed_key):
            self.logger.debug(f"Found uncompressed version: {uncompressed_key}")
            return True
            
        return False

    def download_audio_flexible(self, content_id: str, output_wav_path: str) -> bool:
        """Download and convert audio file from S3, handling multiple formats.
        
        Checks for audio files in priority order: opus, mp3, wav
        Converts compressed formats to 16kHz mono WAV using ffmpeg.
        
        Args:
            content_id: Content ID to find audio for
            output_wav_path: Local path where WAV file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for various audio file formats in S3 (compressed first, then WAV)
            audio_formats = [
                ("audio.opus", "opus"),
                ("audio.mp3", "mp3"),
                ("audio.wav", "wav")
            ]
            
            for filename, format_type in audio_formats:
                s3_audio_key = f"content/{content_id}/{filename}"
                
                if self.file_exists(s3_audio_key):
                    self.logger.info(f"[{content_id}] Found {filename} in S3")
                    
                    if format_type == "wav":
                        # Direct download for WAV
                        if self.download_file(s3_audio_key, output_wav_path):
                            self.logger.info(f"[{content_id}] Downloaded WAV file directly")
                            return True
                    else:
                        # Download compressed file and convert to WAV
                        temp_compressed_path = str(Path(output_wav_path).parent / filename)
                        
                        if self.download_file(s3_audio_key, temp_compressed_path):
                            self.logger.info(f"[{content_id}] Downloaded compressed {filename}")
                            
                            # Convert to WAV using ffmpeg
                            if self._convert_audio_to_wav(temp_compressed_path, output_wav_path, format_type):
                                self.logger.info(f"[{content_id}] Converted {filename} to WAV")
                                # Clean up compressed file
                                try:
                                    Path(temp_compressed_path).unlink()
                                except OSError:
                                    pass
                                return True
                            else:
                                self.logger.error(f"[{content_id}] Failed to convert {filename}")
                                # Clean up compressed file on failure
                                try:
                                    Path(temp_compressed_path).unlink()
                                except OSError:
                                    pass
            
            self.logger.error(f"[{content_id}] No audio file found in S3 (checked: {[f[0] for f in audio_formats]})")
            return False
            
        except Exception as e:
            self.logger.error(f"[{content_id}] Error downloading audio: {str(e)}")
            return False

    def _convert_audio_to_wav(self, input_path: str, output_path: str, format_type: str) -> bool:
        """Convert audio file to WAV format using ffmpeg.
        
        Args:
            input_path: Path to input audio file
            output_path: Path where WAV file should be saved
            format_type: Type of input audio ('opus', 'mp3', etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import subprocess
            
            # Use ffmpeg for all audio conversions
            cmd = [
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',          # 16kHz sample rate
                '-ac', '1',              # Mono
                '-y',                    # Overwrite output
                output_path
            ]
            
            self.logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.debug(f"Successfully converted {format_type} to WAV")
                return True
            else:
                self.logger.error(f"FFmpeg failed with return code {result.returncode}")
                self.logger.error(f"FFmpeg stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"FFmpeg conversion timed out after 5 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Error converting {format_type} audio: {e}")
            return False


    async def download_audio_slice(
        self,
        content_id: str,
        start_time: float,
        end_time: float,
        output_wav_path: str
    ) -> bool:
        """
        Download and slice audio directly from S3 using ffmpeg with HTTP range requests.

        This is efficient because ffmpeg only downloads the needed portion via HTTP range requests
        instead of downloading the entire file first.

        Args:
            content_id: Content ID
            start_time: Start time in seconds
            end_time: End time in seconds
            output_wav_path: Local path where WAV file should be saved

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"[{content_id}] download_audio_slice ENTRY for {start_time:.1f}s-{end_time:.1f}s")

        max_retries = 3
        import random

        # Add small random jitter (0-500ms) to avoid thundering herd
        jitter = random.random() * 0.5
        await asyncio.sleep(jitter)

        logger.info(f"[{content_id}] After jitter sleep, checking for audio files...")

        # Find audio file with single S3 list call
        check_start = time.time()
        content_files = self.list_files(f"content/{content_id}/")
        check_time = time.time() - check_start
        logger.info(f"[{content_id}] list_files() took {check_time:.2f}s, found {len(content_files)} files")

        # Check for audio files in priority order
        audio_formats = [
            ("audio.opus", "opus"),
            ("audio.mp3", "mp3"),
            ("audio.wav", "wav")
        ]

        s3_audio_key = None
        format_type = None

        for filename, fmt in audio_formats:
            key = f"content/{content_id}/{filename}"
            if key in content_files:
                s3_audio_key = key
                format_type = fmt
                logger.info(f"[{content_id}] Found {filename} in S3")
                break

        if not s3_audio_key:
            logger.error(f"[{content_id}] No audio file found in S3 (checked: {[f[0] for f in audio_formats]})")
            return False

        logger.info(f"[{content_id}] Starting attempt loop with {s3_audio_key}")

        for attempt in range(max_retries):
            try:
                import subprocess
                import tempfile

                logger.info(f"[{content_id}] Attempt {attempt + 1}/{max_retries}")

                # Generate presigned URL for ffmpeg to access
                # Try both endpoints since the current endpoint might not be reachable from this worker
                logger.info(f"[{content_id}] Generating presigned URL...")
                url_start = time.time()

                presigned_url = None
                # Try fallback first since it's more likely to be universally reachable
                endpoints_to_try = [self.config.fallback_endpoint_url, self.config.endpoint_url]

                for endpoint in endpoints_to_try:
                    try:
                        # Create temporary client with this endpoint
                        temp_config = Config(
                            signature_version='s3v4',
                            connect_timeout=2,
                            read_timeout=2,
                            retries={'max_attempts': 1}
                        )
                        temp_client = boto3.client('s3',
                            endpoint_url=endpoint,
                            aws_access_key_id=self.config.access_key,
                            aws_secret_access_key=self.config.secret_key,
                            config=temp_config,
                            verify=False
                        )
                        presigned_url = temp_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': self.config.bucket_name, 'Key': s3_audio_key},
                            ExpiresIn=3600
                        )
                        logger.info(f"[{content_id}] Generated presigned URL using endpoint: {endpoint}")
                        break
                    except Exception as e:
                        logger.warning(f"[{content_id}] Failed to generate URL with endpoint {endpoint}: {e}")
                        continue

                url_time = time.time() - url_start
                logger.info(f"[{content_id}] get_file_url() took {url_time:.2f}s")

                if not presigned_url:
                    logger.error(f"[{content_id}] Failed to generate presigned URL on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

                logger.info(f"[{content_id}] Slicing audio {start_time:.1f}s - {end_time:.1f}s using ffmpeg (attempt {attempt + 1}/{max_retries})")

                # Build ffmpeg command with range-based seeking
                cmd = ['ffmpeg', '-loglevel', 'warning']

                # Add seeking BEFORE input for fast seeking with range requests
                cmd.extend(['-ss', str(start_time)])

                # Input URL
                cmd.extend(['-i', presigned_url])

                # Duration
                duration = end_time - start_time
                cmd.extend(['-t', str(duration)])

                # Output options - convert to 16kHz mono WAV
                cmd.extend([
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-ar', '16000',           # 16kHz sample rate
                    '-ac', '1',               # Mono
                    '-y',                     # Overwrite output
                    output_wav_path
                ])

                logger.info(f"[{content_id}] Running ffmpeg command...")

                # Run ffmpeg with timeout
                ffmpeg_start = time.time()
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                try:
                    stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)
                    ffmpeg_time = time.time() - ffmpeg_start
                    logger.info(f"[{content_id}] ffmpeg completed in {ffmpeg_time:.2f}s")
                except asyncio.TimeoutError:
                    ffmpeg_time = time.time() - ffmpeg_start
                    logger.warning(f"[{content_id}] ffmpeg timed out after {ffmpeg_time:.2f}s on attempt {attempt + 1}")
                    result.kill()
                    await result.wait()
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

                if result.returncode != 0:
                    stderr_text = stderr.decode() if stderr else "no stderr"
                    logger.warning(f"[{content_id}] ffmpeg failed on attempt {attempt + 1} (return code {result.returncode}): {stderr_text}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

                # Verify output file exists
                if not os.path.exists(output_wav_path):
                    logger.warning(f"[{content_id}] ffmpeg succeeded but output file not found on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

                file_size = os.path.getsize(output_wav_path)
                logger.info(f"[{content_id}] Successfully sliced audio: {file_size / 1024 / 1024:.2f} MB")
                return True

            except Exception as e:
                logger.warning(f"[{content_id}] Error slicing audio on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    import traceback
                    logger.error(traceback.format_exc())
                    # Don't return False yet - try fallback below
                    break
                await asyncio.sleep(2 ** attempt)

        # Fallback: Download full audio and slice locally
        logger.warning(f"[{content_id}] Range request approach failed after {max_retries} attempts, trying fallback: download full audio and slice locally")
        return await self._download_and_slice_locally(content_id, start_time, end_time, output_wav_path, s3_audio_key)

    async def _download_and_slice_locally(
        self,
        content_id: str,
        start_time: float,
        end_time: float,
        output_wav_path: str,
        s3_audio_key: str = None
    ) -> bool:
        """
        Fallback method: Download full audio file and slice locally using ffmpeg.

        Args:
            content_id: Content ID
            start_time: Start time in seconds
            end_time: End time in seconds
            output_wav_path: Local path where WAV file should be saved
            s3_audio_key: S3 key for audio file (if already discovered)

        Returns:
            True if successful, False otherwise
        """
        import tempfile
        import subprocess

        try:
            # Find audio file in S3 if not provided
            if not s3_audio_key:
                audio_formats = [
                    ("audio.opus", "opus"),
                    ("audio.mp3", "mp3"),
                    ("audio.wav", "wav")
                ]

                for filename, fmt in audio_formats:
                    key = f"content/{content_id}/{filename}"
                    if self.file_exists(key):
                        s3_audio_key = key
                        logger.info(f"[{content_id}] Fallback: Found {filename} in S3")
                        break

                if not s3_audio_key:
                    logger.error(f"[{content_id}] Fallback: No audio file found in S3")
                    return False
            else:
                logger.info(f"[{content_id}] Fallback: Using provided audio key: {s3_audio_key}")

            # Download full audio to temp file
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(s3_audio_key)[1], delete=False) as temp_audio:
                temp_audio_path = temp_audio.name

            try:
                logger.info(f"[{content_id}] Fallback: Downloading full audio file from {s3_audio_key}")
                download_start = time.time()

                if not self.download_file(s3_audio_key, temp_audio_path):
                    logger.error(f"[{content_id}] Fallback: Failed to download full audio")
                    return False

                download_time = time.time() - download_start
                file_size_mb = os.path.getsize(temp_audio_path) / 1024 / 1024
                logger.info(f"[{content_id}] Fallback: Downloaded {file_size_mb:.1f}MB in {download_time:.1f}s")

                # Slice locally using ffmpeg
                duration = end_time - start_time
                cmd = [
                    'ffmpeg', '-loglevel', 'warning',
                    '-ss', str(start_time),
                    '-i', temp_audio_path,
                    '-t', str(duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    output_wav_path
                ]

                logger.info(f"[{content_id}] Fallback: Slicing locally {start_time:.1f}s - {end_time:.1f}s")
                slice_start = time.time()

                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=60)
                slice_time = time.time() - slice_start

                if result.returncode != 0:
                    logger.error(f"[{content_id}] Fallback: ffmpeg failed: {stderr.decode()}")
                    return False

                if not os.path.exists(output_wav_path):
                    logger.error(f"[{content_id}] Fallback: Output file not created")
                    return False

                output_size_mb = os.path.getsize(output_wav_path) / 1024 / 1024
                logger.info(f"[{content_id}] Fallback: Successfully sliced {output_size_mb:.2f}MB in {slice_time:.1f}s")
                return True

            finally:
                # Clean up temp file
                if os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                    except Exception as e:
                        logger.warning(f"[{content_id}] Fallback: Failed to clean up temp file: {e}")

        except Exception as e:
            logger.error(f"[{content_id}] Fallback method failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def create_s3_storage_from_config(config_dict: Dict) -> S3Storage:
    """Create a fresh S3Storage instance from configuration dictionary.

    This is useful for avoiding stale connections by creating new instances as needed.

    Args:
        config_dict: Dictionary containing S3 configuration

    Returns:
        Fresh S3Storage instance
    """
    s3_config = S3StorageConfig.from_dict(config_dict)
    return S3Storage(s3_config) 