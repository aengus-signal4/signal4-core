"""
Migration script to move existing data to the new S3 structure using mc for direct file operations.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import glob
import re
import subprocess
from datetime import datetime, timezone
import logging

from ..utils.logger import setup_task_logger, setup_worker_logger

# Set up loggers
task_logger = setup_task_logger('migration')
worker_logger = setup_worker_logger('migration')

def load_full_config():
    """Load the full configuration from YAML"""
    config_path = Path("config/config.yaml")
    worker_logger.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    worker_logger.info("Config loaded successfully")
    return config

def run_mc_command(cmd: str) -> bool:
    """Run an mc command and return success status"""
    try:
        result = subprocess.run(f"mc {cmd}", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            worker_logger.error(f"mc command failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        worker_logger.error(f"Error running mc command: {str(e)}")
        return False

class StorageMigration:
    """Handles migration of content from file system to S3 using mc for direct operations"""
    
    def __init__(self, base_path: str):
        worker_logger.info(f"Initializing StorageMigration with base_path: {base_path}")
        self.base_path = Path(base_path)
        worker_logger.info(f"Base path set to: {self.base_path}")
        
        # Load config
        self.config = load_full_config()
        worker_logger.info("Config loaded in StorageMigration")
        
        # Ensure bucket exists
        bucket_name = self.config['storage']['s3']['bucket_name']
        if not run_mc_command(f"ls s4/{bucket_name}"):
            worker_logger.info(f"Creating bucket: {bucket_name}")
            if not run_mc_command(f"mb s4/{bucket_name}"):
                raise Exception(f"Failed to create bucket: {bucket_name}")

    def _get_content_id_from_file(self, file_path: Path) -> str:
        """Extract content ID from filename"""
        # Remove extension and any _temp suffix
        content_id = file_path.stem
        if content_id.endswith('_temp'):
            content_id = content_id[:-5]
        return content_id

    def _get_platform_from_extension(self, file_path: Path) -> str:
        """Determine platform based on file extension"""
        if file_path.suffix.lower() in ['.mp3', '.m4a', '.wav']:
            return 'podcast'
        return 'youtube'  # Default to youtube for video formats

    def migrate_file(self, file_path: Path) -> Dict:
        """Migrate a single file to S3"""
        try:
            content_id = self._get_content_id_from_file(file_path)
            platform = self._get_platform_from_extension(file_path)
            bucket_name = self.config['storage']['s3']['bucket_name']

            # Define S3 paths
            s3_base = f"s4/{bucket_name}/content/{content_id}"
            s3_source = f"{s3_base}/source{file_path.suffix}"

            # Create basic metadata
            metadata = {
                'platform': platform,
                'content_id': content_id,
                'original_filename': file_path.name,
                'migrated_at': datetime.now(timezone.utc).isoformat(),
                'source_extension': file_path.suffix
            }

            # Write metadata to temp file
            temp_meta = f"temp_meta_{content_id}.json"
            with open(temp_meta, 'w') as f:
                json.dump(metadata, f)

            # Upload metadata
            if not run_mc_command(f"cp {temp_meta} {s3_base}/meta.json"):
                raise Exception("Failed to upload metadata")

            # Upload source file
            if not run_mc_command(f"cp {file_path} {s3_source}"):
                raise Exception("Failed to upload source file")

            # Clean up temp metadata file
            os.remove(temp_meta)

            return {
                'status': 'success',
                'content_id': content_id,
                'platform': platform,
                's3_path': s3_source
            }

        except Exception as e:
            worker_logger.error(f"Error migrating file {file_path}: {str(e)}")
            return {
                'status': 'error',
                'file': str(file_path),
                'error': str(e)
            }

    def migrate_all(self) -> Dict:
        """Migrate all content files from downloads directory"""
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }

        try:
            # Find all media files
            extensions = ['.mp4', '.mp3', '.m4a', '.wav', '.webm', '.mkv', '.avi', '.mov']
            files = []
            for ext in extensions:
                files.extend(self.base_path.glob(f"*{ext}"))

            stats['total'] = len(files)
            worker_logger.info(f"Found {stats['total']} files to migrate")

            # Process each file
            from tqdm import tqdm
            for file_path in tqdm(files, desc="Migrating content"):
                # Skip temp files
                if '_temp' in file_path.name:
                    stats['skipped'] += 1
                    continue

                result = self.migrate_file(file_path)
                stats['details'].append(result)

                if result['status'] == 'success':
                    stats['success'] += 1
                else:
                    stats['failed'] += 1

            return stats

        except Exception as e:
            worker_logger.error(f"Error during migration: {str(e)}")
            return stats

def migrate_to_s3(base_path: str = None):
    """Main migration function"""
    try:
        # Load full configuration
        worker_logger.info("Loading configuration...")
        config = load_full_config()
        
        # Get base path from config if not provided
        if base_path is None:
            base_path = config['storage']['nas']['mount_point']
            base_path = str(Path(base_path) / "data" / "downloads")
        worker_logger.info(f"Using base path: {base_path}")
            
        # Verify base path exists
        base_path = Path(base_path)
        if not base_path.exists():
            worker_logger.error(f"Base path does not exist: {base_path}")
            worker_logger.error("Please check the path and try again")
            return None
            
        # Run migration
        worker_logger.info(f"Starting migration from: {base_path}")
        stats = StorageMigration(base_path).migrate_all()
        
        # Log results
        task_logger.info(f"Migration complete: {stats['success']}/{stats['total']} successful, {stats['failed']} failed, {stats['skipped']} skipped")
        
        # Log details of failed items
        if stats['failed'] > 0:
            worker_logger.info("\nFailed items:")
            for detail in stats['details']:
                if detail['status'] == 'error':
                    worker_logger.info(f"  {detail['file']}: {detail['error']}")
        
        return stats
        
    except Exception as e:
        worker_logger.error(f"Unexpected error during migration: {str(e)}", exc_info=True)
        return None

if __name__ == '__main__':
    # Load config and get base path
    config = load_full_config()
    base_path = str(Path(config['storage']['nas']['mount_point']) / "data" / "downloads")
    
    # Run migration
    task_logger.info("Starting S3 migration process")
    stats = migrate_to_s3(base_path)
    
    if stats is None:
        task_logger.error("Migration failed to complete")
        exit(1) 