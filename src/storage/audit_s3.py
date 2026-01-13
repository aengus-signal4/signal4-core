"""
S3 storage audit script to verify consistency between database and S3 storage.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from ..database.models import Content, ContentChunk
from .content_storage import ContentStorageManager
from .s3_utils import S3Storage, S3StorageConfig
from .config import get_storage_config

logger = logging.getLogger(__name__)

class S3Auditor:
    """Audits S3 storage against database state"""
    
    def __init__(self, storage_manager: ContentStorageManager):
        self.storage = storage_manager
        self.config = get_storage_config()
        
        # Setup database connection
        db_config = self.config['database']
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
        self.engine = create_engine(db_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def check_content(self, content_id: str) -> Dict:
        """
        Check a single content item's consistency
        
        Returns:
            Dict with audit results
        """
        results = {
            'content_id': content_id,
            'exists_in_db': False,
            'metadata': {'exists': False, 'matches': False},
            'source': {'exists': False},
            'chunks': [],
            'final_transcript': {'exists': False},
            'issues': []
        }
        
        try:
            # Get content from database
            content = self.session.query(Content).filter_by(content_id=content_id).first()
            if not content:
                results['issues'].append('Content not found in database')
                return results
            
            results['exists_in_db'] = True
            
            # Check paths
            paths = self.storage.get_content_paths(content_id)
            
            # Check metadata
            if self.storage.s3.file_exists(paths['meta']):
                results['metadata']['exists'] = True
                # TODO: Add metadata comparison logic
            else:
                results['issues'].append('Missing metadata in S3')
            
            # Check source file
            if self.storage.s3.file_exists(f"{paths['source']}.mp4") or \
               self.storage.s3.file_exists(f"{paths['source']}.mp3"):
                results['source']['exists'] = True
            elif content.is_downloaded:
                results['issues'].append('Source marked as downloaded but missing in S3')
            
            # Check chunks
            chunks = self.session.query(ContentChunk).filter_by(content_id=content.id).all()
            for chunk in chunks:
                chunk_paths = self.storage.get_chunk_paths(content_id, chunk.chunk_index)
                chunk_result = {
                    'index': chunk.chunk_index,
                    'audio_exists': self.storage.s3.file_exists(chunk_paths['audio']),
                    'meta_exists': self.storage.s3.file_exists(chunk_paths['meta']),
                    'transcript_exists': self.storage.s3.file_exists(chunk_paths['transcript']),
                    'status_matches': True  # Default to True, update below
                }
                
                # Check if S3 state matches database state
                if chunk.status == 'completed' and not chunk_result['transcript_exists']:
                    chunk_result['status_matches'] = False
                    results['issues'].append(
                        f'Chunk {chunk.chunk_index} marked completed but missing transcript'
                    )
                
                results['chunks'].append(chunk_result)
            
            # Check final transcript
            if self.storage.s3.file_exists(paths['final']):
                results['final_transcript']['exists'] = True
            elif content.is_transcribed:
                results['issues'].append('Content marked as transcribed but missing final transcript')
            
            return results
            
        except Exception as e:
            results['issues'].append(f'Error during audit: {str(e)}')
            logger.error(f"Error auditing content {content_id}: {str(e)}")
            return results
    
    def audit_all(self, batch_size: int = 100) -> Dict:
        """
        Audit all content
        
        Returns:
            Dict with audit statistics
        """
        stats = {
            'total_content': 0,
            'content_with_issues': 0,
            'total_chunks': 0,
            'chunks_with_issues': 0,
            'issues_by_type': {},
            'content_details': []
        }
        
        # Get all content IDs from database
        content_ids = [r[0] for r in self.session.query(Content.content_id).all()]
        stats['total_content'] = len(content_ids)
        
        # Process in batches with progress bar
        with tqdm(total=len(content_ids), desc="Auditing content") as pbar:
            for i in range(0, len(content_ids), batch_size):
                batch = content_ids[i:i + batch_size]
                for content_id in batch:
                    results = self.check_content(content_id)
                    
                    # Update statistics
                    if results['issues']:
                        stats['content_with_issues'] += 1
                        for issue in results['issues']:
                            stats['issues_by_type'][issue] = stats['issues_by_type'].get(issue, 0) + 1
                    
                    stats['total_chunks'] += len(results['chunks'])
                    stats['chunks_with_issues'] += sum(
                        1 for c in results['chunks'] 
                        if not (c['audio_exists'] and c['meta_exists'] and c['transcript_exists'])
                    )
                    
                    stats['content_details'].append(results)
                    pbar.update(1)
        
        return stats

def audit_s3_storage():
    """Main audit function"""
    # Initialize storage
    config = get_storage_config()
    s3_config = S3StorageConfig(
        endpoint_url=config['s3']['endpoint_url'],
        access_key=config['s3']['access_key'],
        secret_key=config['s3']['secret_key'],
        region=config['s3']['region'],
        bucket_name=config['s3']['bucket_name'],
        use_ssl=config['s3']['use_ssl']
    )
    s3_storage = S3Storage(s3_config)
    storage_manager = ContentStorageManager(s3_storage)
    
    # Run audit
    auditor = S3Auditor(storage_manager)
    stats = auditor.audit_all()
    
    # Log results
    logger.info("\n=== S3 Storage Audit Results ===")
    logger.info(f"Total content items: {stats['total_content']}")
    logger.info(f"Content items with issues: {stats['content_with_issues']}")
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Chunks with issues: {stats['chunks_with_issues']}")
    
    if stats['issues_by_type']:
        logger.info("\nIssues by type:")
        for issue, count in stats['issues_by_type'].items():
            logger.info(f"  {issue}: {count}")
    
    return stats

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    audit_s3_storage() 