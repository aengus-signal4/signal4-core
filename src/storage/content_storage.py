"""
Content storage manager optimized for high-volume chunk processing.
"""
from typing import Optional, Dict, List, Tuple
import os
import json
from pathlib import Path
import logging
from .s3_utils import S3Storage, S3StorageConfig
from .config import get_storage_config

logger = logging.getLogger(__name__)

class ContentStorageManager:
    """Manages content storage operations using S3"""
    
    def __init__(self, s3_storage: S3Storage):
        self.s3_storage = s3_storage
        self.config = get_storage_config()  # This now returns just the s3 section
        
        # Define standard paths
        self.content_prefix = "content"  # All content is stored under this prefix
        
    def _get_content_path(self, content_id: str) -> str:
        """Get the base path for content metadata and source"""
        return f"{self.content_prefix}/{content_id}"
    
    def _get_chunk_path(self, content_id: str, chunk_index: int) -> str:
        """Get the path for a chunk directory"""
        return f"{self._get_content_path(content_id)}/chunks/{chunk_index}"
    
    def get_content_paths(self, content_id: str) -> Dict[str, str]:
        """Get all paths for a content item"""
        content_base = self._get_content_path(content_id)
        return {
            'meta': f"{content_base}/meta.json",
            'source': f"{content_base}/source",  # Extension added during upload
            'transcript': f"{content_base}/transcript_words.json"  # Updated to use transcript_words.json
        }
    
    def get_chunk_paths(self, content_id: str, chunk_index: int) -> Dict[str, str]:
        """Get all paths for a chunk"""
        chunk_base = f"content/{content_id}/chunks/{chunk_index}"
        return {
            'audio': f"{chunk_base}/audio.wav",
            'meta': f"{chunk_base}/meta.json",
            'transcript': f"{chunk_base}/transcript_words.json"
        }
    
    def upload_content_metadata(self, content_id: str, metadata: Dict) -> bool:
        """Upload content metadata"""
        paths = self.get_content_paths(content_id)
        with open('temp_meta.json', 'w') as f:
            json.dump(metadata, f)
        success = self.s3_storage.upload_file('temp_meta.json', paths['meta'])
        os.remove('temp_meta.json')
        return success
    
    def upload_source(self, content_id: str, local_path: str) -> bool:
        """Upload source file"""
        paths = self.get_content_paths(content_id)
        ext = Path(local_path).suffix
        return self.s3_storage.upload_file(local_path, f"{paths['source']}{ext}")
    
    def upload_chunk(self, content_id: str, chunk_index: int, source_path: str, start_time: float, end_time: float) -> bool:
        """Upload an audio chunk with metadata"""
        # First create and upload chunk metadata
        chunk_meta = {
            'content_id': content_id,
            'chunk_index': chunk_index,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
        
        # Create temp meta file
        import tempfile
        import json
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(chunk_meta, f)
            meta_path = f.name
            
        try:
            # Upload metadata first
            meta_key = f"content/{content_id}/chunks/{chunk_index}/meta.json"
            if not self.s3_storage.upload_file(meta_path, meta_key):
                return False
                
            # Then upload chunk
            chunk_key = f"content/{content_id}/chunks/{chunk_index}/audio.wav"
            return self.s3_storage.upload_file(source_path, chunk_key)
            
        finally:
            # Clean up temp meta file
            if os.path.exists(meta_path):
                os.unlink(meta_path)
    
    def upload_chunk_transcript(self, content_id: str, chunk_index: int,
                              local_path: str) -> bool:
        """Upload a chunk transcript"""
        paths = self.get_chunk_paths(content_id, chunk_index)
        return self.s3_storage.upload_file(local_path, paths['transcript'])
    
    def upload_final_transcript(self, content_id: str, local_path: str) -> bool:
        """Upload final stitched transcript"""
        paths = self.get_content_paths(content_id)
        logger.debug(f"Uploading final transcript from {local_path} to {paths['transcript']}")
        success = self.s3_storage.upload_file(local_path, paths['transcript'])
        if success:
            logger.debug(f"Successfully uploaded final transcript to {paths['transcript']}")
        else:
            logger.error(f"Failed to upload final transcript to {paths['transcript']}")
        return success
    
    def download_chunk(self, content_id: str, chunk_index: int,
                      local_path: str) -> bool:
        """
        Download a chunk file
        Returns: True if successful, False if not
        """
        try:
            paths = self.get_chunk_paths(content_id, chunk_index)
            
            # Create temp directory if it doesn't exist
            temp_dir = Path("/tmp/whisper_chunks")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Download audio directly
            if not self.s3_storage.download_file(paths['audio'], local_path):
                logger.error(f"Failed to download audio for chunk {chunk_index} of {content_id}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in download_chunk: {e}")
            return False
    
    def download_chunk_transcript(self, content_id: str, chunk_index: int,
                                local_path: str) -> bool:
        """Download a chunk transcript"""
        paths = self.get_chunk_paths(content_id, chunk_index)
        return self.s3_storage.download_file(paths['transcript'], local_path)
    
    def list_chunk_transcripts(self, content_id: str,
                             start_chunk: int = 0,
                             end_chunk: Optional[int] = None) -> List[Dict]:
        """
        List chunk transcripts for a content item.
        Optionally specify chunk range to list.
        """
        content_path = self._get_content_path(content_id)
        chunks_path = f"{content_path}/chunks"
        
        # If end_chunk is specified, optimize the listing operation
        if end_chunk is not None:
            results = []
            for chunk_idx in range(start_chunk, end_chunk + 1):
                chunk_paths = self.get_chunk_paths(content_id, chunk_idx)
                if self.s3_storage.file_exists(chunk_paths['transcript']):
                    results.append({
                        'chunk_index': chunk_idx,
                        'path': chunk_paths['transcript']
                    })
            return results
        
        # Otherwise, list all chunks
        return self.s3_storage.list_files(prefix=chunks_path)
    
    def delete_content(self, content_id: str) -> bool:
        """Delete all files associated with a content ID"""
        # Delete content metadata and source
        content_path = self._get_content_path(content_id)
        content_files = self.s3_storage.list_files(prefix=content_path)
        
        success = True
        for file_info in content_files:
            if not self.s3_storage.delete_file(file_info['key']):
                success = False
                
        return success
    
    def get_chunk_url(self, content_id: str, chunk_index: int,
                     expires_in: int = 3600) -> Optional[str]:
        """Get a pre-signed URL for a chunk file"""
        paths = self.get_chunk_paths(content_id, chunk_index)
        return self.s3_storage.get_file_url(paths['audio'], expires_in)
    
    def get_transcript_url(self, content_id: str,
                         expires_in: int = 3600) -> Optional[str]:
        """Get a pre-signed URL for the final transcript"""
        paths = self.get_content_paths(content_id)
        return self.s3_storage.get_file_url(paths['transcript'], expires_in)
    
    def download_source(self, content_id: str, target_path: str) -> bool:
        """Download source file for content"""
        # Try common extensions in order of likelihood
        extensions = ['.mp4', '.mp3', '.wav', '.m4a', '.webm', '.mkv', '.avi', '.mov']
        
        for ext in extensions:
            source_key = f"content/{content_id}/source{ext}"
            if self.s3_storage.file_exists(source_key):
                return self.s3_storage.download_file(source_key, target_path)
                
        # If no file found with any extension, try without extension as fallback
        source_key = f"content/{content_id}/source"
        if self.s3_storage.file_exists(source_key):
            return self.s3_storage.download_file(source_key, target_path)
            
        logger.error(f"No source file found for content {content_id}")
        return False
    
    def get_processed_audio_path(self, content_id: str) -> str:
        """Get the S3 path for the main processed audio file (e.g., audio.wav)."""
        content_base = self._get_content_path(content_id)
        # Assuming the standard processed audio filename is audio.wav
        return f"{content_base}/audio.wav"

    def get_speaker_mapping_path(self, content_id: str) -> str:
        """Get the S3 path for the speaker mapping file."""
        content_base = self._get_content_path(content_id)
        return f"{content_base}/speaker_mapping.json"

    def get_diarization_path(self, content_id: str) -> str:
        """Get the S3 path for the diarization file."""
        content_base = self._get_content_path(content_id)
        return f"{content_base}/diarization.json"
    
    def get_speaker_embeddings_path(self, content_id: str) -> str:
        """Get the S3 path for the speaker embeddings file."""
        content_base = self._get_content_path(content_id)
        return f"{content_base}/speaker_embeddings.json"

    def get_chunk_path(self, content_id: str, chunk_index: int) -> str:
        """Get S3 path for a chunk"""
        return f"content/{content_id}/chunks/{chunk_index}/audio.wav" 