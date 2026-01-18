#!/usr/bin/env python3
"""
Cleanup and Compress S3 Storage
================================

Final cleanup step that:
1. Compresses audio files to Opus format
2. Compresses video files to efficient format
3. Deletes chunk audio files (no longer needed)
4. Compresses JSON files
5. Creates a manifest for quick file checking

This step runs after segment_embeddings to optimize storage.
"""

# Centralized environment setup (must be before other imports)
from src.utils.env_setup import setup_env
setup_env()

import sys
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.config import load_config
import asyncio
import logging
import json
import yaml
import gzip
import shutil
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import subprocess
import tempfile
import os

sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.utils.error_codes import (
    ErrorCode, create_error_result, create_success_result, create_skipped_result
)
from src.database.session import get_session
from src.database.models import Content
from src.storage.s3_utils import S3Storage, S3StorageConfig

logger = setup_worker_logger('cleanup_compress')

class StorageOptimizer:
    """Optimizes S3 storage by compressing files and removing unnecessary data."""
    
    def __init__(self):
        """Initialize the storage optimizer."""
        # Load config
        self.config = load_config()
        
        # Initialize S3 storage
        s3_config = S3StorageConfig(
            endpoint_url=self.config['storage']['s3']['endpoint_url'],
            access_key=self.config['storage']['s3']['access_key'],
            secret_key=self.config['storage']['s3']['secret_key'],
            bucket_name=self.config['storage']['s3']['bucket_name'],
            use_ssl=self.config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)
        
        # Compression settings
        compress_config = self.config.get('processing', {}).get('compression', {})
        self.audio_codec = compress_config.get('audio_codec', 'opus')
        self.audio_bitrate = compress_config.get('audio_bitrate', '64k')
        self.video_codec = compress_config.get('video_codec', 'h265')
        self.video_crf = compress_config.get('video_crf', '28')
        self.video_preset = compress_config.get('video_preset', 'medium')
        self.opus_vbr = compress_config.get('opus_vbr', True)
        self.opus_compression_level = compress_config.get('opus_compression_level', 10)
        self.delete_chunks = compress_config.get('delete_chunks', True)
        self.compress_json = compress_config.get('compress_json', True)
        self.min_space_savings = compress_config.get('min_space_savings', 0.3)
        self.preserve_original_on_failure = compress_config.get('preserve_original_on_failure', True)
        
        logger.info(f"Initialized with audio_codec={self.audio_codec}, video_codec={self.video_codec}")
    
    def _audit_s3_files(self, content_id: str, strict: bool = True) -> Dict[str, Any]:
        """Audit S3 files for content.
        
        Args:
            content_id: Content ID to audit
            strict: If True, requires all files for compression. If False, just catalogs what's available.
        """
        audit_result = {
            'content_id': content_id,
            'status': 'pass',
            'errors': [],
            'warnings': [],
            'found_files': [],
            'missing_files': [],
            'file_checks': {},
            'compressible_files': {},  # Files that can be compressed
            'already_compressed': False
        }
        
        try:
            content_prefix = f"content/{content_id}/"
            
            # Get all files for this content
            all_files = set(self.s3_storage.list_files(content_prefix))
            audit_result['found_files'] = sorted(list(all_files))
            
            logger.info(f"[{content_id}] Found {len(all_files)} files in S3")
            
            # Check if already compressed
            storage_manifest_exists = f"{content_prefix}storage_manifest.json" in all_files
            audit_result['already_compressed'] = storage_manifest_exists
            
            # Catalog what files we have that can be processed
            compressible_files = {}
            
            # Check for audio files (original or compressed)
            audio_wav = f"{content_prefix}audio.wav"
            audio_opus = f"{content_prefix}audio.opus"
            audio_mp3 = f"{content_prefix}audio.mp3"
            
            if audio_wav in all_files:
                compressible_files['audio_wav'] = audio_wav
            elif audio_opus in all_files:
                compressible_files['audio_opus'] = audio_opus
            elif audio_mp3 in all_files:
                compressible_files['audio_mp3'] = audio_mp3
            
            # Check for source media files
            source_extensions = ['.mp4', '.mp3', '.wav', '.m4a']
            for ext in source_extensions:
                source_file = f"{content_prefix}source{ext}"
                if source_file in all_files:
                    compressible_files['source_media'] = source_file
                    break
            
            # Check for video files  
            video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv']
            for ext in video_extensions:
                video_file = f"{content_prefix}video{ext}"
                if video_file in all_files:
                    compressible_files['video'] = video_file
                    break
            
            # Check for JSON files (original or compressed)
            json_files = [f for f in all_files if f.startswith(content_prefix) and f.endswith('.json')]
            json_gz_files = [f for f in all_files if f.startswith(content_prefix) and f.endswith('.json.gz')]
            
            for json_file in json_files:
                file_name = json_file[len(content_prefix):]
                compressible_files[file_name] = json_file
                
            for json_gz_file in json_gz_files:
                file_name = json_gz_file[len(content_prefix):]
                compressible_files[file_name] = json_gz_file
            
            # Count chunk files
            chunk_audio_files = [f for f in all_files if '/chunks/' in f and f.endswith('/audio.wav')]
            chunk_transcript_files = [f for f in all_files if '/chunks/' in f and f.endswith('/transcript_words.json')]
            
            if chunk_audio_files:
                compressible_files['chunk_audio_count'] = len(chunk_audio_files)
            if chunk_transcript_files:
                compressible_files['chunk_transcript_count'] = len(chunk_transcript_files)
            
            audit_result['compressible_files'] = compressible_files
            
            # Determine what we can do based on what's available
            can_compress_audio = 'audio_wav' in compressible_files
            can_compress_video = 'video' in compressible_files  
            can_compress_json = any(f.endswith('.json') for f in compressible_files.keys())
            can_cleanup_chunks = chunk_audio_files and chunk_transcript_files
            
            # Set status based on mode
            if strict:
                # For strict mode (compression), we need at least something to compress
                if not (can_compress_audio or can_compress_video or can_compress_json):
                    audit_result['status'] = 'fail'
                    audit_result['errors'].append("No compressible files found")
                elif storage_manifest_exists:
                    audit_result['warnings'].append("Content appears to already be compressed (manifest exists)")
            else:
                # For non-strict mode (decompress, force), just catalog what's available
                audit_result['status'] = 'pass'
            
            # Add warnings for chunk mismatches
            if len(chunk_audio_files) != len(chunk_transcript_files):
                audit_result['warnings'].append(
                    f"Chunk count mismatch: {len(chunk_audio_files)} audio files vs "
                    f"{len(chunk_transcript_files)} transcript files"
                )
            
            # Summary logging
            compressible_count = len([k for k in compressible_files.keys() if not k.endswith('_count')])
            logger.info(f"[{content_id}] Found {compressible_count} compressible file types")
            
            if storage_manifest_exists:
                logger.info(f"[{content_id}] Content is already compressed (manifest exists)")
            
            if audit_result['warnings']:
                logger.warning(f"[{content_id}] Audit has {len(audit_result['warnings'])} warnings:")
                for warning in audit_result['warnings']:
                    logger.warning(f"  - {warning}")
            
            if audit_result['errors']:
                logger.error(f"[{content_id}] Audit FAILED with {len(audit_result['errors'])} errors:")
                for error in audit_result['errors']:
                    logger.error(f"  - {error}")
            
            return audit_result
            
        except Exception as e:
            audit_result['status'] = 'error'
            audit_result['errors'].append(f"Audit failed with exception: {str(e)}")
            logger.error(f"[{content_id}] S3 audit failed: {e}", exc_info=True)
            return audit_result
    
    def _compress_audio_to_opus(self, input_path: Path, output_path: Path) -> Tuple[bool, Path]:
        """Compress audio file to Opus format using dedicated tools.
        
        Returns:
            Tuple of (success: bool, actual_output_path: Path)
        """
        try:
            # First try using dedicated opusenc tool (best quality and compatibility)
            if self._try_opusenc_compression(input_path, output_path):
                return True, output_path
            
            logger.warning("opusenc not available, trying FFmpeg with Opus")
            
            # Fall back to FFmpeg with basic Opus options
            if self._try_ffmpeg_opus_compression(input_path, output_path):
                return True, output_path
            
            logger.warning("FFmpeg Opus failed, falling back to MP3")
            
            # Final fallback to MP3 compression
            mp3_output = output_path.with_suffix('.mp3')
            if self._try_mp3_compression(input_path, mp3_output):
                return True, mp3_output
            
            logger.error("All audio compression methods failed")
            return False, input_path
            
        except Exception as e:
            logger.error(f"Error in audio compression: {e}")
            return False, input_path
    
    def _try_opusenc_compression(self, input_path: Path, output_path: Path) -> bool:
        """Try compression using dedicated opusenc tool."""
        try:
            # Convert bitrate from FFmpeg format (e.g., "64k") to kbps number
            bitrate_kbps = self.audio_bitrate.replace('k', '').replace('K', '')
            
            # Try full path first, then fall back to PATH
            opusenc_paths = ['/opt/homebrew/bin/opusenc', 'opusenc']
            
            for opusenc_cmd in opusenc_paths:
                try:
                    # Build opusenc command optimized for speech
                    cmd = [
                        opusenc_cmd,
                        '--bitrate', bitrate_kbps,
                        '--comp', str(self.opus_compression_level),  # Compression level 0-10
                        '--framesize', '60',  # 60ms frames for speech
                        '--speech',  # Optimize for speech content
                    ]
                    
                    # Add VBR if enabled (before input/output files)
                    if self.opus_vbr:
                        cmd.append('--vbr')
                        
                    # Add input and output files
                    cmd.extend([str(input_path), str(output_path)])
                    
                    logger.debug(f"Running opusenc: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and output_path.exists():
                        self._log_compression_stats(input_path, output_path, "opusenc")
                        return True
                    else:
                        logger.warning(f"opusenc failed (cmd: {opusenc_cmd}): {result.stderr}")
                        continue  # Try next path
                        
                except FileNotFoundError:
                    logger.debug(f"opusenc not found at: {opusenc_cmd}")
                    continue
                    
            # If we get here, none of the paths worked
            logger.debug("opusenc not found in any expected location")
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("opusenc timed out after 5 minutes")
            return False
        except Exception as e:
            logger.warning(f"opusenc error: {e}")
            return False
    
    def _try_ffmpeg_opus_compression(self, input_path: Path, output_path: Path) -> bool:
        """Try compression using FFmpeg with basic Opus options."""
        try:
            # Try full path first, then fall back to PATH
            ffmpeg_paths = ['/opt/homebrew/bin/ffmpeg', 'ffmpeg']
            
            for ffmpeg_cmd in ffmpeg_paths:
                try:
                    cmd = [
                        ffmpeg_cmd, '-i', str(input_path),
                        '-c:a', 'libopus',
                        '-b:a', self.audio_bitrate,
                        '-application', 'voip',  # Optimize for speech
                        '-y',  # Overwrite output
                        str(output_path)
                    ]
                    
                    logger.debug(f"Running FFmpeg Opus: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and output_path.exists():
                        self._log_compression_stats(input_path, output_path, "FFmpeg Opus")
                        return True
                    else:
                        logger.warning(f"FFmpeg Opus failed (cmd: {ffmpeg_cmd}): {result.stderr}")
                        logger.warning(f"FFmpeg stdout: {result.stdout}")
                        continue  # Try next path
                        
                except FileNotFoundError:
                    logger.debug(f"ffmpeg not found at: {ffmpeg_cmd}")
                    continue
                    
            # If we get here, none of the paths worked
            logger.debug("ffmpeg not found in any expected location")
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out after 5 minutes")
            return False
        except Exception as e:
            logger.warning(f"FFmpeg Opus error: {e}")
            return False
    
    def _try_mp3_compression(self, input_path: Path, output_path: Path) -> bool:
        """Try compression using MP3 as final fallback."""
        try:
            # Try full path first, then fall back to PATH  
            ffmpeg_paths = ['/opt/homebrew/bin/ffmpeg', 'ffmpeg']
            
            for ffmpeg_cmd in ffmpeg_paths:
                try:
                    cmd = [
                        ffmpeg_cmd, '-i', str(input_path),
                        '-c:a', 'libmp3lame',
                        '-b:a', self.audio_bitrate,
                        '-q:a', '2',  # High quality VBR
                        '-y',  # Overwrite output
                        str(output_path)
                    ]
                    
                    logger.debug(f"Running FFmpeg MP3: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and output_path.exists():
                        self._log_compression_stats(input_path, output_path, "MP3 fallback")
                        return True
                    else:
                        logger.error(f"MP3 compression failed (cmd: {ffmpeg_cmd}): {result.stderr}")
                        logger.error(f"MP3 stdout: {result.stdout}")
                        continue  # Try next path
                        
                except FileNotFoundError:
                    logger.debug(f"ffmpeg not found at: {ffmpeg_cmd}")
                    continue
                    
            # If we get here, none of the paths worked
            logger.error("ffmpeg not found in any expected location for MP3 compression")
            return False
            
        except Exception as e:
            logger.error(f"MP3 compression error: {e}")
            return False
    
    def _log_compression_stats(self, input_path: Path, output_path: Path, method: str):
        """Log compression statistics."""
        input_size = input_path.stat().st_size
        output_size = output_path.stat().st_size
        compression_ratio = (1 - output_size / input_size) * 100
        
        logger.info(f"Audio compressed using {method}: {input_size:,} -> {output_size:,} bytes "
                   f"({compression_ratio:.1f}% reduction)")
    
    def _decode_opus_to_wav(self, opus_path: str, wav_path: str) -> bool:
        """Decode Opus file to WAV using best available tool."""
        try:
            # First try dedicated opusdec tool
            if self._try_opusdec_decoding(opus_path, wav_path):
                return True
            
            logger.warning("opusdec not available, trying FFmpeg")
            
            # Fall back to FFmpeg
            return self._try_ffmpeg_opus_decoding(opus_path, wav_path)
            
        except Exception as e:
            logger.error(f"Error decoding Opus: {e}")
            return False
    
    def _try_opusdec_decoding(self, opus_path: str, wav_path: str) -> bool:
        """Try decoding using dedicated opusdec tool."""
        try:
            # Try full path first, then fall back to PATH
            opusdec_paths = ['/opt/homebrew/bin/opusdec', 'opusdec']
            
            for opusdec_cmd in opusdec_paths:
                try:
                    cmd = [
                        opusdec_cmd,
                        '--rate', '16000',  # Decode to 16kHz for ML processing
                        opus_path,
                        wav_path
                    ]
                    
                    logger.debug(f"Running opusdec: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and Path(wav_path).exists():
                        logger.debug(f"Successfully decoded Opus to WAV using opusdec")
                        return True
                    else:
                        logger.warning(f"opusdec failed (cmd: {opusdec_cmd}): {result.stderr}")
                        continue  # Try next path
                        
                except FileNotFoundError:
                    logger.debug(f"opusdec not found at: {opusdec_cmd}")
                    continue
                    
            # If we get here, none of the paths worked
            logger.debug("opusdec not found in any expected location")
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("opusdec timed out after 5 minutes")
            return False
        except Exception as e:
            logger.warning(f"opusdec error: {e}")
            return False
    
    def _try_ffmpeg_opus_decoding(self, opus_path: str, wav_path: str) -> bool:
        """Try decoding using FFmpeg."""
        try:
            # Try full path first, then fall back to PATH
            ffmpeg_paths = ['/opt/homebrew/bin/ffmpeg', 'ffmpeg']
            
            for ffmpeg_cmd in ffmpeg_paths:
                try:
                    cmd = [
                        ffmpeg_cmd, '-i', opus_path,
                        '-c:a', 'pcm_s16le',
                        '-ar', '16000',  # 16kHz for ML processing
                        '-y', wav_path
                    ]
                    
                    logger.debug(f"Running FFmpeg decode: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and Path(wav_path).exists():
                        logger.debug(f"Successfully decoded Opus to WAV using FFmpeg")
                        return True
                    else:
                        logger.error(f"FFmpeg decode failed (cmd: {ffmpeg_cmd}): {result.stderr}")
                        logger.error(f"FFmpeg decode stdout: {result.stdout}")
                        continue  # Try next path
                        
                except FileNotFoundError:
                    logger.debug(f"ffmpeg not found at: {ffmpeg_cmd}")
                    continue
                    
            # If we get here, none of the paths worked
            logger.error("ffmpeg not found in any expected location for Opus decoding")
            return False
            
        except Exception as e:
            logger.error(f"FFmpeg decode error: {e}")
            return False
    
    def _decode_mp3_to_wav(self, mp3_path: str, wav_path: str) -> bool:
        """Decode MP3 file to WAV using FFmpeg."""
        try:
            # Try full path first, then fall back to PATH
            ffmpeg_paths = ['/opt/homebrew/bin/ffmpeg', 'ffmpeg']
            
            for ffmpeg_cmd in ffmpeg_paths:
                try:
                    cmd = [
                        ffmpeg_cmd, '-i', mp3_path,
                        '-c:a', 'pcm_s16le',
                        '-ar', '16000',  # 16kHz for ML processing
                        '-y', wav_path
                    ]
                    
                    logger.debug(f"Running FFmpeg MP3 decode: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and Path(wav_path).exists():
                        logger.debug(f"Successfully decoded MP3 to WAV using FFmpeg")
                        return True
                    else:
                        logger.error(f"FFmpeg MP3 decode failed (cmd: {ffmpeg_cmd}): {result.stderr}")
                        logger.error(f"FFmpeg MP3 decode stdout: {result.stdout}")
                        continue  # Try next path
                        
                except FileNotFoundError:
                    logger.debug(f"ffmpeg not found at: {ffmpeg_cmd}")
                    continue
                    
            # If we get here, none of the paths worked
            logger.error("ffmpeg not found in any expected location for MP3 decoding")
            return False
            
        except Exception as e:
            logger.error(f"FFmpeg MP3 decode error: {e}")
            return False
    
    def _compress_video(self, input_path: Path, output_path: Path) -> bool:
        """Compress video file using modern codec."""
        try:
            # Determine codec based on configuration
            if self.video_codec == 'h265':
                codec_name = 'libx265'
                ext = '.mp4'
            elif self.video_codec == 'av1':
                codec_name = 'libaom-av1'
                ext = '.webm'
            else:
                codec_name = 'libx264'
                ext = '.mp4'
            
            # Ensure output has correct extension
            if not str(output_path).endswith(ext):
                output_path = output_path.with_suffix(ext)
            
            # Try full path first, then fall back to PATH
            ffmpeg_paths = ['/opt/homebrew/bin/ffmpeg', 'ffmpeg']
            
            for ffmpeg_cmd in ffmpeg_paths:
                try:
                    cmd = [
                        ffmpeg_cmd, '-i', str(input_path),
                        '-c:v', codec_name,
                        '-crf', str(self.video_crf),
                        '-preset', self.video_preset,
                        '-c:a', 'aac',  # Keep audio in AAC
                        '-b:a', '128k',
                        '-movflags', '+faststart',  # For streaming
                        '-y',
                        str(output_path)
                    ]
                    
                    logger.debug(f"Running FFmpeg video compression: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Check compression results
                        input_size = input_path.stat().st_size
                        output_size = output_path.stat().st_size
                        compression_ratio = (1 - output_size / input_size) * 100
                        
                        logger.info(f"Compressed video: {input_size:,} -> {output_size:,} bytes "
                                   f"({compression_ratio:.1f}% reduction)")
                        
                        return True
                    else:
                        logger.error(f"FFmpeg video compression failed (cmd: {ffmpeg_cmd}): {result.stderr}")
                        logger.error(f"FFmpeg video stdout: {result.stdout}")
                        continue  # Try next path
                        
                except FileNotFoundError:
                    logger.debug(f"ffmpeg not found at: {ffmpeg_cmd}")
                    continue
                    
            # If we get here, none of the paths worked
            logger.error("ffmpeg not found in any expected location for video compression")
            return False
            
        except Exception as e:
            logger.error(f"Error compressing video: {e}")
            return False
    
    def _compress_json(self, input_path: Path, output_path: Path) -> bool:
        """Compress JSON file using gzip."""
        try:
            with open(input_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb', compresslevel=9) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            input_size = input_path.stat().st_size
            output_size = output_path.stat().st_size
            compression_ratio = (1 - output_size / input_size) * 100
            
            logger.info(f"Compressed JSON: {input_size:,} -> {output_size:,} bytes "
                       f"({compression_ratio:.1f}% reduction)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error compressing JSON: {e}")
            return False
    
    def _decompress_json(self, compressed_path: Path) -> Optional[Dict]:
        """Decompress and load JSON file."""
        try:
            with gzip.open(compressed_path, 'rt') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error decompressing JSON: {e}")
            return None
    
    def _create_manifest(self, content_id: str, file_info: Dict[str, Dict]) -> Dict:
        """Create manifest with file information and checksums."""
        manifest = {
            "content_id": content_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "compression": {
                "audio_codec": self.audio_codec,
                "audio_bitrate": self.audio_bitrate,
                "video_codec": self.video_codec,
                "video_crf": self.video_crf,
                "json_compression": "gzip" if self.compress_json else "none"
            },
            "files": file_info,
            "statistics": {
                "original_size": sum(f.get('original_size', 0) for f in file_info.values()),
                "compressed_size": sum(f.get('size', 0) for f in file_info.values()),
                "chunk_files_deleted": len([f for f in file_info.values() if f.get('deleted', False)])
            }
        }
        
        # Calculate total savings
        total_saved = manifest["statistics"]["original_size"] - manifest["statistics"]["compressed_size"]
        manifest["statistics"]["space_saved"] = total_saved
        manifest["statistics"]["compression_ratio"] = (
            (total_saved / manifest["statistics"]["original_size"] * 100) 
            if manifest["statistics"]["original_size"] > 0 else 0
        )
        
        return manifest
    
    async def _process_audio_files(self, content_id: str, manifest_files: Dict) -> int:
        """Process and compress audio files."""
        processed = 0
        
        # Main audio file
        audio_key = f"content/{content_id}/audio.wav"
        if self.s3_storage.file_exists(audio_key):
            logger.info(f"[{content_id}] Processing main audio file")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download original
                local_wav = temp_path / "audio.wav"
                if self.s3_storage.download_file(audio_key, str(local_wav)):
                    original_size = local_wav.stat().st_size
                    
                    # Compress to opus (or MP3 fallback)
                    local_compressed = temp_path / "audio.opus"
                    success, actual_compressed_file = self._compress_audio_to_opus(local_wav, local_compressed)
                    
                    if success and actual_compressed_file != local_wav:
                        # Upload compressed version with correct extension
                        compressed_key = f"content/{content_id}/audio{actual_compressed_file.suffix}"
                        if self.s3_storage.upload_file(str(actual_compressed_file), compressed_key):
                            compressed_size = actual_compressed_file.stat().st_size
                            
                            # Determine format from file extension
                            file_format = actual_compressed_file.suffix.lstrip('.')
                            
                            # Update manifest
                            manifest_files[compressed_key] = {
                                "type": "audio",
                                "format": file_format,
                                "size": compressed_size,
                                "original_size": original_size,
                                "original_key": audio_key,
                                "compression_ratio": (1 - compressed_size / original_size) * 100
                            }
                            
                            # Delete original WAV
                            if self.s3_storage.delete_file(audio_key):
                                manifest_files[audio_key] = {"deleted": True, "reason": "compressed"}
                                logger.info(f"[{content_id}] Deleted original audio.wav")
                            
                            processed += 1
                        else:
                            logger.error(f"[{content_id}] Failed to upload compressed audio to S3")
                    else:
                        logger.warning(f"[{content_id}] Audio compression failed or no compression achieved")
        
        return processed
    
    async def _process_video_files(self, content_id: str, manifest_files: Dict) -> int:
        """Process and compress video files."""
        processed = 0
        
        # Check for video file (various possible formats)
        video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv']
        
        for ext in video_extensions:
            video_key = f"content/{content_id}/video{ext}"
            if self.s3_storage.file_exists(video_key):
                logger.info(f"[{content_id}] Processing video file: {video_key}")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Download original
                    local_video = temp_path / f"video{ext}"
                    if self.s3_storage.download_file(video_key, str(local_video)):
                        original_size = local_video.stat().st_size
                        
                        # Compress video
                        compressed_ext = '.mp4' if self.video_codec in ['h264', 'h265'] else '.webm'
                        local_compressed = temp_path / f"video_compressed{compressed_ext}"
                        
                        if self._compress_video(local_video, local_compressed):
                            # Upload compressed version
                            compressed_key = f"content/{content_id}/video{compressed_ext}"
                            if self.s3_storage.upload_file(str(local_compressed), compressed_key):
                                compressed_size = local_compressed.stat().st_size
                                
                                # Update manifest
                                manifest_files[compressed_key] = {
                                    "type": "video",
                                    "format": compressed_ext.lstrip('.'),
                                    "codec": self.video_codec,
                                    "size": compressed_size,
                                    "original_size": original_size,
                                    "original_key": video_key,
                                    "compression_ratio": (1 - compressed_size / original_size) * 100
                                }
                                
                                # Delete original if different
                                if video_key != compressed_key:
                                    if self.s3_storage.delete_file(video_key):
                                        manifest_files[video_key] = {"deleted": True, "reason": "compressed"}
                                        logger.info(f"[{content_id}] Deleted original {video_key}")
                                
                                processed += 1
                                break  # Only process first video found
        
        return processed
    
    async def _delete_chunk_files(self, content_id: str, manifest_files: Dict) -> int:
        """Delete chunk audio files that are no longer needed."""
        deleted = 0
        
        if not self.delete_chunks:
            logger.info(f"[{content_id}] Chunk deletion disabled, skipping")
            return deleted
        
        # List all chunk directories
        chunk_pattern = f"content/{content_id}/chunks/"
        
        try:
            # Get all files with this prefix using the list_files method
            chunk_files = self.s3_storage.list_files(chunk_pattern)
            
            for chunk_file in chunk_files:
                # Only delete .wav files in chunks
                if chunk_file.endswith('.wav'):
                    if self.s3_storage.delete_file(chunk_file):
                        manifest_files[chunk_file] = {
                            "deleted": True,
                            "reason": "chunk_cleanup"
                        }
                        deleted += 1
                        
                        if deleted % 50 == 0:
                            logger.info(f"[{content_id}] Deleted {deleted} chunk files...")
            
            logger.info(f"[{content_id}] Deleted total {deleted} chunk audio files")
            
        except Exception as e:
            logger.error(f"[{content_id}] Error listing/deleting chunks: {e}")
        
        return deleted
    
    async def _compress_json_files(self, content_id: str, manifest_files: Dict) -> int:
        """Compress JSON files in S3."""
        compressed = 0
        
        if not self.compress_json:
            logger.info(f"[{content_id}] JSON compression disabled")
            return compressed
        
        # Get all JSON files for this content
        content_prefix = f"content/{content_id}/"
        all_files = self.s3_storage.list_files(content_prefix)
        
        # Find all .json files (not already compressed)
        json_files = [f for f in all_files if f.endswith('.json') and not f.endswith('.json.gz')]
        
        logger.info(f"[{content_id}] Found {len(json_files)} JSON files to compress")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for json_key in json_files:
                # Download JSON
                local_json = temp_path / Path(json_key).name
                if self.s3_storage.download_file(json_key, str(local_json)):
                    original_size = local_json.stat().st_size
                    
                    # Compress
                    local_gz = local_json.with_suffix('.json.gz')
                    if self._compress_json(local_json, local_gz):
                        # Upload compressed version
                        gz_key = json_key + '.gz'
                        if self.s3_storage.upload_file(str(local_gz), gz_key):
                            compressed_size = local_gz.stat().st_size
                            
                            # Update manifest
                            manifest_files[gz_key] = {
                                "type": "json",
                                "format": "gzip",
                                "size": compressed_size,
                                "original_size": original_size,
                                "original_key": json_key,
                                "compression_ratio": (1 - compressed_size / original_size) * 100
                            }
                            
                            # Delete original
                            if self.s3_storage.delete_file(json_key):
                                manifest_files[json_key] = {"deleted": True, "reason": "compressed"}
                            
                            compressed += 1
        
        logger.info(f"[{content_id}] Compressed {compressed} JSON files")
        return compressed
    
    async def decompress_content(self, content_id: str, dry_run: bool = False) -> Dict:
        """Decompress content back to original format."""
        logger.info(f"[{content_id}] Starting decompression{' (DRY RUN)' if dry_run else ''}")
        
        try:
            # Audit what compressed files exist
            audit_result = self._audit_s3_files(content_id, strict=False)
            
            if not audit_result['already_compressed']:
                return create_error_result(
                    ErrorCode.NOT_FOUND,
                    'Content is not compressed (no storage manifest found)',
                    {'audit_result': audit_result}
                )
            
            # Download and parse manifest
            manifest_key = f"content/{content_id}/storage_manifest.json"
            manifest = None
            
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_manifest = f.name
            
            try:
                if self.s3_storage.download_file(manifest_key, temp_manifest):
                    with open(temp_manifest, 'r') as f:
                        manifest = json.load(f)
                    os.unlink(temp_manifest)
                else:
                    return create_error_result(
                        ErrorCode.S3_CONNECTION_ERROR,
                        'Could not download storage manifest',
                        {'audit_result': audit_result}
                    )
            except Exception as e:
                if os.path.exists(temp_manifest):
                    os.unlink(temp_manifest)
                return create_error_result(
                    ErrorCode.INVALID_FORMAT,
                    f'Could not parse storage manifest: {e}',
                    {'audit_result': audit_result}
                )
            
            if not manifest:
                return create_error_result(
                    ErrorCode.INVALID_FORMAT,
                    'Storage manifest is empty or invalid',
                    {'audit_result': audit_result}
                )
            
            logger.info(f"[{content_id}] Found storage manifest with {len(manifest.get('files', {}))} file entries")
            
            results = {
                'audio_decompressed': 0,
                'video_decompressed': 0, 
                'json_decompressed': 0,
                'files_restored': 0
            }
            
            if not dry_run:
                # Process each file in the manifest
                manifest_files = manifest.get('files', {})
                
                for s3_key, file_info in manifest_files.items():
                    if file_info.get('deleted'):
                        # This was a file that was deleted during compression
                        original_key = file_info.get('original_key')
                        if original_key:
                            logger.info(f"[{content_id}] File was deleted during compression: {original_key}")
                        continue
                    
                    # Determine original file path from compressed file
                    original_key = file_info.get('original_key')
                    file_type = file_info.get('type')
                    
                    if not original_key:
                        logger.warning(f"[{content_id}] No original key for compressed file: {s3_key}")
                        continue
                    
                    # Check if we need to decompress this file
                    if file_type == 'audio' and (s3_key.endswith('.opus') or s3_key.endswith('.mp3')):
                        success = await self._decompress_audio_file(content_id, s3_key, original_key)
                        if success:
                            results['audio_decompressed'] += 1
                    
                    elif file_type == 'json' and s3_key.endswith('.gz'):
                        success = await self._decompress_json_file(content_id, s3_key, original_key)
                        if success:
                            results['json_decompressed'] += 1
                    
                    elif file_type == 'video':
                        # For video, we might just copy it back if it was renamed
                        # (actual video decompression is usually not reversible)
                        success = await self._restore_video_file(content_id, s3_key, original_key)
                        if success:
                            results['video_decompressed'] += 1
                
                # Remove the manifest file
                if self.s3_storage.delete_file(manifest_key):
                    logger.info(f"[{content_id}] Removed storage manifest")
                
                # Update database
                with get_session() as session:
                    content = session.query(Content).filter_by(content_id=content_id).first()
                    if content:
                        content.is_compressed = False
                        session.commit()
                        logger.info(f"[{content_id}] Updated database: is_compressed=False")
            
            else:
                # Dry run - just report what would be done
                logger.info(f"[{content_id}] DRY RUN - Would decompress:")
                manifest_files = manifest.get('files', {})
                
                for s3_key, file_info in manifest_files.items():
                    if file_info.get('deleted'):
                        continue
                        
                    file_type = file_info.get('type')
                    original_key = file_info.get('original_key')
                    
                    if file_type == 'audio':
                        logger.info(f"  - Decompress {s3_key} -> {original_key}")
                        results['audio_decompressed'] = 1
                    elif file_type == 'json':
                        logger.info(f"  - Decompress {s3_key} -> {original_key}")
                        results['json_decompressed'] += 1
                    elif file_type == 'video':
                        logger.info(f"  - Restore {s3_key} -> {original_key}")
                        results['video_decompressed'] = 1
            
            return create_success_result({
                'content_id': content_id,
                'results': results,
                'dry_run': dry_run,
                'audit_result': audit_result
            })
            
        except Exception as e:
            logger.error(f"[{content_id}] Error in decompression: {e}", exc_info=True)
            return create_error_result(
                ErrorCode.UNKNOWN_ERROR,
                str(e),
                {'audit_result': audit_result if 'audit_result' in locals() else None}
            )
    
    async def _decompress_audio_file(self, content_id: str, compressed_key: str, original_key: str) -> bool:
        """Decompress an audio file from S3."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download compressed file
                compressed_file = temp_path / Path(compressed_key).name
                if not self.s3_storage.download_file(compressed_key, str(compressed_file)):
                    logger.error(f"[{content_id}] Failed to download {compressed_key}")
                    return False
                
                # Decompress to WAV
                original_file = temp_path / Path(original_key).name
                
                if compressed_key.endswith('.opus'):
                    success = self._decode_opus_to_wav(str(compressed_file), str(original_file))
                elif compressed_key.endswith('.mp3'):
                    success = self._decode_mp3_to_wav(str(compressed_file), str(original_file))
                else:
                    logger.error(f"[{content_id}] Unknown compressed audio format: {compressed_key}")
                    return False
                
                if not success:
                    logger.error(f"[{content_id}] Failed to decompress {compressed_key}")
                    return False
                
                # Upload decompressed file
                if self.s3_storage.upload_file(str(original_file), original_key):
                    logger.info(f"[{content_id}] Decompressed {compressed_key} -> {original_key}")
                    
                    # Delete compressed file
                    if self.s3_storage.delete_file(compressed_key):
                        logger.info(f"[{content_id}] Removed compressed file {compressed_key}")
                    
                    return True
                else:
                    logger.error(f"[{content_id}] Failed to upload decompressed file {original_key}")
                    return False
                    
        except Exception as e:
            logger.error(f"[{content_id}] Error decompressing audio {compressed_key}: {e}")
            return False
    
    async def _decompress_json_file(self, content_id: str, compressed_key: str, original_key: str) -> bool:
        """Decompress a JSON file from S3."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download compressed file
                compressed_file = temp_path / Path(compressed_key).name
                if not self.s3_storage.download_file(compressed_key, str(compressed_file)):
                    logger.error(f"[{content_id}] Failed to download {compressed_key}")
                    return False
                
                # Decompress JSON
                original_file = temp_path / Path(original_key).name
                
                with gzip.open(str(compressed_file), 'rb') as f_in:
                    with open(str(original_file), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Upload decompressed file
                if self.s3_storage.upload_file(str(original_file), original_key):
                    logger.info(f"[{content_id}] Decompressed {compressed_key} -> {original_key}")
                    
                    # Delete compressed file
                    if self.s3_storage.delete_file(compressed_key):
                        logger.info(f"[{content_id}] Removed compressed file {compressed_key}")
                    
                    return True
                else:
                    logger.error(f"[{content_id}] Failed to upload decompressed file {original_key}")
                    return False
                    
        except Exception as e:
            logger.error(f"[{content_id}] Error decompressing JSON {compressed_key}: {e}")
            return False
    
    async def _restore_video_file(self, content_id: str, compressed_key: str, original_key: str) -> bool:
        """Restore a video file (usually just a rename/copy since video compression isn't reversible)."""
        try:
            # For video files, we usually can't "decompress" in the traditional sense
            # Instead, we might just rename the file back if it was renamed during compression
            if compressed_key != original_key:
                # Copy the file to the original location
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Download current file
                    temp_file = temp_path / Path(compressed_key).name
                    if not self.s3_storage.download_file(compressed_key, str(temp_file)):
                        logger.error(f"[{content_id}] Failed to download {compressed_key}")
                        return False
                    
                    # Upload to original location
                    if self.s3_storage.upload_file(str(temp_file), original_key):
                        logger.info(f"[{content_id}] Restored {compressed_key} -> {original_key}")
                        
                        # Delete compressed file if different location
                        if self.s3_storage.delete_file(compressed_key):
                            logger.info(f"[{content_id}] Removed compressed file {compressed_key}")
                        
                        return True
                    else:
                        logger.error(f"[{content_id}] Failed to upload restored file {original_key}")
                        return False
            else:
                # File is already in the right place
                logger.info(f"[{content_id}] Video file {compressed_key} already in correct location")
                return True
                
        except Exception as e:
            logger.error(f"[{content_id}] Error restoring video {compressed_key}: {e}")
            return False
    
    async def process_content(self, content_id: str, dry_run: bool = False, force: bool = False) -> Dict:
        """Process content for cleanup and compression."""
        logger.info(f"[{content_id}] Starting cleanup and compression{' (DRY RUN)' if dry_run else ''}")
        
        try:
            # Step 1: Perform S3 audit (non-strict mode if force is enabled)
            logger.info(f"[{content_id}] Step 1: Performing S3 file audit...")
            strict_mode = not force  # Allow force to bypass strict checks
            audit_result = self._audit_s3_files(content_id, strict=strict_mode)
            
            if audit_result['status'] == 'fail' and not force:
                logger.error(f"[{content_id}] S3 audit FAILED - aborting cleanup (use --force to override)")
                return create_error_result(
                    ErrorCode.MISSING_CHUNKS,
                    'S3 audit failed - use --force to override strict validation',
                    {'audit_result': audit_result}
                )
            elif audit_result['status'] == 'error':
                logger.error(f"[{content_id}] S3 audit ERROR - aborting cleanup")
                return create_error_result(
                    ErrorCode.UNKNOWN_ERROR,
                    'S3 audit encountered an error',
                    {'audit_result': audit_result}
                )
            
            if force and audit_result['status'] == 'fail':
                logger.warning(f"[{content_id}] S3 audit failed but --force enabled, proceeding anyway")
            else:
                logger.info(f"[{content_id}] S3 audit passed ✓")
            
            with get_session() as session:
                # Get content record
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return create_error_result(
                        ErrorCode.NOT_FOUND,
                        f'Content {content_id} not found'
                    )
                
                # Check if already processed (but allow force to override)
                manifest_key = f"content/{content_id}/storage_manifest.json"
                if audit_result['already_compressed'] and not force:
                    logger.info(f"[{content_id}] Already compressed. Use --force to reprocess.")
                    return create_skipped_result(
                        'already_compressed',
                        data={'audit_result': audit_result}
                    )
            
            manifest_files = {}
            results = {
                'audio_processed': 0,
                'video_processed': 0,
                'json_compressed': 0,
                'chunks_deleted': 0
            }
            
            if not dry_run:
                # Process different file types
                logger.info(f"[{content_id}] Step 2: Processing audio files...")
                results['audio_processed'] = await self._process_audio_files(content_id, manifest_files)
                
                logger.info(f"[{content_id}] Step 3: Processing video files...")
                results['video_processed'] = await self._process_video_files(content_id, manifest_files)
                
                logger.info(f"[{content_id}] Step 4: Compressing JSON files...")
                results['json_compressed'] = await self._compress_json_files(content_id, manifest_files)
                
                logger.info(f"[{content_id}] Step 5: Deleting chunk files...")
                results['chunks_deleted'] = await self._delete_chunk_files(content_id, manifest_files)
                
                # Create and upload manifest
                logger.info(f"[{content_id}] Step 6: Creating and uploading storage manifest...")
                manifest = self._create_manifest(content_id, manifest_files)
                
                # Save manifest
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(manifest, f, indent=2)
                    temp_manifest = f.name
                
                if self.s3_storage.upload_file(temp_manifest, manifest_key):
                    logger.info(f"[{content_id}] Uploaded storage manifest")
                
                os.unlink(temp_manifest)
                
                # Log summary
                logger.info(f"[{content_id}] Cleanup complete: "
                           f"Audio: {results['audio_processed']}, "
                           f"Video: {results['video_processed']}, "
                           f"JSON: {results['json_compressed']}, "
                           f"Chunks deleted: {results['chunks_deleted']}")
                
                logger.info(f"[{content_id}] Space saved: {manifest['statistics']['space_saved']:,} bytes "
                           f"({manifest['statistics']['compression_ratio']:.1f}% reduction)")
            else:
                # Dry run - just report what would be done
                logger.info(f"[{content_id}] DRY RUN - Would process:")
                
                # Check what files exist
                if self.s3_storage.file_exists(f"content/{content_id}/audio.wav"):
                    logger.info(f"  - Compress audio.wav to opus")
                    results['audio_processed'] = 1
                
                for ext in ['.mp4', '.webm', '.mov', '.avi', '.mkv']:
                    if self.s3_storage.file_exists(f"content/{content_id}/video{ext}"):
                        logger.info(f"  - Compress video{ext}")
                        results['video_processed'] = 1
                        break
                
                # Count chunks using the existing list_files method
                chunk_pattern = f"content/{content_id}/chunks/"
                chunk_files = self.s3_storage.list_files(chunk_pattern)
                chunk_audio_files = [f for f in chunk_files if f.endswith('.wav')]
                chunk_count = len(chunk_audio_files)
                logger.info(f"  - Delete {chunk_count} chunk audio files")
                results['chunks_deleted'] = chunk_count
            
            return create_success_result({
                'content_id': content_id,
                'results': results,
                'dry_run': dry_run,
                'audit_result': audit_result
            })
            
        except Exception as e:
            logger.error(f"[{content_id}] Error in cleanup: {e}", exc_info=True)
            return create_error_result(
                ErrorCode.UNKNOWN_ERROR,
                str(e),
                {'audit_result': audit_result if 'audit_result' in locals() else None}
            )
    
    async def decompress_file(self, s3_key: str, output_path: str) -> bool:
        """Decompress a file from S3 for retrieval."""
        try:
            # Determine file type and decompression method
            if s3_key.endswith('.json.gz'):
                # Download compressed file
                with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as f:
                    temp_gz = f.name
                
                if not self.s3_storage.download_file(s3_key, temp_gz):
                    return False
                
                # Decompress
                with gzip.open(temp_gz, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                os.unlink(temp_gz)
                logger.info(f"Decompressed JSON to {output_path}")
                return True
                
            elif s3_key.endswith('.opus') or s3_key.endswith('.mp3'):
                # For compressed audio, convert back to WAV using appropriate tool
                file_ext = '.opus' if s3_key.endswith('.opus') else '.mp3'
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
                    temp_audio = f.name
                
                if not self.s3_storage.download_file(s3_key, temp_audio):
                    return False
                
                # Convert to WAV if requested
                if output_path.endswith('.wav'):
                    if s3_key.endswith('.opus'):
                        success = self._decode_opus_to_wav(temp_audio, output_path)
                    else:  # MP3
                        success = self._decode_mp3_to_wav(temp_audio, output_path)
                else:
                    # Just copy the compressed file
                    shutil.copy(temp_audio, output_path)
                    success = True
                
                os.unlink(temp_audio)
                return success
                
            else:
                # Direct download for uncompressed files
                return self.s3_storage.download_file(s3_key, output_path)
                
        except Exception as e:
            logger.error(f"Error decompressing {s3_key}: {e}")
            return False

async def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Cleanup and compress S3 storage')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--force', action='store_true', help='Force reprocess even if manifest exists')
    
    # For decompression modes
    parser.add_argument('--decompress', action='store_true', help='Decompress content back to original format')
    parser.add_argument('--decompress-file', help='S3 key to decompress (for testing)')
    parser.add_argument('--output', help='Output path for decompressed file')
    
    # For audit-only mode
    parser.add_argument('--audit-only', action='store_true', help='Only run S3 audit without processing')
    
    args = parser.parse_args()
    
    optimizer = StorageOptimizer()
    
    if args.decompress_file:
        if not args.output:
            print("Error: --output required with --decompress-file")
            return 1
        
        success = await optimizer.decompress_file(args.decompress_file, args.output)
        return 0 if success else 1
    elif args.decompress:
        # Full content decompression
        result = await optimizer.decompress_content(args.content, args.dry_run)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'success' else 1
    elif args.audit_only:
        # Run audit only
        audit_result = optimizer._audit_s3_files(args.content)
        print(json.dumps(audit_result, indent=2))
        return 0 if audit_result['status'] in ['pass', 'warning'] else 1
    else:
        result = await optimizer.process_content(args.content, args.dry_run, args.force)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'success' else 1

if __name__ == "__main__":
    asyncio.run(main())