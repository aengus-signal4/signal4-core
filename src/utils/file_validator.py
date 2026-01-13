#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('file_validator')

class FileValidator:
    """Utility class for validating content files"""
    
    def __init__(self, base_storage_path: Path):
        self.base_path = base_storage_path
        
    def validate_media_file(self, file_path: Path) -> Tuple[bool, Optional[str], Optional[float]]:
        """Validate a media file using ffprobe
        
        Returns:
            Tuple[bool, Optional[str], Optional[float]]: (is_valid, error_message, duration)
        """
        try:
            # Check basic file existence and size
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False, "File does not exist or is empty", None

            # Check file format and get duration
            duration_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            
            if duration_result.returncode != 0:
                return False, f"Invalid media file: {duration_result.stderr}", None
                
            duration = float(duration_result.stdout.strip())
            if duration <= 0:
                return False, f"Invalid duration: {duration}s", None

            return True, None, duration
        except Exception as e:
            return False, str(e), None

    def validate_wav_file(self, file_path: Path) -> Tuple[bool, Optional[str], Optional[float]]:
        """Validate a WAV file
        
        Returns:
            Tuple[bool, Optional[str], Optional[float]]: (is_valid, error_message, duration)
        """
        try:
            # First do basic media validation
            is_valid, error, duration = self.validate_media_file(file_path)
            if not is_valid:
                return False, error, None

            # Check audio stream format
            stream_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_fmt,channels,sample_rate',
                '-of', 'json',
                str(file_path)
            ]
            stream_result = subprocess.run(stream_cmd, capture_output=True, text=True)
            
            if stream_result.returncode != 0:
                return False, f"Error checking audio stream: {stream_result.stderr}", None
                
            audio_info = json.loads(stream_result.stdout)
            if not audio_info.get('streams'):
                return False, "No audio streams found", None
                
            stream = audio_info['streams'][0]
            codec = stream.get('codec_name', '').lower()
            
            if codec != 'pcm_s16le':
                return False, f"Invalid codec: {codec} (expected pcm_s16le)", duration
                
            return True, None, duration
        except Exception as e:
            return False, str(e), None

    def validate_transcription_file(self, file_path: Path) -> Tuple[bool, Optional[str], Optional[dict]]:
        """Validate a transcription JSON file
        
        Returns:
            Tuple[bool, Optional[str], Optional[dict]]: (is_valid, error_message, data)
        """
        try:
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False, "File does not exist or is empty", None

            with open(file_path) as f:
                data = json.load(f)

            # Validate structure
            if not isinstance(data, dict):
                return False, "Not a valid JSON object", None

            required_fields = ['full_text', 'segments']
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}", None

            if not isinstance(data['segments'], list) or not data['segments']:
                return False, "Segments must be a non-empty list", None

            # Validate each segment
            for i, segment in enumerate(data['segments']):
                if not isinstance(segment, dict):
                    return False, f"Segment {i} is not a valid object", None

                # Check required segment fields
                if 'text' not in segment:
                    return False, f"Segment {i} missing text", None

                # Check timing fields (supporting both naming conventions)
                has_start = 'start_time' in segment or 'start' in segment
                has_end = 'end_time' in segment or 'end' in segment
                
                if not (has_start and has_end):
                    return False, f"Segment {i} missing timing information", None

            return True, None, data
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}", None
        except Exception as e:
            return False, str(e), None

    def check_content_files(self, content_id: str, platform: str) -> Dict:
        """Check all files for a piece of content
        
        Args:
            content_id: The ID of the content to check
            platform: The platform (youtube/podcast) to determine file extension
            
        Returns:
            Dict containing validation results for all files
        """
        result = {
            'content_id': content_id,
            'platform': platform,
            'source_file': {
                'exists': False,
                'valid': False,
                'in_recycle': False,
                'path': None,
                'error': None,
                'duration': None
            },
            'wav_file': {
                'exists': False,
                'valid': False,
                'in_recycle': False,
                'path': None,
                'error': None,
                'duration': None
            },
            'transcription_file': {
                'exists': False,
                'valid': False,
                'path': None,
                'error': None,
                'data': None
            }
        }

        try:
            # Check source file
            source_ext = '.mp3' if platform == 'podcast' else '.mp4'
            source_paths = [
                self.base_path / "data" / "downloads" / f"{content_id}{source_ext}",
                self.base_path / "#recycle" / "downloads" / f"{content_id}{source_ext}"
            ]

            for path in source_paths:
                if path.exists() and path.stat().st_size > 0:
                    result['source_file']['exists'] = True
                    result['source_file']['path'] = str(path)
                    result['source_file']['in_recycle'] = '#recycle' in str(path)
                    
                    # Validate source file
                    is_valid, error, duration = self.validate_media_file(path)
                    result['source_file']['valid'] = is_valid
                    result['source_file']['error'] = error
                    result['source_file']['duration'] = duration
                    break

            # Check WAV file
            wav_paths = [
                self.base_path / "data" / "wav" / f"{content_id}.wav",
                self.base_path / "#recycle" / "downloads" / f"{content_id}.wav"
            ]

            for path in wav_paths:
                if path.exists() and path.stat().st_size > 0:
                    result['wav_file']['exists'] = True
                    result['wav_file']['path'] = str(path)
                    result['wav_file']['in_recycle'] = '#recycle' in str(path)
                    
                    # Validate WAV file
                    is_valid, error, duration = self.validate_wav_file(path)
                    result['wav_file']['valid'] = is_valid
                    result['wav_file']['error'] = error
                    result['wav_file']['duration'] = duration
                    break

            # Check transcription file
            transcription_path = self.base_path / "data" / "transcription" / f"{content_id}_transcription.json"
            if transcription_path.exists() and transcription_path.stat().st_size > 0:
                result['transcription_file']['exists'] = True
                result['transcription_file']['path'] = str(transcription_path)
                
                # Validate transcription file
                is_valid, error, data = self.validate_transcription_file(transcription_path)
                result['transcription_file']['valid'] = is_valid
                result['transcription_file']['error'] = error
                result['transcription_file']['data'] = data

            return result
        except Exception as e:
            logger.error(f"Error checking files for {content_id}: {str(e)}")
            return result

    def get_file_paths(self, content_id: str, platform: str) -> Dict[str, List[Path]]:
        """Get all possible file paths for a piece of content
        
        Args:
            content_id: The ID of the content
            platform: The platform (youtube/podcast) to determine file extension
            
        Returns:
            Dict containing lists of possible paths for each file type
        """
        source_ext = '.mp3' if platform == 'podcast' else '.mp4'
        return {
            'source': [
                self.base_path / "data" / "downloads" / f"{content_id}{source_ext}",
                self.base_path / "#recycle" / "downloads" / f"{content_id}{source_ext}"
            ],
            'wav': [
                self.base_path / "data" / "wav" / f"{content_id}.wav",
                self.base_path / "#recycle" / "downloads" / f"{content_id}.wav"
            ],
            'transcription': [
                self.base_path / "data" / "transcription" / f"{content_id}_transcription.json"
            ]
        } 