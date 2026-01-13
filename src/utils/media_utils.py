#!/usr/bin/env python3
import subprocess
import json
import logging
from typing import Optional

# Use a generic logger for utility functions
logger = logging.getLogger(__name__)
# Basic configuration if no handlers are set elsewhere
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_audio_duration(file_path: str) -> Optional[float]:
    """Get duration of an audio/video file using ffprobe with multiple fallback methods."""
    try:
        # First try: Get duration from format
        format_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        format_result = subprocess.run(format_cmd, capture_output=True, text=True, check=False)
        
        if format_result.returncode == 0 and format_result.stdout.strip():
            try:
                duration = float(format_result.stdout.strip())
                if duration > 0:
                    logger.debug(f"Got duration {duration} from format for {file_path}")
                    return duration
            except ValueError:
                 logger.debug(f"Could not parse format duration '{format_result.stdout.strip()}' for {file_path}")


        # Second try: Get duration from audio stream
        stream_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        stream_result = subprocess.run(stream_cmd, capture_output=True, text=True, check=False)
        
        if stream_result.returncode == 0 and stream_result.stdout.strip():
            try:
                duration = float(stream_result.stdout.strip())
                if duration > 0:
                    logger.debug(f"Got duration {duration} from stream for {file_path}")
                    return duration
            except ValueError:
                 logger.debug(f"Could not parse stream duration '{stream_result.stdout.strip()}' for {file_path}")


        # Third try: Get duration from format and stream combined (JSON output)
        combined_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration:stream=duration',
            '-of', 'json',
            str(file_path)
        ]
        combined_result = subprocess.run(combined_cmd, capture_output=True, text=True, check=False)
        
        if combined_result.returncode == 0:
            try:
                data = json.loads(combined_result.stdout)
                # Try format duration first
                if 'format' in data and 'duration' in data['format']:
                    try:
                        duration = float(data['format']['duration'])
                        if duration > 0:
                            logger.debug(f"Got duration {duration} from combined format for {file_path}")
                            return duration
                    except (KeyError, ValueError):
                        pass # Ignore if format duration isn't valid/present
                # Then try stream duration
                if 'streams' in data and data['streams']:
                    for stream in data['streams']:
                        if 'duration' in stream:
                             try:
                                duration = float(stream['duration'])
                                if duration > 0:
                                    logger.debug(f"Got duration {duration} from combined stream for {file_path}")
                                    return duration
                             except (KeyError, ValueError):
                                 pass # Ignore if stream duration isn't valid/present
            except json.JSONDecodeError:
                 logger.debug(f"Could not parse combined JSON output for {file_path}")


        # If all methods fail, log the error and return None
        logger.warning(f"All duration detection methods failed for {file_path}")
        # Log stderr from the probes for debugging
        if format_result.stderr: logger.debug(f"Format probe stderr: {format_result.stderr.strip()}")
        if stream_result.stderr: logger.debug(f"Stream probe stderr: {stream_result.stderr.strip()}")
        if combined_result.stderr: logger.debug(f"Combined probe stderr: {combined_result.stderr.strip()}")
        return None

    except Exception as e:
        # Catch unexpected errors like file not found by subprocess
        logger.error(f"Error getting duration for {file_path}: {str(e)}")
        return None 