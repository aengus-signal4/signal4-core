import json
from pathlib import Path
import webvtt
from typing import Dict, List, Optional
from datetime import datetime
import re
import struct

class SubtitleProcessor:
    @staticmethod
    def convert_vtt_to_standard(vtt_path: Path) -> Dict:
        """Convert VTT subtitles to standardized format with segments"""
        try:
            captions = webvtt.read(str(vtt_path))
            segments = []
            full_text = []
            
            for caption in captions:
                # Parse times to seconds
                start_time = SubtitleProcessor._time_to_seconds(caption.start)
                end_time = SubtitleProcessor._time_to_seconds(caption.end)
                
                # Clean text
                text = SubtitleProcessor._clean_text(caption.text)
                full_text.append(text)
                
                # Create segment
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })
            
            return {
                "segments": segments,
                "full_text": " ".join(full_text)
            }
    
        except Exception as e:
            raise ValueError(f"Failed to process VTT file at {vtt_path}: {e}")

    @staticmethod
    def convert_whisperx_to_standard(whisperx_result: Dict) -> List[tuple]:
        """Convert WhisperX output to minimal format: List of (start_time, text) tuples"""
        if not whisperx_result.get("segments"):
            raise ValueError("Invalid WhisperX result format")
        
        segments = []
        for segment in whisperx_result["segments"]:
            # For each word in the segment
            for word in segment.get("words", []):
                segments.append((
                    word["start"],
                    word["word"]
                ))
        
        return segments
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert VTT timestamp (HH:MM:SS.mmm) to seconds"""
        # Split into hours, minutes, and seconds+milliseconds
        h, m, s = time_str.split(':')
        # Split seconds and milliseconds
        s, ms = s.split('.')
        # Convert to seconds
        total_seconds = float(h) * 3600 + float(m) * 60 + float(s)
        # Add milliseconds
        total_seconds += float('0.' + ms)
        return total_seconds
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean subtitle text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove multiple spaces
        text = ' '.join(text.split())
        # Remove speaker identifications like "SPEAKER:"
        text = re.sub(r'^\s*[A-Z]+:\s*', '', text)
        return text.strip()
    
    @staticmethod
    def save_standardized_transcript(segments: List[tuple], output_path: Path) -> None:
        """Save minimal transcript format to binary file
        Format: sequence of (float32 timestamp, utf-8 text) pairs
        """
        output_path = output_path.with_suffix('.bin')
        with open(output_path, 'wb') as f:
            for timestamp, text in segments:
                # Write timestamp as 4-byte float
                f.write(struct.pack('f', timestamp))
                # Write text length as 2-byte unsigned int
                text_bytes = text.encode('utf-8')
                f.write(struct.pack('H', len(text_bytes)))
                # Write text as UTF-8 bytes
                f.write(text_bytes)
    
    @staticmethod
    def read_standardized_transcript(input_path: Path) -> List[tuple]:
        """Read minimal transcript format from binary file"""
        input_path = input_path.with_suffix('.bin')
        segments = []
        with open(input_path, 'rb') as f:
            while True:
                # Try to read timestamp
                timestamp_bytes = f.read(4)
                if not timestamp_bytes:
                    break
                timestamp = struct.unpack('f', timestamp_bytes)[0]
                
                # Read text length
                length_bytes = f.read(2)
                if not length_bytes:
                    break
                text_length = struct.unpack('H', length_bytes)[0]
                
                # Read text
                text_bytes = f.read(text_length)
                if not text_bytes:
                    break
                text = text_bytes.decode('utf-8')
                
                segments.append((timestamp, text))
        
        return segments
    
    @staticmethod
    def merge_transcripts(transcripts: List[Dict], priority: List[str] = None) -> Dict:
        """Merge multiple transcripts, prioritizing sources in order"""
        if not transcripts:
            raise ValueError("No transcripts provided")
        
        # Default priority: youtube_subtitles > whisperx
        priority = priority or ["youtube_subtitles", "whisperx"]
        
        # Sort transcripts by priority
        sorted_transcripts = sorted(
            transcripts,
            key=lambda x: priority.index(x.get("metadata", {}).get("source", "unknown"))
        )
        
        # Use the highest priority transcript as base
        merged = sorted_transcripts[0]
        
        # Add information about merged sources
        merged["metadata"]["merged_sources"] = [
            t.get("metadata", {}).get("source", "unknown")
            for t in sorted_transcripts
        ]
        
        return merged 