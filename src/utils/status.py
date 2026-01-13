from typing import Dict, Optional, Any
from datetime import datetime
from ..database.session import get_session
from ..database.manager import DatabaseManager
from ..database.models import Content
from sqlalchemy import func

async def get_project_status(project_name: str, date_range: Optional[Dict] = None) -> Dict[str, Any]:
    """Get current status of project content processing"""
    with get_session() as session:
        db = DatabaseManager(session)
        
        # Get total stats (across all dates)
        total_content = db.get_unique_content_count(project_name)
        total_duration = db.get_unique_content_duration(project_name)
        
        # Get content counts for each stage (all dates)
        downloaded_total = db.session.query(Content).filter(
            Content.projects.any(project_name),
            Content.is_downloaded == True,
            Content.is_duplicate == False
        ).count()
        
        # Count content with WAV files
        audio_converted_total = db.session.query(Content).filter(
            Content.projects.any(project_name),
            Content.is_duplicate == False,
            Content.is_converted == True
        ).count()
        
        transcribed_total = db.session.query(Content).filter(
            Content.projects.any(project_name),
            Content.transcriptions.any(),
            Content.is_duplicate == False
        ).count()
        
        # Get content needing processing at each stage (all dates)
        to_download_total = total_content - downloaded_total
        to_convert_total = downloaded_total - audio_converted_total
        to_transcribe_total = audio_converted_total - transcribed_total
        
        # Get date range specific stats if a range is provided
        if date_range:
            # Base query with date range filter
            date_query = db.session.query(Content).filter(
                Content.projects.any(project_name),
                Content.is_duplicate == False,
                Content.publish_date >= date_range['start'],
                Content.publish_date <= date_range['end']
            )
            
            # Get total content in date range
            range_content = date_query.count()
            range_duration = float(date_query.with_entities(func.sum(Content.duration)).scalar() or 0) / 3600
            
            # Get counts for each stage in date range
            downloaded_range = date_query.filter(Content.is_downloaded == True).count()
            
            # Count content with WAV files in date range
            audio_converted_range = date_query.filter(Content.is_converted == True).count()
            
            transcribed_range = date_query.filter(Content.transcriptions.any()).count()
            
            # Get content needing processing at each stage (date range)
            to_download_range = range_content - downloaded_range
            to_convert_range = downloaded_range - audio_converted_range
            to_transcribe_range = audio_converted_range - transcribed_range
            
            # Include both total and range-specific stats
            return {
                'total_content': total_content,
                'total_duration': total_duration,
                'progress': {
                    'download': {
                        'done': downloaded_total,
                        'total': total_content,
                        'percent': (downloaded_total / total_content * 100) if total_content > 0 else 0
                    },
                    'audio': {
                        'done': audio_converted_total,
                        'total': downloaded_total,
                        'percent': (audio_converted_total / downloaded_total * 100) if downloaded_total > 0 else 0
                    },
                    'transcribe': {
                        'done': transcribed_total,
                        'total': audio_converted_total,
                        'percent': (transcribed_total / audio_converted_total * 100) if audio_converted_total > 0 else 0
                    }
                },
                'pending': {
                    'download': to_download_total,
                    'audio': to_convert_total,
                    'transcribe': to_transcribe_total
                },
                'date_range': {
                    'start': date_range['start'].strftime('%Y-%m-%d'),
                    'end': date_range['end'].strftime('%Y-%m-%d'),
                    'total_content': range_content,
                    'total_duration': range_duration,
                    'progress': {
                        'download': {
                            'done': downloaded_range,
                            'total': range_content,
                            'percent': (downloaded_range / range_content * 100) if range_content > 0 else 0
                        },
                        'audio': {
                            'done': audio_converted_range,
                            'total': downloaded_range,
                            'percent': (audio_converted_range / downloaded_range * 100) if downloaded_range > 0 else 0
                        },
                        'transcribe': {
                            'done': transcribed_range,
                            'total': audio_converted_range,
                            'percent': (transcribed_range / audio_converted_range * 100) if audio_converted_range > 0 else 0
                        }
                    },
                    'pending': {
                        'download': to_download_range,
                        'audio': to_convert_range,
                        'transcribe': to_transcribe_range
                    }
                }
            }
        else:
            # Return only total stats if no date range provided
            return {
                'total_content': total_content,
                'total_duration': total_duration,
                'progress': {
                    'download': {
                        'done': downloaded_total,
                        'total': total_content,
                        'percent': (downloaded_total / total_content * 100) if total_content > 0 else 0
                    },
                    'audio': {
                        'done': audio_converted_total,
                        'total': downloaded_total,
                        'percent': (audio_converted_total / downloaded_total * 100) if downloaded_total > 0 else 0
                    },
                    'transcribe': {
                        'done': transcribed_total,
                        'total': audio_converted_total,
                        'percent': (transcribed_total / audio_converted_total * 100) if audio_converted_total > 0 else 0
                    }
                },
                'pending': {
                    'download': to_download_total,
                    'audio': to_convert_total,
                    'transcribe': to_transcribe_total
                }
            }

def print_status_report(status: Dict[str, Any]) -> None:
    """Print a formatted status report"""
    print("\n============================================================")
    print(f"Status Report at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("------------------------------------------------------------")
    
    # Print total stats
    print(f"Total Content: {status['total_content']} items ({status['total_duration']:.1f} hours)")
    print("\nProgress:")
    for stage, data in status['progress'].items():
        percent = data['percent']
        print(f"  {stage.title():8} {data['done']:4}/{data['total']:4} ({percent:6.1f}%)")
    
    print("\nPending Tasks:")
    for stage, count in status['pending'].items():
        print(f"  {stage.title():8} {count:6} items")
    
    # Print date range specific stats if available
    if 'date_range' in status:
        print("\n------------------------------------------------------------")
        print(f"Date Range: {status['date_range']['start']} to {status['date_range']['end']}")
        print(f"Content in Range: {status['date_range']['total_content']} items ({status['date_range']['total_duration']:.1f} hours)")
        
        print("\nProgress (Date Range):")
        for stage, data in status['date_range']['progress'].items():
            percent = data['percent']
            print(f"  {stage.title():8} {data['done']:4}/{data['total']:4} ({percent:6.1f}%)")
        
        print("\nPending Tasks (Date Range):")
        for stage, count in status['date_range']['pending'].items():
            print(f"  {stage.title():8} {count:6} items")
    
    print("============================================================\n") 