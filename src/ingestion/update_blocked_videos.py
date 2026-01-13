import re
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import logging
from ..database.session import get_session
from ..database.models import Content

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_blocked_videos(log_file_path: str) -> list:
    """Extract video IDs from log file that were blocked due to format unavailability"""
    blocked_videos = []
    pattern = r'ERROR - yt-dlp stderr: ERROR: \[youtube\] ([A-Za-z0-9_-]+): Requested format is not available'
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    video_id = match.group(1)
                    blocked_videos.append(video_id)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(blocked_videos))
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return []

async def update_blocked_videos(video_ids: list):
    """Update database records for blocked videos"""
    if not video_ids:
        logger.info("No blocked videos found to update")
        return
    
    logger.info(f"Found {len(video_ids)} blocked videos to update")
    
    with get_session() as session:
        for video_id in video_ids:
            try:
                content = session.query(Content).filter_by(content_id=video_id).first()
                if content:
                    # Only update if not already blocked or if blocked without a reason
                    if not content.blocked_download or not content.meta_data.get('block_reason'):
                        content.blocked_download = True
                        # Preserve existing metadata
                        meta_data = dict(content.meta_data) if content.meta_data else {}
                        meta_data['block_reason'] = "ERROR: [youtube] Requested format is not available. Use --list-formats for a list of available formats"
                        content.meta_data = meta_data
                        content.last_updated = datetime.now(timezone.utc)
                        session.add(content)
                        logger.info(f"Updated blocked status for video {video_id}")
                    else:
                        logger.info(f"Video {video_id} already blocked with reason")
                else:
                    logger.warning(f"Video {video_id} not found in database")
            except Exception as e:
                logger.error(f"Error updating video {video_id}: {e}")
                continue
        
        try:
            session.commit()
            logger.info("Successfully committed all updates")
        except Exception as e:
            logger.error(f"Error committing updates: {e}")
            session.rollback()

async def main():
    log_file = "/Users/signal4/logs/content_processing/worker_logs/10.0.0.209_download_youtube.log"
    
    if not Path(log_file).exists():
        logger.error(f"Log file not found: {log_file}")
        return
    
    blocked_videos = extract_blocked_videos(log_file)
    await update_blocked_videos(blocked_videos)

if __name__ == "__main__":
    asyncio.run(main()) 