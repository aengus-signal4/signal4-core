#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv
from src.utils.paths import get_env_path

# Load env
load_dotenv(get_env_path())

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal
from sqlalchemy import text

db = SessionLocal()

# Search for that specific text
query = text("""
SELECT
    s.id,
    s.content_id,
    s.start_time,
    s.end_time,
    s.text,
    c.title,
    c.channel_name,
    c.content_id as youtube_id,
    c.publish_date
FROM embedding_segments s
JOIN content c ON s.content_id = c.id
WHERE s.text LIKE '%jeans that convert issued immigration arabo-musulmans%'
LIMIT 1
""")

result = db.execute(query).fetchone()

if result:
    print(f'Segment ID: {result.id}')
    print(f'Content ID: {result.content_id}')
    print(f'Start time: {result.start_time}')
    print(f'End time: {result.end_time}')
    print()
    print(f'Content: {result.title}')
    print(f'Channel: {result.channel_name}')
    print(f'YouTube ID: {result.youtube_id}')
    print(f'URL: https://youtube.com/watch?v={result.youtube_id}&t={int(result.start_time)}')
    print(f'Publish date: {result.publish_date}')
    print()
    print('Full segment text:')
    print('='*80)
    print(result.text)
    print('='*80)
else:
    print('Segment not found')

db.close()
