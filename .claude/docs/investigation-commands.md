# Quick Investigation Commands

Use these commands to quickly investigate content issues. Environment variables are loaded from `.env`.

## PostgreSQL Access

The `.env` file doesn't use `export` statements, so use `set -a` (auto-export mode) before sourcing:

```bash
# Quick query (set -a exports all variables to child processes)
cd ~/signal4/core
set -a && source .env && set +a && PGPASSWORD=$POSTGRES_PASSWORD psql -h 10.0.0.4 -U signal4 -d av_content

# One-liner queries
set -a && source .env && set +a && PGPASSWORD=$POSTGRES_PASSWORD psql -h 10.0.0.4 -U signal4 -d av_content -c "SELECT * FROM content WHERE content_id = 'pod_9bb4e73a66d8'"

# Check content and chunks for an item
set -a && source .env && set +a && PGPASSWORD=$POSTGRES_PASSWORD psql -h 10.0.0.4 -U signal4 -d av_content -c "
SELECT c.content_id, c.title, c.is_transcribed, c.processing_stage,
       COUNT(ch.id) as chunks,
       SUM(CASE WHEN ch.transcription_status = 'completed' THEN 1 ELSE 0 END) as completed
FROM content c
LEFT JOIN content_chunks ch ON c.content_id = ch.content_id
WHERE c.content_id = 'CONTENT_ID_HERE'
GROUP BY c.content_id, c.title, c.is_transcribed, c.processing_stage"

# Check chunk statuses for a content item
set -a && source .env && set +a && PGPASSWORD=$POSTGRES_PASSWORD psql -h 10.0.0.4 -U signal4 -d av_content -c "
SELECT chunk_index, transcription_status, transcription_model, diarization_status
FROM content_chunks WHERE content_id = 'CONTENT_ID_HERE' ORDER BY chunk_index"
```

## S3/MinIO Access

```bash
# Configure mc alias (one-time setup)
set -a && source .env && set +a && mc alias set minio http://10.0.0.251:9000 $S3_ACCESS_KEY $S3_SECRET_KEY

# List files for a content item
mc ls minio/av-content/content/CONTENT_ID_HERE/

# List chunk files
mc ls minio/av-content/content/CONTENT_ID_HERE/chunks/

# Check specific chunk
mc ls minio/av-content/content/CONTENT_ID_HERE/chunks/11/

# View transcript file
mc cat minio/av-content/content/CONTENT_ID_HERE/chunks/11/transcript_words.json | jq .

# Check if file exists (returns exit code)
mc stat minio/av-content/content/CONTENT_ID_HERE/chunks/11/transcript_words.json

# Download file for inspection
mc cp minio/av-content/content/CONTENT_ID_HERE/chunks/11/transcript_words.json /tmp/
```

## Using Python Utils

```python
# Quick database session
from src.database.session import get_session
from src.database.models import Content, ContentChunk

with get_session() as session:
    content = session.query(Content).filter_by(content_id='pod_9bb4e73a66d8').first()
    print(f"Title: {content.title}, Stage: {content.processing_stage}")

    chunks = session.query(ContentChunk).filter_by(content_id='pod_9bb4e73a66d8').all()
    for c in chunks:
        print(f"Chunk {c.chunk_index}: transcription={c.transcription_status}")

# Quick S3 access
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.utils.config import load_config

config = load_config()
s3_config = S3StorageConfig.from_dict(config['storage']['s3'])
s3 = S3Storage(s3_config)

# List files
files = s3.list_files('content/pod_9bb4e73a66d8/chunks/11/')
print(files)

# Check if file exists
exists = s3.file_exists('content/pod_9bb4e73a66d8/chunks/11/transcript_words.json')

# Download file content
content = s3.download_json('content/pod_9bb4e73a66d8/chunks/11/transcript_words.json')
```
