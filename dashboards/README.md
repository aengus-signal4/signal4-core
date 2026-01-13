# Audio-Visual Content Analysis Dashboards

This directory contains Streamlit dashboards for analyzing and exploring the audio-visual content processing pipeline data.

## Available Dashboards

### 1. Speaker Annotation Tool (`speaker_annotation.py`)

A web-based tool for reviewing, annotating, and merging speaker identities in transcribed audio content.

#### Features
- View all speakers sorted by appearance count
- Edit speaker information (display name, country, role)
- View recent transcriptions with context
- Find and merge similar speakers using voice embeddings
- Track speaker statistics (total segments, duration, appearances)

#### Usage
1. Select a speaker from the sidebar
2. View speaker details and statistics
3. Edit speaker information using the form
4. Review recent transcriptions with context
5. Find and merge similar speakers if needed

### 2. Semantic Search Dashboard (`semantic_search.py`)

Interactive dashboard for querying content using semantic similarity and filters.

#### Features
- Natural language semantic search across transcribed content
- Filter by project, date range, and keywords
- View matching segments with speaker turns and context
- Audio playback for segments
- Embedding-based similarity scoring

#### Usage
1. Enter a natural language search query
2. Apply filters (project, dates, keywords) in the sidebar
3. Review results with similarity scores
4. Expand results to see context and speaker turns
5. Play audio segments directly in the browser

### 3. Speaker Network Dashboard (`speaker_network.py`)

Interactive network visualization showing speaker co-occurrence patterns based on shared content appearances.

#### Features
- Interactive network graphs with speakers as nodes
- Edges represent co-occurrence in the same content
- Filter by project and date range
- Adjustable network parameters (minimum appearances, co-occurrence threshold)
- Multiple layout algorithms (spring, circular, kamada-kawai)
- Network statistics and analysis
- Top speakers rankings by different metrics
- Connection analysis and centrality measures

#### Network Visualization Elements
- **Nodes**: Speakers (size = total segments, color = content appearances)
- **Edges**: Co-occurrence relationships (thickness = frequency)
- **Hover**: Detailed speaker information and statistics
- **Interactive**: Click, zoom, pan for exploration

#### Usage
1. Set filters in the sidebar (project, date range)
2. Adjust network parameters (minimum appearances, co-occurrence)
3. Choose layout algorithm
4. Click "Generate Network" to create visualization
5. Explore the interactive network graph
6. Review analysis tabs for detailed statistics

## Setup

1. Install dependencies:
```bash
pip install -r requirements-web.txt
```

2. Run any dashboard:
```bash
# Speaker annotation tool
streamlit run dashboards/speaker_annotation.py

# Semantic search dashboard  
streamlit run dashboards/semantic_search.py

# Speaker network dashboard
streamlit run dashboards/speaker_network.py
```

## Dependencies

All dashboards require:
- Streamlit
- Pandas
- SQLAlchemy (for database access)
- PyYAML (for configuration)

Additional dependencies by dashboard:
- **Semantic Search**: Embedding models, S3 storage utilities
- **Speaker Network**: NetworkX, Plotly (for network visualization)

## Configuration

Dashboards use the main configuration file at `config/config.yaml` for:
- Database connection settings
- Storage configuration (S3/MinIO)
- Embedding model settings
- Project definitions

## Speaker Roles

The annotation tool supports the following speaker roles:
- Politician
- Journalist
- Influencer
- Other

## Database Schema

The dashboards use the following main tables:
- `speakers`: Global speaker information
- `speaker_transcriptions`: Speaker turns in content
- `speaker_embeddings`: Voice embeddings for similarity search
- `content`: Media content metadata and processing status
- `embedding_segments`: Optimized segments for semantic search 