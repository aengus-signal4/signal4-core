# Stitch Pipeline Guide

## Overview

The Stitch Pipeline is a modular, 12-stage system that performs word-level speaker attribution for audio content. It takes raw diarization data and transcript data and produces a fully attributed transcript with speaker assignments for each word, enhanced with proper sentence segmentation and capitalization.

## Architecture

The pipeline is built with a modular architecture where each stage is implemented as a separate module:

- **Main Pipeline**: `stitch.py` - Orchestrates all stages and manages caching
- **Word Tracking**: `WordTable` class - Central data structure tracking all words
- **Stage Modules**: `stage1_load.py` through `stage12_output.py`
- **Utilities**: 
  - `util_stitch.py` - Shared helper functions and text processing
  - `util_cache.py` - Stage result caching for test mode
  - `util_wav2vec2.py` - Audio alignment utilities (currently disabled)

## Current Stage Organization

The pipeline consists of 12 stages executed in sequence:

1. **Stage 1**: Data Loading - Load diarization, transcript chunks, and audio
2. **Stage 2**: Data Cleaning - Handle chunk boundaries and remove artifacts
3. **Stage 3**: Table Creation - Create WordTable and categorize words
4. **Stage 4**: Good Grammar + Single Speaker (Slam-Dunk assignments)
5. **Stage 5**: Bad Grammar + Single Speaker (with enhancement)
6. **Stage 6**: Speaker Embeddings - Audio-based speaker identification
7. **Stage 7**: Speaker Centroids - Calculate and store speaker representations
8. **Stage 8**: Good Grammar + Multi-Speaker Analysis
9. **Stage 9**: Bad Grammar + Multi-Speaker Analysis (LLM-based)
10. **Stage 10**: LLM Resolution - Resolve remaining UNKNOWN words
11. **Stage 11**: Final Cleanup - Grammar enhancement and validation
12. **Stage 12**: Output Generation - Create final transcripts

## Key Performance Characteristics

### Computational Complexity by Stage
- **Most Expensive**: Stage 6 (Speaker Embeddings - 20-25% of time)
- **Moderate Cost**: Stages 9 & 10 (LLM inference - 10-15% each)
- **Low Cost**: Stages 2, 3, 4, 5, 7, 8, 11, 12 (data manipulation)
- **I/O Intensive**: Stage 1 (S3/file loading)

### Resource Usage
- **GPU/MPS Intensive**: Stage 6 (pyannote embedding model)
- **LLM Intensive**: Stages 9 & 10 (language model inference)
- **Memory Intensive**: Full audio loaded, pandas DataFrames, embedding matrices
- **Cache Friendly**: Test mode supports stage result caching

## Pipeline Stages

### Stage 1: Data Loading (`stage1_load.py`)
**Purpose**: Load and prepare raw data for processing

**Input**:
- `diarization.json` - Speaker diarization segments
- `transcript_words.json` - Word-level transcription data
- Audio file (optional, for embeddings)

**Process**:
- Downloads files from S3 storage
- Handles compressed files (.json.gz, .opus, .mp3)
- Performs decompression and format conversion
- Validates data integrity

**Output**:
- Loaded diarization data
- Loaded transcript data
- Audio file path (if available)

**Key Features**:
- Supports time range filtering for focused processing
- Automatic format detection and conversion
- Error handling for missing or corrupted files

---

### Stage 2: Data Cleaning (`stage2_clean.py`)
**Purpose**: Remove transcription artifacts and improve data quality

**Input**:
- Raw diarization data from Stage 1
- Raw transcript data from Stage 1

**Process**:
- Remove boundary overlaps (duplicate words at chunk boundaries)
- Filter out transcription artifacts (low confidence words)
- Clean up formatting issues
- Remove words that are too short or have invalid timestamps

**Output**:
- Cleaned diarization data
- Cleaned transcript data
- Cleaning statistics

**Cleaning Operations**:
- **Boundary Overlap Removal**: Eliminates duplicate words at ~300s chunk boundaries
- **Artifact Filtering**: Removes words with confidence < threshold
- **Format Validation**: Ensures proper timestamp ordering and text formatting

---

### Stage 3: Table Creation (`stage3_tables.py`)
**Purpose**: Create central data structures and categorize words

**Input**:
- Cleaned diarization data from Stage 2
- Cleaned transcript data from Stage 2

**Process**:
- Initialize `WordTable` with all words from transcript
- Initialize `SegmentTable` for speaker segment reference
- Analyze grammar quality (capitals + punctuation)
- Determine single vs multi-speaker segments
- Categorize all words into 4 groups:
  - `GOOD_GRAMMAR_SINGLE`: Good grammar + single speaker
  - `GOOD_GRAMMAR_MULTI`: Good grammar + multiple speakers
  - `BAD_GRAMMAR_SINGLE`: Bad grammar + single speaker
  - `BAD_GRAMMAR_MULTI`: Bad grammar + multiple speakers

**Output**:
- `WordTable` - Primary word tracking structure with categorized words
- `SegmentTable` - Speaker segment reference
- Grammar and speaker analysis metadata

**WordTable Structure**:
```python
{
    'word_id': 'unique_identifier',
    'text': 'actual_word',
    'start': 'start_time',
    'end': 'end_time',
    'confidence': 'whisper_confidence',
    'speaker_current': 'GOOD_GRAMMAR_SINGLE',  # Initial category assignment
    'resolution_method': 'none',
    'processing_status': 'initialized',
    'metadata': {},
    'assignment_history': [...],
    'has_good_grammar': True/False,
    'has_multiple_speakers': True/False,
    'segment_has_good_grammar': True/False,
    'segment_has_multiple_speakers': True/False
}
```

---

### Stage 4: Good Grammar + Single Speaker Assignment (`stage4_good_grammar_single.py`)
**Purpose**: High-confidence speaker assignments for well-formed segments

**Input**:
- WordTable from Stage 3 with categorized words
- Diarization segments

**Process**:
- Process all words marked as `GOOD_GRAMMAR_SINGLE`
- Find diarization segment with highest overlap for each word
- Trust the single-speaker categorization from Stage 3
- Assign speaker regardless of overlap percentage
- Track assignments with full history

**Output**:
- WordTable with speaker assignments for good grammar single speaker words
- Assignment statistics and confidence scores

**Key Features**:
- **Slam-Dunk Assignments**: High confidence due to good grammar + single speaker
- **No Threshold**: Assigns to highest overlap speaker even if overlap is low
- **Fast Processing**: Simple overlap calculation without complex analysis

---

### Stage 5: Bad Grammar + Single Speaker Assignment (`stage5_bad_grammar_single.py`)
**Purpose**: Handle poorly punctuated segments with single speakers

**Input**:
- WordTable from Stage 4
- Words marked as `BAD_GRAMMAR_SINGLE`
- Diarization segments

**Process**:
- Process segments with bad grammar but single speaker
- Enhance text with punctuation/capitalization using spaCy NER
- Find diarization overlap for enhanced segments
- Assign speakers based on overlap analysis
- May re-transcribe segments for better quality

**Output**:
- WordTable with speaker assignments for bad grammar single speaker words
- Enhanced text with improved punctuation
- Assignment statistics

**Key Features**:
- **Text Enhancement**: Uses NER to add punctuation and capitals
- **Re-transcription**: Can request better transcription for unclear segments
- **Single Speaker Trust**: Relies on Stage 3's single speaker determination

---

### Stage 6: Speaker Embeddings (`stage6_speaker_embeddings.py`)
**Purpose**: Audio-based speaker identification using pyannote embeddings and clustering

**Input**:
- WordTable from Stage 5
- Audio file (shared resource)
- Configuration settings

**Process**:
- Extract audio segments for analysis
- Generate speaker embeddings using pyannote model
- Build speaker centroids from assigned words
- Merge similar speakers (>0.8 similarity threshold)
- Assign remaining words based on embedding similarity
- Use comprehensive speaker processor for clustering and assignment

**Output**:
- Updated WordTable with embedding-based speaker assignments
- Speaker centroid data for downstream stages
- Speaker processing statistics and confidence scores

**Key Features**:
- **Deep Learning Inference**: Most computationally expensive stage
- **Embedding Model**: pyannote/embedding model on GPU/MPS
- **Similarity Matrices**: O(n*k) cosine similarity calculations
- **Clustering Algorithms**: DBSCAN/Agglomerative clustering overhead
- **Batch Processing**: Up to 64 segments per batch (configurable)

**Performance Impact**:
- **Major Bottleneck**: 20-25% of total pipeline time
- **Memory Intensive**: Large embedding matrices and similarity calculations
- **GPU/MPS Intensive**: Requires hardware acceleration for reasonable performance

---

### Stage 7: Speaker Centroids (`stage7_speaker_centroids.py`)
**Purpose**: Calculate and store speaker centroids for database integration

**Input**:
- WordTable from Stage 6
- Speaker centroid data from Stage 6

**Process**:
- Calculate quality metrics for each speaker
- Filter based on minimum quality thresholds
- Map local speakers to global database IDs
- Store centroids for future use

**Output**:
- Updated WordTable with global speaker mappings
- Speaker centroid statistics

**Performance Characteristics**:
- **Lightweight**: Mostly vector operations
- **Memory Efficient**: Only stores centroid vectors
- **Fast Processing**: O(n) for centroid calculations

---

### Stage 8: Good Grammar + Multi-Speaker Analysis (`stage8_good_grammar_multi.py`)
**Purpose**: Process well-formed segments with multiple speakers using LLM analysis

**Input**:
- WordTable from Stage 7 (with speaker centroids)
- Cleaned diarization data

**Process**:
- Identify segments with good grammar and multiple speakers
- Use LLM-based analysis for speaker disambiguation
- Apply contextual speaker assignment
- Process segments with conversation context

**Output**:
- Updated WordTable with multi-speaker assignments
- Processing statistics for good grammar segments

**Key Features**:
- **LLM-Based Processing**: Uses language models for complex multi-speaker scenarios
- **Context Analysis**: Considers conversation flow and speaker patterns
- **Quality Focus**: Targets well-formed segments for high-confidence assignments

---

### Stage 9: Bad Grammar + Multi-Speaker Analysis (`stage9_bad_grammar_multi.py`)
**Purpose**: Process poorly-formed segments with multiple speakers using enhanced LLM analysis

**Input**:
- WordTable from Stage 8
- Cleaned diarization data
- Audio path for enhanced processing

**Process**:
- Identify segments with poor grammar and multiple speakers
- Apply enhanced LLM processing with audio context
- Use language-aware processing (supports multiple languages)
- Handle complex conversational patterns

**Output**:
- Updated WordTable with enhanced multi-speaker assignments
- Processing statistics for challenging segments

**Key Features**:
- **Enhanced LLM Processing**: More sophisticated analysis for difficult cases
- **Audio Context**: Uses audio information to improve assignments
- **Language Support**: Handles multiple languages
- **Complex Pattern Recognition**: Addresses interruptions and overlapping speech

---

### Stage 10: LLM Resolution (`stage10_resolutions.py`)
**Purpose**: Resolve remaining UNKNOWN words using LLM analysis

**Input**:
- WordTable from Stage 9 (with most assignments completed)
- Speaker centroids from Stage 6
- Cleaned diarization data

**Process**:
- Identify remaining UNKNOWN words
- Use LLM with speaker centroid context
- Apply conversation context analysis
- Resolve final uncertain assignments

**Output**:
- Updated WordTable with final speaker assignments
- Resolution statistics and confidence scores

**Key Features**:
- **Final Resolution**: Addresses remaining uncertain words
- **Context-Aware**: Uses full conversation context for decisions
- **Centroid Integration**: Leverages speaker embeddings for consistency
- **High Precision**: Focused on final quality improvements

---

### Stage 11: Final Cleanup (`stage11_cleanup.py`)
**Purpose**: Apply final grammar enhancements and validation

**Input**:
- WordTable from Stage 10 (with complete speaker assignments)

**Process**:
- Apply final grammar and punctuation improvements
- Validate speaker assignment consistency
- Perform final quality checks
- Prepare data for output generation

**Output**:
- Finalized WordTable ready for output
- Quality validation results

**Key Features**:
- **Quality Assurance**: Final validation of all assignments
- **Grammar Enhancement**: Last-pass punctuation and capitalization
- **Consistency Checks**: Validates temporal and logical consistency

---

### Stage 12: Output Generation (`stage12_output.py`)
**Purpose**: Generate final transcript outputs and save results

**Input**:
- Final WordTable from Stage 11

**Process**:
- Generate readable transcript with proper formatting
- Create detailed transcript with timing and confidence information
- Save processing metadata and statistics
- Create output files for integration with larger system

**Output Formats**:
- **Readable Transcript**: Clean, formatted transcript for human reading
- **Detailed Transcript**: Technical format with all metadata
- **Processing Metadata**: Complete processing statistics and quality metrics
- **Speaker Assignment Results**: Detailed breakdown of assignment methods and confidence

**Performance Characteristics**:
- **Multiple Iterations**: Generates multiple output formats
- **JSON Serialization**: Converting large DataFrames to JSON
- **S3 Uploads**: Network I/O for production mode
- **Disk I/O**: File writes in test mode
- **Smart Word Joining**: Custom punctuation-aware text joining


## Assignment History Tracking

The pipeline now implements comprehensive assignment tracking for debugging and analysis:

### Assignment History System
Every word maintains a complete history of all speaker assignments throughout the pipeline:

**Key Features**:
- **Complete Tracking**: Every speaker assignment change is recorded with timestamp, stage, method, confidence, and reason
- **Debugging Support**: Full visibility into why each word was assigned to each speaker
- **Stage Identification**: Track which pipeline stage made each assignment decision
- **Method Attribution**: Record the specific method used for each assignment (diarization_overlap, embedding_similarity, etc.)
- **Confidence Tracking**: Monitor confidence levels for all assignments
- **Detailed Reasoning**: Human-readable explanations for each assignment decision

**Assignment History Structure**:
```python
{
    'stage': 'stage4_context',                    # Which pipeline stage made the assignment
    'timestamp': 1672531200.0,                   # Unix timestamp of assignment
    'speaker': 'SPEAKER_05',                     # Assigned speaker ID
    'method': 'diarization_overlap',             # Assignment method used
    'confidence': 0.85,                          # Confidence score (0.0-1.0)
    'reason': 'Diarization overlap analysis (boundary tolerance: 0.050s)'  # Detailed explanation
}
```

### Tracked Assignment Methods
The pipeline uses standardized methods to ensure all assignments are tracked:

**Primary Assignment Methods**:
- `word_table.assign_speaker_to_words()` - Batch assignment of multiple words
- `word_table.assign_speaker_to_word_by_index()` - Single word assignment by DataFrame index

**Assignment Tracking Parameters**:
- `stage`: Pipeline stage making the assignment (e.g., 'stage4_context', 'stage5_speaker_assignment')
- `reason`: Detailed explanation of why this assignment was made
- `method`: Technical method used (e.g., 'diarization_overlap', 'embedding_similarity')
- `confidence`: Numeric confidence score for the assignment

### Assignment Methods by Stage

**Stage 5 (Overlap Analysis)**:
- Method: `diarization_overlap`
- Tracks boundary tolerance assignments
- Records overlap percentages and segment confidence

**Stage 6 (LLM Speaker Assignment)**:
- Method: `llm_speaker_assignment`
- Tracks LLM-based speaker decisions for UNKNOWN words
- Records context analysis and confidence scores

**Stage 7 (Speaker Embeddings)**:
- Method: `speaker_embedding_assignment`
- Tracks embedding similarity and clustering decisions
- Records assignment confidence from acoustic analysis

**Stage 8 (Speaker Centroids)**:
- Method: `global_speaker_mapping`
- Tracks mapping of local speakers to global database IDs
- Records universal speaker name assignments

**Stage 9 (Grammar Cleanup)**:
- Method: `grammar_enhancement`
- Tracks grammar and punctuation improvements
- Records text quality enhancements while preserving assignments

**Stage 10 (Split Sentences)**:
- Method: `fragment_cleanup` and `split_sentence_consolidation`
- Tracks fragment-based reassignment and sentence consolidation
- Records cross-speaker sentence fixes and orphan segment handling

### Debug Output Generation
The system generates detailed assignment history reports for debugging:

**Assignment History Report** (`{content_id}_assignment_history.txt`):
- Chronological view of all word assignments
- Complete assignment chain for each word
- Stage-by-stage progression tracking
- Detailed reasoning for each assignment decision

**Example Assignment History Entry**:
```
[298.36s] 'But' -> SPEAKER_05
  ├─ stage3_initialization: UNKNOWN (initialization, conf=0.00) - Word table initialization
  ├─ stage4_context: SPEAKER_05 (diarization_overlap, conf=0.80) - Diarization overlap analysis (boundary tolerance: 0.050s)
  └─ stage5_speaker_assignment: SPEAKER_05 (speaker_embedding_assignment, conf=0.85) - Speaker embedding assignment via diarization_confirmed
```

### Assignment Audit Compliance
All pipeline stages now use tracked assignment methods:

**✅ Fully Compliant Stages**:
- Stage 6: Uses `assign_speaker_to_word_by_index()` with complete tracking
- Stage 7: Uses `assign_speaker_to_words()` with stage and reason parameters
- Stage 7b: Tracks global speaker mapping in assignment history
- Stage 8: Uses utility functions with complete tracking for multi-speaker resolution
- Stage 9: Uses utility functions with complete tracking for sentence embeddings
- Stage 10: Uses `assign_speaker_to_word_by_index()` for fragment cleanup and consolidation
- Stage 11: No speaker assignments (output generation only)

**Tracking Benefits**:
- **Debug Visibility**: See exactly why each word was assigned to each speaker
- **Quality Assurance**: Identify stages with low confidence assignments
- **Method Analysis**: Understand which assignment methods are most effective
- **Error Investigation**: Trace assignment errors back to their source stage
- **Performance Optimization**: Identify stages that frequently override previous assignments

## Shared Resource Management

### Audio Data Sharing
The pipeline now implements efficient resource sharing to avoid redundant loading:

**Shared Audio Loading** (`stitch_utils.py`):
- Audio file loaded once at pipeline start using `load_shared_audio()`
- Shared across Stages 7, 8, 9, and 10 that need audio access
- Reduces memory usage and improves performance
- Uses pyannote Audio loader with consistent 16kHz sample rate

**Shared Embedding Model** (`stitch_utils.py`):
- Pyannote embedding model initialized once using `initialize_shared_embedding_model()`
- Shared across Stages 7, 8, 9, and 10 for consistent embeddings
- Automatic device detection (MPS, CUDA, CPU)
- Avoids redundant model loading and GPU memory allocation

**Batch Processing Utilities**:
- `create_audio_segments_from_words()`: Convert word table to segment list for batch processing
- `create_speaker_segments_from_centroids()`: Extract segments from speaker centroids
- `stack_embeddings_efficiently()`: Stack embeddings into batch tensors with normalization
- `compute_similarity_matrix_batch()`: Efficient cosine similarity computation
- `extract_audio_embeddings_batch()`: Batch embedding extraction from audio segments

**Benefits**:
- **Memory Efficiency**: Audio loaded once instead of 3-4 times across stages
- **Performance**: Model initialization overhead eliminated for subsequent stages
- **Consistency**: All stages use identical audio data and embedding model
- **Resource Management**: Centralized resource lifecycle management

## Data Structures

### WordTable
Central data structure tracking every word through the pipeline:

```python
class WordTable:
    def __init__(self):
        self.df = pd.DataFrame()  # Main word data
        
    def assign_speaker_to_words(self, word_ids, speaker, method, confidence):
        """Assign speaker to specific words"""
        
    def get_speaker_statistics(self):
        """Get current speaker assignment statistics"""
        
    def save_to_file(self, path):
        """Save word table to JSON file"""
```

### SegmentTable
Reference structure for speaker segments:

```python
class SegmentTable:
    def __init__(self):
        self.df = pd.DataFrame()  # Segment data
        
    def get_segments_by_speaker(self, speaker):
        """Get all segments for a specific speaker"""
```

## Configuration

The pipeline uses configuration from `config/config.yaml`:

```yaml
processing:
  speaker_assignment:
    max_batch_size: 64
    min_segment_duration: 1.0
    min_embedding_quality: 0.5
    outlier_threshold: 2.0
  
  model_server:
    enabled: false
    url: "http://localhost:8002"
```

## Usage

### Basic Usage
```bash
python stitch.py --content <content_id>
```

### Test Mode
```bash
python stitch.py --content <content_id> --test
```

### Time Range Processing
```bash
python stitch.py --content <content_id> --start 60.0 --end 120.0
```

### Test Mode Output
When run with `--test` flag, outputs are saved to:
- `tests/content/{content_id}/outputs/word_table.json`
- `tests/content/{content_id}/outputs/segment_table.json`
- `tests/content/{content_id}/outputs/processing_metadata.json`
- `tests/content/{content_id}/outputs/speaker_assignment_results.json`
- `tests/content/{content_id}/outputs/{content_id}_transcript.txt` - Readable transcript
- `tests/content/{content_id}/outputs/{content_id}_transcript_detailed.txt` - Detailed word-by-word breakdown
- `tests/content/{content_id}/outputs/{content_id}_assignment_history.txt` - **Complete assignment tracking report**
- `tests/content/{content_id}/outputs/{content_id}_speaker_turns.json` - Final speaker turns data

## Performance Characteristics

### Processing Time
- **Stage 1**: Fast (file I/O dependent)
- **Stage 2**: Fast (data cleaning operations)
- **Stage 3**: Fast (data structure creation)
- **Stage 4**: Fast (single speaker assignment)
- **Stage 5**: Fast (bad grammar single speaker)
- **Stage 6**: **Slow** (speaker embeddings - major bottleneck)
- **Stage 7**: Fast (speaker centroids)
- **Stage 8**: **Moderate** (LLM-based multi-speaker analysis)
- **Stage 9**: **Moderate** (LLM-based multi-speaker analysis with audio)
- **Stage 10**: **Moderate** (LLM resolution)
- **Stage 11**: Fast (final cleanup)
- **Stage 12**: Fast (output generation)

### Memory Usage
- WordTable scales with transcript length
- **Shared Resource Benefits**: Significant memory reduction from shared audio/model loading
- Peak memory usage reduced through shared resource management
- Audio data loaded once and shared across multiple stages

### Optimization Features
- **Shared Resource Management**: Audio and embedding model loaded once, shared across stages
- **Stacked Batching**: Optimizes embedding calculation with 1.2x duration constraint
- **Quality Filtering**: Reduces processing load by filtering low-quality segments
- **Concurrent Processing**: Batch processing where possible
- **Efficient Batch Operations**: Vectorized similarity computations and embedding stacking
- **Model Caching**: Embedding model cached and reused across stages

## Error Handling

### Common Issues
1. **Missing Audio File**: Falls back to placeholder embeddings in test mode
2. **S3 Connection Errors**: Detailed error reporting with retry suggestions
3. **Model Loading Failures**: Graceful fallback between model server and local models
4. **Memory Issues**: Automatic batch size adjustment
5. **Invalid Data**: Comprehensive validation with detailed error messages

### Recovery Strategies
- **Graceful Degradation**: Pipeline continues with reduced functionality when possible
- **Detailed Logging**: Comprehensive logging for debugging
- **State Preservation**: Word assignments preserved across stage failures
- **Retry Logic**: Automatic retry for transient failures

## Quality Metrics

### Assignment Quality
- **Assignment Rate**: Percentage of words with speaker assignments
- **Confidence Scores**: Average confidence of assignments
- **Outlier Detection**: Percentage of assignments flagged as outliers
- **Method Distribution**: Breakdown of assignment methods used

### Embedding Quality
- **Embedding Norm**: Validates embedding magnitude
- **Similarity Scores**: Cosine similarity to speaker centroids
- **Duration Quality**: Quality score based on segment duration
- **Context Quality**: Quality score based on speaker context

## Troubleshooting

### Debug Mode
Enable debug logging:
```python
logger.setLevel(logging.DEBUG)
```

### Visualization
For debugging speaker assignments:
```bash
python stitch.py --content <content_id> --test --visualize
```

### Assignment History Debugging
For detailed assignment tracking analysis:
```bash
# Run pipeline in test mode to generate assignment history
python stitch.py --content <content_id> --test

# Review the assignment history report
cat tests/content/<content_id>/outputs/<content_id>_assignment_history.txt
```

**Key debugging files**:
- `{content_id}_assignment_history.txt`: Complete word-by-word assignment tracking
- `{content_id}_transcript_detailed.txt`: Technical transcript with confidence scores
- `{content_id}_timeline_visualization.txt`: Stage-by-stage timeline analysis

### Common Problems

1. **Low Assignment Rate**
   - Check diarization quality
   - Verify audio file availability
   - Review embedding quality thresholds
   - **New**: Check assignment history to see where assignments are being lost

2. **High Outlier Rate**
   - Adjust outlier threshold
   - Check speaker centroid quality
   - Review diarization accuracy
   - **New**: Use assignment history to identify which stages are producing outliers

3. **Performance Issues**
   - Verify shared resource loading is working properly
   - Reduce batch size if memory constrained
   - Enable model server for faster embeddings
   - Use time range filtering for testing

4. **Memory Issues**
   - Check if shared resources are being loaded correctly
   - Reduce max_batch_size in configuration
   - Process in smaller time chunks
   - Monitor embedding dimension requirements

5. **LLM Issues**
   - Verify llama3.2:3b model is available in Ollama
   - Check LLM response parsing for "unknown" responses
   - Review simplified prompts for clarity

6. **Speaker Assignment Errors**
   - **New**: Use assignment history to trace incorrect assignments back to their source
   - **Example Investigation**: Word assigned to wrong speaker at 298.36s
     1. Check `assignment_history.txt` for that timestamp
     2. Review stage-by-stage assignments: Stage 4 → Stage 5 → Stage 10
     3. Identify which stage made the incorrect assignment
     4. Review that stage's confidence scores and reasoning
     5. Check if boundary tolerance or embedding similarity thresholds need adjustment
   - **Pattern Analysis**: Look for systematic assignment errors across multiple words
   - **Confidence Analysis**: Identify stages with consistently low confidence scores

## Computational Bottlenecks Summary

### Primary Bottlenecks (80% of compute time)
1. **Stage 6 - Speaker Embeddings**: Deep learning inference with pyannote/embedding
   - Embedding extraction for all viable segments
   - Large similarity matrix computations
   - Clustering algorithms (DBSCAN/Agglomerative)
   - Major bottleneck: 20-25% of total pipeline time

2. **Stage 8 - Good Grammar Multi-Speaker**: LLM inference
   - Processing multi-speaker segments with good grammar
   - Context analysis with conversation flow
   - Language model inference overhead

3. **Stage 9 - Bad Grammar Multi-Speaker**: Enhanced LLM inference
   - Processing complex multi-speaker segments
   - Audio-enhanced LLM processing
   - Language-aware analysis

4. **Stage 10 - LLM Resolution**: Final language model processing
   - Resolution of remaining UNKNOWN words
   - Context-aware speaker assignment
   - Centroid-integrated analysis

### Secondary Bottlenecks (15% of compute time)
5. **Stage 1 - Data Loading**: I/O operations
   - S3 downloads (network dependent)
   - Audio decompression (opus/mp3 to WAV)
   - JSON parsing for large files

6. **Stage 11 - Final Cleanup**: Grammar enhancement
   - Final grammar and punctuation improvements
   - Consistency validation

### Memory Bottlenecks
- **Full Audio in Memory**: Entire WAV file loaded (can be GBs)
- **Multiple Models**: pyannote embeddings, spaCy NLP, LLM models
- **Large DataFrames**: WordTable with full metadata per word
- **Embedding Matrices**: Dense matrices for similarity calculations

### Inefficiencies
- **Sequential Processing**: Stages run sequentially despite some independence
- **Per-Word Processing**: Many operations iterate word-by-word
- **Multiple LLM Calls**: Separate LLM inference for stages 8, 9, and 10
- **No Streaming**: Large files processed entirely in memory

## Integration

The stitch pipeline integrates with the larger content processing system:

- **Input**: Uses output from transcribe and diarize stages
- **Output**: Feeds into segment_embeddings stage
- **Storage**: Reads from and writes to S3 storage
- **Database**: Updates content processing status
- **Orchestration**: Managed by task orchestrator system

## Archive Cleanup and Deletion Plan

### Overview
The stitch pipeline has accumulated extensive archived code from experimental development. This section provides a comprehensive cleanup plan to remove obsolete files and reduce codebase complexity.

### Current Archive Status
- **Total archived files analyzed**: ~83 files across multiple directories
- **Active dependencies**: Only 1 file still referenced (`stage4_wav2vec2.py`)
- **Safe to delete**: ~78 files (94% of archived code)
- **Estimated complexity reduction**: Significant simplification of codebase

### Archive Directories

#### 1. `/src/processing_steps/archive/` - **ALREADY DELETED**
This directory contained legacy processing pipeline components and has been removed from the codebase.

#### 2. `/src/dev/archive/` - **SAFE TO DELETE ENTIRELY**
Contains 63+ files of experimental development code:
- Multiple diarization versions (v2-v8)
- Multiple stitch versions (v2-v6) 
- Backup files and experimental approaches
- Documentation and planning files
- **No active references found in current codebase**

#### 3. `/src/processing_steps/stitch_steps/archived/` - **SAFE TO DELETE ENTIRELY**
Contains 5 recently archived files:
- `stage8_multi_speaker_resolution.py` - Replaced by current stage8
- `stage8_sentence_embeddings.py` - Functionality moved to other stages
- `stage9_whisper_fragment_cleanup.py` - Replaced by current stage9
- `stage10_speaker_consolidation.py` - Replaced by current stage10
- `stage_llm_punctuation.py` - Functionality integrated elsewhere

#### 4. `/src/processing_steps/stitch_steps/archive/` - **MOSTLY SAFE TO DELETE**
Contains 15 files with one active dependency:

**MUST KEEP (1 file):**
- `stage4_wav2vec2.py` - **ACTIVELY USED** by `util_wav2vec2.py` for OptimizedForcedAlignmentProcessor

**SAFE TO DELETE (14 files):**
- `7_word_diarization.py` - Legacy word-level diarization
- `9_llm_coherence.py` - Replaced by current stage9
- `sankey_visualizer.py` - Visualization utility, no longer used
- `single_speaker_embeddings.py` - Functionality moved to stage6
- `spacy_sentence_processor.py` - Legacy NLP processing
- `speaker_assignment.py` - Replaced by current assignment logic
- `speaker_clustering.py` - Replaced by current clustering
- `stage10_post_processing.py` - Replaced by current stage10
- `stage11_sentence_processing.py` - Replaced by current stage11
- `stage1_5_grammar_retranscribe.py` - Intermediate stage no longer used
- `stage5_overlap.py` - Replaced by current stage5
- `stage5_overlap_vectorized.py` - Replaced by current stage5
- `stage5_timing_aware_assignment.py` - Replaced by current stage5
- `stage6_context.py` - Replaced by current stage6
- `stage6_llm_assignment.py` - Replaced by current stage6
- `stage6_speaker.py` - Replaced by current stage6
- `stage6_testing.py` - Test utility no longer needed
- `stage6_unknown_retranscribe.py` - Replaced by current stage6
- `stage9_llm_coherence.py` - Replaced by current stage9
- `stage9_llm_grammar.py` - Replaced by current stage9
- `stage9_llm_grammar.py.archived` - Duplicate archived file
- `test_boundary_tolerances.py` - Legacy test utility
- `timeline_visualizer.py` - Visualization utility, no longer used

### Deletion Commands

**Phase 1: Delete Safe Directories**
```bash
# Delete the entire dev/archive directory
rm -rf /Users/signal4/content_processing/src/dev/archive/

# Delete the archived directory
rm -rf /Users/signal4/content_processing/src/processing_steps/stitch_steps/archived/
```

**Phase 2: Delete Safe Files from Archive Directory**
```bash
cd /Users/signal4/content_processing/src/processing_steps/stitch_steps/archive/

# Delete individual files (keeping stage4_wav2vec2.py)
rm -f 7_word_diarization.py sankey_visualizer.py single_speaker_embeddings.py
rm -f spacy_sentence_processor.py speaker_assignment.py speaker_clustering.py
rm -f stage10_post_processing.py stage11_sentence_processing.py stage1_5_grammar_retranscribe.py
rm -f stage5_overlap.py stage5_overlap_vectorized.py stage5_timing_aware_assignment.py
rm -f stage6_context.py stage6_llm_assignment.py stage6_speaker.py stage6_testing.py
rm -f stage6_unknown_retranscribe.py stage9_llm_coherence.py stage9_llm_grammar.py
rm -f stage9_llm_grammar.py.archived test_boundary_tolerances.py timeline_visualizer.py
```

### Current Active Pipeline
The current pipeline uses only these 12 active stage files:
- `stage1_load.py` - Data loading
- `stage2_clean.py` - Data cleaning  
- `stage3_tables.py` - Table creation
- `stage4_good_grammar_single.py` - Single speaker assignment
- `stage5_bad_grammar_single.py` - Bad grammar single speaker
- `stage6_speaker_embeddings.py` - Speaker embeddings
- `stage7_speaker_centroids.py` - Speaker centroids
- `stage8_good_grammar_multi.py` - Multi-speaker good grammar
- `stage9_bad_grammar_multi.py` - Multi-speaker bad grammar
- `stage10_resolutions.py` - LLM resolution
- `stage11_cleanup.py` - Final cleanup
- `stage12_output.py` - Output generation

### Benefits of Cleanup
1. **Reduced Complexity**: Remove 78 obsolete files (94% of archived code)
2. **Clearer Architecture**: Focus on active 12-stage pipeline
3. **Easier Maintenance**: No confusion about which files are active
4. **Improved Navigation**: Cleaner directory structure
5. **Reduced Storage**: Eliminate redundant experimental code

### Validation
Before deletion, the analysis confirmed:
- Only 1 archived file has active dependencies
- All deleted functionality is superseded by current implementation
- No references to deleted files in current codebase
- All current pipeline stages are well-defined and functional

## Future Enhancements

### Recent Improvements (Completed)
1. **✅ Shared Resource Management**: Audio and embedding model loaded once, shared across stages
2. **✅ Enhanced LLM Integration**: Upgraded to llama3.2:3b with simplified prompts
3. **✅ Speaker Consolidation**: Dedicated stage for fixing sentence-speaker splits
4. **✅ Batch Processing Utilities**: Efficient vectorized operations for embeddings
5. **✅ Comprehensive Assignment Tracking**: Complete word-by-word assignment history through all pipeline stages
6. **✅ Assignment History Debugging**: Detailed reports showing exactly how each word was assigned
7. **✅ Tracked Assignment Methods**: All stages now use standardized assignment tracking
8. **✅ Assignment Audit Compliance**: Every speaker assignment change is logged with stage, method, confidence, and reasoning
9. **✅ Duration-Based Speaker Merging**: Stage 7 now selects primary speaker based on total diarization time
10. **✅ LLM-Only Resolution**: Stages 8 and 10 simplified to remove embedding validation
11. **✅ Archive Cleanup Plan**: Comprehensive analysis and deletion plan for obsolete archived code

### Planned Improvements
1. **Adaptive Batching**: Dynamic batch size based on available memory
2. **Multi-Model Support**: Support for different embedding models
3. **Real-time Processing**: Streaming support for live audio
4. **Quality Feedback**: Iterative improvement based on quality metrics
5. **Cross-Content Learning**: Speaker recognition across multiple content items
6. **Archive Cleanup Execution**: Implement the deletion plan to simplify codebase

### Research Areas
- **Temporal Consistency**: Ensuring speaker assignments are temporally consistent
- **Voice Activity Detection**: Better integration with VAD for segment boundaries
- **Multi-Language Support**: Extension to non-English content (partially implemented)
- **Emotional Context**: Incorporating emotional state in speaker assignment
- **Cross-Modal Learning**: Combining acoustic and linguistic features for better assignments

## End-to-End Pipeline Evaluation

### Architecture Assessment

#### Strengths
1. **Modular Design**: Each stage has clear responsibilities and can be tested/improved independently
2. **Progressive Refinement**: Multiple passes allow correction of earlier mistakes
3. **Hybrid Approach**: Combines acoustic analysis (embeddings) with linguistic context (LLMs)
4. **Comprehensive Tracking**: Full assignment history enables debugging and quality analysis
5. **Resource Efficiency**: Shared audio and model loading reduces memory overhead

#### Weaknesses
1. **Sequential Processing**: Stages must run in order, limiting parallelization opportunities
2. **Multiple Model Dependencies**: Requires Wav2Vec2, pyannote, multiple LLMs (computational overhead)
3. **Memory Intensive**: Full audio and word tables kept in memory throughout
4. **Redundant Operations**: Some cleaning and validation repeated across stages
5. **Limited Error Recovery**: Errors in early stages propagate through pipeline

### Computational Analysis

#### Major Bottlenecks (by computation time)
1. **Stage 5 - Wav2Vec2 Alignment (30-40% of total time)**
   - Transformer inference on audio segments
   - Per-word processing without batching
   - DTW alignment computation
   - **Optimization Opportunity**: Batch multiple words for transformer inference

2. **Stage 7 - Speaker Embeddings (20-25% of total time)**
   - pyannote/embedding model inference
   - Similarity matrix computations
   - Clustering algorithms
   - **Already Optimized**: Uses batching and shared resources

3. **Stage 3 - LLM Punctuation (10-15% of total time)**
   - LLM inference for text enhancement
   - Sequential segment processing
   - **Optimization Opportunity**: Larger batch sizes for LLM

4. **Stages 8 & 10 - LLM Resolution (10-15% of total time)**
   - Multiple LLM calls for uncertain segments
   - Now simplified without embeddings (faster)
   - **Current State**: Already optimized by removing embeddings

#### Memory Bottlenecks
1. **Audio Data**: Full WAV file in memory (can be several GB)
2. **Models**: Multiple large models loaded simultaneously
3. **Word Tables**: Full metadata for every word
4. **Intermediate Results**: Assignment history, embeddings, etc.

### Accuracy vs Performance Trade-offs

#### Current Balance
- **High Accuracy Focus**: Multiple validation stages, comprehensive tracking
- **Performance Sacrifices**: Sequential processing, multiple models, repeated validations
- **LLM Reliance**: Heavy use of LLMs for human-readable output

#### Trade-off Analysis
1. **Embedding Validation Removal (Stages 8 & 10)**
   - **Benefit**: 30-40% faster processing in these stages
   - **Risk**: Potential accuracy loss on acoustically distinct speakers
   - **Mitigation**: LLMs handle conversational context well

2. **Duration-Based Speaker Merging (Stage 7)**
   - **Benefit**: More intuitive primary speaker selection
   - **Risk**: Quality metrics ignored in favor of duration
   - **Mitigation**: Duration often correlates with recording quality

3. **Multiple LLM Usage**
   - **Benefit**: High-quality punctuation and context understanding
   - **Cost**: Significant computational overhead
   - **Alternative**: Could use rule-based systems for simple cases

### Scalability Concerns

1. **Linear Complexity**: Processing time scales linearly with content duration
2. **Memory Scaling**: Memory usage grows with transcript length
3. **Model Loading**: Fixed overhead regardless of content size
4. **No Streaming**: Must process entire file at once

### Recommendations for Production

#### Short-term Optimizations
1. **Batch Processing**: Implement batching in Stage 5 (Wav2Vec2)
2. **Streaming Architecture**: Process audio in chunks to reduce memory
3. **Model Caching**: Keep models warm between pipeline runs
4. **Parallel Stages**: Run independent stages concurrently

#### Long-term Improvements
1. **Unified Model**: Single model for embeddings + alignment
2. **Incremental Processing**: Process new content incrementally
3. **Distributed Processing**: Split long content across workers
4. **Adaptive Quality**: Adjust processing depth based on content complexity

### Quality Metrics Recommendations

1. **Speaker Consistency Score**: Measure speaker changes per minute
2. **Confidence Distribution**: Track assignment confidence across stages
3. **LLM Agreement Rate**: Compare LLM decisions with embedding-based decisions
4. **Temporal Coherence**: Measure speaker assignment stability over time
5. **Cross-Stage Agreement**: Track how often stages override previous assignments

### Final Assessment

The Stitch pipeline represents a sophisticated approach to speaker attribution that prioritizes accuracy and debuggability over raw performance. The recent simplifications (removing embeddings from Stages 8 & 10) show a pragmatic evolution toward trusting LLM context understanding for conversational analysis.

**Key Strengths**:
- Comprehensive speaker attribution with multiple validation layers
- Excellent debugging capabilities through assignment tracking
- Flexible architecture allowing stage-by-stage improvements
- Strong handling of complex conversational patterns

**Key Limitations**:
- High computational cost for production scale
- Memory intensive for long-form content
- Sequential processing limits parallelization
- Multiple model dependencies increase complexity

**Recommended Use Cases**:
- High-value content requiring maximum accuracy
- Content with complex multi-speaker interactions
- Scenarios where debugging and explainability are critical
- Research and development of speaker attribution techniques

**Not Recommended For**:
- Real-time or near-real-time processing
- Very long content (>3 hours) on memory-constrained systems
- High-volume production without significant compute resources
- Simple two-speaker conversations (overengineered for this case)