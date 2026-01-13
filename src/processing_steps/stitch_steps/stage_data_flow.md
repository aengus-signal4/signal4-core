# Stitch Pipeline Stage Data Flow Analysis

## Overview
This document tracks what each stage outputs and which stages consume those outputs in the stitch pipeline. The pipeline consists of 11 stages that progressively refine speaker attribution at the word level.

## Stage-by-Stage Data Flow

### Stage 1: Data Loading (`stage1_load.py`)

**Outputs:**
- `diarization_data` - Raw speaker diarization segments from `diarization.json`
  - Format: Dictionary with 'segments' array containing speaker, start, end times
- `transcript_data` - Raw word-level transcription from `transcript_words.json`
  - Format: Dictionary with 'segments' array containing words, timing, confidence
- `audio_path` - Path to audio file (WAV format, possibly converted from opus/mp3)
- **Stats**: diarization_segments count, transcript_segments count

**Used by:**
- Stage 2: Uses both `diarization_data` and `transcript_data` for cleaning
- Audio path passed through pipeline for stages needing audio access

---

### Stage 2: Data Cleaning (`stage2_clean.py`)

**Outputs:**
- `cleaned_diarization_data` - Filtered and cleaned diarization segments
  - Removes overlapping segments, merges adjacent same-speaker segments
- `cleaned_transcript_data` - Cleaned transcript with artifacts removed
  - Removes boundary overlaps, low confidence words, invalid timestamps
- **Stats**: `transcript_cleaning_stats` including:
  - `total_final_words` - Word count after cleaning
  - `total_artifacts_removed` - Number of removed artifacts

**Used by:**
- Stage 3: Uses `cleaned_transcript_data` for punctuation enhancement
- Stage 4: Uses both cleaned datasets for table creation
- Stage 6: Uses `cleaned_diarization_data` for speaker context analysis
- Stage 8: Uses `cleaned_diarization_data` for multi-speaker resolution

---

### Stage 3: LLM-Based Punctuation Enhancement (`stage3_punctuation.py`)

**Outputs:**
- `enhanced_transcript_data` - Transcript with improved punctuation
  - Same structure as transcript_data but with LLM-enhanced text
- **Stats**: 
  - `segments_processed` - Total segments analyzed
  - `segments_enhanced` - Segments with punctuation improvements

**Used by:**
- Stage 4: Uses `enhanced_transcript_data` to create word table with properly punctuated text

---

### Stage 4: Table Creation (`stage4_tables.py`)

**Outputs:**
- `word_table` (WordTable object) - Central data structure tracking all words
  - DataFrame with columns: word_id, text, start, end, confidence, speaker_current, resolution_method, processing_status, metadata, assignment_history
  - Initially all speakers set to 'UNKNOWN'
- `segment_table` (SegmentTable object) - Reference structure for speaker segments
  - DataFrame tracking diarization segments
- **Stats**: `words_created`, `segments_created`

**Used by:**
- ALL subsequent stages (5-11) use and modify the `word_table`
- Stage 6 uses `segment_table` for reference

---

### Stage 5: Wav2Vec2 Forced Alignment (`stage5_wav2vec2.py`)

**Inputs:**
- `word_table` from Stage 4
- `audio_path` from Stage 1

**Outputs:**
- `word_table` (modified) - Updated with refined word timing boundaries
  - Updates: start/end times adjusted based on phoneme-level alignment
  - Adds: alignment_confidence scores, refined timing metadata
- **Stats**: alignment status, processing metrics

**Used by:**
- Stage 6 and all subsequent stages benefit from improved word timing

---

### Stage 6: Speaker Context Analysis (`stage6_speaker.py`)

**Inputs:**
- `word_table` from Stage 5
- `cleaned_diarization_data` from Stage 2
- `cleaned_transcript_data` from Stage 2 (for reference)

**Outputs:**
- `word_table` (modified) - Updated with speaker context classification
  - Adds: speaker_context field ('single_speaker', 'multi_speaker', 'unknown')
  - Adds: diarization_confidence scores
- `classification_stats` - Dictionary with counts:
  - `single_speaker` - Words with clear single speaker
  - `multi_speaker` - Words overlapping multiple speakers
  - `unknown` - Words with no diarization overlap
- `multi_speaker_segments` - List of segments with multiple speakers

**Used by:**
- Stage 7: Uses `classification_stats` as `speaker_context_stats`
- Pipeline uses stats for configuration decisions

---

### Stage 7: Speaker Assignment (`stage7_speaker_embeddings.py`)

**Inputs:**
- `word_table` from Stage 6
- `speaker_context_stats` from Stage 6
- `audio_path` from Stage 1 (or shared_audio_data)
- `shared_embedding_model` from pipeline initialization

**Outputs:**
- `word_table` (modified) - Major speaker assignments
  - Updates: speaker_current with actual speaker IDs
  - Adds: assignment_method, assignment_confidence
- `assignment_result` - Dictionary containing:
  - `status` - 'completed' or error status
  - `final_stats` - Assignment rate, unique speakers
  - `speaker_centroids` - Speaker embedding centroids
  - `speaker_merge_mappings` - Merged speaker mappings
- `assignment_summary` - High-level summary statistics
- `speaker_centroids` - Dictionary of speaker embeddings
- `speaker_centroid_data` - Detailed centroid metadata

**Used by:**
- Stage 7b: Uses `speaker_centroid_data` for database integration
- Stage 8: Uses `speaker_centroids` and `speaker_merge_mappings`
- Stage 9: Uses `speaker_centroids` for embedding comparison

---

### Stage 7b: Speaker Centroids (`stage7b_speaker_centroids.py`)

**Inputs:**
- `word_table` from Stage 7
- `speaker_centroid_data` from Stage 7

**Outputs:**
- `word_table` (modified) - Updated with global speaker mappings
  - Updates: Maps local speaker IDs to universal database IDs
- `speaker_mapping` - Dictionary mapping local to global speaker IDs
  - Format: {local_id: {universal_name, is_new}}
- **Stats**: Centroid quality metrics, mapping statistics

**Used by:**
- Subsequent stages use global speaker IDs
- Database integration for cross-content speaker tracking

---

### Stage 8: Multi-Speaker Resolution (`stage8_multi_speaker_resolution.py`)

**Inputs:**
- `word_table` from Stage 7/7b
- `speaker_centroids` from Stage 7
- `shared_audio_data` from pipeline
- `shared_embedding_model` from pipeline
- `cleaned_diarization_data` from Stage 2
- `speaker_merge_mappings` from Stage 7

**Outputs:**
- `word_table` (modified) - Resolved multi-speaker and unknown words
  - Updates: speaker_current for MULTI_SPEAKER and UNKNOWN words
  - Method: LLM-based conversational analysis (no embedding validation)
- **Stats**: 
  - `words_resolved` - Number of words assigned
  - Resolution breakdown by method

**Used by:**
- Stage 9 and subsequent stages work with fewer unknowns

---

### Stage 9: Sentence-Level Speaker Embeddings (`stage8_sentence_embeddings.py`)

**Inputs:**
- `word_table` from Stage 8
- `audio_path` (or shared_audio_data)
- `speaker_centroids` from Stage 7
- `shared_embedding_model` from pipeline

**Outputs:**
- `word_table` (modified) - Assigns remaining unknown words
  - Updates: speaker_current for remaining UNKNOWN words
  - Method: Word-level embedding extraction and comparison
  - Includes: Fragment-based propagation, interruption handling
- **Stats**:
  - `embeddings_extracted` - Number of embeddings calculated
  - `words_assigned` - Words assigned via embeddings
  - `words_propagated` - Words assigned via fragment propagation
  - `reason` - 'no_unknown_words' if stage skipped

**Special Processing:**
- Post-stage vectorized same-speaker gap resolution
- Single-word interruption handling
- Fragment-based speaker propagation

**Used by:**
- Stage 10 receives word table with minimal unknowns

---

### Stage 10: Speaker Consolidation (`stage10_speaker_consolidation.py`)

**Inputs:**
- `word_table` from Stage 9

**Outputs:**
- `word_table` (modified) - Consolidated speaker assignments
  - Fixes: Sentences incorrectly split across speakers
  - Method: Rule-based + LLM analysis (no embeddings)
- **Stats**:
  - `sentences_consolidated` - Number of sentences fixed
  - `split_sentences` - Number of split sentences detected
  - Processing metadata

**Note:** This stage previously included fragment cleanup (stage9_whisper_fragment_cleanup.py) which is now integrated

**Used by:**
- Stage 11 for final output generation

---

### Stage 11: Output Generation (`stage11_output.py`)

**Inputs:**
- `word_table` from Stage 10 (final version)

**Outputs:**
- `readable_transcript` - Human-readable formatted transcript
- `detailed_transcript` - Technical transcript with timing/confidence
- `speaker_turns` - Structured speaker turn data
- `output_files` - List of generated files
- **Production outputs** (S3):
  - `transcript_diarized.json` - Final diarized transcript
  - `speaker_centroids.json` - Speaker embedding centroids
- **Test mode outputs**:
  - All word/segment tables as JSON
  - Assignment history report
  - Processing metadata

**Used by:**
- Content processing system for final transcript storage
- Database for marking content as stitched
- Downstream embedding/search systems

---

## Key Data Structures Flow

### WordTable Evolution
1. **Stage 4**: Initialized with all words, speakers='UNKNOWN'
2. **Stage 5**: Timing boundaries refined via Wav2Vec2
3. **Stage 6**: Speaker context added (single/multi/unknown)
4. **Stage 7**: Primary speaker assignments via embeddings
5. **Stage 7b**: Local→global speaker ID mapping
6. **Stage 8**: Multi-speaker/unknown resolution via LLM
7. **Stage 9**: Remaining unknowns via word embeddings
8. **Stage 10**: Sentence consolidation fixes
9. **Stage 11**: Final output generation

### Shared Resources
- **Audio Data**: Loaded once in pipeline, shared by Stages 5, 7, 8, 9
- **Embedding Model**: Initialized once, shared by Stages 7, 8, 9
- **Speaker Centroids**: Created in Stage 7, used by Stages 8, 9

### Assignment History Tracking
Every word maintains complete history through all stages:
- Stage identification
- Assignment method
- Confidence score
- Detailed reasoning
- Timestamp of change

## Cache Dependencies

When using `--start-from-stage`, the following dependencies must be cached:

- **Stage 5**: Requires Stage 1-4 cached results
- **Stage 6**: Requires Stage 1-5 cached results
- **Stage 7**: Requires Stage 1-6 cached results
- **Stage 8**: Requires Stage 1-7 cached results (including speaker centroids)
- **Stage 9**: Requires Stage 1-8 cached results
- **Stage 10**: Requires Stage 1-9 cached results
- **Stage 11**: Requires Stage 1-10 cached results

## Performance Notes

### Memory-Intensive Data
- Full audio file (shared across stages)
- WordTable with complete metadata
- Speaker embeddings and centroids
- Assignment history for all words

### Computational Bottlenecks
- Stage 5: Wav2Vec2 alignment (30-40% of time)
- Stage 7: Speaker embeddings (20-25% of time)
- Stage 3: LLM punctuation (10-15% of time)
- Stages 8 & 10: LLM resolution (10-15% of time)

### Data Dependencies
- Stages 6-10 all depend on Stage 5's improved timing
- Stages 8-9 depend on Stage 7's speaker centroids
- Stage 8 depends on Stage 7's merge mappings
- All stages 5-11 modify the central WordTable