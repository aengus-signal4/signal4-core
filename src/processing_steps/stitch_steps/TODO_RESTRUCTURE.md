# Stitch Pipeline Restructuring TODO

## Overview
This document tracks the restructuring of the stitch pipeline stages to improve accuracy and efficiency.

## Major Changes

### ✅ 1. Move 1.0s calculation to Stage 3
- [x] Add `segment_within_1s_of_non_majority_speaker` column to WordTable
- [x] Initialize the column in word entries
- [x] Implement `_check_1s_of_non_majority_speaker()` method
- [x] Calculate during segment attribute check
- [x] Add logging for the new statistic

### ✅ 2. Update Stage 5 to use new 1.0s calculation
- [x] Add new column to segment aggregation
- [x] Replace complex safety check with simple column check
- [x] Verify EdgeRetranscriber uses Parakeet for English, Whisper for other languages
- [x] Add language parameter to EdgeRetranscriber
- [ ] Test re-transcription logic with new safety check
- [ ] TODO: Get language from transcript data or content metadata (currently defaults to 'en')

### ✅ 3. Reverse stages - Embeddings/Centroids before Multi-speaker
- [x] Rename stage files (9->6, 10->7, 6->8, 7->9, 8->10)
- [x] Update stage imports in main stitch.py
- [x] Create placeholder stages 6 & 7 for speaker embeddings/centroids
- [x] Update stage ordering in stitch.py execution
- [x] Stage 8 now calls good_grammar_multi_speaker_stage
- [x] Stage 9 now calls bad_grammar_multi_speaker_stage  
- [x] Stage 10 now calls llm_resolution
- [x] Update caching keys to match new stage numbers
- [x] Test new stage ordering - working correctly

### ✅ 4. Update new Stage 8 (good grammar + multi-speaker)
- [x] Implement diarization overlap-based assignment
- [x] Respect each diarization segment to have at least 1 sentence fragment
- [x] Fallback to majority speaker when needed
- [x] Add placeholder for speaker embedding confirmation
- [x] Example implementation for the provided case

### ✅ 5. Update new Stage 9 (bad grammar + multi-speaker)
- [x] Re-transcribe all diarization segments
- [x] Use Parakeet for English, Whisper for other languages
- [x] Assign well-formed segments to speakers immediately
- [x] Add placeholder for speaker embedding confirmation

### ⏳ 6. Create Stage 10 - LLM Resolution
- [ ] Create stage10_llm_resolution.py (rename current stage8)
- [ ] Handle remaining UNKNOWN words
- [ ] Implement LLM-based resolution logic

### ⏳ 7. Create Stage 11 - Cleanup
- [ ] Create stage11_cleanup.py (rename current stage11)
- [ ] Merge single words bounded by identical speakers
- [ ] Fix grammar and proper nouns throughout
- [ ] Implement consolidation logic

### ⏳ 8. Update Stage 12 - Final Output
- [ ] Update stage12_output.py (already exists)
- [ ] Ensure it handles the new pipeline structure
- [ ] Generate final transcripts

## Testing Plan

### Test Command
```bash
python src/processing_steps/stitch.py --content 6IYSMEdQkH8 --test --start-from-stage 5 --stages 5
```

### Test Scenarios
1. [ ] Test Stage 5 with segments within 1.0s of non-majority speaker
2. [ ] Test Stage 5 with safe segments (no adjacent speakers)
3. [ ] Verify re-transcription works correctly
4. [ ] Test full pipeline with all stages
5. [ ] Verify speaker assignments are correct
6. [ ] Check that embeddings and centroids are calculated before multi-speaker stages

## Implementation Notes

### Language Detection
- Use Parakeet MLX for English transcription (faster, better word boundaries)
- Use Whisper for non-English languages
- Language detection should be done at the content level

### Speaker Assignment Priority
1. Diarization overlap (primary method)
2. Speaker embeddings (confirmation/refinement)
3. LLM resolution (last resort for ambiguous cases)

### Edge Case Handling
- Segments with <50% dominant speaker → multi-speaker processing
- Segments within 1.0s of non-majority → re-transcription
- Single words between identical speakers → merge in cleanup

## Progress Tracking
- Started: 2025-01-03
- Completed: 
  - Stage 3 and 5 updates (1.0s calculation and re-transcription)
  - Stage reordering (6/7 and 8/9 reversed)
  - Stage 10 (LLM resolution) moved to correct position
  - Stage 8 update (good grammar multi-speaker with diarization coverage)
  - Stage 9 update (bad grammar multi-speaker with language support)
- Tested: 
  - Stage 5 test successful - 1.0s calculation working correctly
  - Stage 6 test successful - speaker embeddings integrated
  - Stage 7 test successful - speaker centroids integrated
- Current Stage: Stages 8 & 9 implementation complete
- Next: Create stage 11 (cleanup) and update stage 12 (output)

## Current State
- All stages are in the correct order
- Stage 6 & 7 are fully integrated speaker embeddings/centroids
- Stage 8 implements good grammar multi-speaker with diarization coverage
- Stage 9 implements bad grammar multi-speaker with language-aware re-transcription
- Stage 10 is LLM resolution (moved from previous stage 8)
- Stage 11 & 12 remain to be updated (cleanup & output)

## Stage Reordering Status - COMPLETE ✅
- Stage 6: speaker_embeddings (fully integrated)
- Stage 7: speaker_centroids (fully integrated)
- Stage 8: good_grammar_multi (updated with diarization coverage)
- Stage 9: bad_grammar_multi (updated with language support)
- Stage 10: llm_resolution (moved from stage 8)

## Notes
- The 1.0s calculation is now a segment-level variable for vectorized operations
- Edge re-transcription should improve speaker boundary detection
- Embeddings/centroids before multi-speaker stages allows for better confirmation
- Stage 6 & 7 are fully implemented and ready:
  - Stage 6 prioritizes single-speaker segments with good grammar (confidence ≥ 0.7, duration ≥ 2s)
  - Stage 7 integrates with database for global speaker tracking
  - These stages verify and refine assignments from earlier stages
- Stage 8 & 9 updates:
  - Stage 8 ensures each diarization segment has at least 1 sentence fragment
  - Stage 9 uses Parakeet MLX for English, Whisper for other languages
  - Both stages add placeholder for speaker embedding confirmation