# Stitch Pipeline Optimization Plan

## Executive Summary

The stitch pipeline is a sophisticated 12-stage system for speaker attribution that prioritizes accuracy over performance. This optimization plan identifies key bottlenecks and provides actionable recommendations to improve performance while maintaining quality.

## Current Performance Profile

### Major Bottlenecks (80% of compute time)
1. **Stage 6 - Speaker Embeddings** (20-25%)
   - pyannote embedding model inference
   - Similarity matrix computations
   - Speaker clustering and merging

2. **Stage 9 - Bad Grammar Multi** (10-15%)
   - LLM inference for complex segments
   - Context building and processing
   - Multiple retry attempts

3. **Stage 10 - LLM Resolution** (10-15%)
   - Final resolution of UNKNOWN words
   - Full context processing
   - Temperature-based retries

4. **Stage 1 - Data Loading** (5-10%)
   - S3 downloads (network dependent)
   - Audio decompression
   - JSON parsing of large files

### Memory Bottlenecks
- Full audio file in memory (can be several GB)
- Multiple large models loaded simultaneously
- Complete WordTable with full metadata
- Dense embedding matrices

## Optimization Recommendations

### 1. High-Impact Optimizations (Implement First)

#### A. Batch Processing Enhancement
**Current Issue**: Many operations process words individually
**Solution**: 
```python
# Instead of:
for word in words:
    embedding = extract_embedding(word)
    
# Use:
embeddings = extract_embeddings_batch(words, batch_size=64)
```
**Impact**: 40-60% reduction in Stage 6 processing time
**Implementation Effort**: Medium

#### B. Lazy Audio Loading
**Current Issue**: Full audio loaded in Stage 1, may not be needed until Stage 6
**Solution**:
```python
class LazyAudioLoader:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self._audio_data = None
    
    @property
    def audio_data(self):
        if self._audio_data is None:
            self._audio_data = load_audio(self.audio_path)
        return self._audio_data
```
**Impact**: Reduced memory usage, faster startup
**Implementation Effort**: Low

#### C. Parallel Stage Execution
**Current Issue**: Stages run sequentially even when independent
**Solution**: Run stages 4 & 5 in parallel after stage 3
```python
async def run_parallel_stages():
    stage4_task = asyncio.create_task(stage4_good_grammar_single())
    stage5_task = asyncio.create_task(stage5_bad_grammar_single())
    await asyncio.gather(stage4_task, stage5_task)
```
**Impact**: 15-20% total pipeline time reduction
**Implementation Effort**: Medium

### 2. Medium-Impact Optimizations

#### A. Smart Caching
**Current Issue**: Models loaded fresh for each content
**Solution**: Implement model server or warm cache
```python
class ModelCache:
    _instance = None
    _models = {}
    
    @classmethod
    def get_model(cls, model_name):
        if model_name not in cls._models:
            cls._models[model_name] = load_model(model_name)
        return cls._models[model_name]
```
**Impact**: 5-10% reduction for multiple content processing
**Implementation Effort**: Low

#### B. Vectorized Operations
**Current Issue**: Many pandas operations use iterrows()
**Solution**: Use vectorized pandas/numpy operations
```python
# Instead of:
for idx, row in df.iterrows():
    df.at[idx, 'result'] = process(row['value'])
    
# Use:
df['result'] = df['value'].apply(process)
# Or better:
df['result'] = vectorized_process(df['value'].values)
```
**Impact**: 10-15% improvement in table operations
**Implementation Effort**: Medium

#### C. Reduce Redundant Computations
**Current Issue**: Some computations repeated across stages
**Solution**: Cache intermediate results in WordTable metadata
```python
# Add to WordTable
def cache_computation(self, word_id, key, value):
    self.df.loc[self.df['word_id'] == word_id, 'metadata'][key] = value
    
def get_cached_computation(self, word_id, key):
    return self.df.loc[self.df['word_id'] == word_id, 'metadata'].get(key)
```
**Impact**: 5-10% reduction in repeated calculations
**Implementation Effort**: Low

### 3. Low-Impact but Easy Optimizations

#### A. Remove Debug Logging in Production
**Current Issue**: Excessive logging in hot paths
**Solution**: Use log levels appropriately
```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Processing {len(words)} words")
```
**Impact**: 2-3% improvement
**Implementation Effort**: Trivial

#### B. Optimize Data Structures
**Current Issue**: Using lists where sets would be faster
**Solution**: Use appropriate data structures
```python
# Instead of:
if speaker in speaker_list:  # O(n)
    
# Use:
if speaker in speaker_set:  # O(1)
```
**Impact**: 1-2% improvement
**Implementation Effort**: Trivial

### 4. Architectural Improvements (Long-term)

#### A. Streaming Architecture
**Current Issue**: Must process entire file at once
**Solution**: Process in chunks with streaming
```python
class StreamingStitchPipeline:
    def process_chunk(self, chunk_data):
        # Process 5-minute chunks
        # Maintain context across chunks
```
**Impact**: Handle arbitrarily long content
**Implementation Effort**: High

#### B. GPU Acceleration
**Current Issue**: Limited GPU utilization
**Solution**: Batch operations for GPU processing
```python
# Use torch.cuda for batch processing
with torch.cuda.amp.autocast():
    embeddings = model(batch_audio)
```
**Impact**: 50-70% reduction in embedding extraction
**Implementation Effort**: Medium

#### C. Microservice Architecture
**Current Issue**: Monolithic pipeline
**Solution**: Split into services
- Embedding service (GPU optimized)
- LLM service (can scale independently)
- Core logic service
**Impact**: Better resource utilization
**Implementation Effort**: Very High

## Stage-Specific Optimizations

### Stage 1: Data Loading
1. **Parallel chunk loading**: Load multiple chunks concurrently
2. **Streaming decompression**: Don't load entire compressed file
3. **Smart prefetching**: Start loading next content while processing

### Stage 2: Data Cleaning
1. **Vectorized cleaning**: Use numpy for artifact detection
2. **Compiled regex**: Pre-compile all regex patterns
3. **Early filtering**: Remove artifacts before creating objects

### Stage 3: Table Creation
1. **Bulk DataFrame creation**: Create DataFrame once, not incrementally
2. **Categorical data types**: Use pandas categoricals for speakers
3. **Index optimization**: Set appropriate indexes upfront

### Stage 4-5: Grammar-based Assignment
1. **Merge stages**: These could be combined into one pass
2. **Vectorized overlap**: Calculate all overlaps at once
3. **Skip recomputation**: Cache overlap calculations

### Stage 6: Speaker Embeddings
1. **Batch extraction**: Process multiple segments together
2. **Reduced precision**: Use float16 for embeddings
3. **Approximate similarity**: Use FAISS for large-scale similarity

### Stage 7: Speaker Centroids
1. **Incremental updates**: Don't recompute all centroids
2. **Sparse operations**: Many speakers have few samples
3. **Database batching**: Batch all DB operations

### Stage 8-9: Multi-Speaker Resolution
1. **Combined processing**: Merge these similar stages
2. **Context caching**: Reuse context windows
3. **Simplified prompts**: Reduce LLM token usage

### Stage 10: LLM Resolution
1. **Batch LLM calls**: Group similar contexts
2. **Caching responses**: Cache similar patterns
3. **Fallback reduction**: Better early assignment reduces load

### Stage 11-12: Output Generation
1. **Streaming output**: Write as we go, not all at end
2. **Parallel formatting**: Format while processing
3. **Lazy evaluation**: Only compute requested outputs

## Implementation Priority

### Phase 1 (Quick Wins - 1 week)
- [ ] Remove debug logging in production
- [ ] Implement lazy audio loading
- [ ] Add basic caching for models
- [ ] Optimize data structures (lists → sets)

### Phase 2 (High Impact - 2-3 weeks)
- [ ] Implement batch processing for embeddings
- [ ] Add vectorized operations for DataFrames
- [ ] Parallelize independent stages (4 & 5)
- [ ] Cache intermediate computations

### Phase 3 (Architecture - 4-6 weeks)
- [ ] Design streaming architecture
- [ ] Implement GPU acceleration
- [ ] Create model server
- [ ] Add comprehensive caching layer

### Phase 4 (Long-term)
- [ ] Microservice architecture
- [ ] Distributed processing
- [ ] Real-time streaming support
- [ ] Cross-content speaker tracking

## Expected Results

### Overall Performance Gains
- **Phase 1**: 10-15% improvement
- **Phase 2**: 30-40% improvement  
- **Phase 3**: 50-60% improvement
- **Phase 4**: 2-3x improvement

### Memory Usage Reduction
- **Phase 1**: 10% reduction
- **Phase 2**: 25% reduction
- **Phase 3**: 40% reduction
- **Phase 4**: 60% reduction (with streaming)

### Quality Impact
- All optimizations maintain current accuracy
- Some may improve quality through better batching
- Streaming enables processing of longer content

## Testing Strategy

### Performance Benchmarks
1. Create benchmark suite with various content lengths
2. Measure time and memory for each stage
3. Track optimization impact stage-by-stage

### Quality Validation
1. Compare outputs before/after optimization
2. Ensure assignment accuracy maintained
3. Validate edge cases still handled correctly

### Regression Testing
1. Comprehensive test suite for each stage
2. A/B testing for major changes
3. Gradual rollout with monitoring

## Conclusion

The stitch pipeline has significant optimization opportunities that can improve performance by 2-3x while maintaining quality. The phased approach allows for incremental improvements with quick wins in Phase 1 and transformative changes in later phases. Priority should be given to batch processing and parallel execution as these provide the best return on investment.