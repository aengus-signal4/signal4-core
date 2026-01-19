"""
Analysis Pipeline
=================

Composable pipeline framework for building RAG analysis workflows.

Features:
- Fluent API for chaining operations
- Full streaming support with async generators (streams within steps)
- Progress tracking with partial results
- Layer 1 integration (SegmentRetriever, EmbeddingIndexer)
- Component sharing (reuse initialized components)
- TextGenerator integration with batch progress
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    name: str
    data: Dict[str, Any]
    execution_time_ms: int
    steps_completed: int
    total_steps: int


class AnalysisPipeline:
    """
    Composable analysis pipeline with fluent API and streaming support.

    Example:
        # Initialize with services
        pipeline = AnalysisPipeline(
            "my_analysis",
            llm_service=llm_service,
            db_session=session
        )

        # Build pipeline
        pipeline = (
            pipeline
            .retrieve_segments(projects=["Europe"], languages=["en", "fr"])
            .extract_themes(n_clusters=10)
            .select_segments(strategy="diversity", n=10)
            .generate_summaries(template="theme_summary", max_concurrent=20)
        )

        # Batch execution
        result = await pipeline.execute()

        # Streaming execution
        async for update in pipeline.execute_stream():
            if update["type"] == "partial":
                print(f"Partial result: {update['data']}")
    """

    def __init__(
        self,
        name: str,
        llm_service=None,
        embedding_service=None,
        db_session=None,
        dashboard_id: str = None,
        config=None
    ):
        """
        Initialize pipeline with shared components.

        Args:
            name: Pipeline identifier for logging
            llm_service: LLMService instance (required for text generation)
            embedding_service: EmbeddingService instance (required for query embeddings)
            db_session: Database session (required for segment retrieval)
            dashboard_id: Dashboard ID (required for SearchService initialization)
            config: Dashboard config (required for SearchService initialization)
        """
        self.name = name
        self.steps: List[tuple] = []
        self.context: Dict[str, Any] = {}

        # Shared components (initialized once, reused across steps)
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.db_session = db_session
        self.dashboard_id = dashboard_id
        self.config = config

        # Lazy-initialized components
        self._retriever = None
        self._extractor = None
        self._selector = None
        self._generator = None
        self._quantitative_analyzer = None
        self._search_service = None

        logger.info(f"Created pipeline: {name}")

    # ========================================================================
    # Component Access (Lazy Initialization)
    # ========================================================================

    def _get_retriever(self):
        """Get or create SegmentRetriever."""
        if self._retriever is None:
            from .segment_retriever import SegmentRetriever
            self._retriever = SegmentRetriever(db=self.db_session)
        return self._retriever

    def _get_extractor(self):
        """Get or create ThemeExtractor."""
        if self._extractor is None:
            from .theme_extractor import ThemeExtractor
            self._extractor = ThemeExtractor()
        return self._extractor

    def _get_selector(self):
        """Get or create SegmentSelector."""
        if self._selector is None:
            from .segment_selector import SegmentSelector
            self._selector = SegmentSelector()
        return self._selector

    def _get_generator(self):
        """Get or create TextGenerator."""
        if self._generator is None:
            if self.llm_service is None:
                raise RuntimeError("TextGenerator requires llm_service to be provided in pipeline initialization")
            from .text_generator import TextGenerator
            self._generator = TextGenerator(self.llm_service)
        return self._generator

    def _get_quantitative_analyzer(self):
        """Get or create QuantitativeAnalyzer."""
        if self._quantitative_analyzer is None:
            from .quantitative_analyzer import QuantitativeAnalyzer
            self._quantitative_analyzer = QuantitativeAnalyzer(db_session=self.db_session)
        return self._quantitative_analyzer

    # ========================================================================
    # Fluent API - Step Builders
    # ========================================================================

    def expand_query(
        self,
        query: str,
        strategy: str = "multi_query",
        **kwargs
    ) -> "AnalysisPipeline":
        """
        Add query expansion step.

        Expands user query into multiple variations for better semantic retrieval.
        Must be called BEFORE retrieve_segments_by_search().

        Args:
            query: Original user query
            strategy: Expansion strategy
                - "multi_query": Generate 10 diverse variations (5 EN + 5 FR) [default]
                - "query2doc": Generate pseudo-document and expand query with it
                - "theme_queries": Generate 10 discourse-focused variations (for thematic analysis)
                - "stance_variation": Generate 3-5 pseudo-docs with different stances/perspectives
            **kwargs: Additional parameters for expansion
                - n_stances: Number of stances (for stance_variation, default 3)

        Returns:
            Self for chaining

        Examples:
            pipeline.expand_query("What is Mark Carney saying about climate?", strategy="multi_query")
            pipeline.expand_query("Pierre Poilievre", strategy="theme_queries")
            pipeline.expand_query("carbon tax", strategy="stance_variation", n_stances=3)
        """
        self.steps.append(("expand_query", {
            "query": query,
            "strategy": strategy,
            **kwargs
        }))
        return self

    def retrieve_segments_by_search(
        self,
        k: int = 500,
        threshold: float = 0.42,
        time_window_days: Optional[int] = None,
        must_contain: Optional[List[str]] = None,
        must_contain_any: Optional[List[str]] = None,
        **filters
    ) -> "AnalysisPipeline":
        """
        Add segment retrieval step using semantic search with optional keyword filtering.

        Requires expand_query() to have been called first to generate query embeddings.
        Uses expanded queries from context["expanded_queries"] and context["query_embeddings"].

        Supports hybrid search: semantic similarity + keyword filtering.

        Args:
            k: Maximum results per query embedding
            threshold: Minimum similarity threshold
            time_window_days: Time window filter (days)
            must_contain: Keywords that ALL must appear in text (AND logic)
                          Useful for entity queries like "Mark Carney"
            must_contain_any: Keywords where AT LEAST ONE must appear (OR logic)
                              Useful for variant names like ["Carney", "Mark Carney"]
            **filters: Additional filters for search
                - projects: List[str]
                - languages: List[str]
                - channels: List[str]

        Returns:
            Self for chaining

        Examples:
            # Pure semantic search
            pipeline.expand_query("climate policy").retrieve_segments_by_search(k=100)

            # Hybrid: semantic + entity filter
            pipeline.expand_query("What is being said about Mark Carney")
                    .retrieve_segments_by_search(must_contain=["Carney"], threshold=0.42)

            # Hybrid with name variants (OR logic)
            pipeline.expand_query("economy criticism")
                    .retrieve_segments_by_search(must_contain_any=["Carney", "Trudeau"])
        """
        self.steps.append(("retrieve_segments_by_search", {
            "k": k,
            "threshold": threshold,
            "time_window_days": time_window_days,
            "must_contain": must_contain,
            "must_contain_any": must_contain_any,
            **filters
        }))
        return self

    def retrieve_segments(self, **filters) -> "AnalysisPipeline":
        """
        Add segment retrieval step.

        Args:
            **filters: Filters for SegmentRetriever.fetch_by_filter()
                - projects: List[str]
                - languages: List[str]
                - channels: List[str]
                - speakers: List[str]
                - date_range: Tuple[datetime, datetime]
                - stitch_versions: List[str]
                - min_duration: float
                - max_duration: float

        Returns:
            Self for chaining
        """
        self.steps.append(("retrieve_segments", filters))
        return self

    def retrieve_all_segments(
        self,
        time_window_days: Optional[int] = 30,
        **filters
    ) -> "AnalysisPipeline":
        """
        Add step to retrieve ALL segments matching filters (no search/query required).

        For landing page workflows - retrieves entire corpus for theme discovery.

        Args:
            time_window_days: Time window in days (default 30)
            **filters: Filters for SegmentRetriever.fetch_by_filter()
                - projects: List[str]
                - languages: List[str]
                - channels: List[str]
                - must_be_embedded: bool (default True)
                - must_be_stitched: bool (default True)

        Returns:
            Self for chaining

        Context updates:
            - segments: List[Segment] - All retrieved segments
        """
        self.steps.append(("retrieve_all_segments", {
            "time_window_days": time_window_days,
            **filters
        }))
        return self

    def corpus_analysis(
        self,
        include_duration: bool = True,
        include_episode_count: bool = True
    ) -> "AnalysisPipeline":
        """
        Add corpus-level quantitative analysis step.

        Analyzes the full corpus (all segments from retrieve_all_segments).
        Returns descriptive stats: episodes, duration, channels, languages, etc.

        Args:
            include_duration: Calculate total duration in hours
            include_episode_count: Count unique episodes

        Returns:
            Self for chaining

        Context updates:
            - corpus_stats: Dict with corpus-level metrics
        """
        self.steps.append(("corpus_analysis", {
            "include_duration": include_duration,
            "include_episode_count": include_episode_count
        }))
        return self

    def analyze_themes_with_subthemes(
        self,
        quick_cluster_check: Optional[Dict[str, Any]] = None,
        quantitative_per_theme: bool = True,
        select_segments_per_subtheme: int = 8,
        select_unclustered: int = 6,
        generate_theme_names: bool = True,
        model: str = "grok-4-fast-non-reasoning-latest",
        max_concurrent: int = 8
    ) -> "AnalysisPipeline":
        """
        Analyze each theme with sub-themes in parallel.

        For each theme:
        - Run quick_cluster_check for sub-theme detection
        - Run quantitative_analysis on theme segments
        - Generate LLM theme name from representative segments
        - Select balanced segments (per sub-theme + unclustered)

        All themes processed concurrently using asyncio.

        Args:
            quick_cluster_check: Config for quick_cluster_check step
            quantitative_per_theme: Run quantitative analysis per theme
            select_segments_per_subtheme: Segments to select per sub-theme
            select_unclustered: Unclustered segments to include
            generate_theme_names: Generate LLM names for themes
            model: LLM model for theme naming
            max_concurrent: Max concurrent theme analysis tasks

        Returns:
            Self for chaining

        Context updates:
            - themes_analysis: Dict[theme_id, analysis_data]
            - themes: List[Theme] with updated names
        """
        self.steps.append(("analyze_themes_with_subthemes", {
            "quick_cluster_check": quick_cluster_check or {},
            "quantitative_per_theme": quantitative_per_theme,
            "select_segments_per_subtheme": select_segments_per_subtheme,
            "select_unclustered": select_unclustered,
            "generate_theme_names": generate_theme_names,
            "model": model,
            "max_concurrent": max_concurrent
        }))
        return self

    def extract_themes(
        self,
        method: str = "hdbscan",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 5,
        **kwargs
    ) -> "AnalysisPipeline":
        """
        Add theme extraction step.

        Args:
            method: Clustering method (hdbscan, kmeans, agglomerative)
            n_clusters: Number of clusters (for kmeans/agglomerative)
            min_cluster_size: Minimum cluster size (for hdbscan)
            **kwargs: Additional params for ThemeExtractor

        Returns:
            Self for chaining
        """
        self.steps.append(("extract_themes", {
            "method": method,
            "n_clusters": n_clusters,
            "min_cluster_size": min_cluster_size,
            **kwargs
        }))
        return self

    def quick_cluster_check(
        self,
        method: str = "hdbscan",
        min_cluster_size: int = 10,
        min_silhouette_score: float = 0.15,
        skip_if_few_segments: int = 30,
        **kwargs
    ) -> "AnalysisPipeline":
        """
        Add quick clustering validation step for sub-theme detection.

        Fast clustering check to detect if retrieved segments contain distinct sub-themes/cultures.
        Only creates themes if clusters are meaningfully distinct (validated by silhouette score).

        Use case: Detect echo chambers or divergent perspectives in simple RAG queries.

        Args:
            method: Clustering method (hdbscan recommended for adaptive clustering)
            min_cluster_size: Minimum segments per cluster
            min_silhouette_score: Minimum silhouette score for valid clusters (0.15 = weak but detectable structure)
            skip_if_few_segments: Skip clustering if fewer than this many segments (default 30)
            **kwargs: Additional params for ThemeExtractor

        Returns:
            Self for chaining

        Context updates:
            - has_subclusters: bool - Whether valid sub-clusters were detected
            - themes: List[Theme] - Detected themes (empty if no valid clusters)
            - cluster_validation: dict - Validation metrics (silhouette, davies_bouldin, etc.)

        Example:
            pipeline.retrieve_segments().quick_cluster_check(min_cluster_size=10).select_segments(n=20)
        """
        self.steps.append(("quick_cluster_check", {
            "method": method,
            "min_cluster_size": min_cluster_size,
            "min_silhouette_score": min_silhouette_score,
            "skip_if_few_segments": skip_if_few_segments,
            **kwargs
        }))
        return self

    def extract_subthemes(
        self,
        method: str = "hdbscan",
        n_subthemes: Optional[int] = None,
        min_cluster_size: int = 3,
        max_concurrent: int = 5,
        require_valid_clusters: bool = True,
        min_silhouette_score: float = 0.15,
        min_clusters_to_validate: int = 2,
        **kwargs
    ) -> "AnalysisPipeline":
        """
        Add sub-theme extraction step (hierarchical clustering with validation).

        NEW: Validates cluster quality before returning sub-themes.
        Themes without meaningful sub-clusters will skip sub-theme extraction.

        Args:
            method: Clustering method
            n_subthemes: Number of sub-themes per theme
            min_cluster_size: Minimum cluster size
            max_concurrent: Max parallel theme processing
            require_valid_clusters: If True, validate clusters before returning
            min_silhouette_score: Minimum silhouette score for valid clusters (0.15 typical)
            min_clusters_to_validate: Minimum number of clusters required (default 2)
            **kwargs: Additional params

        Returns:
            Self for chaining
        """
        self.steps.append(("extract_subthemes", {
            "method": method,
            "n_subthemes": n_subthemes,
            "min_cluster_size": min_cluster_size,
            "max_concurrent": max_concurrent,
            "require_valid_clusters": require_valid_clusters,
            "min_silhouette_score": min_silhouette_score,
            "min_clusters_to_validate": min_clusters_to_validate,
            **kwargs
        }))
        return self

    def select_segments(
        self,
        strategy: str = "diversity",
        n: int = 10,
        **kwargs
    ) -> "AnalysisPipeline":
        """
        Add segment selection step.

        Args:
            strategy: Selection strategy (diversity, centrality, recency, balanced, quality)
            n: Number of segments to select per theme
            **kwargs: Additional params for SegmentSelector

        Returns:
            Self for chaining
        """
        self.steps.append(("select_segments", {
            "strategy": strategy,
            "n": n,
            **kwargs
        }))
        return self

    def rerank_segments(
        self,
        best_per_episode: bool = True,
        max_per_channel: Optional[int] = None,
        similarity_weight: float = 0.4,
        popularity_weight: float = 0.2,
        recency_weight: float = 0.2,
        single_speaker_weight: float = 0.1,
        named_speaker_weight: float = 0.1,
        time_window_days: int = 30
    ) -> "AnalysisPipeline":
        """
        Add segment reranking step.

        Reranks retrieved segments based on multiple quality signals:
        - Semantic similarity (from search)
        - Channel popularity (importance_score)
        - Recency (publish_date freshness)
        - Single speaker (60%+ from one speaker)
        - Named speaker (speaker has identified name)

        Also applies diversity constraints:
        - Best match per episode (one segment per content_id)
        - Max segments per channel

        Call AFTER retrieve_segments_by_search and BEFORE select_segments.

        Args:
            best_per_episode: Keep only best segment per episode (default True)
            max_per_channel: Optional max segments per channel
            similarity_weight: Weight for semantic similarity (default 0.4)
            popularity_weight: Weight for channel popularity (default 0.2)
            recency_weight: Weight for recency (default 0.2)
            single_speaker_weight: Weight for single speaker bonus (default 0.1)
            named_speaker_weight: Weight for named speaker bonus (default 0.1)
            time_window_days: Time window for recency normalization (default 30)

        Returns:
            Self for chaining

        Example:
            pipeline.expand_query("climate policy")
                    .retrieve_segments_by_search(k=500, time_window_days=30)
                    .rerank_segments(best_per_episode=True, popularity_weight=0.3)
                    .select_segments(n=20)
        """
        self.steps.append(("rerank_segments", {
            "best_per_episode": best_per_episode,
            "max_per_channel": max_per_channel,
            "similarity_weight": similarity_weight,
            "popularity_weight": popularity_weight,
            "recency_weight": recency_weight,
            "single_speaker_weight": single_speaker_weight,
            "named_speaker_weight": named_speaker_weight,
            "time_window_days": time_window_days
        }))
        return self

    def generate_summaries(
        self,
        template: str,
        max_concurrent: int = 20,
        level: str = "theme",  # "theme" or "subtheme"
        **kwargs
    ) -> "AnalysisPipeline":
        """
        Add text generation step.

        Args:
            template: Template name for TextGenerator
            max_concurrent: Max concurrent LLM calls
            level: Generate for "theme" or "subtheme"
            **kwargs: Additional params (model, temperature, etc.)

        Returns:
            Self for chaining
        """
        self.steps.append(("generate_summaries", {
            "template": template,
            "max_concurrent": max_concurrent,
            "level": level,
            **kwargs
        }))
        return self

    def quantitative_analysis(
        self,
        include_baseline: bool = False,
        time_window_days: Optional[int] = None,
        **kwargs
    ) -> "AnalysisPipeline":
        """
        Add quantitative analysis step.

        Analyzes segment distributions to provide metrics about:
        - Number of relevant segments
        - Main channels/episodes focused on the issue
        - Temporal distribution
        - Discourse centrality (how central is this topic)

        Args:
            include_baseline: If True, retrieve baseline segments for comparison
            time_window_days: Time window for baseline (if include_baseline=True)
            **kwargs: Additional params for QuantitativeAnalyzer

        Returns:
            Self for chaining

        Example:
            pipeline.retrieve_segments(...).quantitative_analysis(include_baseline=True)
        """
        self.steps.append(("quantitative_analysis", {
            "include_baseline": include_baseline,
            "time_window_days": time_window_days,
            **kwargs
        }))
        return self

    def group_by(self, field: str) -> "AnalysisPipeline":
        """
        Add grouping step.

        Args:
            field: Field to group by (language, channel, date, etc.)

        Returns:
            Self for chaining
        """
        self.steps.append(("group_by", {"field": field}))
        return self

    def custom_step(self, name: str, func: Callable, **params) -> "AnalysisPipeline":
        """
        Add custom processing step.

        Args:
            name: Step name
            func: Async callable that takes (context, **params) and returns updated context
            **params: Parameters to pass to func

        Returns:
            Self for chaining
        """
        self.steps.append(("custom", {
            "name": name,
            "func": func,
            **params
        }))
        return self

    # ========================================================================
    # Execution Methods
    # ========================================================================

    async def execute(self) -> PipelineResult:
        """
        Execute pipeline in batch mode (no streaming).

        Returns:
            PipelineResult with final data and metadata
        """
        start_time = time.time()
        results = {}
        steps_completed = 0

        logger.info(f"Executing pipeline '{self.name}' with {len(self.steps)} steps")

        for step_idx, (step_type, params) in enumerate(self.steps):
            logger.info(f"Step {step_idx + 1}/{len(self.steps)}: {step_type}")

            try:
                results = await self._execute_step(step_type, params, results)
                steps_completed += 1
            except Exception as e:
                logger.error(f"Pipeline '{self.name}' failed at step {step_idx + 1} ({step_type}): {e}", exc_info=True)
                raise

        execution_time = int((time.time() - start_time) * 1000)
        logger.info(f"Pipeline '{self.name}' completed in {execution_time}ms")

        return PipelineResult(
            name=self.name,
            data=results,
            execution_time_ms=execution_time,
            steps_completed=steps_completed,
            total_steps=len(self.steps)
        )

    async def execute_stream(
        self,
        verbose: bool = False,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute pipeline with streaming progress updates and partial results.

        Args:
            verbose: If True, emit progress and partial events. If False, only emit result and complete events.
            initial_context: Optional dict to seed the pipeline context. Use this to skip steps
                            that have already been computed (e.g., cached expand_query results).

        Yields:
            Progress updates and partial results:
            - {"type": "progress", "step": "...", "progress": 0.5, "message": "..."} (verbose only)
            - {"type": "partial", "step": "...", "data": {...}} (verbose only)
            - {"type": "result", "step": "...", "data": {...}} (always)
            - {"type": "complete", "execution_time_ms": 123} (always)
            - {"type": "error", "step": "...", "error": "..."}
        """
        start_time = time.time()
        results = initial_context.copy() if initial_context else {}
        total_steps = len(self.steps)

        logger.info(f"Streaming pipeline '{self.name}' with {total_steps} steps (verbose={verbose})")

        for step_idx, (step_type, params) in enumerate(self.steps):
            # Progress: Starting step (verbose only)
            if verbose:
                yield {
                    "type": "progress",
                    "step": step_type,
                    "step_index": step_idx,
                    "total_steps": total_steps,
                    "progress": step_idx / total_steps,
                    "message": f"Starting {step_type}..."
                }

            try:
                # Execute step with streaming
                step_result_data = None
                async for update in self._execute_step_stream(step_type, params, results):
                    if update["type"] == "result":
                        # Step completed, update results and capture for emission
                        results = update["data"]
                        step_result_data = update["data"]
                    elif update["type"] == "partial" and verbose:
                        # Forward partial results with adjusted progress (verbose only)
                        yield {
                            "type": "partial",
                            "step": step_type,
                            "data": update["data"],
                            "progress": (step_idx + update.get("progress", 0.5)) / total_steps,
                            "message": update.get("message")
                        }

                # Emit step result event with the data produced by this step (always)
                if step_result_data is not None:
                    yield {
                        "type": "result",
                        "step": step_type,
                        "data": self._extract_step_results(step_type, step_result_data),
                        "step_index": step_idx,
                        "total_steps": total_steps
                    }

                # Progress: Completed step (verbose only)
                if verbose:
                    yield {
                        "type": "progress",
                        "step": step_type,
                        "step_index": step_idx,
                        "total_steps": total_steps,
                        "progress": (step_idx + 1) / total_steps,
                        "message": f"Completed {step_type}"
                    }

            except Exception as e:
                logger.error(f"Pipeline '{self.name}' failed at step {step_idx + 1} ({step_type}): {e}", exc_info=True)
                yield {
                    "type": "error",
                    "step": step_type,
                    "error": str(e),
                    "step_index": step_idx
                }
                raise

        # Emit final complete event (no data - just signals pipeline completion)
        execution_time = int((time.time() - start_time) * 1000)
        logger.info(f"Pipeline '{self.name}' streaming completed in {execution_time}ms")
        yield {
            "type": "complete",
            "execution_time_ms": execution_time,
            "steps_completed": len(self.steps),
            "total_steps": len(self.steps)
        }

    # ========================================================================
    # Internal Step Execution
    # ========================================================================

    async def _execute_step(
        self,
        step_type: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single step in batch mode."""

        if step_type == "expand_query":
            query = params.get("query")
            strategy = params.get("strategy", "multi_query")

            if not query:
                raise ValueError("query parameter required for expand_query step")

            logger.info(f"Expanding query: '{query[:50]}...' (strategy={strategy})")

            if strategy == "multi_query":
                # Generate 10 query variations (5 EN + 5 FR)
                result = self.llm_service.optimize_search_query(query)
                expanded_queries = result.get("query_variations", [query])
                keywords = result.get("keywords", [])
            elif strategy == "query2doc":
                # Generate pseudo-document
                pseudo_doc = self.llm_service.query2doc(query)
                expanded_query = f"{query} {pseudo_doc}" if pseudo_doc else query
                expanded_queries = [expanded_query]
                keywords = query.split()
            elif strategy == "theme_queries":
                # Generate discourse-focused query variations using TextGenerator
                from .text_generator import TextGenerator
                generator = TextGenerator(self.llm_service)

                try:
                    import json
                    result = await generator.generate_from_template(
                        template_name="theme_queries",
                        context={"theme_name": query},
                        model="grok-4-fast-non-reasoning-latest",
                        temperature=0.3,
                        max_tokens=800
                    )
                    parsed = json.loads(result)
                    expanded_queries = parsed.get("query_variations", [query])
                    keywords = [query]
                    logger.info(f"Generated {len(expanded_queries)} theme query variations")
                except Exception as e:
                    logger.error(f"Failed to generate theme queries: {e}, falling back to original query")
                    expanded_queries = [query]
                    keywords = query.split()
            elif strategy == "stance_variation":
                # Generate multiple pseudo-documents with different stances
                n_stances = params.get("n_stances", 3)
                stance_docs = self.llm_service.query2doc_stances(query, n_stances=n_stances)
                if stance_docs:
                    expanded_queries = [f"{query} {doc}" for doc in stance_docs]
                else:
                    # Fallback to original query if generation fails
                    logger.warning("Failed to generate stance variations, falling back to original query")
                    expanded_queries = [query]
                keywords = query.split()
            else:
                raise ValueError(f"Unknown expansion strategy: {strategy}")

            # Store expanded queries in context (will be embedded in retrieve_segments_by_search)
            context["original_query"] = query
            context["expanded_queries"] = expanded_queries
            context["keywords"] = keywords
            context["expansion_strategy"] = strategy
            logger.info(f"Expanded to {len(expanded_queries)} query variations")
            return context

        elif step_type == "retrieve_segments_by_search":
            # Get expanded queries from context
            expanded_queries = context.get("expanded_queries")
            if not expanded_queries:
                raise ValueError("expand_query step must be called before retrieve_segments_by_search")

            # Embed all query variations using EmbeddingService
            if not self.embedding_service:
                raise RuntimeError("EmbeddingService required for retrieve_segments_by_search step")

            try:
                embeddings = await self.embedding_service.encode_queries(expanded_queries)
            except RuntimeError as e:
                logger.error(f"Failed to generate embeddings for expanded queries: {e}")
                context["segments"] = []
                return context

            logger.info(f"Generated {len(embeddings)} query embeddings")

            # Get search service (need to initialize if not present)
            if self._search_service is None:
                from ..pgvector_search_service import PgVectorSearchService
                if not self.dashboard_id or not self.config:
                    raise ValueError("dashboard_id and config required for PgVectorSearchService initialization")
                self._search_service = PgVectorSearchService(self.dashboard_id, self.config)

            # Batch search with all embeddings
            # k is optional - if not provided, returns ALL segments above threshold
            k = params.get("k")  # None = no limit
            threshold = params.get("threshold", 0.4)
            time_window_days = params.get("time_window_days")

            # Extract filters to pass through
            search_filters = {}
            if "projects" in params:
                search_filters["projects"] = params["projects"]
            if "languages" in params:
                search_filters["languages"] = params["languages"]
            if "channels" in params:
                search_filters["channels"] = params["channels"]

            # Entity/keyword filters for hybrid search
            must_contain = params.get("must_contain")
            must_contain_any = params.get("must_contain_any")

            # Use unified batch search - returns all segments above threshold
            unique_segments = self._search_service.batch_search_unified(
                embeddings,
                time_window_days=time_window_days,
                k=k,  # None = no limit, returns all above threshold
                threshold=threshold,
                must_contain=must_contain,
                must_contain_any=must_contain_any,
                **search_filters
            )

            context["segments"] = unique_segments
            context["query_embeddings"] = embeddings
            # Store embeddings as serializable lists for caching
            context["query_embeddings_cached"] = [emb.tolist() for emb in embeddings]

            # Log with keyword info
            keyword_info = []
            if must_contain:
                keyword_info.append(f"must_contain={must_contain}")
            if must_contain_any:
                keyword_info.append(f"must_contain_any={must_contain_any}")
            keyword_str = f" [{', '.join(keyword_info)}]" if keyword_info else ""
            logger.info(f"Retrieved {len(unique_segments)} unique segments from unified batch search{keyword_str}")
            return context

        elif step_type == "retrieve_segments":
            retriever = self._get_retriever()
            segments = retriever.fetch_by_filter(**params)
            context["segments"] = segments
            logger.info(f"Retrieved {len(segments)} segments")
            return context

        elif step_type == "retrieve_all_segments":
            # Retrieve ALL segments for filters (landing page workflow)
            retriever = self._get_retriever()

            # Build date range from time_window_days
            time_window_days = params.get("time_window_days", 30)
            from datetime import datetime, timezone, timedelta
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=time_window_days)

            # Build filter params
            filter_params = {
                "date_range": (start_date, end_date),
                "must_be_stitched": params.get("must_be_stitched", True),
                "must_be_embedded": params.get("must_be_embedded", True)
            }

            # Add optional filters
            for key in ["projects", "languages", "channels"]:
                if key in params and params[key]:
                    filter_params[key] = params[key]

            segments = retriever.fetch_by_filter(**filter_params)
            context["segments"] = segments
            context["all_segments"] = segments  # Store for corpus analysis
            logger.info(f"Retrieved all {len(segments)} segments for time window {time_window_days}d")
            return context

        elif step_type == "rerank_segments":
            # Rerank segments based on multiple quality signals
            from .segment_reranker import SegmentReranker, RerankerWeights, DiversityConstraints

            segments = context.get("segments", [])
            if not segments:
                logger.warning("No segments available for reranking")
                return context

            # Build weights from params
            weights = RerankerWeights(
                similarity=params.get("similarity_weight", 0.4),
                popularity=params.get("popularity_weight", 0.2),
                recency=params.get("recency_weight", 0.2),
                single_speaker=params.get("single_speaker_weight", 0.1),
                named_speaker=params.get("named_speaker_weight", 0.1)
            )

            # Build diversity constraints
            diversity = DiversityConstraints(
                best_per_episode=params.get("best_per_episode", True),
                max_per_channel=params.get("max_per_channel")
            )

            time_window_days = params.get("time_window_days", 30)

            # Initialize reranker (uses embedded psycopg2 connection)
            reranker = SegmentReranker()
            reranked = reranker.rerank(
                segments,
                weights=weights,
                diversity=diversity,
                time_window_days=time_window_days
            )

            # Update context with reranked segments
            context["segments"] = reranked
            context["segments_before_rerank"] = len(segments)
            context["segments_after_rerank"] = len(reranked)
            logger.info(f"Reranked segments: {len(segments)} -> {len(reranked)} "
                       f"(best_per_episode={diversity.best_per_episode})")
            return context

        elif step_type == "corpus_analysis":
            # Corpus-level quantitative analysis
            analyzer = self._get_quantitative_analyzer()
            segments = context.get("all_segments") or context.get("segments", [])

            if not segments:
                logger.warning("No segments available for corpus analysis")
                context["corpus_stats"] = analyzer._empty_analysis()
                return context

            # Run quantitative analysis on full corpus
            corpus_stats = analyzer.analyze(
                segments=segments,
                baseline_segments=None  # No baseline for corpus analysis
            )

            context["corpus_stats"] = corpus_stats
            logger.info(
                f"Corpus analysis: {corpus_stats['total_segments']} segments, "
                f"{corpus_stats['episode_count']} episodes, "
                f"{corpus_stats.get('total_duration_hours', 'N/A')}h total"
            )
            return context

        elif step_type == "analyze_themes_with_subthemes":
            # Parallel analysis of themes with sub-themes
            themes = context.get("themes", [])

            if not themes:
                logger.warning("No themes available for sub-theme analysis")
                context["themes_analysis"] = {}
                return context

            # Process all themes in parallel
            max_concurrent = params.get("max_concurrent", 8)
            semaphore = asyncio.Semaphore(max_concurrent)

            async def analyze_one_theme(theme):
                """Analyze single theme with sub-themes."""
                async with semaphore:
                    theme_id = theme.theme_id
                    theme_segments = theme.segments

                    logger.info(f"Analyzing theme {theme_id} ({len(theme_segments)} segments)...")

                    analysis = {
                        "theme_id": theme_id,
                        "theme_name": theme.theme_name,  # Will be updated if generate_theme_names=True
                        "segment_count": len(theme_segments)
                    }

                    # 1. Sub-theme detection (quick_cluster_check)
                    extractor = self._get_extractor()
                    quick_check_config = params.get("quick_cluster_check", {})

                    min_cluster_size = quick_check_config.get("min_cluster_size", 10)
                    min_silhouette = quick_check_config.get("min_silhouette_score", 0.15)
                    skip_threshold = quick_check_config.get("skip_if_few_segments", 30)

                    subthemes = []
                    if len(theme_segments) >= skip_threshold:
                        try:
                            # Run clustering
                            subthemes = extractor.extract_by_clustering(
                                segments=theme_segments,
                                method="hdbscan",
                                min_cluster_size=min_cluster_size,
                                min_segments_per_theme=min_cluster_size
                            )

                            # Validate with silhouette score
                            if subthemes and len(subthemes) >= 2:
                                embeddings = np.vstack([self._get_segment_embedding(seg) for seg in theme_segments if self._get_segment_embedding(seg) is not None])

                                # Build cluster labels
                                cluster_labels = np.full(len(embeddings), -1)
                                valid_segments = [seg for seg in theme_segments if self._get_segment_embedding(seg) is not None]

                                for i, subtheme in enumerate(subthemes):
                                    subtheme_seg_ids = {self._get_segment_id(seg) for seg in subtheme.segments}
                                    for j, seg in enumerate(valid_segments):
                                        if self._get_segment_id(seg) in subtheme_seg_ids:
                                            cluster_labels[j] = i

                                is_valid, metrics = extractor._validate_clusters(embeddings, cluster_labels, min_silhouette)

                                if not is_valid:
                                    logger.info(f"Theme {theme_id}: Sub-clusters not valid (silhouette={metrics.get('silhouette_score', 0):.3f})")
                                    subthemes = []
                                else:
                                    logger.info(f"Theme {theme_id}: {len(subthemes)} valid sub-themes detected")
                        except Exception as e:
                            logger.error(f"Error detecting sub-themes for theme {theme_id}: {e}")
                            subthemes = []

                    analysis["subthemes"] = subthemes
                    analysis["has_subthemes"] = len(subthemes) > 0

                    # 2. Quantitative analysis (per theme)
                    if params.get("quantitative_per_theme", True):
                        analyzer = self._get_quantitative_analyzer()
                        theme_metrics = analyzer.analyze(
                            segments=theme_segments,
                            baseline_segments=None
                        )
                        analysis["quantitative_metrics"] = theme_metrics

                    # 3. Generate theme name with LLM
                    if params.get("generate_theme_names", True):
                        try:
                            generator = self._get_generator()
                            representative_segs = theme.representative_segments[:5]
                            seg_texts = [seg.text[:200] for seg in representative_segs]

                            # Generate theme name
                            prompt = f"Generate a concise 3-5 word theme name for these segments:\n\n" + "\n\n".join([f"- {text}" for text in seg_texts])

                            theme_name = await generator.generate(
                                prompt=prompt,
                                model=params.get("model", "grok-4-fast-non-reasoning-latest"),
                                temperature=0.3,
                                max_tokens=50
                            )

                            # Clean up theme name
                            theme_name = theme_name.strip().strip('"').strip()
                            theme.theme_name = theme_name
                            analysis["theme_name"] = theme_name
                            logger.info(f"Theme {theme_id}: Generated name '{theme_name}'")
                        except Exception as e:
                            logger.error(f"Error generating theme name for {theme_id}: {e}")

                    # 4. Segment selection
                    selector = self._get_selector()
                    if subthemes:
                        # Balanced selection across sub-themes
                        selected = selector.select_from_themes(
                            themes=subthemes,
                            n_per_theme=params.get("select_segments_per_subtheme", 8),
                            strategy="balanced"
                        )
                    else:
                        # Diversity selection from theme
                        selected = selector.select(
                            theme_segments,
                            n=params.get("select_segments_per_subtheme", 8),
                            strategy="diversity"
                        )

                    # Add unclustered segments if sub-themes exist
                    if subthemes and params.get("select_unclustered", 0) > 0:
                        clustered_ids = {id(seg) for subtheme in subthemes for seg in subtheme.segments}
                        unclustered = [seg for seg in theme_segments if id(seg) not in clustered_ids]
                        if unclustered:
                            unclustered_selected = selector.select(
                                unclustered,
                                n=params.get("select_unclustered", 6),
                                strategy="diversity"
                            )
                            selected.extend(unclustered_selected)

                    analysis["selected_segments"] = selected

                    return theme_id, analysis

            # Run all theme analyses in parallel
            results = await asyncio.gather(*[analyze_one_theme(theme) for theme in themes])

            # Store results
            themes_analysis = dict(results)
            context["themes_analysis"] = themes_analysis
            logger.info(f"Analyzed {len(themes)} themes in parallel")
            return context

        elif step_type == "extract_themes":
            extractor = self._get_extractor()
            segments = context.get("segments", [])

            if not segments:
                logger.warning("No segments available for theme extraction")
                context["themes"] = []
                return context

            themes = extractor.extract_by_clustering(
                segments=segments,
                method=params.get("method", "hdbscan"),
                n_clusters=params.get("n_clusters"),
                min_cluster_size=params.get("min_cluster_size", 5),
                max_themes=params.get("max_themes"),
                use_faiss=params.get("use_faiss", False),
                min_theme_percentage=params.get("min_theme_percentage")
            )

            context["themes"] = themes
            logger.info(f"Extracted {len(themes)} themes")
            return context

        elif step_type == "quick_cluster_check":
            extractor = self._get_extractor()
            segments = context.get("segments", [])

            skip_threshold = params.get("skip_if_few_segments", 30)
            min_silhouette = params.get("min_silhouette_score", 0.15)

            # Default: no clusters detected
            context["has_subclusters"] = False
            context["cluster_validation"] = {"skipped": True, "reason": "unknown"}

            if not segments:
                logger.info("No segments available for quick cluster check")
                context["themes"] = []
                context["cluster_validation"] = {"skipped": True, "reason": "no_segments"}
                return context

            if len(segments) < skip_threshold:
                logger.info(f"Skipping clustering: {len(segments)} segments < {skip_threshold} threshold")
                context["themes"] = []
                context["cluster_validation"] = {
                    "skipped": True,
                    "reason": "too_few_segments",
                    "segment_count": len(segments),
                    "threshold": skip_threshold
                }
                return context

            method = params.get("method", "hdbscan")

            # FAST PATH: Use PCA + KMeans (no UMAP, much faster)
            if method == "fast":
                logger.info(f"Running FAST cluster check on {len(segments)} segments (PCA + KMeans)...")
                from .fast_clustering import cluster_segments_fast
                from .theme_extractor import Theme

                # Fast clustering works directly on dict segments with embeddings
                groups, cluster_result = cluster_segments_fast(
                    segments,
                    min_cluster_size=params.get("min_cluster_size", 10),
                    min_silhouette=min_silhouette,
                    max_clusters=params.get("max_themes", 4)
                )

                if cluster_result.is_valid and cluster_result.n_clusters >= 2:
                    # Build Theme objects from groups
                    themes = []
                    for i, group in enumerate(groups):
                        if len(group) >= params.get("min_cluster_size", 10):
                            theme = Theme(
                                theme_id=f"cluster_{i}",
                                theme_name=f"Cluster {i+1}",
                                segments=group,
                                representative_segments=group[:5],
                                keywords=[],
                                embedding=None,
                                metadata={"cluster_size": len(group)}
                            )
                            themes.append(theme)

                    context["has_subclusters"] = True
                    context["themes"] = themes
                    context["cluster_validation"] = {
                        "skipped": False,
                        "silhouette_score": cluster_result.silhouette_score,
                        "num_clusters": cluster_result.n_clusters,
                        "cluster_sizes": cluster_result.cluster_sizes,
                        "method": "fast_pca_kmeans",
                        "elapsed_ms": cluster_result.metadata.get("elapsed_ms", 0)
                    }
                    logger.info(f"✓ Fast clustering: {len(themes)} clusters "
                               f"(sil={cluster_result.silhouette_score:.3f}) in "
                               f"{cluster_result.metadata.get('elapsed_ms', 0):.1f}ms")
                else:
                    context["has_subclusters"] = False
                    context["themes"] = []
                    context["cluster_validation"] = {
                        "skipped": False,
                        "silhouette_score": cluster_result.silhouette_score,
                        "reason": "no_valid_clusters",
                        "method": "fast_pca_kmeans"
                    }
                    logger.info(f"✗ Fast clustering: no valid clusters "
                               f"(sil={cluster_result.silhouette_score:.3f})")

                return context

            # STANDARD PATH: Fetch embeddings for dict segments, use UMAP + HDBSCAN
            if segments and isinstance(segments[0], dict):
                logger.info(f"Fetching embeddings for {len(segments)} dict segments")
                from ..rag.segment_retriever import SegmentRetriever
                retriever = SegmentRetriever(self.db_session)
                segment_ids = [seg.get('segment_id') for seg in segments if seg.get('segment_id')]
                segments = retriever.fetch_by_ids(segment_ids)
                logger.info(f"Fetched {len(segments)} segments with embeddings")

            # Run HDBSCAN clustering first
            logger.info(f"Running quick cluster check on {len(segments)} segments...")
            themes = extractor.extract_by_clustering(
                segments=segments,
                method=method,
                n_clusters=params.get("n_clusters"),
                min_cluster_size=params.get("min_cluster_size", 10),
                min_segments_per_theme=params.get("min_cluster_size", 10),
                max_themes=params.get("max_themes")
            )

            # Handle 1-cluster case: treat as core+fringe structure
            if len(themes) == 1:
                logger.info(f"HDBSCAN found 1 cluster - will use core (cluster) + fringe (noise) structure")
                # We already have unclustered_segments identified above
                # Just keep the single theme and proceed to selection which will handle core+fringe

            # K-means fallback if HDBSCAN found 0 clusters
            fallback_used = False
            if len(themes) == 0 and params.get("force_split", True):
                logger.info(f"HDBSCAN found 0 clusters - trying k-means fallback...")

                # Compute UMAP embeddings for fallback
                import umap
                embeddings = np.vstack([self._get_segment_embedding(seg) for seg in segments if self._get_segment_embedding(seg) is not None])
                valid_segments = [seg for seg in segments if self._get_segment_embedding(seg) is not None]

                umap_params = {
                    "n_neighbors": min(15, len(valid_segments) - 1),
                    "n_components": min(5, len(valid_segments) - 1),
                    "metric": "cosine",
                    "random_state": 42
                }
                reducer = umap.UMAP(**umap_params)
                reduced_embeddings = reducer.fit_transform(embeddings)

                # Try k-means with k=2,3,4
                fallback_themes, fallback_metrics = extractor.extract_with_forced_kmeans(
                    segments=valid_segments,
                    reduced_embeddings=reduced_embeddings,
                    k_values=[2, 3, 4],
                    min_silhouette=min_silhouette,
                    min_segments_per_theme=params.get("min_cluster_size", 10)
                )

                if fallback_themes:
                    themes = fallback_themes
                    fallback_used = True
                    logger.info(f"✓ K-means fallback successful: {len(themes)} clusters")
                else:
                    logger.info(f"✗ K-means fallback failed - treating as homogeneous")

            # Track which segments were clustered to identify noise/outliers
            clustered_segment_ids = set()
            for theme in themes:
                for seg in theme.segments:
                    clustered_segment_ids.add(self._get_segment_id(seg))

            # Collect unclustered segments (noise points that don't fit any cluster)
            unclustered_segments = []
            for seg in segments:
                if self._get_segment_id(seg) not in clustered_segment_ids:
                    unclustered_segments.append(seg)

            context["unclustered_segments"] = unclustered_segments
            context["fallback_used"] = fallback_used
            logger.info(f"Found {len(unclustered_segments)} unclustered segments (noise/outliers)")

            # Validate cluster quality
            if len(themes) == 1:
                # Single cluster: treat as core+fringe structure (no validation needed)
                context["has_subclusters"] = True  # Enable clustered selection mode
                context["themes"] = themes
                context["cluster_validation"] = {
                    "skipped": False,
                    "reason": "single_cluster_core_fringe",
                    "num_clusters": 1
                }
                logger.info(f"✓ 1 cluster detected - using core+fringe structure ({len(themes[0].segments)} core + {len(unclustered_segments)} fringe)")

            elif len(themes) >= 2:
                # Get embeddings and build cluster labels array
                embeddings = np.vstack([self._get_segment_embedding(seg) for seg in segments if self._get_segment_embedding(seg) is not None])
                cluster_labels = np.full(len(embeddings), -1, dtype=int)  # -1 = noise
                valid_segments = [seg for seg in segments if self._get_segment_embedding(seg) is not None]

                # Assign cluster labels based on theme membership
                for theme_idx, theme in enumerate(themes):
                    theme_seg_ids = {self._get_segment_id(seg) for seg in theme.segments}
                    for seg_idx, seg in enumerate(valid_segments):
                        seg_id = self._get_segment_id(seg)
                        if seg_id in theme_seg_ids:
                            cluster_labels[seg_idx] = theme_idx

                # Validate clusters
                is_valid, validation_metrics = extractor._validate_clusters(
                    embeddings,
                    cluster_labels,
                    min_silhouette_score=min_silhouette
                )

                # Add skipped flag to track whether clustering was attempted
                validation_metrics["skipped"] = False
                context["cluster_validation"] = validation_metrics

                if is_valid:
                    context["has_subclusters"] = True
                    context["themes"] = themes
                    logger.info(
                        f"✓ Detected {len(themes)} valid sub-clusters "
                        f"(silhouette={validation_metrics.get('silhouette_score', 0):.3f})"
                    )
                else:
                    context["has_subclusters"] = False
                    context["themes"] = []
                    logger.info(
                        f"✗ Clusters not validated "
                        f"(silhouette={validation_metrics.get('silhouette_score', 0):.3f} < {min_silhouette}). "
                        f"Treating as homogeneous."
                    )
            else:
                context["has_subclusters"] = False
                context["themes"] = []
                context["cluster_validation"] = {
                    "skipped": True,
                    "reason": "insufficient_clusters",
                    "num_clusters": len(themes)
                }
                logger.info(f"Insufficient clusters detected: {len(themes)} < 2. Treating as homogeneous.")

            return context

        elif step_type == "extract_subthemes":
            extractor = self._get_extractor()
            themes = context.get("themes", [])

            if not themes:
                logger.warning("No themes available for sub-theme extraction")
                context["subtheme_map"] = {}
                return context

            # Use batch extraction with parallelization and validation
            subtheme_map = await extractor.extract_subthemes_batch(
                themes=themes,
                method=params.get("method", "hdbscan"),
                n_subthemes_per_theme=params.get("n_subthemes"),
                min_cluster_size=params.get("min_cluster_size", 3),
                max_concurrent=params.get("max_concurrent", 5),
                require_valid_clusters=params.get("require_valid_clusters", True),
                min_silhouette_score=params.get("min_silhouette_score", 0.15),
                min_clusters_to_validate=params.get("min_clusters_to_validate", 2)
            )

            context["subtheme_map"] = subtheme_map
            total_subthemes = sum(len(subs) for subs in subtheme_map.values())
            themes_with_subthemes = sum(1 for subs in subtheme_map.values() if len(subs) > 0)
            logger.info(
                f"Extracted {total_subthemes} sub-themes from {themes_with_subthemes}/{len(themes)} themes "
                f"(validation={'enabled' if params.get('require_valid_clusters', True) else 'disabled'})"
            )
            return context

        elif step_type == "select_segments":
            selector = self._get_selector()
            strategy = params.get("strategy", "diversity")
            n = params.get("n", 10)

            # Check if we have clusters from quick_cluster_check
            has_subclusters = context.get("has_subclusters", False)

            # Case 1: Themes detected (from quick_cluster_check or extract_themes)
            if "themes" in context and context.get("themes"):
                all_selected = []
                num_clusters = len(context['themes'])

                # Adaptive per-cluster sizing
                # 1 cluster (core+fringe): 8-14 from core + 6 from fringe
                # 2 clusters → 8 each, 3 clusters → 8 each, 4 clusters → 6 each
                if num_clusters == 1:
                    n_per_cluster = min(14, n)  # More from core since it represents main discourse
                elif num_clusters == 2:
                    n_per_cluster = min(8, n)
                elif num_clusters == 3:
                    n_per_cluster = min(8, n)
                else:  # 4+ clusters
                    n_per_cluster = min(6, n)

                for theme in context["themes"]:
                    theme.selected = selector.select(
                        theme.segments,
                        n=n_per_cluster,
                        strategy=strategy
                    )
                    all_selected.extend(theme.selected)

                logger.info(f"Selected {n_per_cluster} segments per theme ({num_clusters} themes) using {strategy} strategy")

                # Also select from unclustered segments if available (noise/outliers)
                unclustered_segments = context.get("unclustered_segments", [])
                unclustered_selected = []
                if unclustered_segments and has_subclusters:
                    # Use explicit n_unclustered param if provided, otherwise default to 6
                    n_unclustered = params.get("n_unclustered", 6)
                    n_unclustered = min(n_unclustered, len(unclustered_segments))
                    unclustered_selected = selector.select(
                        unclustered_segments,
                        n=n_unclustered,
                        strategy=strategy
                    )
                    all_selected.extend(unclustered_selected)
                    logger.info(f"Selected {len(unclustered_selected)} unclustered segments (outliers/noise)")

                # Store unclustered selection separately for prompt formatting
                context["unclustered_selected"] = unclustered_selected

                # ALWAYS set selected_segments when we have themes (even if validation failed)
                # This ensures downstream generate_summaries has segments to work with
                context["selected_segments"] = all_selected
                context["selected_by_cluster"] = has_subclusters

                total_clustered = len(all_selected) - len(unclustered_selected)
                if has_subclusters:
                    logger.info(f"Aggregated {len(all_selected)} total segments ({total_clustered} clustered + {len(unclustered_selected)} unclustered)")
                else:
                    logger.info(f"Selected {len(all_selected)} segments from themes (clustering validation failed, no sub-themes)")

            # Case 2: Select from subthemes if available
            elif "subtheme_map" in context:
                for theme_id, subthemes in context["subtheme_map"].items():
                    for subtheme in subthemes:
                        subtheme.selected = selector.select(
                            subtheme.segments,
                            n=n,
                            strategy=strategy
                        )
                logger.info(f"Selected {n} segments per sub-theme using {strategy} strategy")

            # Case 3: No themes - select from flat segment list (standard simple RAG)
            else:
                segments = context.get("segments", [])

                # If clustering was attempted but failed validation, increase selection
                # to compensate for lack of cluster-based diversity sampling
                if context.get("cluster_validation", {}).get("skipped") == False:
                    # Clustering was attempted but didn't pass validation
                    # Use a higher baseline: n + n_unclustered for better coverage
                    n_unclustered = params.get("n_unclustered", 6)
                    n_total = n + n_unclustered
                    logger.info(f"Clustering validation failed - increasing selection to {n_total} segments for better coverage")
                else:
                    n_total = n

                if segments:
                    selected = selector.select(segments, n=n_total, strategy=strategy)
                    context["selected_segments"] = selected
                    context["selected_by_cluster"] = False
                    logger.info(f"Selected {len(selected)} segments from {len(segments)} using {strategy} strategy (no clustering)")

            return context

        elif step_type == "generate_summaries":
            generator = self._get_generator()
            template = params.get("template")
            max_concurrent = params.get("max_concurrent", 20)
            level = params.get("level", "theme")

            # Prepare generation tasks (now returns tuple with segment IDs)
            tasks, segment_id_map = self._prepare_generation_tasks(context, template, level, params)

            if not tasks:
                logger.warning(f"No tasks to generate for level '{level}'")
                context.setdefault("summaries", {})[level] = []
                context.setdefault("segment_ids", {})[level] = {}
                return context

            # Check if two-pass generation is needed
            two_pass = context.get("_two_pass_generation", False)

            if two_pass:
                # TWO-PASS GENERATION
                themes = context.get("themes", [])
                query = context.get("_two_pass_query", "the topic")

                # Pass 1: Generate group summaries in parallel
                logger.info(f"Pass 1: Generating {len(tasks)} group summaries...")
                pass1_results = await generator.generate_batch(tasks, max_concurrent=max_concurrent)

                # Pass 2: Generate overall synthesis
                logger.info("Pass 2: Generating overall synthesis...")
                group_summaries_with_sources = self._format_group_summaries_with_sources(
                    group_summaries=pass1_results,
                    themes=themes
                )

                synthesis_task = {
                    "template_name": "rag_synthesis",
                    "context": {
                        "theme_name": query,
                        "group_summaries_with_sources": group_summaries_with_sources
                    },
                    "model": params.get("model", "grok-2-1212"),
                    "temperature": params.get("temperature"),
                    "max_tokens": params.get("max_tokens", 800),
                    "metadata": {
                        "task_id": "overall_synthesis",
                        "type": "overall_synthesis"
                    }
                }

                synthesis_result = await generator.generate_from_template(
                    template_name="rag_synthesis",
                    context=synthesis_task["context"],
                    model=synthesis_task["model"],
                    temperature=synthesis_task.get("temperature"),
                    max_tokens=synthesis_task.get("max_tokens")
                )

                # Collect all segment IDs from all groups
                all_segment_ids = []
                for task_id, ids in segment_id_map.items():
                    all_segment_ids.extend(ids)
                segment_id_map["overall_synthesis"] = all_segment_ids

                # Store results in structured format
                if "summaries" not in context:
                    context["summaries"] = {}

                context["summaries"][level] = {
                    "overall_summary": synthesis_result,
                    "group_summaries": [
                        {
                            "group_index": i + 1,
                            "summary": summary,
                            "segment_ids": segment_id_map.get(f"group_{i+1}", [])
                        }
                        for i, summary in enumerate(pass1_results)
                    ],
                    "has_groups": True,
                    "num_groups": len(pass1_results)
                }

                # Clean up temporary context flags
                context.pop("_two_pass_generation", None)
                context.pop("_two_pass_query", None)

                logger.info(f"Two-pass generation complete: {len(pass1_results)} group summaries + 1 synthesis")

            else:
                # SINGLE-PASS GENERATION (no clusters)
                results = await generator.generate_batch(tasks, max_concurrent=max_concurrent)

                # Store results
                if "summaries" not in context:
                    context["summaries"] = {}

                # For single-pass, wrap in consistent structure
                context["summaries"][level] = {
                    "overall_summary": results[0] if results else None,
                    "group_summaries": [],
                    "has_groups": False,
                    "num_groups": 0
                }

                logger.info(f"Single-pass generation complete: 1 summary")

            # Store segment ID mapping
            if "segment_ids" not in context:
                context["segment_ids"] = {}
            context["segment_ids"][level] = segment_id_map

            return context

        elif step_type == "quantitative_analysis":
            analyzer = self._get_quantitative_analyzer()
            segments = context.get("segments", [])

            if not segments:
                logger.warning("No segments available for quantitative analysis")
                context["quantitative_metrics"] = analyzer._empty_analysis()
                return context

            # Get baseline segments if requested
            baseline_segments = None
            if params.get("include_baseline", False):
                retriever = self._get_retriever()
                time_window_days = params.get("time_window_days", 7)

                # Calculate date range
                from datetime import datetime, timezone, timedelta
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=time_window_days)

                # Retrieve baseline with same filters as original query (if available in context)
                baseline_filters = {
                    "date_range": (start_date, end_date),
                    "must_be_stitched": True,
                    "must_be_embedded": True
                }

                # Copy project/language/channel filters from original retrieval if available
                for filter_key in ["projects", "languages", "channels"]:
                    if filter_key in context:
                        baseline_filters[filter_key] = context[filter_key]

                # Use SQL aggregation instead of loading all segments (100x faster!)
                baseline_segments = retriever.get_baseline_stats(**baseline_filters)
                logger.info(
                    f"Retrieved baseline stats: {baseline_segments['total_segments']} segments, "
                    f"{baseline_segments['unique_videos']} videos, {baseline_segments['unique_channels']} channels"
                )

            # Perform analysis
            metrics = analyzer.analyze(
                segments=segments,
                baseline_segments=baseline_segments,
                time_window_days=params.get("time_window_days")
            )

            context["quantitative_metrics"] = metrics

            if metrics.get('discourse_centrality'):
                logger.info(
                    f"Quantitative analysis complete: {metrics['total_segments']} segments, "
                    f"centrality={metrics['discourse_centrality']['score']:.2f}"
                )
            else:
                logger.info(
                    f"Quantitative analysis complete: {metrics['total_segments']} segments "
                    f"(no centrality - baseline not provided)"
                )

            return context

        elif step_type == "group_by":
            field = params.get("field")
            segments = context.get("segments", [])

            from collections import defaultdict
            groups = defaultdict(list)
            for seg in segments:
                key = getattr(seg, field, "unknown")
                groups[key].append(seg)

            context["groups"] = dict(groups)
            logger.info(f"Grouped {len(segments)} segments by '{field}' into {len(groups)} groups")
            return context

        elif step_type == "custom":
            func = params.pop("func")
            name = params.get("name", "custom")
            logger.info(f"Executing custom step: {name}")
            return await func(context, **params)

        else:
            raise ValueError(f"Unknown step type: {step_type}")

    async def _execute_step_stream(
        self,
        step_type: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single step with streaming partial results."""

        if step_type == "expand_query":
            # Stream query expansion progress
            query = params.get("query")
            strategy = params.get("strategy", "multi_query")

            if not query:
                raise ValueError("query parameter required for expand_query step")

            logger.info(f"Expanding query: '{query[:50]}...' (strategy={strategy})")

            # Yield initial progress
            yield {
                "type": "partial",
                "data": {
                    "original_query": query,
                    "strategy": strategy
                },
                "progress": 0.1,
                "message": f"Expanding query using {strategy} strategy"
            }

            # Execute expansion (LLM call)
            if strategy == "multi_query":
                result = self.llm_service.optimize_search_query(query)
                expanded_queries = result.get("query_variations", [query])
                keywords = result.get("keywords", [])
            elif strategy == "query2doc":
                pseudo_doc = self.llm_service.query2doc(query)
                expanded_query = f"{query} {pseudo_doc}" if pseudo_doc else query
                expanded_queries = [expanded_query]
                keywords = query.split()
            elif strategy == "theme_queries":
                from .text_generator import TextGenerator
                generator = TextGenerator(self.llm_service)
                try:
                    import json
                    result = await generator.generate_from_template(
                        template_name="theme_queries",
                        context={"theme_name": query},
                        model="grok-4-fast-non-reasoning-latest",
                        temperature=0.3,
                        max_tokens=800
                    )
                    parsed = json.loads(result)
                    expanded_queries = parsed.get("query_variations", [query])
                    keywords = [query]
                    logger.info(f"Generated {len(expanded_queries)} theme query variations")
                except Exception as e:
                    logger.error(f"Failed to generate theme queries: {e}, falling back to original query")
                    expanded_queries = [query]
                    keywords = query.split()
            elif strategy == "stance_variation":
                n_stances = params.get("n_stances", 3)
                stance_docs = self.llm_service.query2doc_stances(query, n_stances=n_stances)
                if stance_docs:
                    expanded_queries = [f"{query} {doc}" for doc in stance_docs]
                else:
                    logger.warning("Failed to generate stance variations, falling back to original query")
                    expanded_queries = [query]
                keywords = query.split()
            else:
                raise ValueError(f"Unknown expansion strategy: {strategy}")

            # Yield completion with expanded queries
            yield {
                "type": "partial",
                "data": {
                    "expanded_query_count": len(expanded_queries),
                    "keywords": keywords[:10]  # Top 10 keywords only
                },
                "progress": 1.0,
                "message": f"Expanded to {len(expanded_queries)} query variations"
            }

            # Store in context
            context["original_query"] = query
            context["expanded_queries"] = expanded_queries
            context["keywords"] = keywords
            context["expansion_strategy"] = strategy
            logger.info(f"Expanded to {len(expanded_queries)} query variations")

            yield {"type": "result", "data": context}

        elif step_type == "retrieve_segments_by_search":
            # Stream semantic search retrieval progress
            # Get expanded queries from context (or fall back to original query for quick_summary)
            expanded_queries = context.get("expanded_queries")
            if not expanded_queries:
                # Support direct query embedding without expand_query step (quick_summary workflow)
                original_query = context.get("original_query")
                if original_query:
                    expanded_queries = [original_query]
                    logger.info(f"Using original query directly (no expansion): '{original_query[:60]}...'")
                else:
                    raise ValueError("expand_query step must be called before retrieve_segments_by_search, or original_query must be in context")

            # Check if embeddings are already in context (from cache)
            embeddings = context.get("query_embeddings")
            if embeddings:
                logger.info(f"Using {len(embeddings)} cached embeddings for search")
            else:
                # Embed all query variations using EmbeddingService
                if not self.embedding_service:
                    raise RuntimeError("EmbeddingService required for retrieve_segments_by_search step")

                try:
                    embeddings = await self.embedding_service.encode_queries(expanded_queries)
                    logger.info(f"Generated {len(embeddings)} embeddings for search")
                except RuntimeError as e:
                    logger.error(f"Failed to generate embeddings for expanded queries: {e}")
                    context["segments"] = []
                    yield {"type": "result", "data": context}
                    return

            # Get search service (need to initialize if not present)
            if self._search_service is None:
                from ..pgvector_search_service import PgVectorSearchService
                if not self.dashboard_id or not self.config:
                    raise ValueError("dashboard_id and config required for PgVectorSearchService initialization")
                self._search_service = PgVectorSearchService(self.dashboard_id, self.config)

            # Search with all embeddings
            k = params.get("k", 100)
            threshold = params.get("threshold", 0.4)
            time_window_days = params.get("time_window_days")

            # Extract other filters
            filters = {k: v for k, v in params.items() if k not in ["k", "threshold", "time_window_days"]}

            all_segments = []
            seen_ids = set()

            for i, (query, embedding) in enumerate(zip(expanded_queries, embeddings)):
                results = self._search_service.search(
                    query_embedding=embedding,
                    k=k,
                    threshold=threshold,
                    time_window_days=time_window_days,
                    **filters
                )

                # Deduplicate
                for seg in results:
                    seg_id = (seg.get('content_id'), seg.get('start_time'))
                    if seg_id not in seen_ids:
                        seen_ids.add(seg_id)
                        all_segments.append(seg)

                # Yield progress
                yield {
                    "type": "partial",
                    "data": {
                        "query_num": i + 1,
                        "total_queries": len(expanded_queries),
                        "segment_count": len(all_segments)
                    },
                    "progress": (i + 1) / len(expanded_queries),
                    "message": f"Searched query {i+1}/{len(expanded_queries)}: {len(all_segments)} segments found"
                }

            context["segments"] = all_segments
            context["query_embeddings"] = embeddings
            # Store embeddings as serializable lists for caching
            context["query_embeddings_cached"] = [emb.tolist() for emb in embeddings]
            logger.info(f"Retrieved {len(all_segments)} unique segments via semantic search")

            # Final yield with segment count
            yield {
                "type": "partial",
                "data": {
                    "segment_count": len(all_segments)
                },
                "progress": 1.0,
                "message": f"Retrieved {len(all_segments)} segments"
            }

            yield {"type": "result", "data": context}

        elif step_type == "retrieve_segments":
            # Stream segment retrieval progress
            retriever = self._get_retriever()
            segments = retriever.fetch_by_filter(**params)

            # Yield progress event immediately after retrieval
            yield {
                "type": "partial",
                "data": {
                    "segment_count": len(segments),
                    "filters": {k: v for k, v in params.items() if k in ["projects", "languages", "channels", "date_range"]}
                },
                "progress": 1.0,
                "message": f"Retrieved {len(segments)} segments"
            }

            context["segments"] = segments
            logger.info(f"Retrieved {len(segments)} segments")
            yield {"type": "result", "data": context}

        elif step_type == "rerank_segments":
            # Stream reranking progress
            from .segment_reranker import SegmentReranker, RerankerWeights, DiversityConstraints

            segments = context.get("segments", [])
            if not segments:
                logger.warning("No segments available for reranking")
                yield {"type": "result", "data": context}
                return

            yield {
                "type": "partial",
                "data": {"input_segments": len(segments)},
                "progress": 0.3,
                "message": f"Reranking {len(segments)} segments..."
            }

            # Build weights from params
            weights = RerankerWeights(
                similarity=params.get("similarity_weight", 0.4),
                popularity=params.get("popularity_weight", 0.2),
                recency=params.get("recency_weight", 0.2),
                single_speaker=params.get("single_speaker_weight", 0.1),
                named_speaker=params.get("named_speaker_weight", 0.1)
            )

            # Build diversity constraints
            diversity = DiversityConstraints(
                best_per_episode=params.get("best_per_episode", True),
                max_per_channel=params.get("max_per_channel")
            )

            time_window_days = params.get("time_window_days", 30)

            # Initialize reranker (uses embedded psycopg2 connection)
            reranker = SegmentReranker()
            reranked = reranker.rerank(
                segments,
                weights=weights,
                diversity=diversity,
                time_window_days=time_window_days
            )

            # Update context with reranked segments
            context["segments"] = reranked
            context["segments_before_rerank"] = len(segments)
            context["segments_after_rerank"] = len(reranked)

            yield {
                "type": "partial",
                "data": {
                    "input_segments": len(segments),
                    "output_segments": len(reranked),
                    "best_per_episode": diversity.best_per_episode
                },
                "progress": 1.0,
                "message": f"Reranked: {len(segments)} -> {len(reranked)} segments"
            }

            logger.info(f"Reranked segments: {len(segments)} -> {len(reranked)}")
            yield {"type": "result", "data": context}

        elif step_type == "select_segments":
            # Stream segment selection progress
            selector = self._get_selector()
            strategy = params.get("strategy", "diversity")
            n = params.get("n", 10)
            has_subclusters = context.get("has_subclusters", False)

            segments = context.get("segments", [])

            # Case 1: Themes detected (from quick_cluster_check or extract_themes)
            if "themes" in context and context.get("themes"):
                themes = context["themes"]
                all_selected = []
                num_clusters = len(themes)

                # Adaptive per-cluster sizing: 2 clusters → 8 each, 3 clusters → 8 each, 4 clusters → 6 each
                if num_clusters == 2:
                    n_per_cluster = min(8, n)
                elif num_clusters == 3:
                    n_per_cluster = min(8, n)
                else:  # 4+ clusters
                    n_per_cluster = min(6, n)

                # Yield progress
                yield {
                    "type": "partial",
                    "data": {
                        "strategy": strategy,
                        "n_per_cluster": n_per_cluster,
                        "num_clusters": num_clusters,
                        "has_subclusters": has_subclusters
                    },
                    "progress": 0.3,
                    "message": f"Selecting {n_per_cluster} segments from each of {num_clusters} clusters using {strategy} strategy"
                }

                for theme in themes:
                    theme.selected = selector.select(
                        theme.segments,
                        n=n_per_cluster,
                        strategy=strategy
                    )
                    all_selected.extend(theme.selected)

                logger.info(f"Selected {n_per_cluster} segments per theme ({num_clusters} themes) using {strategy} strategy")

                # Also select from unclustered segments if available (noise/outliers)
                unclustered_segments = context.get("unclustered_segments", [])
                unclustered_selected = []
                if unclustered_segments and has_subclusters:
                    # Use explicit n_unclustered param if provided, otherwise default to 6
                    n_unclustered = params.get("n_unclustered", 6)
                    n_unclustered = min(n_unclustered, len(unclustered_segments))
                    unclustered_selected = selector.select(
                        unclustered_segments,
                        n=n_unclustered,
                        strategy=strategy
                    )
                    all_selected.extend(unclustered_selected)
                    logger.info(f"Selected {len(unclustered_selected)} unclustered segments (outliers/noise)")

                # Store unclustered selection separately for prompt formatting
                context["unclustered_selected"] = unclustered_selected

                # Aggregate for simple RAG with clustering
                if has_subclusters:
                    context["selected_segments"] = all_selected
                    context["selected_by_cluster"] = True
                    total_clustered = len(all_selected) - len(unclustered_selected)
                    logger.info(f"Aggregated {len(all_selected)} total segments ({total_clustered} clustered + {len(unclustered_selected)} unclustered)")

                # Yield completion
                yield {
                    "type": "partial",
                    "data": {
                        "selected_count": len(all_selected),
                        "clusters_processed": num_clusters,
                        "unclustered_count": len(unclustered_selected),
                        "per_cluster_count": n_per_cluster
                    },
                    "progress": 1.0,
                    "message": f"Selected {len(all_selected)} segments ({num_clusters}×{n_per_cluster} clustered + {len(unclustered_selected)} unclustered)"
                }

            # Case 2: Select from subthemes if available
            elif "subtheme_map" in context:
                for theme_id, subthemes in context["subtheme_map"].items():
                    for subtheme in subthemes:
                        subtheme.selected = selector.select(
                            subtheme.segments,
                            n=n,
                            strategy=strategy
                        )
                logger.info(f"Selected {n} segments per sub-theme using {strategy} strategy")

            # Case 3: No themes - select from flat segment list (standard simple RAG)
            elif segments:
                # Yield progress before selection
                yield {
                    "type": "partial",
                    "data": {
                        "strategy": strategy,
                        "n": n,
                        "total_segments": len(segments)
                    },
                    "progress": 0.5,
                    "message": f"Selecting {n} segments using {strategy} strategy"
                }

                # Fetch embeddings for dict segments if needed
                if segments and isinstance(segments[0], dict):
                    from ..rag.segment_retriever import SegmentRetriever
                    retriever = SegmentRetriever(self.db_session)

                    # Get segment IDs
                    segment_ids = [seg.get('segment_id') for seg in segments if seg.get('segment_id')]

                    # Fetch full segments with embeddings
                    segments = retriever.fetch_by_ids(segment_ids)
                    context["segments"] = segments
                    logger.info(f"Fetched embeddings for {len(segments)} segments")

                # Select segments
                selected = selector.select(segments, n=n, strategy=strategy)
                context["selected_segments"] = selected
                context["selected_by_cluster"] = False
                logger.info(f"Selected {len(selected)} segments from {len(segments)} using {strategy} strategy (no clustering)")

                # Yield completion
                yield {
                    "type": "partial",
                    "data": {
                        "selected_count": len(selected)
                    },
                    "progress": 1.0,
                    "message": f"Selected {len(selected)} segments"
                }

            yield {"type": "result", "data": context}

        elif step_type == "quantitative_analysis":
            # Stream quantitative analysis metrics as they compute
            analyzer = self._get_quantitative_analyzer()
            segments = context.get("segments", [])

            if not segments:
                logger.warning("No segments available for quantitative analysis")
                context["quantitative_metrics"] = analyzer._empty_analysis()
                yield {"type": "result", "data": context}
                return

            # Get baseline segments if requested
            baseline_segments = None
            if params.get("include_baseline", False):
                retriever = self._get_retriever()
                time_window_days = params.get("time_window_days", 7)

                # Calculate date range
                from datetime import datetime, timezone, timedelta
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=time_window_days)

                # Retrieve baseline with same filters as original query
                baseline_filters = {
                    "date_range": (start_date, end_date),
                    "must_be_stitched": True,
                    "must_be_embedded": True
                }

                # Copy project/language/channel filters from context if available
                for filter_key in ["projects", "languages", "channels"]:
                    if filter_key in context:
                        baseline_filters[filter_key] = context[filter_key]

                # If no filters in context, get from dashboard config
                if "projects" not in baseline_filters and self.config:
                    if hasattr(self.config, 'project'):
                        baseline_filters["projects"] = [self.config.project]
                if "languages" not in baseline_filters and self.config and hasattr(self.config, 'languages'):
                    baseline_filters["languages"] = self.config.languages

                # Use SQL aggregation instead of loading all segments (100x faster!)
                baseline_segments = retriever.get_baseline_stats(**baseline_filters)

                # Yield partial result after baseline retrieval
                yield {
                    "type": "partial",
                    "data": {
                        "baseline_segment_count": baseline_segments['total_segments'],
                        "analysis_type": "with_baseline"
                    },
                    "progress": 0.3,
                    "message": f"Retrieved {baseline_segments['total_segments']} baseline segments for comparison"
                }

            # Perform analysis
            metrics = analyzer.analyze(
                segments=segments,
                baseline_segments=baseline_segments,
                time_window_days=params.get("time_window_days")
            )

            # Yield partial result with computed metrics
            # Include ALL metrics so frontend can use them immediately
            yield {
                "type": "partial",
                "data": {
                    "total_segments": metrics.get("total_segments", 0),
                    "unique_videos": metrics.get("unique_videos", 0),
                    "unique_channels": metrics.get("unique_channels", 0),
                    "channel_distribution": metrics.get("channel_distribution", []),
                    "video_distribution": metrics.get("video_distribution", []),
                    "temporal_distribution": metrics.get("temporal_distribution", {}),
                    "concentration_metrics": metrics.get("concentration_metrics", {}),
                    "discourse_centrality": metrics.get("discourse_centrality")
                },
                "progress": 1.0,
                "message": f"Analyzed {metrics.get('total_segments', 0)} segments"
            }

            context["quantitative_metrics"] = metrics

            if metrics.get('discourse_centrality'):
                logger.info(
                    f"Quantitative analysis complete: {metrics['total_segments']} segments, "
                    f"centrality={metrics['discourse_centrality']['score']:.2f}"
                )
            else:
                logger.info(
                    f"Quantitative analysis complete: {metrics['total_segments']} segments "
                    f"(no centrality - baseline not provided)"
                )

            yield {"type": "result", "data": context}

        elif step_type == "extract_subthemes":
            # Stream subtheme extraction (one theme at a time)
            extractor = self._get_extractor()
            themes = context.get("themes", [])

            if not themes:
                logger.warning("No themes available for sub-theme extraction")
                context["subtheme_map"] = {}
                yield {"type": "result", "data": context}
                return

            subtheme_map = {}
            for i, theme in enumerate(themes):
                subthemes = extractor.extract_subthemes(
                    theme=theme,
                    method=params.get("method", "hdbscan"),
                    n_subthemes=params.get("n_subthemes"),
                    min_cluster_size=params.get("min_cluster_size", 3),
                    require_valid_clusters=params.get("require_valid_clusters", True),
                    min_silhouette_score=params.get("min_silhouette_score", 0.15),
                    min_clusters_to_validate=params.get("min_clusters_to_validate", 2)
                )
                subtheme_map[theme.theme_id] = subthemes

                # Extract validation metadata if available
                validation_info = None
                if subthemes and "cluster_validation" in subthemes[0].metadata:
                    validation_info = subthemes[0].metadata["cluster_validation"]

                # Yield partial result for this theme
                yield {
                    "type": "partial",
                    "data": {
                        "parent_theme_id": theme.theme_id,
                        "parent_theme_name": theme.theme_name,
                        "subtheme_count": len(subthemes),
                        "validation": validation_info,
                        "subthemes": [
                            {
                                "theme_id": st.theme_id,
                                "theme_name": st.theme_name,
                                "segment_count": len(st.segments)
                            }
                            for st in subthemes
                        ]
                    },
                    "progress": (i + 1) / len(themes),
                    "message": f"Extracted sub-themes for theme {i+1}/{len(themes)}: {theme.theme_name}"
                }

            context["subtheme_map"] = subtheme_map
            yield {"type": "result", "data": context}

        elif step_type == "generate_summaries":
            # Stream summary generation (yield each summary as it completes)
            generator = self._get_generator()
            template = params.get("template")
            max_concurrent = params.get("max_concurrent", 20)
            level = params.get("level", "theme")

            # Prepare generation tasks (now returns tuple with segment IDs)
            tasks, segment_id_map = self._prepare_generation_tasks(context, template, level, params)

            if not tasks:
                logger.warning(f"No tasks to generate for level '{level}'")
                context.setdefault("summaries", {})[level] = []
                context.setdefault("segment_ids", {})[level] = {}
                yield {"type": "result", "data": context}
                return

            # Check if two-pass generation is needed
            two_pass = context.get("_two_pass_generation", False)

            if two_pass:
                # TWO-PASS GENERATION (streaming)
                themes = context.get("themes", [])
                query = context.get("_two_pass_query", "the topic")

                # Pass 1: Stream group summaries
                pass1_results = [None] * len(tasks)
                async for update in generator.generate_batch_stream(tasks, max_concurrent=max_concurrent):
                    idx = update["index"]
                    pass1_results[idx] = update.get("result")

                    # Yield partial SSE event for Pass 1
                    yield {
                        "type": "partial",
                        "data": {
                            "phase": "group_summaries",
                            "index": idx,
                            "summary": update.get("result"),
                            "completed": update["completed"],
                            "total": update["total"],
                            "task_metadata": tasks[idx].get("metadata", {}),
                            "error": update.get("error")
                        },
                        "progress": update["progress"] * 0.8,  # Reserve 20% for synthesis
                        "message": f"Pass 1: Generated group summary {update['completed']}/{update['total']}"
                    }

                # Pass 2: Generate overall synthesis
                yield {
                    "type": "partial",
                    "data": {"phase": "synthesis", "message": "Generating overall synthesis..."},
                    "progress": 0.85,
                    "message": "Pass 2: Generating overall synthesis..."
                }

                group_summaries_with_sources = self._format_group_summaries_with_sources(
                    group_summaries=pass1_results,
                    themes=themes
                )

                synthesis_result = await generator.generate_from_template(
                    template_name="rag_synthesis",
                    context={
                        "theme_name": query,
                        "group_summaries_with_sources": group_summaries_with_sources
                    },
                    model=params.get("model", "grok-2-1212"),
                    temperature=params.get("temperature"),
                    max_tokens=params.get("max_tokens", 800)
                )

                # Collect all segment IDs from all groups
                all_segment_ids = []
                for task_id, ids in segment_id_map.items():
                    all_segment_ids.extend(ids)
                segment_id_map["overall_synthesis"] = all_segment_ids

                # Store results in structured format
                if "summaries" not in context:
                    context["summaries"] = {}

                context["summaries"][level] = {
                    "overall_summary": synthesis_result,
                    "group_summaries": [
                        {
                            "group_index": i + 1,
                            "summary": summary,
                            "segment_ids": segment_id_map.get(f"group_{i+1}", [])
                        }
                        for i, summary in enumerate(pass1_results)
                    ],
                    "has_groups": True,
                    "num_groups": len(pass1_results)
                }

                # Clean up temporary context flags
                context.pop("_two_pass_generation", None)
                context.pop("_two_pass_query", None)

                logger.info(f"Two-pass streaming generation complete: {len(pass1_results)} group summaries + 1 synthesis")

            else:
                # SINGLE-PASS GENERATION (streaming)
                results = [None] * len(tasks)
                async for update in generator.generate_batch_stream(tasks, max_concurrent=max_concurrent):
                    idx = update["index"]
                    results[idx] = update.get("result")

                    # Yield partial SSE event immediately
                    yield {
                        "type": "partial",
                        "data": {
                            "phase": "single_pass",
                            "index": idx,
                            "summary": update.get("result"),
                            "completed": update["completed"],
                            "total": update["total"],
                            "task_metadata": tasks[idx].get("metadata", {}),
                            "error": update.get("error")
                        },
                        "progress": update["progress"],
                        "message": f"Generated summary {update['completed']}/{update['total']}"
                    }

                # Store results in consistent structure
                if "summaries" not in context:
                    context["summaries"] = {}

                context["summaries"][level] = {
                    "overall_summary": results[0] if results else None,
                    "group_summaries": [],
                    "has_groups": False,
                    "num_groups": 0
                }

                logger.info(f"Single-pass streaming generation complete: 1 summary")

            # Store segment ID mapping
            if "segment_ids" not in context:
                context["segment_ids"] = {}
            context["segment_ids"][level] = segment_id_map

            yield {"type": "result", "data": context}

        else:
            # For other steps, execute in batch and yield result
            result = await self._execute_step(step_type, params, context)
            yield {"type": "result", "data": result}

    def _prepare_generation_tasks(
        self,
        context: Dict[str, Any],
        template: str,
        level: str,
        params: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        """
        Prepare generation tasks from context.

        For simple_rag with clusters, uses TWO-PASS generation:
        - Pass 1: Generate group summaries (parallel, one per cluster)
        - Pass 2: Generate overall synthesis using group summaries + source transcripts

        Returns:
            Tuple of (tasks, segment_id_map)
            - tasks: List of generation task dicts
            - segment_id_map: Dict mapping task_id -> list of segment IDs used
        """
        tasks = []
        segment_id_map = {}  # Track segment IDs per task

        if level == "theme":
            themes = context.get("themes", [])

            # Simple RAG case: if we have selected_segments
            if "selected_segments" in context:
                selected_segments = context["selected_segments"]
                if selected_segments:
                    query = context.get("original_query", "the topic")

                    # Check if segments were selected from clusters
                    selected_by_cluster = context.get("selected_by_cluster", False)
                    has_subclusters = context.get("has_subclusters", False)
                    themes = context.get("themes", [])
                    num_clusters = len(themes)

                    # TWO-PASS GENERATION when clusters detected
                    if selected_by_cluster and has_subclusters and num_clusters >= 2:
                        # Mark context for two-pass generation
                        context["_two_pass_generation"] = True
                        context["_two_pass_query"] = query

                        # Pass 1: Create group summary tasks (one per cluster)
                        for i, theme in enumerate(themes):
                            theme_segments = getattr(theme, "selected", theme.segments[:10])
                            if not theme_segments:
                                continue

                            theme_segments_text = self._format_segments_for_prompt(theme_segments)
                            group_task_id = f"group_{i+1}"

                            tasks.append({
                                "template_name": "cluster_perspective_summary",
                                "context": {
                                    "parent_query": query,
                                    "segments_text": theme_segments_text
                                },
                                "model": params.get("model", "grok-2-1212"),
                                "temperature": params.get("temperature"),
                                "max_tokens": 300,
                                "metadata": {
                                    "task_id": group_task_id,
                                    "type": "group_summary",
                                    "group_index": i + 1,
                                    "theme_id": theme.theme_id
                                }
                            })

                            # Track segment IDs
                            segment_id_map[group_task_id] = [self._get_segment_id(seg) for seg in theme_segments]

                        logger.info(f"Pass 1: Prepared {num_clusters} group summary tasks for two-pass generation")

                    # SINGLE-PASS when no clusters (use human-friendly rag_answer)
                    else:
                        segments_text = self._format_segments_for_prompt(selected_segments)
                        task_id = "simple_rag_summary"

                        tasks.append({
                            "template_name": template,  # rag_answer
                            "context": {
                                "theme_name": query,
                                "segments_text": segments_text
                            },
                            "model": params.get("model", "grok-2-1212"),
                            "temperature": params.get("temperature"),
                            "max_tokens": params.get("max_tokens"),
                            "metadata": {
                                "task_id": task_id,
                                "type": "overall",
                                "query": query,
                                "has_subclusters": False,
                                "num_clusters": 0
                            }
                        })

                        # Track segment IDs
                        segment_id_map[task_id] = [self._get_segment_id(seg) for seg in selected_segments]

                        logger.info(f"Single-pass: Prepared summary task with {len(selected_segments)} segments (no clustering)")

            # Hierarchical mode: Process themes individually (when NO selected_segments)
            # This is for workflows that want per-theme summaries without an overall summary
            elif themes and "selected_segments" not in context:
                for theme in themes:
                    theme_selected = getattr(theme, "selected", theme.segments[:10])
                    segments_text = self._format_segments_for_prompt(theme_selected)

                    task_id = f"theme_{theme.theme_id}"
                    tasks.append({
                        "template_name": template,
                        "context": {
                            "theme_name": theme.theme_name,
                            "segments_text": segments_text,
                            "cluster_context": "",
                            "cluster_instruction": ""
                        },
                        "model": params.get("model", "grok-2-1212"),
                        "temperature": params.get("temperature"),
                        "max_tokens": params.get("max_tokens"),
                        "metadata": {
                            "theme_id": theme.theme_id,
                            "theme_name": theme.theme_name,
                            "task_id": task_id,
                            "type": "theme"
                        }
                    })

                    # Track segment IDs for this task
                    segment_id_map[task_id] = [self._get_segment_id(seg) for seg in theme_selected]

                logger.info(f"Prepared {len(themes)} theme summary tasks (hierarchical mode)")

        elif level == "subtheme":
            subtheme_map = context.get("subtheme_map", {})
            themes = context.get("themes", [])

            # Build parent theme name lookup
            theme_name_map = {t.theme_id: t.theme_name for t in themes}

            for parent_theme_id, subthemes in subtheme_map.items():
                parent_theme_name = theme_name_map.get(parent_theme_id, "Unknown Theme")

                for subtheme in subthemes:
                    selected_segments = getattr(subtheme, "selected", subtheme.segments[:10])
                    segments_text = self._format_segments_for_prompt(selected_segments)

                    task_id = f"subtheme_{subtheme.theme_id}"
                    tasks.append({
                        "template_name": template,
                        "context": {
                            "parent_theme_name": parent_theme_name,
                            "subtheme_name": subtheme.theme_name,
                            "segments_text": segments_text
                        },
                        "model": params.get("model", "grok-2-1212"),
                        "temperature": params.get("temperature"),
                        "max_tokens": params.get("max_tokens"),
                        "metadata": {
                            "parent_theme_id": parent_theme_id,
                            "subtheme_id": subtheme.theme_id,
                            "subtheme_name": subtheme.theme_name,
                            "task_id": task_id
                        }
                    })

                    # Track segment IDs for this task
                    segment_id_map[task_id] = [self._get_segment_id(seg) for seg in selected_segments]

        return tasks, segment_id_map

    def _format_segments_for_prompt(self, segments) -> str:
        """Format segments for LLM prompt with simple IDs."""
        parts = []
        for i, seg in enumerate(segments):
            citation_id = f"{{seg_{i+1}}}"
            text = self._get_segment_attr(seg, 'text', str(seg))
            channel = self._get_segment_attr(seg, 'channel_name', 'Unknown')
            date = self._get_segment_attr(seg, 'publish_date', 'Unknown')

            parts.append(
                f"{citation_id}\n"
                f"Channel: {channel}\n"
                f"Date: {date}\n"
                f"Text: {text}\n"
            )

        return "\n---\n".join(parts)

    def _format_group_summaries_with_sources(
        self,
        group_summaries: List[str],
        themes: List,
        citation_offset: int = 0
    ) -> str:
        """
        Format group summaries with their source transcripts for synthesis (Pass 2).

        Creates the input format expected by the rag_synthesis template:
        === GROUP 1 ===
        Summary: {group_1_summary}

        Source transcripts:
        {seg_1}
        Channel: ...
        ...

        Args:
            group_summaries: List of generated summaries from Pass 1
            themes: List of Theme objects with selected segments
            citation_offset: Starting offset for citation numbering (for continuity across groups)

        Returns:
            Formatted string for the synthesis prompt
        """
        sections = []
        current_seg_id = 1  # Global segment counter for citations

        for i, (summary, theme) in enumerate(zip(group_summaries, themes)):
            theme_segments = getattr(theme, "selected", theme.segments[:10])

            # Format transcripts for this group with globally unique segment IDs
            transcript_parts = []
            for seg in theme_segments:
                citation_id = f"{{seg_{current_seg_id}}}"
                text = self._get_segment_attr(seg, 'text', str(seg))
                channel = self._get_segment_attr(seg, 'channel_name', 'Unknown')
                date = self._get_segment_attr(seg, 'publish_date', 'Unknown')

                transcript_parts.append(
                    f"{citation_id}\n"
                    f"Channel: {channel}\n"
                    f"Date: {date}\n"
                    f"Text: {text}"
                )
                current_seg_id += 1

            transcripts_text = "\n\n---\n".join(transcript_parts)

            sections.append(f"""=== GROUP {i+1} ===
Summary: {summary}

Source transcripts:
{transcripts_text}""")

        return "\n\n".join(sections)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def _get_segment_attr(seg, attr: str, default=None):
        """Get attribute from segment (handles both dict and object segments)."""
        if isinstance(seg, dict):
            return seg.get(attr, default)
        return getattr(seg, attr, default)

    @staticmethod
    def _get_segment_id(seg):
        """Get segment ID (handles both dict and object segments)."""
        if isinstance(seg, dict):
            return seg.get("segment_id")
        return seg.id

    @staticmethod
    def _get_segment_embedding(seg):
        """Get segment embedding (handles both dict and object segments)."""
        if isinstance(seg, dict):
            # Check _embedding first (transient field from search), then embedding
            emb = seg.get("_embedding") or seg.get("embedding")
            if emb is not None and not isinstance(emb, np.ndarray):
                return np.array(emb)
            return emb
        return seg.embedding

    def _extract_step_results(self, step_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant results for a specific step type (avoid sending full context)."""

        if step_type == "expand_query":
            return {
                "original_query": context.get("query"),
                "expanded_queries": context.get("expanded_queries", []),
                "keywords": context.get("keywords", []),
                "expansion_strategy": context.get("expansion_strategy"),
                # Include serialized embeddings for caching
                "query_embeddings_cached": context.get("query_embeddings_cached", [])
            }

        elif step_type == "retrieve_segments_by_search":
            segments = context.get("segments", [])
            return {
                "segment_count": len(segments),
                "segments": segments,  # Already dicts from search
                # Include serialized embeddings for caching (generated during search)
                "query_embeddings_cached": context.get("query_embeddings_cached", [])
            }

        elif step_type == "rerank_segments":
            segments = context.get("segments", [])
            return {
                "segment_count": len(segments),
                "segments_before_rerank": context.get("segments_before_rerank", 0),
                "segments_after_rerank": context.get("segments_after_rerank", 0),
                "segments": segments  # Reranked dicts
            }

        elif step_type == "quick_cluster_check":
            themes = context.get("themes", [])
            return {
                "has_subclusters": context.get("has_subclusters", False),
                "num_clusters": len(themes),
                "cluster_validation": context.get("cluster_validation", {}),
                "cluster_sizes": [len(theme.segments) for theme in themes] if themes else [],
                "fallback_used": context.get("fallback_used", False)
            }

        elif step_type == "quantitative_analysis":
            return {
                "quantitative_metrics": context.get("quantitative_metrics", {})
            }

        elif step_type == "select_segments":
            selected = context.get("selected_segments", [])
            # Convert segment objects to dicts
            selected_dicts = []
            for seg in selected:
                if isinstance(seg, dict):
                    selected_dicts.append({
                        "segment_id": seg.get("segment_id"),
                        "content_id": seg.get("content_id"),
                        "content_id_string": seg.get("content_id_string"),
                        "start_time": seg.get("start_time"),
                        "end_time": seg.get("end_time"),
                        "text": seg.get("text"),
                        "channel_name": seg.get("channel_name", "Unknown"),
                        "title": seg.get("title", "Unknown"),
                        "publish_date": seg.get("publish_date")
                    })
                else:
                    # Object segment - extract fields
                    publish_date = getattr(seg.content, 'publish_date', None) if hasattr(seg, 'content') else None
                    if publish_date and hasattr(publish_date, 'isoformat'):
                        publish_date = publish_date.isoformat()

                    selected_dicts.append({
                        "segment_id": seg.id,
                        "content_id": seg.content_id,
                        "content_id_string": seg.content_id_string,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text,
                        "channel_name": getattr(seg.content, 'channel_name', 'Unknown') if hasattr(seg, 'content') else 'Unknown',
                        "title": getattr(seg.content, 'title', 'Unknown') if hasattr(seg, 'content') else 'Unknown',
                        "publish_date": publish_date
                    })

            return {
                "selected_segments": selected_dicts,
                "selection_strategy": context.get("selection_strategy")
            }

        elif step_type == "generate_summaries":
            return {
                "summaries": context.get("summaries", {}),
                "segment_ids": context.get("segment_ids", {})
            }

        else:
            # For unknown steps, return minimal context
            return {
                "step_type": step_type,
                "status": "completed"
            }

    def describe(self) -> str:
        """Get human-readable description of pipeline."""
        steps_str = "\n".join([
            f"  {i + 1}. {step_type}: {params}"
            for i, (step_type, params) in enumerate(self.steps)
        ])
        return f"Pipeline '{self.name}':\n{steps_str}"

    def __repr__(self) -> str:
        return f"<AnalysisPipeline '{self.name}' with {len(self.steps)} steps>"


# ============================================================================
# Helper Functions for Cache Fast Path
# ============================================================================

async def compute_fresh_quantitative_analysis(
    db_session,
    expanded_queries: List[str],
    segment_ids: List[int],
    time_window_days: int,
    global_filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute fresh quantitative metrics for cached segments.

    Used in Scenario A (full cache hit) to provide current stats
    even though segments + summary are cached.

    Args:
        db_session: Database session
        expanded_queries: Expanded query variations (for context)
        segment_ids: List of segment IDs to analyze
        time_window_days: Time window in days
        global_filters: Project/language/channel filters

    Returns:
        Quantitative metrics dict with fresh statistics
    """
    from .segment_retriever import SegmentRetriever
    from .quantitative_analyzer import QuantitativeAnalyzer
    from datetime import datetime, timezone, timedelta

    retriever = SegmentRetriever(db_session)
    analyzer = QuantitativeAnalyzer(db_session=db_session)

    # Fetch segments by IDs
    segments = retriever.fetch_by_ids(segment_ids)

    # Get baseline stats for comparison
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=time_window_days)

    baseline_filters = {
        "date_range": (start_date, end_date),
        "must_be_stitched": True,
        "must_be_embedded": True
    }

    # Copy project/language/channel filters
    for filter_key in ["projects", "languages", "channels"]:
        if filter_key in global_filters:
            baseline_filters[filter_key] = global_filters[filter_key]

    baseline_segments = retriever.get_baseline_stats(**baseline_filters)

    # Analyze with fresh baseline
    metrics = analyzer.analyze(
        segments=segments,
        baseline_segments=baseline_segments,
        time_window_days=time_window_days
    )

    logger.info(
        f"Computed fresh quantitative analysis: {metrics['total_segments']} segments, "
        f"centrality={metrics.get('discourse_centrality', {}).get('score', 0):.2f}"
    )

    return metrics
