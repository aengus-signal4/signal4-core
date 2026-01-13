"""
Health Query Generator
======================

Generates diverse query variations for comprehensive health & wellness analysis.

The generator creates queries across multiple health domains (nutrition, fitness,
mental health, sleep, longevity, etc.) using multiple strategies:

1. Direct domain queries - Base topics from config
2. LLM-expanded variations - Discourse-focused variations of each topic
3. Cross-domain connections - Queries linking related domains
4. Perspective variations - Different stances (mainstream, alternative, skeptical)

Usage:
    generator = HealthQueryGenerator(llm_service, config)
    queries = await generator.generate_all_queries()
    # Returns: List[Dict] with query, domain, strategy metadata
"""

import asyncio
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import yaml
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path, get_dashboard_config_path

logger = logging.getLogger(__name__)


@dataclass
class HealthQuery:
    """A health-related query with metadata."""
    query: str
    domain: str
    subdomain: Optional[str] = None
    strategy: str = "direct"  # direct, llm_expanded, cross_domain, perspective
    perspective: Optional[str] = None  # mainstream, alternative, skeptical
    weight: float = 1.0  # For sampling priority


@dataclass
class DomainAnalysis:
    """Analysis results for a health domain."""
    domain: str
    queries: List[HealthQuery]
    segments: List[Any] = field(default_factory=list)
    themes: List[Any] = field(default_factory=list)
    subthemes: Dict[str, List[Any]] = field(default_factory=dict)
    summary: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class HealthQueryGenerator:
    """
    Generates comprehensive health query variations for domain analysis.

    Strategies:
    1. Direct queries from config health_domains
    2. LLM expansion for discourse-focused variations
    3. Cross-domain queries connecting related topics
    4. Perspective variations (mainstream vs alternative medicine)
    """

    # Cross-domain connections for generating relationship queries
    DOMAIN_CONNECTIONS = {
        ("nutrition", "fitness"): [
            "protein timing around workouts",
            "pre-workout nutrition strategies",
            "post-exercise recovery nutrition",
            "carb cycling for performance",
        ],
        ("nutrition", "mental_health"): [
            "gut-brain axis and mood",
            "nutrition for anxiety and depression",
            "blood sugar and mental clarity",
            "omega-3 and brain health",
        ],
        ("sleep", "mental_health"): [
            "sleep deprivation and anxiety",
            "insomnia and depression connection",
            "sleep quality for emotional regulation",
            "dreams and psychological processing",
        ],
        ("fitness", "longevity"): [
            "exercise for healthy aging",
            "zone 2 cardio and lifespan",
            "strength training for longevity",
            "VO2 max as longevity biomarker",
        ],
        ("hormones", "fitness"): [
            "testosterone and muscle building",
            "cortisol from overtraining",
            "growth hormone and exercise",
            "thyroid and metabolism",
        ],
        ("sleep", "fitness"): [
            "sleep and muscle recovery",
            "training timing and sleep quality",
            "overtraining and sleep disruption",
            "sleep for athletic performance",
        ],
        ("nutrition", "longevity"): [
            "fasting and autophagy",
            "caloric restriction research",
            "anti-inflammatory diet",
            "polyphenols and aging",
        ],
        ("biohacking", "longevity"): [
            "blood biomarker optimization",
            "continuous glucose monitoring insights",
            "NAD precursor supplementation",
            "senolytic therapies",
        ],
        ("mental_health", "wellness_practices"): [
            "meditation for anxiety",
            "yoga for stress reduction",
            "breathwork for mental clarity",
            "nature therapy benefits",
        ],
    }

    # Perspective templates for generating varied viewpoints
    PERSPECTIVE_TEMPLATES = {
        "mainstream": [
            "evidence-based approach to {topic}",
            "doctor recommended {topic} strategies",
            "scientific research on {topic}",
            "medical guidelines for {topic}",
        ],
        "alternative": [
            "natural alternatives for {topic}",
            "holistic approach to {topic}",
            "functional medicine perspective on {topic}",
            "ancestral health view of {topic}",
        ],
        "skeptical": [
            "myths about {topic}",
            "is {topic} actually effective",
            "problems with {topic} claims",
            "what the research really says about {topic}",
        ],
        "practical": [
            "how to actually improve {topic}",
            "beginner guide to {topic}",
            "simple steps for {topic}",
            "daily habits for {topic}",
        ],
    }

    def __init__(
        self,
        llm_service,
        config_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize generator.

        Args:
            llm_service: LLMService for query expansion
            config_path: Path to dashboard config.yaml
            config: Pre-loaded config dict (alternative to config_path)
        """
        self.llm_service = llm_service

        # Load config
        if config:
            self.config = config
        elif config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config path
            default_path = get_dashboard_config_path("health_wellness")
            with open(default_path) as f:
                self.config = yaml.safe_load(f)

        self.health_domains = self.config.get("health_domains", {})
        self.analysis_config = self.config.get("analysis", {})

        logger.info(f"HealthQueryGenerator initialized with {len(self.health_domains)} domains")

    async def generate_all_queries(
        self,
        include_llm_expansion: bool = True,
        include_cross_domain: bool = True,
        include_perspectives: bool = True,
        queries_per_domain: Optional[int] = None
    ) -> List[HealthQuery]:
        """
        Generate comprehensive query set across all health domains.

        Args:
            include_llm_expansion: Expand queries with LLM variations
            include_cross_domain: Include cross-domain connection queries
            include_perspectives: Include perspective variation queries
            queries_per_domain: Override queries per domain from config

        Returns:
            List of HealthQuery objects with full metadata
        """
        queries_per = queries_per_domain or self.analysis_config.get("queries_per_domain", 5)
        all_queries: List[HealthQuery] = []

        # 1. Direct domain queries
        for domain, topics in self.health_domains.items():
            # Sample topics if we have more than needed
            sampled_topics = random.sample(topics, min(len(topics), queries_per))

            for topic in sampled_topics:
                all_queries.append(HealthQuery(
                    query=topic,
                    domain=domain,
                    strategy="direct",
                    weight=1.0
                ))

        logger.info(f"Generated {len(all_queries)} direct domain queries")

        # 2. LLM-expanded variations (async batch)
        if include_llm_expansion and self.llm_service:
            expanded = await self._expand_with_llm(all_queries[:20])  # Limit for cost
            all_queries.extend(expanded)
            logger.info(f"Added {len(expanded)} LLM-expanded queries")

        # 3. Cross-domain connection queries
        if include_cross_domain:
            cross_domain = self._generate_cross_domain_queries()
            all_queries.extend(cross_domain)
            logger.info(f"Added {len(cross_domain)} cross-domain queries")

        # 4. Perspective variations
        if include_perspectives:
            perspective_queries = self._generate_perspective_queries(all_queries[:15])
            all_queries.extend(perspective_queries)
            logger.info(f"Added {len(perspective_queries)} perspective queries")

        logger.info(f"Total queries generated: {len(all_queries)}")
        return all_queries

    async def _expand_with_llm(
        self,
        base_queries: List[HealthQuery],
        variations_per_query: int = 3
    ) -> List[HealthQuery]:
        """
        Expand queries using LLM for discourse-focused variations.

        Uses the theme_queries template pattern but adapted for health.
        """
        expanded = []

        system_message = """You are a health discourse analyst. Generate search query variations
that reflect how people ACTUALLY DISCUSS health topics in podcasts, YouTube videos, and social media.
Include both professional/scientific language AND casual/colloquial phrasing."""

        prompt_template = """Generate {n} distinct search query variations for this health topic:

Topic: "{topic}"
Domain: {domain}

Requirements:
1. Vary between scientific/medical language and everyday speech
2. Include questions people actually ask
3. Include claims/statements people make
4. Cover different aspects (benefits, risks, how-to, myths)
5. 2-5 words each, natural language

Return ONLY a JSON array of strings, no explanation:
["query 1", "query 2", ...]"""

        async def expand_one(query: HealthQuery) -> List[HealthQuery]:
            try:
                prompt = prompt_template.format(
                    n=variations_per_query,
                    topic=query.query,
                    domain=query.domain
                )

                response = await self.llm_service.call_grok_async(
                    prompt=prompt,
                    system_message=system_message,
                    model="grok-2-1212",
                    temperature=0.5,
                    max_tokens=300
                )

                # Parse JSON response
                import json
                variations = json.loads(response.strip())

                return [
                    HealthQuery(
                        query=v,
                        domain=query.domain,
                        subdomain=query.query,  # Original topic as subdomain
                        strategy="llm_expanded",
                        weight=0.8
                    )
                    for v in variations if isinstance(v, str)
                ]
            except Exception as e:
                logger.warning(f"Failed to expand query '{query.query}': {e}")
                return []

        # Run expansions concurrently with rate limiting
        semaphore = asyncio.Semaphore(10)

        async def limited_expand(q):
            async with semaphore:
                return await expand_one(q)

        results = await asyncio.gather(*[limited_expand(q) for q in base_queries])

        for result in results:
            expanded.extend(result)

        return expanded

    def _generate_cross_domain_queries(self) -> List[HealthQuery]:
        """Generate queries connecting related health domains."""
        queries = []

        for (domain1, domain2), topics in self.DOMAIN_CONNECTIONS.items():
            # Only include if both domains are in our config
            if domain1 in self.health_domains and domain2 in self.health_domains:
                for topic in topics:
                    queries.append(HealthQuery(
                        query=topic,
                        domain=f"{domain1}+{domain2}",
                        strategy="cross_domain",
                        weight=1.2  # Slightly higher weight for integrative topics
                    ))

        return queries

    def _generate_perspective_queries(
        self,
        base_queries: List[HealthQuery]
    ) -> List[HealthQuery]:
        """Generate queries with different perspectives/stances."""
        queries = []

        for query in base_queries:
            # Pick 1-2 random perspectives for each base query
            perspectives = random.sample(
                list(self.PERSPECTIVE_TEMPLATES.keys()),
                k=min(2, len(self.PERSPECTIVE_TEMPLATES))
            )

            for perspective in perspectives:
                templates = self.PERSPECTIVE_TEMPLATES[perspective]
                template = random.choice(templates)

                # Extract key topic from query
                topic = query.query.lower()

                queries.append(HealthQuery(
                    query=template.format(topic=topic),
                    domain=query.domain,
                    subdomain=query.query,
                    strategy="perspective",
                    perspective=perspective,
                    weight=0.9
                ))

        return queries

    def get_domains(self) -> List[str]:
        """Get list of configured health domains."""
        return list(self.health_domains.keys())

    def get_domain_topics(self, domain: str) -> List[str]:
        """Get topics for a specific domain."""
        return self.health_domains.get(domain, [])

    def sample_queries(
        self,
        queries: List[HealthQuery],
        n: int = 50,
        strategy: str = "weighted"
    ) -> List[HealthQuery]:
        """
        Sample a subset of queries for analysis.

        Args:
            queries: Full query list
            n: Number to sample
            strategy: "weighted" (by weight), "stratified" (by domain), "random"

        Returns:
            Sampled subset of queries
        """
        if len(queries) <= n:
            return queries

        if strategy == "weighted":
            # Weighted random sampling
            weights = [q.weight for q in queries]
            total = sum(weights)
            probs = [w/total for w in weights]
            indices = random.choices(range(len(queries)), weights=probs, k=n)
            return [queries[i] for i in set(indices)]

        elif strategy == "stratified":
            # Equal samples per domain
            by_domain: Dict[str, List[HealthQuery]] = {}
            for q in queries:
                by_domain.setdefault(q.domain, []).append(q)

            per_domain = max(1, n // len(by_domain))
            sampled = []
            for domain_queries in by_domain.values():
                sampled.extend(random.sample(
                    domain_queries,
                    min(len(domain_queries), per_domain)
                ))
            return sampled[:n]

        else:  # random
            return random.sample(queries, n)


class HealthDomainAnalyzer:
    """
    Orchestrates full health domain analysis using the RAG pipeline.

    Workflow:
    1. Generate diverse queries across health domains
    2. Run semantic search for each query batch
    3. Aggregate and deduplicate segments by domain
    4. Extract themes within each domain
    5. Extract sub-themes within each theme
    6. Generate domain summaries
    7. Generate cross-domain synthesis
    """

    def __init__(
        self,
        llm_service,
        embedding_service,
        db_session,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize analyzer.

        Args:
            llm_service: LLMService for summaries
            embedding_service: For query embeddings
            db_session: Database session
            config: Dashboard config dict
        """
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.db_session = db_session

        self.query_generator = HealthQueryGenerator(llm_service, config=config)
        self.config = config or {}

        # Lazy-load pipeline components
        self._search_service = None
        self._theme_extractor = None

    async def analyze_all_domains(
        self,
        time_window_days: int = 90,
        max_queries: int = 100,
        projects: Optional[List[str]] = None
    ) -> Dict[str, DomainAnalysis]:
        """
        Run comprehensive analysis across all health domains.

        Args:
            time_window_days: Time window for content
            max_queries: Maximum queries to process
            projects: Filter by projects

        Returns:
            Dict mapping domain -> DomainAnalysis
        """
        # 1. Generate queries
        all_queries = await self.query_generator.generate_all_queries()
        sampled_queries = self.query_generator.sample_queries(
            all_queries, n=max_queries, strategy="stratified"
        )

        logger.info(f"Analyzing {len(sampled_queries)} queries across {len(self.query_generator.get_domains())} domains")

        # 2. Group queries by domain
        queries_by_domain: Dict[str, List[HealthQuery]] = {}
        for q in sampled_queries:
            # Handle cross-domain queries
            primary_domain = q.domain.split("+")[0] if "+" in q.domain else q.domain
            queries_by_domain.setdefault(primary_domain, []).append(q)

        # 3. Analyze each domain in parallel
        domain_results: Dict[str, DomainAnalysis] = {}

        async def analyze_domain(domain: str, queries: List[HealthQuery]) -> DomainAnalysis:
            return await self._analyze_single_domain(
                domain=domain,
                queries=queries,
                time_window_days=time_window_days,
                projects=projects
            )

        tasks = [
            analyze_domain(domain, queries)
            for domain, queries in queries_by_domain.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for domain, result in zip(queries_by_domain.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Domain {domain} analysis failed: {result}")
                domain_results[domain] = DomainAnalysis(domain=domain, queries=[])
            else:
                domain_results[domain] = result

        return domain_results

    async def _analyze_single_domain(
        self,
        domain: str,
        queries: List[HealthQuery],
        time_window_days: int,
        projects: Optional[List[str]]
    ) -> DomainAnalysis:
        """Analyze a single health domain."""
        from .pgvector_search_service import PgVectorSearchService
        from .theme_extractor import ThemeExtractor
        from .segment_selector import SegmentSelector

        # Initialize services lazily
        if self._search_service is None:
            # Get dashboard config for search service
            dashboard_config = self.config.get("search", {})
            self._search_service = PgVectorSearchService(
                db_session=self.db_session,
                embedding_service=self.embedding_service,
                use_alt_embeddings=dashboard_config.get("use_alt_embeddings", False)
            )

        if self._theme_extractor is None:
            self._theme_extractor = ThemeExtractor()

        # Collect all segments for this domain's queries
        all_segments = []
        seen_ids = set()

        for query in queries:
            try:
                # Get embedding for query
                embedding = await self.embedding_service.get_embedding(query.query)

                # Search
                results = await self._search_service.search(
                    query_embedding=embedding,
                    k=50,
                    time_window_days=time_window_days,
                    projects=projects
                )

                # Deduplicate
                for seg in results:
                    if seg.id not in seen_ids:
                        seen_ids.add(seg.id)
                        all_segments.append(seg)

            except Exception as e:
                logger.warning(f"Query failed for '{query.query}': {e}")

        logger.info(f"Domain '{domain}': {len(all_segments)} unique segments from {len(queries)} queries")

        # Skip if too few segments
        min_segments = self.config.get("analysis", {}).get("min_segments_per_domain", 10)
        if len(all_segments) < min_segments:
            return DomainAnalysis(
                domain=domain,
                queries=queries,
                segments=all_segments,
                metrics={"segment_count": len(all_segments), "skipped": True}
            )

        # Extract themes
        theme_config = self.config.get("analysis", {}).get("theme_extraction", {})
        themes = self._theme_extractor.extract(
            segments=all_segments,
            method=theme_config.get("method", "hdbscan"),
            min_cluster_size=theme_config.get("min_cluster_size", 8),
            max_themes=theme_config.get("max_themes", 6)
        )

        # Extract sub-themes for each theme
        subtheme_config = self.config.get("analysis", {}).get("subtheme_extraction", {})
        subthemes: Dict[str, List[Any]] = {}

        if subtheme_config.get("enabled", True):
            for theme in themes:
                if len(theme.segments) >= subtheme_config.get("min_cluster_size", 4) * 2:
                    try:
                        sub = self._theme_extractor.extract(
                            segments=theme.segments,
                            method="hdbscan",
                            min_cluster_size=subtheme_config.get("min_cluster_size", 4),
                            max_themes=subtheme_config.get("max_subthemes", 4)
                        )
                        subthemes[theme.id] = sub
                    except Exception as e:
                        logger.warning(f"Sub-theme extraction failed for theme {theme.id}: {e}")

        return DomainAnalysis(
            domain=domain,
            queries=queries,
            segments=all_segments,
            themes=themes,
            subthemes=subthemes,
            metrics={
                "segment_count": len(all_segments),
                "theme_count": len(themes),
                "query_count": len(queries)
            }
        )

    async def generate_domain_summary(
        self,
        analysis: DomainAnalysis,
        template: str = "health_theme_summary"
    ) -> str:
        """Generate LLM summary for a domain analysis."""
        from .text_generator import TextGenerator

        generator = TextGenerator(self.llm_service)

        # Build segments text with citations
        segments_text = self._format_segments_for_summary(analysis.segments[:20])

        # Format theme info
        theme_descriptions = []
        for theme in analysis.themes:
            theme_descriptions.append(f"- {theme.name}: {len(theme.segments)} segments")

        context = {
            "domain": analysis.domain,
            "theme_name": f"Health Domain: {analysis.domain.replace('_', ' ').title()}",
            "segments_text": segments_text,
            "theme_count": len(analysis.themes),
            "theme_descriptions": "\n".join(theme_descriptions),
            "segment_count": len(analysis.segments),
            "metrics": analysis.metrics
        }

        summary = await generator.generate_from_template(
            template_name=template,
            context=context,
            model="grok-4-fast-non-reasoning-latest",
            max_tokens=500
        )

        return summary

    def _format_segments_for_summary(self, segments: List[Any]) -> str:
        """Format segments with citation IDs for LLM summary."""
        lines = []
        for i, seg in enumerate(segments):
            citation_id = f"citation{i+1}"
            text = getattr(seg, 'text', str(seg))[:500]  # Truncate long segments
            lines.append(f"[{citation_id}] {text}")
        return "\n\n".join(lines)


async def run_health_analysis_pipeline(
    llm_service,
    embedding_service,
    db_session,
    time_window_days: int = 90,
    projects: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run full health analysis pipeline.

    Returns comprehensive analysis with:
    - Per-domain analysis (themes, sub-themes, metrics)
    - Domain summaries
    - Cross-domain synthesis
    """
    # Load config
    config_path = get_dashboard_config_path("health_wellness")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    analyzer = HealthDomainAnalyzer(
        llm_service=llm_service,
        embedding_service=embedding_service,
        db_session=db_session,
        config=config
    )

    # Run analysis
    domain_results = await analyzer.analyze_all_domains(
        time_window_days=time_window_days,
        projects=projects
    )

    # Generate summaries for each domain
    summaries = {}
    for domain, analysis in domain_results.items():
        if analysis.themes:  # Only summarize domains with themes
            summaries[domain] = await analyzer.generate_domain_summary(analysis)

    return {
        "domains": {
            domain: {
                "segment_count": a.metrics.get("segment_count", 0),
                "theme_count": len(a.themes),
                "themes": [{"id": t.id, "name": t.name, "size": len(t.segments)} for t in a.themes],
                "subtheme_count": sum(len(subs) for subs in a.subthemes.values()),
                "summary": summaries.get(domain)
            }
            for domain, a in domain_results.items()
        },
        "total_segments": sum(a.metrics.get("segment_count", 0) for a in domain_results.values()),
        "total_themes": sum(len(a.themes) for a in domain_results.values()),
        "domains_analyzed": len(domain_results)
    }
