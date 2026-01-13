"""
LLM Service
============

Client for xAI Grok and Google Gemini APIs with query conversion and RAG summaries.
Includes embedding model for semantic search.
"""

import os
import logging
import time
import requests
import json
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from collections import Counter
# Lazy imports to avoid PyTorch/MPS initialization issues on macOS:
# - numpy (can trigger MPS issues)
# - sentence_transformers (triggers PyTorch init)
# - google.generativeai (can cause issues on some systems)
# These imports are moved to respective methods
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

import sys
from pathlib import Path
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend.llm")
usage_logger = setup_worker_logger("backend.usage")

# Import activity logger
try:
    from ..utils.logger import activity_logger
except ImportError:
    activity_logger = None
    logger.warning("Could not import activity_logger - logging disabled")

# Import PostgreSQL LLM cache
from .pg_cache_service import PgLLMCache


class LLMService:
    """Client for Google Gemini API with RAG capabilities"""

    def __init__(self, config, dashboard_id='unknown'):
        """
        Initialize LLM service

        Args:
            config: DashboardConfig object
            dashboard_id: Dashboard identifier for logging
        """
        self.config = config
        self.dashboard_id = dashboard_id

        # Initialize PostgreSQL LLM response cache
        self._cache = PgLLMCache(
            dashboard_id=dashboard_id,
            similarity_threshold=0.85,  # 85% similarity for cache hit
            ttl_hours=None  # Permanent cache (no expiration)
        )
        logger.info(f"[{self.dashboard_id}] PostgreSQL LLM cache enabled (permanent)")

        # Configure Google API
        self._configure_google_api()

        # Initialize TextGenerator (lazy-loaded on first use)
        self._text_generator = None

        logger.info(f"[{self.dashboard_id}] LLM Service initialized (model: {config.llm_model})")

    def _configure_google_api(self):
        """Configure Google Gemini API (lazy import)"""
        try:
            import google.generativeai as genai
            from ..config import settings
            if not settings.GOOGLE_API_KEY:
                logger.error("GOOGLE_API_KEY not found in environment")
                return

            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Google Gemini API configured successfully")

        except Exception as e:
            logger.error(f"Error configuring Google API: {e}")

    def _get_text_generator(self):
        """Get or create TextGenerator instance (lazy loading)"""
        if self._text_generator is None:
            from .rag.text_generator import TextGenerator
            self._text_generator = TextGenerator(self)
            logger.debug(f"[{self.dashboard_id}] TextGenerator initialized")
        return self._text_generator

    # ============================================================================
    # DEPRECATED: Embedding methods moved to EmbeddingService
    # ============================================================================
    # These methods are kept temporarily for backwards compatibility
    # but will be removed in future versions. Use EmbeddingService instead.

    async def batch_convert_queries_to_embeddings(self, queries: List[str]):
        """
        DEPRECATED: Use EmbeddingService.encode_queries() instead.

        This method is kept for backwards compatibility only.
        """
        logger.warning(
            f"[{self.dashboard_id}] DEPRECATED: LLMService.batch_convert_queries_to_embeddings() "
            "is deprecated. Use EmbeddingService.encode_queries() instead."
        )
        from .embedding_service import EmbeddingService
        embedding_service = EmbeddingService(self.config, self.dashboard_id)
        try:
            return await embedding_service.encode_queries(queries)
        except RuntimeError as e:
            logger.error(f"[{self.dashboard_id}] Embedding generation failed: {e}")
            return None

    def convert_query_to_embedding(self, query: str):
        """
        DEPRECATED: Use EmbeddingService.encode_query() instead.

        This method is kept for backwards compatibility only.
        """
        logger.warning(
            f"[{self.dashboard_id}] DEPRECATED: LLMService.convert_query_to_embedding() "
            "is deprecated. Use EmbeddingService.encode_query() instead."
        )
        from .embedding_service import EmbeddingService
        embedding_service = EmbeddingService(self.config, self.dashboard_id)
        try:
            return embedding_service.encode_query(query)
        except RuntimeError as e:
            logger.error(f"[{self.dashboard_id}] Embedding generation failed: {e}")
            return None

    def query2doc(self, query: str) -> str:
        """
        Query2doc: Generate a pseudo-document for the query (faster than multi-query)

        Based on https://arxiv.org/pdf/2503.03417 - generates a single pseudo-document
        that represents what a relevant document might say, then embeds [query + pseudo-doc].

        This is much faster than generating 10 query variations:
        - 1 LLM call instead of 1
        - 1 embedding instead of 10
        - 1 FAISS search instead of 10

        Args:
            query: Natural language query

        Returns:
            Pseudo-document text to append to query
        """
        # Check cache first
        cache_key = f"query2doc:{query}"
        if self._cache:
            cached_response = self._cache.get(cache_key, query_embedding=None, cache_type='query2doc')
            if cached_response:
                return cached_response

        start_time = time.time()
        try:
            from ..config import settings
            if not settings.XAI_API_KEY:
                logger.warning("XAI_API_KEY not found, skipping query2doc")
                return ""

            # Use TextGenerator with query2doc template
            text_gen = self._get_text_generator()

            # Note: Using sync requests.post instead of async to maintain backward compatibility
            # The template is designed for sync generation
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.XAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-4-fast-non-reasoning-latest",
                    "messages": [{
                        "role": "system",
                        "content": text_gen.prompt_manager.get("query2doc").system_message
                    }, {
                        "role": "user",
                        "content": text_gen.prompt_manager.get("query2doc").prompt_template.format(query=query)
                    }],
                    "temperature": 0.3,
                    "max_tokens": 150
                },
                timeout=10
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                pseudo_doc = response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                logger.info(f"[{self.dashboard_id}] query2doc generated in {duration_ms:.0f}ms: {pseudo_doc[:80]}...")

                # Cache result
                if self._cache:
                    self._cache.put(cache_key, query_embedding=None, response=pseudo_doc, cache_type='query2doc')

                return pseudo_doc
            else:
                logger.warning(f"[{self.dashboard_id}] query2doc failed: {response.status_code}")
                return ""

        except Exception as e:
            logger.warning(f"[{self.dashboard_id}] query2doc error: {e}")
            return ""

    def generate_retrieval_queries(
        self,
        query: str,
        strategy: str = 'multi_stance',
        n_variations: int = 5,
        include_keywords: bool = False
    ) -> Dict[str, Any]:
        """
        Generate retrieval queries using different strategies, optionally with bilingual keywords.

        IMPORTANT: Generated queries will be embedded WITH the instruction prefix.
        Documents in the database are embedded WITHOUT any prefix (plain text).

        NOTE: The include_keywords feature generates bilingual keywords for text-based search,
        but testing shows minimal benefit (0% improvement in 30-day test). Keywords are kept
        for flexibility but not recommended. Use conversational strategy at 0.43 threshold instead.

        Args:
            query: User's question or topic (e.g., "immigration policy")
            strategy: Strategy to use:
                - 'multi_stance': Generate queries from different ideological perspectives (default)
                - 'conversational': Generate conversational paragraphs simulating natural speech
            n_variations: Number of query variations to generate (default 5)
            include_keywords: Whether to also generate bilingual keyword phrases (default False, not recommended)

        Returns:
            Dictionary containing:
                - 'queries': List[str] of generated queries
                - 'keywords': List[str] of keyword phrases (empty if include_keywords=False)

        Example:
            result = llm.generate_retrieval_queries("climate change", strategy='conversational')
            # result['queries'] = ["I've been reading about climate change...", ...]
            # result['keywords'] = []  # Not using keywords
        """
        # Check cache first
        cache_key = f"generate_retrieval_queries:{query}:{strategy}:{n_variations}:{include_keywords}"
        if self._cache:
            cached_response = self._cache.get(cache_key, query_embedding=None, cache_type='query2doc_stances')
            if cached_response:
                return cached_response

        start_time = time.time()
        try:
            from ..config import settings
            if not settings.XAI_API_KEY:
                logger.warning("XAI_API_KEY not found, skipping generate_retrieval_queries")
                return {'queries': [], 'keywords': []}

            # Select template based on strategy
            text_gen = self._get_text_generator()
            if strategy == 'multi_stance':
                template_name = 'multi_stance_queries'
            elif strategy == 'conversational':
                template_name = 'conversational_queries'
            else:
                logger.error(f"Unknown strategy: {strategy}")
                return {'queries': [], 'keywords': []}

            template = text_gen.prompt_manager.get(template_name)
            if not template:
                logger.error(f"Template '{template_name}' not found")
                return {'queries': [], 'keywords': []}

            # Format prompt
            prompt_text = template.prompt_template.format(query=query, n_variations=n_variations)

            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.XAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-4-fast-non-reasoning-latest",
                    "messages": [{
                        "role": "system",
                        "content": template.system_message
                    }, {
                        "role": "user",
                        "content": prompt_text
                    }],
                    "temperature": template.default_temperature,
                    "max_tokens": template.default_max_tokens
                },
                timeout=15
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                # Parse JSON response
                try:
                    # Extract JSON from response (may have markdown code blocks)
                    text = content
                    if '```json' in text:
                        text = text.split('```json')[1].split('```')[0].strip()
                    elif '```' in text:
                        text = text.split('```')[1].split('```')[0].strip()

                    parsed = json.loads(text)
                    queries = parsed.get('queries', [])

                    if not queries or len(queries) < n_variations:
                        logger.warning(f"[{self.dashboard_id}] Expected {n_variations} queries, got {len(queries)}")

                    # Extract keywords separately if requested (for true keyword search)
                    keywords = []
                    if include_keywords:
                        keywords_en = parsed.get('keywords_en', [])
                        keywords_fr = parsed.get('keywords_fr', [])
                        keywords = keywords_en + keywords_fr

                    logger.info(f"[{self.dashboard_id}] generate_retrieval_queries ({strategy}) generated {len(queries)} queries + {len(keywords)} keywords in {duration_ms:.0f}ms")
                    logger.debug(f"[{self.dashboard_id}] Queries: {[q[:60]+'...' for q in queries[:5]]}")
                    if keywords:
                        logger.debug(f"[{self.dashboard_id}] Keywords: {keywords[:10]}")

                    result = {
                        'queries': queries,
                        'keywords': keywords  # Separate list for true keyword search
                    }

                    # Cache result
                    if self._cache:
                        self._cache.put(cache_key, query_embedding=None, response=result, cache_type='query2doc_stances')

                    return result

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"[{self.dashboard_id}] Failed to parse retrieval queries JSON: {e}")
                    logger.warning(f"[{self.dashboard_id}] Raw content: {content[:1000]}")
                    return {'queries': [], 'keywords': []}
            else:
                logger.warning(f"[{self.dashboard_id}] generate_retrieval_queries failed: {response.status_code}")
                return {'queries': [], 'keywords': []}

        except Exception as e:
            logger.warning(f"[{self.dashboard_id}] generate_retrieval_queries error: {e}")
            return {'queries': [], 'keywords': []}

    def query2doc_stances(self, query: str, n_stances: int = 3) -> List[str]:
        """
        DEPRECATED: Use generate_retrieval_queries() instead.

        Backward compatibility wrapper for old method name.
        Returns only queries (not keywords).
        """
        result = self.generate_retrieval_queries(
            query=query,
            strategy='multi_stance',
            n_variations=n_stances,
            include_keywords=False
        )
        return result.get('queries', [])

    def optimize_search_query(self, query: str, user_email: str = None) -> Dict[str, Any]:
        """
        Use xAI Grok to generate multiple query variations for expanded search

        Generates 5-6 query variations including:
        - Original query reformulated
        - Related concepts and synonyms
        - Named entity variations
        - French translations (for bilingual coverage)
        - All formatted with Qwen3 instruction prefix

        Args:
            query: Natural language query
            user_email: User email for logging

        Returns:
            Dict with 'keywords', 'query_variations', 'search_type', 'variation_embeddings' (optional)
        """
        if not self.config.llm_enabled:
            return {
                'keywords': query.split(),
                'query_variations': [query],
                'search_type': 'keyword'
            }

        # Check cache with exact text match first (fast path, no embedding needed)
        if self._cache:
            cached_response = self._cache.get(query, query_embedding=None, cache_type='query_optimization')
            if cached_response:
                logger.info(f"[{self.dashboard_id}] ✓ Cache hit for query optimization (exact text match)")
                return cached_response

        # No exact match - generate embedding for semantic similarity lookup
        query_embedding = self.convert_query_to_embedding(query)

        # Check cache with semantic similarity (catches paraphrases)
        if self._cache and query_embedding is not None:
            cached_response = self._cache.get(query, query_embedding, cache_type='query_optimization')
            if cached_response:
                logger.info(f"[{self.dashboard_id}] ✓ Cache hit for query optimization (semantic similarity)")
                return cached_response

        start_time = time.time()
        try:
            logger.info(f"[{self.dashboard_id}] Generating query variations via Grok: '{query[:50]}...'")

            from ..config import settings
            if not settings.XAI_API_KEY:
                logger.error("XAI_API_KEY not found in environment")
                return self._create_fallback_variations(query)
            xai_api_key = settings.XAI_API_KEY

            prompt = f"""You are assisting academic research on political discourse and media content analysis. Generate 10 diverse search query variations for this research question. The goal is to maximize search recall by covering different perspectives, phrasings that reflect how people actually discuss these topics, and both English and French.

CONTEXT: This is for analyzing existing media content and discourse (not generating or advocating for any viewpoint).

User query: "{query}"

Instructions:
1. Generate 10 short query variations (1-2 sentences max)
2. Structure: 5 English variations + 5 French variations
3. For each language, include:
   - Reformulated queries with key entities and concepts
   - Different phrasings reflecting how people actually discuss this topic
   - Language that matches the discourse style (formal analysis + colloquial perspectives)
   - Synonyms and related concepts
4. Return ONLY the query text - we'll add the instruction prefix automatically
5. Use neutral, analytical language appropriate for academic research

Respond in JSON format:
{{
    "keywords": ["main", "keywords", "from", "query"],
    "query_variations": [
        "english variation 1",
        "english variation 2",
        "english variation 3",
        "english variation 4",
        "english variation 5",
        "french variation 1",
        "french variation 2",
        "french variation 3",
        "french variation 4",
        "french variation 5"
    ],
    "search_type": "semantic"
}}

Example for "What is Mark Carney saying about climate?":
{{
    "keywords": ["Mark Carney", "climate", "environment", "Liberal"],
    "query_variations": [
        "Mark Carney discusses climate change policy and environmental action",
        "Liberal leader environmental stance and green initiatives",
        "Political climate action promises and carbon reduction plans",
        "Canadian political figure comments on global warming",
        "Mark Carney's position on fighting climate change",
        "Mark Carney parle de changement climatique et action environnementale",
        "Position du chef libéral sur l'environnement et initiatives vertes",
        "Promesses d'action climatique politique et réduction du carbone",
        "Commentaires de figure politique canadienne sur réchauffement planétaire",
        "Position de Mark Carney sur la lutte contre le changement climatique"
        "Mark Carney parle du changement climatique et de l'environnement"
    ],
    "search_type": "semantic"
}}"""

            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {xai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-4-fast-non-reasoning-latest",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant for academic research query expansion. Always respond with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.5,
                    "max_tokens": 600
                },
                timeout=30
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                logger.error(f"[{self.dashboard_id}] Grok API error: {response.status_code} - {response.text[:200]}")
                return self._create_fallback_variations(query)

            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            if not content:
                logger.error(f"[{self.dashboard_id}] Empty response from Grok")
                return self._create_fallback_variations(query)

            # Parse JSON response
            try:
                # Extract JSON from response (may have markdown code blocks)
                text = content.strip()
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()

                parsed = json.loads(text)

                # Validate we got variations
                raw_variations = parsed.get('query_variations', [])
                if not raw_variations:
                    raise ValueError("No query variations generated")

                # Wrap each variation with Qwen3 instruction prefix
                formatted_variations = [
                    f"Instruct: Retrieve relevant passages.\nQuery: {var}"
                    for var in raw_variations
                ]

                # Update parsed dict with formatted variations
                parsed['query_variations'] = formatted_variations

                logger.info(f"[{self.dashboard_id}] ✓ Generated {len(formatted_variations)} query variations via Grok in {duration_ms:.0f}ms")
                logger.debug(f"[{self.dashboard_id}] Raw variations: {[v[:40]+'...' for v in raw_variations]}")

                # Log to activity logger
                if activity_logger and user_email:
                    activity_logger.log_llm_request(
                        user_email,
                        self.dashboard_id,
                        'query_expansion',
                        query,
                        int(duration_ms),
                        success=True
                    )

                # Don't generate embeddings here - will be done by search router if needed
                # This avoids nested async issues and lets the router handle embedding caching
                parsed['variation_embeddings'] = None

                # Cache the response WITH query embedding for semantic similarity matching
                if self._cache and query_embedding is not None:
                    self._cache.put(query, query_embedding, response=parsed, cache_type='query_optimization')
                    logger.debug(f"[{self.dashboard_id}] Cached query optimization with query embedding + variation embeddings")

                return parsed

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[{self.dashboard_id}] Failed to parse Grok response: {str(e)[:100]}")
                logger.debug(f"[{self.dashboard_id}] Raw content: {content[:500]}")
                return self._create_fallback_variations(query)

        except requests.exceptions.RequestException as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{self.dashboard_id}] Grok API request error: {e}")

            # Log failure
            if activity_logger and user_email:
                activity_logger.log_llm_request(
                    user_email,
                    self.dashboard_id,
                    'query_expansion',
                    query,
                    int(duration_ms),
                    success=False,
                    error=str(e)
                )

            return self._create_fallback_variations(query)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{self.dashboard_id}] Error generating query variations: {e}")

            # Log failure
            if activity_logger and user_email:
                activity_logger.log_llm_request(
                    user_email,
                    self.dashboard_id,
                    'query_expansion',
                    query,
                    int(duration_ms),
                    success=False,
                    error=str(e)
                )

            return self._create_fallback_variations(query)

    def generate_theme_query_variations_no_embeddings(self, theme_name: str, theme_description: str, user_email: str = None) -> Dict[str, Any]:
        """
        Generate query variations via Grok WITHOUT computing embeddings.

        This is used by the batch caching script to separate Grok API calls
        from embedding computation, allowing models to be loaded once.

        Args:
            theme_name: Name of the theme
            theme_description: Description of the theme
            user_email: User email for logging

        Returns:
            Dict with 'query_variations' (10 variations: 5 EN + 5 FR) and 'search_type'
        """
        if not self.config.llm_enabled:
            query = f"What is being said about {theme_name}? {theme_description[:100]}"
            return {
                'query_variations': [f"Instruct: Retrieve relevant passages.\nQuery: {query}"],
                'search_type': 'semantic'
            }

        # Create base query for cache lookup
        cache_query = f"THEME:{theme_name}|{theme_description[:100]}"

        # Check cache first (variations only, no embeddings)
        if self._cache:
            cached_response = self._cache.get(cache_query, query_embedding=None, cache_type='theme_queries_no_embed')
            if cached_response:
                return cached_response

        start_time = time.time()
        try:
            logger.info(f"[{self.dashboard_id}] Generating theme query variations via Grok: '{theme_name}'")

            from ..config import settings
            if not settings.XAI_API_KEY:
                logger.error("XAI_API_KEY not found in environment")
                return self._create_fallback_theme_variations(theme_name, theme_description)
            xai_api_key = settings.XAI_API_KEY

            # Use TextGenerator with theme_queries template
            text_gen = self._get_text_generator()
            template = text_gen.prompt_manager.get('theme_queries')
            if not template:
                logger.error("Template 'theme_queries' not found")
                return self._create_fallback_theme_variations(theme_name, theme_description)

            # Format prompt
            prompt_text = template.prompt_template.format(
                theme_name=theme_name,
                theme_description=theme_description
            )

            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {xai_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'grok-4-fast-non-reasoning-latest',
                    'messages': [{
                        'role': 'system',
                        'content': template.system_message
                    }, {
                        'role': 'user',
                        'content': prompt_text
                    }],
                    'temperature': template.default_temperature,
                    'max_tokens': template.default_max_tokens
                },
                timeout=30
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                logger.error(f"[{self.dashboard_id}] Grok API error: {response.status_code} - {response.text[:500]}")
                return self._create_fallback_theme_variations(theme_name, theme_description)

            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            if not content:
                logger.error(f"[{self.dashboard_id}] Empty response from Grok")
                return self._create_fallback_theme_variations(theme_name, theme_description)

            # Parse JSON response
            try:
                text = content.strip()
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()

                parsed = json.loads(text)
                raw_variations = parsed.get('query_variations', [])

                if not raw_variations or len(raw_variations) < 10:
                    raise ValueError(f"Expected 10 variations, got {len(raw_variations)}")

                # Wrap with Qwen3 instruction prefix
                formatted_variations = [
                    f"Instruct: Retrieve relevant passages.\nQuery: {var}"
                    for var in raw_variations
                ]

                result = {
                    'query_variations': formatted_variations,
                    'search_type': 'semantic'
                }

                logger.info(f"[{self.dashboard_id}] ✓ Generated {len(formatted_variations)} theme query variations in {duration_ms:.0f}ms")

                # Cache variations only (no embeddings)
                if self._cache:
                    self._cache.put(cache_query, query_embedding=None, response=result, cache_type='theme_queries_no_embed')

                return result

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[{self.dashboard_id}] Failed to parse Grok response: {str(e)[:100]}")
                return self._create_fallback_theme_variations(theme_name, theme_description)

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Error generating theme variations: {e}")
            return self._create_fallback_theme_variations(theme_name, theme_description)

    def generate_theme_query_variations(self, theme_name: str, theme_description: str, user_email: str = None) -> Dict[str, Any]:
        """
        Generate query variations for theme exploration (with embeddings).

        This is a convenience wrapper that calls generate_theme_query_variations_no_embeddings()
        and then computes embeddings for a single theme. For batch processing, use
        batch_compute_theme_embeddings() instead to load models only once.

        Args:
            theme_name: Name of the theme
            theme_description: Description of the theme
            user_email: User email for logging

        Returns:
            Dict with 'query_variations', 'search_type', 'embeddings_1024', 'embeddings_2000'
        """
        import numpy as np
        
        # Create base query for cache lookup
        cache_query = f"THEME:{theme_name}|{theme_description[:100]}"

        # Check full cache first (with embeddings)
        if self._cache:
            cached_response = self._cache.get(cache_query, query_embedding=None, cache_type='theme_queries')
            if cached_response:
                return cached_response

        # Generate variations via Grok
        result = self.generate_theme_query_variations_no_embeddings(theme_name, theme_description, user_email)
        variations = result.get('query_variations', [])

        if not variations:
            return result

        # Compute embeddings for this single theme using pre-loaded models
        logger.info(f"[{self.dashboard_id}] Computing embeddings (1024-dim + 2000-dim) for {len(variations)} variations...")

        try:
            # Use pre-loaded models from global cache
            from ..main import _embedding_models

            if '0.6B' not in _embedding_models or '4B' not in _embedding_models:
                logger.error(f"[{self.dashboard_id}] Pre-loaded models not available!")
                return result

            model_1024 = _embedding_models['0.6B']
            model_2000 = _embedding_models['4B']

            # Embed all variations
            result['embeddings_1024'] = []
            result['embeddings_2000'] = []

            for var in variations:
                # Use batch_size=1 to prevent multiprocessing conflicts
                emb_1024 = model_1024.encode(var, convert_to_numpy=True, normalize_embeddings=True, batch_size=1).astype(np.float32)
                emb_2000 = model_2000.encode(var, convert_to_numpy=True, normalize_embeddings=True, batch_size=1).astype(np.float32)
                result['embeddings_1024'].append(emb_1024)
                result['embeddings_2000'].append(emb_2000)

            logger.info(f"[{self.dashboard_id}] ✓ Computed embeddings for both dimensions using pre-loaded models")

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Failed to compute embeddings: {e}", exc_info=True)
            return result

        # Cache the full result
        if self._cache:
            self._cache.put(cache_query, query_embedding=None, response=result, cache_type='theme_queries')

        return result

    def batch_compute_theme_embeddings(self, themes_data: List[Dict[str, Any]]) -> None:
        """
        Batch compute embeddings for multiple themes efficiently.

        Uses pre-loaded models from global cache, no need to load/unload.

        Args:
            themes_data: List of dicts with 'theme_name', 'theme_desc', 'variations'
        """
        import numpy as np

        if not themes_data:
            logger.warning(f"[{self.dashboard_id}] No themes to process")
            return

        total_variations = sum(len(t.get('variations', [])) for t in themes_data)
        logger.info(f"[{self.dashboard_id}] Batch computing embeddings for {len(themes_data)} themes ({total_variations} total variations)")

        try:
            # Use pre-loaded models from global cache
            from ..main import _embedding_models

            if '0.6B' not in _embedding_models or '4B' not in _embedding_models:
                logger.error(f"[{self.dashboard_id}] Pre-loaded models not available!")
                return

            model_1024 = _embedding_models['0.6B']
            model_2000 = _embedding_models['4B']

            # Embed ALL variations with 0.6B
            logger.info(f"[{self.dashboard_id}] Embedding {total_variations} variations with 0.6B...")
            for theme_data in themes_data:
                variations = theme_data.get('variations', [])
                theme_data['embeddings_1024'] = []
                for var in variations:
                    # Use batch_size=1 to prevent multiprocessing conflicts
                    emb = model_1024.encode(var, convert_to_numpy=True, normalize_embeddings=True, batch_size=1).astype(np.float32)
                    theme_data['embeddings_1024'].append(emb)

            logger.info(f"[{self.dashboard_id}] ✓ Completed 1024-dim embeddings")

            # Embed ALL variations with 4B
            logger.info(f"[{self.dashboard_id}] Embedding {total_variations} variations with 4B...")
            for theme_data in themes_data:
                variations = theme_data.get('variations', [])
                theme_data['embeddings_2000'] = []
                for var in variations:
                    # Use batch_size=1 to prevent multiprocessing conflicts
                    emb = model_2000.encode(var, convert_to_numpy=True, normalize_embeddings=True, batch_size=1).astype(np.float32)
                    theme_data['embeddings_2000'].append(emb)

            logger.info(f"[{self.dashboard_id}] ✓ Completed 2000-dim embeddings")

            # Cache all results
            if self._cache:
                logger.info(f"[{self.dashboard_id}] Caching {len(themes_data)} theme embeddings...")
                for theme_data in themes_data:
                    cache_query = f"THEME:{theme_data['theme_name']}|{theme_data['theme_desc'][:100]}"
                    result = {
                        'query_variations': theme_data['variations'],
                        'search_type': 'semantic',
                        'embeddings_1024': theme_data['embeddings_1024'],
                        'embeddings_2000': theme_data['embeddings_2000']
                    }
                    self._cache.put(cache_query, query_embedding=None, response=result, cache_type='theme_queries')

            logger.info(f"[{self.dashboard_id}] ✓ Batch embedding complete")

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Batch embedding failed: {e}", exc_info=True)

    def _create_fallback_theme_variations(self, theme_name: str, theme_description: str) -> Dict[str, Any]:
        """Create basic theme query variations if Grok fails"""
        query = f"What is being said about {theme_name}? {theme_description[:100]}"
        formatted_query = f"Instruct: Retrieve relevant passages.\nQuery: {query}"

        return {
            'query_variations': [formatted_query],
            'search_type': 'semantic'
        }

    def _create_fallback_variations(self, query: str) -> Dict[str, Any]:
        """Create basic query variations if Grok fails"""
        # Just format the original query with Qwen3 instruction
        formatted_query = f"Instruct: Retrieve relevant passages.\nQuery: {query}"

        return {
            'keywords': query.split(),
            'query_variations': [formatted_query],
            'search_type': 'semantic'
        }

    def generate_rag_summary(self, query: str, segments: List[Dict]) -> Dict[str, Any]:
        """
        Generate RAG summary from search results

        Args:
            query: User's original query
            segments: List of segment dicts with 'text', 'similarity', etc.

        Returns:
            Dict with 'summary', 'keywords', 'segment_count', 'volume_stats'
        """
        if not self.config.llm_enabled or not segments:
            return {
                'summary': None,
                'keywords': [],
                'segment_count': len(segments),
                'volume_stats': {}
            }

        # Check cache first
        query_embedding = None
        if self._cache:
            query_embedding = self.convert_query_to_embedding(query)
            if query_embedding is not None:
                cached_response = self._cache.get(query, query_embedding, cache_type='rag_summary')
                if cached_response:
                    return cached_response

        # Sample diverse segments
        sampled = self._sample_diverse_segments(segments, self.config.llm_max_sample_segments)

        # Build context from sampled segments
        context = self._build_rag_context(sampled)

        # Generate summary with Grok
        start_time = time.time()
        try:
            from ..config import settings
            if not settings.XAI_API_KEY:
                logger.error("XAI_API_KEY not found in environment")
                summary = None
            else:
                xai_api_key = settings.XAI_API_KEY
                prompt = f"""Based on these audio transcript segments from Canadian political media, answer the user's question.

User question: "{query}"

Transcript segments:
{context}

INSTRUCTIONS:
1. First line: Single-sentence direct answer to the question.

2. Second paragraph: Brief summary of key points and arguments (3-5 sentences). CITE segments in parentheses like (Segment 1) or (Segments 3, 7) when referencing specific content.

Keep it concise and factual."""

                response = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {xai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "grok-4-fast-non-reasoning-latest",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant for academic research analysis. Provide objective, neutral summaries of media discourse."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": self.config.llm_temperature,
                        "max_tokens": 400
                    },
                    timeout=30
                )

                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    summary = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    if summary:
                        logger.info(f"[{self.dashboard_id}] ✓ Generated RAG summary via Grok ({len(summary)} chars) in {duration_ms:.0f}ms")
                    else:
                        logger.error(f"[{self.dashboard_id}] Empty summary from Grok")
                        summary = None
                else:
                    logger.error(f"[{self.dashboard_id}] Grok API error: {response.status_code} - {response.text[:200]}")
                    summary = None

        except requests.exceptions.RequestException as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{self.dashboard_id}] Grok API request error for RAG: {e}")
            summary = None
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{self.dashboard_id}] Error generating summary with Grok: {e}")
            summary = None

        # Extract keywords from segments
        keywords = self._extract_keywords(segments)

        # Calculate volume stats
        volume_stats = self._calculate_volume_stats(segments)

        result = {
            'summary': summary,
            'keywords': keywords,
            'segment_count': len(segments),
            'volume_stats': volume_stats,
            'sampled_segments': sampled  # Include sampled segments for citation linking
        }

        # Cache the result
        if self._cache and query_embedding is not None and summary:
            self._cache.put(query, query_embedding, result, cache_type='rag_summary')

        return result

    def _sample_diverse_segments(self, segments: List[Dict], max_samples: int) -> List[Dict]:
        """
        Sample diverse segments for RAG

        Strategies:
        - Top similarity scores
        - Diverse speakers
        - Temporal spread
        """
        if len(segments) <= max_samples:
            return segments

        # Strategy: Take top by similarity, but ensure diversity
        sorted_segments = sorted(segments, key=lambda x: x.get('similarity', 0), reverse=True)

        sampled = []
        seen_speakers = set()
        content_counts = Counter()

        for seg in sorted_segments:
            if len(sampled) >= max_samples:
                break

            # Prefer segments from different speakers and content
            speakers = seg.get('speaker_hashes', [])
            content_id = seg.get('content_id')

            # Add if: high similarity OR new speaker OR underrepresented content
            if (len(sampled) < max_samples // 2 or  # Always take top half
                not any(spk in seen_speakers for spk in speakers) or
                content_counts[content_id] < 2):

                sampled.append(seg)
                seen_speakers.update(speakers)
                content_counts[content_id] += 1

        logger.info(f"Sampled {len(sampled)} diverse segments from {len(segments)} total")
        return sampled

    def _build_rag_context(self, segments: List[Dict]) -> str:
        """Build context string from segments with clear numbering for citation"""
        context_parts = []

        for i, seg in enumerate(segments, 1):
            speakers = ', '.join(seg.get('speaker_hashes', ['Unknown']))
            text = seg['text'][:400]  # Slightly longer for better context
            content_id = seg.get('content_id_string') or seg.get('content_id', 'Unknown')
            context_parts.append(f"[Segment {i}] (Source: {content_id}, Speakers: {speakers})\n{text}\n")

        return '\n'.join(context_parts)

    def _extract_keywords(self, segments: List[Dict], top_n: int = 10) -> List[str]:
        """Extract top keywords from segments using frequency"""
        # Simple frequency-based keyword extraction
        # TODO: Could enhance with TF-IDF or LLM-based extraction

        words = []
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
                     'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
                     'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
                     'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
                     'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                     'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'now'}

        for seg in segments:
            text = seg['text'].lower()
            for word in text.split():
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 3 and word not in stopwords:
                    words.append(word)

        # Count and return top keywords
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(top_n)]

    def _calculate_volume_stats(self, segments: List[Dict]) -> Dict[str, Any]:
        """Calculate volume statistics"""
        try:
            # Convert Pydantic models to dicts if needed
            seg_dicts = []
            for seg in segments:
                if hasattr(seg, 'model_dump'):  # Pydantic v2
                    seg_dicts.append(seg.model_dump())
                elif hasattr(seg, 'dict'):  # Pydantic v1
                    seg_dicts.append(seg.dict())
                elif isinstance(seg, dict):
                    seg_dicts.append(seg)
                else:
                    seg_dicts.append(dict(seg))

            # Total duration
            total_duration = sum(seg.get('end_time', 0) - seg.get('start_time', 0) for seg in seg_dicts)

            # Unique content
            unique_content = len(set(seg.get('content_id', 0) for seg in seg_dicts))

            # Unique speakers
            all_speakers = set()
            for seg in seg_dicts:
                speakers = seg.get('speaker_hashes', [])
                if speakers:
                    all_speakers.update(speakers)

            return {
                'total_segments': len(seg_dicts),
                'total_duration_seconds': total_duration,
                'unique_content_items': unique_content,
                'unique_speakers': len(all_speakers)
            }
        except Exception as e:
            logger.error(f"Error calculating volume stats: {e}")
            return {
                'total_segments': len(segments),
                'total_duration_seconds': 0,
                'unique_content_items': 0,
                'unique_speakers': 0
            }

    async def call_grok_async(
        self,
        prompt: str,
        system_message: str = "You are an expert analyst synthesizing discourse analysis. Provide detailed, well-cited analysis.",
        model: str = "grok-2-1212",
        temperature: float = 0.3,
        max_tokens: int = 3000,
        timeout: int = 90
    ) -> str:
        """
        Async call to xAI Grok API for LLM generation.

        Args:
            prompt: User prompt
            system_message: System message for the LLM
            model: Model name (default: grok-2-1212)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds

        Returns:
            Generated text from the LLM

        Raises:
            RuntimeError: If API call fails
        """
        from ..config import settings

        if not settings.XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY not configured")

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.XAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_message
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    duration_ms = int((time.time() - start_time) * 1000)

                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"[{self.dashboard_id}] Grok API error: {response.status} {text[:200]}")
                        raise RuntimeError(f"LLM API returned {response.status}")

                    result = await response.json()

                    if 'choices' not in result or not result['choices']:
                        raise RuntimeError("No response from LLM API")

                    generated_text = result['choices'][0]['message']['content'].strip()

                    # Extract token usage and calculate cost
                    usage = result.get('usage', {})
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)

                    # Calculate cost: $0.20 per 1M input tokens, $0.50 per 1M output tokens
                    input_cost = (prompt_tokens / 1_000_000) * 0.20
                    output_cost = (completion_tokens / 1_000_000) * 0.50
                    total_cost = input_cost + output_cost

                    # Calculate requests per dollar
                    requests_per_dollar = 1.0 / total_cost if total_cost > 0 else 0

                    # Log to usage log
                    usage_logger.info(
                        f"model={model} | "
                        f"dashboard={self.dashboard_id} | "
                        f"prompt_tokens={prompt_tokens} | "
                        f"completion_tokens={completion_tokens} | "
                        f"total_tokens={total_tokens} | "
                        f"input_cost=${input_cost:.6f} | "
                        f"output_cost=${output_cost:.6f} | "
                        f"total_cost=${total_cost:.6f} | "
                        f"requests_per_dollar={requests_per_dollar:.1f} | "
                        f"duration_ms={duration_ms}"
                    )

                    logger.info(f"[{self.dashboard_id}] ✓ Async Grok call complete ({len(generated_text)} chars, {duration_ms}ms)")

                    return generated_text

            except asyncio.TimeoutError:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.error(f"[{self.dashboard_id}] Grok API timeout after {duration_ms}ms")
                raise RuntimeError("LLM API timeout")
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.error(f"[{self.dashboard_id}] Grok API error: {e}", exc_info=True)
                raise
