"""
Topic Labeling via LLM
========================

Generates human-readable topic labels and descriptions for clusters
using LLM analysis of representative segments.
"""

import json
import logging
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend.topic_labeler")


@dataclass
class TopicLabel:
    """Topic label with metadata"""
    cluster_id: int
    topic_name: str
    topic_description: str
    keywords: List[str]
    representative_quotes: List[str]
    confidence: float  # LLM confidence in labeling (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'cluster_id': self.cluster_id,
            'topic_name': self.topic_name,
            'topic_description': self.topic_description,
            'keywords': self.keywords,
            'representative_quotes': self.representative_quotes,
            'confidence': round(self.confidence, 2)
        }


class TopicLabeler:
    """
    Generates topic labels using LLM analysis of cluster samples.

    Uses xAI Grok to analyze representative segments and generate:
    - Concise topic name
    - Brief description
    - Keywords
    - Representative quotes
    """

    def __init__(self, llm_service, cache_dir: Optional[Path] = None):
        """
        Initialize TopicLabeler

        Args:
            llm_service: LLMService instance (for embeddings/caching)
            cache_dir: Optional cache directory override
        """
        self.llm_service = llm_service

        # Use LLM service's cache if available
        self._cache = self.llm_service._cache if hasattr(self.llm_service, '_cache') else None

        logger.info("TopicLabeler initialized")

    def label_topic(
        self,
        cluster_id: int,
        text_samples: List[Dict[str, Any]],
        cluster_metrics: Optional[Dict[str, Any]] = None
    ) -> TopicLabel:
        """
        Generate label for a single topic cluster

        Args:
            cluster_id: Cluster ID
            text_samples: List of dicts with {segment_id, text, channel, date, title}
            cluster_metrics: Optional cluster metrics (size, breadth, etc.)

        Returns:
            TopicLabel with name, description, keywords, quotes
        """
        logger.info(f"Labeling topic cluster {cluster_id} ({len(text_samples)} samples)")

        # Check cache first
        cache_key = self._make_cache_key(cluster_id, text_samples)
        if self._cache:
            cached_label = self._cache.get(cache_key, query_embedding=None, cache_type='topic_labels')
            if cached_label:
                logger.info(f"[CACHE HIT] Topic {cluster_id} label from cache")
                return TopicLabel(**cached_label)

        # Generate label via LLM
        start_time = time.time()

        try:
            # Prepare context for LLM
            context = self._prepare_context(text_samples, cluster_metrics)

            # Call Grok API
            response = self._call_grok_api(context)

            # Parse response
            label = self._parse_response(cluster_id, response)

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"✓ Topic {cluster_id} labeled: '{label.topic_name}' ({duration_ms:.0f}ms)")

            # Cache result
            if self._cache:
                self._cache.put(cache_key, query_embedding=None, value=label.to_dict(), cache_type='topic_labels')

            return label

        except Exception as e:
            logger.error(f"Error labeling topic {cluster_id}: {e}", exc_info=True)
            # Return fallback label
            return TopicLabel(
                cluster_id=cluster_id,
                topic_name=f"Topic {cluster_id}",
                topic_description="Unable to generate description",
                keywords=[],
                representative_quotes=[],
                confidence=0.0
            )

    def label_topics_batch(
        self,
        clusters_with_samples: List[Dict[str, Any]]
    ) -> List[TopicLabel]:
        """
        Label multiple topics in batch

        Args:
            clusters_with_samples: List of dicts with {cluster_id, samples, metrics}

        Returns:
            List of TopicLabels
        """
        labels = []

        for item in clusters_with_samples:
            label = self.label_topic(
                cluster_id=item['cluster_id'],
                text_samples=item['samples'],
                cluster_metrics=item.get('metrics')
            )
            labels.append(label)

        return labels

    def _make_cache_key(self, cluster_id: int, text_samples: List[Dict]) -> str:
        """Create cache key from cluster content"""
        # Use first 3 sample texts as fingerprint
        fingerprint = '|'.join(
            s['text'][:100] for s in text_samples[:3]
        )
        return f"topic_{cluster_id}_{hash(fingerprint)}"

    def _prepare_context(
        self,
        text_samples: List[Dict[str, Any]],
        cluster_metrics: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare context string for LLM"""
        # Build context with samples
        context_parts = []

        # Add metrics if available
        if cluster_metrics:
            context_parts.append("=== CLUSTER METRICS ===")
            context_parts.append(f"Size: {cluster_metrics.get('size', 'unknown')} segments")
            context_parts.append(f"Channels: {cluster_metrics.get('breadth', 0):.2f} (diversity score)")
            context_parts.append(f"Intensity: {cluster_metrics.get('intensity', 0):.2f} (discussion concentration)")
            context_parts.append("")

        # Add sample texts
        context_parts.append("=== REPRESENTATIVE SEGMENTS ===")
        for i, sample in enumerate(text_samples[:10], 1):  # Max 10 samples
            context_parts.append(f"\n--- Sample {i} ---")
            context_parts.append(f"Channel: {sample.get('channel', 'unknown')}")
            context_parts.append(f"Date: {sample.get('date', 'unknown')}")
            if sample.get('title'):
                context_parts.append(f"Title: {sample['title']}")
            context_parts.append(f"Text: {sample['text']}")

        return '\n'.join(context_parts)

    def _call_grok_api(self, context: str) -> Dict[str, Any]:
        """Call xAI Grok API to generate topic label"""
        from ...config import settings

        if not settings.XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY not configured")

        prompt = f"""You are analyzing political discourse in Canadian far-right media. Based on the provided text segments from a topic cluster, generate a concise topic label and description.

{context}

Analyze these segments and provide:
1. **Topic Name**: 3-6 word concise label capturing the main theme
2. **Description**: 1-2 sentence description of what this topic covers
3. **Keywords**: 5-8 key terms that characterize this topic
4. **Representative Quotes**: 2-3 short quotes (1 sentence each) that exemplify the topic

Guidelines:
- Be precise and specific to the actual content
- Use neutral, analytical language
- Capture the primary theme, not every detail
- Focus on what makes this cluster distinct

Respond in JSON format:
{{
    "topic_name": "Concise Topic Label",
    "topic_description": "Brief description of the topic theme and key points discussed.",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "representative_quotes": [
        "First representative quote from the segments",
        "Second representative quote from the segments",
        "Third representative quote from the segments"
    ],
    "confidence": 0.85
}}

The confidence score (0-1) reflects how coherent and well-defined the topic appears."""

        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-2-1212",
                "messages": [
                    {"role": "system", "content": "You are an expert analyst of political discourse and media content."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            },
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Grok API error: {response.status_code} {response.text}")
            raise RuntimeError(f"Grok API returned {response.status_code}")

        result = response.json()

        if 'choices' not in result or not result['choices']:
            raise RuntimeError("No response from Grok API")

        content = result['choices'][0]['message']['content']

        # Parse JSON from response (handle markdown code blocks)
        if '```json' in content:
            json_str = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            json_str = content.split('```')[1].split('```')[0].strip()
        else:
            json_str = content.strip()

        return json.loads(json_str)

    def _parse_response(self, cluster_id: int, response: Dict[str, Any]) -> TopicLabel:
        """Parse LLM response into TopicLabel"""
        return TopicLabel(
            cluster_id=cluster_id,
            topic_name=response.get('topic_name', f'Topic {cluster_id}'),
            topic_description=response.get('topic_description', ''),
            keywords=response.get('keywords', []),
            representative_quotes=response.get('representative_quotes', []),
            confidence=float(response.get('confidence', 0.5))
        )
