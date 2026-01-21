"""
Theme Summarizer
================

Generates first-pass theme summaries with embedded citations.
Uses LLM to create 7-8 paragraph summaries with citation tracking.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from ...utils.backend_logger import get_logger
logger = get_logger("theme_summarizer")

from .citation_manager import CitationManager, Citation
from src.database.models import EmbeddingSegment as SampledSegment


@dataclass
class ThemeSummary:
    """Theme summary with citations and metadata"""
    group_id: str
    theme_id: str
    theme_name: str
    summary_text: str
    citations: Dict[str, Dict[str, Any]]  # citation_id -> segment metadata
    segment_count: int
    invalid_citations: List[str]
    generation_time_ms: int
    model: str = "grok-2-1212"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'group_id': self.group_id,
            'theme_id': self.theme_id,
            'theme_name': self.theme_name,
            'summary_text': self.summary_text,
            'citations': self.citations,
            'segment_count': self.segment_count,
            'invalid_citations': self.invalid_citations,
            'generation_time_ms': self.generation_time_ms,
            'model': self.model
        }


class ThemeSummarizer:
    """
    Generates theme summaries with embedded citations.

    First-pass summarization: analyzes ~20 representative segments
    per theme and generates a 7-8 paragraph summary with citations.
    """

    def __init__(
        self,
        llm_service,
        citation_manager: CitationManager,
        max_context_segments: int = 20
    ):
        """
        Initialize theme summarizer.

        Args:
            llm_service: LLMService instance (for API calls)
            citation_manager: CitationManager instance
            max_context_segments: Max segments to include in LLM context
        """
        self.llm_service = llm_service
        self.citation_manager = citation_manager
        self.max_context_segments = max_context_segments

        logger.info(f"ThemeSummarizer initialized (max_context={max_context_segments})")

    def generate_summary(
        self,
        group_id: str,
        theme_id: str,
        theme_name: str,
        sampled_segments: List[SampledSegment],
        additional_context: Optional[str] = None
    ) -> ThemeSummary:
        """
        Generate theme summary with citations.

        Args:
            group_id: Group identifier
            theme_id: Theme identifier
            theme_name: Human-readable theme name
            sampled_segments: List of SampledSegment objects
            additional_context: Optional context about the theme

        Returns:
            ThemeSummary with embedded citations
        """
        logger.info(f"Generating summary for theme '{theme_name}' (group='{group_id}', {len(sampled_segments)} segments)")

        start_time = time.time()

        # Build citation map and context
        citation_map = {}
        context_parts = []

        for seg in sampled_segments[:self.max_context_segments]:
            # Generate citation ID
            citation_id = self.citation_manager.generate(
                group_id=group_id,
                theme_id=theme_id,
                segment_id=seg.segment_id
            )

            # Store in map
            citation_map[citation_id] = seg.to_dict()

            # Build context entry
            context_parts.append(
                f"{citation_id}\n"
                f"Channel: {seg.channel_name or seg.channel_url}\n"
                f"Date: {seg.publish_date.strftime('%Y-%m-%d') if seg.publish_date else 'Unknown'}\n"
                f"Text: {seg.text}\n"
            )

        context = "\n---\n".join(context_parts)

        # Build prompt
        prompt = self._build_prompt(
            theme_name=theme_name,
            context=context,
            additional_context=additional_context
        )

        # Call LLM API
        try:
            summary_text = self._call_llm(prompt)

            # Extract and validate citations
            used_citations = self.citation_manager.extract_all(summary_text)
            invalid_citations = [
                c.raw for c in used_citations
                if c.raw not in citation_map
            ]

            if invalid_citations:
                logger.warning(
                    f"Theme '{theme_name}': {len(invalid_citations)} invalid citations found: "
                    f"{invalid_citations[:5]}"
                )

            generation_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"✓ Theme summary generated: '{theme_name}' "
                f"({len(summary_text)} chars, {len(used_citations)} citations, {generation_time}ms)"
            )

            return ThemeSummary(
                group_id=group_id,
                theme_id=theme_id,
                theme_name=theme_name,
                summary_text=summary_text,
                citations=citation_map,
                segment_count=len(sampled_segments),
                invalid_citations=invalid_citations,
                generation_time_ms=generation_time
            )

        except Exception as e:
            logger.error(f"Error generating theme summary for '{theme_name}': {e}", exc_info=True)
            raise

    def _build_prompt(
        self,
        theme_name: str,
        context: str,
        additional_context: Optional[str]
    ) -> str:
        """Build LLM prompt for theme summarization"""

        context_section = f"\n\nADDITIONAL CONTEXT:\n{additional_context}\n" if additional_context else ""

        prompt = f"""You are analyzing discourse patterns in online media. Generate a comprehensive summary for the theme: "{theme_name}".

Below are representative text segments from this theme cluster. Each segment has a citation ID.

{context}{context_section}

TASK: Generate a 7-8 paragraph summary that:
1. Describes the main discourse patterns and arguments within this theme
2. Identifies key narratives, frames, and rhetorical strategies
3. Notes any contradictions or tensions within the theme
4. Highlights the most prominent voices/channels
5. Observes temporal patterns if evident

CRITICAL CITATION RULES:
- After EVERY factual claim or quote, cite the source using its citation ID
- Format: "Content creators argue X {{citation}}."
- Multiple sources: "This narrative appears frequently {{citation1}} {{citation2}}."
- ONLY use citation IDs provided above - do NOT create new ones
- Every paragraph should have multiple citations

STYLE:
- Objective, analytical tone
- Focus on what is being said, not your judgment of it
- Be specific: quote exact phrases when helpful
- Keep it dense with information

Generate the summary now:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API for summary generation.

        Args:
            prompt: Formatted prompt

        Returns:
            Summary text with embedded citations
        """
        import requests
        from ...config import settings

        if not settings.XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY not configured")

        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-2-1212",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert analyst of political discourse and media content. Provide detailed, well-cited summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            },
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} {response.text}")
            raise RuntimeError(f"LLM API returned {response.status_code}")

        result = response.json()

        if 'choices' not in result or not result['choices']:
            raise RuntimeError("No response from LLM API")

        summary_text = result['choices'][0]['message']['content'].strip()

        return summary_text

    def generate_batch(
        self,
        group_id: str,
        themes_with_segments: List[Dict[str, Any]]
    ) -> List[ThemeSummary]:
        """
        Generate summaries for multiple themes in batch.

        Args:
            group_id: Group identifier
            themes_with_segments: List of dicts with:
                {
                    'theme_id': str,
                    'theme_name': str,
                    'segments': List[SampledSegment],
                    'context': Optional[str]
                }

        Returns:
            List of ThemeSummary objects
        """
        summaries = []

        logger.info(f"Starting batch summary generation for {len(themes_with_segments)} themes in group '{group_id}'")

        for i, theme_data in enumerate(themes_with_segments, 1):
            logger.info(f"[{i}/{len(themes_with_segments)}] Processing theme: {theme_data['theme_name']}")

            try:
                summary = self.generate_summary(
                    group_id=group_id,
                    theme_id=theme_data['theme_id'],
                    theme_name=theme_data['theme_name'],
                    sampled_segments=theme_data['segments'],
                    additional_context=theme_data.get('context')
                )
                summaries.append(summary)

            except Exception as e:
                logger.error(f"Failed to generate summary for theme '{theme_data['theme_name']}': {e}")
                # Continue with other themes
                continue

        logger.info(f"✓ Batch complete: {len(summaries)}/{len(themes_with_segments)} summaries generated")

        return summaries

    async def generate_summary_async(
        self,
        group_id: str,
        theme_id: str,
        theme_name: str,
        sampled_segments: List[SampledSegment],
        additional_context: Optional[str] = None
    ) -> ThemeSummary:
        """
        Async version: Generate theme summary with citations.

        Args:
            group_id: Group identifier
            theme_id: Theme identifier
            theme_name: Human-readable theme name
            sampled_segments: List of SampledSegment objects
            additional_context: Optional context about the theme

        Returns:
            ThemeSummary with embedded citations
        """
        logger.info(f"Generating summary (async) for theme '{theme_name}' (group='{group_id}', {len(sampled_segments)} segments)")

        start_time = time.time()

        # Build citation map and context (same as sync version)
        citation_map = {}
        context_parts = []

        for seg in sampled_segments[:self.max_context_segments]:
            citation_id = self.citation_manager.generate(
                group_id=group_id,
                theme_id=theme_id,
                segment_id=seg.segment_id
            )
            citation_map[citation_id] = seg.to_dict()
            context_parts.append(
                f"{citation_id}\n"
                f"Channel: {seg.channel_name or seg.channel_url}\n"
                f"Date: {seg.publish_date.strftime('%Y-%m-%d') if seg.publish_date else 'Unknown'}\n"
                f"Text: {seg.text}\n"
            )

        context = "\n---\n".join(context_parts)
        prompt = self._build_prompt(
            theme_name=theme_name,
            context=context,
            additional_context=additional_context
        )

        try:
            # Async LLM call
            summary_text = await self.llm_service.call_grok_async(
                prompt=prompt,
                system_message="You are an expert analyst of political discourse and media content. Provide detailed, well-cited summaries.",
                model="grok-2-1212",
                temperature=0.3,
                max_tokens=2000,
                timeout=60
            )

            # Extract and validate citations
            used_citations = self.citation_manager.extract_all(summary_text)
            invalid_citations = [
                c.raw for c in used_citations
                if c.raw not in citation_map
            ]

            if invalid_citations:
                logger.warning(
                    f"Theme '{theme_name}': {len(invalid_citations)} invalid citations found: "
                    f"{invalid_citations[:5]}"
                )

            generation_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"✓ Theme summary generated (async): '{theme_name}' "
                f"({len(summary_text)} chars, {len(used_citations)} citations, {generation_time}ms)"
            )

            return ThemeSummary(
                group_id=group_id,
                theme_id=theme_id,
                theme_name=theme_name,
                summary_text=summary_text,
                citations=citation_map,
                segment_count=len(sampled_segments),
                invalid_citations=invalid_citations,
                generation_time_ms=generation_time
            )

        except Exception as e:
            logger.error(f"Error generating theme summary (async) for '{theme_name}': {e}", exc_info=True)
            raise

    def validate_summary_citations(
        self,
        summary: ThemeSummary
    ) -> Dict[str, Any]:
        """
        Validate all citations in a summary.

        Args:
            summary: ThemeSummary to validate

        Returns:
            Validation report with statistics
        """
        used_citations = self.citation_manager.extract_all(summary.summary_text)

        valid_citations = [c for c in used_citations if c.raw in summary.citations]
        invalid_citations = [c for c in used_citations if c.raw not in summary.citations]

        # Citation frequency
        frequency = self.citation_manager.count_citation_frequency(summary.summary_text)

        # Segment coverage (how many segments were actually cited)
        cited_segment_ids = set()
        for citation in valid_citations:
            parsed = self.citation_manager.parse(citation.raw)
            if parsed:
                cited_segment_ids.add(parsed.segment_id)

        report = {
            'total_citations': len(used_citations),
            'valid_citations': len(valid_citations),
            'invalid_citations': len(invalid_citations),
            'invalid_citation_list': [c.raw for c in invalid_citations],
            'unique_segments_cited': len(cited_segment_ids),
            'total_segments_available': summary.segment_count,
            'segment_coverage_rate': len(cited_segment_ids) / summary.segment_count if summary.segment_count > 0 else 0,
            'citation_frequency': frequency,
            'avg_citations_per_segment': len(used_citations) / len(cited_segment_ids) if cited_segment_ids else 0
        }

        logger.debug(f"Citation validation for '{summary.theme_name}': {report}")

        return report
