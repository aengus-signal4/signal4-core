"""
Meta Summarizer
===============

Generates second-pass meta-summaries by synthesizing across theme summaries.
Preserves and reuses citations from theme summaries.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from ...utils.backend_logger import get_logger
logger = get_logger("meta_summarizer")

from .citation_manager import CitationManager
from .theme_summarizer import ThemeSummary


@dataclass
class MetaSummary:
    """Cross-theme synthesis with preserved citations"""
    synthesis_id: str
    group_ids: List[str]
    theme_ids: List[str]
    synthesis_text: str
    all_citations: Dict[str, Dict[str, Any]]  # Aggregated from all themes
    invalid_citations: List[str]
    theme_count: int
    generation_time_ms: int
    synthesis_type: str = "cross_theme"  # cross_theme, cross_group, temporal
    model: str = "grok-2-1212"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'synthesis_id': self.synthesis_id,
            'group_ids': self.group_ids,
            'theme_ids': self.theme_ids,
            'synthesis_text': self.synthesis_text,
            'citation_count': len(self.all_citations),
            'invalid_citations': self.invalid_citations,
            'theme_count': self.theme_count,
            'generation_time_ms': self.generation_time_ms,
            'synthesis_type': self.synthesis_type,
            'model': self.model
        }


class MetaSummarizer:
    """
    Generates meta-summaries by synthesizing across theme summaries.

    Second-pass summarization: identifies patterns, contradictions,
    and cross-theme narratives while preserving citation integrity.
    """

    def __init__(
        self,
        llm_service,
        citation_manager: CitationManager
    ):
        """
        Initialize meta summarizer.

        Args:
            llm_service: LLMService instance
            citation_manager: CitationManager instance
        """
        self.llm_service = llm_service
        self.citation_manager = citation_manager

        logger.info("MetaSummarizer initialized")

    def generate_synthesis(
        self,
        theme_summaries: List[ThemeSummary],
        synthesis_type: str = "cross_theme",
        synthesis_id: Optional[str] = None
    ) -> MetaSummary:
        """
        Generate meta-summary across theme summaries.

        Args:
            theme_summaries: List of ThemeSummary objects
            synthesis_type: Type of synthesis:
                - "cross_theme": Identify patterns across themes
                - "cross_group": Compare different groups
                - "temporal": Track evolution over time
            synthesis_id: Optional identifier for this synthesis

        Returns:
            MetaSummary with cross-theme synthesis
        """
        if not theme_summaries:
            raise ValueError("Cannot generate synthesis with no theme summaries")

        logger.info(
            f"Generating {synthesis_type} synthesis across {len(theme_summaries)} themes"
        )

        start_time = time.time()

        # Aggregate all citations
        all_citations = {}
        for theme in theme_summaries:
            all_citations.update(theme.citations)

        logger.info(f"Aggregated {len(all_citations)} total citations from themes")

        # Build context from theme summaries
        context = self._build_context(theme_summaries)

        # Build prompt based on synthesis type
        prompt = self._build_prompt(synthesis_type, context, len(theme_summaries))

        # Call LLM API
        try:
            synthesis_text = self._call_llm(prompt)

            # Extract and validate citations
            used_citations = self.citation_manager.extract_all(synthesis_text)
            invalid_citations = [
                c.raw for c in used_citations
                if c.raw not in all_citations
            ]

            if invalid_citations:
                logger.warning(
                    f"Synthesis generated {len(invalid_citations)} invalid citations: "
                    f"{invalid_citations[:5]}"
                )

            generation_time = int((time.time() - start_time) * 1000)

            # Extract group and theme IDs
            group_ids = list(set(t.group_id for t in theme_summaries))
            theme_ids = [t.theme_id for t in theme_summaries]

            # Generate synthesis ID if not provided
            if not synthesis_id:
                import hashlib
                synthesis_id = hashlib.md5(
                    f"{'_'.join(group_ids)}:{'_'.join(theme_ids)}".encode()
                ).hexdigest()[:12]

            logger.info(
                f"✓ Meta-summary generated: {synthesis_type} "
                f"({len(synthesis_text)} chars, {len(used_citations)} citations, {generation_time}ms)"
            )

            return MetaSummary(
                synthesis_id=synthesis_id,
                group_ids=group_ids,
                theme_ids=theme_ids,
                synthesis_text=synthesis_text,
                all_citations=all_citations,
                invalid_citations=invalid_citations,
                theme_count=len(theme_summaries),
                generation_time_ms=generation_time,
                synthesis_type=synthesis_type
            )

        except Exception as e:
            logger.error(f"Error generating meta-summary: {e}", exc_info=True)
            raise

    def _build_context(self, theme_summaries: List[ThemeSummary]) -> str:
        """Build context from theme summaries"""
        context_parts = []

        for i, theme in enumerate(theme_summaries, 1):
            context_parts.append(
                f"## THEME {i}: {theme.theme_name}\n"
                f"Group: {theme.group_id}\n"
                f"Segments analyzed: {theme.segment_count}\n\n"
                f"{theme.summary_text}\n"
            )

        return "\n".join(context_parts)

    def _build_prompt(
        self,
        synthesis_type: str,
        context: str,
        theme_count: int
    ) -> str:
        """Build LLM prompt for meta-summarization"""

        if synthesis_type == "cross_theme":
            task_description = """TASK: Generate a 10-paragraph cross-theme synthesis that:
1. Identifies common narratives and frames appearing across multiple themes
2. Highlights contradictions and tensions between themes
3. Notes which themes overlap vs. which are distinct
4. Identifies the most cited sources and their influence patterns
5. Describes the overall discourse landscape across all themes
6. Observes any dominant ideological patterns
7. Notes gaps or silences in the discourse"""

        elif synthesis_type == "cross_group":
            task_description = """TASK: Generate a 10-paragraph cross-group comparison that:
1. Identifies how different groups frame similar themes differently
2. Notes shared narratives vs. group-specific narratives
3. Highlights ideological differences between groups
4. Observes language and rhetorical differences
5. Identifies which groups cite which sources
6. Notes areas of agreement and disagreement across groups"""

        elif synthesis_type == "group_internal":
            task_description = """TASK: Generate a 10-paragraph group-level synthesis that:
1. Describes the overall discourse landscape within this group
2. Identifies dominant narratives and frames across themes
3. Notes connections and relationships between themes
4. Highlights the most prominent voices and sources
5. Describes the ideological patterns and rhetorical strategies
6. Identifies any contradictions or tensions within the group
7. Provides a coherent picture of this group's discourse"""

        elif synthesis_type == "temporal":
            task_description = """TASK: Generate a 10-paragraph temporal analysis that:
1. Tracks how themes have evolved over time
2. Identifies emerging vs. declining narratives
3. Notes shifts in language and framing
4. Highlights events or moments that changed the discourse
5. Observes patterns of attention (what gets discussed when)
6. Identifies long-term trends vs. temporary spikes"""

        else:
            task_description = """TASK: Generate a 10-paragraph synthesis that:
1. Identifies key patterns across all theme summaries
2. Highlights important connections and contradictions
3. Provides an overview of the discourse landscape"""

        prompt = f"""You are synthesizing discourse analysis across {theme_count} theme summaries.

Each theme summary below includes embedded citations in the format [G{{...}}-T{{...}}-S{{...}}].

{context}

{task_description}

CRITICAL CITATION RULES:
- ONLY reuse citations that already appear in the theme summaries above
- Do NOT create new citation IDs
- When referencing a pattern across multiple themes, include citations from each: "This narrative appears {{citation1}} {{citation2}} {{citation3}}"
- Cite specific examples to support every claim

STYLE:
- Objective, analytical, scholarly tone
- Focus on patterns and structures of discourse
- Be specific with examples
- Dense with information and citations

Generate the synthesis now:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API for synthesis generation.

        Args:
            prompt: Formatted prompt

        Returns:
            Synthesis text with embedded citations
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
                        "content": "You are an expert analyst synthesizing discourse analysis. Provide detailed, well-cited meta-analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 3000
            },
            timeout=90
        )

        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} {response.text}")
            raise RuntimeError(f"LLM API returned {response.status_code}")

        result = response.json()

        if 'choices' not in result or not result['choices']:
            raise RuntimeError("No response from LLM API")

        synthesis_text = result['choices'][0]['message']['content'].strip()

        return synthesis_text

    def validate_synthesis_citations(
        self,
        synthesis: MetaSummary
    ) -> Dict[str, Any]:
        """
        Validate all citations in synthesis.

        Args:
            synthesis: MetaSummary to validate

        Returns:
            Validation report
        """
        used_citations = self.citation_manager.extract_all(synthesis.synthesis_text)

        valid_citations = [c for c in used_citations if c.raw in synthesis.all_citations]
        invalid_citations = [c for c in used_citations if c.raw not in synthesis.all_citations]

        # Group citations by theme
        citations_by_theme = self.citation_manager.group_citations_by_theme(valid_citations)

        # Group citations by group
        citations_by_group = self.citation_manager.group_citations_by_group(valid_citations)

        report = {
            'total_citations': len(used_citations),
            'valid_citations': len(valid_citations),
            'invalid_citations': len(invalid_citations),
            'invalid_citation_list': [c.raw for c in invalid_citations],
            'unique_citations': len(set(c.raw for c in valid_citations)),
            'themes_cited': len(citations_by_theme),
            'groups_cited': len(citations_by_group),
            'citations_per_theme': {
                theme_id: len(cites)
                for theme_id, cites in citations_by_theme.items()
            },
            'citations_per_group': {
                group_id: len(cites)
                for group_id, cites in citations_by_group.items()
            }
        }

        logger.debug(f"Synthesis validation: {report}")

        return report

    async def generate_synthesis_async(
        self,
        theme_summaries: List[ThemeSummary],
        synthesis_type: str = "cross_theme",
        synthesis_id: Optional[str] = None,
        include_citations: bool = True
    ) -> MetaSummary:
        """
        Async version: Generate meta-summary across theme summaries.

        Args:
            theme_summaries: List of ThemeSummary objects
            synthesis_type: Type of synthesis:
                - "cross_theme": Identify patterns across themes
                - "cross_group": Compare different groups
                - "temporal": Track evolution over time
                - "group_internal": Synthesize themes within a single group
            synthesis_id: Optional identifier for this synthesis
            include_citations: Whether to include citation reuse (False for high-level comparison)

        Returns:
            MetaSummary with cross-theme synthesis
        """
        if not theme_summaries:
            raise ValueError("Cannot generate synthesis with no theme summaries")

        logger.info(
            f"Generating {synthesis_type} synthesis (async) across {len(theme_summaries)} themes"
        )

        start_time = time.time()

        # Aggregate all citations (if needed)
        all_citations = {}
        if include_citations:
            for theme in theme_summaries:
                all_citations.update(theme.citations)
            logger.info(f"Aggregated {len(all_citations)} total citations from themes")

        # Build context from theme summaries
        context = self._build_context(theme_summaries)

        # Build prompt based on synthesis type
        prompt = self._build_prompt(synthesis_type, context, len(theme_summaries))

        try:
            # Async LLM call
            synthesis_text = await self.llm_service.call_grok_async(
                prompt=prompt,
                system_message="You are an expert analyst synthesizing discourse analysis. Provide detailed, well-cited meta-analysis.",
                model="grok-2-1212",
                temperature=0.3,
                max_tokens=3000,
                timeout=90
            )

            # Extract and validate citations (if applicable)
            invalid_citations = []
            if include_citations:
                used_citations = self.citation_manager.extract_all(synthesis_text)
                invalid_citations = [
                    c.raw for c in used_citations
                    if c.raw not in all_citations
                ]

                if invalid_citations:
                    logger.warning(
                        f"Synthesis generated {len(invalid_citations)} invalid citations: "
                        f"{invalid_citations[:5]}"
                    )

            generation_time = int((time.time() - start_time) * 1000)

            # Extract group and theme IDs
            group_ids = list(set(t.group_id for t in theme_summaries))
            theme_ids = [t.theme_id for t in theme_summaries]

            # Generate synthesis ID if not provided
            if not synthesis_id:
                import hashlib
                synthesis_id = hashlib.md5(
                    f"{'_'.join(group_ids)}:{'_'.join(theme_ids)}".encode()
                ).hexdigest()[:12]

            logger.info(
                f"✓ Meta-summary generated (async): {synthesis_type} "
                f"({len(synthesis_text)} chars, {generation_time}ms)"
            )

            return MetaSummary(
                synthesis_id=synthesis_id,
                group_ids=group_ids,
                theme_ids=theme_ids,
                synthesis_text=synthesis_text,
                all_citations=all_citations,
                invalid_citations=invalid_citations,
                theme_count=len(theme_summaries),
                generation_time_ms=generation_time,
                synthesis_type=synthesis_type
            )

        except Exception as e:
            logger.error(f"Error generating meta-summary (async): {e}", exc_info=True)
            raise

    def generate_comparative_synthesis(
        self,
        group_summaries: Dict[str, List[ThemeSummary]]
    ) -> MetaSummary:
        """
        Generate synthesis comparing multiple groups.

        Args:
            group_summaries: Dict mapping group_id -> List[ThemeSummary]

        Returns:
            MetaSummary with cross-group comparison
        """
        # Flatten all theme summaries
        all_themes = []
        for summaries in group_summaries.values():
            all_themes.extend(summaries)

        logger.info(
            f"Generating comparative synthesis across {len(group_summaries)} groups "
            f"({len(all_themes)} total themes)"
        )

        return self.generate_synthesis(
            theme_summaries=all_themes,
            synthesis_type="cross_group"
        )
