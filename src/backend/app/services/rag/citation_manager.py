"""
Citation Manager
================

Manages citation ID generation, parsing, and validation for hierarchical summaries.
Supports flexible citation formats with group, theme, and segment identifiers.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass

from ...utils.backend_logger import get_logger
logger = get_logger("citation_manager")


@dataclass
class Citation:
    """Parsed citation with identifiers"""
    raw: str
    group_id: str
    theme_id: str
    segment_id: int

    def __str__(self) -> str:
        return self.raw


class CitationManager:
    """
    Manages citation IDs for hierarchical summarization.

    Default format: [G{group_id}-T{theme_id}-S{segment_id}]
    Example: [G_religious_fr-T5-S12847]

    Supports custom formats via format string.
    """

    def __init__(self, citation_format: str = "[G{group_id}-T{theme_id}-S{segment_id}]"):
        """
        Initialize citation manager.

        Args:
            citation_format: Format string with placeholders {group_id}, {theme_id}, {segment_id}
        """
        self.format = citation_format

        # Build regex pattern from format string
        self.pattern = self._build_pattern(citation_format)

        logger.info(f"CitationManager initialized with format: {citation_format}")

    def _build_pattern(self, format_string: str) -> re.Pattern:
        """
        Convert format string to regex pattern.

        Example:
            Format: "[G{group_id}-T{theme_id}-S{segment_id}]"
            Regex:  r"\\[G([^-]+)-T([^-]+)-S(\\d+)\\]"
        """
        # Escape special regex characters except our placeholders
        pattern = re.escape(format_string)

        # Replace placeholders with capture groups
        pattern = pattern.replace(r'\{group_id\}', r'([^-\]]+)')  # Capture until - or ]
        pattern = pattern.replace(r'\{theme_id\}', r'([^-\]]+)')  # Capture until - or ]
        pattern = pattern.replace(r'\{segment_id\}', r'(\d+)')    # Capture digits

        logger.debug(f"Built citation pattern: {pattern}")
        return re.compile(pattern)

    def generate(self, group_id: str, theme_id: str, segment_id: int) -> str:
        """
        Generate citation ID.

        Args:
            group_id: Group identifier (e.g., "religious_fr", "masculinist_en")
            theme_id: Theme identifier (e.g., "T5", "gender_schools")
            segment_id: Segment ID from database

        Returns:
            Citation string (e.g., "[G_religious_fr-T5-S12847]")
        """
        citation = self.format.format(
            group_id=group_id,
            theme_id=theme_id,
            segment_id=segment_id
        )
        logger.debug(f"Generated citation: {citation}")
        return citation

    def parse(self, citation: str) -> Optional[Citation]:
        """
        Parse citation ID into components.

        Args:
            citation: Citation string (e.g., "[G_religious_fr-T5-S12847]")

        Returns:
            Citation object with parsed components, or None if invalid
        """
        match = self.pattern.match(citation)
        if not match:
            logger.warning(f"Failed to parse citation: {citation}")
            return None

        try:
            group_id, theme_id, segment_id = match.groups()
            return Citation(
                raw=citation,
                group_id=group_id,
                theme_id=theme_id,
                segment_id=int(segment_id)
            )
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing citation '{citation}': {e}")
            return None

    def extract_all(self, text: str) -> List[Citation]:
        """
        Extract all citations from text.

        Args:
            text: Text containing citations

        Returns:
            List of parsed Citation objects
        """
        citations = []
        for match in self.pattern.finditer(text):
            try:
                group_id, theme_id, segment_id = match.groups()
                citations.append(Citation(
                    raw=match.group(0),
                    group_id=group_id,
                    theme_id=theme_id,
                    segment_id=int(segment_id)
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid citation match: {match.group(0)} - {e}")

        logger.debug(f"Extracted {len(citations)} citations from text")
        return citations

    def validate(self, citation: str, valid_segment_ids: Set[int]) -> bool:
        """
        Validate citation against known segment IDs.

        Args:
            citation: Citation string
            valid_segment_ids: Set of valid segment IDs

        Returns:
            True if citation is valid and segment exists
        """
        parsed = self.parse(citation)
        if parsed is None:
            return False

        return parsed.segment_id in valid_segment_ids

    def validate_all(self, citations: List[Citation], valid_segment_ids: Set[int]) -> Dict[str, List[Citation]]:
        """
        Validate multiple citations and categorize results.

        Args:
            citations: List of Citation objects
            valid_segment_ids: Set of valid segment IDs

        Returns:
            Dict with keys:
                - 'valid': List of valid citations
                - 'invalid_format': Citations that failed to parse
                - 'invalid_segment': Citations with non-existent segment IDs
        """
        results = {
            'valid': [],
            'invalid_format': [],
            'invalid_segment': []
        }

        for citation in citations:
            if citation.segment_id in valid_segment_ids:
                results['valid'].append(citation)
            else:
                results['invalid_segment'].append(citation)

        logger.info(
            f"Validation results: {len(results['valid'])} valid, "
            f"{len(results['invalid_segment'])} invalid segments"
        )

        return results

    def build_citation_map(
        self,
        group_id: str,
        theme_segments: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        Build map of all possible citations to segment metadata.

        Args:
            group_id: Group identifier
            theme_segments: Dict mapping theme_id -> list of segment dicts

        Returns:
            Dict mapping citation_id -> segment metadata

        Example:
            {
                "[G_religious_fr-T5-S12847]": {
                    "segment_id": 12847,
                    "text": "...",
                    "channel_name": "...",
                    ...
                }
            }
        """
        citation_map = {}

        for theme_id, segments in theme_segments.items():
            for segment in segments:
                citation = self.generate(
                    group_id=group_id,
                    theme_id=theme_id,
                    segment_id=segment['segment_id']
                )
                citation_map[citation] = segment

        logger.info(f"Built citation map with {len(citation_map)} entries for group '{group_id}'")
        return citation_map

    def get_segment_id_from_citation(self, citation: str) -> Optional[int]:
        """
        Extract segment ID from citation string.

        Args:
            citation: Citation string

        Returns:
            Segment ID or None if invalid
        """
        parsed = self.parse(citation)
        return parsed.segment_id if parsed else None

    def group_citations_by_theme(self, citations: List[Citation]) -> Dict[str, List[Citation]]:
        """
        Group citations by theme ID.

        Args:
            citations: List of Citation objects

        Returns:
            Dict mapping theme_id -> list of citations
        """
        grouped = {}
        for citation in citations:
            if citation.theme_id not in grouped:
                grouped[citation.theme_id] = []
            grouped[citation.theme_id].append(citation)

        return grouped

    def group_citations_by_group(self, citations: List[Citation]) -> Dict[str, List[Citation]]:
        """
        Group citations by group ID.

        Args:
            citations: List of Citation objects

        Returns:
            Dict mapping group_id -> list of citations
        """
        grouped = {}
        for citation in citations:
            if citation.group_id not in grouped:
                grouped[citation.group_id] = []
            grouped[citation.group_id].append(citation)

        return grouped

    def count_citation_frequency(self, text: str) -> Dict[str, int]:
        """
        Count frequency of each citation in text.

        Args:
            text: Text containing citations

        Returns:
            Dict mapping citation_id -> frequency count
        """
        citations = self.extract_all(text)
        frequency = {}

        for citation in citations:
            if citation.raw not in frequency:
                frequency[citation.raw] = 0
            frequency[citation.raw] += 1

        return frequency
