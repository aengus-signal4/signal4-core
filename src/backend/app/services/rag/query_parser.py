"""
QueryParser
===========

Convert natural language queries into structured retrieval filters using LLM.

Uses structured prompt pattern: <task> <context> <response_rules>

Examples:
    "Recent Canadian content about immigration"
    → projects=["Canadian"], time_window_days=30, keywords=["immigration"]

    "French videos from the past week"
    → languages=["fr"], time_window_days=7

    "Analyze Europe project from 2024"
    → projects=["Europe"], date_range=(2024-01-01, 2024-12-31)
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import requests
import os

from ..llm_service import LLMService
from ...utils.backend_logger import get_logger

logger = get_logger("query_parser")


# Prompt template following <task> <context> <response_rules> pattern
QUERY_PARSING_SYSTEM_PROMPT = """You are a query parser for a content analysis system.

# TASK
Parse natural language queries into structured filters for content retrieval.

# CONTEXT
{context}

# RESPONSE RULES
1. Return ONLY valid JSON, no explanation or markdown formatting
2. Use exact project/language codes from available options
3. Be flexible with variations (e.g., "Canada" → "Canadian", "EU" → "Europe")
4. Extract all relevant keywords/topics from the query
5. Infer user intent from action verbs:
   - "find", "search", "show" → intent: "search"
   - "analyze", "understand", "explore" → intent: "analyze"
   - "compare", "contrast", "versus" → intent: "compare"
   - "summarize", "overview" → intent: "summarize"
6. If no project specified, include all available projects
7. If no language specified, leave as null (all languages)
8. If no time period specified, use "past 30 days" as default

# OUTPUT FORMAT
{{
    "projects": ["project_name"],     // From available projects list
    "languages": ["en", "fr"],        // ISO 639-1 codes, or null for all
    "channels": ["channel_name"],     // Specific channels, or null
    "time_period": "past 30 days",    // Human-readable description
    "time_window_days": 30,           // Integer days
    "keywords": ["keyword1"],         // Main topics/themes
    "intent": "search"                // One of: search, analyze, compare, summarize
}}"""


QUERY_PARSING_USER_PROMPT = """Parse this query:
"{query}"

Return only the JSON object, nothing else."""


class QueryParser:
    """Parse natural language queries into structured retrieval filters."""

    # Available projects (could be dynamically loaded)
    KNOWN_PROJECTS = [
        "CPRMV",
        "Canadian",
        "Europe",
        "BigChannels"
    ]

    # Common language mappings
    LANGUAGE_MAP = {
        "english": "en",
        "french": "fr",
        "francais": "fr",
        "français": "fr",
        "german": "de",
        "deutsch": "de",
        "spanish": "es",
        "italiano": "it",
        "italian": "it"
    }

    # Time period mappings
    TIME_PERIODS = {
        "today": 1,
        "yesterday": 1,
        "this week": 7,
        "past week": 7,
        "last week": 7,
        "this month": 30,
        "past month": 30,
        "last month": 30,
        "past 2 months": 60,
        "past 3 months": 90,
        "this quarter": 90,
        "past 6 months": 180,
        "this year": 365,
        "past year": 365,
        "recent": 30,  # default for "recent"
    }

    def __init__(self, llm_service: LLMService):
        """
        Initialize parser.

        Args:
            llm_service: LLMService instance (required)
        """
        if llm_service is None:
            raise ValueError("LLMService is required for QueryParser")
        self.llm_service = llm_service

    def parse(
        self,
        query: str,
        default_time_window_days: int = 30,
        available_projects: Optional[List[str]] = None,
        available_channels: Optional[List[str]] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language query into structured filters using LLM.

        Args:
            query: Natural language query string
            default_time_window_days: Default time window if not specified
            available_projects: Override default project list
            available_channels: List of known channels (for better parsing)
            additional_context: Extra context to help LLM parsing

        Returns:
            Dictionary with structured filters:
            {
                "projects": List[str],
                "languages": List[str],
                "channels": List[str],
                "date_range": Tuple[datetime, datetime],
                "time_window_days": int,
                "keywords": List[str],
                "intent": str,  # "search", "analyze", "compare", "summarize"
                "original_query": str
            }
        """
        projects = available_projects or self.KNOWN_PROJECTS

        # Build context section
        context_parts = [
            f"Available projects: {', '.join(projects)}",
            "Available languages: en (English), fr (French), de (German), es (Spanish), it (Italian)",
            f"Default time window: {default_time_window_days} days",
            f"Current date: {datetime.now().strftime('%Y-%m-%d')}"
        ]

        if available_channels:
            context_parts.append(f"Known channels: {', '.join(available_channels[:20])}" +
                               (" and more..." if len(available_channels) > 20 else ""))

        if additional_context:
            context_parts.append(f"\nAdditional context:\n{additional_context}")

        context = "\n".join(context_parts)

        # Build prompts using template
        system_prompt = QUERY_PARSING_SYSTEM_PROMPT.format(context=context)
        user_prompt = QUERY_PARSING_USER_PROMPT.format(query=query)

        # Call LLM (sync request)
        try:
            api_key = os.getenv('XAI_API_KEY', '')
            if not api_key:
                raise Exception("XAI_API_KEY not found in environment")

            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-2-1212",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,  # Low temperature for consistent parsing
                    "max_tokens": 500
                },
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")

            response_text = response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()

            # Parse JSON response
            parsed = json.loads(response_text)

            # Convert time_window_days to date_range
            date_range = self._time_window_to_date_range(
                parsed.get("time_window_days", default_time_window_days)
            )

            # Build structured filters
            filters = {
                "projects": parsed.get("projects", projects),
                "languages": parsed.get("languages"),
                "channels": parsed.get("channels"),
                "date_range": date_range,
                "time_window_days": parsed.get("time_window_days", default_time_window_days),
                "keywords": parsed.get("keywords", []),
                "intent": parsed.get("intent", "search"),
                "original_query": query
            }

            logger.info(f"Parsed query: '{query}' → {filters}")
            return filters

        except Exception as e:
            logger.error(f"Failed to parse query '{query}': {e}", exc_info=True)
            # Return default filters as fallback
            return {
                "projects": projects,
                "languages": None,
                "channels": None,
                "date_range": self._time_window_to_date_range(default_time_window_days),
                "time_window_days": default_time_window_days,
                "keywords": [query],  # Use query as keyword
                "intent": "search",
                "original_query": query,
                "parse_error": str(e)
            }

    def _time_window_to_date_range(self, days: int) -> Tuple[datetime, datetime]:
        """
        Convert time window in days to date range.

        Args:
            days: Number of days

        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return (start_date, end_date)

    def validate_filters(self, filters: Dict[str, Any]) -> bool:
        """
        Validate parsed filters.

        Args:
            filters: Parsed filters dictionary

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required = ["projects", "date_range", "time_window_days", "original_query"]
        for field in required:
            if field not in filters:
                logger.warning(f"Missing required field: {field}")
                return False

        # Check date range is valid
        if filters["date_range"]:
            start, end = filters["date_range"]
            if start > end:
                logger.warning(f"Invalid date range: start > end")
                return False

        return True

    def filters_to_retriever_params(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parsed filters to SegmentRetriever parameters.

        Args:
            filters: Parsed filters from parse()

        Returns:
            Dictionary of parameters for SegmentRetriever.fetch_by_filter()
        """
        params = {}

        if filters.get("projects"):
            params["projects"] = filters["projects"]

        if filters.get("languages"):
            params["languages"] = filters["languages"]

        if filters.get("channels"):
            params["channels"] = filters["channels"]

        if filters.get("date_range"):
            params["date_range"] = filters["date_range"]

        return params

    def get_available_projects(self) -> List[str]:
        """
        Get list of available projects.

        In the future, this could query the database dynamically.

        Returns:
            List of project names
        """
        return self.KNOWN_PROJECTS.copy()

    def get_available_languages(self) -> List[str]:
        """
        Get list of available language codes.

        Returns:
            List of ISO 639-1 language codes
        """
        return list(set(self.LANGUAGE_MAP.values()))
