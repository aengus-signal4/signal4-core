"""
Hierarchical Summary Workflow
==============================

Multi-group, multi-theme analysis with hierarchical summarization.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..analysis_pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)


class HierarchicalSummaryWorkflow:
    """
    Hierarchical summary workflow for complex multi-group analysis.

    Workflow:
    1. For each group: retrieve segments by filters
    2. Extract themes (clustering)
    3. Extract sub-themes (hierarchical clustering, optional)
    4. Select representative segments per theme
    5. Generate theme summaries (parallel)
    6. Return structured hierarchy with segment ID traceability

    Example:
        workflow = HierarchicalSummaryWorkflow(llm_service, db_session)
        result = await workflow.run(
            groupings=[
                {"group_id": "en", "group_name": "English", "filter": {"languages": ["en"]}},
                {"group_id": "fr", "group_name": "French", "filter": {"languages": ["fr"]}}
            ],
            num_themes=5,
            samples_per_theme=20
        )
    """

    def __init__(self, llm_service, db_session):
        """
        Initialize workflow.

        Args:
            llm_service: LLMService for text generation
            db_session: Database session for segment retrieval
        """
        self.llm_service = llm_service
        self.db_session = db_session
        logger.info("HierarchicalSummaryWorkflow initialized")

    async def run(
        self,
        groupings: Optional[List[Dict]] = None,
        time_window_days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        num_themes: int = 5,
        samples_per_theme: int = 20,
        extract_subthemes: bool = False,
        n_subthemes: int = 3,
        max_concurrent_themes: int = 20,
        model: str = "grok-2-1212",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        generate_quantitative_metrics: bool = False,
        include_baseline_for_centrality: bool = False
    ) -> Dict[str, Any]:
        """
        Execute hierarchical summary workflow.

        Args:
            groupings: List of group configs (None = all content)
                Each: {"group_id": str, "group_name": str, "filter": dict}
                Filter can contain: projects, languages, channels, speakers
            time_window_days: Time window for analysis (days back from now)
            start_date: Explicit start date (overrides time_window)
            end_date: Explicit end date
            num_themes: Target number of themes per group
            samples_per_theme: Segments to sample per theme
            extract_subthemes: Whether to extract sub-themes
            n_subthemes: Number of sub-themes per theme
            max_concurrent_themes: Max parallel theme summaries
            model: LLM model to use
            temperature: Generation temperature
            max_tokens: Max tokens per summary
            generate_quantitative_metrics: Generate quantitative analysis
            include_baseline_for_centrality: Include baseline for discourse centrality

        Returns:
            Dict with:
                - theme_summaries: List of theme summary dicts
                - segment_ids_by_theme: Dict mapping theme_id -> segment IDs
                - group_results: Dict of group-level results
                - total_themes: Number of themes discovered
                - total_segments: Total segments analyzed
                - quantitative_metrics: Optional quantitative analysis (if requested)
        """
        # Default to all content if no groupings
        if not groupings:
            groupings = [{"group_id": "all", "group_name": "All Content", "filter": {}}]

        logger.info(
            f"Running HierarchicalSummary: {len(groupings)} groups, "
            f"time_window={time_window_days}d, themes={num_themes}, "
            f"subthemes={'yes' if extract_subthemes else 'no'}"
        )

        # Process each group
        group_results = {}
        all_theme_summaries = []
        segment_ids_by_theme = {}

        for group_config in groupings:
            group_id = group_config["group_id"]
            group_name = group_config.get("group_name", group_id)
            filter_config = group_config.get("filter", {})

            logger.info(f"Processing group: {group_id} ({group_name})")

            # Build pipeline for this group
            pipeline = AnalysisPipeline(
                f"hierarchical_{group_id}",
                llm_service=self.llm_service,
                db_session=self.db_session
            )

            # Build filter arguments
            filter_args = {
                "projects": filter_config.get("projects"),
                "languages": filter_config.get("languages"),
                "channels": filter_config.get("channels"),
                "speakers": filter_config.get("speakers"),
            }

            # Add time filtering
            if start_date and end_date:
                filter_args["date_range"] = (start_date, end_date)
            elif time_window_days:
                # Calculate date range from time_window_days
                from datetime import datetime, timezone, timedelta
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=time_window_days)
                filter_args["date_range"] = (start, end)

            # Build pipeline steps
            pipeline = pipeline.retrieve_segments(**filter_args)

            # Add quantitative analysis if requested
            if generate_quantitative_metrics:
                pipeline = pipeline.quantitative_analysis(
                    include_baseline=include_baseline_for_centrality,
                    time_window_days=time_window_days
                )

            pipeline = pipeline.extract_themes(
                method="hdbscan",
                n_clusters=num_themes,
                min_cluster_size=5
            )

            if extract_subthemes:
                pipeline = pipeline.extract_subthemes(
                    method="hdbscan",
                    n_subthemes=n_subthemes,
                    min_cluster_size=3,
                    max_concurrent=5
                )

            # Select representative segments
            pipeline = pipeline.select_segments(
                strategy="diversity",
                n=samples_per_theme
            )

            # Generate summaries
            summary_level = "subtheme" if extract_subthemes else "theme"
            summary_template = "subtheme_summary" if extract_subthemes else "theme_summary"

            pipeline = pipeline.generate_summaries(
                template=summary_template,
                level=summary_level,
                max_concurrent=max_concurrent_themes,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Execute pipeline
            try:
                result = await pipeline.execute()
            except Exception as e:
                logger.error(f"Error processing group {group_id}: {e}", exc_info=True)
                group_results[group_id] = {
                    "error": str(e),
                    "themes": [],
                    "summaries": [],
                    "segment_count": 0
                }
                continue

            # Extract results
            themes = result.data.get("themes", [])
            summaries = result.data.get("summaries", {}).get(summary_level, [])
            segment_ids_map = result.data.get("segment_ids", {}).get(summary_level, {})
            segments = result.data.get("segments", [])
            quantitative_metrics = result.data.get("quantitative_metrics")

            logger.info(f"Group {group_id}: extracted {len(themes)} themes, generated {len(summaries)} summaries")

            # Build theme summary objects
            if extract_subthemes:
                # For subthemes, we need to match summaries to subthemes
                subtheme_map = result.data.get("subtheme_map", {})
                summary_idx = 0

                for parent_theme_id, subthemes in subtheme_map.items():
                    for subtheme in subthemes:
                        if summary_idx >= len(summaries):
                            break

                        theme_id = f"{group_id}_subtheme_{summary_idx}"
                        task_id = f"subtheme_{subtheme.theme_id}"

                        theme_summary = {
                            "group_id": group_id,
                            "group_name": group_name,
                            "theme_id": theme_id,
                            "theme_name": subtheme.theme_name,
                            "parent_theme_id": parent_theme_id,
                            "summary": summaries[summary_idx],
                            "segment_count": len(subtheme.segments),
                            "keywords": subtheme.keywords
                        }

                        all_theme_summaries.append(theme_summary)
                        segment_ids_by_theme[theme_id] = segment_ids_map.get(task_id, [])
                        summary_idx += 1
            else:
                # Standard theme-level summaries
                for i, (theme, summary) in enumerate(zip(themes, summaries)):
                    theme_id = f"{group_id}_theme_{i}"
                    task_id = f"theme_{theme.theme_id}"

                    theme_summary = {
                        "group_id": group_id,
                        "group_name": group_name,
                        "theme_id": theme_id,
                        "theme_name": theme.theme_name,
                        "summary": summary,
                        "segment_count": len(theme.segments),
                        "keywords": theme.keywords
                    }

                    all_theme_summaries.append(theme_summary)
                    segment_ids_by_theme[theme_id] = segment_ids_map.get(task_id, [])

            # Store group results
            group_results[group_id] = {
                "group_name": group_name,
                "themes": themes,
                "summaries": summaries,
                "segment_count": len(segments),
                "theme_count": len(themes),
                "quantitative_metrics": quantitative_metrics
            }

        # Calculate totals
        total_themes = len(all_theme_summaries)
        total_segments = sum(g.get("segment_count", 0) for g in group_results.values())

        # Aggregate quantitative metrics if generated
        aggregated_metrics = None
        if generate_quantitative_metrics:
            # Collect metrics from each group
            group_metrics = [
                g.get("quantitative_metrics")
                for g in group_results.values()
                if g.get("quantitative_metrics")
            ]

            if group_metrics:
                # For single group, use its metrics directly
                if len(group_metrics) == 1:
                    aggregated_metrics = group_metrics[0]
                else:
                    # For multiple groups, aggregate key metrics
                    aggregated_metrics = {
                        "total_segments": sum(m["total_segments"] for m in group_metrics),
                        "unique_videos": sum(m["unique_videos"] for m in group_metrics),
                        "unique_channels": sum(m["unique_channels"] for m in group_metrics),
                        "by_group": {
                            group_id: group_results[group_id]["quantitative_metrics"]
                            for group_id in group_results
                            if group_results[group_id].get("quantitative_metrics")
                        }
                    }

        logger.info(f"HierarchicalSummary complete: {total_themes} themes, {total_segments} segments")

        result = {
            "theme_summaries": all_theme_summaries,
            "segment_ids_by_theme": segment_ids_by_theme,
            "group_results": group_results,
            "total_themes": total_themes,
            "total_segments": total_segments
        }

        if aggregated_metrics:
            result["quantitative_metrics"] = aggregated_metrics

        return result
