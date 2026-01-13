"""
Validators
==========

Validates test results for quality, performance, and correctness.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class ResultValidator:
    """
    Validates API response structure and content quality.
    """

    @staticmethod
    def validate_search_response(data: Dict[str, Any], expected_min_results: int = 1) -> ValidationResult:
        """
        Validate search response structure.

        Checks:
        - Has required fields
        - Result count is reasonable
        - Results have proper structure
        """
        # Check required fields
        required_fields = ['results', 'total_results', 'query', 'processing_time_ms']
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            return ValidationResult(
                passed=False,
                message=f"Missing required fields: {missing_fields}"
            )

        # Check result count
        total_results = data.get('total_results', 0)
        if total_results < expected_min_results:
            return ValidationResult(
                passed=False,
                message=f"Expected at least {expected_min_results} results, got {total_results}"
            )

        # Validate result structure
        results = data.get('results', [])
        if len(results) > 0:
            first_result = results[0]
            result_required_fields = ['segment_id', 'text', 'similarity']
            missing_result_fields = [f for f in result_required_fields if f not in first_result]

            if missing_result_fields:
                return ValidationResult(
                    passed=False,
                    message=f"Results missing fields: {missing_result_fields}"
                )

        return ValidationResult(
            passed=True,
            message=f"Valid search response with {total_results} results",
            details={'total_results': total_results}
        )

    @staticmethod
    def validate_theme_summary(data: Dict[str, Any]) -> ValidationResult:
        """
        Validate theme summary structure and quality.

        Checks:
        - Theme summary exists
        - Has summary text
        - Has citations
        - Citations are valid format
        """
        theme_summary = data.get('theme_summary')

        if not theme_summary:
            return ValidationResult(
                passed=False,
                message="No theme summary in response"
            )

        # Check summary text
        summary_text = theme_summary.get('summary_text', '')
        if len(summary_text) < 100:
            return ValidationResult(
                passed=False,
                message=f"Summary text too short: {len(summary_text)} chars"
            )

        # Check citations
        citations = theme_summary.get('citations', {})
        if not citations:
            return ValidationResult(
                passed=False,
                message="Theme summary has no citations"
            )

        # Validate citation format (should be dict with citation_id -> segment data)
        if not isinstance(citations, dict):
            return ValidationResult(
                passed=False,
                message="Citations must be a dictionary"
            )

        return ValidationResult(
            passed=True,
            message=f"Valid theme summary ({len(summary_text)} chars, {len(citations)} citations)",
            details={
                'summary_chars': len(summary_text),
                'citation_count': len(citations)
            }
        )

    @staticmethod
    def validate_subthemes(data: Dict[str, Any], allow_zero: bool = True) -> ValidationResult:
        """
        Validate sub-theme detection results.

        Args:
            data: Response data
            allow_zero: If True, 0 sub-themes is valid (homogeneous discourse)

        Checks:
        - Sub-theme structure
        - Cluster validation metrics
        - Sub-theme summaries
        """
        subtheme_summaries = data.get('subtheme_summaries')

        if subtheme_summaries is None:
            if allow_zero:
                return ValidationResult(
                    passed=True,
                    message="No sub-themes (likely homogeneous discourse)",
                    details={'subtheme_count': 0}
                )
            else:
                return ValidationResult(
                    passed=False,
                    message="Expected sub-themes but none found"
                )

        if not isinstance(subtheme_summaries, list):
            return ValidationResult(
                passed=False,
                message="Sub-theme summaries must be a list"
            )

        # Validate each sub-theme
        for i, subtheme in enumerate(subtheme_summaries):
            # Check required fields
            required_fields = ['theme_name', 'summary_text', 'citations', 'segment_count']
            missing_fields = [f for f in required_fields if f not in subtheme]

            if missing_fields:
                return ValidationResult(
                    passed=False,
                    message=f"Sub-theme {i} missing fields: {missing_fields}"
                )

            # Check validation metrics
            metadata = subtheme.get('metadata', {})
            validation = metadata.get('cluster_validation', {})

            if validation:
                silhouette_score = validation.get('silhouette_score')
                if silhouette_score is not None and silhouette_score < 0:
                    return ValidationResult(
                        passed=False,
                        message=f"Sub-theme {i} has negative silhouette score: {silhouette_score}"
                    )

        return ValidationResult(
            passed=True,
            message=f"Valid sub-themes ({len(subtheme_summaries)} detected)",
            details={'subtheme_count': len(subtheme_summaries)}
        )


class CacheValidator:
    """
    Validates cache behavior.
    """

    @staticmethod
    def validate_cache_hit(
        stream_data: Dict[str, Any],
        should_be_cached: bool
    ) -> ValidationResult:
        """
        Validate whether request hit cache as expected.

        Args:
            stream_data: Complete event data (from SSE stream)
            should_be_cached: Whether cache hit is expected

        For non-SSE responses, checks for fast response time as proxy for cache.
        """
        # Check if there's a 'cache_hit' indicator in response
        complete_event = stream_data.get('complete', {})
        if isinstance(complete_event, dict):
            cache_hit = complete_event.get('data', {}).get('cache_hit', False)
        else:
            # Check processing time as proxy
            processing_time = stream_data.get('processing_time_ms', 0)
            cache_hit = processing_time < 100  # <100ms likely cached

        if should_be_cached and not cache_hit:
            return ValidationResult(
                passed=False,
                message="Expected cache hit but got cache miss"
            )

        if not should_be_cached and cache_hit:
            return ValidationResult(
                passed=False,
                message="Expected cache miss but got cache hit"
            )

        cache_status = "hit" if cache_hit else "miss"
        return ValidationResult(
            passed=True,
            message=f"Cache behavior correct: {cache_status}",
            details={'cache_hit': cache_hit}
        )


class PerformanceValidator:
    """
    Validates performance metrics against thresholds.
    """

    # Default performance thresholds (ms)
    THRESHOLDS = {
        'faiss_search_ms': 1000,
        'theme_generation_ms': 15000,
        'subtheme_detection_ms': 5000,
        'total_search_ms': 20000,
        'query_expansion_ms': 2000,
        'embedding_batch_ms': 500
    }

    @staticmethod
    def validate_performance(
        metric_name: str,
        actual_value_ms: int,
        custom_threshold_ms: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate a performance metric against threshold.

        Args:
            metric_name: Name of metric (must be in THRESHOLDS)
            actual_value_ms: Actual measured time in milliseconds
            custom_threshold_ms: Optional custom threshold (overrides default)
        """
        threshold_ms = custom_threshold_ms or PerformanceValidator.THRESHOLDS.get(metric_name)

        if threshold_ms is None:
            return ValidationResult(
                passed=False,
                message=f"Unknown metric: {metric_name}"
            )

        passed = actual_value_ms <= threshold_ms

        if passed:
            message = f"{metric_name}: {actual_value_ms}ms (threshold: <{threshold_ms}ms) ✓"
        else:
            message = f"{metric_name}: {actual_value_ms}ms EXCEEDED threshold ({threshold_ms}ms) ✗"

        return ValidationResult(
            passed=passed,
            message=message,
            details={
                'metric': metric_name,
                'actual_ms': actual_value_ms,
                'threshold_ms': threshold_ms,
                'margin_ms': threshold_ms - actual_value_ms
            }
        )

    @staticmethod
    def validate_multiple_metrics(
        metrics: Dict[str, int],
        custom_thresholds: Optional[Dict[str, int]] = None
    ) -> List[ValidationResult]:
        """
        Validate multiple performance metrics.

        Args:
            metrics: Dict of {metric_name: actual_value_ms}
            custom_thresholds: Optional dict of {metric_name: threshold_ms}

        Returns:
            List of ValidationResult for each metric
        """
        results = []

        for metric_name, actual_value in metrics.items():
            custom_threshold = None
            if custom_thresholds:
                custom_threshold = custom_thresholds.get(metric_name)

            result = PerformanceValidator.validate_performance(
                metric_name,
                actual_value,
                custom_threshold
            )
            results.append(result)

        return results


class SimilarityValidator:
    """
    Validates similarity scores and thresholds.
    """

    @staticmethod
    def validate_similarity_threshold(
        results: List[Dict[str, Any]],
        threshold: float
    ) -> ValidationResult:
        """
        Validate that all results meet similarity threshold.

        Args:
            results: List of search results with 'similarity' field
            threshold: Minimum similarity threshold

        Returns:
            ValidationResult
        """
        if not results:
            return ValidationResult(
                passed=True,
                message="No results to validate"
            )

        # Check each result
        below_threshold = []
        for i, result in enumerate(results):
            similarity = result.get('similarity', 0)
            if similarity < threshold:
                below_threshold.append((i, similarity))

        if below_threshold:
            return ValidationResult(
                passed=False,
                message=f"{len(below_threshold)} results below threshold {threshold}",
                details={'violations': below_threshold[:5]}  # First 5 violations
            )

        # Get min similarity
        min_similarity = min(r.get('similarity', 1.0) for r in results)

        return ValidationResult(
            passed=True,
            message=f"All {len(results)} results meet threshold {threshold} (min: {min_similarity:.3f})",
            details={'min_similarity': min_similarity, 'threshold': threshold}
        )
