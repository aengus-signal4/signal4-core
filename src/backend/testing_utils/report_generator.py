"""
Report Generator
================

Generates comprehensive test reports in multiple formats.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ReportGenerator:
    """
    Generates test reports in JSON, Markdown, and console formats.
    """

    def __init__(self, output_dir: str = "test_results/latest"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SSE logs directory
        self.sse_log_dir = self.output_dir / "sse_logs"
        self.sse_log_dir.mkdir(exist_ok=True)

    def save_json_report(self, summary: Dict[str, Any], filename: str = "results.json"):
        """Save JSON report"""
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ JSON report saved: {output_path}")

    def save_markdown_report(
        self,
        summary: Dict[str, Any],
        filename: str = "report.md"
    ):
        """Generate and save Markdown report"""
        output_path = self.output_dir / filename

        md_lines = []
        md_lines.append("# Backend Test Report")
        md_lines.append("")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"**Duration:** {summary.get('duration_seconds', 0):.1f}s")
        md_lines.append("")

        # Overall summary
        md_lines.append("## Summary")
        md_lines.append("")
        md_lines.append(f"- **Total Tests:** {summary.get('total_tests', 0)}")
        md_lines.append(f"- **Passed:** {summary.get('passed', 0)}")
        md_lines.append(f"- **Failed:** {summary.get('failed', 0)}")
        md_lines.append("")

        # Test suites
        md_lines.append("## Test Suites")
        md_lines.append("")

        for suite in summary.get('suites', []):
            suite_name = suite.get('suite_name', 'Unknown')
            md_lines.append(f"### {suite_name}")
            md_lines.append("")
            md_lines.append(f"- Duration: {suite.get('duration_seconds', 0):.2f}s")
            md_lines.append(f"- Tests: {suite.get('passed', 0)}/{suite.get('total_tests', 0)} passed")
            md_lines.append("")

            # Test results
            md_lines.append("| Test | Status | Duration | Details |")
            md_lines.append("|------|--------|----------|---------|")

            for test in suite.get('tests', []):
                test_name = test.get('test_name', 'Unknown')
                status = test.get('status', 'unknown')
                duration_ms = test.get('duration_ms', 0)
                error = test.get('error', '')

                status_icon = "✓" if status == "passed" else "✗"
                details = error if error else self._format_details(test.get('details', {}))

                md_lines.append(f"| {test_name} | {status_icon} {status} | {duration_ms}ms | {details} |")

            md_lines.append("")

        # Performance metrics
        if 'performance' in summary:
            md_lines.append("## Performance")
            md_lines.append("")
            perf = summary['performance']

            for metric_name, metric_data in perf.items():
                md_lines.append(f"- **{metric_name}:** {metric_data.get('avg_ms', 0):.0f}ms avg (threshold: {metric_data.get('threshold_ms', 'N/A')}ms)")

            md_lines.append("")

        # Cache behavior
        if 'cache_behavior' in summary:
            md_lines.append("## Cache Behavior")
            md_lines.append("")
            cache = summary['cache_behavior']

            for test_type, cache_status in cache.items():
                status_icon = "✓" if cache_status.get('passed', False) else "✗"
                md_lines.append(f"- **{test_type}:** {status_icon} {cache_status.get('message', '')}")

            md_lines.append("")

        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(md_lines))

        print(f"✓ Markdown report saved: {output_path}")

    def save_sse_log(
        self,
        test_name: str,
        events: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save SSE event log for a single test"""
        # Sanitize test name for filename
        safe_name = test_name.replace(' ', '_').replace('/', '_').lower()
        log_path = self.sse_log_dir / f"{safe_name}.sse.log"

        with open(log_path, 'w') as f:
            # Write metadata
            if metadata:
                f.write("# Metadata\n")
                f.write(json.dumps(metadata, indent=2))
                f.write("\n\n")

            # Write events
            f.write("# SSE Events\n\n")
            for event in events:
                event_type = event.get('event_type', 'unknown')
                timestamp_ms = event.get('timestamp_ms', 0)
                data = event.get('data', {})

                f.write(f"[{timestamp_ms}ms] {event_type}\n")
                if data:
                    f.write(f"  {json.dumps(data, indent=2)}\n")
                f.write("\n")

    def print_performance_summary(self, performance_results: List[Any]):
        """Print performance summary to console"""
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")

        for result in performance_results:
            if result.passed:
                print(f"✓ {result.message}")
            else:
                print(f"✗ {result.message}")

    def print_cache_summary(self, cache_results: List[Any]):
        """Print cache behavior summary"""
        print(f"\n{'='*80}")
        print("CACHE BEHAVIOR SUMMARY")
        print(f"{'='*80}")

        for result in cache_results:
            if result.passed:
                print(f"✓ {result.message}")
            else:
                print(f"✗ {result.message}")

    def _format_details(self, details: Dict[str, Any]) -> str:
        """Format details dict as compact string"""
        if not details:
            return ""

        parts = []
        for key, value in details.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value}")
            elif isinstance(value, str) and len(value) < 30:
                parts.append(f"{key}={value}")

        return ", ".join(parts[:3])  # Max 3 items

    def generate_timestamp_directory(self) -> Path:
        """Create timestamped directory for archival"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = self.output_dir.parent / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)
        return archive_dir


def print_test_header(title: str):
    """Print formatted test header"""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}\n")


def print_sse_event(event_type: str, timestamp_ms: int, details: str = ""):
    """Print formatted SSE event"""
    if details:
        print(f"  ⏱  {event_type} ({timestamp_ms}ms) - {details}")
    else:
        print(f"  ⏱  {event_type} ({timestamp_ms}ms)")


def print_result(passed: bool, message: str, duration_ms: Optional[int] = None):
    """Print formatted test result"""
    icon = "✓" if passed else "✗"
    if duration_ms is not None:
        print(f"  {icon} {message} - {duration_ms}ms")
    else:
        print(f"  {icon} {message}")


def print_section(title: str):
    """Print section divider"""
    print(f"\n{'-'*80}")
    print(title)
    print(f"{'-'*80}")
