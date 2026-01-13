"""
Test Runner
===========

Orchestrates test execution with progress reporting and result aggregation.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class TestStatus(Enum):
    """Test status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Result from a single test"""
    test_name: str
    status: TestStatus
    duration_ms: int
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_name': self.test_name,
            'status': self.status.value,
            'duration_ms': self.duration_ms,
            'details': self.details,
            'error': self.error,
            'warnings': self.warnings
        }


@dataclass
class TestSuiteResult:
    """Results from a test suite"""
    suite_name: str
    tests: List[TestResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def passed_count(self) -> int:
        """Number of passed tests"""
        return sum(1 for t in self.tests if t.status == TestStatus.PASSED)

    @property
    def failed_count(self) -> int:
        """Number of failed tests"""
        return sum(1 for t in self.tests if t.status == TestStatus.FAILED)

    @property
    def total_count(self) -> int:
        """Total number of tests"""
        return len(self.tests)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'suite_name': self.suite_name,
            'duration_seconds': self.duration_seconds,
            'total_tests': self.total_count,
            'passed': self.passed_count,
            'failed': self.failed_count,
            'tests': [t.to_dict() for t in self.tests]
        }


class TestRunner:
    """
    Test runner that executes tests and aggregates results.

    Supports:
    - Async test execution
    - Progress reporting
    - Result aggregation
    - Error handling
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestSuiteResult] = []

    async def run_test(
        self,
        test_name: str,
        test_func: Callable,
        *args,
        **kwargs
    ) -> TestResult:
        """
        Run a single test function.

        Args:
            test_name: Human-readable test name
            test_func: Async function to execute
            *args, **kwargs: Arguments to pass to test function

        Returns:
            TestResult with status and details
        """
        start_time = time.time()

        try:
            if self.verbose:
                print(f"  Running: {test_name}")

            # Execute test function
            result_data = await test_func(*args, **kwargs)

            duration_ms = int((time.time() - start_time) * 1000)

            # Test passed if no exception
            return TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                duration_ms=duration_ms,
                details=result_data or {}
            )

        except AssertionError as e:
            # Test failed (assertion error)
            duration_ms = int((time.time() - start_time) * 1000)
            return TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                duration_ms=duration_ms,
                error=f"Assertion failed: {str(e)}"
            )

        except Exception as e:
            # Test failed (unexpected error)
            duration_ms = int((time.time() - start_time) * 1000)
            return TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                duration_ms=duration_ms,
                error=f"Error: {str(e)}"
            )

    async def run_suite(
        self,
        suite_name: str,
        tests: List[tuple]
    ) -> TestSuiteResult:
        """
        Run a suite of tests.

        Args:
            suite_name: Name of test suite
            tests: List of (test_name, test_func, args, kwargs) tuples

        Returns:
            TestSuiteResult with all test results
        """
        print(f"\n{'='*80}")
        print(f"Test Suite: {suite_name}")
        print(f"{'='*80}")

        suite_result = TestSuiteResult(suite_name=suite_name)

        for test_item in tests:
            if len(test_item) == 2:
                test_name, test_func = test_item
                args, kwargs = (), {}
            elif len(test_item) == 3:
                test_name, test_func, args = test_item
                kwargs = {}
            else:
                test_name, test_func, args, kwargs = test_item

            # Run test
            result = await self.run_test(test_name, test_func, *args, **kwargs)
            suite_result.tests.append(result)

            # Print result
            status_icon = "✓" if result.status == TestStatus.PASSED else "✗"
            duration_str = f"{result.duration_ms}ms"

            if result.status == TestStatus.PASSED:
                print(f"  {status_icon} {test_name} - {duration_str}")
                if self.verbose and result.details:
                    for key, value in result.details.items():
                        print(f"      {key}: {value}")
            else:
                print(f"  {status_icon} {test_name} - {duration_str}")
                if result.error:
                    print(f"      Error: {result.error}")

        suite_result.end_time = time.time()
        self.results.append(suite_result)

        # Print suite summary
        print(f"\n  Suite completed: {suite_result.passed_count}/{suite_result.total_count} passed")
        print(f"  Duration: {suite_result.duration_seconds:.2f}s")

        return suite_result

    def get_summary(self) -> Dict[str, Any]:
        """Get overall test summary"""
        total_tests = sum(r.total_count for r in self.results)
        total_passed = sum(r.passed_count for r in self.results)
        total_failed = sum(r.failed_count for r in self.results)
        total_duration = sum(r.duration_seconds for r in self.results)

        return {
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'duration_seconds': total_duration,
            'suites': [r.to_dict() for r in self.results]
        }

    def print_summary(self):
        """Print overall test summary"""
        summary = self.get_summary()

        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Tests Run: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Duration: {summary['duration_seconds']:.1f}s")
        print(f"{'='*80}\n")


# Helper for running async tests from sync context
def run_async_tests(runner: TestRunner, suite_name: str, tests: List[tuple]) -> TestSuiteResult:
    """Run async test suite from synchronous context"""
    return asyncio.run(runner.run_suite(suite_name, tests))
