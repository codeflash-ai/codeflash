from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from codeflash_core.config import TestConfig
    from codeflash_core.models import (
        BenchmarkResults,
        Candidate,
        CodeContext,
        CoverageData,
        FunctionToOptimize,
        GeneratedTestSuite,
        OptimizationResult,
        ScoredCandidate,
        TestDiff,
        TestResults,
        TestReviewResult,
    )


@runtime_checkable
class LanguagePlugin(Protocol):
    """Protocol that language packages must implement.

    A language plugin provides all language-specific functionality:
    discovering functions, extracting code context, running tests,
    replacing code, running benchmarks, and formatting.
    """

    def discover_functions(self, paths: list[Path]) -> list[FunctionToOptimize]:
        """Discover optimizable functions in the given file paths."""
        ...

    def build_index(self, files: list[Path], on_progress: Callable[[Any], None] | None = None) -> None:
        """Pre-index source files for dependency analysis (e.g. call graph).

        Called after discovery, before optimization begins. Plugins that
        maintain a dependency resolver should index the given files here.
        The optional on_progress callback is called once per file with an
        implementation-defined result object.
        """
        ...

    def rank_functions(
        self,
        functions: list[FunctionToOptimize],
        trace_file: Path | None = None,
        test_counts: dict[tuple[Path, str], int] | None = None,
    ) -> list[FunctionToOptimize]:
        """Rank functions in optimization order (most impactful first).

        Ranking priority:
        1. Trace-based addressable time (when trace_file is provided)
        2. Dependency count from the call graph (fallback)
        3. Existing unit test count as secondary sort key (when test_counts provided)

        Returns the functions unchanged if no ranking is possible.
        """
        ...

    def get_dependency_counts(self) -> dict[str, int]:
        """Return {qualified_name: callee_count} from the most recent ranking.

        Called after rank_functions() so the UI can display per-function
        dependency information.  Plugins that don't track a call graph may
        return an empty dict.
        """
        ...

    def get_candidates(self, context: CodeContext, trace_id: str = "") -> list[Candidate]:
        """Request optimization candidates from the AI service."""
        ...

    def extract_context(self, function: FunctionToOptimize) -> CodeContext:
        """Extract all code context needed to optimize a function."""
        ...

    @overload
    def run_tests(
        self,
        test_config: TestConfig,
        test_files: list[Path] | None = ...,
        test_iteration: int = ...,
        enable_coverage: Literal[False] = ...,
    ) -> TestResults: ...

    @overload
    def run_tests(
        self,
        test_config: TestConfig,
        test_files: list[Path] | None = ...,
        test_iteration: int = ...,
        enable_coverage: Literal[True] = ...,
    ) -> tuple[TestResults, CoverageData | None]: ...

    def run_tests(
        self,
        test_config: TestConfig,
        test_files: list[Path] | None = None,
        test_iteration: int = 0,
        enable_coverage: bool = False,
    ) -> TestResults | tuple[TestResults, CoverageData | None]:
        """Run tests and return structured results.

        If test_files is provided, run only those files.
        Otherwise discover test files from test_config.
        test_iteration is passed as CODEFLASH_TEST_ITERATION env var.
        When enable_coverage is True, returns (TestResults, CoverageData | None).
        """
        ...

    def replace_function(self, file: Path, function: FunctionToOptimize, new_code: str) -> None:
        """Replace a function's source code in a file."""
        ...

    def restore_function(self, file: Path, function: FunctionToOptimize, original_code: str) -> None:
        """Restore a function's original source code in a file."""
        ...

    def run_benchmarks(
        self,
        function: FunctionToOptimize,
        test_config: TestConfig,
        test_files: list[Path] | None = None,
        test_iteration: int = 0,
    ) -> BenchmarkResults:
        """Run benchmarks for a function and return timing data.

        If test_files is provided, run only those files.
        Otherwise discover test files from test_config.
        test_iteration is passed as CODEFLASH_TEST_ITERATION env var.
        """
        ...

    def format_code(self, code: str, file: Path) -> str:
        """Format code according to the project's style."""
        ...

    def validate_candidate(self, code: str) -> bool:
        """Return True if the candidate code is syntactically valid."""
        ...

    def normalize_code(self, code: str) -> str:
        """Normalize code for deduplication (e.g. strip comments, whitespace, docstrings)."""
        ...

    # -- Phase 1: Test Generation -----------------------------------------------

    def generate_tests(
        self, function: FunctionToOptimize, context: CodeContext, test_config: TestConfig, trace_id: str = ""
    ) -> GeneratedTestSuite | None:
        """Generate regression tests for the target function."""
        ...

    # -- Phase 2: Split behavioral / performance test running --------------------

    def run_behavioral_tests(self, test_files: list[Path], test_config: TestConfig) -> TestResults:
        """Run behavioral tests and return pass/fail with captured outputs."""
        ...

    def run_performance_tests(
        self, test_files: list[Path], function: FunctionToOptimize, test_config: TestConfig
    ) -> BenchmarkResults:
        """Run performance-instrumented tests and return timing data."""
        ...

    # -- Phase 3: Multi-round candidate generation ------------------------------

    def run_line_profiler(
        self, function: FunctionToOptimize, test_config: TestConfig, test_files: list[Path] | None = None
    ) -> str:
        """Run line profiler on the function and return formatted profiler output.

        Returns an empty string if profiling is not possible (e.g. JIT-decorated code).
        """
        ...

    def get_line_profiler_candidates(
        self, context: CodeContext, line_profile_data: str, trace_id: str = ""
    ) -> list[Candidate]:
        """Generate candidates guided by line profiler hotspot data."""
        ...

    def repair_candidate(
        self, context: CodeContext, candidate: Candidate, test_diffs: list[TestDiff], trace_id: str = ""
    ) -> Candidate | None:
        """Fix a failing candidate using test failure info."""
        ...

    def refine_candidate(
        self, context: CodeContext, candidate: ScoredCandidate, baseline_bench: BenchmarkResults, trace_id: str = ""
    ) -> list[Candidate]:
        """Refine a passing candidate for further improvement."""
        ...

    def adaptive_optimize(
        self, context: CodeContext, scored: list[ScoredCandidate], trace_id: str = ""
    ) -> Candidate | None:
        """Combine insights from evaluated candidates."""
        ...

    # -- Phase 4: Test review & repair ------------------------------------------

    def review_generated_tests(
        self, suite: GeneratedTestSuite, context: CodeContext, test_results: TestResults, trace_id: str = ""
    ) -> list[TestReviewResult]:
        """Review generated tests for quality issues."""
        ...

    def repair_generated_tests(
        self,
        suite: GeneratedTestSuite,
        reviews: list[TestReviewResult],
        context: CodeContext,
        trace_id: str = "",
        previous_repair_errors: dict[str, str] | None = None,
        coverage_data: CoverageData | None = None,
    ) -> GeneratedTestSuite | None:
        """Repair generated tests based on review feedback."""
        ...

    # -- Phase 5: AI-assisted ranking & explanation -----------------------------

    def rank_candidates(
        self, scored: list[ScoredCandidate], context: CodeContext, trace_id: str = ""
    ) -> list[int] | None:
        """Rank candidates using AI. Returns indices in decreasing preference order, or None."""
        ...

    def generate_explanation(
        self, result: OptimizationResult, context: CodeContext, trace_id: str = "", annotated_tests: str = ""
    ) -> str:
        """Generate a human-readable explanation for the winning optimization."""
        ...

    # -- Cleanup & environment -------------------------------------------------

    def cleanup_run(self, tests_root: Path) -> None:
        """Clean up leftover files from previous or current runs.

        Called before and after the optimization loop. Implementations should
        remove instrumented test files, temporary return-value files, trace
        files, and any shared temp directories their tooling creates.
        """
        ...

    def compare_outputs(self, baseline_output: object, candidate_output: object) -> bool:
        """Compare two captured test outputs for equivalence.

        Called during verification to decide whether a candidate preserved
        behavior. Implementations may use deep/structural comparison
        (e.g. handling NaN, custom objects). The default core fallback is ``==``.
        """
        ...

    def validate_environment(self, config: Any) -> bool:
        """Validate that the environment is ready to run optimizations.

        Called before the optimization loop starts. Implementations should
        check that required tools (formatters, test runners, etc.) are
        installed and accessible. Return True if everything is OK.
        """
        ...

    # -- Phase 6: PR creation & result logging ----------------------------------

    def create_pr(
        self,
        result: OptimizationResult,
        context: CodeContext,
        trace_id: str = "",
        generated_tests: GeneratedTestSuite | None = None,
    ) -> str | None:
        """Create a pull request with the optimization. Returns the PR URL or None."""
        ...

    def log_results(
        self,
        result: OptimizationResult,
        trace_id: str,
        all_speedups: dict[str, float] | None = None,
        all_runtimes: dict[str, float] | None = None,
        all_correct: dict[str, bool] | None = None,
    ) -> None:
        """Log optimization results to the backend."""
        ...
