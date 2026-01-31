from __future__ import annotations

import ast
import concurrent.futures
import logging
import os
import queue
import random
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import libcst as cst
import sentry_sdk
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

from codeflash.api.aiservice import AiServiceClient, AIServiceRefinerRequest, LocalAiServiceClient
from codeflash.api.cfapi import add_code_context_hash, create_staging, get_cfapi_base_urls, mark_optimization_success
from codeflash.benchmarking.utils import process_benchmark_data
from codeflash.cli_cmds.console import code_print, console, logger, lsp_log, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_extractor import get_opt_review_metrics, is_numerical_code
from codeflash.code_utils.code_replacer import (
    add_custom_marker_to_all_tests,
    modify_autouse_fixture,
    replace_function_definitions_in_module,
)
from codeflash.code_utils.code_utils import (
    choose_weights,
    cleanup_paths,
    create_rank_dictionary_compact,
    create_score_dictionary_from_metrics,
    diff_length,
    encoded_tokens_len,
    extract_unique_errors,
    file_name_from_test_module_name,
    get_run_tmp_file,
    normalize_by_max,
    restore_conftest,
    unified_diff_strings,
)
from codeflash.code_utils.config_consts import (
    COVERAGE_THRESHOLD,
    INDIVIDUAL_TESTCASE_TIMEOUT,
    MIN_CORRECT_CANDIDATES,
    OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    REFINED_CANDIDATE_RANKING_WEIGHTS,
    REPEAT_OPTIMIZATION_PROBABILITY,
    TOTAL_LOOPING_TIME_EFFECTIVE,
    EffortKeys,
    EffortLevel,
    get_effort_value,
)
from codeflash.code_utils.deduplicate_code import normalize_code
from codeflash.code_utils.edit_generated_tests import (
    add_runtime_comments_to_generated_tests,
    disable_ts_check,
    inject_test_globals,
    normalize_generated_tests_imports,
    remove_functions_from_generated_tests,
)
from codeflash.code_utils.env_utils import get_pr_number
from codeflash.code_utils.formatter import format_code, format_generated_code, sort_imports
from codeflash.code_utils.git_utils import git_root_dir
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.code_utils.line_profile_utils import add_decorator_imports, contains_jit_decorator
from codeflash.code_utils.static_analysis import get_first_top_level_function_or_method_ast
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.context import code_context_extractor
from codeflash.context.unused_definition_remover import detect_unused_helper_functions, revert_unused_helper_functions
from codeflash.discovery.functions_to_optimize import was_function_previously_optimized
from codeflash.either import Failure, Success, is_successful
from codeflash.languages import is_java, is_python
from codeflash.languages.base import FunctionInfo, Language
from codeflash.languages.current import current_language_support, is_typescript
from codeflash.languages.javascript.module_system import detect_module_system
from codeflash.lsp.helpers import is_LSP_enabled, report_to_markdown_table, tree_to_markdown
from codeflash.lsp.lsp_message import LspCodeMessage, LspMarkdownMessage, LSPMessageId
from codeflash.models.ExperimentMetadata import ExperimentMetadata
from codeflash.models.models import (
    AdaptiveOptimizedCandidate,
    AIServiceAdaptiveOptimizeRequest,
    AIServiceCodeRepairRequest,
    BestOptimization,
    CandidateEvaluationContext,
    CodeOptimizationContext,
    GeneratedTests,
    GeneratedTestsList,
    OptimizationReviewResult,
    OptimizationSet,
    OptimizedCandidate,
    OptimizedCandidateResult,
    OptimizedCandidateSource,
    OriginalCodeBaseline,
    TestFile,
    TestFiles,
    TestingMode,
    TestResults,
    TestType,
)
from codeflash.result.create_pr import check_create_pr, existing_tests_source_for
from codeflash.result.critic import (
    concurrency_gain,
    coverage_critic,
    get_acceptance_reason,
    performance_gain,
    quantity_of_tests_critic,
    speedup_critic,
    throughput_gain,
)
from codeflash.result.explanation import Explanation
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.concolic_testing import generate_concolic_tests
from codeflash.verification.equivalence import compare_test_results
from codeflash.verification.instrument_codeflash_capture import instrument_codeflash_capture
from codeflash.verification.parse_line_profile_test_output import parse_line_profile_results
from codeflash.verification.parse_test_output import (
    calculate_function_throughput_from_test_results,
    parse_concurrency_metrics,
    parse_test_results,
)
from codeflash.verification.test_runner import run_behavioral_tests, run_benchmarking_tests, run_line_profile_tests
from codeflash.verification.verification_utils import get_test_file_path
from codeflash.verification.verifier import generate_tests

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.either import Result
    from codeflash.models.models import (
        BenchmarkKey,
        CodeStringsMarkdown,
        ConcurrencyMetrics,
        CoverageData,
        FunctionCalledInTest,
        FunctionSource,
        TestDiff,
    )
    from codeflash.verification.verification_utils import TestConfig


def log_optimization_context(function_name: str, code_context: CodeOptimizationContext) -> None:
    """Log optimization context details when in verbose mode using Rich formatting."""
    if logger.getEffectiveLevel() > logging.DEBUG:
        return

    console.rule()
    read_writable_tokens = encoded_tokens_len(code_context.read_writable_code.markdown)
    read_only_tokens = (
        encoded_tokens_len(code_context.read_only_context_code) if code_context.read_only_context_code else 0
    )
    total_tokens = read_writable_tokens + read_only_tokens
    token_pct = min(total_tokens / OPTIMIZATION_CONTEXT_TOKEN_LIMIT, 1.0)

    # Token bar color based on usage
    bar_color = "green" if token_pct < 0.7 else "yellow" if token_pct < 0.9 else "red"

    # Build compact info line
    helper_names = [hf.qualified_name for hf in code_context.helper_functions]
    helpers_str = f"[magenta]{', '.join(helper_names)}[/]" if helper_names else "[dim]none[/]"
    read_writable_files = [str(cs.file_path) for cs in code_context.read_writable_code.code_strings]

    # Create a tree view for the context
    tree = Tree(f"[bold cyan]Context for {function_name}[/]")
    tree.add(
        Text.assemble(
            ("Tokens: ", "dim"),
            (f"{total_tokens:,}", "bold " + bar_color),
            (f"/{OPTIMIZATION_CONTEXT_TOKEN_LIMIT:,} ", "dim"),
            (f"({token_pct:.0%})", bar_color),
            ("  [", "dim"),
            (f"{read_writable_tokens:,}", "green"),
            (" rw", "dim green"),
            (" + ", "dim"),
            (f"{read_only_tokens:,}", "yellow"),
            (" ro", "dim yellow"),
            ("]", "dim"),
        )
    )
    tree.add(f"[dim]Helpers:[/] {helpers_str}")
    files_branch = tree.add("[dim]Files:[/]")
    for f in read_writable_files:
        files_branch.add(f"[blue]{f}[/]")

    console.print(tree)

    console.print(
        Panel(
            Syntax(code_context.read_writable_code.markdown, "markdown", theme="monokai", word_wrap=True),
            title="[green]Read-Writable Code[/]",
            border_style="green",
        )
    )

    if code_context.read_only_context_code:
        console.print(
            Panel(
                Syntax(code_context.read_only_context_code, "markdown", theme="monokai", word_wrap=True),
                title="[yellow]Read-Only Dependencies[/]",
                border_style="yellow",
            )
        )


class CandidateNode:
    __slots__ = ("candidate", "children", "parent")

    def __init__(self, candidate: OptimizedCandidate) -> None:
        self.candidate = candidate
        self.parent: CandidateNode | None = None
        self.children: list[CandidateNode] = []

    def is_leaf(self) -> bool:
        return not self.children

    def path_to_root(self) -> list[OptimizedCandidate]:
        path = []
        node: CandidateNode | None = self
        while node:
            path.append(node.candidate)
            node = node.parent
        return path[::-1]


class CandidateForest:
    def __init__(self) -> None:
        self.nodes: dict[str, CandidateNode] = {}

    def add(self, candidate: OptimizedCandidate) -> CandidateNode:
        cid = candidate.optimization_id
        pid = candidate.parent_id

        node = self.nodes.get(cid)
        if node is None:
            node = CandidateNode(candidate)
            self.nodes[cid] = node

        if pid is not None:
            parent = self.nodes.get(pid)
            if parent is None:
                parent = CandidateNode(candidate=None)  # placeholder
                self.nodes[pid] = parent

            node.parent = parent
            parent.children.append(node)

        return node

    def get_node(self, cid: str) -> CandidateNode | None:
        return self.nodes.get(cid)


class CandidateProcessor:
    """Handles candidate processing using a queue-based approach."""

    def __init__(
        self,
        initial_candidates: list[OptimizedCandidate],
        future_line_profile_results: concurrent.futures.Future,
        eval_ctx: CandidateEvaluationContext,
        effort: str,
        original_markdown_code: str,
        future_all_refinements: list[concurrent.futures.Future],
        future_all_code_repair: list[concurrent.futures.Future],
        future_adaptive_optimizations: list[concurrent.futures.Future],
    ) -> None:
        self.candidate_queue = queue.Queue()
        self.forest = CandidateForest()
        self.line_profiler_done = False
        self.refinement_done = False
        self.eval_ctx = eval_ctx
        self.effort = effort
        self.candidate_len = len(initial_candidates)
        self.refinement_calls_count = 0
        self.original_markdown_code = original_markdown_code

        # Initialize queue with initial candidates
        for candidate in initial_candidates:
            self.forest.add(candidate)
            self.candidate_queue.put(candidate)

        self.future_line_profile_results = future_line_profile_results
        self.future_all_refinements = future_all_refinements
        self.future_all_code_repair = future_all_code_repair
        self.future_adaptive_optimizations = future_adaptive_optimizations

    def get_total_llm_calls(self) -> int:
        return self.refinement_calls_count

    def get_next_candidate(self) -> CandidateNode | None:
        """Get the next candidate from the queue, handling async results as needed."""
        try:
            return self.forest.get_node(self.candidate_queue.get_nowait().optimization_id)
        except queue.Empty:
            return self._handle_empty_queue()

    def _handle_empty_queue(self) -> CandidateNode | None:
        """Handle empty queue by checking for pending async results."""
        if not self.line_profiler_done:
            return self._process_candidates(
                [self.future_line_profile_results],
                "all candidates processed, await candidates from line profiler",
                "Added results from line profiler to candidates, total candidates now: {1}",
                lambda: setattr(self, "line_profiler_done", True),
            )
        if len(self.future_all_code_repair) > 0:
            return self._process_candidates(
                self.future_all_code_repair,
                "Repairing {0} candidates",
                "Added {0} candidates from repair, total candidates now: {1}",
                lambda: self.future_all_code_repair.clear(),
            )
        if self.line_profiler_done and not self.refinement_done:
            return self._process_candidates(
                self.future_all_refinements,
                "Refining generated code for improved quality and performance...",
                "Added {0} candidates from refinement, total candidates now: {1}",
                lambda: setattr(self, "refinement_done", True),
                filter_candidates_func=self._filter_refined_candidates,
            )
        if len(self.future_adaptive_optimizations) > 0:
            return self._process_candidates(
                self.future_adaptive_optimizations,
                "Applying adaptive optimizations to {0} candidates",
                "Added {0} candidates from adaptive optimization, total candidates now: {1}",
                lambda: self.future_adaptive_optimizations.clear(),
            )
        return None  # All done

    def _process_candidates(
        self,
        future_candidates: list[concurrent.futures.Future],
        loading_msg: str,
        success_msg: str,
        callback: Callable[[], None],
        filter_candidates_func: Callable[[list[OptimizedCandidate]], list[OptimizedCandidate]] | None = None,
    ) -> CandidateNode | None:
        if len(future_candidates) == 0:
            return None
        with progress_bar(
            loading_msg.format(len(future_candidates)), transient=True, revert_to_print=bool(get_pr_number())
        ):
            concurrent.futures.wait(future_candidates)
            candidates: list[OptimizedCandidate] = []
            for future_c in future_candidates:
                candidate_result = future_c.result()
                if not candidate_result:
                    continue

                if isinstance(candidate_result, list):
                    candidates.extend(candidate_result)
                else:
                    candidates.append(candidate_result)

            candidates = filter_candidates_func(candidates) if filter_candidates_func else candidates
            for candidate in candidates:
                self.forest.add(candidate)
                self.candidate_queue.put(candidate)
                self.candidate_len += 1

            if len(candidates) > 0:
                logger.info(success_msg.format(len(candidates), self.candidate_len))

            callback()
            return self.get_next_candidate()

    def _filter_refined_candidates(self, candidates: list[OptimizedCandidate]) -> list[OptimizedCandidate]:
        """We generate a weighted ranking based on the runtime and diff lines and select the best of valid optimizations to be tested."""
        self.refinement_calls_count += len(candidates)

        top_n_candidates = int(
            min(int(get_effort_value(EffortKeys.TOP_VALID_CANDIDATES_FOR_REFINEMENT, self.effort)), len(candidates))
        )

        if len(candidates) == top_n_candidates:
            # no need for ranking since we will return all candidates
            return candidates

        diff_lens_list = []
        runtimes_list = []
        for c in candidates:
            # current refined candidates is not benchmarked yet, a close values we would expect to be the parent candidate
            parent_id = c.parent_id
            parent_candidate_node = self.forest.get_node(parent_id)
            parent_optimized_runtime = self.eval_ctx.get_optimized_runtime(parent_id)
            if not parent_optimized_runtime or not parent_candidate_node:
                continue
            diff_lens_list.append(
                diff_length(self.original_markdown_code, parent_candidate_node.candidate.source_code.markdown)
            )
            runtimes_list.append(parent_optimized_runtime)

        if not runtimes_list or not diff_lens_list:
            # should not happen
            logger.warning("No valid candidates for refinement while filtering")
            return candidates

        runtime_w, diff_w = REFINED_CANDIDATE_RANKING_WEIGHTS
        weights = choose_weights(runtime=runtime_w, diff=diff_w)

        runtime_norm = normalize_by_max(runtimes_list)
        diffs_norm = normalize_by_max(diff_lens_list)
        # the lower the better
        score_dict = create_score_dictionary_from_metrics(weights, runtime_norm, diffs_norm)
        top_indecies = sorted(score_dict, key=score_dict.get)[:top_n_candidates]

        return [candidates[idx] for idx in top_indecies]

    def is_done(self) -> bool:
        """Check if processing is complete."""
        return (
            self.line_profiler_done
            and self.refinement_done
            and len(self.future_all_code_repair) == 0
            and len(self.future_adaptive_optimizations) == 0
            and self.candidate_queue.empty()
        )


class FunctionOptimizer:
    def __init__(
        self,
        function_to_optimize: FunctionToOptimize,
        test_cfg: TestConfig,
        function_to_optimize_source_code: str = "",
        function_to_tests: dict[str, set[FunctionCalledInTest]] | None = None,
        function_to_optimize_ast: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
        aiservice_client: AiServiceClient | None = None,
        function_benchmark_timings: dict[BenchmarkKey, int] | None = None,
        total_benchmark_timings: dict[BenchmarkKey, int] | None = None,
        args: Namespace | None = None,
        replay_tests_dir: Path | None = None,
    ) -> None:
        self.project_root = test_cfg.project_root_path
        self.test_cfg = test_cfg
        self.aiservice_client = aiservice_client if aiservice_client else AiServiceClient()
        self.function_to_optimize = function_to_optimize
        self.function_to_optimize_source_code = (
            function_to_optimize_source_code
            if function_to_optimize_source_code
            else function_to_optimize.file_path.read_text(encoding="utf8")
        )
        self.language_support = current_language_support()
        if not function_to_optimize_ast:
            # Skip Python AST parsing for non-Python languages
            if not is_python():
                self.function_to_optimize_ast = None
            else:
                original_module_ast = ast.parse(function_to_optimize_source_code)
                self.function_to_optimize_ast = get_first_top_level_function_or_method_ast(
                    function_to_optimize.function_name, function_to_optimize.parents, original_module_ast
                )
        else:
            self.function_to_optimize_ast = function_to_optimize_ast
        self.function_to_tests = function_to_tests if function_to_tests else {}

        self.experiment_id = os.getenv("CODEFLASH_EXPERIMENT_ID", None)
        self.local_aiservice_client = LocalAiServiceClient() if self.experiment_id else None
        self.test_files = TestFiles(test_files=[])

        self.effort = getattr(args, "effort", EffortLevel.MEDIUM.value) if args else EffortLevel.MEDIUM.value

        self.args = args  # Check defaults for these
        self.function_trace_id: str = str(uuid.uuid4())
        # Get module path using language support (handles Python vs JavaScript differences)
        self.original_module_path = self.language_support.get_module_path(
            source_file=self.function_to_optimize.file_path,
            project_root=self.project_root,
            tests_root=test_cfg.tests_root,
        )

        self.function_benchmark_timings = function_benchmark_timings if function_benchmark_timings else {}
        self.total_benchmark_timings = total_benchmark_timings if total_benchmark_timings else {}
        self.replay_tests_dir = replay_tests_dir if replay_tests_dir else None
        n_tests = get_effort_value(EffortKeys.N_GENERATED_TESTS, self.effort)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=n_tests + 3 if self.experiment_id is None else n_tests + 4
        )
        self.optimization_review = ""
        self.future_all_code_repair: list[concurrent.futures.Future] = []
        self.future_all_refinements: list[concurrent.futures.Future] = []
        self.future_adaptive_optimizations: list[concurrent.futures.Future] = []
        self.repair_counter = 0  # track how many repairs we did for each function
        self.adaptive_optimization_counter = 0  # track how many adaptive optimizations we did for each function
        self.is_numerical_code: bool | None = None
        self.code_already_exists: bool = False

    def can_be_optimized(self) -> Result[tuple[bool, CodeOptimizationContext, dict[Path, str]], str]:
        should_run_experiment = self.experiment_id is not None
        logger.info(f"!lsp|Function Trace ID: {self.function_trace_id}")
        ph("cli-optimize-function-start", {"function_trace_id": self.function_trace_id})
        self.cleanup_leftover_test_return_values()
        file_name_from_test_module_name.cache_clear()
        ctx_result = self.get_code_optimization_context()
        if not is_successful(ctx_result):
            return Failure(ctx_result.failure())
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        log_optimization_context(self.function_to_optimize.function_name, code_context)
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code

        # Random here means that we still attempt optimization with a fractional chance to see if
        # last time we could not find an optimization, maybe this time we do.
        # Random is before as a performance optimization, swapping the two 'and' statements has the same effect
        self.code_already_exists = was_function_previously_optimized(self.function_to_optimize, code_context, self.args)
        if random.random() > REPEAT_OPTIMIZATION_PROBABILITY and self.code_already_exists:  # noqa: S311
            return Failure("Function optimization previously attempted, skipping.")

        return Success((should_run_experiment, code_context, original_helper_code))

    def generate_and_instrument_tests(
        self, code_context: CodeOptimizationContext
    ) -> Result[
        tuple[
            GeneratedTestsList,
            dict[str, set[FunctionCalledInTest]],
            str,
            list[Path],
            list[Path],
            set[Path],
            dict | None,
        ],
        str,
    ]:
        """Generate and instrument tests for the function."""
        n_tests = get_effort_value(EffortKeys.N_GENERATED_TESTS, self.effort)
        generated_test_paths = [
            get_test_file_path(
                self.test_cfg.tests_root, self.function_to_optimize.function_name, test_index, test_type="unit"
            )
            for test_index in range(n_tests)
        ]
        generated_perf_test_paths = [
            get_test_file_path(
                self.test_cfg.tests_root, self.function_to_optimize.function_name, test_index, test_type="perf"
            )
            for test_index in range(n_tests)
        ]

        # Note: JavaScript/TypeScript runtime is provided by codeflash npm package
        # which is installed automatically by test_runner.py._ensure_runtime_files()
        # No manual file copying is needed here.

        test_results = self.generate_tests(
            testgen_context=code_context.testgen_context,
            helper_functions=code_context.helper_functions,
            generated_test_paths=generated_test_paths,
            generated_perf_test_paths=generated_perf_test_paths,
        )

        if not is_successful(test_results):
            return Failure(test_results.failure())

        count_tests, generated_tests, function_to_concolic_tests, concolic_test_str = test_results.unwrap()

        # Normalize codeflash imports in JS/TS tests to use npm package
        if not is_python():
            module_system = detect_module_system(self.project_root)
            if module_system == "esm":
                generated_tests = inject_test_globals(generated_tests)
            if is_typescript():
                # disable ts check for typescript tests
                generated_tests = disable_ts_check(generated_tests)

            generated_tests = normalize_generated_tests_imports(generated_tests)

        logger.debug(f"[PIPELINE] Processing {count_tests} generated tests")
        for i, generated_test in enumerate(generated_tests.generated_tests):
            behavior_path = generated_test.behavior_file_path
            perf_path = generated_test.perf_file_path

            # For Java, fix paths to match package structure
            if is_java():
                behavior_path, perf_path = self._fix_java_test_paths(
                    generated_test.instrumented_behavior_test_source,
                    generated_test.instrumented_perf_test_source,
                )
                generated_test.behavior_file_path = behavior_path
                generated_test.perf_file_path = perf_path

            logger.debug(
                f"[PIPELINE] Test {i + 1}: behavior_path={behavior_path}, perf_path={perf_path}"
            )

            with behavior_path.open("w", encoding="utf8") as f:
                f.write(generated_test.instrumented_behavior_test_source)
            logger.debug(f"[PIPELINE] Wrote behavioral test to {behavior_path}")

            with perf_path.open("w", encoding="utf8") as f:
                f.write(generated_test.instrumented_perf_test_source)
            logger.debug(f"[PIPELINE] Wrote perf test to {perf_path}")

            # File paths are expected to be absolute - resolved at their source (CLI, TestConfig, etc.)
            test_file_obj = TestFile(
                instrumented_behavior_file_path=generated_test.behavior_file_path,
                benchmarking_file_path=generated_test.perf_file_path,
                original_file_path=None,
                original_source=generated_test.generated_original_test_source,
                test_type=TestType.GENERATED_REGRESSION,
                tests_in_file=None,  # This is currently unused. We can discover the tests in the file if needed.
            )
            self.test_files.add(test_file_obj)
            logger.debug(
                f"[PIPELINE] Added test file to collection: behavior={test_file_obj.instrumented_behavior_file_path}, perf={test_file_obj.benchmarking_file_path}"
            )

            logger.info(f"Generated test {i + 1}/{count_tests}:")
            # Use correct extension based on language
            test_ext = self.language_support.get_test_file_suffix()
            code_print(
                generated_test.generated_original_test_source,
                file_name=f"test_{i + 1}{test_ext}",
                language=self.function_to_optimize.language,
            )
        if concolic_test_str:
            logger.info(f"Generated test {count_tests}/{count_tests}:")
            code_print(concolic_test_str, language=self.function_to_optimize.language)

        function_to_all_tests = {
            key: self.function_to_tests.get(key, set()) | function_to_concolic_tests.get(key, set())
            for key in set(self.function_to_tests) | set(function_to_concolic_tests)
        }
        instrumented_unittests_created_for_function = self.instrument_existing_tests(function_to_all_tests)

        original_conftest_content = None
        if self.args.override_fixtures:
            logger.info("Disabling all autouse fixtures associated with the generated test files")
            original_conftest_content = modify_autouse_fixture(generated_test_paths + generated_perf_test_paths)
            logger.info("Add custom marker to generated test files")
            add_custom_marker_to_all_tests(generated_test_paths + generated_perf_test_paths)

        return Success(
            (
                generated_tests,
                function_to_concolic_tests,
                concolic_test_str,
                generated_test_paths,
                generated_perf_test_paths,
                instrumented_unittests_created_for_function,
                original_conftest_content,
            )
        )

    def _fix_java_test_paths(
        self, behavior_source: str, perf_source: str
    ) -> tuple[Path, Path]:
        """Fix Java test file paths to match package structure.

        Java requires test files to be in directories matching their package.
        This method extracts the package and class from the generated tests
        and returns correct paths.

        Args:
            behavior_source: Source code of the behavior test.
            perf_source: Source code of the performance test.

        Returns:
            Tuple of (behavior_path, perf_path) with correct package structure.

        """
        import re

        # Extract package from behavior source
        package_match = re.search(r'^\s*package\s+([\w.]+)\s*;', behavior_source, re.MULTILINE)
        package_name = package_match.group(1) if package_match else ""

        # Extract class name from behavior source
        # Use more specific pattern to avoid matching words like "command" or text in comments
        class_match = re.search(r'^(?:public\s+)?class\s+(\w+)', behavior_source, re.MULTILINE)
        behavior_class = class_match.group(1) if class_match else "GeneratedTest"

        # Extract class name from perf source
        perf_class_match = re.search(r'^(?:public\s+)?class\s+(\w+)', perf_source, re.MULTILINE)
        perf_class = perf_class_match.group(1) if perf_class_match else "GeneratedPerfTest"

        # Build paths with package structure
        test_dir = self.test_cfg.tests_root

        if package_name:
            package_path = package_name.replace(".", "/")
            behavior_path = test_dir / package_path / f"{behavior_class}.java"
            perf_path = test_dir / package_path / f"{perf_class}.java"
        else:
            behavior_path = test_dir / f"{behavior_class}.java"
            perf_path = test_dir / f"{perf_class}.java"

        # Create directories if needed
        behavior_path.parent.mkdir(parents=True, exist_ok=True)
        perf_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"[JAVA] Fixed paths: behavior={behavior_path}, perf={perf_path}")
        return behavior_path, perf_path

    # note: this isn't called by the lsp, only called by cli
    def optimize_function(self) -> Result[BestOptimization, str]:
        initialization_result = self.can_be_optimized()
        if not is_successful(initialization_result):
            return Failure(initialization_result.failure())
        should_run_experiment, code_context, original_helper_code = initialization_result.unwrap()
        self.is_numerical_code = is_numerical_code(code_string=code_context.read_writable_code.flat)
        code_print(
            code_context.read_writable_code.flat,
            file_name=self.function_to_optimize.file_path,
            function_name=self.function_to_optimize.function_name,
            language=self.function_to_optimize.language,
        )

        with progress_bar(
            f"Generating new tests and optimizations for function '{self.function_to_optimize.function_name}'",
            transient=True,
            revert_to_print=bool(get_pr_number()),
        ):
            console.rule()
            new_code_context = code_context
            if (
                self.is_numerical_code and not self.args.no_jit_opts
            ):  # if the code is numerical in nature (uses numpy/tensorflow/math/pytorch/jax)
                jit_compiled_opt_candidate = self.aiservice_client.get_jit_rewritten_code(
                    code_context.read_writable_code.markdown, self.function_trace_id
                )
                if jit_compiled_opt_candidate:  # jit rewrite was successful
                    # write files
                    # Try to replace function with optimized code
                    self.replace_function_and_helpers_with_optimized_code(
                        code_context=code_context,
                        optimized_code=jit_compiled_opt_candidate[0].source_code,
                        original_helper_code=original_helper_code,
                    )
                    # get code context
                    try:
                        new_code_context = self.get_code_optimization_context().unwrap()
                    except Exception as e:
                        sentry_sdk.capture_exception(e)
                        logger.debug("!lsp|Getting new code context failed, revert to original one")
                    # unwrite files
                    self.write_code_and_helpers(
                        self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                    )
            # Generate tests and optimizations in parallel
            future_tests = self.executor.submit(self.generate_and_instrument_tests, new_code_context)
            future_optimizations = self.executor.submit(
                self.generate_optimizations,
                read_writable_code=code_context.read_writable_code,
                read_only_context_code=code_context.read_only_context_code,
                run_experiment=should_run_experiment,
                is_numerical_code=self.is_numerical_code and not self.args.no_jit_opts,
            )

            concurrent.futures.wait([future_tests, future_optimizations])

            test_setup_result = future_tests.result()
            optimization_result = future_optimizations.result()
            console.rule()

        if not is_successful(test_setup_result):
            return Failure(test_setup_result.failure())

        if not is_successful(optimization_result):
            return Failure(optimization_result.failure())

        (
            generated_tests,
            function_to_concolic_tests,
            concolic_test_str,
            generated_test_paths,
            generated_perf_test_paths,
            instrumented_unittests_created_for_function,
            original_conftest_content,
        ) = test_setup_result.unwrap()

        optimizations_set, function_references = optimization_result.unwrap()

        baseline_setup_result = self.setup_and_establish_baseline(
            code_context=code_context,
            original_helper_code=original_helper_code,
            function_to_concolic_tests=function_to_concolic_tests,
            generated_test_paths=generated_test_paths,
            generated_perf_test_paths=generated_perf_test_paths,
            instrumented_unittests_created_for_function=instrumented_unittests_created_for_function,
            original_conftest_content=original_conftest_content,
        )

        if not is_successful(baseline_setup_result):
            return Failure(baseline_setup_result.failure())

        (
            function_to_optimize_qualified_name,
            function_to_all_tests,
            original_code_baseline,
            test_functions_to_remove,
            file_path_to_helper_classes,
        ) = baseline_setup_result.unwrap()

        best_optimization = self.find_and_process_best_optimization(
            optimizations_set=optimizations_set,
            code_context=code_context,
            original_code_baseline=original_code_baseline,
            original_helper_code=original_helper_code,
            file_path_to_helper_classes=file_path_to_helper_classes,
            function_to_optimize_qualified_name=function_to_optimize_qualified_name,
            function_to_all_tests=function_to_all_tests,
            generated_tests=generated_tests,
            test_functions_to_remove=test_functions_to_remove,
            concolic_test_str=concolic_test_str,
            function_references=function_references,
        )

        # Add function to code context hash if in gh actions and code doesn't already exist
        if not self.code_already_exists:
            add_code_context_hash(code_context.hashing_code_context_hash)

        if self.args.override_fixtures:
            restore_conftest(original_conftest_content)
        if not best_optimization:
            return Failure(f"No best optimizations found for function {self.function_to_optimize.qualified_name}")
        return Success(best_optimization)

    def get_trace_id(self, exp_type: str) -> str:
        """Get the trace ID for the current experiment type."""
        if self.experiment_id:
            return self.function_trace_id[:-4] + exp_type
        return self.function_trace_id

    def build_runtime_info_tree(
        self,
        candidate_index: int,
        candidate_result: OptimizedCandidateResult,
        original_code_baseline: OriginalCodeBaseline,
        perf_gain: float,
        *,
        is_successful_candidate: bool,
    ) -> Tree:
        """Build a Tree display for runtime information of a candidate."""
        tree = Tree(f"Candidate #{candidate_index} - Runtime Information ⌛")

        is_async = original_code_baseline.async_throughput is not None and candidate_result.async_throughput is not None

        if is_successful_candidate:
            if is_async:
                throughput_gain_value = throughput_gain(
                    original_throughput=original_code_baseline.async_throughput,
                    optimized_throughput=candidate_result.async_throughput,
                )
                tree.add("This candidate has better async throughput than the original code. 🚀")
                tree.add(f"Original async throughput: {original_code_baseline.async_throughput} executions")
                tree.add(f"Optimized async throughput: {candidate_result.async_throughput} executions")
                tree.add(f"Throughput improvement: {throughput_gain_value * 100:.1f}%")
                tree.add(f"Throughput ratio: {throughput_gain_value + 1:.3f}X")

                # Display concurrency metrics if available
                if candidate_result.concurrency_metrics and original_code_baseline.concurrency_metrics:
                    orig_ratio = original_code_baseline.concurrency_metrics.concurrency_ratio
                    cand_ratio = candidate_result.concurrency_metrics.concurrency_ratio
                    conc_gain = ((cand_ratio - orig_ratio) / orig_ratio * 100) if orig_ratio > 0 else 0
                    tree.add(f"Concurrency ratio: {orig_ratio:.2f}x → {cand_ratio:.2f}x ({conc_gain:+.1f}%)")
            else:
                tree.add("This candidate is faster than the original code. 🚀")
                tree.add(f"Original summed runtime: {humanize_runtime(original_code_baseline.runtime)}")
                tree.add(
                    f"Best summed runtime: {humanize_runtime(candidate_result.best_test_runtime)} "
                    f"(measured over {candidate_result.max_loop_count} "
                    f"loop{'s' if candidate_result.max_loop_count > 1 else ''})"
                )
                tree.add(f"Speedup percentage: {perf_gain * 100:.1f}%")
                tree.add(f"Speedup ratio: {perf_gain + 1:.3f}X")
        # Not a successful optimization candidate
        elif is_async:
            throughput_gain_value = throughput_gain(
                original_throughput=original_code_baseline.async_throughput,
                optimized_throughput=candidate_result.async_throughput,
            )
            tree.add(f"Async throughput: {candidate_result.async_throughput} executions")
            tree.add(f"Throughput change: {throughput_gain_value * 100:.1f}%")

            # Display concurrency metrics if available
            if candidate_result.concurrency_metrics and original_code_baseline.concurrency_metrics:
                orig_ratio = original_code_baseline.concurrency_metrics.concurrency_ratio
                cand_ratio = candidate_result.concurrency_metrics.concurrency_ratio
                conc_gain = ((cand_ratio - orig_ratio) / orig_ratio * 100) if orig_ratio > 0 else 0
                tree.add(f"Concurrency ratio: {orig_ratio:.2f}x → {cand_ratio:.2f}x ({conc_gain:+.1f}%)")

            tree.add(
                f"(Runtime for reference: {humanize_runtime(candidate_result.best_test_runtime)} over "
                f"{candidate_result.max_loop_count} loop{'s' if candidate_result.max_loop_count > 1 else ''})"
            )
        else:
            tree.add(
                f"Summed runtime: {humanize_runtime(candidate_result.best_test_runtime)} "
                f"(measured over {candidate_result.max_loop_count} "
                f"loop{'s' if candidate_result.max_loop_count > 1 else ''})"
            )
            tree.add(f"Speedup percentage: {perf_gain * 100:.1f}%")
            tree.add(f"Speedup ratio: {perf_gain + 1:.3f}X")

        return tree

    def handle_successful_candidate(
        self,
        candidate: OptimizedCandidate,
        candidate_result: OptimizedCandidateResult,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        candidate_index: int,
        eval_ctx: CandidateEvaluationContext,
    ) -> tuple[BestOptimization, Tree | None]:
        """Handle a successful optimization candidate.

        Returns the BestOptimization and optional benchmark tree.
        """
        with progress_bar("Running line-by-line profiling"):
            line_profile_test_results = self.line_profiler_step(
                code_context=code_context, original_helper_code=original_helper_code, candidate_index=candidate_index
            )

        eval_ctx.record_line_profiler_result(candidate.optimization_id, line_profile_test_results["str_out"])

        replay_perf_gain = {}
        benchmark_tree = None

        if self.args.benchmark:
            test_results_by_benchmark = candidate_result.benchmarking_test_results.group_by_benchmarks(
                self.total_benchmark_timings.keys(), self.replay_tests_dir, self.project_root
            )
            if len(test_results_by_benchmark) > 0:
                benchmark_tree = Tree("Speedup percentage on benchmarks:")
            for benchmark_key, candidate_test_results in test_results_by_benchmark.items():
                original_code_replay_runtime = original_code_baseline.replay_benchmarking_test_results[
                    benchmark_key
                ].total_passed_runtime()
                candidate_replay_runtime = candidate_test_results.total_passed_runtime()
                replay_perf_gain[benchmark_key] = performance_gain(
                    original_runtime_ns=original_code_replay_runtime, optimized_runtime_ns=candidate_replay_runtime
                )
                benchmark_tree.add(f"{benchmark_key}: {replay_perf_gain[benchmark_key] * 100:.1f}%")

        best_optimization = BestOptimization(
            candidate=candidate,
            helper_functions=code_context.helper_functions,
            code_context=code_context,
            runtime=candidate_result.best_test_runtime,
            line_profiler_test_results=line_profile_test_results,
            winning_behavior_test_results=candidate_result.behavior_test_results,
            replay_performance_gain=replay_perf_gain if self.args.benchmark else None,
            winning_benchmarking_test_results=candidate_result.benchmarking_test_results,
            winning_replay_benchmarking_test_results=candidate_result.benchmarking_test_results,
            async_throughput=candidate_result.async_throughput,
            concurrency_metrics=candidate_result.concurrency_metrics,
        )

        return best_optimization, benchmark_tree

    def select_best_optimization(
        self,
        eval_ctx: CandidateEvaluationContext,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        ai_service_client: AiServiceClient,
        exp_type: str,
        function_references: str,
    ) -> BestOptimization | None:
        """Select the best optimization from valid candidates."""
        if not eval_ctx.valid_optimizations:
            return None

        valid_candidates_with_shorter_code = []
        diff_lens_list = []  # character level diff
        speedups_list = []
        optimization_ids = []
        diff_strs = []
        runtimes_list = []

        for valid_opt in eval_ctx.valid_optimizations:
            valid_opt_normalized_code = normalize_code(valid_opt.candidate.source_code.flat.strip())
            new_candidate_with_shorter_code = OptimizedCandidate(
                source_code=eval_ctx.ast_code_to_id[valid_opt_normalized_code]["shorter_source_code"],
                optimization_id=valid_opt.candidate.optimization_id,
                explanation=valid_opt.candidate.explanation,
                source=valid_opt.candidate.source,
                parent_id=valid_opt.candidate.parent_id,
            )
            new_best_opt = BestOptimization(
                candidate=new_candidate_with_shorter_code,
                helper_functions=valid_opt.helper_functions,
                code_context=valid_opt.code_context,
                runtime=valid_opt.runtime,
                line_profiler_test_results=valid_opt.line_profiler_test_results,
                winning_behavior_test_results=valid_opt.winning_behavior_test_results,
                replay_performance_gain=valid_opt.replay_performance_gain,
                winning_benchmarking_test_results=valid_opt.winning_benchmarking_test_results,
                winning_replay_benchmarking_test_results=valid_opt.winning_replay_benchmarking_test_results,
                async_throughput=valid_opt.async_throughput,
                concurrency_metrics=valid_opt.concurrency_metrics,
            )
            valid_candidates_with_shorter_code.append(new_best_opt)
            diff_lens_list.append(
                diff_length(new_best_opt.candidate.source_code.flat, code_context.read_writable_code.flat)
            )
            diff_strs.append(
                unified_diff_strings(code_context.read_writable_code.flat, new_best_opt.candidate.source_code.flat)
            )
            speedups_list.append(
                1
                + performance_gain(
                    original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=new_best_opt.runtime
                )
            )
            optimization_ids.append(new_best_opt.candidate.optimization_id)
            runtimes_list.append(new_best_opt.runtime)

        if len(optimization_ids) > 1:
            future_ranking = self.executor.submit(
                ai_service_client.generate_ranking,
                diffs=diff_strs,
                optimization_ids=optimization_ids,
                speedups=speedups_list,
                trace_id=self.get_trace_id(exp_type),
                function_references=function_references,
            )
            concurrent.futures.wait([future_ranking])
            ranking = future_ranking.result()
            if ranking:
                min_key = ranking[0]
            else:
                diff_lens_ranking = create_rank_dictionary_compact(diff_lens_list)
                runtimes_ranking = create_rank_dictionary_compact(runtimes_list)
                overall_ranking = {key: diff_lens_ranking[key] + runtimes_ranking[key] for key in diff_lens_ranking}
                min_key = min(overall_ranking, key=overall_ranking.get)
        elif len(optimization_ids) == 1:
            min_key = 0
        else:
            return None

        return valid_candidates_with_shorter_code[min_key]

    def log_evaluation_results(
        self,
        eval_ctx: CandidateEvaluationContext,
        best_optimization: BestOptimization,
        original_code_baseline: OriginalCodeBaseline,
        ai_service_client: AiServiceClient,
        exp_type: str,
    ) -> None:
        """Log evaluation results to the AI service."""
        ai_service_client.log_results(
            function_trace_id=self.get_trace_id(exp_type),
            speedup_ratio=eval_ctx.speedup_ratios,
            original_runtime=original_code_baseline.runtime,
            optimized_runtime=eval_ctx.optimized_runtimes,
            is_correct=eval_ctx.is_correct,
            optimized_line_profiler_results=eval_ctx.optimized_line_profiler_results,
            optimizations_post=eval_ctx.optimizations_post,
            metadata={"best_optimization_id": best_optimization.candidate.optimization_id},
        )

    def process_single_candidate(
        self,
        candidate_node: CandidateNode,
        candidate_index: int,
        total_candidates: int,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        eval_ctx: CandidateEvaluationContext,
        exp_type: str,
        function_references: str,
    ) -> BestOptimization | None:
        """Process a single optimization candidate.

        Returns the BestOptimization if the candidate is successful, None otherwise.
        Updates eval_ctx with results and may append to all_refinements_data.
        """
        # Cleanup temp files
        get_run_tmp_file(Path(f"test_return_values_{candidate_index}.bin")).unlink(missing_ok=True)
        get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite")).unlink(missing_ok=True)

        logger.info(f"h3|Optimization candidate {candidate_index}/{total_candidates}:")
        candidate = candidate_node.candidate
        # Use correct extension based on language
        ext = self.language_support.file_extensions[0]
        code_print(
            candidate.source_code.flat,
            file_name=f"candidate_{candidate_index}{ext}",
            lsp_message_id=LSPMessageId.CANDIDATE.value,
            language=self.function_to_optimize.language,
        )

        # Try to replace function with optimized code
        try:
            did_update = self.replace_function_and_helpers_with_optimized_code(
                code_context=code_context,
                optimized_code=candidate.source_code,
                original_helper_code=original_helper_code,
            )
            if not did_update:
                logger.info("No functions were replaced in the optimized code. Skipping optimization candidate.")
                console.rule()
                return None
        except (ValueError, SyntaxError, cst.ParserSyntaxError, AttributeError) as e:
            logger.error(e)
            self.write_code_and_helpers(
                self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
            )
            return None

        # Check for duplicate candidates
        normalized_code = normalize_code(candidate.source_code.flat.strip())
        if normalized_code in eval_ctx.ast_code_to_id:
            logger.info("Current candidate has been encountered before in testing, Skipping optimization candidate.")
            eval_ctx.handle_duplicate_candidate(candidate, normalized_code, code_context)
            return None

        eval_ctx.register_new_candidate(normalized_code, candidate, code_context)

        # Run the optimized candidate
        run_results = self.run_optimized_candidate(
            optimization_candidate_index=candidate_index,
            baseline_results=original_code_baseline,
            original_helper_code=original_helper_code,
            file_path_to_helper_classes=file_path_to_helper_classes,
            eval_ctx=eval_ctx,
            code_context=code_context,
            candidate=candidate,
            exp_type=exp_type,
        )
        console.rule()

        if not is_successful(run_results):
            eval_ctx.record_failed_candidate(candidate.optimization_id)
            return None

        candidate_result: OptimizedCandidateResult = run_results.unwrap()
        perf_gain = performance_gain(
            original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=candidate_result.best_test_runtime
        )
        eval_ctx.record_successful_candidate(candidate.optimization_id, candidate_result.best_test_runtime, perf_gain)

        # Check if this is a successful optimization
        is_successful_opt = speedup_critic(
            candidate_result,
            original_code_baseline.runtime,
            best_runtime_until_now=None,
            original_async_throughput=original_code_baseline.async_throughput,
            best_throughput_until_now=None,
            original_concurrency_metrics=original_code_baseline.concurrency_metrics,
            best_concurrency_ratio_until_now=None,
        ) and quantity_of_tests_critic(candidate_result)

        tree = self.build_runtime_info_tree(
            candidate_index=candidate_index,
            candidate_result=candidate_result,
            original_code_baseline=original_code_baseline,
            perf_gain=perf_gain,
            is_successful_candidate=is_successful_opt,
        )

        best_optimization = None
        benchmark_tree = None

        if is_successful_opt:
            best_optimization, benchmark_tree = self.handle_successful_candidate(
                candidate=candidate,
                candidate_result=candidate_result,
                code_context=code_context,
                original_code_baseline=original_code_baseline,
                original_helper_code=original_helper_code,
                candidate_index=candidate_index,
                eval_ctx=eval_ctx,
            )
            eval_ctx.valid_optimizations.append(best_optimization)

            current_tree_candidates = candidate_node.path_to_root()
            is_candidate_refined_before = any(
                c.source == OptimizedCandidateSource.REFINE for c in current_tree_candidates
            )

            aiservice_client = self.aiservice_client if exp_type == "EXP0" else self.local_aiservice_client

            if is_candidate_refined_before:
                future_adaptive_optimization = self.call_adaptive_optimize(
                    trace_id=self.get_trace_id(exp_type),
                    original_source_code=code_context.read_writable_code.markdown,
                    prev_candidates=current_tree_candidates,
                    eval_ctx=eval_ctx,
                    ai_service_client=aiservice_client,
                )
                if future_adaptive_optimization:
                    self.future_adaptive_optimizations.append(future_adaptive_optimization)
            else:
                # Refinement for all languages (Python, JavaScript, TypeScript)
                future_refinement = self.executor.submit(
                    aiservice_client.optimize_code_refinement,
                    request=[
                        AIServiceRefinerRequest(
                            optimization_id=best_optimization.candidate.optimization_id,
                            original_source_code=code_context.read_writable_code.markdown,
                            read_only_dependency_code=code_context.read_only_context_code,
                            original_code_runtime=original_code_baseline.runtime,
                            optimized_source_code=best_optimization.candidate.source_code.markdown,
                            optimized_explanation=best_optimization.candidate.explanation,
                            optimized_code_runtime=best_optimization.runtime,
                            speedup=f"{int(performance_gain(original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=best_optimization.runtime) * 100)}%",
                            trace_id=self.get_trace_id(exp_type),
                            original_line_profiler_results=original_code_baseline.line_profile_results["str_out"],
                            optimized_line_profiler_results=best_optimization.line_profiler_test_results["str_out"],
                            function_references=function_references,
                            language=self.function_to_optimize.language,
                        )
                    ],
                )
                self.future_all_refinements.append(future_refinement)

        # Display runtime information
        if is_LSP_enabled():
            lsp_log(LspMarkdownMessage(markdown=tree_to_markdown(tree)))
        else:
            console.print(tree)
        if self.args.benchmark and benchmark_tree:
            console.print(benchmark_tree)
        console.rule()

        return best_optimization

    def determine_best_candidate(
        self,
        *,
        candidates: list[OptimizedCandidate],
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        exp_type: str,
        function_references: str,
    ) -> BestOptimization | None:
        """Determine the best optimization candidate from a list of candidates."""
        logger.info(
            f"Determining best optimization candidate (out of {len(candidates)}) for "
            f"{self.function_to_optimize.qualified_name}…"
        )
        console.rule()

        # Initialize evaluation context and async tasks
        eval_ctx = CandidateEvaluationContext()

        self.future_all_refinements.clear()
        self.future_all_code_repair.clear()
        self.future_adaptive_optimizations.clear()

        self.repair_counter = 0
        self.adaptive_optimization_counter = 0

        ai_service_client = self.aiservice_client if exp_type == "EXP0" else self.local_aiservice_client
        assert ai_service_client is not None, "AI service client must be set for optimization"

        future_line_profile_results = self.executor.submit(
            ai_service_client.optimize_python_code_line_profiler,
            source_code=code_context.read_writable_code.markdown,
            dependency_code=code_context.read_only_context_code,
            trace_id=self.get_trace_id(exp_type),
            line_profiler_results=original_code_baseline.line_profile_results["str_out"],
            n_candidates=get_effort_value(EffortKeys.N_OPTIMIZER_LP_CANDIDATES, self.effort),
            experiment_metadata=ExperimentMetadata(
                id=self.experiment_id, group="control" if exp_type == "EXP0" else "experiment"
            )
            if self.experiment_id
            else None,
            is_numerical_code=self.is_numerical_code and not self.args.no_jit_opts,
            language=self.function_to_optimize.language,
        )

        processor = CandidateProcessor(
            candidates,
            future_line_profile_results,
            eval_ctx,
            self.effort,
            code_context.read_writable_code.markdown,
            self.future_all_refinements,
            self.future_all_code_repair,
            self.future_adaptive_optimizations,
        )
        candidate_index = 0

        # Process candidates using queue-based approach
        while not processor.is_done():
            candidate_node = processor.get_next_candidate()
            if candidate_node is None:
                logger.debug("everything done, exiting")
                break

            try:
                candidate_index += 1
                self.process_single_candidate(
                    candidate_node=candidate_node,
                    candidate_index=candidate_index,
                    total_candidates=processor.candidate_len,
                    code_context=code_context,
                    original_code_baseline=original_code_baseline,
                    original_helper_code=original_helper_code,
                    file_path_to_helper_classes=file_path_to_helper_classes,
                    eval_ctx=eval_ctx,
                    exp_type=exp_type,
                    function_references=function_references,
                )
            except KeyboardInterrupt as e:
                logger.exception(f"Optimization interrupted: {e}")
                raise
            finally:
                self.write_code_and_helpers(
                    self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                )

        # Select and return the best optimization
        best_optimization = self.select_best_optimization(
            eval_ctx=eval_ctx,
            code_context=code_context,
            original_code_baseline=original_code_baseline,
            ai_service_client=ai_service_client,
            exp_type=exp_type,
            function_references=function_references,
        )

        if best_optimization:
            self.log_evaluation_results(
                eval_ctx=eval_ctx,
                best_optimization=best_optimization,
                original_code_baseline=original_code_baseline,
                ai_service_client=ai_service_client,
                exp_type=exp_type,
            )

        return best_optimization

    def call_adaptive_optimize(
        self,
        trace_id: str,
        original_source_code: str,
        prev_candidates: list[OptimizedCandidate],
        eval_ctx: CandidateEvaluationContext,
        ai_service_client: AiServiceClient,
    ) -> concurrent.futures.Future[OptimizedCandidate | None] | None:
        if self.adaptive_optimization_counter >= get_effort_value(
            EffortKeys.MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE, self.effort
        ):
            logger.debug(
                f"Max adaptive optimizations reached for {self.function_to_optimize.qualified_name}: {self.adaptive_optimization_counter}"
            )
            return None

        adaptive_count = sum(1 for c in prev_candidates if c.source == OptimizedCandidateSource.ADAPTIVE)

        if adaptive_count >= get_effort_value(EffortKeys.ADAPTIVE_OPTIMIZATION_THRESHOLD, self.effort):
            return None

        request_candidates = []

        for c in prev_candidates:
            speedup = eval_ctx.get_speedup_ratio(c.optimization_id)
            request_candidates.append(
                AdaptiveOptimizedCandidate(
                    optimization_id=c.optimization_id,
                    source_code=c.source_code.markdown,
                    explanation=c.explanation,
                    source=c.source,
                    speedup=f"Performance gain: {int(speedup * 100 + 0.5)}%"
                    if speedup
                    else "Candidate didn't match the behavior of the original code",
                )
            )

        request = AIServiceAdaptiveOptimizeRequest(
            trace_id=trace_id, original_source_code=original_source_code, candidates=request_candidates
        )
        self.adaptive_optimization_counter += 1
        return self.executor.submit(ai_service_client.adaptive_optimize, request=request)

    def repair_optimization(
        self,
        original_source_code: str,
        modified_source_code: str,
        test_diffs: list[TestDiff],
        trace_id: str,
        optimization_id: str,
        ai_service_client: AiServiceClient,
        executor: concurrent.futures.ThreadPoolExecutor,
        language: str = "python",
    ) -> concurrent.futures.Future[OptimizedCandidate | None]:
        request = AIServiceCodeRepairRequest(
            optimization_id=optimization_id,
            original_source_code=original_source_code,
            modified_source_code=modified_source_code,
            test_diffs=test_diffs,
            trace_id=trace_id,
            language=language,
        )
        return executor.submit(ai_service_client.code_repair, request=request)

    def log_successful_optimization(
        self, explanation: Explanation, generated_tests: GeneratedTestsList, exp_type: str
    ) -> None:
        if is_LSP_enabled():
            md_lines = [
                "### ⚡️ Optimization Summary",
                f"Function: `{self.function_to_optimize.qualified_name}`",
                f"File: {explanation.file_path}",
                f"Performance: {explanation.perf_improvement_line}",
                "",
                "#### Explanation\n",
                explanation.__str__(),
            ]

            optimization_summary_markdown = "\n".join(md_lines)
            tests_messages: list[LspCodeMessage] = []

            if generated_tests.generated_tests:
                tests_messages.extend(
                    [
                        LspCodeMessage(code=test.generated_original_test_source, file_name=f"test_{i + 1}.py")
                        for i, test in enumerate(generated_tests.generated_tests)
                    ]
                )

            logger.info("h3|Validated Tests")
            test_report = explanation.winning_behavior_test_results.get_test_pass_fail_report_by_type()
            test_report_md = report_to_markdown_table(test_report, "")

            # displaying tests summary
            lsp_log(LspMarkdownMessage(markdown=test_report_md))
            for test in tests_messages:
                lsp_log(test)

            # displaying optimization summary
            lsp_log(LspMarkdownMessage(markdown=optimization_summary_markdown))

        else:
            # normal console output
            explanation_panel = Panel(
                f"⚡️ Optimization successful! 📄 {self.function_to_optimize.qualified_name} in {explanation.file_path}\n"
                f"📈 {explanation.perf_improvement_line}\n"
                f"Explanation: \n{explanation.__str__()}",
                title="Optimization Summary",
                border_style="green",
            )

            if self.args.no_pr:
                tests_panel = Panel(
                    Syntax(
                        "\n".join([test.generated_original_test_source for test in generated_tests.generated_tests]),
                        "python",
                        line_numbers=True,
                    ),
                    title="Validated Tests",
                    border_style="blue",
                )

                console.print(Group(explanation_panel, tests_panel))
            else:
                console.print(explanation_panel)

        ph(
            "cli-optimize-success",
            {
                "function_trace_id": self.function_trace_id[:-4] + exp_type
                if self.experiment_id
                else self.function_trace_id,
                "speedup_x": explanation.speedup_x,
                "speedup_pct": explanation.speedup_pct,
                "best_runtime": explanation.best_runtime_ns,
                "original_runtime": explanation.original_runtime_ns,
                "winning_test_results": {
                    tt.to_name(): v
                    for tt, v in explanation.winning_behavior_test_results.get_test_pass_fail_report_by_type().items()
                },
            },
        )

    @staticmethod
    def write_code_and_helpers(original_code: str, original_helper_code: dict[Path, str], path: Path) -> None:
        with path.open("w", encoding="utf8") as f:
            f.write(original_code)
        for module_abspath, helper_code in original_helper_code.items():
            with Path(module_abspath).open("w", encoding="utf8") as f:
                f.write(helper_code)

    def reformat_code_and_helpers(
        self,
        helper_functions: list[FunctionSource],
        path: Path,
        original_code: str,
        optimized_context: CodeStringsMarkdown,
    ) -> tuple[str, dict[Path, str]]:
        should_sort_imports = not self.args.disable_imports_sorting
        if should_sort_imports and sort_imports(code=original_code) != original_code:
            should_sort_imports = False

        optimized_code = ""
        if optimized_context is not None:
            file_to_code_context = optimized_context.file_to_path()
            optimized_code = file_to_code_context.get(str(path.relative_to(self.project_root)), "")

        new_code = format_code(
            self.args.formatter_cmds, path, optimized_code=optimized_code, check_diff=True, exit_on_failure=False
        )
        if should_sort_imports:
            new_code = sort_imports(new_code)

        new_helper_code: dict[Path, str] = {}
        for hp in helper_functions:
            module_abspath = hp.file_path
            hp_source_code = hp.source_code
            formatted_helper_code = format_code(
                self.args.formatter_cmds,
                module_abspath,
                optimized_code=hp_source_code,
                check_diff=True,
                exit_on_failure=False,
            )
            if should_sort_imports:
                formatted_helper_code = sort_imports(formatted_helper_code)
            new_helper_code[module_abspath] = formatted_helper_code

        return new_code, new_helper_code

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        optimized_code: CodeStringsMarkdown,
        original_helper_code: dict[Path, str],
    ) -> bool:
        did_update = False
        read_writable_functions_by_file_path = defaultdict(set)
        read_writable_functions_by_file_path[self.function_to_optimize.file_path].add(
            self.function_to_optimize.qualified_name
        )
        for helper_function in code_context.helper_functions:
            # Skip class definitions (jedi_definition may be None for non-Python languages)
            if helper_function.jedi_definition is None or helper_function.jedi_definition.type != "class":
                read_writable_functions_by_file_path[helper_function.file_path].add(helper_function.qualified_name)
        for module_abspath, qualified_names in read_writable_functions_by_file_path.items():
            did_update |= replace_function_definitions_in_module(
                function_names=list(qualified_names),
                optimized_code=optimized_code,
                module_abspath=module_abspath,
                preexisting_objects=code_context.preexisting_objects,
                project_root_path=self.project_root,
            )
        unused_helpers = detect_unused_helper_functions(self.function_to_optimize, code_context, optimized_code)

        # Revert unused helper functions to their original definitions
        if unused_helpers:
            revert_unused_helper_functions(self.project_root, unused_helpers, original_helper_code)

        return did_update

    def get_code_optimization_context(self) -> Result[CodeOptimizationContext, str]:
        try:
            new_code_ctx = code_context_extractor.get_code_optimization_context(
                self.function_to_optimize, self.project_root
            )
        except ValueError as e:
            return Failure(str(e))

        return Success(
            CodeOptimizationContext(
                testgen_context=new_code_ctx.testgen_context,
                read_writable_code=new_code_ctx.read_writable_code,
                read_only_context_code=new_code_ctx.read_only_context_code,
                hashing_code_context=new_code_ctx.hashing_code_context,
                hashing_code_context_hash=new_code_ctx.hashing_code_context_hash,
                helper_functions=new_code_ctx.helper_functions,  # only functions that are read writable
                preexisting_objects=new_code_ctx.preexisting_objects,
            )
        )

    @staticmethod
    def cleanup_leftover_test_return_values() -> None:
        # remove leftovers from previous run
        get_run_tmp_file(Path("test_return_values_0.bin")).unlink(missing_ok=True)
        get_run_tmp_file(Path("test_return_values_0.sqlite")).unlink(missing_ok=True)

    def instrument_existing_tests(self, function_to_all_tests: dict[str, set[FunctionCalledInTest]]) -> set[Path]:
        existing_test_files_count = 0
        replay_test_files_count = 0
        concolic_coverage_test_files_count = 0
        unique_instrumented_test_files = set()

        func_qualname = self.function_to_optimize.qualified_name_with_modules_from_root(self.project_root)
        if func_qualname not in function_to_all_tests:
            logger.info(f"Did not find any pre-existing tests for '{func_qualname}', will only use generated tests.")
        # Handle non-Python existing test instrumentation
        elif not is_python():
            test_file_invocation_positions = defaultdict(list)
            for tests_in_file in function_to_all_tests.get(func_qualname):
                test_file_invocation_positions[
                    (tests_in_file.tests_in_file.test_file, tests_in_file.tests_in_file.test_type)
                ].append(tests_in_file)

            for (test_file, test_type), tests_in_file_list in test_file_invocation_positions.items():
                path_obj_test_file = Path(test_file)
                if test_type == TestType.EXISTING_UNIT_TEST:
                    existing_test_files_count += 1
                elif test_type == TestType.REPLAY_TEST:
                    replay_test_files_count += 1
                elif test_type == TestType.CONCOLIC_COVERAGE_TEST:
                    concolic_coverage_test_files_count += 1
                else:
                    msg = f"Unexpected test type: {test_type}"
                    raise ValueError(msg)

                # Use language-specific instrumentation
                success, injected_behavior_test = self.language_support.instrument_existing_test(
                    test_path=path_obj_test_file,
                    call_positions=[test.position for test in tests_in_file_list],
                    function_to_optimize=self.function_to_optimize,
                    tests_project_root=self.test_cfg.tests_project_rootdir,
                    mode="behavior",
                )
                if not success:
                    logger.debug(f"Failed to instrument test file {test_file} for behavior testing")
                    continue

                success, injected_perf_test = self.language_support.instrument_existing_test(
                    test_path=path_obj_test_file,
                    call_positions=[test.position for test in tests_in_file_list],
                    function_to_optimize=self.function_to_optimize,
                    tests_project_root=self.test_cfg.tests_project_rootdir,
                    mode="performance",
                )
                if not success:
                    logger.debug(f"Failed to instrument test file {test_file} for performance testing")
                    continue

                # Generate instrumented test file paths
                # For JS/TS, preserve .test.ts or .spec.ts suffix for Jest pattern matching
                def get_instrumented_path(original_path: str, suffix: str) -> Path:
                    """Generate instrumented test file path preserving .test/.spec pattern."""
                    path_obj = Path(original_path)
                    stem = path_obj.stem  # e.g., "fibonacci.test"
                    ext = path_obj.suffix  # e.g., ".ts"

                    # Check for .test or .spec in stem (JS/TS pattern)
                    if ".test" in stem:
                        # fibonacci.test -> fibonacci__suffix.test
                        base, _ = stem.rsplit(".test", 1)
                        new_stem = f"{base}{suffix}.test"
                    elif ".spec" in stem:
                        base, _ = stem.rsplit(".spec", 1)
                        new_stem = f"{base}{suffix}.spec"
                    else:
                        # Default Python-style: add suffix before extension
                        new_stem = f"{stem}{suffix}"

                    return path_obj.parent / f"{new_stem}{ext}"

                new_behavioral_test_path = get_instrumented_path(test_file, "__perfinstrumented")
                new_perf_test_path = get_instrumented_path(test_file, "__perfonlyinstrumented")

                if injected_behavior_test is not None:
                    with new_behavioral_test_path.open("w", encoding="utf8") as _f:
                        _f.write(injected_behavior_test)
                    logger.debug(f"[PIPELINE] Wrote instrumented behavior test to {new_behavioral_test_path}")
                else:
                    msg = "injected_behavior_test is None"
                    raise ValueError(msg)

                if injected_perf_test is not None:
                    with new_perf_test_path.open("w", encoding="utf8") as _f:
                        _f.write(injected_perf_test)
                    logger.debug(f"[PIPELINE] Wrote instrumented perf test to {new_perf_test_path}")

                unique_instrumented_test_files.add(new_behavioral_test_path)
                unique_instrumented_test_files.add(new_perf_test_path)

                if not self.test_files.get_by_original_file_path(path_obj_test_file):
                    self.test_files.add(
                        TestFile(
                            instrumented_behavior_file_path=new_behavioral_test_path,
                            benchmarking_file_path=new_perf_test_path,
                            original_source=None,
                            original_file_path=Path(test_file),
                            test_type=test_type,
                            tests_in_file=[t.tests_in_file for t in tests_in_file_list],
                        )
                    )

            if existing_test_files_count > 0 or replay_test_files_count > 0 or concolic_coverage_test_files_count > 0:
                logger.info(
                    f"Instrumented {existing_test_files_count} existing unit test file"
                    f"{'s' if existing_test_files_count != 1 else ''}, {replay_test_files_count} replay test file"
                    f"{'s' if replay_test_files_count != 1 else ''}, and "
                    f"{concolic_coverage_test_files_count} concolic coverage test file"
                    f"{'s' if concolic_coverage_test_files_count != 1 else ''} for {func_qualname}"
                )
                console.rule()
        else:
            test_file_invocation_positions = defaultdict(list)
            for tests_in_file in function_to_all_tests.get(func_qualname):
                test_file_invocation_positions[
                    (tests_in_file.tests_in_file.test_file, tests_in_file.tests_in_file.test_type)
                ].append(tests_in_file)
            for (test_file, test_type), tests_in_file_list in test_file_invocation_positions.items():
                path_obj_test_file = Path(test_file)
                if test_type == TestType.EXISTING_UNIT_TEST:
                    existing_test_files_count += 1
                elif test_type == TestType.REPLAY_TEST:
                    replay_test_files_count += 1
                elif test_type == TestType.CONCOLIC_COVERAGE_TEST:
                    concolic_coverage_test_files_count += 1
                else:
                    msg = f"Unexpected test type: {test_type}"
                    raise ValueError(msg)
                success, injected_behavior_test = inject_profiling_into_existing_test(
                    mode=TestingMode.BEHAVIOR,
                    test_path=path_obj_test_file,
                    call_positions=[test.position for test in tests_in_file_list],
                    function_to_optimize=self.function_to_optimize,
                    tests_project_root=self.test_cfg.tests_project_rootdir,
                )
                if not success:
                    continue
                success, injected_perf_test = inject_profiling_into_existing_test(
                    mode=TestingMode.PERFORMANCE,
                    test_path=path_obj_test_file,
                    call_positions=[test.position for test in tests_in_file_list],
                    function_to_optimize=self.function_to_optimize,
                    tests_project_root=self.test_cfg.tests_project_rootdir,
                )
                if not success:
                    continue
                # TODO: this naming logic should be moved to a function and made more standard
                new_behavioral_test_path = Path(
                    f"{os.path.splitext(test_file)[0]}__perfinstrumented{os.path.splitext(test_file)[1]}"  # noqa: PTH122
                )
                new_perf_test_path = Path(
                    f"{os.path.splitext(test_file)[0]}__perfonlyinstrumented{os.path.splitext(test_file)[1]}"  # noqa: PTH122
                )
                if injected_behavior_test is not None:
                    with new_behavioral_test_path.open("w", encoding="utf8") as _f:
                        _f.write(injected_behavior_test)
                else:
                    msg = "injected_behavior_test is None"
                    raise ValueError(msg)
                if injected_perf_test is not None:
                    with new_perf_test_path.open("w", encoding="utf8") as _f:
                        _f.write(injected_perf_test)

                unique_instrumented_test_files.add(new_behavioral_test_path)
                unique_instrumented_test_files.add(new_perf_test_path)

                if not self.test_files.get_by_original_file_path(path_obj_test_file):
                    self.test_files.add(
                        TestFile(
                            instrumented_behavior_file_path=new_behavioral_test_path,
                            benchmarking_file_path=new_perf_test_path,
                            original_source=None,
                            original_file_path=Path(test_file),
                            test_type=test_type,
                            tests_in_file=[t.tests_in_file for t in tests_in_file_list],
                        )
                    )

            logger.info(
                f"Discovered {existing_test_files_count} existing unit test file"
                f"{'s' if existing_test_files_count != 1 else ''}, {replay_test_files_count} replay test file"
                f"{'s' if replay_test_files_count != 1 else ''}, and "
                f"{concolic_coverage_test_files_count} concolic coverage test file"
                f"{'s' if concolic_coverage_test_files_count != 1 else ''} for {func_qualname}"
            )
            console.rule()
        return unique_instrumented_test_files

    def generate_tests(
        self,
        testgen_context: CodeStringsMarkdown,
        helper_functions: list[FunctionSource],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
    ) -> Result[tuple[int, GeneratedTestsList, dict[str, set[FunctionCalledInTest]], str], str]:
        """Generate unit tests and concolic tests for the function."""
        n_tests = get_effort_value(EffortKeys.N_GENERATED_TESTS, self.effort)
        assert len(generated_test_paths) == n_tests

        if not self.args.no_gen_tests:
            # Submit test generation tasks
            future_tests = self.submit_test_generation_tasks(
                self.executor,
                testgen_context.markdown,
                [definition.fully_qualified_name for definition in helper_functions],
                generated_test_paths,
                generated_perf_test_paths,
            )

        future_concolic_tests = self.executor.submit(
            generate_concolic_tests, self.test_cfg, self.args, self.function_to_optimize, self.function_to_optimize_ast
        )

        if not self.args.no_gen_tests:
            # Wait for test futures to complete
            concurrent.futures.wait([*future_tests, future_concolic_tests])
        else:
            concurrent.futures.wait([future_concolic_tests])
        # Process test generation results
        tests: list[GeneratedTests] = []
        if not self.args.no_gen_tests:
            for future in future_tests:
                res = future.result()
                if res:
                    (
                        generated_test_source,
                        instrumented_behavior_test_source,
                        instrumented_perf_test_source,
                        test_behavior_path,
                        test_perf_path,
                    ) = res
                    tests.append(
                        GeneratedTests(
                            generated_original_test_source=generated_test_source,
                            instrumented_behavior_test_source=instrumented_behavior_test_source,
                            instrumented_perf_test_source=instrumented_perf_test_source,
                            behavior_file_path=test_behavior_path,
                            perf_file_path=test_perf_path,
                        )
                    )

            if not tests:
                logger.warning(f"Failed to generate and instrument tests for {self.function_to_optimize.function_name}")
                return Failure(f"/!\\ NO TESTS GENERATED for {self.function_to_optimize.function_name}")

        function_to_concolic_tests, concolic_test_str = future_concolic_tests.result()
        count_tests = len(tests)
        if concolic_test_str:
            count_tests += 1

        logger.info(f"!lsp|Generated {count_tests} tests for '{self.function_to_optimize.function_name}'")
        console.rule()

        generated_tests = GeneratedTestsList(generated_tests=tests)
        return Success((count_tests, generated_tests, function_to_concolic_tests, concolic_test_str))

    def generate_optimizations(
        self,
        read_writable_code: CodeStringsMarkdown,
        read_only_context_code: str,
        run_experiment: bool = False,
        is_numerical_code: bool | None = None,
    ) -> Result[tuple[OptimizationSet, str], str]:
        """Generate optimization candidates for the function. Backend handles multi-model diversity."""
        n_candidates = get_effort_value(EffortKeys.N_OPTIMIZER_CANDIDATES, self.effort)
        future_optimization_candidates = self.executor.submit(
            self.aiservice_client.optimize_code,
            read_writable_code.markdown,
            read_only_context_code,
            self.function_trace_id[:-4] + "EXP0" if run_experiment else self.function_trace_id,
            ExperimentMetadata(id=self.experiment_id, group="control") if run_experiment else None,
            language=self.function_to_optimize.language,
            is_async=self.function_to_optimize.is_async,
            n_candidates=n_candidates,
            is_numerical_code=is_numerical_code,
        )

        future_references = self.executor.submit(
            get_opt_review_metrics,
            self.function_to_optimize_source_code,
            self.function_to_optimize.file_path,
            self.function_to_optimize.qualified_name,
            self.project_root,
            self.test_cfg.tests_root,
            Language(self.function_to_optimize.language),
        )

        futures = [future_optimization_candidates, future_references]
        future_candidates_exp = None

        if run_experiment:
            future_candidates_exp = self.executor.submit(
                self.local_aiservice_client.optimize_code,
                read_writable_code.markdown,
                read_only_context_code,
                self.function_trace_id[:-4] + "EXP1",
                ExperimentMetadata(id=self.experiment_id, group="experiment"),
                language=self.function_to_optimize.language,
                is_async=self.function_to_optimize.is_async,
                n_candidates=n_candidates,
            )
            futures.append(future_candidates_exp)

        # Wait for optimization futures to complete
        concurrent.futures.wait(futures)

        # Retrieve results - optimize_python_code returns list of candidates
        candidates = future_optimization_candidates.result()

        if not candidates:
            return Failure(f"/!\\ NO OPTIMIZATIONS GENERATED for {self.function_to_optimize.function_name}")

        # Handle experiment results
        candidates_experiment = None
        if future_candidates_exp:
            candidates_experiment = future_candidates_exp.result()
        function_references = future_references.result()

        return Success((OptimizationSet(control=candidates, experiment=candidates_experiment), function_references))

    def setup_and_establish_baseline(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        function_to_concolic_tests: dict[str, set[FunctionCalledInTest]],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
        instrumented_unittests_created_for_function: set[Path],
        original_conftest_content: str | None,
    ) -> Result[
        tuple[str, dict[str, set[FunctionCalledInTest]], OriginalCodeBaseline, list[str], dict[Path, set[str]]], str
    ]:
        """Set up baseline context and establish original code baseline."""
        function_to_optimize_qualified_name = self.function_to_optimize.qualified_name
        function_to_all_tests = {
            key: self.function_to_tests.get(key, set()) | function_to_concolic_tests.get(key, set())
            for key in set(self.function_to_tests) | set(function_to_concolic_tests)
        }

        # Get a dict of file_path_to_classes of fto and helpers_of_fto
        file_path_to_helper_classes = defaultdict(set)
        for function_source in code_context.helper_functions:
            if (
                function_source.qualified_name != self.function_to_optimize.qualified_name
                and "." in function_source.qualified_name
            ):
                file_path_to_helper_classes[function_source.file_path].add(function_source.qualified_name.split(".")[0])

        baseline_result = self.establish_original_code_baseline(
            code_context=code_context,
            original_helper_code=original_helper_code,
            file_path_to_helper_classes=file_path_to_helper_classes,
        )

        console.rule()
        paths_to_cleanup = (
            generated_test_paths + generated_perf_test_paths + list(instrumented_unittests_created_for_function)
        )

        if not is_successful(baseline_result):
            if self.args.override_fixtures:
                restore_conftest(original_conftest_content)
            cleanup_paths(paths_to_cleanup)
            return Failure(baseline_result.failure())

        original_code_baseline, test_functions_to_remove = baseline_result.unwrap()
        if isinstance(original_code_baseline, OriginalCodeBaseline) and (
            not coverage_critic(original_code_baseline.coverage_results)
            or not quantity_of_tests_critic(original_code_baseline)
        ):
            if self.args.override_fixtures:
                restore_conftest(original_conftest_content)
            cleanup_paths(paths_to_cleanup)
            return Failure("The threshold for test confidence was not met.")

        return Success(
            (
                function_to_optimize_qualified_name,
                function_to_all_tests,
                original_code_baseline,
                test_functions_to_remove,
                file_path_to_helper_classes,
            )
        )

    def find_and_process_best_optimization(
        self,
        optimizations_set: OptimizationSet,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        function_to_optimize_qualified_name: str,
        function_to_all_tests: dict[str, set[FunctionCalledInTest]],
        generated_tests: GeneratedTestsList,
        test_functions_to_remove: list[str],
        concolic_test_str: str | None,
        function_references: str,
    ) -> BestOptimization | None:
        """Find the best optimization candidate and process it with all required steps."""
        best_optimization = None
        for _u, (candidates, exp_type) in enumerate(
            zip([optimizations_set.control, optimizations_set.experiment], ["EXP0", "EXP1"])
        ):
            if candidates is None:
                continue

            best_optimization = self.determine_best_candidate(
                candidates=candidates,
                code_context=code_context,
                original_code_baseline=original_code_baseline,
                original_helper_code=original_helper_code,
                file_path_to_helper_classes=file_path_to_helper_classes,
                exp_type=exp_type,
                function_references=function_references,
            )
            ph(
                "cli-optimize-function-finished",
                {
                    "function_trace_id": self.function_trace_id[:-4] + exp_type
                    if self.experiment_id
                    else self.function_trace_id
                },
            )

            if best_optimization:
                logger.info("h2|Best candidate 🚀")
                # Use correct extension based on language
                best_ext = self.language_support.file_extensions[0]
                code_print(
                    best_optimization.candidate.source_code.flat,
                    file_name=f"best_candidate{best_ext}",
                    function_name=self.function_to_optimize.function_name,
                    lsp_message_id=LSPMessageId.BEST_CANDIDATE.value,
                    language=self.function_to_optimize.language,
                )
                processed_benchmark_info = None
                if self.args.benchmark:
                    processed_benchmark_info = process_benchmark_data(
                        replay_performance_gain=best_optimization.replay_performance_gain,
                        fto_benchmark_timings=self.function_benchmark_timings,
                        total_benchmark_timings=self.total_benchmark_timings,
                    )
                acceptance_reason = get_acceptance_reason(
                    original_runtime_ns=original_code_baseline.runtime,
                    optimized_runtime_ns=best_optimization.runtime,
                    original_async_throughput=original_code_baseline.async_throughput,
                    optimized_async_throughput=best_optimization.async_throughput,
                    original_concurrency_metrics=original_code_baseline.concurrency_metrics,
                    optimized_concurrency_metrics=best_optimization.concurrency_metrics,
                )
                explanation = Explanation(
                    raw_explanation_message=best_optimization.candidate.explanation,
                    winning_behavior_test_results=best_optimization.winning_behavior_test_results,
                    winning_benchmarking_test_results=best_optimization.winning_benchmarking_test_results,
                    original_runtime_ns=original_code_baseline.runtime,
                    best_runtime_ns=best_optimization.runtime,
                    function_name=function_to_optimize_qualified_name,
                    file_path=self.function_to_optimize.file_path,
                    benchmark_details=processed_benchmark_info.benchmark_details if processed_benchmark_info else None,
                    original_async_throughput=original_code_baseline.async_throughput,
                    best_async_throughput=best_optimization.async_throughput,
                    original_concurrency_metrics=original_code_baseline.concurrency_metrics,
                    best_concurrency_metrics=best_optimization.concurrency_metrics,
                    acceptance_reason=acceptance_reason,
                )

                self.replace_function_and_helpers_with_optimized_code(
                    code_context=code_context,
                    optimized_code=best_optimization.candidate.source_code,
                    original_helper_code=original_helper_code,
                )

                new_code, new_helper_code = self.reformat_code_and_helpers(
                    code_context.helper_functions,
                    explanation.file_path,
                    self.function_to_optimize_source_code,
                    optimized_context=best_optimization.candidate.source_code,
                )

                original_code_combined = original_helper_code.copy()
                original_code_combined[explanation.file_path] = self.function_to_optimize_source_code
                new_code_combined = new_helper_code.copy()
                new_code_combined[explanation.file_path] = new_code
                self.process_review(
                    original_code_baseline,
                    best_optimization,
                    generated_tests,
                    test_functions_to_remove,
                    concolic_test_str,
                    original_code_combined,
                    new_code_combined,
                    explanation,
                    function_to_all_tests,
                    exp_type,
                    original_helper_code,
                    code_context,
                    function_references,
                )
        return best_optimization

    def process_review(
        self,
        original_code_baseline: OriginalCodeBaseline,
        best_optimization: BestOptimization,
        generated_tests: GeneratedTestsList,
        test_functions_to_remove: list[str],
        concolic_test_str: str | None,
        original_code_combined: dict[Path, str],
        new_code_combined: dict[Path, str],
        explanation: Explanation,
        function_to_all_tests: dict[str, set[FunctionCalledInTest]],
        exp_type: str,
        original_helper_code: dict[Path, str],
        code_context: CodeOptimizationContext,
        function_references: str,
    ) -> None:
        coverage_message = (
            original_code_baseline.coverage_results.build_message()
            if original_code_baseline.coverage_results
            else "Coverage data not available"
        )

        generated_tests = remove_functions_from_generated_tests(
            generated_tests=generated_tests, test_functions_to_remove=test_functions_to_remove
        )
        map_gen_test_file_to_no_of_tests = original_code_baseline.behavior_test_results.file_to_no_of_tests(
            test_functions_to_remove
        )

        original_runtime_by_test = original_code_baseline.benchmarking_test_results.usable_runtime_data_by_test_case()
        optimized_runtime_by_test = (
            best_optimization.winning_benchmarking_test_results.usable_runtime_data_by_test_case()
        )

        generated_tests = add_runtime_comments_to_generated_tests(
            generated_tests, original_runtime_by_test, optimized_runtime_by_test, self.test_cfg.tests_project_rootdir
        )

        generated_tests_str = ""
        code_lang = self.function_to_optimize.language
        for test in generated_tests.generated_tests:
            if map_gen_test_file_to_no_of_tests[test.behavior_file_path] > 0:
                formatted_generated_test = format_generated_code(
                    test.generated_original_test_source, self.args.formatter_cmds
                )
                generated_tests_str += f"```{code_lang}\n{formatted_generated_test}\n```"
                generated_tests_str += "\n\n"

        if concolic_test_str:
            formatted_generated_test = format_generated_code(concolic_test_str, self.args.formatter_cmds)
            generated_tests_str += f"```{code_lang}\n{formatted_generated_test}\n```\n\n"

        existing_tests, replay_tests, concolic_tests = existing_tests_source_for(
            self.function_to_optimize.qualified_name_with_modules_from_root(self.project_root),
            function_to_all_tests,
            test_cfg=self.test_cfg,
            original_runtimes_all=original_runtime_by_test,
            optimized_runtimes_all=optimized_runtime_by_test,
            test_files_registry=self.test_files,
        )
        original_throughput_str = None
        optimized_throughput_str = None
        throughput_improvement_str = None
        original_concurrency_ratio_str = None
        optimized_concurrency_ratio_str = None
        concurrency_improvement_str = None

        if (
            self.function_to_optimize.is_async
            and is_python()
            and original_code_baseline.async_throughput is not None
            and best_optimization.async_throughput is not None
        ):
            original_throughput_str = f"{original_code_baseline.async_throughput} operations/second"
            optimized_throughput_str = f"{best_optimization.async_throughput} operations/second"
            throughput_improvement_value = throughput_gain(
                original_throughput=original_code_baseline.async_throughput,
                optimized_throughput=best_optimization.async_throughput,
            )
            throughput_improvement_str = f"{throughput_improvement_value * 100:.1f}%"

        if original_code_baseline.concurrency_metrics is not None and best_optimization.concurrency_metrics is not None:
            original_concurrency_ratio_str = f"{original_code_baseline.concurrency_metrics.concurrency_ratio:.2f}x"
            optimized_concurrency_ratio_str = f"{best_optimization.concurrency_metrics.concurrency_ratio:.2f}x"
            conc_improvement_value = concurrency_gain(
                original_code_baseline.concurrency_metrics, best_optimization.concurrency_metrics
            )
            concurrency_improvement_str = f"{conc_improvement_value * 100:.1f}%"

        new_explanation_raw_str = self.aiservice_client.get_new_explanation(
            source_code=code_context.read_writable_code.flat,
            dependency_code=code_context.read_only_context_code,
            trace_id=self.function_trace_id[:-4] + exp_type if self.experiment_id else self.function_trace_id,
            optimized_code=best_optimization.candidate.source_code.flat,
            original_line_profiler_results=original_code_baseline.line_profile_results["str_out"],
            optimized_line_profiler_results=best_optimization.line_profiler_test_results["str_out"],
            original_code_runtime=humanize_runtime(original_code_baseline.runtime),
            optimized_code_runtime=humanize_runtime(best_optimization.runtime),
            speedup=f"{int(performance_gain(original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=best_optimization.runtime) * 100)}%",
            annotated_tests=generated_tests_str,
            optimization_id=best_optimization.candidate.optimization_id,
            original_explanation=best_optimization.candidate.explanation,
            original_throughput=original_throughput_str,
            optimized_throughput=optimized_throughput_str,
            throughput_improvement=throughput_improvement_str,
            function_references=function_references,
            acceptance_reason=explanation.acceptance_reason.value,
            original_concurrency_ratio=original_concurrency_ratio_str,
            optimized_concurrency_ratio=optimized_concurrency_ratio_str,
            concurrency_improvement=concurrency_improvement_str,
        )
        new_explanation = Explanation(
            raw_explanation_message=new_explanation_raw_str or explanation.raw_explanation_message,
            winning_behavior_test_results=explanation.winning_behavior_test_results,
            winning_benchmarking_test_results=explanation.winning_benchmarking_test_results,
            original_runtime_ns=explanation.original_runtime_ns,
            best_runtime_ns=explanation.best_runtime_ns,
            function_name=explanation.function_name,
            file_path=explanation.file_path,
            benchmark_details=explanation.benchmark_details,
            original_async_throughput=explanation.original_async_throughput,
            best_async_throughput=explanation.best_async_throughput,
            original_concurrency_metrics=explanation.original_concurrency_metrics,
            best_concurrency_metrics=explanation.best_concurrency_metrics,
            acceptance_reason=explanation.acceptance_reason,
        )
        self.log_successful_optimization(new_explanation, generated_tests, exp_type)

        best_optimization.explanation_v2 = new_explanation.explanation_message()

        data = {
            "original_code": original_code_combined,
            "new_code": new_code_combined,
            "explanation": new_explanation,
            "existing_tests_source": existing_tests,
            "generated_original_test_source": generated_tests_str,
            "function_trace_id": self.function_trace_id[:-4] + exp_type
            if self.experiment_id
            else self.function_trace_id,
            "coverage_message": coverage_message,
            "replay_tests": replay_tests,
            "concolic_tests": concolic_tests,
            "language": self.function_to_optimize.language,
            "original_line_profiler": original_code_baseline.line_profile_results.get("str_out", ""),
            "optimized_line_profiler": best_optimization.line_profiler_test_results.get("str_out", ""),
        }

        raise_pr = not self.args.no_pr
        staging_review = self.args.staging_review
        opt_review_result = OptimizationReviewResult(review="", explanation="")
        # this will now run regardless of pr, staging review flags
        try:
            opt_review_result = self.aiservice_client.get_optimization_review(
                **data, calling_fn_details=function_references
            )
        except Exception as e:
            logger.debug(f"optimization review response failed, investigate {e}")
        data["optimization_review"] = opt_review_result.review
        self.optimization_review = opt_review_result.review

        # Display the reviewer result to the user
        if opt_review_result.review:
            review_display = {
                "high": ("[bold green]High[/bold green]", "green", "Recommended to merge"),
                "medium": ("[bold yellow]Medium[/bold yellow]", "yellow", "Review recommended before merging"),
                "low": ("[bold red]Low[/bold red]", "red", "Not recommended to merge"),
            }
            display_info = review_display.get(opt_review_result.review.lower(), ("[bold]Unknown[/bold]", "white", ""))
            explanation_text = opt_review_result.explanation.strip() if opt_review_result.explanation else ""
            if is_LSP_enabled():
                md_content = f"### Reviewer Assessment: {opt_review_result.review.capitalize()}\n{display_info[2]}"
                if explanation_text:
                    md_content += f"\n\n{explanation_text}"
                lsp_log(LspMarkdownMessage(markdown=md_content))
            else:
                panel_content = f"Reviewer Assessment: {display_info[0]}\n{display_info[2]}"
                if explanation_text:
                    panel_content += f"\n\n[dim]{explanation_text}[/dim]"
                console.print(Panel(panel_content, title="Optimization Review", border_style=display_info[1]))

        if raise_pr or staging_review:
            data["root_dir"] = git_root_dir()
        if raise_pr and not staging_review and opt_review_result.review != "low":
            # Ensure root_dir is set for PR creation (needed for async functions that skip opt_review)
            if "root_dir" not in data:
                data["root_dir"] = git_root_dir()
            data["git_remote"] = self.args.git_remote
            # Remove language from data dict as check_create_pr doesn't accept it
            pr_data = {k: v for k, v in data.items() if k != "language"}
            check_create_pr(**pr_data)
        elif staging_review:
            response = create_staging(**data)
            if response.status_code == 200:
                trace_id = self.function_trace_id[:-4] + exp_type if self.experiment_id else self.function_trace_id
                staging_url = f"{get_cfapi_base_urls().cfwebapp_base_url}/review-optimizations/{trace_id}"
                console.print(
                    Panel(
                        f"[bold green]✅ Staging created:[/bold green]\n[link={staging_url}]{staging_url}[/link]",
                        title="Staging Link",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[bold red]❌ Failed to create staging[/bold red]\nStatus: {response.status_code}",
                        title="Staging Error",
                        border_style="red",
                    )
                )

        else:
            # Mark optimization success since no PR will be created
            mark_optimization_success(
                trace_id=self.function_trace_id, is_optimization_found=best_optimization is not None
            )

        # If worktree mode, do not revert code and helpers, otherwise we would have an empty diff when writing the patch in the lsp
        if self.args.worktree:
            return

        if raise_pr and (
            self.args.all
            or env_utils.get_pr_number()
            or self.args.replay_test
            or (self.args.file and not self.args.function)
        ):
            self.revert_code_and_helpers(original_helper_code)
            return

        if staging_review:
            # always revert code and helpers when staging review
            self.revert_code_and_helpers(original_helper_code)
            return

    def revert_code_and_helpers(self, original_helper_code: dict[Path, str]) -> None:
        logger.info("Reverting code and helpers...")
        self.write_code_and_helpers(
            self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
        )

    def establish_original_code_baseline(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
    ) -> Result[tuple[OriginalCodeBaseline, list[str]], str]:
        line_profile_results = {"timings": {}, "unit": 0, "str_out": ""}
        # For the original function - run the tests and get the runtime, plus coverage
        success = True

        test_env = self.get_test_env(codeflash_loop_index=0, codeflash_test_iteration=0, codeflash_tracer_disable=1)

        if self.function_to_optimize.is_async and is_python():
            from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

            success = add_async_decorator_to_function(
                self.function_to_optimize.file_path, self.function_to_optimize, TestingMode.BEHAVIOR
            )

        # Instrument codeflash capture
        with progress_bar("Running tests to establish original code behavior..."):
            try:
                # Only instrument Python code here - non-Python languages use their own runtime helpers
                # which are already included in the generated/instrumented tests
                if is_python():
                    instrument_codeflash_capture(
                        self.function_to_optimize, file_path_to_helper_classes, self.test_cfg.tests_root
                    )

                total_looping_time = TOTAL_LOOPING_TIME_EFFECTIVE
                logger.debug(f"[PIPELINE] Establishing baseline with {len(self.test_files)} test files")
                for idx, tf in enumerate(self.test_files):
                    logger.debug(
                        f"[PIPELINE] Test file {idx}: behavior={tf.instrumented_behavior_file_path}, perf={tf.benchmarking_file_path}"
                    )
                behavioral_results, coverage_results = self.run_and_parse_tests(
                    testing_type=TestingMode.BEHAVIOR,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=0,
                    testing_time=total_looping_time,
                    enable_coverage=True,
                    code_context=code_context,
                )
            finally:
                # Remove codeflash capture
                self.write_code_and_helpers(
                    self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                )
        if not behavioral_results:
            logger.warning(
                f"force_lsp|Couldn't run any tests for original function {self.function_to_optimize.function_name}. Skipping optimization."
            )
            console.rule()
            return Failure("Failed to establish a baseline for the original code - bevhavioral tests failed.")
        # Skip coverage check for non-Python languages (coverage not yet supported)
        if is_python() and not coverage_critic(coverage_results):
            did_pass_all_tests = all(result.did_pass for result in behavioral_results)
            if not did_pass_all_tests:
                return Failure("Tests failed to pass for the original code.")
            coverage_pct = coverage_results.coverage if coverage_results else 0
            return Failure(
                f"Test coverage is {coverage_pct}%, which is below the required threshold of {COVERAGE_THRESHOLD}%."
            )

        with progress_bar("Running line profiler to identify performance bottlenecks..."):
            line_profile_results = self.line_profiler_step(
                code_context=code_context, original_helper_code=original_helper_code, candidate_index=0
            )
        console.rule()
        with progress_bar("Running performance benchmarks..."):
            if self.function_to_optimize.is_async and is_python():
                from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

                add_async_decorator_to_function(
                    self.function_to_optimize.file_path, self.function_to_optimize, TestingMode.PERFORMANCE
                )

            try:
                benchmarking_results, _ = self.run_and_parse_tests(
                    testing_type=TestingMode.PERFORMANCE,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=0,
                    testing_time=total_looping_time,
                    enable_coverage=False,
                    code_context=code_context,
                )
            finally:
                if self.function_to_optimize.is_async:
                    self.write_code_and_helpers(
                        self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                    )

        console.print(
            TestResults.report_to_tree(
                behavioral_results.get_test_pass_fail_report_by_type(), title="Overall test results for original code"
            )
        )
        console.rule()

        total_timing = benchmarking_results.total_passed_runtime()  # caution: doesn't handle the loop index
        functions_to_remove = [
            result.id.test_function_name
            for result in behavioral_results
            if (result.test_type == TestType.GENERATED_REGRESSION and not result.did_pass)
        ]

        if total_timing == 0:
            logger.warning("The overall summed benchmark runtime of the original function is 0, couldn't run tests.")
            console.rule()
            success = False
        if not total_timing:
            logger.warning("Failed to run the tests for the original function, skipping optimization")
            console.rule()
            success = False
        if not success:
            return Failure("Failed to establish a baseline for the original code.")

        loop_count = max([int(result.loop_index) for result in benchmarking_results.test_results])
        logger.info(
            f"h3|⌚ Original code summed runtime measured over '{loop_count}' loop{'s' if loop_count > 1 else ''}: "
            f"'{humanize_runtime(total_timing)}' per full loop"
        )
        console.rule()
        logger.debug(f"Total original code runtime (ns): {total_timing}")

        async_throughput = None
        concurrency_metrics = None
        if self.function_to_optimize.is_async and is_python():
            async_throughput = calculate_function_throughput_from_test_results(
                benchmarking_results, self.function_to_optimize.function_name
            )
            logger.debug(f"Original async function throughput: {async_throughput} calls/second")

            concurrency_metrics = self.run_concurrency_benchmark(
                code_context=code_context, original_helper_code=original_helper_code, test_env=test_env
            )
            if concurrency_metrics:
                logger.debug(
                    f"Original concurrency metrics: ratio={concurrency_metrics.concurrency_ratio:.2f}, "
                    f"seq={concurrency_metrics.sequential_time_ns}ns, conc={concurrency_metrics.concurrent_time_ns}ns"
                )

        if self.args.benchmark:
            replay_benchmarking_test_results = benchmarking_results.group_by_benchmarks(
                self.total_benchmark_timings.keys(), self.replay_tests_dir, self.project_root
            )
        return Success(
            (
                OriginalCodeBaseline(
                    behavior_test_results=behavioral_results,
                    benchmarking_test_results=benchmarking_results,
                    replay_benchmarking_test_results=replay_benchmarking_test_results if self.args.benchmark else None,
                    runtime=total_timing,
                    coverage_results=coverage_results,
                    line_profile_results=line_profile_results,
                    async_throughput=async_throughput,
                    concurrency_metrics=concurrency_metrics,
                ),
                functions_to_remove,
            )
        )

    def get_results_not_matched_error(self) -> Failure:
        logger.info("h4|Test results did not match the test results of the original code ❌")
        console.rule()
        return Failure("Test results did not match the test results of the original code.")

    def repair_if_possible(
        self,
        candidate: OptimizedCandidate,
        diffs: list[TestDiff],
        eval_ctx: CandidateEvaluationContext,
        code_context: CodeOptimizationContext,
        test_results_count: int,
        exp_type: str,
    ) -> None:
        max_repairs = get_effort_value(EffortKeys.MAX_CODE_REPAIRS_PER_TRACE, self.effort)
        if self.repair_counter >= max_repairs:
            logger.debug(f"Repair counter reached {max_repairs}, skipping repair")
            return

        successful_candidates_count = sum(1 for is_correct in eval_ctx.is_correct.values() if is_correct)
        if successful_candidates_count >= MIN_CORRECT_CANDIDATES:
            logger.debug(f"{successful_candidates_count} of the candidates were correct, no need to repair")
            return

        if candidate.source not in (OptimizedCandidateSource.OPTIMIZE, OptimizedCandidateSource.OPTIMIZE_LP):
            # only repair the first pass of the candidates for now
            logger.debug(f"Candidate is a result of {candidate.source.value}, skipping repair")
            return
        if not diffs:
            logger.debug("No diffs found, skipping repair")
            return
        result_unmatched_perc = len(diffs) / test_results_count
        if result_unmatched_perc > get_effort_value(EffortKeys.REPAIR_UNMATCHED_PERCENTAGE_LIMIT, self.effort):
            logger.debug(f"Result unmatched percentage is {result_unmatched_perc * 100}%, skipping repair")
            return

        logger.debug(
            f"Adding a candidate for repair, with {len(diffs)} diffs, ({result_unmatched_perc * 100}% unmatched)"
        )
        # start repairing
        ai_service_client = self.aiservice_client if exp_type == "EXP0" else self.local_aiservice_client
        self.repair_counter += 1
        self.future_all_code_repair.append(
            self.repair_optimization(
                original_source_code=code_context.read_writable_code.markdown,
                modified_source_code=candidate.source_code.markdown,
                test_diffs=diffs,
                trace_id=self.function_trace_id[:-4] + exp_type if self.experiment_id else self.function_trace_id,
                ai_service_client=ai_service_client,
                optimization_id=candidate.optimization_id,
                executor=self.executor,
                language=self.function_to_optimize.language,
            )
        )

    def run_optimized_candidate(
        self,
        *,
        optimization_candidate_index: int,
        baseline_results: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        eval_ctx: CandidateEvaluationContext,
        code_context: CodeOptimizationContext,
        candidate: OptimizedCandidate,
        exp_type: str,
    ) -> Result[OptimizedCandidateResult, str]:
        with progress_bar("Testing optimization candidate"):
            test_env = self.get_test_env(
                codeflash_loop_index=0,
                codeflash_test_iteration=optimization_candidate_index,
                codeflash_tracer_disable=1,
            )

            get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite")).unlink(missing_ok=True)
            # Instrument codeflash capture
            candidate_fto_code = Path(self.function_to_optimize.file_path).read_text("utf-8")
            candidate_helper_code = {}
            for module_abspath in original_helper_code:
                candidate_helper_code[module_abspath] = Path(module_abspath).read_text("utf-8")
            if self.function_to_optimize.is_async and is_python():
                from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

                add_async_decorator_to_function(
                    self.function_to_optimize.file_path, self.function_to_optimize, TestingMode.BEHAVIOR
                )

            try:
                # Only instrument Python code here - non-Python languages use their own runtime helpers
                if is_python():
                    instrument_codeflash_capture(
                        self.function_to_optimize, file_path_to_helper_classes, self.test_cfg.tests_root
                    )

                total_looping_time = TOTAL_LOOPING_TIME_EFFECTIVE
                candidate_behavior_results, _ = self.run_and_parse_tests(
                    testing_type=TestingMode.BEHAVIOR,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=optimization_candidate_index,
                    testing_time=total_looping_time,
                    enable_coverage=False,
                )
            # Remove instrumentation
            finally:
                # Only restore code for Python - non-Python tests are self-contained
                if is_python():
                    self.write_code_and_helpers(
                        candidate_fto_code, candidate_helper_code, self.function_to_optimize.file_path
                    )
            console.print(
                TestResults.report_to_tree(
                    candidate_behavior_results.get_test_pass_fail_report_by_type(),
                    title=f"Behavioral Test Results for candidate {optimization_candidate_index}",
                )
            )
            console.rule()

            # Use language-appropriate comparison
            if not is_python():
                # Non-Python: Compare using language support with SQLite results if available
                original_sqlite = get_run_tmp_file(Path("test_return_values_0.sqlite"))
                candidate_sqlite = get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite"))

                if original_sqlite.exists() and candidate_sqlite.exists():
                    # Full comparison using captured return values via language support
                    # Use js_project_root where node_modules is located
                    js_root = self.test_cfg.js_project_root or self.args.project_root
                    match, diffs = self.language_support.compare_test_results(
                        original_sqlite, candidate_sqlite, project_root=js_root
                    )
                    # Cleanup SQLite files after comparison
                    candidate_sqlite.unlink(missing_ok=True)
                else:
                    # Fallback: compare test pass/fail status (tests aren't instrumented yet)
                    # If all tests that passed for original also pass for candidate, consider it a match
                    match, diffs = compare_test_results(
                        baseline_results.behavior_test_results, candidate_behavior_results, pass_fail_only=True
                    )
            else:
                # Python: Compare using Python comparator
                match, diffs = compare_test_results(baseline_results.behavior_test_results, candidate_behavior_results)

            if match:
                logger.info("h3|Test results matched ✅")
                console.rule()
            else:
                self.repair_if_possible(
                    candidate, diffs, eval_ctx, code_context, len(candidate_behavior_results), exp_type
                )
                return self.get_results_not_matched_error()

            logger.info(f"loading|Running performance tests for candidate {optimization_candidate_index}...")
            console.rule()

            # For async functions, instrument at definition site for performance benchmarking
            if self.function_to_optimize.is_async and is_python():
                from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

                add_async_decorator_to_function(
                    self.function_to_optimize.file_path, self.function_to_optimize, TestingMode.PERFORMANCE
                )

            try:
                candidate_benchmarking_results, _ = self.run_and_parse_tests(
                    testing_type=TestingMode.PERFORMANCE,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=optimization_candidate_index,
                    testing_time=total_looping_time,
                    enable_coverage=False,
                )
            finally:
                # Restore original source if we instrumented it
                if self.function_to_optimize.is_async and is_python():
                    self.write_code_and_helpers(
                        candidate_fto_code, candidate_helper_code, self.function_to_optimize.file_path
                    )
            loop_count = (
                max(all_loop_indices)
                if (all_loop_indices := {result.loop_index for result in candidate_benchmarking_results.test_results})
                else 0
            )

            if (total_candidate_timing := candidate_benchmarking_results.total_passed_runtime()) == 0:
                logger.warning("The overall test runtime of the optimized function is 0, couldn't run tests.")
                console.rule()

            logger.debug(f"Total optimized code {optimization_candidate_index} runtime (ns): {total_candidate_timing}")

            candidate_async_throughput = None
            candidate_concurrency_metrics = None
            if self.function_to_optimize.is_async and is_python():
                candidate_async_throughput = calculate_function_throughput_from_test_results(
                    candidate_benchmarking_results, self.function_to_optimize.function_name
                )
                logger.debug(f"Candidate async function throughput: {candidate_async_throughput} calls/second")

                # Run concurrency benchmark for candidate
                candidate_concurrency_metrics = self.run_concurrency_benchmark(
                    code_context=code_context, original_helper_code=candidate_helper_code, test_env=test_env
                )
                if candidate_concurrency_metrics:
                    logger.debug(
                        f"Candidate concurrency metrics: ratio={candidate_concurrency_metrics.concurrency_ratio:.2f}, "
                        f"seq={candidate_concurrency_metrics.sequential_time_ns}ns, conc={candidate_concurrency_metrics.concurrent_time_ns}ns"
                    )

            if self.args.benchmark:
                candidate_replay_benchmarking_results = candidate_benchmarking_results.group_by_benchmarks(
                    self.total_benchmark_timings.keys(), self.replay_tests_dir, self.project_root
                )
                for benchmark_name, benchmark_results in candidate_replay_benchmarking_results.items():
                    logger.debug(
                        f"Benchmark {benchmark_name} runtime (ns): {humanize_runtime(benchmark_results.total_passed_runtime())}"
                    )
            return Success(
                OptimizedCandidateResult(
                    max_loop_count=loop_count,
                    best_test_runtime=total_candidate_timing,
                    behavior_test_results=candidate_behavior_results,
                    benchmarking_test_results=candidate_benchmarking_results,
                    replay_benchmarking_test_results=candidate_replay_benchmarking_results
                    if self.args.benchmark
                    else None,
                    optimization_candidate_index=optimization_candidate_index,
                    total_candidate_timing=total_candidate_timing,
                    async_throughput=candidate_async_throughput,
                    concurrency_metrics=candidate_concurrency_metrics,
                )
            )

    def run_and_parse_tests(
        self,
        testing_type: TestingMode,
        test_env: dict[str, str],
        test_files: TestFiles,
        optimization_iteration: int,
        testing_time: float = TOTAL_LOOPING_TIME_EFFECTIVE,
        *,
        enable_coverage: bool = False,
        pytest_min_loops: int = 5,
        pytest_max_loops: int = 250,
        code_context: CodeOptimizationContext | None = None,
        line_profiler_output_file: Path | None = None,
    ) -> tuple[TestResults | dict, CoverageData | None]:
        coverage_database_file = None
        coverage_config_file = None
        try:
            if testing_type == TestingMode.BEHAVIOR:
                result_file_path, run_result, coverage_database_file, coverage_config_file = run_behavioral_tests(
                    test_files,
                    test_framework=self.test_cfg.test_framework,
                    cwd=self.project_root,
                    test_env=test_env,
                    pytest_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    enable_coverage=enable_coverage,
                    js_project_root=self.test_cfg.js_project_root,
                    candidate_index=optimization_iteration,
                )
            elif testing_type == TestingMode.LINE_PROFILE:
                result_file_path, run_result = run_line_profile_tests(
                    test_files,
                    cwd=self.project_root,
                    test_env=test_env,
                    pytest_cmd=self.test_cfg.pytest_cmd,
                    pytest_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    pytest_target_runtime_seconds=testing_time,
                    test_framework=self.test_cfg.test_framework,
                    js_project_root=self.test_cfg.js_project_root,
                    line_profiler_output_file=line_profiler_output_file,
                )
            elif testing_type == TestingMode.PERFORMANCE:
                result_file_path, run_result = run_benchmarking_tests(
                    test_files,
                    cwd=self.project_root,
                    test_env=test_env,
                    pytest_cmd=self.test_cfg.pytest_cmd,
                    pytest_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    pytest_target_runtime_seconds=testing_time,
                    pytest_min_loops=pytest_min_loops,
                    pytest_max_loops=pytest_max_loops,
                    test_framework=self.test_cfg.test_framework,
                    js_project_root=self.test_cfg.js_project_root,
                )
            else:
                msg = f"Unexpected testing type: {testing_type}"
                raise ValueError(msg)
        except subprocess.TimeoutExpired:
            logger.exception(
                f"Error running tests in {', '.join(str(f) for f in test_files.test_files)}.\nTimeout Error"
            )
            return TestResults(), None
        if run_result.returncode != 0 and testing_type == TestingMode.BEHAVIOR:
            logger.debug(
                f"!lsp|Nonzero return code {run_result.returncode} when running tests in "
                f"{', '.join([str(f.instrumented_behavior_file_path) for f in test_files.test_files])}.\n"
                f"stdout: {run_result.stdout}\n"
                f"stderr: {run_result.stderr}\n"
            )

            unique_errors = extract_unique_errors(run_result.stdout)

            if unique_errors:
                from rich.text import Text

                if is_LSP_enabled():
                    combined_errors = "\n\n".join(unique_errors)
                    lsp_log(LspCodeMessage(code=combined_errors, file_name="errors", collapsed=True))
                else:
                    for error in unique_errors:
                        panel = Panel(Text.from_markup(f"⚠️  {error} ", style="bold red"), expand=False)
                        console.print(panel)

        if testing_type in {TestingMode.BEHAVIOR, TestingMode.PERFORMANCE}:
            # For non-Python behavior tests, skip SQLite cleanup - files needed for language-native comparison
            non_python_original_code = not is_python() and optimization_iteration == 0
            skip_cleanup = (not is_python() and testing_type == TestingMode.BEHAVIOR) or non_python_original_code

            results, coverage_results = parse_test_results(
                test_xml_path=result_file_path,
                test_files=test_files,
                test_config=self.test_cfg,
                optimization_iteration=optimization_iteration,
                run_result=run_result,
                function_name=self.function_to_optimize.function_name,
                source_file=self.function_to_optimize.file_path,
                code_context=code_context,
                coverage_database_file=coverage_database_file,
                coverage_config_file=coverage_config_file,
                skip_sqlite_cleanup=skip_cleanup,
            )
            if testing_type == TestingMode.PERFORMANCE:
                results.perf_stdout = run_result.stdout
            return results, coverage_results
        # For LINE_PROFILE mode, Python uses .lprof files while JavaScript uses JSON
        # Return TestResults for JavaScript so _line_profiler_step_javascript can parse the JSON
        if not is_python():
            # Return TestResults to indicate tests ran, actual parsing happens in _line_profiler_step_javascript
            return TestResults(test_results=[]), None
        results, coverage_results = parse_line_profile_results(line_profiler_output_file=line_profiler_output_file)
        return results, coverage_results

    def submit_test_generation_tasks(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        source_code_being_tested: str,
        helper_function_names: list[str],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
    ) -> list[concurrent.futures.Future]:
        return [
            executor.submit(
                generate_tests,
                self.aiservice_client,
                source_code_being_tested,
                self.function_to_optimize,
                helper_function_names,
                Path(self.original_module_path),
                self.test_cfg,
                INDIVIDUAL_TESTCASE_TIMEOUT,
                self.function_trace_id,
                test_index,
                test_path,
                test_perf_path,
                self.is_numerical_code,
            )
            for test_index, (test_path, test_perf_path) in enumerate(
                zip(generated_test_paths, generated_perf_test_paths)
            )
        ]

    def cleanup_generated_files(self) -> None:
        paths_to_cleanup = []
        for test_file in self.test_files:
            paths_to_cleanup.append(test_file.instrumented_behavior_file_path)
            paths_to_cleanup.append(test_file.benchmarking_file_path)

        cleanup_paths(paths_to_cleanup)

    def get_test_env(
        self, codeflash_loop_index: int, codeflash_test_iteration: int, codeflash_tracer_disable: int = 1
    ) -> dict:
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = str(codeflash_test_iteration)
        test_env["CODEFLASH_TRACER_DISABLE"] = str(codeflash_tracer_disable)
        test_env["CODEFLASH_LOOP_INDEX"] = str(codeflash_loop_index)
        if "PYTHONPATH" not in test_env:
            test_env["PYTHONPATH"] = str(self.args.project_root)
        else:
            test_env["PYTHONPATH"] += os.pathsep + str(self.args.project_root)
        return test_env

    def line_profiler_step(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], candidate_index: int
    ) -> dict:
        # Dispatch to language-specific implementation
        if is_python():
            return self._line_profiler_step_python(code_context, original_helper_code, candidate_index)

        if self.language_support is not None and hasattr(self.language_support, "instrument_source_for_line_profiler"):
            try:
                line_profiler_output_path = get_run_tmp_file(Path("line_profiler_output.json"))
                # NOTE: currently this handles single file only, add support to multi file instrumentation (or should it be kept for the main file only)
                original_source = Path(self.function_to_optimize.file_path).read_text()
                # Instrument source code
                func_info = FunctionInfo(
                    name=self.function_to_optimize.function_name,
                    file_path=self.function_to_optimize.file_path,
                    start_line=self.function_to_optimize.starting_line,
                    end_line=self.function_to_optimize.ending_line,
                    start_col=self.function_to_optimize.starting_col,
                    end_col=self.function_to_optimize.ending_col,
                    is_async=self.function_to_optimize.is_async,
                    language=self.language_support.language,
                )
                success = self.language_support.instrument_source_for_line_profiler(
                    func_info=func_info, line_profiler_output_file=line_profiler_output_path
                )
                if not success:
                    return {"timings": {}, "unit": 0, "str_out": ""}

                test_env = self.get_test_env(
                    codeflash_loop_index=0, codeflash_test_iteration=candidate_index, codeflash_tracer_disable=1
                )

                _test_results, _ = self.run_and_parse_tests(
                    testing_type=TestingMode.LINE_PROFILE,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=0,
                    testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                    enable_coverage=False,
                    code_context=code_context,
                    line_profiler_output_file=line_profiler_output_path,
                )

                if not hasattr(self.language_support, "parse_line_profile_results"):
                    raise ValueError("Language support does not implement parse_line_profile_results")  # noqa: TRY301

                return self.language_support.parse_line_profile_results(line_profiler_output_path)
            except Exception as e:
                logger.warning(f"Failed to run line profiling: {e}")
                return {"timings": {}, "unit": 0, "str_out": ""}
            finally:
                # restore original source
                Path(self.function_to_optimize.file_path).write_text(original_source)

        logger.warning(f"Language support for {self.language_support.language} doesn't support line profiling")
        return {"timings": {}, "unit": 0, "str_out": ""}

    def _line_profiler_step_python(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], candidate_index: int
    ) -> dict:
        """Python-specific line profiler using decorator imports."""
        # Check if candidate code contains JIT decorators - line profiler doesn't work with JIT compiled code
        candidate_fto_code = Path(self.function_to_optimize.file_path).read_text("utf-8")
        if contains_jit_decorator(candidate_fto_code):
            logger.info(
                f"Skipping line profiler for {self.function_to_optimize.function_name} - code contains JIT decorator"
            )
            return {"timings": {}, "unit": 0, "str_out": ""}

        # Check helper code for JIT decorators
        for module_abspath in original_helper_code:
            candidate_helper_code = Path(module_abspath).read_text("utf-8")
            if contains_jit_decorator(candidate_helper_code):
                logger.info(
                    f"Skipping line profiler for {self.function_to_optimize.function_name} - helper code contains JIT decorator"
                )
                return {"timings": {}, "unit": 0, "str_out": ""}

        try:
            console.rule()

            test_env = self.get_test_env(
                codeflash_loop_index=0, codeflash_test_iteration=candidate_index, codeflash_tracer_disable=1
            )
            line_profiler_output_file = add_decorator_imports(self.function_to_optimize, code_context)
            line_profile_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.LINE_PROFILE,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=0,
                testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                enable_coverage=False,
                code_context=code_context,
                line_profiler_output_file=line_profiler_output_file,
            )
        finally:
            # Remove codeflash capture
            self.write_code_and_helpers(
                self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
            )
        # this will happen when a timeoutexpired exception happens
        if isinstance(line_profile_results, TestResults) and not line_profile_results.test_results:
            logger.warning(
                f"Timeout occurred while running line profiler for original function {self.function_to_optimize.function_name}"
            )
            # set default value for line profiler results
            return {"timings": {}, "unit": 0, "str_out": ""}
        if line_profile_results["str_out"] == "":
            logger.warning(
                f"Couldn't run line profiler for original function {self.function_to_optimize.function_name}"
            )
        return line_profile_results

    def run_concurrency_benchmark(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], test_env: dict[str, str]
    ) -> ConcurrencyMetrics | None:
        """Run concurrency benchmark to measure sequential vs concurrent execution for async functions.

        This benchmark detects blocking vs non-blocking async code by comparing:
        - Sequential execution time (running N iterations one after another)
        - Concurrent execution time (running N iterations in parallel with asyncio.gather)

        Blocking code (like time.sleep) will have similar sequential and concurrent times.
        Non-blocking code (like asyncio.sleep) will be much faster when run concurrently.

        Returns:
            ConcurrencyMetrics if benchmark ran successfully, None otherwise.

        """
        if not self.function_to_optimize.is_async:
            return None

        from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

        try:
            # Add concurrency decorator to the source function
            add_async_decorator_to_function(
                self.function_to_optimize.file_path, self.function_to_optimize, TestingMode.CONCURRENCY
            )

            # Run the concurrency benchmark tests
            concurrency_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.PERFORMANCE,  # Use performance mode for running
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=0,
                testing_time=5.0,  # Short benchmark time
                enable_coverage=False,
                code_context=code_context,
                pytest_min_loops=1,
                pytest_max_loops=3,
            )
        except Exception as e:
            logger.debug(f"Concurrency benchmark failed: {e}")
            return None
        finally:
            # Restore original code
            self.write_code_and_helpers(
                self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
            )

        # Parse concurrency metrics from stdout
        if concurrency_results and concurrency_results.perf_stdout:
            return parse_concurrency_metrics(concurrency_results, self.function_to_optimize.function_name)

        return None
