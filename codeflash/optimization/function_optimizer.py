from __future__ import annotations

import ast
import concurrent.futures
import os
import random
import subprocess
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

import isort
import libcst as cst
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from codeflash.api.aiservice import AiServiceClient, LocalAiServiceClient
from codeflash.api.cfapi import add_code_context_hash, mark_optimization_success
from codeflash.benchmarking.utils import process_benchmark_data
from codeflash.cli_cmds.console import code_print, console, logger, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_replacer import (
    add_custom_marker_to_all_tests,
    modify_autouse_fixture,
    replace_function_definitions_in_module,
)
from codeflash.code_utils.code_utils import (
    ImportErrorPattern,
    cleanup_paths,
    file_name_from_test_module_name,
    get_run_tmp_file,
    has_any_async_functions,
    module_name_from_file_path,
    restore_conftest,
)
from codeflash.code_utils.config_consts import (
    INDIVIDUAL_TESTCASE_TIMEOUT,
    N_CANDIDATES,
    N_TESTS_TO_GENERATE,
    REPEAT_OPTIMIZATION_PROBABILITY,
    TOTAL_LOOPING_TIME,
)
from codeflash.code_utils.edit_generated_tests import (
    add_runtime_comments_to_generated_tests,
    remove_functions_from_generated_tests,
)
from codeflash.code_utils.env_utils import get_pr_number
from codeflash.code_utils.formatter import format_code, sort_imports
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.code_utils.line_profile_utils import add_decorator_imports
from codeflash.code_utils.static_analysis import get_first_top_level_function_or_method_ast
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.context import code_context_extractor
from codeflash.context.unused_definition_remover import detect_unused_helper_functions, revert_unused_helper_functions
from codeflash.discovery.functions_to_optimize import was_function_previously_optimized
from codeflash.either import Failure, Success, is_successful
from codeflash.models.ExperimentMetadata import ExperimentMetadata
from codeflash.models.models import (
    BestOptimization,
    CodeOptimizationContext,
    GeneratedTests,
    GeneratedTestsList,
    OptimizationSet,
    OptimizedCandidateResult,
    OriginalCodeBaseline,
    TestFile,
    TestFiles,
    TestingMode,
    TestResults,
    TestType,
)
from codeflash.result.create_pr import check_create_pr, existing_tests_source_for
from codeflash.result.critic import coverage_critic, performance_gain, quantity_of_tests_critic, speedup_critic
from codeflash.result.explanation import Explanation
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.concolic_testing import generate_concolic_tests
from codeflash.verification.equivalence import compare_test_results
from codeflash.verification.instrument_codeflash_capture import instrument_codeflash_capture
from codeflash.verification.parse_line_profile_test_output import parse_line_profile_results
from codeflash.verification.parse_test_output import parse_test_results
from codeflash.verification.test_runner import run_behavioral_tests, run_benchmarking_tests, run_line_profile_tests
from codeflash.verification.verification_utils import get_test_file_path
from codeflash.verification.verifier import generate_tests

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.either import Result
    from codeflash.models.models import (
        BenchmarkKey,
        CoverageData,
        FunctionCalledInTest,
        FunctionSource,
        OptimizedCandidate,
    )
    from codeflash.verification.verification_utils import TestConfig


class FunctionOptimizer:
    def __init__(
        self,
        function_to_optimize: FunctionToOptimize,
        test_cfg: TestConfig,
        function_to_optimize_source_code: str = "",
        function_to_tests: dict[str, set[FunctionCalledInTest]] | None = None,
        function_to_optimize_ast: ast.FunctionDef | None = None,
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
        if not function_to_optimize_ast:
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
        self.args = args  # Check defaults for these
        self.function_trace_id: str = str(uuid.uuid4())
        self.original_module_path = module_name_from_file_path(self.function_to_optimize.file_path, self.project_root)

        self.function_benchmark_timings = function_benchmark_timings if function_benchmark_timings else {}
        self.total_benchmark_timings = total_benchmark_timings if total_benchmark_timings else {}
        self.replay_tests_dir = replay_tests_dir if replay_tests_dir else None

    def optimize_function(self) -> Result[BestOptimization, str]:  # noqa: PLR0911
        should_run_experiment = self.experiment_id is not None
        logger.debug(f"Function Trace ID: {self.function_trace_id}")
        ph("cli-optimize-function-start", {"function_trace_id": self.function_trace_id})
        self.cleanup_leftover_test_return_values()
        file_name_from_test_module_name.cache_clear()
        ctx_result = self.get_code_optimization_context()
        if not is_successful(ctx_result):
            return Failure(ctx_result.failure())
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code

        if has_any_async_functions(code_context.read_writable_code):
            return Failure("Codeflash does not support async functions in the code to optimize.")
        # Random here means that we still attempt optimization with a fractional chance to see if
        # last time we could not find an optimization, maybe this time we do.
        # Random is before as a performance optimization, swapping the two 'and' statements has the same effect
        if random.random() > REPEAT_OPTIMIZATION_PROBABILITY and was_function_previously_optimized(  # noqa: S311
            self.function_to_optimize, code_context
        ):
            return Failure("Function optimization previously attempted, skipping.")

        code_print(code_context.read_writable_code)
        generated_test_paths = [
            get_test_file_path(
                self.test_cfg.tests_root, self.function_to_optimize.function_name, test_index, test_type="unit"
            )
            for test_index in range(N_TESTS_TO_GENERATE)
        ]
        generated_perf_test_paths = [
            get_test_file_path(
                self.test_cfg.tests_root, self.function_to_optimize.function_name, test_index, test_type="perf"
            )
            for test_index in range(N_TESTS_TO_GENERATE)
        ]

        with progress_bar(
            f"Generating new tests and optimizations for function {self.function_to_optimize.function_name}",
            transient=True,
            revert_to_print=bool(get_pr_number()),
        ):
            generated_results = self.generate_tests_and_optimizations(
                testgen_context_code=code_context.testgen_context_code,
                read_writable_code=code_context.read_writable_code,
                read_only_context_code=code_context.read_only_context_code,
                helper_functions=code_context.helper_functions,
                generated_test_paths=generated_test_paths,
                generated_perf_test_paths=generated_perf_test_paths,
                run_experiment=should_run_experiment,
            )

        if not is_successful(generated_results):
            return Failure(generated_results.failure())
        generated_tests: GeneratedTestsList
        optimizations_set: OptimizationSet
        generated_tests, function_to_concolic_tests, concolic_test_str, optimizations_set = generated_results.unwrap()
        count_tests = len(generated_tests.generated_tests)
        if concolic_test_str:
            count_tests += 1

        for i, generated_test in enumerate(generated_tests.generated_tests):
            with generated_test.behavior_file_path.open("w", encoding="utf8") as f:
                f.write(generated_test.instrumented_behavior_test_source)
            with generated_test.perf_file_path.open("w", encoding="utf8") as f:
                f.write(generated_test.instrumented_perf_test_source)
            self.test_files.add(
                TestFile(
                    instrumented_behavior_file_path=generated_test.behavior_file_path,
                    benchmarking_file_path=generated_test.perf_file_path,
                    original_file_path=None,
                    original_source=generated_test.generated_original_test_source,
                    test_type=TestType.GENERATED_REGRESSION,
                    tests_in_file=None,  # This is currently unused. We can discover the tests in the file if needed.
                )
            )
            logger.info(f"Generated test {i + 1}/{count_tests}:")
            code_print(generated_test.generated_original_test_source)
        if concolic_test_str:
            logger.info(f"Generated test {count_tests}/{count_tests}:")
            code_print(concolic_test_str)

        function_to_optimize_qualified_name = self.function_to_optimize.qualified_name
        function_to_all_tests = {
            key: self.function_to_tests.get(key, set()) | function_to_concolic_tests.get(key, set())
            for key in set(self.function_to_tests) | set(function_to_concolic_tests)
        }
        instrumented_unittests_created_for_function = self.instrument_existing_tests(function_to_all_tests)
        if self.args.override_fixtures:
            logger.info("Disabling all autouse fixtures associated with the generated test files")
            original_conftest_content = modify_autouse_fixture(generated_test_paths + generated_perf_test_paths)
            logger.info("Add custom marker to generated test files")
            add_custom_marker_to_all_tests(generated_test_paths + generated_perf_test_paths)

        # Get a dict of file_path_to_classes of fto and helpers_of_fto
        file_path_to_helper_classes = defaultdict(set)
        for function_source in code_context.helper_functions:
            if (
                function_source.qualified_name != self.function_to_optimize.qualified_name
                and "." in function_source.qualified_name
            ):
                file_path_to_helper_classes[function_source.file_path].add(function_source.qualified_name.split(".")[0])

        baseline_result = self.establish_original_code_baseline(  # this needs better typing
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
        if isinstance(original_code_baseline, OriginalCodeBaseline) and not coverage_critic(
            original_code_baseline.coverage_results, self.args.test_framework
        ):
            if self.args.override_fixtures:
                restore_conftest(original_conftest_content)
            cleanup_paths(paths_to_cleanup)
            return Failure("The threshold for test coverage was not met.")
        # request for new optimizations but don't block execution, check for completion later
        # adding to control and experiment set but with same traceid
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
                logger.info("Best candidate:")
                code_print(best_optimization.candidate.source_code)
                console.print(
                    Panel(
                        best_optimization.candidate.explanation, title="Best Candidate Explanation", border_style="blue"
                    )
                )
                processed_benchmark_info = None
                if self.args.benchmark:
                    processed_benchmark_info = process_benchmark_data(
                        replay_performance_gain=best_optimization.replay_performance_gain,
                        fto_benchmark_timings=self.function_benchmark_timings,
                        total_benchmark_timings=self.total_benchmark_timings,
                    )
                explanation = Explanation(
                    raw_explanation_message=best_optimization.candidate.explanation,
                    winning_behavioral_test_results=best_optimization.winning_behavioral_test_results,
                    winning_benchmarking_test_results=best_optimization.winning_benchmarking_test_results,
                    original_runtime_ns=original_code_baseline.runtime,
                    best_runtime_ns=best_optimization.runtime,
                    function_name=function_to_optimize_qualified_name,
                    file_path=self.function_to_optimize.file_path,
                    benchmark_details=processed_benchmark_info.benchmark_details if processed_benchmark_info else None,
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
                    optimized_function=best_optimization.candidate.source_code,
                )

                original_code_combined = original_helper_code.copy()
                original_code_combined[explanation.file_path] = self.function_to_optimize_source_code
                new_code_combined = new_helper_code.copy()
                new_code_combined[explanation.file_path] = new_code
                if not self.args.no_pr:
                    coverage_message = (
                        original_code_baseline.coverage_results.build_message()
                        if original_code_baseline.coverage_results
                        else "Coverage data not available"
                    )
                    generated_tests = remove_functions_from_generated_tests(
                        generated_tests=generated_tests, test_functions_to_remove=test_functions_to_remove
                    )
                    original_runtime_by_test = (
                        original_code_baseline.benchmarking_test_results.usable_runtime_data_by_test_case()
                    )
                    optimized_runtime_by_test = (
                        best_optimization.winning_benchmarking_test_results.usable_runtime_data_by_test_case()
                    )
                    # Add runtime comments to generated tests before creating the PR
                    generated_tests = add_runtime_comments_to_generated_tests(
                        self.test_cfg, generated_tests, original_runtime_by_test, optimized_runtime_by_test
                    )
                    generated_tests_str = "\n\n".join(
                        [test.generated_original_test_source for test in generated_tests.generated_tests]
                    )
                    existing_tests = existing_tests_source_for(
                        self.function_to_optimize.qualified_name_with_modules_from_root(self.project_root),
                        function_to_all_tests,
                        test_cfg=self.test_cfg,
                        original_runtimes_all=original_runtime_by_test,
                        optimized_runtimes_all=optimized_runtime_by_test,
                    )
                    if concolic_test_str:
                        generated_tests_str += "\n\n" + concolic_test_str

                    check_create_pr(
                        original_code=original_code_combined,
                        new_code=new_code_combined,
                        explanation=explanation,
                        existing_tests_source=existing_tests,
                        generated_original_test_source=generated_tests_str,
                        function_trace_id=self.function_trace_id[:-4] + exp_type
                        if self.experiment_id
                        else self.function_trace_id,
                        coverage_message=coverage_message,
                        git_remote=self.args.git_remote,
                    )
                    if self.args.all or env_utils.get_pr_number() or (self.args.file and not self.args.function):
                        self.write_code_and_helpers(
                            self.function_to_optimize_source_code,
                            original_helper_code,
                            self.function_to_optimize.file_path,
                        )
                else:
                    # Mark optimization success since no PR will be created
                    mark_optimization_success(
                        trace_id=self.function_trace_id, is_optimization_found=best_optimization is not None
                    )
                self.log_successful_optimization(explanation, generated_tests, exp_type)

        # Add function to code context hash if in gh actions

        add_code_context_hash(code_context.hashing_code_context_hash)

        if self.args.override_fixtures:
            restore_conftest(original_conftest_content)
        if not best_optimization:
            return Failure(f"No best optimizations found for function {self.function_to_optimize.qualified_name}")
        return Success(best_optimization)

    def determine_best_candidate(
        self,
        *,
        candidates: list[OptimizedCandidate],
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        exp_type: str,
    ) -> BestOptimization | None:
        best_optimization: BestOptimization | None = None
        best_runtime_until_now = original_code_baseline.runtime

        speedup_ratios: dict[str, float | None] = {}
        optimized_runtimes: dict[str, float | None] = {}
        is_correct = {}

        logger.info(
            f"Determining best optimization candidate (out of {len(candidates)}) for "
            f"{self.function_to_optimize.qualified_name}…"
        )
        console.rule()
        candidates = deque(candidates)
        # Start a new thread for AI service request, start loop in main thread
        # check if aiservice request is complete, when it is complete, append result to the candidates list
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ai_service_client = self.aiservice_client if exp_type == "EXP0" else self.local_aiservice_client
            future_line_profile_results = executor.submit(
                ai_service_client.optimize_python_code_line_profiler,
                source_code=code_context.read_writable_code,
                dependency_code=code_context.read_only_context_code,
                trace_id=self.function_trace_id[:-4] + exp_type if self.experiment_id else self.function_trace_id,
                line_profiler_results=original_code_baseline.line_profile_results["str_out"],
                num_candidates=10,
                experiment_metadata=ExperimentMetadata(
                    id=self.experiment_id, group="control" if exp_type == "EXP0" else "experiment"
                )
                if self.experiment_id
                else None,
            )
            try:
                candidate_index = 0
                original_len = len(candidates)
                while candidates:
                    done = True if future_line_profile_results is None else future_line_profile_results.done()
                    if done and (future_line_profile_results is not None):
                        line_profile_results = future_line_profile_results.result()
                        candidates.extend(line_profile_results)
                        original_len += len(line_profile_results)
                        logger.info(
                            f"Added results from line profiler to candidates, total candidates now: {original_len}"
                        )
                        future_line_profile_results = None
                    candidate = candidates.popleft()
                    candidate_index += 1
                    get_run_tmp_file(Path(f"test_return_values_{candidate_index}.bin")).unlink(missing_ok=True)
                    get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite")).unlink(missing_ok=True)
                    logger.info(f"Optimization candidate {candidate_index}/{original_len}:")
                    code_print(candidate.source_code)
                    try:
                        did_update = self.replace_function_and_helpers_with_optimized_code(
                            code_context=code_context,
                            optimized_code=candidate.source_code,
                            original_helper_code=original_helper_code,
                        )
                        if not did_update:
                            logger.warning(
                                "No functions were replaced in the optimized code. Skipping optimization candidate."
                            )
                            console.rule()
                            continue
                    except (ValueError, SyntaxError, cst.ParserSyntaxError, AttributeError) as e:
                        logger.error(e)
                        self.write_code_and_helpers(
                            self.function_to_optimize_source_code,
                            original_helper_code,
                            self.function_to_optimize.file_path,
                        )
                        continue

                    run_results = self.run_optimized_candidate(
                        optimization_candidate_index=candidate_index,
                        baseline_results=original_code_baseline,
                        original_helper_code=original_helper_code,
                        file_path_to_helper_classes=file_path_to_helper_classes,
                    )
                    console.rule()

                    if not is_successful(run_results):
                        optimized_runtimes[candidate.optimization_id] = None
                        is_correct[candidate.optimization_id] = False
                        speedup_ratios[candidate.optimization_id] = None
                    else:
                        candidate_result: OptimizedCandidateResult = run_results.unwrap()
                        best_test_runtime = candidate_result.best_test_runtime
                        optimized_runtimes[candidate.optimization_id] = best_test_runtime
                        is_correct[candidate.optimization_id] = True
                        perf_gain = performance_gain(
                            original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=best_test_runtime
                        )
                        speedup_ratios[candidate.optimization_id] = perf_gain

                        tree = Tree(f"Candidate #{candidate_index} - Runtime Information")
                        benchmark_tree = None
                        if speedup_critic(
                            candidate_result, original_code_baseline.runtime, best_runtime_until_now
                        ) and quantity_of_tests_critic(candidate_result):
                            tree.add("This candidate is faster than the previous best candidate. 🚀")
                            tree.add(f"Original summed runtime: {humanize_runtime(original_code_baseline.runtime)}")
                            tree.add(
                                f"Best summed runtime: {humanize_runtime(candidate_result.best_test_runtime)} "
                                f"(measured over {candidate_result.max_loop_count} "
                                f"loop{'s' if candidate_result.max_loop_count > 1 else ''})"
                            )
                            tree.add(f"Speedup percentage: {perf_gain * 100:.1f}%")
                            tree.add(f"Speedup ratio: {perf_gain + 1:.3f}X")
                            replay_perf_gain = {}
                            if self.args.benchmark:
                                test_results_by_benchmark = (
                                    candidate_result.benchmarking_test_results.group_by_benchmarks(
                                        self.total_benchmark_timings.keys(), self.replay_tests_dir, self.project_root
                                    )
                                )
                                if len(test_results_by_benchmark) > 0:
                                    benchmark_tree = Tree("Speedup percentage on benchmarks:")
                                for benchmark_key, candidate_test_results in test_results_by_benchmark.items():
                                    original_code_replay_runtime = (
                                        original_code_baseline.replay_benchmarking_test_results[
                                            benchmark_key
                                        ].total_passed_runtime()
                                    )
                                    candidate_replay_runtime = candidate_test_results.total_passed_runtime()
                                    replay_perf_gain[benchmark_key] = performance_gain(
                                        original_runtime_ns=original_code_replay_runtime,
                                        optimized_runtime_ns=candidate_replay_runtime,
                                    )
                                    benchmark_tree.add(f"{benchmark_key}: {replay_perf_gain[benchmark_key] * 100:.1f}%")

                            best_optimization = BestOptimization(
                                candidate=candidate,
                                helper_functions=code_context.helper_functions,
                                runtime=best_test_runtime,
                                winning_behavioral_test_results=candidate_result.behavior_test_results,
                                replay_performance_gain=replay_perf_gain if self.args.benchmark else None,
                                winning_benchmarking_test_results=candidate_result.benchmarking_test_results,
                                winning_replay_benchmarking_test_results=candidate_result.benchmarking_test_results,
                            )
                            best_runtime_until_now = best_test_runtime
                        else:
                            tree.add(
                                f"Summed runtime: {humanize_runtime(best_test_runtime)} "
                                f"(measured over {candidate_result.max_loop_count} "
                                f"loop{'s' if candidate_result.max_loop_count > 1 else ''})"
                            )
                            tree.add(f"Speedup percentage: {perf_gain * 100:.1f}%")
                            tree.add(f"Speedup ratio: {perf_gain + 1:.3f}X")
                        console.print(tree)
                        if self.args.benchmark and benchmark_tree:
                            console.print(benchmark_tree)
                        console.rule()

                    self.write_code_and_helpers(
                        self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                    )
                    if (not len(candidates)) and (
                        not done
                    ):  # all original candidates processed but lp results haven't been processed
                        concurrent.futures.wait([future_line_profile_results])
                        line_profile_results = future_line_profile_results.result()
                        candidates.extend(line_profile_results)
                        original_len += len(line_profile_results)
                        logger.info(
                            f"Added results from line profiler to candidates, total candidates now: {original_len}"
                        )
                        future_line_profile_results = None
            except KeyboardInterrupt as e:
                self.write_code_and_helpers(
                    self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                )
                logger.exception(f"Optimization interrupted: {e}")
                raise

        ai_service_client.log_results(
            function_trace_id=self.function_trace_id[:-4] + exp_type if self.experiment_id else self.function_trace_id,
            speedup_ratio=speedup_ratios,
            original_runtime=original_code_baseline.runtime,
            optimized_runtime=optimized_runtimes,
            is_correct=is_correct,
        )
        return best_optimization

    def log_successful_optimization(
        self, explanation: Explanation, generated_tests: GeneratedTestsList, exp_type: str
    ) -> None:
        explanation_panel = Panel(
            f"⚡️ Optimization successful! 📄 {self.function_to_optimize.qualified_name} in {explanation.file_path}\n"
            f"📈 {explanation.perf_improvement_line}\n"
            f"Explanation: \n{explanation.to_console_string()}",
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
                    for tt, v in explanation.winning_behavioral_test_results.get_test_pass_fail_report_by_type().items()
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
        self, helper_functions: list[FunctionSource], path: Path, original_code: str, optimized_function: str
    ) -> tuple[str, dict[Path, str]]:
        should_sort_imports = not self.args.disable_imports_sorting
        if should_sort_imports and isort.code(original_code) != original_code:
            should_sort_imports = False

        new_code = format_code(self.args.formatter_cmds, path, optimized_function=optimized_function, check_diff=True)
        if should_sort_imports:
            new_code = sort_imports(new_code)

        new_helper_code: dict[Path, str] = {}
        for hp in helper_functions:
            module_abspath = hp.file_path
            hp_source_code = hp.source_code
            formatted_helper_code = format_code(
                self.args.formatter_cmds, module_abspath, optimized_function=hp_source_code, check_diff=True
            )
            if should_sort_imports:
                formatted_helper_code = sort_imports(formatted_helper_code)
            new_helper_code[module_abspath] = formatted_helper_code

        return new_code, new_helper_code

    def replace_function_and_helpers_with_optimized_code(
        self, code_context: CodeOptimizationContext, optimized_code: str, original_helper_code: str
    ) -> bool:
        did_update = False
        read_writable_functions_by_file_path = defaultdict(set)
        read_writable_functions_by_file_path[self.function_to_optimize.file_path].add(
            self.function_to_optimize.qualified_name
        )
        for helper_function in code_context.helper_functions:
            if helper_function.jedi_definition.type != "class":
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
                testgen_context_code=new_code_ctx.testgen_context_code,
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
                    test_framework=self.args.test_framework,
                )
                if not success:
                    continue
                success, injected_perf_test = inject_profiling_into_existing_test(
                    mode=TestingMode.PERFORMANCE,
                    test_path=path_obj_test_file,
                    call_positions=[test.position for test in tests_in_file_list],
                    function_to_optimize=self.function_to_optimize,
                    tests_project_root=self.test_cfg.tests_project_rootdir,
                    test_framework=self.args.test_framework,
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
        return unique_instrumented_test_files

    def generate_tests_and_optimizations(
        self,
        testgen_context_code: str,
        read_writable_code: str,
        read_only_context_code: str,
        helper_functions: list[FunctionSource],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
        run_experiment: bool = False,  # noqa: FBT001, FBT002
    ) -> Result[tuple[GeneratedTestsList, dict[str, set[FunctionCalledInTest]], OptimizationSet], str]:
        assert len(generated_test_paths) == N_TESTS_TO_GENERATE
        max_workers = N_TESTS_TO_GENERATE + 2 if not run_experiment else N_TESTS_TO_GENERATE + 3
        console.rule()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit the test generation task as future
            future_tests = self.generate_and_instrument_tests(
                executor,
                testgen_context_code,
                [definition.fully_qualified_name for definition in helper_functions],
                generated_test_paths,
                generated_perf_test_paths,
            )
            future_optimization_candidates = executor.submit(
                self.aiservice_client.optimize_python_code,
                read_writable_code,
                read_only_context_code,
                self.function_trace_id[:-4] + "EXP0" if run_experiment else self.function_trace_id,
                N_CANDIDATES,
                ExperimentMetadata(id=self.experiment_id, group="control") if run_experiment else None,
            )
            future_candidates_exp = None

            future_concolic_tests = executor.submit(
                generate_concolic_tests,
                self.test_cfg,
                self.args,
                self.function_to_optimize,
                self.function_to_optimize_ast,
            )
            futures = [*future_tests, future_optimization_candidates, future_concolic_tests]
            if run_experiment:
                future_candidates_exp = executor.submit(
                    self.local_aiservice_client.optimize_python_code,
                    read_writable_code,
                    read_only_context_code,
                    self.function_trace_id[:-4] + "EXP1",
                    N_CANDIDATES,
                    ExperimentMetadata(id=self.experiment_id, group="experiment"),
                )
                futures.append(future_candidates_exp)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

            # Retrieve results
            candidates: list[OptimizedCandidate] = future_optimization_candidates.result()
            if not candidates:
                return Failure(f"/!\\ NO OPTIMIZATIONS GENERATED for {self.function_to_optimize.function_name}")

            candidates_experiment = future_candidates_exp.result() if future_candidates_exp else None

            # Process test generation results

            tests: list[GeneratedTests] = []
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
            logger.info(f"Generated {len(tests)} tests for {self.function_to_optimize.function_name}")
            console.rule()
            generated_tests = GeneratedTestsList(generated_tests=tests)

        return Success(
            (
                generated_tests,
                function_to_concolic_tests,
                concolic_test_str,
                OptimizationSet(control=candidates, experiment=candidates_experiment),
            )
        )

    def establish_original_code_baseline(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
    ) -> Result[tuple[OriginalCodeBaseline, list[str]], str]:
        line_profile_results = {"timings": {}, "unit": 0, "str_out": ""}
        # For the original function - run the tests and get the runtime, plus coverage
        with progress_bar(f"Establishing original code baseline for {self.function_to_optimize.function_name}"):
            assert (test_framework := self.args.test_framework) in {"pytest", "unittest"}  # noqa: RUF018
            success = True

            test_env = os.environ.copy()
            test_env["CODEFLASH_TEST_ITERATION"] = "0"
            test_env["CODEFLASH_TRACER_DISABLE"] = "1"
            test_env["CODEFLASH_LOOP_INDEX"] = "0"
            if "PYTHONPATH" not in test_env:
                test_env["PYTHONPATH"] = str(self.args.project_root)
            else:
                test_env["PYTHONPATH"] += os.pathsep + str(self.args.project_root)

            coverage_results = None
            # Instrument codeflash capture
            try:
                instrument_codeflash_capture(
                    self.function_to_optimize, file_path_to_helper_classes, self.test_cfg.tests_root
                )
                behavioral_results, coverage_results = self.run_and_parse_tests(
                    testing_type=TestingMode.BEHAVIOR,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=0,
                    testing_time=TOTAL_LOOPING_TIME,
                    enable_coverage=test_framework == "pytest",
                    code_context=code_context,
                )
            finally:
                # Remove codeflash capture
                self.write_code_and_helpers(
                    self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                )
            if not behavioral_results:
                logger.warning(
                    f"Couldn't run any tests for original function {self.function_to_optimize.function_name}. SKIPPING OPTIMIZING THIS FUNCTION."
                )
                console.rule()
                return Failure("Failed to establish a baseline for the original code - bevhavioral tests failed.")
            if not coverage_critic(coverage_results, self.args.test_framework):
                return Failure("The threshold for test coverage was not met.")
            if test_framework == "pytest":
                try:
                    line_profiler_output_file = add_decorator_imports(self.function_to_optimize, code_context)
                    line_profile_results, _ = self.run_and_parse_tests(
                        testing_type=TestingMode.LINE_PROFILE,
                        test_env=test_env,
                        test_files=self.test_files,
                        optimization_iteration=0,
                        testing_time=TOTAL_LOOPING_TIME,
                        enable_coverage=False,
                        code_context=code_context,
                        line_profiler_output_file=line_profiler_output_file,
                    )
                finally:
                    # Remove codeflash capture
                    self.write_code_and_helpers(
                        self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                    )
                if line_profile_results["str_out"] == "":
                    logger.warning(
                        f"Couldn't run line profiler for original function {self.function_to_optimize.function_name}"
                    )
                    console.rule()
                benchmarking_results, _ = self.run_and_parse_tests(
                    testing_type=TestingMode.PERFORMANCE,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=0,
                    testing_time=TOTAL_LOOPING_TIME,
                    enable_coverage=False,
                    code_context=code_context,
                )
            else:
                benchmarking_results = TestResults()
                start_time: float = time.time()
                for i in range(100):
                    if i >= 5 and time.time() - start_time >= TOTAL_LOOPING_TIME * 1.5:
                        # * 1.5 to give unittest a bit more time to run
                        break
                    test_env["CODEFLASH_LOOP_INDEX"] = str(i + 1)
                    unittest_loop_results, _ = self.run_and_parse_tests(
                        testing_type=TestingMode.PERFORMANCE,
                        test_env=test_env,
                        test_files=self.test_files,
                        optimization_iteration=0,
                        testing_time=TOTAL_LOOPING_TIME,
                        enable_coverage=False,
                        code_context=code_context,
                        unittest_loop_index=i + 1,
                    )
                    benchmarking_results.merge(unittest_loop_results)

            console.print(
                TestResults.report_to_tree(
                    behavioral_results.get_test_pass_fail_report_by_type(),
                    title="Overall test results for original code",
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
                logger.warning(
                    "The overall summed benchmark runtime of the original function is 0, couldn't run tests."
                )
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
                f"Original code summed runtime measured over {loop_count} loop{'s' if loop_count > 1 else ''}: "
                f"{humanize_runtime(total_timing)} per full loop"
            )
            console.rule()
            logger.debug(f"Total original code runtime (ns): {total_timing}")

            if self.args.benchmark:
                replay_benchmarking_test_results = benchmarking_results.group_by_benchmarks(
                    self.total_benchmark_timings.keys(), self.replay_tests_dir, self.project_root
                )
            return Success(
                (
                    OriginalCodeBaseline(
                        behavioral_test_results=behavioral_results,
                        benchmarking_test_results=benchmarking_results,
                        replay_benchmarking_test_results=replay_benchmarking_test_results
                        if self.args.benchmark
                        else None,
                        runtime=total_timing,
                        coverage_results=coverage_results,
                        line_profile_results=line_profile_results,
                    ),
                    functions_to_remove,
                )
            )

    def run_optimized_candidate(
        self,
        *,
        optimization_candidate_index: int,
        baseline_results: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
    ) -> Result[OptimizedCandidateResult, str]:
        assert (test_framework := self.args.test_framework) in {"pytest", "unittest"}  # noqa: RUF018

        with progress_bar("Testing optimization candidate"):
            test_env = os.environ.copy()
            test_env["CODEFLASH_LOOP_INDEX"] = "0"
            test_env["CODEFLASH_TEST_ITERATION"] = str(optimization_candidate_index)
            test_env["CODEFLASH_TRACER_DISABLE"] = "1"
            if "PYTHONPATH" not in test_env:
                test_env["PYTHONPATH"] = str(self.project_root)
            else:
                test_env["PYTHONPATH"] += os.pathsep + str(self.project_root)

            get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite")).unlink(missing_ok=True)
            # Instrument codeflash capture
            candidate_fto_code = Path(self.function_to_optimize.file_path).read_text("utf-8")
            candidate_helper_code = {}
            for module_abspath in original_helper_code:
                candidate_helper_code[module_abspath] = Path(module_abspath).read_text("utf-8")
            try:
                instrument_codeflash_capture(
                    self.function_to_optimize, file_path_to_helper_classes, self.test_cfg.tests_root
                )

                candidate_behavior_results, _ = self.run_and_parse_tests(
                    testing_type=TestingMode.BEHAVIOR,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=optimization_candidate_index,
                    testing_time=TOTAL_LOOPING_TIME,
                    enable_coverage=False,
                )
            # Remove instrumentation
            finally:
                self.write_code_and_helpers(
                    candidate_fto_code, candidate_helper_code, self.function_to_optimize.file_path
                )
            console.print(
                TestResults.report_to_tree(
                    candidate_behavior_results.get_test_pass_fail_report_by_type(),
                    title="Behavioral Test Results for candidate",
                )
            )
            console.rule()
            if compare_test_results(baseline_results.behavioral_test_results, candidate_behavior_results):
                logger.info("Test results matched!")
                console.rule()
            else:
                logger.info("Test results did not match the test results of the original code.")
                console.rule()
                return Failure("Test results did not match the test results of the original code.")

            if test_framework == "pytest":
                candidate_benchmarking_results, _ = self.run_and_parse_tests(
                    testing_type=TestingMode.PERFORMANCE,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=optimization_candidate_index,
                    testing_time=TOTAL_LOOPING_TIME,
                    enable_coverage=False,
                )
                loop_count = (
                    max(all_loop_indices)
                    if (
                        all_loop_indices := {
                            result.loop_index for result in candidate_benchmarking_results.test_results
                        }
                    )
                    else 0
                )

            else:
                candidate_benchmarking_results = TestResults()
                start_time: float = time.time()
                loop_count = 0
                for i in range(100):
                    if i >= 5 and time.time() - start_time >= TOTAL_LOOPING_TIME * 1.5:
                        # * 1.5 to give unittest a bit more time to run
                        break
                    test_env["CODEFLASH_LOOP_INDEX"] = str(i + 1)
                    unittest_loop_results, cov = self.run_and_parse_tests(
                        testing_type=TestingMode.PERFORMANCE,
                        test_env=test_env,
                        test_files=self.test_files,
                        optimization_iteration=optimization_candidate_index,
                        testing_time=TOTAL_LOOPING_TIME,
                        unittest_loop_index=i + 1,
                    )
                    loop_count = i + 1
                    candidate_benchmarking_results.merge(unittest_loop_results)

            if (total_candidate_timing := candidate_benchmarking_results.total_passed_runtime()) == 0:
                logger.warning("The overall test runtime of the optimized function is 0, couldn't run tests.")
                console.rule()

            logger.debug(f"Total optimized code {optimization_candidate_index} runtime (ns): {total_candidate_timing}")
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
                )
            )

    def run_and_parse_tests(
        self,
        testing_type: TestingMode,
        test_env: dict[str, str],
        test_files: TestFiles,
        optimization_iteration: int,
        testing_time: float = TOTAL_LOOPING_TIME,
        *,
        enable_coverage: bool = False,
        pytest_min_loops: int = 5,
        pytest_max_loops: int = 100_000,
        code_context: CodeOptimizationContext | None = None,
        unittest_loop_index: int | None = None,
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
                    verbose=True,
                    enable_coverage=enable_coverage,
                )
            elif testing_type == TestingMode.LINE_PROFILE:
                result_file_path, run_result = run_line_profile_tests(
                    test_files,
                    cwd=self.project_root,
                    test_env=test_env,
                    pytest_cmd=self.test_cfg.pytest_cmd,
                    pytest_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    pytest_target_runtime_seconds=testing_time,
                    pytest_min_loops=1,
                    pytest_max_loops=1,
                    test_framework=self.test_cfg.test_framework,
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
                f"Nonzero return code {run_result.returncode} when running tests in "
                f"{', '.join([str(f.instrumented_behavior_file_path) for f in test_files.test_files])}.\n"
                f"stdout: {run_result.stdout}\n"
                f"stderr: {run_result.stderr}\n"
            )
            if "ModuleNotFoundError" in run_result.stdout:
                from rich.text import Text

                match = ImportErrorPattern.search(run_result.stdout).group()
                panel = Panel(Text.from_markup(f"⚠️  {match} ", style="bold red"), expand=False)
                console.print(panel)
        if testing_type in {TestingMode.BEHAVIOR, TestingMode.PERFORMANCE}:
            results, coverage_results = parse_test_results(
                test_xml_path=result_file_path,
                test_files=test_files,
                test_config=self.test_cfg,
                optimization_iteration=optimization_iteration,
                run_result=run_result,
                unittest_loop_index=unittest_loop_index,
                function_name=self.function_to_optimize.function_name,
                source_file=self.function_to_optimize.file_path,
                code_context=code_context,
                coverage_database_file=coverage_database_file,
                coverage_config_file=coverage_config_file,
            )
        else:
            results, coverage_results = parse_line_profile_results(line_profiler_output_file=line_profiler_output_file)
        return results, coverage_results

    def generate_and_instrument_tests(
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
            )
            for test_index, (test_path, test_perf_path) in enumerate(
                zip(generated_test_paths, generated_perf_test_paths)
            )
        ]

    def cleanup_generated_files(self) -> None:
        paths_to_cleanup = (
            [
                test_file.instrumented_behavior_file_path
                for test_type in [
                    TestType.GENERATED_REGRESSION,
                    TestType.EXISTING_UNIT_TEST,
                    TestType.CONCOLIC_COVERAGE_TEST,
                ]
                for test_file in self.test_files.get_by_type(test_type).test_files
            ]
            + [
                test_file.benchmarking_file_path
                for test_type in [TestType.GENERATED_REGRESSION, TestType.EXISTING_UNIT_TEST]
                for test_file in self.test_files.get_by_type(test_type).test_files
            ]
            + [self.test_cfg.concolic_test_root_dir]
        )
        cleanup_paths(paths_to_cleanup)
        if hasattr(get_run_tmp_file, "tmpdir"):
            get_run_tmp_file.tmpdir.cleanup()
