from __future__ import annotations

import ast
import concurrent.futures
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import isort
import libcst as cst
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from codeflash.api.aiservice import AiServiceClient, LocalAiServiceClient
from codeflash.cli_cmds.console import code_print, console, logger, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_extractor import add_needed_imports_from_module, extract_code
from codeflash.code_utils.code_replacer import normalize_code, normalize_node, replace_function_definitions_in_module
from codeflash.code_utils.code_utils import (
    cleanup_paths,
    file_name_from_test_module_name,
    get_run_tmp_file,
    module_name_from_file_path,
)
from codeflash.code_utils.config_consts import (
    INDIVIDUAL_TESTCASE_TIMEOUT,
    N_CANDIDATES,
    N_TESTS_TO_GENERATE,
    TOTAL_LOOPING_TIME,
)
from codeflash.code_utils.formatter import format_code, sort_imports
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.code_utils.remove_generated_tests import remove_functions_from_generated_tests
from codeflash.code_utils.static_analysis import analyze_imported_modules, get_first_top_level_function_or_method_ast
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.context import code_context_extractor
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.discovery.functions_to_optimize import FunctionToOptimize, get_functions_to_optimize
from codeflash.either import Failure, Success, is_successful
from codeflash.models.ExperimentMetadata import ExperimentMetadata
from codeflash.models.models import (
    BestOptimization,
    CodeOptimizationContext,
    FunctionCalledInTest,
    FunctionParent,
    GeneratedTests,
    GeneratedTestsList,
    OptimizationSet,
    OptimizedCandidateResult,
    OriginalCodeBaseline,
    TestFile,
    TestFiles,
    TestingMode,
    ValidCode,
)
from codeflash.optimization.function_context import get_constrained_function_context_and_helper_functions
from codeflash.result.create_pr import check_create_pr, existing_tests_source_for
from codeflash.result.critic import coverage_critic, performance_gain, quantity_of_tests_critic, speedup_critic
from codeflash.result.explanation import Explanation
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.bayesian_analysis import compare_function_runtime_distributions
from codeflash.verification.concolic_testing import generate_concolic_tests
from codeflash.verification.equivalence import compare_test_results
from codeflash.verification.parse_test_output import parse_test_results
from codeflash.verification.test_results import TestResults, TestType
from codeflash.verification.test_runner import run_behavioral_tests, run_benchmarking_tests
from codeflash.verification.verification_utils import TestConfig, get_test_file_path
from codeflash.verification.verifier import generate_tests

if TYPE_CHECKING:
    from argparse import Namespace

    import numpy as np
    import numpy.typing as npt

    from codeflash.either import Result
    from codeflash.models.models import CoverageData, FunctionSource, OptimizedCandidate


class Optimizer:
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.test_cfg = TestConfig(
            tests_root=args.tests_root,
            tests_project_rootdir=args.test_project_root,
            project_root_path=args.project_root,
            test_framework=args.test_framework,
            pytest_cmd=args.pytest_cmd,
        )

        self.aiservice_client = AiServiceClient()
        self.experiment_id = os.getenv("CODEFLASH_EXPERIMENT_ID", None)
        self.local_aiservice_client = LocalAiServiceClient() if self.experiment_id else None

        self.test_files = TestFiles(test_files=[])

    def run(self) -> None:
        ph("cli-optimize-run-start")
        logger.info("Running optimizer.")
        console.rule()
        if not env_utils.ensure_codeflash_api_key():
            return

        file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]]
        num_optimizable_functions: int
        (file_to_funcs_to_optimize, num_optimizable_functions) = get_functions_to_optimize(
            optimize_all=self.args.all,
            replay_test=self.args.replay_test,
            file=self.args.file,
            only_get_this_function=self.args.function,
            test_cfg=self.test_cfg,
            ignore_paths=self.args.ignore_paths,
            project_root=self.args.project_root,
            module_root=self.args.module_root,
        )

        optimizations_found: int = 0
        function_iterator_count: int = 0
        if self.args.test_framework == "pytest":
            self.test_cfg.concolic_test_root_dir = Path(
                tempfile.mkdtemp(dir=self.args.tests_root, prefix="codeflash_concolic_")
            )
        try:
            ph("cli-optimize-functions-to-optimize", {"num_functions": num_optimizable_functions})
            if num_optimizable_functions == 0:
                logger.info("No functions found to optimize. Exiting…")
                return

            console.rule()
            logger.info(f"Discovering existing unit tests in {self.test_cfg.tests_root}…")
            console.rule()
            function_to_tests: dict[str, list[FunctionCalledInTest]] = discover_unit_tests(self.test_cfg)
            num_discovered_tests: int = sum([len(value) for value in function_to_tests.values()])
            console.rule()
            logger.info(f"Discovered {num_discovered_tests} existing unit tests in {self.test_cfg.tests_root}")
            console.rule()
            ph("cli-optimize-discovered-tests", {"num_tests": num_discovered_tests})

            for original_module_path in file_to_funcs_to_optimize:
                logger.info(f"Examining file {original_module_path!s}…")
                console.rule()

                original_module_code: str = original_module_path.read_text(encoding="utf8")
                try:
                    original_module_ast = ast.parse(original_module_code)
                except SyntaxError as e:
                    logger.warning(f"Syntax error parsing code in {original_module_path}: {e}")
                    logger.info("Skipping optimization due to file error.")
                    continue
                normalized_original_module_code = ast.unparse(normalize_node(original_module_ast))
                validated_original_code: dict[Path, ValidCode] = {
                    original_module_path: ValidCode(
                        source_code=original_module_code, normalized_code=normalized_original_module_code
                    )
                }

                imported_module_analyses = analyze_imported_modules(
                    original_module_code, original_module_path, self.args.project_root
                )

                has_syntax_error = False
                for analysis in imported_module_analyses:
                    callee_original_code = analysis.file_path.read_text(encoding="utf8")
                    try:
                        normalized_callee_original_code = normalize_code(callee_original_code)
                    except SyntaxError as e:
                        logger.warning(f"Syntax error parsing code in callee module {analysis.file_path}: {e}")
                        logger.info("Skipping optimization due to helper file error.")
                        has_syntax_error = True
                        break
                    validated_original_code[analysis.file_path] = ValidCode(
                        source_code=callee_original_code, normalized_code=normalized_callee_original_code
                    )
                if has_syntax_error:
                    continue

                for function_to_optimize in file_to_funcs_to_optimize[original_module_path]:
                    function_iterator_count += 1
                    logger.info(
                        f"Optimizing function {function_iterator_count} of {num_optimizable_functions}: "
                        f"{function_to_optimize.qualified_name}"
                    )

                    if not (
                        function_to_optimize_ast := get_first_top_level_function_or_method_ast(
                            function_to_optimize.function_name, function_to_optimize.parents, original_module_ast
                        )
                    ):
                        logger.info(
                            f"Function {function_to_optimize.qualified_name} not found in {original_module_path}.\n"
                            f"Skipping optimization."
                        )
                        continue

                    best_optimization = self.optimize_function(
                        function_to_optimize, function_to_optimize_ast, function_to_tests, validated_original_code
                    )
                    self.test_files = TestFiles(test_files=[])
                    if is_successful(best_optimization):
                        optimizations_found += 1
                    else:
                        logger.warning(best_optimization.failure())
                        console.rule()
                        continue
            ph("cli-optimize-run-finished", {"optimizations_found": optimizations_found})
            if optimizations_found == 0:
                logger.info("❌ No optimizations found.")
            elif self.args.all:
                logger.info("✨ All functions have been optimized! ✨")
        finally:
            for test_file in self.test_files.get_by_type(TestType.GENERATED_REGRESSION).test_files:
                test_file.instrumented_behavior_file_path.unlink(missing_ok=True)
                test_file.benchmarking_file_path.unlink(missing_ok=True)
            for test_file in self.test_files.get_by_type(TestType.EXISTING_UNIT_TEST).test_files:
                test_file.instrumented_behavior_file_path.unlink(missing_ok=True)
                test_file.benchmarking_file_path.unlink(missing_ok=True)
            for test_file in self.test_files.get_by_type(TestType.CONCOLIC_COVERAGE_TEST).test_files:
                test_file.instrumented_behavior_file_path.unlink(missing_ok=True)
            if hasattr(get_run_tmp_file, "tmpdir"):
                get_run_tmp_file.tmpdir.cleanup()
            if self.test_cfg.concolic_test_root_dir:
                shutil.rmtree(self.test_cfg.concolic_test_root_dir, ignore_errors=True)

    def optimize_function(
        self,
        function_to_optimize: FunctionToOptimize,
        function_to_optimize_ast: ast.FunctionDef,
        function_to_tests: dict[str, list[FunctionCalledInTest]],
        validated_original_code: dict[Path, ValidCode],
    ) -> Result[BestOptimization, str]:
        should_run_experiment = self.experiment_id is not None
        function_trace_id: str = str(uuid.uuid4())
        logger.debug(f"Function Trace ID: {function_trace_id}")
        ph("cli-optimize-function-start", {"function_trace_id": function_trace_id})
        self.cleanup_leftover_test_return_values()
        file_name_from_test_module_name.cache_clear()
        ctx_result = self.get_code_optimization_context(
            function_to_optimize,
            self.args.project_root,
            validated_original_code[function_to_optimize.file_path].source_code,
        )
        if not is_successful(ctx_result):
            return Failure(ctx_result.failure())
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code

        logger.info("Code to be optimized:")
        code_print(code_context.read_writable_code)

        original_module_path = module_name_from_file_path(function_to_optimize.file_path, self.args.project_root)

        for module_abspath, helper_code_source in original_helper_code.items():
            code_context.code_to_optimize_with_helpers = add_needed_imports_from_module(
                helper_code_source,
                code_context.code_to_optimize_with_helpers,
                module_abspath,
                function_to_optimize.file_path,
                self.args.project_root,
            )

        generated_test_paths = [
            get_test_file_path(
                self.test_cfg.tests_root, function_to_optimize.function_name, test_index, test_type="unit"
            )
            for test_index in range(N_TESTS_TO_GENERATE)
        ]
        generated_perf_test_paths = [
            get_test_file_path(
                self.test_cfg.tests_root, function_to_optimize.function_name, test_index, test_type="perf"
            )
            for test_index in range(N_TESTS_TO_GENERATE)
        ]

        with progress_bar(
            f"Generating new tests and optimizations for function {function_to_optimize.function_name}", transient=True
        ):
            generated_results = self.generate_tests_and_optimizations(
                code_to_optimize_with_helpers=code_context.code_to_optimize_with_helpers,
                read_writable_code=code_context.read_writable_code,
                read_only_context_code=code_context.read_only_context_code,
                function_to_optimize=function_to_optimize,
                helper_functions=code_context.helper_functions,
                module_path=Path(original_module_path),
                function_trace_id=function_trace_id,
                generated_test_paths=generated_test_paths,
                generated_perf_test_paths=generated_perf_test_paths,
                function_to_optimize_ast=function_to_optimize_ast,
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

        function_to_optimize_qualified_name = function_to_optimize.qualified_name
        function_to_all_tests = {
            key: function_to_tests.get(key, []) + function_to_concolic_tests.get(key, [])
            for key in set(function_to_tests) | set(function_to_concolic_tests)
        }
        instrumented_unittests_created_for_function = self.instrument_existing_tests(
            function_to_optimize=function_to_optimize, function_to_tests=function_to_all_tests
        )

        baseline_result = self.establish_original_code_baseline(  # this needs better typing
            function_name=function_to_optimize_qualified_name,
            function_file_path=function_to_optimize.file_path,
            code_context=code_context,
        )

        console.rule()
        paths_to_cleanup = (
            generated_test_paths + generated_perf_test_paths + list(instrumented_unittests_created_for_function)
        )

        if not is_successful(baseline_result):
            cleanup_paths(paths_to_cleanup)
            return Failure(baseline_result.failure())

        (
            original_code_baseline,
            original_code_runtime_distribution,
            original_code_runtime_statistics,
            test_functions_to_remove,
        ) = baseline_result.unwrap()
        if isinstance(original_code_baseline, OriginalCodeBaseline) and not coverage_critic(
            original_code_baseline.coverage_results, self.args.test_framework
        ):
            cleanup_paths(paths_to_cleanup)
            return Failure("The threshold for test coverage was not met.")

        best_optimization = None

        for u, candidates in enumerate([optimizations_set.control, optimizations_set.experiment]):
            if candidates is None:
                continue

            best_optimization = self.determine_best_candidate(
                candidates=candidates,
                code_context=code_context,
                function_to_optimize=function_to_optimize,
                original_code=validated_original_code[function_to_optimize.file_path].source_code,
                original_code_baseline=original_code_baseline,
                original_code_runtime_distribution=original_code_runtime_distribution,
                original_helper_code=original_helper_code,
                function_trace_id=function_trace_id[:-4] + f"EXP{u}" if should_run_experiment else function_trace_id,
            )
            ph("cli-optimize-function-finished", {"function_trace_id": function_trace_id})

            generated_tests = remove_functions_from_generated_tests(
                generated_tests=generated_tests, test_functions_to_remove=test_functions_to_remove
            )

            if best_optimization:
                logger.info("Best candidate:")
                code_print(best_optimization.candidate.source_code)
                console.print(
                    Panel(
                        best_optimization.candidate.explanation, title="Best Candidate Explanation", border_style="blue"
                    )
                )
                explanation = Explanation(
                    raw_explanation_message=best_optimization.candidate.explanation,
                    winning_behavioral_test_results=best_optimization.winning_behavioral_test_results,
                    winning_benchmarking_test_results=best_optimization.winning_benchmarking_test_results,
                    original_runtime_ns=original_code_baseline.runtime,
                    best_runtime_ns=best_optimization.runtime,
                    function_name=function_to_optimize_qualified_name,
                    file_path=function_to_optimize.file_path,
                )

                self.log_successful_optimization(explanation, function_to_optimize, function_trace_id, generated_tests)

                self.replace_function_and_helpers_with_optimized_code(
                    code_context=code_context,
                    function_to_optimize_file_path=explanation.file_path,
                    optimized_code=best_optimization.candidate.source_code,
                    qualified_function_name=function_to_optimize_qualified_name,
                )

                new_code, new_helper_code = self.reformat_code_and_helpers(
                    code_context.helper_functions,
                    explanation.file_path,
                    validated_original_code[function_to_optimize.file_path].source_code,
                )

                existing_tests = existing_tests_source_for(
                    function_to_optimize.qualified_name_with_modules_from_root(self.args.project_root),
                    function_to_all_tests,
                    tests_root=self.test_cfg.tests_root,
                )

                original_code_combined = original_helper_code.copy()
                original_code_combined[explanation.file_path] = validated_original_code[
                    function_to_optimize.file_path
                ].source_code
                new_code_combined = new_helper_code.copy()
                new_code_combined[explanation.file_path] = new_code
                if not self.args.no_pr:
                    coverage_message = (
                        original_code_baseline.coverage_results.build_message()
                        if original_code_baseline.coverage_results
                        else "Coverage data not available"
                    )
                    generated_tests_str = "\n\n".join(
                        [test.generated_original_test_source for test in generated_tests.generated_tests]
                    )
                    if concolic_test_str:
                        generated_tests_str += "\n\n" + concolic_test_str

                    check_create_pr(
                        original_code=original_code_combined,
                        new_code=new_code_combined,
                        explanation=explanation,
                        existing_tests_source=existing_tests,
                        generated_original_test_source=generated_tests_str,
                        function_trace_id=function_trace_id,
                        coverage_message=coverage_message,
                        git_remote=self.args.git_remote,
                    )
                    if self.args.all or env_utils.get_pr_number():
                        self.write_code_and_helpers(
                            validated_original_code[function_to_optimize.file_path].source_code,
                            original_helper_code,
                            function_to_optimize.file_path,
                        )
        for generated_test_path in generated_test_paths:
            generated_test_path.unlink(missing_ok=True)
        for generated_perf_test_path in generated_perf_test_paths:
            generated_perf_test_path.unlink(missing_ok=True)
        for test_paths in instrumented_unittests_created_for_function:
            test_paths.unlink(missing_ok=True)
        for fn in function_to_concolic_tests:
            for test in function_to_concolic_tests[fn]:
                if not test.tests_in_file.test_file.parent.exists():
                    logger.warning(
                        f"Concolic test directory {test.tests_in_file.test_file.parent} does not exist so could not be deleted."
                    )
                shutil.rmtree(test.tests_in_file.test_file.parent, ignore_errors=True)
                break  # need to delete only one test directory

        if not best_optimization:
            return Failure(f"No best optimizations found for function {function_to_optimize.qualified_name}")
        return Success(best_optimization)

    def determine_best_candidate(
        self,
        *,
        candidates: list[OptimizedCandidate],
        code_context: CodeOptimizationContext,
        function_to_optimize: FunctionToOptimize,
        original_code: str,
        original_code_baseline: OriginalCodeBaseline,
        original_code_runtime_distribution: npt.NDArray[np.float64],
        original_helper_code: dict[Path, str],
        function_trace_id: str,
    ) -> BestOptimization | None:
        best_optimization: BestOptimization | None = None
        best_runtime_until_now = original_code_baseline.runtime
        best_speedup_ratio_until_now = 1.0

        speedup_ratios: dict[str, float | None] = {}
        optimized_runtimes: dict[str, float | None] = {}
        is_correct = {}

        logger.info(
            f"Determining best optimization candidate (out of {len(candidates)}) for "
            f"{function_to_optimize.qualified_name}…"
        )
        console.rule()
        try:
            for candidate_index, candidate in enumerate(candidates, start=1):
                get_run_tmp_file(Path(f"test_return_values_{candidate_index}.bin")).unlink(missing_ok=True)
                get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite")).unlink(missing_ok=True)
                logger.info(f"Optimization candidate {candidate_index}/{len(candidates)}:")
                code_print(candidate.source_code)
                try:
                    did_update = self.replace_function_and_helpers_with_optimized_code(
                        code_context=code_context,
                        function_to_optimize_file_path=function_to_optimize.file_path,
                        optimized_code=candidate.source_code,
                        qualified_function_name=function_to_optimize.qualified_name,
                    )
                    if not did_update:
                        logger.warning(
                            "No functions were replaced in the optimized code. Skipping optimization candidate."
                        )
                        console.rule()
                        continue
                except (ValueError, SyntaxError, cst.ParserSyntaxError, AttributeError) as e:
                    logger.error(e)
                    self.write_code_and_helpers(original_code, original_helper_code, function_to_optimize.file_path)
                    continue

                run_results = self.run_optimized_candidate(
                    optimization_candidate_index=candidate_index, baseline_results=original_code_baseline
                )
                console.rule()
                if not is_successful(run_results):
                    optimized_runtimes[candidate.optimization_id] = None
                    is_correct[candidate.optimization_id] = False
                    speedup_ratios[candidate.optimization_id] = None
                else:
                    candidate_result, candidate_runtime_distribution, candidate_runtime_statistics = (
                        run_results.unwrap()
                    )
                    best_test_runtime = candidate_result.best_test_runtime
                    optimized_runtimes[candidate.optimization_id] = best_test_runtime
                    is_correct[candidate.optimization_id] = True
                    perf_gain = performance_gain(
                        original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=best_test_runtime
                    )
                    speedup_ratios[candidate.optimization_id] = perf_gain

                    speedup_stats = compare_function_runtime_distributions(
                        original_code_runtime_distribution, candidate_runtime_distribution
                    )

                    tree = Tree(f"Candidate #{candidate_index} - Sum of Minimum Runtimes")
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
                        tree.add(f"Speedup ratio: {perf_gain + 1:.1f}X")

                        best_optimization = BestOptimization(
                            candidate=candidate,
                            helper_functions=code_context.helper_functions,
                            runtime=best_test_runtime,
                            winning_behavioral_test_results=candidate_result.behavior_test_results,
                            winning_benchmarking_test_results=candidate_result.benchmarking_test_results,
                        )
                        best_runtime_until_now = best_test_runtime
                    else:
                        tree.add(
                            f"Summed runtime: {humanize_runtime(best_test_runtime)} "
                            f"(measured over {candidate_result.max_loop_count} "
                            f"loop{'s' if candidate_result.max_loop_count > 1 else ''})"
                        )
                        tree.add(f"Speedup percentage: {perf_gain * 100:.1f}%")
                        tree.add(f"Speedup ratio: {perf_gain + 1:.1f}X")
                    console.print(tree)
                    console.rule()

                    tree = Tree(f"Candidate #{candidate_index} - Bayesian Bootstrapping Nonparametric Analysis")
                    tree.add(
                        f"Expected candidate runtime (95% Credible Interval) = ["
                        f"{humanize_runtime(candidate_runtime_statistics['credible_interval_lower_bound'])}, "
                        f"{humanize_runtime(candidate_runtime_statistics['credible_interval_upper_bound'])}], "
                        f"\nmedian = {humanize_runtime(candidate_runtime_statistics['median'])}"
                        f"\nSpeedup ratio of candidate vs original:"
                        f"\n95% Credible Interval = [{speedup_stats['credible_interval_lower_bound']:.3f}X, "
                        f"{speedup_stats['credible_interval_upper_bound']:.3f}X]"
                        f"\nmedian = {speedup_stats['median']:.3f}X"
                    )
                    if speedup_stats["credible_interval_lower_bound"] > 1.0:
                        tree.add("The candidate is faster than the original code with a 95% probability.")
                        if speedup_stats["median"] > best_speedup_ratio_until_now:
                            best_speedup_ratio_until_now = speedup_stats["median"]
                            tree.add("This candidate is the best candidate so far.")
                        else:
                            tree.add("This candidate is not faster than the current fastest candidate.")
                    else:
                        tree.add("It is inconclusive whether the candidate is faster than the original code.")
                    console.print(tree)
                    console.rule()

                self.write_code_and_helpers(original_code, original_helper_code, function_to_optimize.file_path)
        except KeyboardInterrupt as e:
            self.write_code_and_helpers(original_code, original_helper_code, function_to_optimize.file_path)
            logger.exception(f"Optimization interrupted: {e}")
            raise

        self.aiservice_client.log_results(
            function_trace_id=function_trace_id,
            speedup_ratio=speedup_ratios,
            original_runtime=original_code_baseline.runtime,
            optimized_runtime=optimized_runtimes,
            is_correct=is_correct,
        )
        return best_optimization

    def log_successful_optimization(
        self,
        explanation: Explanation,
        function_to_optimize: FunctionToOptimize,
        function_trace_id: str,
        generated_tests: GeneratedTestsList,
    ) -> None:
        explanation_panel = Panel(
            f"⚡️ Optimization successful! 📄 {function_to_optimize.qualified_name} in {explanation.file_path}\n"
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
        console.print(explanation_panel)

        ph(
            "cli-optimize-success",
            {
                "function_trace_id": function_trace_id,
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
        for module_abspath in original_helper_code:
            with Path(module_abspath).open("w", encoding="utf8") as f:
                f.write(original_helper_code[module_abspath])

    def reformat_code_and_helpers(
        self, helper_functions: list[FunctionSource], path: Path, original_code: str
    ) -> tuple[str, dict[Path, str]]:
        should_sort_imports = not self.args.disable_imports_sorting
        if should_sort_imports and isort.code(original_code) != original_code:
            should_sort_imports = False

        new_code = format_code(self.args.formatter_cmds, path)
        if should_sort_imports:
            new_code = sort_imports(new_code)

        new_helper_code: dict[Path, str] = {}
        helper_functions_paths = {hf.file_path for hf in helper_functions}
        for module_abspath in helper_functions_paths:
            formatted_helper_code = format_code(self.args.formatter_cmds, module_abspath)
            if should_sort_imports:
                formatted_helper_code = sort_imports(formatted_helper_code)
            new_helper_code[module_abspath] = formatted_helper_code

        return new_code, new_helper_code

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        function_to_optimize_file_path: Path,
        optimized_code: str,
        qualified_function_name: str,
    ) -> bool:
        did_update = False
        read_writable_functions_by_file_path = defaultdict(set)
        read_writable_functions_by_file_path[function_to_optimize_file_path].add(qualified_function_name)
        for helper_function in code_context.helper_functions:
            if helper_function.jedi_definition.type != "class":
                read_writable_functions_by_file_path[helper_function.file_path].add(helper_function.qualified_name)
        for module_abspath, qualified_names in read_writable_functions_by_file_path.items():
            did_update |= replace_function_definitions_in_module(
                function_names=list(qualified_names),
                optimized_code=optimized_code,
                file_path_of_module_with_function_to_optimize=function_to_optimize_file_path,
                module_abspath=module_abspath,
                preexisting_objects=code_context.preexisting_objects,
                project_root_path=self.args.project_root,
            )
        return did_update

    def get_code_optimization_context(
        self, function_to_optimize: FunctionToOptimize, project_root: Path, original_source_code: str
    ) -> Result[CodeOptimizationContext, str]:
        code_to_optimize, contextual_dunder_methods = extract_code([function_to_optimize])
        if code_to_optimize is None:
            return Failure("Could not find function to optimize.")
        (helper_code, helper_functions, helper_dunder_methods) = get_constrained_function_context_and_helper_functions(
            function_to_optimize, self.args.project_root, code_to_optimize
        )
        if function_to_optimize.parents:
            function_class = function_to_optimize.parents[0].name
            same_class_helper_methods = [
                df
                for df in helper_functions
                if df.qualified_name.count(".") > 0 and df.qualified_name.split(".")[0] == function_class
            ]
            optimizable_methods = [
                FunctionToOptimize(
                    df.qualified_name.split(".")[-1],
                    df.file_path,
                    [FunctionParent(df.qualified_name.split(".")[0], "ClassDef")],
                    None,
                    None,
                )
                for df in same_class_helper_methods
            ] + [function_to_optimize]
            dedup_optimizable_methods = []
            added_methods = set()
            for method in reversed(optimizable_methods):
                if f"{method.file_path}.{method.qualified_name}" not in added_methods:
                    dedup_optimizable_methods.append(method)
                    added_methods.add(f"{method.file_path}.{method.qualified_name}")
            if len(dedup_optimizable_methods) > 1:
                code_to_optimize, contextual_dunder_methods = extract_code(list(reversed(dedup_optimizable_methods)))
                if code_to_optimize is None:
                    return Failure("Could not find function to optimize.")
        code_to_optimize_with_helpers = helper_code + "\n" + code_to_optimize

        code_to_optimize_with_helpers_and_imports = add_needed_imports_from_module(
            original_source_code,
            code_to_optimize_with_helpers,
            function_to_optimize.file_path,
            function_to_optimize.file_path,
            project_root,
            helper_functions,
        )

        try:
            new_code_ctx = code_context_extractor.get_code_optimization_context(function_to_optimize, project_root)
        except ValueError as e:
            return Failure(str(e))

        return Success(
            CodeOptimizationContext(
                code_to_optimize_with_helpers=code_to_optimize_with_helpers_and_imports,
                read_writable_code=new_code_ctx.read_writable_code,
                read_only_context_code=new_code_ctx.read_only_context_code,
                helper_functions=new_code_ctx.helper_functions,  # only functions that are read writable
                preexisting_objects=new_code_ctx.preexisting_objects,
            )
        )

    @staticmethod
    def cleanup_leftover_test_return_values() -> None:
        # remove leftovers from previous run
        get_run_tmp_file(Path("test_return_values_0.bin")).unlink(missing_ok=True)
        get_run_tmp_file(Path("test_return_values_0.sqlite")).unlink(missing_ok=True)

    def instrument_existing_tests(
        self, function_to_optimize: FunctionToOptimize, function_to_tests: dict[str, list[FunctionCalledInTest]]
    ) -> set[Path]:
        existing_test_files_count = 0
        replay_test_files_count = 0
        concolic_coverage_test_files_count = 0
        unique_instrumented_test_files = set()

        func_qualname = function_to_optimize.qualified_name_with_modules_from_root(self.args.project_root)
        if func_qualname not in function_to_tests:
            logger.info(f"Did not find any pre-existing tests for '{func_qualname}', will only use generated tests.")
            console.rule()
        else:
            test_file_invocation_positions = defaultdict(list[FunctionCalledInTest])
            for tests_in_file in function_to_tests.get(func_qualname):
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
                    function_to_optimize=function_to_optimize,
                    tests_project_root=self.test_cfg.tests_project_rootdir,
                    test_framework=self.args.test_framework,
                )
                if not success:
                    continue
                success, injected_perf_test = inject_profiling_into_existing_test(
                    mode=TestingMode.PERFORMANCE,
                    test_path=path_obj_test_file,
                    call_positions=[test.position for test in tests_in_file_list],
                    function_to_optimize=function_to_optimize,
                    tests_project_root=self.test_cfg.tests_project_rootdir,
                    test_framework=self.args.test_framework,
                )
                if not success:
                    continue
                # TODO: this naming logic should be moved to a function and made more standard
                new_behavioral_test_path = Path(
                    f"{os.path.splitext(test_file)[0]}__perfinstrumented{os.path.splitext(test_file)[1]}"
                )
                new_perf_test_path = Path(
                    f"{os.path.splitext(test_file)[0]}__perfonlyinstrumented{os.path.splitext(test_file)[1]}"
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
        code_to_optimize_with_helpers: str,
        read_writable_code: str,
        read_only_context_code: str,
        function_to_optimize: FunctionToOptimize,
        helper_functions: list[FunctionSource],
        module_path: Path,
        function_trace_id: str,
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
        function_to_optimize_ast: ast.FunctionDef,
        run_experiment: bool = False,
    ) -> Result[tuple[GeneratedTestsList, dict[str, list[FunctionCalledInTest]], OptimizationSet], str]:
        assert len(generated_test_paths) == N_TESTS_TO_GENERATE
        max_workers = N_TESTS_TO_GENERATE + 2 if not run_experiment else N_TESTS_TO_GENERATE + 3
        console.rule()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit the test generation task as future
            future_tests = self.generate_and_instrument_tests(
                executor,
                code_to_optimize_with_helpers,
                function_to_optimize,
                [definition.fully_qualified_name for definition in helper_functions],
                module_path,
                generated_test_paths,
                generated_perf_test_paths,
                (function_trace_id[:-4] + "EXP0" if run_experiment else function_trace_id),
            )
            future_optimization_candidates = executor.submit(
                self.aiservice_client.optimize_python_code,
                read_writable_code,
                read_only_context_code,
                function_trace_id[:-4] + "EXP0" if run_experiment else function_trace_id,
                N_CANDIDATES,
                ExperimentMetadata(id=self.experiment_id, group="control") if run_experiment else None,
            )
            future_candidates_exp = None

            future_concolic_tests = executor.submit(
                generate_concolic_tests, self.test_cfg, self.args, function_to_optimize, function_to_optimize_ast
            )
            futures = [*future_tests, future_optimization_candidates, future_concolic_tests]
            if run_experiment:
                future_candidates_exp = executor.submit(
                    self.local_aiservice_client.optimize_python_code,
                    read_writable_code,
                    read_only_context_code,
                    function_trace_id[:-4] + "EXP1",
                    N_CANDIDATES,
                    ExperimentMetadata(id=self.experiment_id, group="experiment"),
                )
                futures.append(future_candidates_exp)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

            # Retrieve results
            candidates: list[OptimizedCandidate] = future_optimization_candidates.result()
            if not candidates:
                return Failure(f"/!\\ NO OPTIMIZATIONS GENERATED for {function_to_optimize.function_name}")

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
                logger.warning(f"Failed to generate and instrument tests for {function_to_optimize.function_name}")
                return Failure(f"/!\\ NO TESTS GENERATED for {function_to_optimize.function_name}")
            function_to_concolic_tests, concolic_test_str = future_concolic_tests.result()
            logger.info(f"Generated {len(tests)} tests for {function_to_optimize.function_name}")
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
        self, function_name: str, function_file_path: Path, code_context: CodeOptimizationContext
    ) -> Result[tuple[OriginalCodeBaseline, npt.NDArray[np.float64], dict[str, np.float64], list[str]], str]:
        # For the original function - run the tests and get the runtime, plus coverage
        with progress_bar(f"Establishing original code baseline for {function_name}"):
            assert (test_framework := self.args.test_framework) in ["pytest", "unittest"]
            success = True

            test_env = os.environ.copy()
            test_env["CODEFLASH_TEST_ITERATION"] = "0"
            test_env["CODEFLASH_TRACER_DISABLE"] = "1"
            if "PYTHONPATH" not in test_env:
                test_env["PYTHONPATH"] = str(self.args.project_root)
            else:
                test_env["PYTHONPATH"] += os.pathsep + str(self.args.project_root)

            coverage_results = None
            behavioral_results, coverage_results = self.run_and_parse_tests(
                testing_type=TestingMode.BEHAVIOR,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=0,
                testing_time=TOTAL_LOOPING_TIME,
                enable_coverage=test_framework == "pytest",
                function_name=function_name,
                source_file=function_file_path,
                code_context=code_context,
            )
            if test_framework == "pytest":
                benchmarking_results, _ = self.run_and_parse_tests(
                    testing_type=TestingMode.PERFORMANCE,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=0,
                    testing_time=TOTAL_LOOPING_TIME,
                    enable_coverage=False,
                    function_name=function_name,
                    source_file=function_file_path,
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
                        function_name=function_name,
                        source_file=function_file_path,
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
            runtime_distribution, runtime_statistics = benchmarking_results.bayesian_nonparametric_bootstrap_analysis(
                100_000
            )
            functions_to_remove = [
                result.id.test_function_name
                for result in behavioral_results
                if (result.test_type == TestType.GENERATED_REGRESSION and not result.did_pass)
            ]

            if not behavioral_results:
                logger.warning(
                    f"Couldn't run any tests for original function {function_name}. SKIPPING OPTIMIZING THIS FUNCTION."
                )
                console.rule()
                success = False
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
            logger.debug(f"Total original code summed runtime (ns): {total_timing}")
            console.rule()
            logger.info(
                f"Bayesian Bootstrapping Nonparametric Analysis"
                f"\nExpected original code runtime (95% Credible Interval) = ["
                f"{humanize_runtime(round(runtime_statistics['credible_interval_lower_bound']))}, "
                f"{humanize_runtime(round(runtime_statistics['credible_interval_upper_bound']))}], "
                f"\nmedian: {humanize_runtime(round(runtime_statistics['median']))}"
            )

            return Success(
                (
                    OriginalCodeBaseline(
                        behavioral_test_results=behavioral_results,
                        benchmarking_test_results=benchmarking_results,
                        runtime=total_timing,
                        coverage_results=coverage_results,
                    ),
                    runtime_distribution,
                    runtime_statistics,
                    functions_to_remove,
                )
            )

    def run_optimized_candidate(
        self, *, optimization_candidate_index: int, baseline_results: OriginalCodeBaseline
    ) -> Result[tuple[OptimizedCandidateResult, npt.NDArray[np.float64], dict[str, np.float64]], str]:
        assert (test_framework := self.args.test_framework) in ["pytest", "unittest"]

        with progress_bar("Testing optimization candidate"):
            test_env = os.environ.copy()
            test_env["CODEFLASH_TEST_ITERATION"] = str(optimization_candidate_index)
            test_env["CODEFLASH_TRACER_DISABLE"] = "1"
            if "PYTHONPATH" not in test_env:
                test_env["PYTHONPATH"] = str(self.args.project_root)
            else:
                test_env["PYTHONPATH"] += os.pathsep + str(self.args.project_root)

            get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite")).unlink(missing_ok=True)
            get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite")).unlink(missing_ok=True)

            candidate_behavior_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.BEHAVIOR,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=optimization_candidate_index,
                testing_time=TOTAL_LOOPING_TIME,
                enable_coverage=False,
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
            runtime_distribution, runtime_statistics = (
                candidate_benchmarking_results.bayesian_nonparametric_bootstrap_analysis(100_000)
            )

            logger.debug(f"Total optimized code {optimization_candidate_index} runtime (ns): {total_candidate_timing}")
            console.rule()
            logger.debug(
                f"Overall code runtime (95% Credible Interval) = ["
                f"{humanize_runtime(round(runtime_statistics['credible_interval_lower_bound']))}, "
                f"{humanize_runtime(round(runtime_statistics['credible_interval_upper_bound']))}], median: "
                f"{humanize_runtime(round(runtime_statistics['median']))}"
            )
            return Success(
                (
                    OptimizedCandidateResult(
                        max_loop_count=loop_count,
                        best_test_runtime=total_candidate_timing,
                        behavior_test_results=candidate_behavior_results,
                        benchmarking_test_results=candidate_benchmarking_results,
                        optimization_candidate_index=optimization_candidate_index,
                        total_candidate_timing=total_candidate_timing,
                    ),
                    runtime_distribution,
                    runtime_statistics,
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
        function_name: str | None = None,
        source_file: Path | None = None,
        code_context: CodeOptimizationContext | None = None,
        unittest_loop_index: int | None = None,
    ) -> tuple[TestResults, CoverageData | None]:
        coverage_database_file = None
        try:
            if testing_type == TestingMode.BEHAVIOR:
                result_file_path, run_result, coverage_database_file = run_behavioral_tests(
                    test_files,
                    test_framework=self.args.test_framework,
                    cwd=self.args.project_root,
                    test_env=test_env,
                    pytest_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    pytest_cmd=self.test_cfg.pytest_cmd,
                    verbose=True,
                    enable_coverage=enable_coverage,
                )
            elif testing_type == TestingMode.PERFORMANCE:
                result_file_path, run_result = run_benchmarking_tests(
                    test_files,
                    cwd=self.args.project_root,
                    test_env=test_env,
                    pytest_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    pytest_cmd=self.test_cfg.pytest_cmd,
                    pytest_target_runtime_seconds=testing_time,
                    pytest_min_loops=pytest_min_loops,
                    pytest_max_loops=pytest_max_loops,
                    test_framework=self.args.test_framework,
                )
            else:
                raise ValueError(f"Unexpected testing type: {testing_type}")
        except subprocess.TimeoutExpired:
            logger.exception(
                f'Error running tests in {", ".join(str(f) for f in test_files.test_files)}.\nTimeout Error'
            )
            return TestResults(), None
        if run_result.returncode != 0:
            logger.debug(
                f'Nonzero return code {run_result.returncode} when running tests in '
                f'{", ".join([str(f.instrumented_behavior_file_path) for f in test_files.test_files])}.\n'
                f"stdout: {run_result.stdout}\n"
                f"stderr: {run_result.stderr}\n"
            )

        results, coverage_results = parse_test_results(
            test_xml_path=result_file_path,
            test_files=test_files,
            test_config=self.test_cfg,
            optimization_iteration=optimization_iteration,
            run_result=run_result,
            unittest_loop_index=unittest_loop_index,
            function_name=function_name,
            source_file=source_file,
            code_context=code_context,
            coverage_database_file=coverage_database_file,
        )
        return results, coverage_results

    def generate_and_instrument_tests(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        source_code_being_tested: str,
        function_to_optimize: FunctionToOptimize,
        helper_function_names: list[str],
        module_path: Path,
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
        function_trace_id: str,
    ) -> list[concurrent.futures.Future]:
        return [
            executor.submit(
                generate_tests,
                self.aiservice_client,
                source_code_being_tested,
                function_to_optimize,
                helper_function_names,
                module_path,
                self.test_cfg,
                INDIVIDUAL_TESTCASE_TIMEOUT,
                function_trace_id,
                test_index,
                test_path,
                test_perf_path,
            )
            for test_index, (test_path, test_perf_path) in enumerate(
                zip(generated_test_paths, generated_perf_test_paths)
            )
        ]


def run_with_args(args: Namespace) -> None:
    optimizer = Optimizer(args)
    optimizer.run()
