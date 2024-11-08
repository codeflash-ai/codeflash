from __future__ import annotations

import concurrent.futures
import os
import re
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
from returns.pipeline import is_successful
from returns.result import Failure, Success
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from codeflash.api.aiservice import AiServiceClient, LocalAiServiceClient
from codeflash.cli_cmds.console import code_print, console, logger, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_extractor import add_needed_imports_from_module, extract_code, find_preexisting_objects
from codeflash.code_utils.code_replacer import (
    normalize_code,
    replace_function_definitions_in_module,
    replace_functions_and_add_imports,
)
from codeflash.code_utils.code_utils import (
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
from codeflash.code_utils.git_utils import git_root_dir
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.code_utils.remove_generated_tests import remove_functions_from_generated_tests
from codeflash.code_utils.static_analysis import analyze_imported_modules
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.discovery.functions_to_optimize import FunctionToOptimize, get_functions_to_optimize
from codeflash.models.ExperimentMetadata import ExperimentMetadata
from codeflash.models.models import (
    BestOptimization,
    CodeOptimizationContext,
    DiffbehaviorReturnCode,
    FunctionParent,
    GeneratedTests,
    GeneratedTestsList,
    OptimizationSet,
    OptimizedCandidateResult,
    OriginalCodeBaseline,
    TestFile,
    TestFiles,
    ValidCode,
)
from codeflash.optimization.function_context import get_constrained_function_context_and_helper_functions
from codeflash.result.create_pr import check_create_pr, existing_tests_source_for
from codeflash.result.critic import performance_gain, quantity_of_tests_critic, speedup_critic
from codeflash.result.explanation import Explanation
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.equivalence import compare_test_results
from codeflash.verification.parse_test_output import parse_test_results
from codeflash.verification.test_results import TestResults, TestType
from codeflash.verification.test_runner import run_tests
from codeflash.verification.verification_utils import TestConfig, get_test_file_path
from codeflash.verification.verifier import generate_tests

if TYPE_CHECKING:
    from argparse import Namespace

    from returns.result import Result

    from codeflash.models.models import FunctionCalledInTest, FunctionSource, OptimizedCandidate


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

        try:
            ph("cli-optimize-functions-to-optimize", {"num_functions": num_optimizable_functions})
            if num_optimizable_functions == 0:
                logger.info("No functions found to optimize. Exiting…")
                return

            console.rule()
            logger.info(f"Discovering existing unit tests in {self.test_cfg.tests_root}…")
            function_to_tests: dict[str, list[FunctionCalledInTest]] = discover_unit_tests(self.test_cfg)
            num_discovered_tests: int = sum([len(value) for value in function_to_tests.values()])
            logger.info(f"Discovered {num_discovered_tests} existing unit tests in {self.test_cfg.tests_root}")
            console.rule()
            ph("cli-optimize-discovered-tests", {"num_tests": num_discovered_tests})

            # TODO CROSSHAIR: Handle no git case.
            git_root = git_root_dir()

            for original_module_path in file_to_funcs_to_optimize:
                logger.info(f"Examining file {original_module_path!s}…")

                original_module_code: str = original_module_path.read_text(encoding="utf8")
                imported_module_analyses = analyze_imported_modules(
                    original_module_code, original_module_path, self.args.project_root
                )
                callee_module_paths = {analysis.file_path for analysis in imported_module_analyses}

                try:
                    normalized_original_module_code = normalize_code(original_module_code)
                except SyntaxError as e:
                    logger.warning(f"Syntax error parsing code in {original_module_path}: {e}")
                    continue
                validated_original_code: dict[Path, ValidCode] = {
                    original_module_path: ValidCode(
                        source_code=original_module_code, normalized_code=normalized_original_module_code
                    )
                }

                has_syntax_error = False
                for analysis in imported_module_analyses:
                    callee_original_code = analysis.file_path.read_text(encoding="utf8")
                    try:
                        normalized_callee_original_code = normalize_code(callee_original_code)
                    except SyntaxError as e:
                        logger.warning(f"Syntax error parsing code in {analysis.file_path}: {e}")
                        has_syntax_error = True
                        break
                    validated_original_code[analysis.file_path] = ValidCode(
                        source_code=callee_original_code, normalized_code=normalized_callee_original_code
                    )
                if has_syntax_error:
                    continue

                for function_to_optimize in file_to_funcs_to_optimize[original_module_path]:
                    worktree_root: Path = Path(tempfile.mkdtemp())
                    worktrees: list[Path] = [Path(tempfile.mkdtemp(dir=worktree_root)) for _ in range(N_CANDIDATES + 1)]
                    # TODO CROSSHAIR Handle no git case.
                    for worktree in worktrees:
                        subprocess.run(
                            ["git", "worktree", "add", "-d", worktree], cwd=self.args.module_root, check=True
                        )

                    function_iterator_count += 1
                    logger.info(
                        f"Optimizing function {function_iterator_count} of {num_optimizable_functions}: "
                        f"{function_to_optimize.qualified_name}"
                    )

                    best_optimization = self.optimize_function(
                        function_to_optimize,
                        function_to_tests,
                        callee_module_paths,
                        validated_original_code,
                        worktree_root,
                        worktrees,
                        git_root,
                    )
                    self.test_files = TestFiles(test_files=[])

                    try:
                        for worktree in worktrees:
                            subprocess.run(["git", "worktree", "remove", "-f", worktree], check=True)
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Error removing worktrees: {e}")
                    shutil.rmtree(worktree_root)

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
                test_file.instrumented_file_path.unlink(missing_ok=True)
            for test_file in self.test_files.get_by_type(TestType.EXISTING_UNIT_TEST).test_files:
                test_file.instrumented_file_path.unlink(missing_ok=True)
            if hasattr(get_run_tmp_file, "tmpdir"):
                get_run_tmp_file.tmpdir.cleanup()

    def optimize_function(
        self,
        function_to_optimize: FunctionToOptimize,
        function_to_tests: dict[str, list[FunctionCalledInTest]],
        callee_module_paths: set[Path],
        validated_original_code: dict[Path, ValidCode],
        worktree_root: Path,
        worktrees: list[Path],
        git_root: Path,
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

        code_print(code_context.code_to_optimize_with_helpers)

        original_module_path = module_name_from_file_path(function_to_optimize.file_path, self.args.project_root)

        for module_abspath in original_helper_code:
            code_context.code_to_optimize_with_helpers = add_needed_imports_from_module(
                original_helper_code[module_abspath],
                code_context.code_to_optimize_with_helpers,
                module_abspath,
                function_to_optimize.file_path,
                self.args.project_root,
            )

        instrumented_unittests_created_for_function = self.instrument_existing_tests(
            function_to_optimize=function_to_optimize, function_to_tests=function_to_tests
        )

        with progress_bar(
            f"Generating new tests and optimizations for function {function_to_optimize.function_name}", transient=True
        ):
            generated_results = self.generate_tests_and_optimizations(
                code_context.code_to_optimize_with_helpers,
                function_to_optimize,
                code_context.helper_functions,
                Path(original_module_path),
                function_trace_id,
                run_experiment=should_run_experiment,
            )

        if not is_successful(generated_results):
            return Failure(generated_results.failure())
        generated_tests: GeneratedTestsList
        optimizations_set: OptimizationSet
        generated_tests, optimizations_set = generated_results.unwrap()
        count_tests = len(generated_tests.generated_tests)
        generated_tests_paths = [
            get_test_file_path(self.args.tests_root, function_to_optimize.function_name, i) for i in range(count_tests)
        ]

        for i, generated_test in enumerate(generated_tests.generated_tests):
            generated_tests_path = generated_tests_paths[i]
            with generated_tests_path.open("w", encoding="utf8") as f:
                f.write(generated_test.instrumented_test_source)
            self.test_files.add(
                TestFile(
                    instrumented_file_path=generated_tests_path,
                    original_file_path=None,
                    original_source=generated_test.generated_original_test_source,
                    test_type=TestType.GENERATED_REGRESSION,
                )
            )
            logger.info(f"Generated test {i + 1}/{count_tests}:")
            code_print(generated_test.generated_original_test_source)

        function_to_optimize_qualified_name = function_to_optimize.qualified_name
        baseline_result = self.establish_original_code_baseline(
            function_to_optimize_qualified_name,
            function_to_tests.get(original_module_path + "." + function_to_optimize_qualified_name, []),
        )
        console.rule()
        if not is_successful(baseline_result):
            for generated_test_path in generated_tests_paths:
                generated_test_path.unlink(missing_ok=True)

            for instrumented_path in instrumented_unittests_created_for_function:
                instrumented_path.unlink(missing_ok=True)
            return Failure(baseline_result.failure())
        original_code_baseline, test_functions_to_remove = baseline_result.unwrap()

        best_optimization = None

        for u, candidates in enumerate([optimizations_set.control, optimizations_set.experiment]):
            if candidates is None:
                continue

            initial_optimized_code = {
                candidate.optimization_id: replace_functions_and_add_imports(
                    validated_original_code[function_to_optimize.file_path].source_code,
                    [function_to_optimize_qualified_name],
                    candidate.source_code,
                    function_to_optimize.file_path,
                    function_to_optimize.file_path,
                    code_context.preexisting_objects,
                    code_context.contextual_dunder_methods,
                    self.args.project_root,
                )
                for candidate in candidates
            }
            callee_original_code = {
                module_path: validated_original_code[module_path].source_code for module_path in callee_module_paths
            }
            intermediate_original_code = {
                candidate.optimization_id: callee_original_code for candidate in candidates
            } | {
                candidate.optimization_id: {
                    function_to_optimize.file_path: initial_optimized_code[candidate.optimization_id]
                }
                for candidate in candidates
            }
            module_paths = callee_module_paths | {function_to_optimize.file_path}
            optimized_code = {
                candidate.optimization_id: {
                    module_path: replace_functions_and_add_imports(
                        intermediate_original_code[candidate.optimization_id][module_path],
                        (
                            [
                                callee.qualified_name
                                for callee in code_context.helper_functions
                                if callee.file_path == module_path and callee.jedi_definition.type != "class"
                            ]
                        ),
                        candidate.source_code,
                        function_to_optimize.file_path,
                        module_path,
                        [],
                        code_context.contextual_dunder_methods,
                        self.args.project_root,
                    )
                    for module_path in module_paths
                }
                for candidate in candidates
            }

            are_optimized_callee_module_code_strings_zero_diff = {
                candidate.optimization_id: {
                    callee_module_path: normalize_code(optimized_code[candidate.optimization_id][callee_module_path])
                    == validated_original_code[callee_module_path].normalized_code
                    for callee_module_path in module_paths
                }
                for candidate in candidates
            }
            candidates_with_diffs = [
                candidate
                for candidate in candidates
                if not all(are_optimized_callee_module_code_strings_zero_diff[candidate.optimization_id].values())
            ]

            for candidate, worktree in zip(candidates_with_diffs, worktrees[1:]):
                for module_path in optimized_code[candidate.optimization_id]:
                    if are_optimized_callee_module_code_strings_zero_diff[candidate.optimization_id][module_path]:
                        (worktree / module_path.relative_to(git_root)).write_text(
                            optimized_code[candidate.optimization_id][module_path], encoding="utf8"
                        )

            function_to_optimize_original_worktree_fqn = (
                str(worktrees[0].name / function_to_optimize.file_path.relative_to(git_root).with_suffix("")).replace(
                    "/", "."
                )
                + "."
                + function_to_optimize_qualified_name
            )

            logger.info("Running concolic coverage checking for the original code…")
            original_code_coverage_tests = subprocess.run(
                [
                    "crosshair",
                    "cover",
                    "--example_output_format=pytest",
                    "--max_uninteresting_iterations=256",
                    function_to_optimize_original_worktree_fqn,
                ],
                capture_output=True,
                text=True,
                cwd=worktree_root,
                check=False,
            )
            logger.info(f"Tests generated through concolic coverage checking:\n{original_code_coverage_tests.stdout}")
            console.rule()

            diffbehavior_results: dict[str, DiffbehaviorReturnCode] = {}
            logger.info("Running concolic behavior correctness and coverage checking on optimized code…")
            console.rule()
            for candidate_index, candidate in enumerate(candidates_with_diffs, start=1):
                logger.info(f"Optimization candidate {candidate_index}/{len(candidates_with_diffs)}:")
                code_print(candidate.source_code)
                function_to_optimize_optimized_worktree_fqn = (
                    str(
                        worktrees[candidate_index].name
                        / function_to_optimize.file_path.relative_to(git_root).with_suffix("")
                    ).replace("/", ".")
                    + "."
                    + function_to_optimize_qualified_name
                )
                result = subprocess.run(
                    [
                        "crosshair",
                        "diffbehavior",
                        "--max_uninteresting_iterations=256",
                        function_to_optimize_optimized_worktree_fqn,
                        function_to_optimize_original_worktree_fqn,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=worktree_root,
                    check=False,
                )
                optimized_code_tests = subprocess.run(
                    [
                        "crosshair",
                        "cover",
                        "--example_output_format=pytest",
                        "--max_uninteresting_iterations=256",
                        function_to_optimize_optimized_worktree_fqn,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=worktree_root,
                    check=False,
                )

                if result.returncode == DiffbehaviorReturnCode.ERROR:
                    diffbehavior_results[candidate.optimization_id] = DiffbehaviorReturnCode.ERROR
                    logger.info("Inconclusive results from concolic behavior correctness checking.")
                    logger.warning(
                        f"Error running crosshair diffbehavior{': ' + result.stderr if result.stderr else '.'}"
                    )
                elif result.returncode == DiffbehaviorReturnCode.COUNTER_EXAMPLES:
                    split_counter_examples = re.split("(Given: )", result.stdout)[1:]
                    joined_counter_examples = [
                        "".join(map(str, split_counter_examples[i : i + 2]))
                        for i in range(0, len(split_counter_examples), 2)
                    ]
                    concrete_counter_examples = "".join(
                        [elt for elt in joined_counter_examples if not re.search(r" object at 0x[0-9a-fA-F]+", elt)]
                    )
                    if concrete_counter_examples:
                        diffbehavior_results[candidate.optimization_id] = DiffbehaviorReturnCode.COUNTER_EXAMPLES
                        logger.info(
                            f"Optimization candidate failed concolic behavior correctness "
                            f"checking:\n{concrete_counter_examples}"
                        )
                        if result.stdout != concrete_counter_examples:
                            objectid_counter_examples = "".join(
                                [elt for elt in joined_counter_examples if re.search(r" object at 0x[0-9a-fA-F]+", elt)]
                            )
                            logger.warning(f"Counter-examples with object ID found:\n{objectid_counter_examples}")
                    else:
                        diffbehavior_results[candidate.optimization_id] = DiffbehaviorReturnCode.ERROR
                        logger.info("Inconclusive results from concolic behavior correctness checking.")
                        console.rule()
                        logger.warning(f"Counter-examples with object ID found:\n{result.stdout}")
                elif result.returncode == DiffbehaviorReturnCode.NO_DIFFERENCES:
                    diffbehavior_results[candidate.optimization_id] = DiffbehaviorReturnCode.NO_DIFFERENCES
                    logger.info(
                        f"Optimization candidate passed concolic behavior correctness checking"
                        f"{': ' + chr(10) + result.stdout.split(chr(10), 1)[0] if chr(10) in result.stdout else '.'}"
                    )
                    if result.stdout.endswith("All paths exhausted, functions are likely the same!\n"):
                        logger.info("All paths exhausted, functions are likely the same!")
                    else:
                        logger.warning("Consider increasing the --max_uninteresting_iterations option.")
                else:
                    logger.info("Inconclusive results from concolic behavior correctness checking.")
                    logger.error("Unknown return code running crosshair diffbehavior.")
                console.rule()
                logger.info(f"Tests generated through concolic coverage checking:\n{optimized_code_tests.stdout}")
                console.rule()

            tests_in_file: list[FunctionCalledInTest] = function_to_tests.get(
                function_to_optimize.qualified_name_with_modules_from_root(self.args.project_root), []
            )

            best_optimization = self.determine_best_candidate(
                candidates=candidates_with_diffs,
                code_context=code_context,
                function_to_optimize=function_to_optimize,
                original_code=validated_original_code[function_to_optimize.file_path].source_code,
                original_code_baseline=original_code_baseline,
                original_helper_code=original_helper_code,
                function_trace_id=function_trace_id[:-4] + f"EXP{u}" if should_run_experiment else function_trace_id,
                only_run_this_test_function=tests_in_file,
                diffbehavior_results=diffbehavior_results,
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
                    winning_test_results=best_optimization.winning_test_results,
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
                    function_to_tests,
                    tests_root=self.test_cfg.tests_root,
                )

                original_code_combined = original_helper_code.copy()
                original_code_combined[explanation.file_path] = validated_original_code[
                    function_to_optimize.file_path
                ].source_code
                new_code_combined = new_helper_code.copy()
                new_code_combined[explanation.file_path] = new_code
                if not self.args.no_pr:
                    check_create_pr(
                        original_code=original_code_combined,
                        new_code=new_code_combined,
                        explanation=explanation,
                        existing_tests_source=existing_tests,
                        generated_original_test_source="\n".join(
                            [test.generated_original_test_source for test in generated_tests.generated_tests]
                        ),
                        function_trace_id=function_trace_id,
                    )
                    if self.args.all or env_utils.get_pr_number():
                        self.write_code_and_helpers(
                            validated_original_code[function_to_optimize.file_path].source_code,
                            original_helper_code,
                            function_to_optimize.file_path,
                        )
        for generated_test_path in generated_tests_paths:
            generated_test_path.unlink(missing_ok=True)
        for test_paths in instrumented_unittests_created_for_function:
            test_paths.unlink(missing_ok=True)
        if not best_optimization:
            return Failure(f"No best optimizations found for function {function_to_optimize.qualified_name}")
        logger.info("----------------")
        return Success(best_optimization)

    def determine_best_candidate(
        self,
        *,
        candidates: list[OptimizedCandidate],
        code_context: CodeOptimizationContext,
        function_to_optimize: FunctionToOptimize,
        original_code: str,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        function_trace_id: str,
        only_run_this_test_function: list[FunctionCalledInTest] | None = None,
        diffbehavior_results: dict[str, DiffbehaviorReturnCode],
    ) -> BestOptimization | None:
        best_optimization: BestOptimization | None = None
        best_runtime_until_now = original_code_baseline.runtime  # The fastest code runtime until now

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
                        continue
                except (ValueError, SyntaxError, cst.ParserSyntaxError, AttributeError) as e:
                    logger.error(e)
                    self.write_code_and_helpers(original_code, original_helper_code, function_to_optimize.file_path)
                    continue

                run_results = self.run_optimized_candidate(
                    optimization_candidate_index=candidate_index,
                    original_test_results=original_code_baseline.overall_test_results,
                    tests_in_file=only_run_this_test_function,
                    diffbehavior_result=diffbehavior_results[candidate.optimization_id],
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
                    if speedup_critic(
                        candidate_result, original_code_baseline.runtime, best_runtime_until_now
                    ) and quantity_of_tests_critic(candidate_result):
                        tree.add("This candidate is faster than the previous best candidate. 🚀")
                        tree.add(f"Original runtime: {humanize_runtime(original_code_baseline.runtime)}")
                        tree.add(
                            f"Best test runtime: {humanize_runtime(candidate_result.best_test_runtime)} "
                            f"(measured over {candidate_result.max_loop_count} "
                            f"loop{'s' if candidate_result.max_loop_count > 1 else ''})"
                        )
                        tree.add(f"Speedup ratio: {perf_gain:.3f}")

                        best_optimization = BestOptimization(
                            candidate=candidate,
                            helper_functions=code_context.helper_functions,
                            runtime=best_test_runtime,
                            winning_test_results=candidate_result.test_results,
                        )
                        best_runtime_until_now = best_test_runtime
                    else:
                        tree.add(
                            f"Runtime: {humanize_runtime(best_test_runtime)} "
                            f"(measured over {candidate_result.max_loop_count} "
                            f"loop{'s' if candidate_result.max_loop_count > 1 else ''})"
                        )
                        tree.add(f"Speedup ratio: {perf_gain:.3f}")
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

    @staticmethod
    def log_successful_optimization(
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
                    for tt, v in explanation.winning_test_results.get_test_pass_fail_report_by_type().items()
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
        if should_sort_imports and new_code is not None:
            new_code = sort_imports(new_code)

        new_helper_code: dict[Path, str] = {}
        helper_functions_paths = {hf.file_path for hf in helper_functions}
        for module_abspath in helper_functions_paths:
            formatted_helper_code = format_code(self.args.formatter_cmds, module_abspath)
            if should_sort_imports and formatted_helper_code is not None:
                formatted_helper_code = sort_imports(formatted_helper_code)
            if formatted_helper_code is not None:
                new_helper_code[module_abspath] = formatted_helper_code

        return new_code or "", new_helper_code

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        function_to_optimize_file_path: Path,
        optimized_code: str,
        qualified_function_name: str,
    ) -> bool:
        did_update = replace_function_definitions_in_module(
            function_names=[qualified_function_name],
            optimized_code=optimized_code,
            file_path_of_module_with_function_to_optimize=function_to_optimize_file_path,
            module_abspath=function_to_optimize_file_path,
            preexisting_objects=code_context.preexisting_objects,
            contextual_functions=code_context.contextual_dunder_methods,
            project_root_path=self.args.project_root,
        )
        helper_functions_by_module_abspath = defaultdict(set)
        for helper_function in code_context.helper_functions:
            if helper_function.jedi_definition.type != "class":
                helper_functions_by_module_abspath[helper_function.file_path].add(helper_function.qualified_name)
        for module_abspath, qualified_names in helper_functions_by_module_abspath.items():
            did_update |= replace_function_definitions_in_module(
                function_names=list(qualified_names),
                optimized_code=optimized_code,
                file_path_of_module_with_function_to_optimize=function_to_optimize_file_path,
                module_abspath=module_abspath,
                preexisting_objects=[],
                contextual_functions=code_context.contextual_dunder_methods,
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
        preexisting_objects = find_preexisting_objects(code_to_optimize_with_helpers)
        contextual_dunder_methods.update(helper_dunder_methods)
        return Success(
            CodeOptimizationContext(
                code_to_optimize_with_helpers=code_to_optimize_with_helpers_and_imports,
                contextual_dunder_methods=contextual_dunder_methods,
                helper_functions=helper_functions,
                preexisting_objects=preexisting_objects,
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
        unique_instrumented_test_files = set()

        func_qualname = function_to_optimize.qualified_name_with_modules_from_root(self.args.project_root)
        if func_qualname not in function_to_tests:
            logger.info(f"Did not find any pre-existing tests for '{func_qualname}', will only use generated tests.")
        else:
            test_file_invocation_positions = defaultdict(list)
            for tests_in_file in function_to_tests.get(func_qualname):
                test_file_invocation_positions[
                    (tests_in_file.tests_in_file.test_file, tests_in_file.tests_in_file.test_type)
                ].append(tests_in_file.position)
            for (test_file, test_type), positions in test_file_invocation_positions.items():
                path_obj_test_file = Path(test_file)
                if test_type == TestType.EXISTING_UNIT_TEST:
                    existing_test_files_count += 1
                elif test_type == TestType.REPLAY_TEST:
                    replay_test_files_count += 1
                else:
                    msg = f"Unexpected test type: {test_type}"
                    raise ValueError(msg)
                success, injected_test = inject_profiling_into_existing_test(
                    test_path=path_obj_test_file,
                    call_positions=positions,
                    function_to_optimize=function_to_optimize,
                    tests_project_root=self.test_cfg.tests_project_rootdir,
                    test_framework=self.args.test_framework,
                )
                if not success:
                    continue

                new_test_path = Path(
                    f"{os.path.splitext(test_file)[0]}__perfinstrumented{os.path.splitext(test_file)[1]}"
                )
                if injected_test is not None:
                    with new_test_path.open("w", encoding="utf8") as _f:
                        _f.write(injected_test)
                else:
                    msg = "injected_test is None"
                    raise ValueError(msg)

                unique_instrumented_test_files.add(new_test_path)
                if not self.test_files.get_by_original_file_path(path_obj_test_file):
                    self.test_files.add(
                        TestFile(
                            instrumented_file_path=new_test_path,
                            original_source=None,
                            original_file_path=Path(test_file),
                            test_type=test_type,
                        )
                    )
            logger.info(
                f"Discovered {existing_test_files_count} existing unit test file"
                f"{'s' if existing_test_files_count != 1 else ''} and {replay_test_files_count} replay test file"
                f"{'s' if replay_test_files_count != 1 else ''} for {func_qualname}"
            )
        return unique_instrumented_test_files

    def generate_tests_and_optimizations(
        self,
        code_to_optimize_with_helpers: str,
        function_to_optimize: FunctionToOptimize,
        helper_functions: list[FunctionSource],
        module_path: Path,
        function_trace_id: str,
        run_experiment: bool = False,
    ) -> Result[tuple[GeneratedTestsList, OptimizationSet], str]:
        max_workers = N_TESTS_TO_GENERATE + 1 if not run_experiment else N_TESTS_TO_GENERATE + 2
        console.rule()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit the test generation task as future
            future_tests = self.generate_and_instrument_tests(
                executor,
                code_to_optimize_with_helpers,
                function_to_optimize,
                [definition.fully_qualified_name for definition in helper_functions],
                module_path,
                (function_trace_id[:-4] + "EXP0" if run_experiment else function_trace_id),
            )
            future_optimization_candidates = executor.submit(
                self.aiservice_client.optimize_python_code,
                code_to_optimize_with_helpers,
                function_trace_id[:-4] + "EXP0" if run_experiment else function_trace_id,
                N_CANDIDATES,
                ExperimentMetadata(id=self.experiment_id, group="control") if run_experiment else None,
            )
            future_candidates_exp = None
            futures: list[concurrent.futures.Future] = [*future_tests, future_optimization_candidates]
            if run_experiment:
                future_candidates_exp = executor.submit(
                    self.local_aiservice_client.optimize_python_code,
                    code_to_optimize_with_helpers,
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
                    generated_test_source, instrumented_test_source = res
                    tests.append(
                        GeneratedTests(
                            generated_original_test_source=generated_test_source,
                            instrumented_test_source=instrumented_test_source,
                        )
                    )
            if not tests:
                logger.warning(f"Failed to generate and instrument tests for {function_to_optimize.function_name}")
                return Failure(f"/!\\ NO TESTS GENERATED for {function_to_optimize.function_name}")
            logger.info(f"Generated {len(tests)} tests for {function_to_optimize.function_name}")
            console.rule()
            generated_tests = GeneratedTestsList(generated_tests=tests)

        return Success((generated_tests, OptimizationSet(control=candidates, experiment=candidates_experiment)))

    def establish_original_code_baseline(
        self, function_name: str, tests_in_file: list[FunctionCalledInTest]
    ) -> Result[tuple[OriginalCodeBaseline, list[str]], str]:
        # For the original function - run the tests and get the runtime

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

            only_run_these_test_functions_for_test_files: dict[str, str] = {}

            # Replay tests can have hundreds of test functions and running them can be very slow,
            # so we only run the test functions that are relevant to the function we are optimizing
            for test_file in self.test_files.get_by_type(TestType.REPLAY_TEST).test_files:
                relevant_tests_in_file = [
                    test_in_file
                    for test_in_file in tests_in_file
                    if test_in_file.tests_in_file.test_file == test_file.original_file_path
                ]
                only_run_these_test_functions_for_test_files[test_file.instrumented_file_path] = relevant_tests_in_file[
                    0
                ].tests_in_file.test_function

                if len(relevant_tests_in_file) > 1:
                    logger.warning(
                        f"Multiple tests found ub the replay test {test_file} for {function_name}. Should not happen"
                    )

            if test_framework == "pytest":
                unittest_results = self.run_and_parse_tests(
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=0,
                    test_functions=only_run_these_test_functions_for_test_files,
                    testing_time=TOTAL_LOOPING_TIME,
                )
            else:
                unittest_results = TestResults()
                start_time: float = time.time()
                for i in range(100):
                    if i >= 5 and time.time() - start_time >= TOTAL_LOOPING_TIME:
                        break
                    test_env["CODEFLASH_LOOP_INDEX"] = str(i + 1)
                    unittest_loop_results = self.run_and_parse_tests(
                        test_env=test_env,
                        test_files=self.test_files,
                        optimization_iteration=0,
                        test_functions=only_run_these_test_functions_for_test_files,
                        testing_time=TOTAL_LOOPING_TIME,
                    )
                    unittest_results.merge(unittest_loop_results)

            initial_loop_unittest_results = TestResults(
                test_results=[result for result in unittest_results.test_results if result.loop_index == 1]
            )

            console.print(
                TestResults.report_to_tree(
                    initial_loop_unittest_results.get_test_pass_fail_report_by_type(),
                    title="Overall initial loop test results for original code",
                )
            )
            console.rule()

            existing_test_results = TestResults(
                test_results=[result for result in unittest_results if result.test_type == TestType.EXISTING_UNIT_TEST]
            )
            generated_test_results = TestResults(
                test_results=[
                    result for result in unittest_results if result.test_type == TestType.GENERATED_REGRESSION
                ]
            )

            total_timing = unittest_results.total_passed_runtime()

            functions_to_remove = [
                result.id.test_function_name for result in generated_test_results.test_results if not result.did_pass
            ]

            if not initial_loop_unittest_results:
                logger.warning(
                    f"Couldn't run any tests for original function {function_name}. SKIPPING OPTIMIZING THIS FUNCTION."
                )
                console.rule()
                success = False
            if total_timing == 0:
                logger.warning("The overall test runtime of the original function is 0, couldn't run tests.")
                console.rule()
                success = False
            if not total_timing:
                logger.warning("Failed to run the tests for the original function, skipping optimization")
                console.rule()
                success = False
            if not success:
                return Failure("Failed to establish a baseline for the original code.")

            loop_count = max([int(result.loop_index) for result in unittest_results.test_results])
            logger.info(
                f"Original code runtime measured over {loop_count} loop{'s' if loop_count > 1 else ''}: "
                f"{humanize_runtime(total_timing)} per full loop"
            )
            console.rule()
            logger.debug(f"Total original code runtime (ns): {total_timing}")
            return Success(
                (
                    OriginalCodeBaseline(
                        generated_test_results=generated_test_results,
                        existing_test_results=existing_test_results,
                        overall_test_results=unittest_results,
                        runtime=total_timing,
                    ),
                    functions_to_remove,
                )
            )

    def run_optimized_candidate(
        self,
        *,
        optimization_candidate_index: int,
        original_test_results: TestResults | None,
        tests_in_file: list[FunctionCalledInTest] | None,
        diffbehavior_result: DiffbehaviorReturnCode,
    ) -> Result[OptimizedCandidateResult, str]:
        assert (test_framework := self.args.test_framework) in ["pytest", "unittest"]

        with progress_bar("Testing optimization candidate"):
            success = True

            test_env = os.environ.copy()
            test_env["CODEFLASH_TEST_ITERATION"] = str(optimization_candidate_index)
            test_env["CODEFLASH_TRACER_DISABLE"] = "1"
            if "PYTHONPATH" not in test_env:
                test_env["PYTHONPATH"] = str(self.args.project_root)
            else:
                test_env["PYTHONPATH"] += os.pathsep + str(self.args.project_root)

            get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite")).unlink(missing_ok=True)
            get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite")).unlink(missing_ok=True)

            only_run_these_test_functions_for_test_files: dict[str, str] = {}
            # Replay tests can have hundreds of test functions and running them can be very slow,
            # so we only run the test functions that are relevant to the function we are optimizing
            for test_file in self.test_files.get_by_type(TestType.REPLAY_TEST).test_files:
                relevant_tests_in_file = [
                    test_in_file
                    for test_in_file in tests_in_file
                    if test_in_file.tests_in_file.test_file == test_file.original_file_path
                ]
                only_run_these_test_functions_for_test_files[test_file.instrumented_file_path] = relevant_tests_in_file[
                    0
                ].tests_in_file.test_function

            if test_framework == "pytest":
                candidate_results = self.run_and_parse_tests(
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=optimization_candidate_index,
                    test_functions=only_run_these_test_functions_for_test_files,
                    testing_time=TOTAL_LOOPING_TIME,
                )
                loop_count = (
                    max(all_loop_indices)
                    if (all_loop_indices := {result.loop_index for result in candidate_results.test_results})
                    else 0
                )
            else:
                candidate_results = TestResults()
                start_time: float = time.time()
                loop_count = 0
                for i in range(100):
                    if i >= 5 and time.time() - start_time >= TOTAL_LOOPING_TIME:
                        break
                    test_env["CODEFLASH_LOOP_INDEX"] = str(i + 1)
                    candidate_loop_results = self.run_and_parse_tests(
                        test_env=test_env,
                        test_files=self.test_files,
                        optimization_iteration=optimization_candidate_index,
                        test_functions=only_run_these_test_functions_for_test_files,
                        testing_time=TOTAL_LOOPING_TIME,
                    )
                    loop_count = i + 1
                    candidate_results.merge(candidate_loop_results)

            initial_loop_candidate_results = TestResults(
                test_results=[result for result in candidate_results.test_results if result.loop_index == 1]
            )

            console.print(
                TestResults.report_to_tree(
                    initial_loop_candidate_results.get_test_pass_fail_report_by_type(),
                    title="Overall initial loop test results for candidate",
                )
            )
            console.rule()

        # TestType.CONCOLIC_TESTING: "🔍 Dynamic Symbolic Execution",
        # 🔍 Dynamic Symbolic Execution - No differences found. (attempted 154 iterations)
        # console.rule
        initial_loop_original_test_results = TestResults(
            test_results=[result for result in original_test_results.test_results if result.loop_index == 1]
        )

        if compare_test_results(initial_loop_original_test_results, initial_loop_candidate_results):
            logger.info("Test results matched!")
            equal_results = True
        else:
            logger.info("Test results did not match the test results of the original code.")
            success = False
            equal_results = False
        console.rule()

        if diffbehavior_result == DiffbehaviorReturnCode.NO_DIFFERENCES:
            logger.info("Concolic behavior correctness check successful!")
            console.rule()
            if equal_results:
                logger.info("True negative: Concolic behavior correctness check successful and test results matched.")
            else:
                logger.warning(
                    "False negative for concolic testing: Concolic behavior correctness check successful but test "
                    "results did not match."
                )
            console.rule()
        elif diffbehavior_result == DiffbehaviorReturnCode.COUNTER_EXAMPLES:
            logger.warning("Concolic behavior correctness check failed.")
            console.rule()
            if equal_results:
                logger.warning(
                    "False negative for regression testing: Concolic behavior correctness check failed but test "
                    "results matched."
                )
                success = False
                equal_results = False
            else:
                logger.info("True positive: Concolic behavior correctness check failed and test results did not match.")
            console.rule()
        else:
            logger.warning("Concolic behavior correctness check inconclusive.")
            console.rule()

        if (total_candidate_timing := candidate_results.total_passed_runtime()) == 0:
            logger.warning("The overall test runtime of the optimized function is 0, couldn't run tests.")
            console.rule()
        get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.bin")).unlink(missing_ok=True)

        get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite")).unlink(missing_ok=True)
        if not equal_results:
            success = False

        if not success:
            return Failure("Failed to run the optimization candidate.")
        logger.debug(f"Total optimized code {optimization_candidate_index} runtime (ns): {total_candidate_timing}")
        return Success(
            OptimizedCandidateResult(
                max_loop_count=loop_count,
                best_test_runtime=total_candidate_timing,
                test_results=candidate_results,
                optimization_candidate_index=optimization_candidate_index,
                total_candidate_timing=total_candidate_timing,
            )
        )

    def run_and_parse_tests(
        self,
        test_env: dict[str, str],
        test_files: TestFiles,
        optimization_iteration: int,
        test_functions: list[str | None] | None = None,
        testing_time: float = TOTAL_LOOPING_TIME,
        pytest_min_loops: int = 5,
        pytest_max_loops: int = 100_000,
    ) -> TestResults:
        try:
            result_file_path, run_result = run_tests(
                test_files,
                test_framework=self.args.test_framework,
                cwd=self.args.project_root,
                test_env=test_env,
                pytest_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                pytest_cmd=self.test_cfg.pytest_cmd,
                verbose=True,
                only_run_these_test_functions=test_functions,
                pytest_target_runtime_seconds=testing_time,
                pytest_min_loops=pytest_min_loops,
                pytest_max_loops=pytest_max_loops,
            )
        except subprocess.TimeoutExpired:
            logger.exception(
                f'Error running tests in {", ".join(str(f) for f in test_files.test_files)}.\nTimeout Error'
            )
            return TestResults()
        if run_result.returncode != 0:
            logger.debug(
                f'Nonzero return code {run_result.returncode} when running tests in '
                f'{", ".join([str(f.instrumented_file_path) for f in test_files.test_files])}.\n'
                f"stdout: {run_result.stdout}\n"
                f"stderr: {run_result.stderr}\n"
            )
        return parse_test_results(
            test_xml_path=result_file_path,
            test_files=test_files,
            test_config=self.test_cfg,
            optimization_iteration=optimization_iteration,
            run_result=run_result,
        )

    def generate_and_instrument_tests(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        source_code_being_tested: str,
        function_to_optimize: FunctionToOptimize,
        helper_function_names: list[str],
        module_path: Path,
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
                self.args.use_cached_tests,
                function_trace_id,
                test_index,
            )
            for test_index in range(N_TESTS_TO_GENERATE)
        ]


def run_with_args(args: Namespace) -> None:
    optimizer = Optimizer(args)
    optimizer.run()
