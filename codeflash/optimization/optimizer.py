import concurrent.futures
import logging
import os
import pathlib
import uuid
from argparse import Namespace
from collections import defaultdict
from typing import IO, Dict, List, Optional, Tuple, Union

import libcst as cst
from pydantic import BaseModel
from returns.pipeline import is_successful
from returns.result import Failure, Result, Success

from codeflash.api.aiservice import Optimization, log_results, optimize_python_code
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_extractor import extract_code
from codeflash.code_utils.code_replacer import replace_function_definitions_in_module
from codeflash.code_utils.code_utils import (
    get_all_function_names,
    get_run_tmp_file,
    module_name_from_file_path,
)
from codeflash.code_utils.config_consts import (
    INDIVIDUAL_TEST_TIMEOUT,
    MAX_CUMULATIVE_TEST_RUNTIME_NANOSECONDS,
    MAX_FUNCTION_TEST_SECONDS,
    MAX_TEST_FUNCTION_RUNS,
    MAX_TEST_RUN_ITERATIONS,
    N_CANDIDATES,
)
from codeflash.code_utils.formatter import format_code, sort_imports
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.discovery.discover_unit_tests import TestsInFile, discover_unit_tests
from codeflash.discovery.functions_to_optimize import (
    FunctionParent,
    FunctionToOptimize,
    get_functions_to_optimize_by_file,
)
from codeflash.optimization.function_context import (
    Source,
    get_constrained_function_context_and_dependent_functions,
)
from codeflash.result.create_pr import check_create_pr
from codeflash.result.explanation import Explanation
from codeflash.telemetry import posthog
from codeflash.telemetry.posthog import ph
from codeflash.telemetry.sentry import init_sentry
from codeflash.verification.equivalence import compare_results
from codeflash.verification.parse_test_output import parse_test_results
from codeflash.verification.test_results import TestResults, TestType
from codeflash.verification.test_runner import run_tests
from codeflash.verification.verification_utils import TestConfig, get_test_file_path
from codeflash.verification.verifier import generate_tests


class OptimizedCandidateResult(BaseModel):
    times_run: int
    best_test_runtime: int
    best_test_results: TestResults


class OriginalCodeBaseline(BaseModel):
    generated_test_results: TestResults
    overall_test_results: Optional[TestResults]
    runtime: int


class TestsAndOptimizationsResult(BaseModel):
    generated_original_test_source: str
    instrumented_test_source: str
    optimizations: List[Optimization]


class BestOptimization(BaseModel):
    source_code: str
    explanation: str
    dependent_functions: list[Tuple[Source, str, str]]


class CodeOptimizationContext(BaseModel):
    code_to_optimize_with_dependents: str
    contextual_dunder_methods: set[tuple[str, str]]
    dependent_functions: List[Tuple[Source, str, str]]
    preexisting_functions: List[str]


class Optimizer:
    def __init__(self, args: Namespace):
        self.args = args
        init_sentry(not args.disable_telemetry)
        posthog.initialize_posthog(not args.disable_telemetry)

        self.test_cfg = TestConfig(
            tests_root=args.tests_root,
            project_root_path=args.project_root,
            test_framework=args.test_framework,
            pytest_cmd=args.pytest_cmd,
        )

    def run(self) -> None:
        ph("cli-optimize-run-start")
        logging.info("Running optimizer.")
        if not env_utils.ensure_codeflash_api_key():
            return
        if not env_utils.ensure_git_repo(module_root=self.args.module_root):
            logging.error("No git repository detected and user aborted run. Exiting...")
            exit(1)

        file_to_funcs_to_optimize: dict[str, list[FunctionToOptimize]]
        num_optimizable_functions: int
        (
            file_to_funcs_to_optimize,
            num_optimizable_functions,
        ) = get_functions_to_optimize_by_file(
            optimize_all=self.args.all,
            file=self.args.file,
            function=self.args.function,
            test_cfg=self.test_cfg,
            ignore_paths=self.args.ignore_paths,
            project_root=self.args.project_root,
            module_root=self.args.module_root,
        )

        test_files_created: set[str] = set()
        instrumented_unittests_created: set[str] = set()
        optimizations_found: int = 0

        function_iterator_count: int = 0
        try:
            ph("cli-optimize-functions-to-optimize", {"num_functions": num_optimizable_functions})
            if num_optimizable_functions == 0:
                logging.info("No functions found to optimize. Exiting...")
                return
            logging.info(f"Discovering existing unit tests in {self.test_cfg.tests_root} ...")
            function_to_tests: dict[str, list[TestsInFile]] = discover_unit_tests(self.test_cfg)
            num_discovered_tests: int = sum([len(value) for value in function_to_tests.values()])
            logging.info(
                f"Discovered {num_discovered_tests} existing unit tests in {self.test_cfg.tests_root}",
            )
            ph("cli-optimize-discovered-tests", {"num_tests": num_discovered_tests})
            path: str
            for path in file_to_funcs_to_optimize:
                logging.info(f"Examining file {path} ...")
                # TODO: Sequence the functions one goes through intelligently. If we are optimizing f(g(x)),
                #  then we might want to first optimize f rather than g because optimizing f would already
                #  optimize g as it is a dependency
                f: IO[str]
                with open(path, encoding="utf8") as f:
                    original_code: str = f.read()
                should_sort_imports = True
                if sort_imports(self.args.imports_sort_cmd, should_sort_imports, path) != original_code:
                    should_sort_imports = False

                function_to_optimize: FunctionToOptimize
                for function_to_optimize in file_to_funcs_to_optimize[path]:
                    function_trace_id: str = str(uuid.uuid4())
                    ph("cli-optimize-function-start", {"function_trace_id": function_trace_id})
                    qualified_function_name: str = function_to_optimize.qualified_name
                    function_iterator_count += 1
                    logging.info(
                        f"Optimizing function {function_iterator_count} of {num_optimizable_functions} - {qualified_function_name}",
                    )
                    winning_test_results = None
                    self.cleanup_leftover_test_return_values()
                    ctx_result = self.get_code_optimization_context(function_to_optimize)
                    if not is_successful(ctx_result):
                        logging.error(ctx_result.failure())
                        continue
                    code_context = ctx_result.unwrap()
                    dependent_functions_by_module_abspath = defaultdict(set)
                    for _, module_abspath, qualified_name in code_context.dependent_functions:
                        dependent_functions_by_module_abspath[module_abspath].add(qualified_name)
                    original_dependent_code = {}
                    for module_abspath in dependent_functions_by_module_abspath.keys():
                        with open(module_abspath, encoding="utf8") as f:
                            dependent_code = f.read()
                            original_dependent_code[module_abspath] = dependent_code
                    logging.info(f"Code to be optimized:\n{code_context.code_to_optimize_with_dependents}")
                    module_path = module_name_from_file_path(path, self.args.project_root)

                    instrumented_unittests_created_for_function = self.instrument_existing_tests(
                        function_name=qualified_function_name,
                        module_path=module_path,
                        function_to_tests=function_to_tests,
                    )
                    instrumented_unittests_created.update(
                        instrumented_unittests_created_for_function,
                    )

                    tests_result = self.generate_tests_and_optimizations(
                        code_context.code_to_optimize_with_dependents,
                        function_to_optimize,
                        code_context.dependent_functions,
                        module_path,
                        function_trace_id,
                    )
                    if not is_successful(tests_result):
                        logging.error(tests_result.failure())
                        continue
                    tests_and_optimizations: TestsAndOptimizationsResult = tests_result.unwrap()

                    generated_tests_path = get_test_file_path(
                        self.args.tests_root,
                        function_to_optimize.function_name,
                        0,
                    )
                    with open(generated_tests_path, "w", encoding="utf8") as file:
                        file.write(tests_and_optimizations.instrumented_test_source)

                    test_files_created.add(generated_tests_path)
                    baseline_result = self.establish_original_code_baseline(
                        qualified_function_name,
                        instrumented_unittests_created_for_function,
                        generated_tests_path,
                    )
                    if not is_successful(baseline_result):
                        logging.error(baseline_result.failure())
                        continue
                    original_code_baseline: OriginalCodeBaseline = baseline_result.unwrap()
                    best_runtime = original_code_baseline.runtime  # The fastest code runtime until now
                    logging.info("Optimizing code ...")
                    # TODO: Postprocess the optimized function to include the original docstring and such

                    best_optimization: Optional[BestOptimization] = None
                    speedup_ratios: Dict[str, float | None] = dict()
                    optimized_runtimes = dict()
                    is_correct = dict()

                    for i, optimization in enumerate(tests_and_optimizations.optimizations):
                        j = i + 1
                        if optimization.source_code is None:
                            continue
                        # remove left overs from previous run
                        pathlib.Path(get_run_tmp_file(f"test_return_values_{j}.bin")).unlink(
                            missing_ok=True,
                        )
                        pathlib.Path(get_run_tmp_file(f"test_return_values_{j}.sqlite")).unlink(
                            missing_ok=True,
                        )
                        logging.info("Optimized candidate:")
                        logging.info(optimization.source_code)
                        try:
                            replace_function_definitions_in_module(
                                [qualified_function_name],
                                optimization.source_code,
                                path,
                                code_context.preexisting_functions,
                                code_context.contextual_dunder_methods,
                            )
                            for (
                                module_abspath,
                                qualified_names,
                            ) in dependent_functions_by_module_abspath.items():
                                replace_function_definitions_in_module(
                                    list(qualified_names),
                                    optimization.source_code,
                                    module_abspath,
                                    [],
                                    code_context.contextual_dunder_methods,
                                )
                        except (
                            ValueError,
                            SyntaxError,
                            cst.ParserSyntaxError,
                            AttributeError,
                        ) as e:
                            logging.exception(e)
                            with open(path, "w", encoding="utf8") as f:
                                f.write(original_code)
                            for module_abspath in dependent_functions_by_module_abspath.keys():
                                with open(module_abspath, "w", encoding="utf8") as f:
                                    f.write(original_dependent_code[module_abspath])
                            continue

                        run_results = self.run_optimized_candidate(
                            optimization_index=j,
                            instrumented_unittests_created_for_function=instrumented_unittests_created_for_function,
                            overall_original_test_results=original_code_baseline.overall_test_results,
                            original_gen_results=original_code_baseline.generated_test_results,
                            generated_tests_path=generated_tests_path,
                            best_runtime_until_now=best_runtime,
                        )
                        if not is_successful(run_results):
                            optimized_runtimes[optimization.optimization_id] = None
                            is_correct[optimization.optimization_id] = False
                            speedup_ratios[optimization.optimization_id] = None
                        else:
                            candidate_result: OptimizedCandidateResult = run_results.unwrap()
                            best_test_runtime = candidate_result.best_test_runtime
                            optimized_runtimes[optimization.optimization_id] = best_test_runtime
                            is_correct[optimization.optimization_id] = True
                            speedup_ratios[optimization.optimization_id] = (
                                original_code_baseline.runtime - best_test_runtime
                            ) / best_test_runtime

                            logging.info(
                                f"Candidate runtime measured over {candidate_result.times_run} run{'s' if candidate_result.times_run > 1 else ''}: "
                                f"{humanize_runtime(best_test_runtime)}, speedup ratio = "
                                f"{((original_code_baseline.runtime - best_test_runtime) / best_test_runtime):.3f}",
                            )
                            if (
                                ((original_code_baseline.runtime - best_test_runtime) / best_test_runtime)
                                > self.args.minimum_performance_gain
                            ) and best_test_runtime < best_runtime:
                                logging.info(
                                    "This candidate is better than the previous best candidate.",
                                )

                                logging.info(
                                    f"Original runtime: {humanize_runtime(original_code_baseline.runtime)} Best test runtime: "
                                    f"{humanize_runtime(best_test_runtime)}, ratio = "
                                    f"{((original_code_baseline.runtime - best_test_runtime) / best_test_runtime)}",
                                )
                                best_optimization = BestOptimization(
                                    source_code=optimization.source_code,
                                    explanation=optimization.explanation,
                                    dependent_functions=code_context.dependent_functions,
                                )
                                best_runtime = best_test_runtime
                                winning_test_results = candidate_result.best_test_results
                        with open(path, "w", encoding="utf8") as f:
                            f.write(original_code)
                        for module_abspath in dependent_functions_by_module_abspath.keys():
                            with open(module_abspath, "w", encoding="utf8") as f:
                                f.write(original_dependent_code[module_abspath])
                        logging.info("----------------")
                    log_results(
                        function_trace_id=function_trace_id,
                        speedup_ratio=speedup_ratios,
                        original_runtime=original_code_baseline.runtime,
                        optimized_runtime=optimized_runtimes,
                        is_correct=is_correct,
                    )
                    ph("cli-optimize-function-finished", {"function_trace_id": function_trace_id})

                    if best_optimization:
                        optimizations_found += 1
                        logging.info(
                            f"Best candidate:\n{best_optimization.source_code}, {best_optimization.explanation}",
                        )

                        optimized_code = best_optimization.source_code
                        replace_function_definitions_in_module(
                            [qualified_function_name],
                            optimized_code,
                            path,
                            code_context.preexisting_functions,
                            code_context.contextual_dunder_methods,
                        )
                        for (
                            module_abspath,
                            qualified_names,
                        ) in dependent_functions_by_module_abspath.items():
                            replace_function_definitions_in_module(
                                list(qualified_names),
                                optimized_code,
                                module_abspath,
                                [],
                                code_context.contextual_dunder_methods,
                            )
                        explanation_final = Explanation(
                            raw_explanation_message=best_optimization.explanation,
                            winning_test_results=winning_test_results,
                            original_runtime_ns=original_code_baseline.runtime,
                            best_runtime_ns=best_runtime,
                            function_name=qualified_function_name,
                            path=path,
                        )
                        logging.info(f"Explanation: \n{explanation_final.to_console_string()}")

                        new_code = format_code(
                            self.args.formatter_cmd,
                            self.args.imports_sort_cmd,
                            should_sort_imports,
                            path,
                        )
                        new_dependent_code: dict[str, str] = {
                            module_abspath: format_code(
                                self.args.formatter_cmd,
                                self.args.imports_sort_cmd,
                                should_sort_imports,
                                module_abspath,
                            )
                            for module_abspath in dependent_functions_by_module_abspath.keys()
                        }
                        logging.info(
                            f"Optimization was validated for correctness by running the following tests - "
                            f"\n{tests_and_optimizations.generated_original_test_source}",
                        )

                        logging.info(f"⚡️ Optimization successful! 📄 {qualified_function_name} in {path}")
                        logging.info(f"📈 {explanation_final.perf_improvement_line}")

                        ph(
                            "cli-optimize-success",
                            {
                                "function_trace_id": function_trace_id,
                                "speedup_x": explanation_final.speedup_x,
                                "speedup_pct": explanation_final.speedup_pct,
                                "best_runtime": explanation_final.best_runtime_ns,
                                "original_runtime": explanation_final.original_runtime_ns,
                                "winning_test_results": {
                                    tt.to_name(): v
                                    for tt, v in explanation_final.winning_test_results.get_test_pass_fail_report_by_type().items()
                                },
                            },
                        )
                        test_files = function_to_tests.get(module_path + "." + qualified_function_name)
                        existing_tests = ""
                        if test_files:
                            for test_file in test_files:
                                with open(test_file.test_file, encoding="utf8") as f:
                                    new_test = "".join(f.readlines())
                                    if new_test not in existing_tests:
                                        existing_tests += new_test

                        check_create_pr(
                            optimize_all=self.args.all,
                            path=path,
                            original_code=original_dependent_code | {path: original_code},
                            new_code=new_dependent_code | {path: new_code},
                            explanation=explanation_final,
                            existing_tests_source=existing_tests,
                            generated_original_test_source=tests_and_optimizations.generated_original_test_source,
                        )
                        if self.args.all or env_utils.get_pr_number():
                            # Reverting to original code, because optimizing functions in a sequence can lead to
                            #  a) Error propagation, where error in one function can cause the next optimization to fail
                            #  b) Performance estimates become unstable, as the runtime of an optimization might be
                            #     dependent on the runtime of the previous optimization
                            with open(path, "w", encoding="utf8") as f:
                                f.write(original_code)
                            for module_abspath in dependent_functions_by_module_abspath.keys():
                                with open(module_abspath, "w", encoding="utf8") as f:
                                    f.write(original_dependent_code[module_abspath])
                    # Delete all the generated tests to not cause any clutter.
                    pathlib.Path(generated_tests_path).unlink(missing_ok=True)
                    for test_paths in instrumented_unittests_created_for_function:
                        pathlib.Path(test_paths).unlink(missing_ok=True)
            ph("cli-optimize-run-finished", {"optimizations_found": optimizations_found})
            if optimizations_found == 0:
                logging.info("❌ No optimizations found.")
            elif self.args.all:
                logging.info("✨ All functions have been optimized! ✨")
        finally:
            # TODO: Also revert the file/function being optimized if the process did not succeed
            for test_file in instrumented_unittests_created:
                pathlib.Path(test_file).unlink(missing_ok=True)
            for test_file in test_files_created:
                pathlib.Path(test_file).unlink(missing_ok=True)
            if hasattr(get_run_tmp_file, "tmpdir"):
                get_run_tmp_file.tmpdir.cleanup()

    def get_code_optimization_context(
        self,
        function_to_optimize: FunctionToOptimize,
    ) -> Result[CodeOptimizationContext, str]:
        code_to_optimize, contextual_dunder_methods = extract_code(
            [function_to_optimize],
        )
        if code_to_optimize is None:
            return Failure("Could not find function to optimize.")
        success, preexisting_functions = get_all_function_names(code_to_optimize)
        if not success:
            return Failure("Error in parsing the code, skipping optimization.")
        dependent_code, dependent_functions = get_constrained_function_context_and_dependent_functions(
            function_to_optimize,
            self.args.project_root,
            code_to_optimize,
        )
        if function_to_optimize.parents:
            function_class = function_to_optimize.parents[0].name
            dependent_methods = [
                df
                for df in dependent_functions
                if df[2].count(".") > 0 and df[2].split(".")[0] == function_class
            ]
            optimizable_methods = [function_to_optimize] + [
                FunctionToOptimize(
                    df[2].split(".")[0],
                    "",
                    [FunctionParent(df[2].split(".")[0], "ClassDef")],
                    None,
                    None,
                )
                for df in dependent_methods
            ]
            if len(optimizable_methods) > 1:
                code_to_optimize, contextual_dunder_methods = extract_code(
                    optimizable_methods,
                )
                if code_to_optimize is None:
                    return Failure("Could not find function to optimize.")
        code_to_optimize_with_dependents = dependent_code + "\n" + code_to_optimize
        preexisting_functions.extend(
            [fn[0].full_name.split(".")[-1] for fn in dependent_functions],
        )
        return Success(
            CodeOptimizationContext(
                code_to_optimize_with_dependents=code_to_optimize_with_dependents,
                contextual_dunder_methods=contextual_dunder_methods,
                dependent_functions=dependent_functions,
                preexisting_functions=preexisting_functions,
            ),
        )

    def cleanup_leftover_test_return_values(self):
        # remove leftovers from previous run
        pathlib.Path(get_run_tmp_file("test_return_values_0.bin")).unlink(
            missing_ok=True,
        )
        pathlib.Path(get_run_tmp_file("test_return_values_0.sqlite")).unlink(
            missing_ok=True,
        )

    def instrument_existing_tests(
        self,
        function_name: str,
        module_path: str,
        function_to_tests: dict[str, list[TestsInFile]],
    ):
        relevant_test_files_count = 0
        unique_original_test_files = set()
        unique_instrumented_test_files = set()

        full_module_function_path = module_path + "." + function_name
        if full_module_function_path not in function_to_tests:
            logging.info(
                "Did not find any pre-existing tests for '%s', will only use generated tests.",
                full_module_function_path,
            )
        else:
            for tests_in_file in function_to_tests.get(full_module_function_path):
                if tests_in_file.test_file in unique_original_test_files:
                    continue
                relevant_test_files_count += 1
                success, injected_test = inject_profiling_into_existing_test(
                    tests_in_file.test_file,
                    function_name,
                    self.args.project_root,
                )
                if not success:
                    continue
                new_test_path = (
                    os.path.splitext(tests_in_file.test_file)[0]
                    + "__perfinstrumented"
                    + os.path.splitext(tests_in_file.test_file)[1]
                )
                with open(new_test_path, "w", encoding="utf8") as f:
                    f.write(injected_test)
                unique_instrumented_test_files.add(new_test_path)
                unique_original_test_files.add(tests_in_file.test_file)
            logging.info(
                f"Discovered {relevant_test_files_count} existing unit test file"
                f"{'s' if relevant_test_files_count != 1 else ''} for {full_module_function_path}",
            )
        return unique_instrumented_test_files

    def generate_tests_and_optimizations(
        self,
        code_to_optimize_with_dependents: str,
        function_to_optimize: FunctionToOptimize,
        dependent_functions: list[Tuple[Source, str, str]],
        module_path: str,
        function_trace_id: str,
    ) -> Result[TestsAndOptimizationsResult, str]:
        generated_original_test_source = None
        instrumented_test_source = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_tests = executor.submit(
                self.generate_and_instrument_tests,
                code_to_optimize_with_dependents,
                function_to_optimize,
                [definition[0].full_name for definition in dependent_functions],
                module_path,
                function_trace_id,
            )
            future_optimization = executor.submit(
                optimize_python_code,
                code_to_optimize_with_dependents,
                function_trace_id,
                N_CANDIDATES,
            )

            future_tests_result = future_tests.result()
            optimizations: List[Optimization] = future_optimization.result()
        if future_tests_result and isinstance(future_tests_result, tuple) and len(future_tests_result) == 2:
            (
                generated_original_test_source,
                instrumented_test_source,
            ) = future_tests_result

        else:
            return Failure(f"/!\\ NO TESTS GENERATED for {function_to_optimize.function_name}")
        if not optimizations:
            return Failure(f"/!\\ NO OPTIMIZATIONS GENERATED for {function_to_optimize.function_name}")
        return Success(
            TestsAndOptimizationsResult(
                generated_original_test_source=generated_original_test_source,
                instrumented_test_source=instrumented_test_source,
                optimizations=optimizations,
            ),
        )

    def establish_original_code_baseline(
        self,
        function_name: str,
        instrumented_unittests_created_for_function: set[str],
        generated_tests_path: str,
    ) -> Result[OriginalCodeBaseline, str]:
        original_runtime = None
        best_runtime = None
        original_gen_results = None
        overall_original_test_results = None
        times_run = 0
        success = True
        # Keep the runtime in some acceptable range
        generated_tests_elapsed_time = 0.0

        # For the original function - run the tests and get the runtime
        # TODO: Compare the function return values over the multiple runs and check if they are any different,
        #  if they are different, then we can't optimize this function because it is a non-deterministic function
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = str(0)
        if "PYTHONPATH" not in test_env:
            test_env["PYTHONPATH"] = self.args.project_root
        else:
            test_env["PYTHONPATH"] += os.pathsep + self.args.project_root
        cumulative_test_runtime = 0
        cumulative_test_runs = 0
        first_run = True
        do_break = False
        while (
            cumulative_test_runtime < MAX_CUMULATIVE_TEST_RUNTIME_NANOSECONDS
            and cumulative_test_runs < MAX_TEST_FUNCTION_RUNS
        ):
            for i in range(MAX_TEST_RUN_ITERATIONS):
                if generated_tests_elapsed_time > MAX_FUNCTION_TEST_SECONDS:
                    do_break = True
                    break
                instrumented_existing_test_timing = []
                original_test_results_iter = TestResults()
                for test_file in instrumented_unittests_created_for_function:
                    unittest_results = self.run_and_parse_tests(
                        test_env,
                        test_file,
                        TestType.EXISTING_UNIT_TEST,
                        0,
                    )

                    timing = unittest_results.total_passed_runtime()
                    original_test_results_iter.merge(unittest_results)
                    instrumented_existing_test_timing.append(timing)
                if i == 0 and first_run:
                    logging.info(
                        f"original code, existing unit test results -> {original_test_results_iter.get_test_pass_fail_report()}",
                    )

                original_gen_results = self.run_and_parse_tests(
                    test_env,
                    generated_tests_path,
                    TestType.GENERATED_REGRESSION,
                    0,
                )

                # TODO: Implement the logic to disregard the timing info of the tests that errored out. That is remove test cases that failed to run.

                if not original_gen_results and len(instrumented_existing_test_timing) == 0:
                    logging.warning(
                        f"Couldn't run any tests for original function {function_name}. SKIPPING OPTIMIZING THIS FUNCTION.",
                    )
                    success = False
                    do_break = True
                    break
                # TODO: Doing a simple sum of test runtime, Improve it by looking at test by test runtime, or a better scheme
                # TODO: If the runtime is None, that happens in the case where an exception is expected and is successfully
                #  caught by the test framework. This makes the test pass, but we can't find runtime because the exception caused
                #  the execution to not reach the runtime measurement part. We are currently ignoring such tests, because the performance
                #  for such a execution that raises an exception should not matter.
                if i == 0 and first_run:
                    logging.info(
                        f"original generated tests results -> {original_gen_results.get_test_pass_fail_report()}",
                    )

                original_total_runtime_iter = original_gen_results.total_passed_runtime() + sum(
                    instrumented_existing_test_timing,
                )
                if original_total_runtime_iter == 0:
                    logging.warning(
                        "The overall test runtime of the original function is 0, couldn't run tests.",
                    )
                    logging.warning(original_gen_results.test_results)
                    do_break = True
                    break
                original_test_results_iter.merge(original_gen_results)
                if i == 0 and first_run:
                    logging.info(
                        f"Original overall test results = {TestResults.report_to_string(original_test_results_iter.get_test_pass_fail_report_by_type())}",
                    )
                if original_runtime is None or original_total_runtime_iter < original_runtime:
                    original_runtime = best_runtime = original_total_runtime_iter
                    overall_original_test_results = original_test_results_iter
                cumulative_test_runs += 1
                cumulative_test_runtime += original_total_runtime_iter
                times_run += 1
            if first_run:
                first_run = False
            if do_break:
                break

        if times_run == 0:
            logging.warning(
                "Failed to run the tests for the original function, skipping optimization",
            )
            success = False
        if not success:
            return Failure("Failed to establish a baseline for the original code.")
        logging.info(
            f"Original code runtime measured over {times_run} run{'s' if times_run > 1 else ''}: {humanize_runtime(original_runtime)}",
        )
        return Success(
            OriginalCodeBaseline(
                generated_test_results=original_gen_results,
                overall_test_results=overall_original_test_results,
                runtime=best_runtime,
            ),
        )

    def run_optimized_candidate(
        self,
        optimization_index: int,
        instrumented_unittests_created_for_function: set[str],
        overall_original_test_results: TestResults,
        original_gen_results: TestResults,
        generated_tests_path: str,
        best_runtime_until_now: int,
    ) -> Result[OptimizedCandidateResult, str]:
        success = True
        best_test_runtime = None
        best_test_results = None
        equal_results = True
        generated_tests_elapsed_time = 0.0

        times_run = 0
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = str(optimization_index)
        if "PYTHONPATH" not in test_env:
            test_env["PYTHONPATH"] = self.args.project_root
        else:
            test_env["PYTHONPATH"] += os.pathsep + self.args.project_root
        cumulative_test_runtime = 0
        cumulative_test_runs = 0
        first_run = True
        do_break = False
        while (
            cumulative_test_runtime < MAX_CUMULATIVE_TEST_RUNTIME_NANOSECONDS
            and cumulative_test_runs < MAX_TEST_FUNCTION_RUNS
        ):
            for test_index in range(MAX_TEST_RUN_ITERATIONS):
                pathlib.Path(
                    get_run_tmp_file(f"test_return_values_{optimization_index}.bin"),
                ).unlink(missing_ok=True)
                pathlib.Path(
                    get_run_tmp_file(f"test_return_values_{optimization_index}.sqlite"),
                ).unlink(missing_ok=True)
                if generated_tests_elapsed_time > MAX_FUNCTION_TEST_SECONDS:
                    do_break = True
                    break

                optimized_test_results_iter = TestResults()
                instrumented_test_timing = []
                for instrumented_test_file in instrumented_unittests_created_for_function:
                    unittest_results_optimized = self.run_and_parse_tests(
                        test_env,
                        instrumented_test_file,
                        TestType.EXISTING_UNIT_TEST,
                        optimization_index,
                    )
                    timing = unittest_results_optimized.total_passed_runtime()
                    optimized_test_results_iter.merge(unittest_results_optimized)
                    instrumented_test_timing.append(timing)
                if first_run and test_index == 0:
                    equal_results = True
                    logging.info(
                        f"optimized existing unit tests result -> {optimized_test_results_iter.get_test_pass_fail_report()}",
                    )
                    for test_invocation in optimized_test_results_iter:
                        if (
                            overall_original_test_results.get_by_id(test_invocation.id) is None
                            or test_invocation.did_pass
                            != overall_original_test_results.get_by_id(test_invocation.id).did_pass
                        ):
                            logging.info("Results did not match.")
                            logging.info(
                                f"Test {test_invocation.id} failed on the optimized code. Skipping this optimization",
                            )
                            equal_results = False
                            do_break = True
                            break
                    if not equal_results:
                        do_break = True
                        break

                test_results = self.run_and_parse_tests(
                    test_env,
                    generated_tests_path,
                    TestType.GENERATED_REGRESSION,
                    optimization_index,
                )

                if first_run and test_index == 0:
                    logging.info(
                        f"generated test_results optimized -> {test_results.get_test_pass_fail_report()}",
                    )
                    if test_results:
                        if compare_results(original_gen_results, test_results):
                            equal_results = True
                            logging.info("Results matched!")
                        else:
                            logging.info("Results did not match.")
                            equal_results = False
                if not equal_results:
                    do_break = True
                    break

                test_runtime = test_results.total_passed_runtime() + sum(instrumented_test_timing)

                if test_runtime == 0:
                    logging.warning(
                        "The overall test runtime of the optimized function is 0, couldn't run tests.",
                    )
                    do_break = True
                    break
                if best_test_runtime is None or test_runtime < best_test_runtime:
                    optimized_test_results_iter.merge(test_results)
                    best_test_runtime = test_runtime
                    best_test_results = optimized_test_results_iter
                cumulative_test_runs += 1
                cumulative_test_runtime += test_runtime
                times_run += 1
            if first_run:
                first_run = False
            if best_test_runtime is not None and (best_test_runtime > 3 * best_runtime_until_now):
                # If after 5 runs, the optimized candidate is taking 3 times longer than the best code until now,
                # then it is not a good optimization. Early exit to save time.
                success = True
                do_break = True
            if do_break:
                break

        pathlib.Path(get_run_tmp_file(f"test_return_values_{optimization_index}.bin")).unlink(
            missing_ok=True,
        )
        pathlib.Path(get_run_tmp_file(f"test_return_values_{optimization_index}.sqlite")).unlink(
            missing_ok=True,
        )
        if not (equal_results and times_run > 0):
            success = False

        if not success:
            return Failure("Failed to run the optimized candidate.")
        return Success(
            OptimizedCandidateResult(
                times_run=times_run,
                best_test_runtime=best_test_runtime,
                best_test_results=best_test_results,
            ),
        )

    def run_and_parse_tests(
        self,
        test_env: dict[str, str],
        test_file: str,
        test_type: TestType,
        optimization_iteration: int,
    ) -> TestResults:
        result_file_path, run_result = run_tests(
            test_file,
            test_framework=self.args.test_framework,
            cwd=self.args.project_root,
            pytest_timeout=INDIVIDUAL_TEST_TIMEOUT,
            pytest_cmd=self.test_cfg.pytest_cmd,
            verbose=True,
            test_env=test_env,
        )
        if run_result.returncode != 0:
            logging.debug(
                f"Nonzero return code {run_result.returncode} when running tests in {test_file}.\n"
                f"stdout: {run_result.stdout}\n"
                f"stderr: {run_result.stderr}\n",
            )
        unittest_results = parse_test_results(
            test_xml_path=result_file_path,
            test_py_path=test_file,
            test_config=self.test_cfg,
            test_type=test_type,
            run_result=run_result,
            optimization_iteration=optimization_iteration,
        )
        return unittest_results

    def generate_and_instrument_tests(
        self,
        source_code_being_tested: str,
        function_to_optimize: FunctionToOptimize,
        dependent_function_names: list[str],
        module_path: str,
        function_trace_id: str,
    ) -> Union[Tuple[str, str], None]:
        tests = generate_tests(
            source_code_being_tested=source_code_being_tested,
            function_to_optimize=function_to_optimize,
            dependent_function_names=dependent_function_names,
            module_path=module_path,
            test_cfg=self.test_cfg,
            test_timeout=INDIVIDUAL_TEST_TIMEOUT,
            use_cached_tests=self.args.use_cached_tests,
            function_trace_id=function_trace_id,
        )
        if tests is None:
            logging.error(
                f"Failed to generate and instrument tests for {function_to_optimize.function_name}",
            )
            return None

        generated_original_test_source, instrumented_test_source = tests

        return generated_original_test_source, instrumented_test_source


def run_with_args(args: Namespace) -> None:
    optimizer = Optimizer(args)
    optimizer.run()
