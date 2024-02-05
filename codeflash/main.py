import concurrent.futures
import logging
import os
import pathlib
from argparse import ArgumentParser, SUPPRESS, Namespace
from typing import Tuple, Union

import libcst as cst

import codeflash.cli_cmds.logging_config  # intializes logging, has to be the first non-system import # noqa
from codeflash.api.aiservice import optimize_python_code
from codeflash.cli_cmds.cli import process_cmd_args
from codeflash.cli_cmds.cmd_init import CODEFLASH_LOGO
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_extractor import get_code
from codeflash.code_utils.code_replacer import replace_function_in_file
from codeflash.code_utils.code_utils import (
    module_name_from_file_path,
    get_all_function_names,
    get_run_tmp_file,
)
from codeflash.code_utils.config_consts import (
    MAX_TEST_RUN_ITERATIONS,
    MAX_FUNCTION_TEST_SECONDS,
    INDIVIDUAL_TEST_TIMEOUT,
    N_CANDIDATES,
)
from codeflash.code_utils.instrument_existing_tests import (
    inject_profiling_into_existing_test,
)
from codeflash.code_utils.linter import lint_code
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.discovery.discover_unit_tests import discover_unit_tests, TestsInFile
from codeflash.discovery.functions_to_optimize import (
    get_functions_to_optimize_by_file,
    FunctionToOptimize,
)
from codeflash.optimization.function_context import (
    get_constrained_function_context_and_dependent_functions,
    Source,
)
from codeflash.result.create_pr import check_create_pr
from codeflash.result.explanation import Explanation
from codeflash.verification.equivalence import compare_results
from codeflash.verification.parse_test_output import (
    TestType,
    parse_test_results,
)
from codeflash.verification.test_results import TestResults
from codeflash.verification.test_runner import run_tests
from codeflash.verification.verification_utils import (
    get_test_file_path,
    TestConfig,
)
from codeflash.verification.verifier import generate_tests


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("command", nargs="?", help="The command to run (e.g., 'init')")
    parser.add_argument("--file", help="Try to optimize only this file")
    parser.add_argument(
        "--function",
        help="Try to optimize only this function within the given file path",
    )
    parser.add_argument(
        "--all",
        help="Try to optimize all functions. Can take a really long time. Can pass an optional starting directory to"
        " optimize code from. If no args specified (just --all), will optimize all code in the project.",
        nargs="?",
        const="",
        default=SUPPRESS,
    )
    parser.add_argument(
        "--module-root",
        type=str,
        help="Path to the project's Python module that you want to optimize."
        " This is the top-level root directory where all the Python source code is located.",
    )
    parser.add_argument(
        "--tests-root",
        type=str,
        help="Path to the test directory of the project, where all the tests are located.",
    )
    parser.add_argument("--test-framework", choices=["pytest", "unittest"])
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the pyproject.toml with codeflash configs.",
    )
    parser.add_argument(
        "--pytest-cmd",
        type=str,
        help="Command that codeflash will use to run pytest. If not specified, codeflash will use 'pytest'",
    )
    parser.add_argument(
        "--use-cached-tests",
        action="store_true",
        help="Use cached tests from a specified file for debugging.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose logs"
    )
    parser.add_argument(
        "--version", action="store_true", help="Print the version of codeflash"
    )
    args: Namespace = parser.parse_args()
    return process_cmd_args(args)


class Optimizer:
    def __init__(self, args: Namespace):
        self.args = args
        self.test_cfg = TestConfig(
            tests_root=args.tests_root,
            project_root_path=args.project_root,
            test_framework=args.test_framework,
            pytest_cmd=args.pytest_cmd,
        )

    def run(self):
        logging.info(CODEFLASH_LOGO)
        logging.info("Running optimizer.")
        if not env_utils.ensure_codeflash_api_key():
            return

        (
            file_to_funcs_to_optimize,
            num_modified_functions,
        ) = get_functions_to_optimize_by_file(
            optimize_all=self.args.all,
            file=self.args.file,
            function=self.args.function,
            test_cfg=self.test_cfg,
            ignore_paths=self.args.ignore_paths,
        )

        test_files_created = set()
        instrumented_unittests_created = set()
        found_atleast_one_optimization = False

        function_iterator_count = 0
        try:
            if num_modified_functions == 0:
                logging.info("No functions found to optimize. Exiting...")
                return
            logging.info(
                f"Discovering existing unit tests in {self.test_cfg.tests_root} ..."
            )
            function_to_tests: dict[str, list[TestsInFile]] = discover_unit_tests(
                self.test_cfg
            )
            logging.info(
                f"Discovered {sum([len(value) for value in function_to_tests.values()])} "
                f"existing unit tests in {self.test_cfg.tests_root}"
            )
            for path in file_to_funcs_to_optimize:
                logging.info(f"Examining file {path} ...")
                # TODO: Sequence the functions one goes through intelligently. If we are optimizing f(g(x)),
                #  then we might want to first optimize f rather than g because optimizing f would already
                #  optimize g as it is a dependency
                with open(path, "r") as f:
                    original_code = f.read()
                for function_to_optimize in file_to_funcs_to_optimize[path]:
                    function_name = function_to_optimize.function_name
                    function_iterator_count += 1
                    logging.info(
                        f"Optimizing function {function_iterator_count} of {num_modified_functions} - {function_name}"
                    )
                    winning_test_results = None
                    # remove left overs from previous run
                    pathlib.Path(get_run_tmp_file("test_return_values_0.bin")).unlink(
                        missing_ok=True
                    )
                    pathlib.Path(
                        get_run_tmp_file("test_return_values_0.sqlite")
                    ).unlink(missing_ok=True)
                    code_to_optimize = get_code(function_to_optimize)
                    if code_to_optimize is None:
                        logging.error("Could not find function to optimize")
                        continue

                    preexisting_functions = get_all_function_names(code_to_optimize)

                    (
                        code_to_optimize_with_dependents,
                        dependent_functions,
                    ) = get_constrained_function_context_and_dependent_functions(
                        function_to_optimize, self.args.project_root, code_to_optimize
                    )
                    logging.info(
                        f"Code to be optimized:\n{code_to_optimize_with_dependents}"
                    )
                    module_path = module_name_from_file_path(
                        path, self.args.project_root
                    )

                    instrumented_unittests_created_for_function = (
                        self.prepare_existing_tests(
                            function_name=function_name,
                            module_path=module_path,
                            function_to_tests=function_to_tests,
                        )
                    )
                    instrumented_unittests_created.update(
                        instrumented_unittests_created_for_function
                    )

                    (
                        success,
                        generated_original_test_source,
                        instrumented_test_source,
                        optimizations,
                    ) = self.generate_tests_and_optimizations(
                        code_to_optimize_with_dependents,
                        function_to_optimize,
                        dependent_functions,
                        module_path,
                    )
                    if not success:
                        continue

                    generated_tests_path = get_test_file_path(
                        self.args.tests_root, function_to_optimize.function_name, 0
                    )
                    with open(generated_tests_path, "w") as file:
                        file.write(instrumented_test_source)

                    test_files_created.add(generated_tests_path)
                    (
                        success,
                        original_gen_results,
                        overall_original_test_results,
                        original_runtime,
                    ) = self.establish_original_code_baseline(
                        function_name,
                        instrumented_unittests_created_for_function,
                        generated_tests_path,
                    )
                    if not success:
                        continue
                    best_runtime = (
                        original_runtime  # The fastest code runtime until now
                    )
                    logging.info("OPTIMIZING CODE....")
                    # TODO: Postprocess the optimized function to include the original docstring and such

                    best_optimization = []
                    for i, (optimized_code, explanation) in enumerate(optimizations):
                        j = i + 1
                        if optimized_code is None:
                            continue
                        # remove left overs from previous run
                        pathlib.Path(
                            get_run_tmp_file(f"test_return_values_{j}.bin")
                        ).unlink(missing_ok=True)
                        pathlib.Path(
                            get_run_tmp_file(f"test_return_values_{j}.sqlite")
                        ).unlink(missing_ok=True)
                        logging.info("Optimized Candidate:")
                        logging.info(optimized_code)
                        try:
                            new_code = replace_function_in_file(
                                path,
                                function_name,
                                optimized_code,
                                preexisting_functions,
                            )
                        except (
                            ValueError,
                            SyntaxError,
                            cst.ParserSyntaxError,
                            AttributeError,
                        ) as e:
                            logging.error(e)
                            continue
                        with open(path, "w") as f:
                            f.write(new_code)
                        (
                            success,
                            times_run,
                            best_test_runtime,
                            best_test_results,
                        ) = self.run_optimized_candidate(
                            optimization_index=j,
                            instrumented_unittests_created_for_function=instrumented_unittests_created_for_function,
                            overall_original_test_results=overall_original_test_results,
                            original_gen_results=original_gen_results,
                            generated_tests_path=generated_tests_path,
                        )

                        if success:
                            logging.info(
                                f"NEW CODE RUNTIME OVER {times_run} RUN{'S' if times_run > 1 else ''} = "
                                f"{humanize_runtime(best_test_runtime)}, SPEEDUP RATIO = "
                                f"{((original_runtime - best_test_runtime) / best_test_runtime):.3f}"
                            )
                            if (
                                (
                                    (original_runtime - best_test_runtime)
                                    / best_test_runtime
                                )
                                > self.args.minimum_performance_gain
                            ) and best_test_runtime < best_runtime:
                                logging.info("THIS IS BETTER!")

                                logging.info(
                                    f"original_test_time={humanize_runtime(original_runtime)} new_test_time="
                                    f"{humanize_runtime(best_test_runtime)}, FASTER RATIO = "
                                    f"{((original_runtime - best_test_runtime) / best_test_runtime)}"
                                )
                                best_optimization = [
                                    optimized_code,
                                    explanation,
                                    dependent_functions,
                                ]
                                best_runtime = best_test_runtime
                                winning_test_results = best_test_results
                        with open(path, "w") as f:
                            f.write(original_code)
                        logging.info("----------------")
                    logging.info(f"BEST OPTIMIZATION {best_optimization[0:2]}")
                    if best_optimization:
                        found_atleast_one_optimization = True
                        logging.info(f"BEST OPTIMIZED CODE\n{best_optimization[0]}")

                        new_code = replace_function_in_file(
                            path,
                            function_name,
                            best_optimization[0],
                            preexisting_functions,
                        )
                        with open(path, "w") as f:
                            f.write(new_code)
                        explanation_final = Explanation(
                            raw_explanation_message=best_optimization[1],
                            winning_test_results=winning_test_results,
                            original_runtime_ns=original_runtime,
                            best_runtime_ns=best_runtime,
                            function_name=function_name,
                            path=path,
                        )
                        logging.info(
                            f"EXPLANATION\n{explanation_final.to_console_string()}"
                        )

                        new_code = lint_code(path)

                        logging.info(
                            f"Optimization was validated for correctness by running the following test - "
                            f"\n{generated_original_test_source}"
                        )

                        logging.info(
                            f"⚡️ Optimization successful! 📄 {function_name} in {path}"
                        )
                        logging.info(f"📈 {explanation_final.perf_improvement_line}")

                        # TODO: Create multi-file PR for dependent functions, extract dependent functions from new code
                        # and overwrite original definitions, with replace_function_in_file. Also need to lint edited
                        # files.
                        check_create_pr(
                            optimize_all=self.args.all,
                            path=path,
                            original_code=original_code,
                            new_code=new_code,
                            explanation=explanation_final,
                            generated_original_test_source=generated_original_test_source,
                        )
                        if self.args.all or env_utils.get_pr_number():
                            # Reverting to original code, because optimizing functions in a sequence can lead to
                            #  a. Error propagation, where error in one function can cause the next optimization to fail
                            #  b. Performance estimates become unstable, as the runtime of an optimization might be
                            #     dependent on the runtime of the previous optimization
                            with open(path, "w") as f:
                                f.write(original_code)
                    # Delete all the generated tests to not cause any clutter.
                    pathlib.Path(generated_tests_path).unlink(missing_ok=True)
                    for test_paths in instrumented_unittests_created_for_function:
                        pathlib.Path(test_paths).unlink(missing_ok=True)
            if not found_atleast_one_optimization:
                logging.info("❌ No optimizations found.")

        finally:
            # TODO: Also revert the file/function being optimized if the process did not succeed
            for test_file in instrumented_unittests_created:
                pathlib.Path(test_file).unlink(missing_ok=True)
            for test_file in test_files_created:
                pathlib.Path(test_file).unlink(missing_ok=True)
            if hasattr(get_run_tmp_file, "tmpdir"):
                get_run_tmp_file.tmpdir.cleanup()

    def prepare_existing_tests(
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
            logging.warning(
                "Could not find any pre-existing tests for '%s', will only use generated tests.",
                full_module_function_path,
            )
        else:
            for tests_in_file in function_to_tests.get(full_module_function_path):
                if tests_in_file.test_file in unique_original_test_files:
                    continue
                relevant_test_files_count += 1
                injected_test = inject_profiling_into_existing_test(
                    tests_in_file.test_file,
                    function_name,
                    self.args.project_root,
                )
                new_test_path = (
                    os.path.splitext(tests_in_file.test_file)[0]
                    + "__perfinstrumented"
                    + os.path.splitext(tests_in_file.test_file)[1]
                )
                with open(new_test_path, "w") as f:
                    f.write(injected_test)
                unique_instrumented_test_files.add(new_test_path)
                unique_original_test_files.add(tests_in_file.test_file)
            logging.info(
                f"Discovered {relevant_test_files_count} existing unit test file"
                f"{'s' if relevant_test_files_count != 1 else ''} for {full_module_function_path}"
            )
        return unique_instrumented_test_files

    def generate_tests_and_optimizations(
        self,
        code_to_optimize_with_dependents: str,
        function_to_optimize: FunctionToOptimize,
        dependent_functions: list[Tuple[Source, str]],
        module_path: str,
    ):
        generated_original_test_source = None
        instrumented_test_source = None
        success = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_tests = executor.submit(
                self.generate_and_instrument_tests,
                code_to_optimize_with_dependents,
                function_to_optimize,
                [definition[0].full_name for definition in dependent_functions],
                module_path,
            )
            future_optimization = executor.submit(
                optimize_python_code,
                code_to_optimize_with_dependents,
                N_CANDIDATES,
            )

            future_tests_result = future_tests.result()
            optimizations = future_optimization.result()
        if (
            future_tests_result
            and isinstance(future_tests_result, tuple)
            and len(future_tests_result) == 2
        ):
            (
                generated_original_test_source,
                instrumented_test_source,
            ) = future_tests_result

        else:
            logging.error(
                "/!\\ NO TESTS GENERATED for %s", function_to_optimize.function_name
            )
            success = False
        if len(optimizations) == 1 and optimizations[0][0] is None:
            logging.error(
                "/!\\ NO OPTIMIZATIONS GENERATED for %s",
                function_to_optimize.function_name,
            )
            success = False
        return (
            success,
            generated_original_test_source,
            instrumented_test_source,
            optimizations,
        )

    def establish_original_code_baseline(
        self,
        function_name: str,
        instrumented_unittests_created_for_function: set[str],
        generated_tests_path: str,
    ):
        original_runtime = None
        best_runtime = None
        original_gen_results = None
        overall_original_test_results = None
        times_run = 0
        success = True
        # TODO : Dynamically determine the number of times to run the tests based on the runtime of the tests.
        # Keep the runtime in some acceptable range
        generated_tests_elapsed_time = 0.0

        # For the original function - run the tests and get the runtime
        # TODO: Compare the function return values over the multiple runs and check if they are any different,
        #  if they are different, then we can't optimize this function because it is a non-deterministic function
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = str(0)
        for i in range(MAX_TEST_RUN_ITERATIONS):
            if generated_tests_elapsed_time > MAX_FUNCTION_TEST_SECONDS:
                break
            instrumented_existing_test_timing = []
            original_test_results_iter = TestResults()
            for test_file in instrumented_unittests_created_for_function:
                unittest_results = self.run_and_parse_tests(
                    test_env, test_file, TestType.EXISTING_UNIT_TEST, 0
                )

                timing = unittest_results.total_passed_runtime()
                original_test_results_iter.merge(unittest_results)
                instrumented_existing_test_timing.append(timing)
            if i == 0:
                logging.info(
                    f"original code, existing unit test results -> "
                    f"{original_test_results_iter.get_test_pass_fail_report()}"
                )

            original_gen_results = self.run_and_parse_tests(
                test_env, generated_tests_path, TestType.GENERATED_REGRESSION, 0
            )

            # TODO: Implement the logic to disregard the timing info of the tests that ERRORed out.
            # That is remove test cases that failed to run.

            if not original_gen_results and len(instrumented_existing_test_timing) == 0:
                logging.warning(
                    f"Couldn't run any tests for original function {function_name}. SKIPPING OPTIMIZING THIS FUNCTION."
                )
                success = False
                break
            # TODO: Doing a simple sum of test runtime, Improve it by looking at test by test runtime,
            #  or a better scheme
            # TODO: If the runtime is None, that happens in the case where an exception is expected and is successfully
            #  caught by the test framework. This makes the test pass, but we can't find runtime because the exception
            #  caused the execution to not reach the runtime measurement part. We are currently ignoring such tests,
            #  because the performance for such a execution that raises an exception should not matter.
            if i == 0:
                logging.info(
                    f"original generated tests results -> {original_gen_results.get_test_pass_fail_report()}"
                )

            original_total_runtime_iter = (
                original_gen_results.total_passed_runtime()
                + sum(instrumented_existing_test_timing)
            )
            if original_total_runtime_iter == 0:
                logging.warning(
                    "The overall test runtime of the original function is 0, trying again..."
                )
                logging.warning(original_gen_results.test_results)
                continue
            original_test_results_iter.merge(original_gen_results)
            if i == 0:
                logging.info(
                    f"Original overall test results = {TestResults.report_to_string(
                        original_test_results_iter.get_test_pass_fail_report_by_type())}"
                )
            if (
                original_runtime is None
                or original_total_runtime_iter < original_runtime
            ):
                original_runtime = best_runtime = original_total_runtime_iter
                overall_original_test_results = original_test_results_iter

            times_run += 1

        if times_run == 0:
            logging.warning(
                "Failed to run the tests for the original function, skipping optimization"
            )
            success = False
        if success:
            logging.info(
                f"ORIGINAL CODE RUNTIME OVER {times_run} RUN{'S' if times_run > 1 else ''} = {original_runtime}ns"
            )
        return (
            success,
            original_gen_results,
            overall_original_test_results,
            best_runtime,
        )

    def run_optimized_candidate(
        self,
        optimization_index: int,
        instrumented_unittests_created_for_function: set[str],
        overall_original_test_results: TestResults,
        original_gen_results: TestResults,
        generated_tests_path: str,
    ):
        success = True
        best_test_runtime = None
        best_test_results = None
        equal_results = True
        generated_tests_elapsed_time = 0.0

        times_run = 0
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = str(optimization_index)
        for test_index in range(MAX_TEST_RUN_ITERATIONS):
            pathlib.Path(
                get_run_tmp_file(f"test_return_values_{optimization_index}.bin")
            ).unlink(missing_ok=True)
            pathlib.Path(
                get_run_tmp_file(f"test_return_values_{optimization_index}.sqlite")
            ).unlink(missing_ok=True)
            if generated_tests_elapsed_time > MAX_FUNCTION_TEST_SECONDS:
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
            if test_index == 0:
                equal_results = True
                logging.info(
                    f"optimized existing unit tests result -> {optimized_test_results_iter.get_test_pass_fail_report()}"
                )
                for test_invocation in optimized_test_results_iter:
                    if (
                        overall_original_test_results.get_by_id(test_invocation.id)
                        is None
                        or test_invocation.did_pass
                        != overall_original_test_results.get_by_id(
                            test_invocation.id
                        ).did_pass
                    ):
                        logging.info("RESULTS DID NOT MATCH")
                        logging.info(
                            f"Test {test_invocation.id} failed on the optimized code. Skipping this optimization"
                        )
                        equal_results = False
                        break
                if not equal_results:
                    break

            test_results = self.run_and_parse_tests(
                test_env,
                generated_tests_path,
                TestType.GENERATED_REGRESSION,
                optimization_index,
            )

            if test_index == 0:
                logging.info(
                    f"generated test_results optimized -> {test_results.get_test_pass_fail_report()}"
                )
                if test_results:
                    if compare_results(original_gen_results, test_results):
                        equal_results = True
                        logging.info("RESULTS MATCHED!")
                    else:
                        logging.info("RESULTS DID NOT MATCH")
                        equal_results = False
            if not equal_results:
                break

            test_runtime = test_results.total_passed_runtime() + sum(
                instrumented_test_timing
            )

            if test_runtime == 0:
                logging.warning(
                    "The overall test runtime of the optimized function is 0, trying again..."
                )
                continue
            if best_test_runtime is None or test_runtime < best_test_runtime:
                optimized_test_results_iter.merge(test_results)
                best_test_runtime = test_runtime
                best_test_results = optimized_test_results_iter

            times_run += 1
        pathlib.Path(
            get_run_tmp_file(f"test_return_values_{optimization_index}.bin")
        ).unlink(missing_ok=True)
        pathlib.Path(
            get_run_tmp_file(f"test_return_values_{optimization_index}.sqlite")
        ).unlink(missing_ok=True)
        if not (equal_results and times_run > 0):
            success = False

        return (
            success,
            times_run,
            best_test_runtime,
            best_test_results,
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
    ) -> Union[Tuple[str, str], None]:
        tests = generate_tests(
            source_code_being_tested=source_code_being_tested,
            function_to_optimize=function_to_optimize,
            dependent_function_names=dependent_function_names,
            module_path=module_path,
            test_cfg=self.test_cfg,
            test_timeout=INDIVIDUAL_TEST_TIMEOUT,
            use_cached_tests=self.args.use_cached_tests,
        )
        if tests is None:
            logging.error(
                f"Failed to generate and instrument tests for {function_to_optimize.function_name}"
            )
            return None

        generated_original_test_source, instrumented_test_source = tests

        return generated_original_test_source, instrumented_test_source


def main():
    """Entry point for the codeflash command-line interface."""
    Optimizer(parse_args()).run()


if __name__ == "__main__":
    main()
