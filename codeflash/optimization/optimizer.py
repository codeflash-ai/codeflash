from __future__ import annotations

import concurrent.futures
import logging
import os
import pathlib
import subprocess
import uuid
from argparse import Namespace
from collections import defaultdict

import isort
import libcst as cst
from returns.pipeline import is_successful
from returns.result import Failure, Result, Success

from codeflash.api.aiservice import (
    AiServiceClient,
    LocalAiServiceClient,
    OptimizedCandidate,
)
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_extractor import (
    add_needed_imports_from_module,
    extract_code,
    find_preexisting_objects,
)
from codeflash.code_utils.code_replacer import replace_function_definitions_in_module
from codeflash.code_utils.code_utils import (
    get_run_tmp_file,
    module_name_from_file_path,
)
from codeflash.code_utils.config_consts import (
    INDIVIDUAL_TESTCASE_TIMEOUT,
    MAX_CUMULATIVE_TEST_RUNTIME_NANOSECONDS,
    MAX_FUNCTION_TEST_SECONDS,
    MAX_TEST_FUNCTION_RUNS,
    MAX_TEST_RUN_ITERATIONS,
    N_CANDIDATES,
)
from codeflash.code_utils.formatter import format_code, sort_imports
from codeflash.code_utils.instrument_existing_tests import (
    inject_profiling_into_existing_test,
)
from codeflash.code_utils.remove_generated_tests import (
    remove_functions_from_generated_tests,
)
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.discovery.discover_unit_tests import (
    TestsInFile,
    discover_unit_tests,
)
from codeflash.discovery.functions_to_optimize import (
    FunctionParent,
    FunctionToOptimize,
    get_functions_to_optimize,
)
from codeflash.models.ExperimentMetadata import ExperimentMetadata
from codeflash.models.models import (
    BestOptimization,
    CodeOptimizationContext,
    FunctionSource,
    GeneratedTests,
    OptimizationSet,
    OptimizedCandidateResult,
    OriginalCodeBaseline,
)
from codeflash.optimization.function_context import (
    get_constrained_function_context_and_helper_functions,
)
from codeflash.result.create_pr import check_create_pr, existing_tests_source_for
from codeflash.result.critic import performance_gain, quantity_of_tests_critic, speedup_critic
from codeflash.result.explanation import Explanation
from codeflash.telemetry.posthog import ph
from codeflash.verification.equivalence import compare_test_results
from codeflash.verification.parse_test_output import parse_test_results
from codeflash.verification.test_results import TestResults, TestType
from codeflash.verification.test_runner import run_tests
from codeflash.verification.verification_utils import TestConfig, get_test_file_path
from codeflash.verification.verifier import generate_tests


class Optimizer:
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.test_cfg = TestConfig(
            tests_root=args.tests_root,
            project_root_path=args.project_root,
            test_framework=args.test_framework,
            pytest_cmd=args.pytest_cmd,
        )

        self.aiservice_client = AiServiceClient()
        self.experiment_id = os.getenv("CODEFLASH_EXPERIMENT_ID", None)
        self.local_aiservice_client = LocalAiServiceClient() if self.experiment_id else None

        self.test_files_created: set[str] = set()
        self.instrumented_unittests_created: set[str] = set()

    def run(self) -> None:
        ph("cli-optimize-run-start")
        logging.info("Running optimizer.")
        if not env_utils.ensure_codeflash_api_key():
            return

        file_to_funcs_to_optimize: dict[str, list[FunctionToOptimize]]
        num_optimizable_functions: int

        (
            file_to_funcs_to_optimize,
            num_optimizable_functions,
        ) = get_functions_to_optimize(
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
            ph(
                "cli-optimize-functions-to-optimize",
                {"num_functions": num_optimizable_functions},
            )
            if num_optimizable_functions == 0:
                logging.info("No functions found to optimize. Exiting...")
                return
            logging.info(
                f"Discovering existing unit tests in {self.test_cfg.tests_root} ...",
            )
            function_to_tests: dict[str, list[TestsInFile]] = discover_unit_tests(
                self.test_cfg,
            )
            num_discovered_tests: int = sum(
                [len(value) for value in function_to_tests.values()],
            )
            logging.info(
                f"Discovered {num_discovered_tests} existing unit tests in {self.test_cfg.tests_root}",
            )
            ph("cli-optimize-discovered-tests", {"num_tests": num_discovered_tests})
            for path in file_to_funcs_to_optimize:
                logging.info(f"Examining file {path} ...")
                # TODO @afik.cohen: Sequence the functions one goes through intelligently. If we are
                #  optimizing f(g(x)), then we might want to first optimize f rather than g because optimizing
                #  f would already optimize g as it is a dependency.
                with pathlib.Path(path).open(encoding="utf8") as f:
                    original_code: str = f.read()

                for function_to_optimize in file_to_funcs_to_optimize[path]:
                    function_iterator_count += 1
                    logging.info(
                        f"Optimizing function {function_iterator_count} of {num_optimizable_functions} - {function_to_optimize.qualified_name}",
                    )
                    best_optimization = self.optimize_function(
                        function_to_optimize,
                        function_to_tests,
                        original_code,
                    )
                    if is_successful(best_optimization):
                        optimizations_found += 1
                    else:
                        logging.warning(best_optimization.failure())
                        continue
            ph("cli-optimize-run-finished", {"optimizations_found": optimizations_found})
            if optimizations_found == 0:
                logging.info("❌ No optimizations found.")
            elif self.args.all:
                logging.info("✨ All functions have been optimized! ✨")
        finally:
            for test_file in self.instrumented_unittests_created:
                pathlib.Path(test_file).unlink(missing_ok=True)
            for test_file in self.test_files_created:
                pathlib.Path(test_file).unlink(missing_ok=True)
            if hasattr(get_run_tmp_file, "tmpdir"):
                get_run_tmp_file.tmpdir.cleanup()

    def optimize_function(
        self,
        function_to_optimize: FunctionToOptimize,
        function_to_tests: dict[str, list[TestsInFile]],
        original_code: str,
    ) -> Result[BestOptimization, str]:
        should_run_experiment = self.experiment_id is not None
        function_trace_id: str = str(uuid.uuid4())
        logging.debug(f"Function Trace ID: {function_trace_id}")
        ph("cli-optimize-function-start", {"function_trace_id": function_trace_id})
        self.cleanup_leftover_test_return_values()
        ctx_result = self.get_code_optimization_context(
            function_to_optimize,
            self.args.project_root,
            original_code,
        )
        if not is_successful(ctx_result):
            return Failure(ctx_result.failure())
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with pathlib.Path(helper_function_path).open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        logging.info(f"Code to be optimized:\n{code_context.code_to_optimize_with_helpers}")
        module_path = module_name_from_file_path(function_to_optimize.file_path, self.args.project_root)

        for module_abspath in original_helper_code:
            code_context.code_to_optimize_with_helpers = add_needed_imports_from_module(
                original_helper_code[module_abspath],
                code_context.code_to_optimize_with_helpers,
                module_abspath,
                function_to_optimize.file_path,
                self.args.project_root,
            )

        instrumented_unittests_created_for_function = self.instrument_existing_tests(
            function_to_optimize=function_to_optimize,
            function_to_tests=function_to_tests,
        )
        self.instrumented_unittests_created.update(instrumented_unittests_created_for_function)

        generated_results = self.generate_tests_and_optimizations(
            code_context.code_to_optimize_with_helpers,
            function_to_optimize,
            code_context.helper_functions,
            module_path,
            function_trace_id,
            run_experiment=should_run_experiment,
        )
        if not is_successful(generated_results):
            return Failure(generated_results.failure())
        tests_and_opts: tuple[GeneratedTests, OptimizationSet] = generated_results.unwrap()
        generated_tests, optimizations_set = tests_and_opts
        generated_tests_path = get_test_file_path(
            self.args.tests_root,
            function_to_optimize.function_name,
            0,
        )
        with pathlib.Path(generated_tests_path).open("w", encoding="utf8") as file:
            file.write(generated_tests.instrumented_test_source)
        logging.info(f"Generated tests:\n{generated_tests.generated_original_test_source}")
        self.test_files_created.add(generated_tests_path)
        baseline_result, test_functions_to_remove = self.establish_original_code_baseline(
            function_to_optimize.qualified_name,
            instrumented_unittests_created_for_function,
            generated_tests_path,
            function_to_tests.get(module_path + "." + function_to_optimize.qualified_name, []),
        )
        if not is_successful(baseline_result):
            pathlib.Path(generated_tests_path).unlink(missing_ok=True)
            for instrumented_path in instrumented_unittests_created_for_function:
                pathlib.Path(instrumented_path).unlink(missing_ok=True)
            return Failure(baseline_result.failure())
        original_code_baseline: OriginalCodeBaseline = baseline_result.unwrap()
        # TODO: Postprocess the optimized function to include the original docstring and such

        best_optimization = None
        for u, candidates in enumerate(
            [optimizations_set.control, optimizations_set.experiment],
        ):
            if candidates is None:
                continue

            tests_in_file: list[TestsInFile] = function_to_tests.get(
                function_to_optimize.qualified_name_with_modules_from_root(self.args.project_root),
                [],
            )

            best_optimization = self.determine_best_candidate(
                candidates,
                code_context,
                function_to_optimize,
                generated_tests_path,
                instrumented_unittests_created_for_function,
                original_code,
                original_code_baseline,
                original_helper_code,
                function_trace_id[:-4] + f"EXP{u}" if should_run_experiment else function_trace_id,
                tests_in_file,
            )
            ph("cli-optimize-function-finished", {"function_trace_id": function_trace_id})

            generated_tests = remove_functions_from_generated_tests(generated_tests, test_functions_to_remove)

            if best_optimization:
                logging.info(
                    f"Best candidate:\n{best_optimization.candidate.source_code}, {best_optimization.candidate.explanation}",
                )

                explanation = Explanation(
                    raw_explanation_message=best_optimization.candidate.explanation,
                    winning_test_results=best_optimization.winning_test_results,
                    original_runtime_ns=original_code_baseline.runtime,
                    best_runtime_ns=best_optimization.runtime,
                    function_name=function_to_optimize.qualified_name,
                    file_path=function_to_optimize.file_path,
                )

                self.log_successful_optimization(
                    explanation,
                    function_to_optimize,
                    function_trace_id,
                    generated_tests,
                )

                self.replace_function_and_helpers_with_optimized_code(
                    code_context=code_context,
                    function_to_optimize_file_path=explanation.file_path,
                    optimized_code=best_optimization.candidate.source_code,
                    qualified_function_name=function_to_optimize.qualified_name,
                )

                new_code, new_helper_code = self.reformat_code_and_helpers(
                    code_context.helper_functions,
                    explanation.file_path,
                    original_code,
                )

                existing_tests = existing_tests_source_for(
                    function_to_optimize.qualified_name_with_modules_from_root(self.args.project_root),
                    function_to_tests,
                    tests_root=self.test_cfg.tests_root,
                )

                original_code_combined = original_helper_code.copy()
                original_code_combined[explanation.file_path] = original_code
                new_code_combined = new_helper_code.copy()
                new_code_combined[explanation.file_path] = new_code
                if not self.args.no_pr:
                    check_create_pr(
                        original_code=original_code_combined,
                        new_code=new_code_combined,
                        explanation=explanation,
                        existing_tests_source=existing_tests,
                        generated_original_test_source=generated_tests.generated_original_test_source,
                        function_trace_id=function_trace_id,
                    )
                    if self.args.all or env_utils.get_pr_number():
                        # Reverting to original code, because optimizing functions in a sequence can lead to
                        #  a) Error propagation, where error in one function can cause the next optimization to fail
                        #  b) Performance estimates become unstable, as the runtime of an optimization might be
                        #     dependent on the runtime of the previous optimization
                        self.write_code_and_helpers(
                            original_code,
                            original_helper_code,
                            function_to_optimize.file_path,
                        )
        # Delete all the generated tests to not cause any clutter.
        pathlib.Path(generated_tests_path).unlink(missing_ok=True)
        for test_paths in instrumented_unittests_created_for_function:
            pathlib.Path(test_paths).unlink(missing_ok=True)
        if not best_optimization:
            return Failure(f"No best optimizations found for function {function_to_optimize.qualified_name}")
        return Success(best_optimization)

    def determine_best_candidate(
        self,
        candidates: list[OptimizedCandidate],
        code_context: CodeOptimizationContext,
        function_to_optimize: FunctionToOptimize,
        generated_tests_path: str,
        instrumented_unittests_created_for_function: set[str],
        original_code: str,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[str, str],
        function_trace_id: str,
        only_run_this_test_function: list[TestsInFile] | None = None,
    ) -> BestOptimization | None:
        best_optimization: BestOptimization | None = None
        best_runtime_until_now = original_code_baseline.runtime  # The fastest code runtime until now

        speedup_ratios: dict[str, float | None] = {}
        optimized_runtimes: dict[str, float | None] = {}
        is_correct = {}

        logging.info(
            f"Determining best optimized candidate (out of {len(candidates)}) for {function_to_optimize.qualified_name} ...",
        )
        try:
            for j, candidate in enumerate(candidates, start=1):
                if candidate.source_code is None:
                    continue
                # remove left overs from previous run
                pathlib.Path(get_run_tmp_file(f"test_return_values_{j}.bin")).unlink(
                    missing_ok=True,
                )
                pathlib.Path(get_run_tmp_file(f"test_return_values_{j}.sqlite")).unlink(
                    missing_ok=True,
                )
                logging.info(f"Optimized candidate {j}/{len(candidates)}:")
                logging.info(candidate.source_code)
                try:
                    did_update = self.replace_function_and_helpers_with_optimized_code(
                        code_context=code_context,
                        function_to_optimize_file_path=function_to_optimize.file_path,
                        optimized_code=candidate.source_code,
                        qualified_function_name=function_to_optimize.qualified_name,
                    )
                    if not did_update:
                        logging.warning(
                            "No functions were replaced in the optimized code. Skipping optimization candidate.",
                        )
                        continue
                except (
                    ValueError,
                    SyntaxError,
                    cst.ParserSyntaxError,
                    AttributeError,
                ) as e:
                    logging.error(e)  # noqa: TRY400
                    self.write_code_and_helpers(
                        original_code,
                        original_helper_code,
                        function_to_optimize.file_path,
                    )
                    continue

                # Run generated tests if at least one of them passed
                run_generated_tests = False
                if original_code_baseline.generated_test_results:
                    for test_result in original_code_baseline.generated_test_results.test_results:
                        if test_result.did_pass:
                            run_generated_tests = True
                            break

                run_results = self.run_optimized_candidate(
                    optimization_index=j,
                    instrumented_unittests_created_for_function=instrumented_unittests_created_for_function,
                    overall_original_test_results=original_code_baseline.overall_test_results,
                    original_existing_test_results=original_code_baseline.existing_test_results,
                    original_generated_test_results=original_code_baseline.generated_test_results,
                    generated_tests_path=generated_tests_path,
                    best_runtime_until_now=best_runtime_until_now,
                    tests_in_file=only_run_this_test_function,
                    run_generated_tests=run_generated_tests,
                )
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
                        original_runtime_ns=original_code_baseline.runtime,
                        optimized_runtime_ns=best_test_runtime,
                    )
                    speedup_ratios[candidate.optimization_id] = perf_gain
                    logging.info(
                        f"Candidate runtime measured over {candidate_result.times_run} run{'s' if candidate_result.times_run > 1 else ''}: "
                        f"{humanize_runtime(best_test_runtime)}, speedup ratio = "
                        f"{perf_gain:.3f}",
                    )

                    if speedup_critic(
                        candidate_result,
                        original_code_baseline.runtime,
                        best_runtime_until_now,
                    ) and quantity_of_tests_critic(candidate_result):
                        logging.info("This candidate is better than the previous best candidate.")
                        logging.info(
                            f"Original runtime: {humanize_runtime(original_code_baseline.runtime)} Best test runtime: "
                            f"{humanize_runtime(candidate_result.best_test_runtime)}, ratio = "
                            f"{perf_gain:.3f}",
                        )
                        best_optimization = BestOptimization(
                            candidate=candidate,
                            helper_functions=code_context.helper_functions,
                            runtime=best_test_runtime,
                            winning_test_results=candidate_result.best_test_results,
                        )
                        best_runtime_until_now = best_test_runtime

                self.write_code_and_helpers(
                    original_code,
                    original_helper_code,
                    function_to_optimize.file_path,
                )
                logging.info("----------------")
        except KeyboardInterrupt as e:
            self.write_code_and_helpers(
                original_code,
                original_helper_code,
                function_to_optimize.file_path,
            )
            logging.exception(f"Optimization interrupted: {e}")
            raise e

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
        generated_tests: GeneratedTests,
    ) -> None:
        logging.info(
            f"⚡️ Optimization successful! 📄 {function_to_optimize.qualified_name} in {explanation.file_path}",
        )
        logging.info(f"📈 {explanation.perf_improvement_line}")
        logging.info(f"Explanation: \n{explanation.to_console_string()}")
        logging.info(
            f"Optimization was validated for correctness by running the following tests - "
            f"\n{generated_tests.generated_original_test_source}",
        )
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

    def write_code_and_helpers(
        self,
        original_code: str,
        original_helper_code: dict[str, str],
        path: str,
    ) -> None:
        with pathlib.Path(path).open("w", encoding="utf8") as f:
            f.write(original_code)
        for module_abspath in original_helper_code:
            with pathlib.Path(module_abspath).open("w", encoding="utf8") as f:
                f.write(original_helper_code[module_abspath])

    def reformat_code_and_helpers(
        self,
        helper_functions: list[FunctionSource],
        path: str,
        original_code: str,
    ) -> tuple[str, dict[str, str]]:
        should_sort_imports = not self.args.disable_imports_sorting
        if should_sort_imports and isort.code(original_code) != original_code:
            should_sort_imports = False

        new_code = format_code(
            self.args.formatter_cmds,
            path,
        )
        if should_sort_imports:
            new_code = sort_imports(new_code)

        new_helper_code: dict[str, str] = {}
        helper_functions_paths = {hf.file_path for hf in helper_functions}
        for module_abspath in helper_functions_paths:
            formatted_helper_code = format_code(
                self.args.formatter_cmds,
                module_abspath,
            )
            if should_sort_imports:
                formatted_helper_code = sort_imports(formatted_helper_code)
            new_helper_code[module_abspath] = formatted_helper_code

        return new_code, new_helper_code

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        function_to_optimize_file_path: str,
        optimized_code: str,
        qualified_function_name: str,
    ) -> bool:
        """Raises many exceptions if the code is not valid. Catch them where using"""
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
                helper_functions_by_module_abspath[helper_function.file_path].add(
                    helper_function.qualified_name,
                )
        for (
            module_abspath,
            qualified_names,
        ) in helper_functions_by_module_abspath.items():
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
        self,
        function_to_optimize: FunctionToOptimize,
        project_root: str,
        original_source_code: str,
    ) -> Result[CodeOptimizationContext, str]:
        code_to_optimize, contextual_dunder_methods = extract_code(
            [function_to_optimize],
        )
        if code_to_optimize is None:
            return Failure("Could not find function to optimize.")
        (
            helper_code,
            helper_functions,
            helper_dunder_methods,
        ) = get_constrained_function_context_and_helper_functions(
            function_to_optimize,
            self.args.project_root,
            code_to_optimize,
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
                code_to_optimize, contextual_dunder_methods = extract_code(
                    list(reversed(dedup_optimizable_methods)),
                )
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
            ),
        )

    def cleanup_leftover_test_return_values(self) -> None:
        # remove leftovers from previous run
        pathlib.Path(get_run_tmp_file("test_return_values_0.bin")).unlink(
            missing_ok=True,
        )
        pathlib.Path(get_run_tmp_file("test_return_values_0.sqlite")).unlink(
            missing_ok=True,
        )

    def instrument_existing_tests(
        self,
        function_to_optimize: FunctionToOptimize,
        function_to_tests: dict[str, list[TestsInFile]],
    ) -> set[str]:
        relevant_test_files_count = 0
        unique_instrumented_test_files = set()

        func_qualname = function_to_optimize.qualified_name_with_modules_from_root(
            self.args.project_root,
        )
        if func_qualname not in function_to_tests:
            logging.info(
                f"Did not find any pre-existing tests for '{func_qualname}', will only use generated tests.",
            )
        else:
            test_file_invocation_positions = defaultdict(list)
            for tests_in_file in function_to_tests.get(func_qualname):
                test_file_invocation_positions[tests_in_file.test_file].append(tests_in_file.position)
            for test_file, positions in test_file_invocation_positions.items():
                relevant_test_files_count += 1
                success, injected_test = inject_profiling_into_existing_test(
                    test_file,
                    positions,
                    function_to_optimize,
                    self.args.project_root,
                    self.args.test_framework,
                )
                if not success:
                    continue
                new_test_path = (
                    f"{os.path.splitext(test_file)[0]}__perfinstrumented{os.path.splitext(test_file)[1]}"
                )
                with pathlib.Path(new_test_path).open("w", encoding="utf8") as f:
                    f.write(injected_test)
                unique_instrumented_test_files.add(new_test_path)
            logging.info(
                f"Discovered {relevant_test_files_count} existing unit test file"
                f"{'s' if relevant_test_files_count != 1 else ''} for {func_qualname}",
            )
        return unique_instrumented_test_files

    def generate_tests_and_optimizations(
        self,
        code_to_optimize_with_helpers: str,
        function_to_optimize: FunctionToOptimize,
        helper_functions: list[FunctionSource],
        module_path: str,
        function_trace_id: str,
        run_experiment: bool = False,
    ) -> Result[tuple[GeneratedTests, OptimizationSet], str]:
        max_workers = 2 if not run_experiment else 3
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_tests = executor.submit(
                self.generate_and_instrument_tests,
                code_to_optimize_with_helpers,
                function_to_optimize,
                [definition.fully_qualified_name for definition in helper_functions],
                module_path,
                function_trace_id[:-4] + "EXP0" if run_experiment else function_trace_id,
            )
            future_optimization_candidates = executor.submit(
                self.aiservice_client.optimize_python_code,
                code_to_optimize_with_helpers,
                function_trace_id[:-4] + "EXP0" if run_experiment else function_trace_id,
                N_CANDIDATES,
                ExperimentMetadata(id=self.experiment_id, group="control") if run_experiment else None,
            )
            if run_experiment:
                future_candidates_exp = executor.submit(
                    self.local_aiservice_client.optimize_python_code,
                    code_to_optimize_with_helpers,
                    function_trace_id[:-4] + "EXP1",
                    N_CANDIDATES,
                    ExperimentMetadata(id=self.experiment_id, group="experiment"),
                )

            future_tests_result = future_tests.result()
            candidates: list[OptimizedCandidate] = future_optimization_candidates.result()

            candidates_experiment = future_candidates_exp.result() if run_experiment else None

        if future_tests_result and isinstance(future_tests_result, tuple) and len(future_tests_result) == 2:
            (
                generated_original_test_source,
                instrumented_test_source,
            ) = future_tests_result

        else:
            return Failure(f"/!\\ NO TESTS GENERATED for {function_to_optimize.function_name}")
        if not candidates:
            return Failure(f"/!\\ NO OPTIMIZATIONS GENERATED for {function_to_optimize.function_name}")
        return Success(
            (
                GeneratedTests(
                    generated_original_test_source=generated_original_test_source,
                    instrumented_test_source=instrumented_test_source,
                ),
                OptimizationSet(
                    control=candidates,
                    experiment=candidates_experiment,
                ),
            ),
        )

    def establish_original_code_baseline(
        self,
        function_name: str,
        instrumented_unittests_created_for_function: set[str],
        generated_tests_path: str,
        tests_in_file: list[TestsInFile],
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
        logging.info(f"Establishing original code baseline runtime for {function_name}.")
        # TODO: Compare the function return values over the multiple runs and check if they are any different,
        #  if they are different, then we can't optimize this function because it is a non-deterministic function
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = str(0)
        test_env["CODEFLASH_TRACER_DISABLE"] = "1"
        if "PYTHONPATH" not in test_env:
            test_env["PYTHONPATH"] = self.args.project_root
        else:
            test_env["PYTHONPATH"] += os.pathsep + self.args.project_root
        cumulative_test_runtime = 0
        cumulative_test_runs = 0
        test_times_list = []
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
                existing_test_results = TestResults()
                for test_file in instrumented_unittests_created_for_function:
                    relevant_tests_in_file = [
                        test_in_file
                        for test_in_file in tests_in_file
                        if test_in_file.test_file == test_file.replace("__perfinstrumented", "")
                    ]
                    is_replay_test = relevant_tests_in_file[0].test_type == TestType.REPLAY_TEST
                    if is_replay_test and len(relevant_tests_in_file) > 1:
                        logging.warning(
                            f"Multiple tests found for the replay test {test_file}. Should not happen",
                        )

                    unittest_results = self.run_and_parse_tests(
                        test_env,
                        test_file,
                        relevant_tests_in_file[0].test_type,
                        0,
                        relevant_tests_in_file[0].test_function if is_replay_test else None,
                    )

                    timing = unittest_results.total_passed_runtime()
                    original_test_results_iter.merge(unittest_results)
                    existing_test_results.merge(unittest_results)
                    instrumented_existing_test_timing.append(timing)
                if i == 0 and first_run:
                    logging.info(
                        f"Existing unit test results for original code: {original_test_results_iter.get_test_pass_fail_report()}",
                    )

                original_gen_results = self.run_and_parse_tests(
                    test_env,
                    generated_tests_path,
                    TestType.GENERATED_REGRESSION,
                    0,
                )
                functions_to_remove = [
                    result.id.test_function_name
                    for result in original_gen_results.test_results
                    if not result.did_pass
                ]

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
                        f"Generated tests results for original code: {original_gen_results.get_test_pass_fail_report()}",
                    )

                if not original_gen_results:
                    original_total_runtime_iter = sum(instrumented_existing_test_timing)
                else:
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
                        f"Overall test results for original code: {TestResults.report_to_string(original_test_results_iter.get_test_pass_fail_report_by_type())}",
                    )
                test_times_list.append(original_total_runtime_iter)
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

        if times_run == 0 and original_runtime is None:
            logging.warning(
                "Failed to run the tests for the original function, skipping optimization",
            )
            success = False
        if not success:
            return Failure("Failed to establish a baseline for the original code."), []
        logging.info(
            f"Original code runtime measured over {times_run} run{'s' if times_run > 1 else ''}: {humanize_runtime(original_runtime)}",
        )
        logging.debug(f"Original code test runtimes: {test_times_list}")
        return (
            Success(
                OriginalCodeBaseline(
                    generated_test_results=original_gen_results,
                    existing_test_results=existing_test_results,
                    overall_test_results=overall_original_test_results,
                    runtime=best_runtime,
                ),
            ),
            functions_to_remove,
        )

    def run_optimized_candidate(
        self,
        optimization_index: int,
        instrumented_unittests_created_for_function: set[str],
        overall_original_test_results: TestResults,
        original_existing_test_results: TestResults,
        original_generated_test_results: TestResults,
        generated_tests_path: str,
        best_runtime_until_now: int,
        tests_in_file: list[TestsInFile] | None,
        run_generated_tests: bool,
    ) -> Result[OptimizedCandidateResult, str]:
        success = True
        best_test_runtime = None
        best_test_results = None
        equal_results = True
        generated_tests_elapsed_time = 0.0

        times_run = 0
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = str(optimization_index)
        test_env["CODEFLASH_TRACER_DISABLE"] = "1"
        if "PYTHONPATH" not in test_env:
            test_env["PYTHONPATH"] = self.args.project_root
        else:
            test_env["PYTHONPATH"] += os.pathsep + self.args.project_root
        cumulative_test_runtime = 0
        cumulative_test_runs = 0
        test_times_list = []
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

                candidate_existing_test_results = TestResults()
                instrumented_test_timing = []
                for instrumented_test_file in instrumented_unittests_created_for_function:
                    relevant_tests_in_file = [
                        test_in_file
                        for test_in_file in tests_in_file
                        if test_in_file.test_file == instrumented_test_file.replace("__perfinstrumented", "")
                    ]
                    is_replay_test = relevant_tests_in_file[0].test_type == TestType.REPLAY_TEST
                    if is_replay_test and len(relevant_tests_in_file) > 1:
                        logging.warning(
                            f"Multiple tests found for the replay test {instrumented_test_file}. Should not happen",
                        )
                    candidate_existing_test_result = self.run_and_parse_tests(
                        test_env,
                        instrumented_test_file,
                        relevant_tests_in_file[0].test_type,
                        optimization_index,
                        relevant_tests_in_file[0].test_function if is_replay_test else None,
                    )
                    timing = candidate_existing_test_result.total_passed_runtime()
                    candidate_existing_test_results.merge(candidate_existing_test_result)
                    instrumented_test_timing.append(timing)
                if first_run and test_index == 0:
                    equal_results = True
                    logging.info(
                        f"Existing unit tests results for candidate: {candidate_existing_test_results.get_test_pass_fail_report()}",
                    )
                    return_values_are_equal = compare_test_results(
                        original_existing_test_results,
                        candidate_existing_test_results,
                    )
                    for test_invocation in candidate_existing_test_results:
                        original_test_invocation = original_existing_test_results.get_by_id(
                            test_invocation.id,
                        )
                        if (
                            original_test_invocation is not None
                            and not original_test_invocation.timed_out
                            and (test_invocation.did_pass != original_test_invocation.did_pass)
                        ) or not return_values_are_equal:
                            logging.info(
                                "Test results did not match the test results of the original code.",
                            )
                            logging.info(
                                f"Test {test_invocation.id} failed. Skipping this candidate.",
                            )
                            equal_results = False
                            do_break = True
                            break
                    if not equal_results:
                        do_break = True
                        break

                candidate_generated_test_results = None
                if run_generated_tests:
                    candidate_generated_test_results = self.run_and_parse_tests(
                        test_env,
                        generated_tests_path,
                        TestType.GENERATED_REGRESSION,
                        optimization_index,
                    )

                if candidate_generated_test_results and first_run and test_index == 0:
                    logging.info(
                        f"Generated tests results for candidate: {candidate_generated_test_results.get_test_pass_fail_report()}",
                    )
                    if compare_test_results(
                        original_generated_test_results,
                        candidate_generated_test_results,
                    ):
                        equal_results = True
                        logging.info("Test results matched!")
                    else:
                        logging.info("Test results did not match the test results of the original code.")
                        equal_results = False
                if not equal_results:
                    do_break = True
                    break

                if not candidate_generated_test_results:
                    test_runtime = sum(instrumented_test_timing)
                else:
                    test_runtime = candidate_generated_test_results.total_passed_runtime() + sum(
                        instrumented_test_timing,
                    )

                if test_runtime == 0:
                    logging.warning(
                        "The overall test runtime of the optimized function is 0, couldn't run tests.",
                    )
                    do_break = True
                    break
                test_times_list.append(test_runtime)
                if best_test_runtime is None or test_runtime < best_test_runtime:
                    if candidate_generated_test_results:
                        candidate_existing_test_results.merge(candidate_generated_test_results)
                    best_test_runtime = test_runtime
                    best_test_results = candidate_existing_test_results
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
        logging.debug(f"Optimized code {optimization_index} runtime: {test_times_list}")
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
        test_function: str | None = None,
    ) -> TestResults:
        try:
            result_file_path, run_result = run_tests(
                test_file,
                test_framework=self.args.test_framework,
                cwd=self.args.project_root,
                pytest_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                pytest_cmd=self.test_cfg.pytest_cmd,
                verbose=True,
                test_env=test_env,
                only_run_this_test_function=test_function,
            )
        except subprocess.TimeoutExpired:
            logging.exception(
                f"Error running tests in {test_file}.\nTimeout Error",
            )
            return TestResults()
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
        helper_function_names: list[str],
        module_path: str,
        function_trace_id: str,
    ) -> tuple[str, str] | None:
        tests = generate_tests(
            self.aiservice_client,
            source_code_being_tested=source_code_being_tested,
            function_to_optimize=function_to_optimize,
            helper_function_names=helper_function_names,
            module_path=module_path,
            test_cfg=self.test_cfg,
            test_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
            use_cached_tests=self.args.use_cached_tests,
            function_trace_id=function_trace_id,
        )
        if tests is None:
            logging.warning(
                f"Failed to generate and instrument tests for {function_to_optimize.function_name}",
            )
            return None

        generated_original_test_source, instrumented_test_source = tests

        return generated_original_test_source, instrumented_test_source


def run_with_args(args: Namespace) -> None:
    optimizer = Optimizer(args)
    optimizer.run()
