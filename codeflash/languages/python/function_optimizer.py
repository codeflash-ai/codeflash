from __future__ import annotations

import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.languages.python.context.unused_definition_remover import (
    detect_unused_helper_functions,
    revert_unused_helper_functions,
)
from codeflash.languages.python.optimizer import resolve_python_function_ast
from codeflash.languages.python.static_analysis.code_extractor import get_opt_review_metrics, is_numerical_code
from codeflash.languages.python.static_analysis.code_replacer import (
    add_custom_marker_to_all_tests,
    modify_autouse_fixture,
)
from codeflash.languages.python.static_analysis.line_profile_utils import add_decorator_imports, contains_jit_decorator
from codeflash.models.models import TestFile, TestingMode, TestResults, TestType
from codeflash.optimization.function_optimizer import FunctionOptimizer

if TYPE_CHECKING:
    from codeflash.languages.base import Language
    from codeflash.models.models import CodeOptimizationContext, CodeStringsMarkdown, FunctionCalledInTest


class PythonFunctionOptimizer(FunctionOptimizer):
    def _resolve_function_ast(
        self, source_code: str, function_name: str, parents: list
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        original_module_ast = ast.parse(source_code)
        return resolve_python_function_ast(function_name, parents, original_module_ast)

    def analyze_code_characteristics(self, code_context: CodeOptimizationContext) -> None:
        self.is_numerical_code = is_numerical_code(code_string=code_context.read_writable_code.flat)

    def get_optimization_review_metrics(
        self,
        source_code: str,
        file_path: Path,
        qualified_name: str,
        project_root: Path,
        tests_root: Path,
        language: Language,
    ) -> str:
        return get_opt_review_metrics(source_code, file_path, qualified_name, project_root, tests_root, language)

    def instrument_test_fixtures(self, test_paths: list[Path]) -> dict[Path, list[str]] | None:
        logger.info("Disabling all autouse fixtures associated with the generated test files")
        original_conftest_content = modify_autouse_fixture(test_paths)
        logger.info("Add custom marker to generated test files")
        add_custom_marker_to_all_tests(test_paths)
        return original_conftest_content

    def instrument_existing_tests(self, function_to_all_tests: dict[str, set[FunctionCalledInTest]]) -> set[Path]:
        existing_test_files_count = 0
        replay_test_files_count = 0
        concolic_coverage_test_files_count = 0
        unique_instrumented_test_files = set()

        func_qualname = self.function_to_optimize.qualified_name_with_modules_from_root(self.project_root)
        if func_qualname not in function_to_all_tests:
            logger.info(f"Did not find any pre-existing tests for '{func_qualname}', will only use generated tests.")
            return unique_instrumented_test_files

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

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        optimized_code: CodeStringsMarkdown,
        original_helper_code: dict[Path, str],
    ) -> bool:
        did_update = super().replace_function_and_helpers_with_optimized_code(
            code_context, optimized_code, original_helper_code
        )
        unused_helpers = detect_unused_helper_functions(self.function_to_optimize, code_context, optimized_code)
        if unused_helpers:
            revert_unused_helper_functions(self.project_root, unused_helpers, original_helper_code)
        return did_update

    def _line_profiler_step_python(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], candidate_index: int
    ) -> dict:
        candidate_fto_code = Path(self.function_to_optimize.file_path).read_text("utf-8")
        if contains_jit_decorator(candidate_fto_code):
            logger.info(
                f"Skipping line profiler for {self.function_to_optimize.function_name} - code contains JIT decorator"
            )
            return {"timings": {}, "unit": 0, "str_out": ""}

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
            self.write_code_and_helpers(
                self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
            )
        if isinstance(line_profile_results, TestResults) and not line_profile_results.test_results:
            logger.warning(
                f"Timeout occurred while running line profiler for original function {self.function_to_optimize.function_name}"
            )
            return {"timings": {}, "unit": 0, "str_out": ""}
        if line_profile_results["str_out"] == "":
            logger.warning(
                f"Couldn't run line profiler for original function {self.function_to_optimize.function_name}"
            )
        return line_profile_results
