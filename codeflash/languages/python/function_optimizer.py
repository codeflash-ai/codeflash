from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash.languages.python.context.unused_definition_remover import (
    detect_unused_helper_functions,
    revert_unused_helper_functions,
)
from codeflash.languages.python.optimizer import resolve_python_function_ast
from codeflash.languages.python.static_analysis.code_extractor import get_opt_review_metrics, is_numerical_code
from codeflash.languages.python.static_analysis.code_replacer import (
    add_custom_marker_to_all_tests,
    modify_autouse_fixture,
    replace_function_definitions_in_module,
)
from codeflash.languages.python.static_analysis.line_profile_utils import add_decorator_imports, contains_jit_decorator
from codeflash.models.models import TestingMode, TestResults
from codeflash.optimization.function_optimizer import FunctionOptimizer

if TYPE_CHECKING:
    from codeflash.languages.base import Language
    from codeflash.models.models import CodeOptimizationContext, CodeStringsMarkdown


class PythonFunctionOptimizer(FunctionOptimizer):
    def _resolve_function_ast(
        self, source_code: str, function_name: str, parents: list
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        original_module_ast = _cached_parse_source(source_code)
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
            if helper_function.definition_type != "class":
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


@lru_cache(maxsize=128)
def _cached_parse_source(source_code: str) -> ast.Module:
    return ast.parse(source_code)
