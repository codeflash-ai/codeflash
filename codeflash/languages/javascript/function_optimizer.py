from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash.models.models import TestingMode, TestResults
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.equivalence import compare_test_results

if TYPE_CHECKING:
    from codeflash.models.models import (
        CodeOptimizationContext,
        CodeStringsMarkdown,
        CoverageData,
        OriginalCodeBaseline,
        TestDiff,
    )


class JavaScriptFunctionOptimizer(FunctionOptimizer):
    def compare_candidate_results(
        self,
        baseline_results: OriginalCodeBaseline,
        candidate_behavior_results: TestResults,
        optimization_candidate_index: int,
    ) -> tuple[bool, list[TestDiff]]:
        original_sqlite = get_run_tmp_file(Path("test_return_values_0.sqlite"))
        candidate_sqlite = get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite"))

        if original_sqlite.exists() and candidate_sqlite.exists():
            js_root = self.test_cfg.js_project_root or self.args.project_root
            match, diffs = self.language_support.compare_test_results(
                original_sqlite, candidate_sqlite, project_root=js_root
            )
            candidate_sqlite.unlink(missing_ok=True)
        else:
            match, diffs = compare_test_results(
                baseline_results.behavior_test_results, candidate_behavior_results, pass_fail_only=True
            )
        return match, diffs

    def should_skip_sqlite_cleanup(self, testing_type: TestingMode, optimization_iteration: int) -> bool:
        return testing_type == TestingMode.BEHAVIOR or optimization_iteration == 0

    def parse_line_profile_test_results(
        self, line_profiler_output_file: Path | None
    ) -> tuple[TestResults | dict, CoverageData | None]:
        if line_profiler_output_file is None or not line_profiler_output_file.exists():
            return TestResults(test_results=[]), None
        if hasattr(self.language_support, "parse_line_profile_results"):
            return self.language_support.parse_line_profile_results(line_profiler_output_file), None
        return TestResults(test_results=[]), None

    def line_profiler_step(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], candidate_index: int
    ) -> dict:
        if not hasattr(self.language_support, "instrument_source_for_line_profiler"):
            logger.warning(f"Language support for {self.language_support.language} doesn't support line profiling")
            return {"timings": {}, "unit": 0, "str_out": ""}

        try:
            line_profiler_output_path = get_run_tmp_file(Path("line_profiler_output.json"))
            original_source = Path(self.function_to_optimize.file_path).read_text()

            success = self.language_support.instrument_source_for_line_profiler(
                func_info=self.function_to_optimize, line_profiler_output_file=line_profiler_output_path
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

            return self.language_support.parse_line_profile_results(line_profiler_output_path)
        except Exception as e:
            logger.warning(f"Failed to run line profiling: {e}")
            return {"timings": {}, "unit": 0, "str_out": ""}
        finally:
            Path(self.function_to_optimize.file_path).write_text(original_source)

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        optimized_code: CodeStringsMarkdown,
        original_helper_code: dict[Path, str],
    ) -> bool:
        from codeflash.languages.python.static_analysis.code_replacer import replace_function_definitions_for_language

        did_update = False
        read_writable_functions_by_file_path: dict[Path, set[str]] = defaultdict(set)
        read_writable_functions_by_file_path[self.function_to_optimize.file_path].add(
            self.function_to_optimize.qualified_name
        )
        for helper_function in code_context.helper_functions:
            if helper_function.definition_type != "class":
                read_writable_functions_by_file_path[helper_function.file_path].add(helper_function.qualified_name)
        for module_abspath, qualified_names in read_writable_functions_by_file_path.items():
            did_update |= replace_function_definitions_for_language(
                function_names=list(qualified_names),
                optimized_code=optimized_code,
                module_abspath=module_abspath,
                project_root_path=self.project_root,
                function_to_optimize=self.function_to_optimize,
            )
        return did_update
