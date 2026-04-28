from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.languages.base import LanguageSupport
from codeflash.languages.golang.comparator import compare_test_results as _compare_results
from codeflash.languages.golang.config import detect_go_project, detect_go_version
from codeflash.languages.golang.context import extract_code_context as _extract_context
from codeflash.languages.golang.context import find_helper_functions as _find_helpers
from codeflash.languages.golang.discovery import discover_functions_from_source
from codeflash.languages.golang.formatter import format_go_code, normalize_go_code
from codeflash.languages.golang.parser import GoAnalyzer
from codeflash.languages.golang.replacement import add_global_declarations as _add_globals
from codeflash.languages.golang.replacement import remove_test_functions as _remove_tests
from codeflash.languages.golang.replacement import replace_function as _replace_func
from codeflash.languages.golang.test_discovery import discover_tests as _discover_tests
from codeflash.languages.golang.test_runner import parse_test_results as _parse_results
from codeflash.languages.golang.test_runner import run_behavioral_tests as _run_behavioral
from codeflash.languages.golang.test_runner import run_benchmarking_tests as _run_benchmarking
from codeflash.languages.language_enum import Language
from codeflash.languages.registry import register_language

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeflash.languages.base import (
        CodeContext,
        DependencyResolver,
        FunctionFilterCriteria,
        HelperFunction,
        ReferenceInfo,
        TestInfo,
    )
    from codeflash.models.function_types import FunctionToOptimize
    from codeflash.models.models import GeneratedTestsList, InvocationId

logger = logging.getLogger(__name__)


@register_language
class GoSupport(LanguageSupport):
    def __init__(self) -> None:
        self._analyzer = GoAnalyzer()
        self._go_version: str | None = None
        self._go_version_detected = False

    @property
    def language(self) -> Language:
        return Language.GO

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".go",)

    @property
    def default_file_extension(self) -> str:
        return ".go"

    @property
    def test_framework(self) -> str:
        return "go-test"

    @property
    def comment_prefix(self) -> str:
        return "//"

    @property
    def dir_excludes(self) -> frozenset[str]:
        return frozenset({"vendor", "testdata", ".git", "node_modules"})

    @property
    def language_version(self) -> str | None:
        if not self._go_version_detected:
            self._go_version = detect_go_version()
            self._go_version_detected = True
        return self._go_version

    @property
    def valid_test_frameworks(self) -> tuple[str, ...]:
        return ("go-test",)

    @property
    def test_result_serialization_format(self) -> str:
        return "json"

    @property
    def function_optimizer_class(self) -> type:
        from codeflash.languages.golang.function_optimizer import GoFunctionOptimizer

        return GoFunctionOptimizer

    def discover_functions(
        self, source: str, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionToOptimize]:
        return discover_functions_from_source(source, file_path, filter_criteria, self._analyzer)

    def discover_tests(
        self, test_root: Path, source_functions: Sequence[FunctionToOptimize]
    ) -> dict[str, list[TestInfo]]:
        return _discover_tests(test_root, source_functions)

    def validate_syntax(self, source: str, file_path: Path | None = None) -> bool:
        return self._analyzer.validate_syntax(source)

    def parse_test_xml(
        self, test_xml_file_path: Path, test_files: Any, test_config: Any, run_result: Any = None
    ) -> Any:
        from codeflash.languages.golang.parse import parse_go_test_output

        return parse_go_test_output(test_xml_file_path, test_files, test_config, run_result)

    def extract_code_context(self, function: FunctionToOptimize, project_root: Path, module_root: Path) -> CodeContext:
        return _extract_context(function, project_root, module_root, self._analyzer)

    def find_helper_functions(self, function: FunctionToOptimize, project_root: Path) -> list[HelperFunction]:
        try:
            source = function.file_path.read_text(encoding="utf-8")
        except Exception:
            return []
        return _find_helpers(source, function, self._analyzer)

    def find_references(
        self, function: FunctionToOptimize, project_root: Path, tests_root: Path | None = None, max_files: int = 100
    ) -> list[ReferenceInfo]:
        return []

    def replace_function(self, source: str, function: FunctionToOptimize, new_source: str) -> str:
        return _replace_func(source, function, new_source, self._analyzer)

    def format_code(self, source: str, file_path: Path | None = None) -> str:
        return format_go_code(source, file_path)

    def normalize_code(self, source: str) -> str:
        return normalize_go_code(source)

    def add_global_declarations(self, optimized_code: str, original_source: str, module_abspath: Path) -> str:
        return _add_globals(optimized_code, original_source, self._analyzer)

    def get_module_path(self, source_file: Path, project_root: Path, tests_root: Path | None = None) -> str:
        return str(source_file)

    def prepare_module(
        self, module_code: str, module_path: Path, project_root: Path
    ) -> tuple[dict[Path, Any], None] | None:
        from codeflash.models.models import ValidCode

        if not self._analyzer.validate_syntax(module_code):
            return None
        validated: dict[Path, ValidCode] = {
            module_path: ValidCode(source_code=module_code, normalized_code=normalize_go_code(module_code))
        }
        return validated, None

    def setup_test_config(self, test_cfg: Any, file_path: Path, current_worktree: Path | None = None) -> bool:
        _ = file_path, current_worktree
        project_root = getattr(test_cfg, "project_root_path", Path.cwd())
        config = detect_go_project(project_root)
        if config is not None and config.go_version:
            self._go_version = config.go_version
            self._go_version_detected = True
        return True

    def detect_module_system(self, project_root: Path, source_file: Path | None = None) -> str | None:
        return None

    def run_behavioral_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        enable_coverage: bool = False,
        candidate_index: int = 0,
    ) -> tuple[Path, Any, Path | None, Path | None]:
        return _run_behavioral(test_paths, test_env, cwd, timeout, project_root, enable_coverage, candidate_index)

    def run_benchmarking_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        min_loops: int = 5,
        max_loops: int = 100_000,
        target_duration_seconds: float = 10.0,
        inner_iterations: int = 100,
    ) -> tuple[Path, Any]:
        return _run_benchmarking(
            test_paths,
            test_env,
            cwd,
            timeout,
            project_root,
            min_loops,
            max_loops,
            target_duration_seconds,
            inner_iterations,
        )

    def generate_concolic_tests(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        return {}, ""

    def run_line_profile_tests(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def compare_test_results(
        self,
        original_results_path: Path,
        candidate_results_path: Path,
        project_root: Path | None = None,
        project_classpath: str | None = None,
    ) -> tuple[bool, list[Any]]:
        return _compare_results(original_results_path, candidate_results_path, project_root, project_classpath)

    def instrument_for_behavior(self, source: str, functions: Sequence[FunctionToOptimize]) -> str:
        return source

    def instrument_for_benchmarking(self, test_source: str, target_function: FunctionToOptimize) -> str:
        from codeflash.languages.golang.instrumentation import convert_tests_to_benchmarks

        func_name = target_function.function_name if target_function else ""
        return convert_tests_to_benchmarks(test_source, func_name)

    def instrument_existing_test(
        self, test_path: Path, call_positions: Any, function_to_optimize: Any, tests_project_root: Path, mode: str
    ) -> tuple[bool, str | None]:
        _ = call_positions, tests_project_root
        try:
            source = test_path.read_text(encoding="utf-8")
        except Exception:
            return False, None
        if mode == "performance":
            from codeflash.languages.golang.instrumentation import convert_tests_to_benchmarks

            func_name = function_to_optimize.function_name if function_to_optimize else ""
            source = convert_tests_to_benchmarks(source, func_name)
        return True, source

    def postprocess_generated_tests(
        self, generated_tests: GeneratedTestsList, test_framework: str, project_root: Path, source_file_path: Path
    ) -> GeneratedTestsList:
        _ = test_framework, project_root, source_file_path
        return generated_tests

    def process_generated_test_strings(
        self,
        generated_test_source: str,
        instrumented_behavior_test_source: str,
        instrumented_perf_test_source: str,
        function_to_optimize: Any,
        test_path: Path,
        test_cfg: Any,
        project_module_system: str | None,
    ) -> tuple[str, str, str]:
        _ = test_path, test_cfg, project_module_system
        from codeflash.languages.golang.instrumentation import convert_tests_to_benchmarks

        func_name = function_to_optimize.function_name if function_to_optimize else ""
        instrumented_perf_test_source = convert_tests_to_benchmarks(instrumented_perf_test_source, func_name)
        return generated_test_source, instrumented_behavior_test_source, instrumented_perf_test_source

    def load_coverage(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def get_test_file_suffix(self) -> str:
        return "_test.go"

    def resolve_test_file_from_class_path(self, test_class_path: str, base_dir: Path) -> Path | None:
        return None

    def resolve_test_module_path_for_pr(
        self, test_module_path: str, tests_project_rootdir: Path, non_generated_tests: set[Path]
    ) -> Path | None:
        return None

    def find_test_root(self, project_root: Path) -> Path | None:
        return project_root

    def get_runtime_files(self) -> list[Path]:
        return []

    def ensure_runtime_environment(self, project_root: Path) -> bool:
        return detect_go_version() is not None

    def create_dependency_resolver(self, project_root: Path) -> DependencyResolver | None:
        return None

    def adjust_test_config_for_discovery(self, test_cfg: Any) -> None:
        pass

    def add_runtime_comments(
        self, test_source: str, original_runtimes: dict[str, Any], optimized_runtimes: dict[str, Any]
    ) -> str:
        return test_source

    def remove_test_functions(self, test_source: str, functions_to_remove: list[str]) -> str:
        return _remove_tests(test_source, functions_to_remove, self._analyzer)

    def add_runtime_comments_to_generated_tests(
        self,
        generated_tests: GeneratedTestsList,
        original_runtimes: dict[InvocationId, list[int]],
        optimized_runtimes: dict[InvocationId, list[int]],
        tests_project_rootdir: Path | None = None,
    ) -> GeneratedTestsList:
        _ = original_runtimes, optimized_runtimes, tests_project_rootdir
        return generated_tests

    def remove_test_functions_from_generated_tests(
        self, generated_tests: GeneratedTestsList, functions_to_remove: list[str]
    ) -> GeneratedTestsList:
        from codeflash.models.models import GeneratedTests

        updated_tests: list[GeneratedTests] = []
        for test in generated_tests.generated_tests:
            updated_tests.append(
                GeneratedTests(
                    generated_original_test_source=self.remove_test_functions(
                        test.generated_original_test_source, functions_to_remove
                    ),
                    instrumented_behavior_test_source=test.instrumented_behavior_test_source,
                    instrumented_perf_test_source=test.instrumented_perf_test_source,
                    behavior_file_path=test.behavior_file_path,
                    perf_file_path=test.perf_file_path,
                )
            )
        return type(generated_tests)(generated_tests=updated_tests)

    def get_test_dir_for_source(self, test_dir: Path, source_file: Path | None = None) -> Path | None:
        if source_file is not None:
            return source_file.parent
        return test_dir

    def parse_test_results(self, json_output_path: Path, stdout: str) -> Any:
        return _parse_results(json_output_path, stdout)
