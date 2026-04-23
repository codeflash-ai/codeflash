from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

logger = logging.getLogger(__name__)


@register_language
class GoSupport:
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
        raise NotImplementedError

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

    def prepare_module(
        self, module_code: str, module_path: Path, project_root: Path
    ) -> tuple[dict[Path, Any], None] | None:
        if not self._analyzer.validate_syntax(module_code):
            return None
        return {module_path: module_code}, None

    def setup_test_config(self, test_cfg: Any) -> None:
        project_root = getattr(test_cfg, "project_root_path", Path.cwd())
        config = detect_go_project(project_root)
        if config is not None and config.go_version:
            self._go_version = config.go_version
            self._go_version_detected = True

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
        return test_source

    def instrument_existing_test(self, *args: Any, **kwargs: Any) -> tuple[bool, str | None]:
        raise NotImplementedError

    def postprocess_generated_tests(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def process_generated_test_strings(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def load_coverage(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def get_test_file_suffix(self) -> str:
        return "_test.go"

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

    def get_test_dir_for_source(self, test_dir: Path, source_file: Path | None = None) -> Path | None:
        if source_file is not None:
            return source_file.parent
        return test_dir

    def parse_test_results(self, json_output_path: Path, stdout: str) -> Any:
        return _parse_results(json_output_path, stdout)
