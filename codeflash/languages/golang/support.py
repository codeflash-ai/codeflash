from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.languages.golang.config import detect_go_project, detect_go_version
from codeflash.languages.golang.discovery import discover_functions_from_source
from codeflash.languages.golang.parser import GoAnalyzer
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

    def discover_tests(self, test_root: Path, source_functions: Sequence[FunctionToOptimize]) -> dict[str, list[Any]]:
        raise NotImplementedError

    def validate_syntax(self, source: str, file_path: Path | None = None) -> bool:
        return self._analyzer.validate_syntax(source)

    def extract_code_context(self, function: FunctionToOptimize, project_root: Path, module_root: Path) -> CodeContext:
        raise NotImplementedError

    def find_helper_functions(self, function: FunctionToOptimize, project_root: Path) -> list[HelperFunction]:
        raise NotImplementedError

    def find_references(
        self, function: FunctionToOptimize, project_root: Path, tests_root: Path | None = None, max_files: int = 100
    ) -> list[ReferenceInfo]:
        raise NotImplementedError

    def replace_function(self, source: str, function: FunctionToOptimize, new_source: str) -> str:
        raise NotImplementedError

    def format_code(self, source: str, file_path: Path | None = None) -> str:
        raise NotImplementedError

    def normalize_code(self, source: str) -> str:
        raise NotImplementedError

    def add_global_declarations(self, optimized_code: str, original_source: str, module_abspath: Path) -> str:
        raise NotImplementedError

    def prepare_module(
        self, module_code: str, module_path: Path, project_root: Path
    ) -> tuple[dict[Path, Any], None] | None:
        raise NotImplementedError

    def setup_test_config(self, test_cfg: Any) -> None:
        project_root = getattr(test_cfg, "project_root_path", Path.cwd())
        config = detect_go_project(project_root)
        if config is not None and config.go_version:
            self._go_version = config.go_version
            self._go_version_detected = True

    def detect_module_system(self, project_root: Path, source_file: Path | None = None) -> str | None:
        return None

    def run_behavioral_tests(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def run_benchmarking_tests(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def run_line_profile_tests(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def compare_test_results(self, *args: Any, **kwargs: Any) -> tuple[bool, list[Any]]:
        raise NotImplementedError

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
        raise NotImplementedError

    def get_test_dir_for_source(self, test_dir: Path, source_file: Path | None = None) -> Path | None:
        if source_file is not None:
            return source_file.parent
        return test_dir

    def parse_test_results(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
