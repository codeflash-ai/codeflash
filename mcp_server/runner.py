from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.code_utils.shell_utils import make_env_with_project_root

if TYPE_CHECKING:
    import subprocess

    from codeflash.models.models import TestFiles, TestResults
    from codeflash.verification.verification_utils import TestConfig


class TestingMode(str, Enum):
    BEHAVIORAL = "behavioral"
    BENCHMARKING = "benchmarking"


@dataclass(frozen=True)
class _ResolvedTestFile:
    original_path: Path
    effective_path: Path


def build_test_env(project_root: Path) -> dict[str, str]:
    env = make_env_with_project_root(project_root)
    env["CODEFLASH_TEST_ITERATION"] = "0"
    env["CODEFLASH_TRACER_DISABLE"] = "1"
    env["CODEFLASH_LOOP_INDEX"] = "0"
    if "PYTHONHASHSEED" not in env:
        env["PYTHONHASHSEED"] = "0"
    return env


def _build_test_files(test_files: list[_ResolvedTestFile], mode: TestingMode) -> TestFiles:
    from codeflash.models.models import TestFile, TestFiles
    from codeflash.models.test_type import TestType

    test_files_objs = []
    for test_file in test_files:
        effective_path = test_file.effective_path.resolve()
        original_path = test_file.original_path.resolve()
        test_files_objs.append(
            TestFile(
                instrumented_behavior_file_path=effective_path,
                benchmarking_file_path=effective_path if mode == TestingMode.BENCHMARKING else None,
                original_file_path=original_path,
                test_type=TestType.EXISTING_UNIT_TEST,
            )
        )
    return TestFiles(test_files=test_files_objs)


def _build_test_config(project_root: Path, tests_dir: Path | None = None) -> TestConfig:
    from codeflash.verification.verification_utils import TestConfig

    effective_tests_dir = tests_dir or project_root
    return TestConfig(tests_root=effective_tests_dir, project_root_path=project_root, tests_project_rootdir=effective_tests_dir)


def _build_fallback_function_to_optimize(module_path: Path, function_name: str, language: str):
    from codeflash.models.function_types import FunctionParent, FunctionToOptimize

    qualified_name_parts = function_name.split(".")
    simple_name = qualified_name_parts[-1]
    parents = [FunctionParent(name=part, type="ClassDef") for part in qualified_name_parts[:-1]]
    return FunctionToOptimize(
        function_name=simple_name,
        file_path=module_path,
        parents=parents,
        is_method=bool(parents),
        language=language,
    )


def _resolve_function_to_optimize(lang_support: object, module_path: str, function_name: str, language: str):
    from codeflash.languages.base import FunctionFilterCriteria

    source_path = Path(module_path).resolve()
    fallback = _build_fallback_function_to_optimize(source_path, function_name, language)

    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError:
        return fallback

    criteria = FunctionFilterCriteria(require_return=False, require_export=False)
    discovered_functions = lang_support.discover_functions(source, source_path, criteria)
    if not discovered_functions:
        return fallback

    requested_name = function_name.rsplit(".", 1)[-1]

    qualified_matches = [func for func in discovered_functions if func.qualified_name == function_name]
    if len(qualified_matches) == 1:
        return qualified_matches[0]

    top_level_matches = [func for func in discovered_functions if func.qualified_name == requested_name]
    if len(top_level_matches) == 1:
        return top_level_matches[0]

    simple_matches = [func for func in discovered_functions if func.function_name == requested_name]
    if len(simple_matches) == 1:
        return simple_matches[0]

    return fallback


def _find_call_positions(test_path: Path, function_name: str, language: str) -> list:
    """Scan a Python test file's AST to find all call sites of the target function."""
    import ast

    from codeflash.models.models import CodePosition

    if language != "python":
        return []

    try:
        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    positions: list[CodePosition] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Direct call: function_name(...)
        if (isinstance(func, ast.Name) and func.id == function_name) or (isinstance(func, ast.Attribute) and func.attr == function_name):
            positions.append(CodePosition(line_no=node.lineno, col_no=node.col_offset))

    return positions


def _invoke_with_optional_test_framework(run_callable: object, *, test_framework: str | None = None, **kwargs: object):
    try:
        if test_framework is not None and "test_framework" in inspect.signature(run_callable).parameters:
            kwargs["test_framework"] = test_framework
    except (TypeError, ValueError):
        pass
    return run_callable(**kwargs)


def _resolve_test_files(test_file_paths: list[str]) -> list[_ResolvedTestFile]:
    return [_ResolvedTestFile(original_path=Path(path).resolve(), effective_path=Path(path).resolve()) for path in test_file_paths]


def _instrumented_test_path(test_path: Path, language: str, mode: TestingMode) -> Path:
    if language != "java":
        return test_path

    suffix = "__perfinstrumented" if mode == TestingMode.BEHAVIORAL else "__perfonlyinstrumented"
    if test_path.stem.endswith(suffix):
        return test_path
    return test_path.with_name(f"{test_path.stem}{suffix}{test_path.suffix}")


def _reset_java_compilation_cache(language: str) -> None:
    if language != "java":
        return

    from codeflash.languages.java.test_runner import CompilationCache

    CompilationCache.clear()


class _InstrumentedFiles:
    """Context manager that instruments MCP test files and restores originals on exit."""

    def __init__(
        self,
        test_file_paths: list[str],
        function_name: str,
        module_path: str,
        project_root: Path,
        language: str,
        mode: TestingMode,
    ) -> None:
        self.test_file_paths = test_file_paths
        self.function_name = function_name
        self.module_path = module_path
        self.project_root = project_root
        self.language = language
        self.mode = mode
        self._backups: dict[Path, str] = {}
        self._created_files: set[Path] = set()

    def _write_instrumented_source(self, target_path: Path, code: str) -> None:
        if target_path.exists():
            self._backups[target_path] = target_path.read_text(encoding="utf-8")
        else:
            self._created_files.add(target_path)

        target_path.write_text(code, encoding="utf-8")

    def __enter__(self) -> list[_ResolvedTestFile]:
        from codeflash.languages.current import set_current_language
        from codeflash.languages.registry import get_language_support

        set_current_language(self.language)
        lang_support = get_language_support(self.language)

        func_to_optimize = _resolve_function_to_optimize(
            lang_support=lang_support,
            module_path=self.module_path,
            function_name=self.function_name,
            language=self.language,
        )

        instrument_mode = "behavior" if self.mode == TestingMode.BEHAVIORAL else "performance"

        instrumented_paths: list[_ResolvedTestFile] = []
        for test_file in self.test_file_paths:
            test_path = Path(test_file).resolve()
            instrumented_path = _instrumented_test_path(test_path, self.language, self.mode)

            call_positions = _find_call_positions(test_path, func_to_optimize.function_name, self.language)
            if self.language == "python" and not call_positions:
                instrumented_paths.append(_ResolvedTestFile(original_path=test_path, effective_path=test_path))
                continue

            success, code = lang_support.instrument_existing_test(
                test_path=test_path,
                call_positions=call_positions,
                function_to_optimize=func_to_optimize,
                tests_project_root=self.project_root,
                mode=instrument_mode,
            )

            if success and code:
                self._write_instrumented_source(instrumented_path, code)
                instrumented_paths.append(_ResolvedTestFile(original_path=test_path, effective_path=instrumented_path))
            else:
                instrumented_paths.append(_ResolvedTestFile(original_path=test_path, effective_path=test_path))

        return instrumented_paths

    def __exit__(self, *_exc: object) -> None:
        # restore original code for backup files
        for path, original_content in self._backups.items():
            path.write_text(original_content, encoding="utf-8")

        # remove new files
        for path in self._created_files:
            path.unlink(missing_ok=True)
        self._backups.clear()
        self._created_files.clear()


def run_and_parse(
    mode: TestingMode,
    test_files: list[str],
    project_root: Path,
    language: str,
    timeout: int = 300,
    min_loops: int = 1,
    max_loops: int = 1,
    target_duration_seconds: float = 0.5,
    function_name: str | None = None,
    module_path: str | None = None,
    test_framework: str | None = None,
) -> tuple[TestResults, subprocess.CompletedProcess[str]]:
    from codeflash.languages.current import set_current_language
    from codeflash.languages.registry import get_language_support
    from codeflash.verification.parse_test_output import parse_test_results

    set_current_language(language)
    lang_support = get_language_support(language)
    _reset_java_compilation_cache(language)

    test_env = build_test_env(project_root)
    test_config = _build_test_config(project_root)

    def _execute(effective_files: list[_ResolvedTestFile]) -> tuple[TestResults, subprocess.CompletedProcess[str]]:
        test_files_obj = _build_test_files(effective_files, mode)

        if mode == TestingMode.BEHAVIORAL:
            result_file_path, run_result, _, _ = _invoke_with_optional_test_framework(
                lang_support.run_behavioral_tests,
                test_framework=test_framework,
                test_paths=test_files_obj,
                test_env=test_env,
                cwd=project_root,
                timeout=timeout,
                project_root=project_root,
            )
        else:
            result_file_path, run_result = _invoke_with_optional_test_framework(
                lang_support.run_benchmarking_tests,
                test_framework=test_framework,
                test_paths=test_files_obj,
                test_env=test_env,
                cwd=project_root,
                timeout=timeout,
                project_root=project_root,
                min_loops=min_loops,
                max_loops=max_loops,
                target_duration_seconds=target_duration_seconds,
            )

        test_results, _ = parse_test_results(
            test_xml_path=result_file_path,
            test_files=test_files_obj,
            test_config=test_config,
            optimization_iteration=0,
            function_name=function_name,
            source_file=Path(module_path) if module_path else None,
            coverage_database_file=None,
            coverage_config_file=None,
            run_result=run_result,
        )

        return test_results, run_result

    if function_name and module_path:
        with _InstrumentedFiles(
            test_file_paths=test_files,
            function_name=function_name,
            module_path=module_path,
            project_root=project_root,
            language=language,
            mode=mode,
        ) as effective_files:
            return _execute(effective_files)
    else:
        return _execute(_resolve_test_files(test_files))
