from __future__ import annotations

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


def build_test_env(project_root: Path) -> dict[str, str]:
    env = make_env_with_project_root(project_root)
    env["CODEFLASH_TEST_ITERATION"] = "0"
    env["CODEFLASH_TRACER_DISABLE"] = "1"
    env["CODEFLASH_LOOP_INDEX"] = "0"
    if "PYTHONHASHSEED" not in env:
        env["PYTHONHASHSEED"] = "0"
    return env


def _build_test_files(test_file_paths: list[str], mode: TestingMode) -> TestFiles:
    from codeflash.models.models import TestFile, TestFiles
    from codeflash.models.test_type import TestType

    test_files_objs = []
    for path_str in test_file_paths:
        p = Path(path_str).resolve()
        test_files_objs.append(
            TestFile(
                instrumented_behavior_file_path=p,
                benchmarking_file_path=p if mode == TestingMode.BENCHMARKING else None,
                original_file_path=p,
                test_type=TestType.EXISTING_UNIT_TEST,
            )
        )
    return TestFiles(test_files=test_files_objs)


def _build_test_config(project_root: Path, tests_dir: Path | None = None) -> TestConfig:
    from codeflash.verification.verification_utils import TestConfig

    effective_tests_dir = tests_dir or project_root
    return TestConfig(tests_root=effective_tests_dir, project_root_path=project_root, tests_project_rootdir=effective_tests_dir)


def _find_call_positions(test_path: Path, function_name: str) -> list:
    """Scan a test file's AST to find all call sites of the target function."""
    import ast

    from codeflash.models.models import CodePosition

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


class _InstrumentedFiles:
    """Context manager that instruments test files in-place and restores originals on exit."""

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

    def __enter__(self) -> list[str]:
        from codeflash.languages.current import set_current_language
        from codeflash.languages.registry import get_language_support
        from codeflash.models.function_types import FunctionToOptimize

        set_current_language(self.language)
        lang_support = get_language_support(self.language)

        func_to_optimize = FunctionToOptimize(
            function_name=self.function_name,
            file_path=Path(self.module_path),
            parents=(),
            qualified_name=self.function_name,
        )

        instrument_mode = "behavior" if self.mode == TestingMode.BEHAVIORAL else "performance"

        instrumented_paths: list[str] = []
        for test_file in self.test_file_paths:
            test_path = Path(test_file).resolve()

            call_positions = _find_call_positions(test_path, self.function_name)
            if not call_positions:
                instrumented_paths.append(test_file)
                continue

            success, code = lang_support.instrument_existing_test(
                test_path=test_path,
                call_positions=call_positions,
                function_to_optimize=func_to_optimize,
                tests_project_root=self.project_root,
                mode=instrument_mode,
            )

            if success and code:
                self._backups[test_path] = test_path.read_text(encoding="utf-8")
                test_path.write_text(code, encoding="utf-8")
                instrumented_paths.append(str(test_path))
            else:
                instrumented_paths.append(test_file)

        return instrumented_paths

    def __exit__(self, *_exc: object) -> None:
        for path, original_content in self._backups.items():
            path.write_text(original_content, encoding="utf-8")
        self._backups.clear()


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
) -> tuple[TestResults, subprocess.CompletedProcess[str]]:
    from codeflash.languages.current import set_current_language
    from codeflash.languages.registry import get_language_support
    from codeflash.verification.parse_test_output import parse_test_results

    set_current_language(language)
    lang_support = get_language_support(language)

    test_env = build_test_env(project_root)
    test_config = _build_test_config(project_root)

    def _execute(effective_files: list[str]) -> tuple[TestResults, subprocess.CompletedProcess[str]]:
        test_files_obj = _build_test_files(effective_files, mode)

        if mode == TestingMode.BEHAVIORAL:
            result_file_path, run_result, _, _ = lang_support.run_behavioral_tests(
                test_paths=test_files_obj, test_env=test_env, cwd=project_root, timeout=timeout, project_root=project_root
            )
        else:
            result_file_path, run_result = lang_support.run_benchmarking_tests(
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
        return _execute(test_files)
