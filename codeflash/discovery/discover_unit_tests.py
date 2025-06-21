# ruff: noqa: SLF001
from __future__ import annotations

import ast
import hashlib
import os
import pickle
import re
import sqlite3
import subprocess
import unittest
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize

import pytest
from pydantic.dataclasses import dataclass
from rich.panel import Panel
from rich.text import Text

from codeflash.cli_cmds.console import console, logger, test_files_progress_bar
from codeflash.code_utils.code_utils import (
    ImportErrorPattern,
    custom_addopts,
    get_run_tmp_file,
    module_name_from_file_path,
)
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE, codeflash_cache_db
from codeflash.models.models import CodePosition, FunctionCalledInTest, TestsInFile, TestType

if TYPE_CHECKING:
    from codeflash.verification.verification_utils import TestConfig


@dataclass(frozen=True)
class TestFunction:
    function_name: str
    test_class: Optional[str]
    parameters: Optional[str]
    test_type: TestType


ERROR_PATTERN = re.compile(r"={3,}\s*ERRORS\s*={3,}\n([\s\S]*?)(?:={3,}|$)")
PYTEST_PARAMETERIZED_TEST_NAME_REGEX = re.compile(r"[\[\]]")
UNITTEST_PARAMETERIZED_TEST_NAME_REGEX = re.compile(r"^test_\w+_\d+(?:_\w+)*")
UNITTEST_STRIP_NUMBERED_SUFFIX_REGEX = re.compile(r"_\d+(?:_\w+)*$")
FUNCTION_NAME_REGEX = re.compile(r"([^.]+)\.([a-zA-Z0-9_]+)$")


class TestsCache:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(codeflash_cache_db)
        self.cur = self.connection.cursor()

        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS discovered_tests(
                file_path TEXT,
                file_hash TEXT,
                qualified_name_with_modules_from_root TEXT,
                function_name TEXT,
                test_class TEXT,
                test_function TEXT,
                test_type TEXT,
                line_number INTEGER,
                col_number INTEGER
            )
            """
        )
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_discovered_tests_file_path_hash
            ON discovered_tests (file_path, file_hash)
            """
        )
        self._memory_cache = {}

    def insert_test(
        self,
        file_path: str,
        file_hash: str,
        qualified_name_with_modules_from_root: str,
        function_name: str,
        test_class: str,
        test_function: str,
        test_type: TestType,
        line_number: int,
        col_number: int,
    ) -> None:
        test_type_value = test_type.value if hasattr(test_type, "value") else test_type
        self.cur.execute(
            "INSERT INTO discovered_tests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_path,
                file_hash,
                qualified_name_with_modules_from_root,
                function_name,
                test_class,
                test_function,
                test_type_value,
                line_number,
                col_number,
            ),
        )
        self.connection.commit()

    def get_tests_for_file(self, file_path: str, file_hash: str) -> list[FunctionCalledInTest]:
        cache_key = (file_path, file_hash)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        self.cur.execute("SELECT * FROM discovered_tests WHERE file_path = ? AND file_hash = ?", (file_path, file_hash))
        result = [
            FunctionCalledInTest(
                tests_in_file=TestsInFile(
                    test_file=Path(row[0]), test_class=row[4], test_function=row[5], test_type=TestType(int(row[6]))
                ),
                position=CodePosition(line_no=row[7], col_no=row[8]),
            )
            for row in self.cur.fetchall()
        ]
        self._memory_cache[cache_key] = result
        return result

    @staticmethod
    def compute_file_hash(path: str) -> str:
        h = hashlib.sha256(usedforsecurity=False)
        with Path(path).open("rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def close(self) -> None:
        self.cur.close()
        self.connection.close()


class ImportAnalyzer(ast.NodeVisitor):
    """AST-based analyzer to check if any qualified names from function_names_to_find are imported or used in a test file."""

    def __init__(self, function_names_to_find: set[str]) -> None:
        self.function_names_to_find = function_names_to_find
        self.found_any_target_function: bool = False
        self.found_qualified_name = None
        self.imported_modules: set[str] = set()
        self.has_dynamic_imports: bool = False
        self.wildcard_modules: set[str] = set()

        # Precompute function_names for prefix search
        # For prefix match, store mapping from prefix-root to candidates for O(1) matching
        self._exact_names = function_names_to_find
        self._prefix_roots = {}
        for name in function_names_to_find:
            if "." in name:
                root = name.split(".", 1)[0]
                self._prefix_roots.setdefault(root, []).append(name)

    def visit_Import(self, node: ast.Import) -> None:
        """Handle 'import module' statements."""
        if self.found_any_target_function:
            return

        for alias in node.names:
            module_name = alias.asname if alias.asname else alias.name
            self.imported_modules.add(module_name)

            # Check for dynamic import modules
            if alias.name == "importlib":
                self.has_dynamic_imports = True

            # Check if module itself is a target qualified name
            if module_name in self.function_names_to_find:
                self.found_any_target_function = True
                self.found_qualified_name = module_name
                return
            # Check if any target qualified name starts with this module
            for target_func in self.function_names_to_find:
                if target_func.startswith(f"{module_name}."):
                    self.found_any_target_function = True
                    self.found_qualified_name = target_func
                    return

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle 'from module import name' statements."""
        if self.found_any_target_function:
            return

        mod = node.module
        if not mod:
            return

        fnames = self._exact_names
        proots = self._prefix_roots

        for alias in node.names:
            aname = alias.name
            if aname == "*":
                self.wildcard_modules.add(mod)
                continue

            imported_name = alias.asname if alias.asname else aname
            self.imported_modules.add(imported_name)

            # Fast check for dynamic import
            if mod == "importlib" and aname == "import_module":
                self.has_dynamic_imports = True

            qname = f"{mod}.{aname}"

            # Fast exact match check
            if aname in fnames:
                self.found_any_target_function = True
                self.found_qualified_name = aname
                return
            if qname in fnames:
                self.found_any_target_function = True
                self.found_qualified_name = qname
                return

            # Fast prefix match: only for relevant roots
            prefix = qname + "."
            # Only bother if one of the targets startswith the prefix-root
            candidates = proots.get(qname, ())
            for target_func in candidates:
                if target_func.startswith(prefix):
                    self.found_any_target_function = True
                    self.found_qualified_name = target_func
                    return

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Handle attribute access like module.function_name."""
        if self.found_any_target_function:
            return

        # Check if this is accessing a target function through an imported module
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.imported_modules
            and node.attr in self.function_names_to_find
        ):
            self.found_any_target_function = True
            self.found_qualified_name = node.attr
            return

        # Check if this is accessing a target function through a dynamically imported module
        # Only if we've detected dynamic imports are being used
        if self.has_dynamic_imports and node.attr in self.function_names_to_find:
            self.found_any_target_function = True
            self.found_qualified_name = node.attr
            return

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Handle direct name usage like target_function()."""
        if self.found_any_target_function:
            return

        # Check for __import__ usage
        if node.id == "__import__":
            self.has_dynamic_imports = True

        if node.id in self.function_names_to_find:
            self.found_any_target_function = True
            self.found_qualified_name = node.id
            return

        # Check if this name could come from a wildcard import
        for wildcard_module in self.wildcard_modules:
            for target_func in self.function_names_to_find:
                # Check if target_func is from this wildcard module and name matches
                if target_func.startswith(f"{wildcard_module}.") and target_func.endswith(f".{node.id}"):
                    self.found_any_target_function = True
                    self.found_qualified_name = target_func
                    return

        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Override generic_visit to stop traversal if a target function is found."""
        if self.found_any_target_function:
            return
        super().generic_visit(node)


def analyze_imports_in_test_file(test_file_path: Path | str, target_functions: set[str]) -> bool:
    """Analyze a test file to see if it imports any of the target functions."""
    try:
        with Path(test_file_path).open("r", encoding="utf-8") as f:
            source_code = f.read()
        tree = ast.parse(source_code, filename=str(test_file_path))
        analyzer = ImportAnalyzer(target_functions)
        analyzer.visit(tree)
    except (SyntaxError, FileNotFoundError) as e:
        logger.debug(f"Failed to analyze imports in {test_file_path}: {e}")
        return True
    else:
        if analyzer.found_any_target_function:
            logger.debug(f"Test file {test_file_path} imports target function: {analyzer.found_qualified_name}")
            return True
        logger.debug(f"Test file {test_file_path} does not import any target functions.")
        return False


def filter_test_files_by_imports(
    file_to_test_map: dict[Path, list[TestsInFile]], target_functions: set[str]
) -> dict[Path, list[TestsInFile]]:
    """Filter test files based on import analysis to reduce Jedi processing.

    Args:
        file_to_test_map: Original mapping of test files to test functions
        target_functions: Set of function names we're optimizing

    Returns:
        Filtered mapping of test files to test functions

    """
    if not target_functions:
        return file_to_test_map

    logger.debug(f"Target functions for import filtering: {target_functions}")

    filtered_map = {}
    for test_file, test_functions in file_to_test_map.items():
        should_process = analyze_imports_in_test_file(test_file, target_functions)
        if should_process:
            filtered_map[test_file] = test_functions

    logger.debug(
        f"analyzed {len(file_to_test_map)} test files for imports, filtered down to {len(filtered_map)} relevant files"
    )
    return filtered_map


def discover_unit_tests(
    cfg: TestConfig,
    discover_only_these_tests: list[Path] | None = None,
    file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]] | None = None,
) -> tuple[dict[str, set[FunctionCalledInTest]], int]:
    framework_strategies: dict[str, Callable] = {"pytest": discover_tests_pytest, "unittest": discover_tests_unittest}
    strategy = framework_strategies.get(cfg.test_framework, None)
    if not strategy:
        error_message = f"Unsupported test framework: {cfg.test_framework}"
        raise ValueError(error_message)

    # Extract all functions to optimize for import filtering
    functions_to_optimize = None
    if file_to_funcs_to_optimize:
        functions_to_optimize = [func for funcs_list in file_to_funcs_to_optimize.values() for func in funcs_list]
    function_to_tests, num_discovered_tests = strategy(cfg, discover_only_these_tests, functions_to_optimize)
    return function_to_tests, num_discovered_tests


def discover_tests_pytest(
    cfg: TestConfig,
    discover_only_these_tests: list[Path] | None = None,
    functions_to_optimize: list[FunctionToOptimize] | None = None,
) -> tuple[dict[str, set[FunctionCalledInTest]], int]:
    tests_root = cfg.tests_root
    project_root = cfg.project_root_path

    tmp_pickle_path = get_run_tmp_file("collected_tests.pkl")
    with custom_addopts():
        result = subprocess.run(
            [
                SAFE_SYS_EXECUTABLE,
                Path(__file__).parent / "pytest_new_process_discovery.py",
                str(project_root),
                str(tests_root),
                str(tmp_pickle_path),
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    try:
        with tmp_pickle_path.open(mode="rb") as f:
            exitcode, tests, pytest_rootdir = pickle.load(f)
    except Exception as e:
        logger.exception(f"Failed to discover tests: {e}")
        exitcode = -1
    finally:
        tmp_pickle_path.unlink(missing_ok=True)
    if exitcode != 0:
        if exitcode == 2 and "ERROR collecting" in result.stdout:
            # Pattern matches "===== ERRORS =====" (any number of =) and captures everything after
            match = ERROR_PATTERN.search(result.stdout)
            error_section = match.group(1) if match else result.stdout

            logger.warning(
                f"Failed to collect tests. Pytest Exit code: {exitcode}={pytest.ExitCode(exitcode).name}\n {error_section}"
            )
            if "ModuleNotFoundError" in result.stdout:
                match = ImportErrorPattern.search(result.stdout).group()
                panel = Panel(Text.from_markup(f"⚠️  {match} ", style="bold red"), expand=False)
                console.print(panel)

        elif 0 <= exitcode <= 5:
            logger.warning(f"Failed to collect tests. Pytest Exit code: {exitcode}={pytest.ExitCode(exitcode).name}")
        else:
            logger.warning(f"Failed to collect tests. Pytest Exit code: {exitcode}")
        console.rule()
    else:
        logger.debug(f"Pytest collection exit code: {exitcode}")
    if pytest_rootdir is not None:
        cfg.tests_project_rootdir = Path(pytest_rootdir)
    file_to_test_map: dict[Path, list[FunctionCalledInTest]] = defaultdict(list)
    for test in tests:
        if "__replay_test" in test["test_file"]:
            test_type = TestType.REPLAY_TEST
        elif "test_concolic_coverage" in test["test_file"]:
            test_type = TestType.CONCOLIC_COVERAGE_TEST
        else:
            test_type = TestType.EXISTING_UNIT_TEST

        test_obj = TestsInFile(
            test_file=Path(test["test_file"]),
            test_class=test["test_class"],
            test_function=test["test_function"],
            test_type=test_type,
        )
        if discover_only_these_tests and test_obj.test_file not in discover_only_these_tests:
            continue
        file_to_test_map[test_obj.test_file].append(test_obj)
    # Within these test files, find the project functions they are referring to and return their names/locations
    return process_test_files(file_to_test_map, cfg, functions_to_optimize)


def discover_tests_unittest(
    cfg: TestConfig,
    discover_only_these_tests: list[str] | None = None,
    functions_to_optimize: list[FunctionToOptimize] | None = None,
) -> tuple[dict[str, set[FunctionCalledInTest]], int]:
    tests_root: Path = cfg.tests_root
    loader: unittest.TestLoader = unittest.TestLoader()
    tests: unittest.TestSuite = loader.discover(str(tests_root))
    file_to_test_map: defaultdict[str, list[TestsInFile]] = defaultdict(list)

    def get_test_details(_test: unittest.TestCase) -> TestsInFile | None:
        _test_function, _test_module, _test_suite_name = (
            _test._testMethodName,
            _test.__class__.__module__,
            _test.__class__.__qualname__,
        )

        _test_module_path = Path(_test_module.replace(".", os.sep)).with_suffix(".py")
        _test_module_path = tests_root / _test_module_path
        if not _test_module_path.exists() or (
            discover_only_these_tests and str(_test_module_path) not in discover_only_these_tests
        ):
            return None
        if "__replay_test" in str(_test_module_path):
            test_type = TestType.REPLAY_TEST
        elif "test_concolic_coverage" in str(_test_module_path):
            test_type = TestType.CONCOLIC_COVERAGE_TEST
        else:
            test_type = TestType.EXISTING_UNIT_TEST
        return TestsInFile(
            test_file=str(_test_module_path),
            test_function=_test_function,
            test_type=test_type,
            test_class=_test_suite_name,
        )

    for _test_suite in tests._tests:
        for test_suite_2 in _test_suite._tests:
            if not hasattr(test_suite_2, "_tests"):
                logger.warning(f"Didn't find tests for {test_suite_2}")
                continue

            for test in test_suite_2._tests:
                # some test suites are nested, so we need to go deeper
                if not hasattr(test, "_testMethodName") and hasattr(test, "_tests"):
                    for test_2 in test._tests:
                        if not hasattr(test_2, "_testMethodName"):
                            logger.warning(f"Didn't find tests for {test_2}")  # it goes deeper?
                            continue
                        details = get_test_details(test_2)
                        if details is not None:
                            file_to_test_map[str(details.test_file)].append(details)
                else:
                    details = get_test_details(test)
                    if details is not None:
                        file_to_test_map[str(details.test_file)].append(details)
    return process_test_files(file_to_test_map, cfg, functions_to_optimize)


def discover_parameters_unittest(function_name: str) -> tuple[bool, str, str | None]:
    function_name = function_name.split("_")
    if len(function_name) > 1 and function_name[-1].isdigit():
        return True, "_".join(function_name[:-1]), function_name[-1]

    return False, function_name, None


def process_test_files(
    file_to_test_map: dict[Path, list[TestsInFile]],
    cfg: TestConfig,
    functions_to_optimize: list[FunctionToOptimize] | None = None,
) -> tuple[dict[str, set[FunctionCalledInTest]], int]:
    import jedi

    project_root_path = cfg.project_root_path
    test_framework = cfg.test_framework

    if functions_to_optimize:
        target_function_names = {func.qualified_name for func in functions_to_optimize}
        file_to_test_map = filter_test_files_by_imports(file_to_test_map, target_function_names)

    function_to_test_map = defaultdict(set)
    num_discovered_tests = 0
    jedi_project = jedi.Project(path=project_root_path)

    with test_files_progress_bar(total=len(file_to_test_map), description="Processing test files") as (
        progress,
        task_id,
    ):
        for test_file, functions in file_to_test_map.items():
            try:
                script = jedi.Script(path=test_file, project=jedi_project)
                test_functions = set()

                # Single call to get all names with references and definitions
                all_names = script.get_names(all_scopes=True, references=True, definitions=True)

                # Filter once and create lookup dictionaries
                top_level_functions = {}
                top_level_classes = {}
                all_defs = []

                for name in all_names:
                    if name.type == "function":
                        top_level_functions[name.name] = name
                        all_defs.append(name)
                    elif name.type == "class":
                        top_level_classes[name.name] = name
            except Exception as e:
                logger.debug(f"Failed to get jedi script for {test_file}: {e}")
                progress.advance(task_id)
                continue

            if test_framework == "pytest":
                for function in functions:
                    if "[" in function.test_function:
                        function_name = PYTEST_PARAMETERIZED_TEST_NAME_REGEX.split(function.test_function)[0]
                        parameters = PYTEST_PARAMETERIZED_TEST_NAME_REGEX.split(function.test_function)[1]
                        if function_name in top_level_functions:
                            test_functions.add(
                                TestFunction(function_name, function.test_class, parameters, function.test_type)
                            )
                    elif function.test_function in top_level_functions:
                        test_functions.add(
                            TestFunction(function.test_function, function.test_class, None, function.test_type)
                        )
                    elif UNITTEST_PARAMETERIZED_TEST_NAME_REGEX.match(function.test_function):
                        base_name = UNITTEST_STRIP_NUMBERED_SUFFIX_REGEX.sub("", function.test_function)
                        if base_name in top_level_functions:
                            test_functions.add(
                                TestFunction(
                                    function_name=base_name,
                                    test_class=function.test_class,
                                    parameters=function.test_function,
                                    test_type=function.test_type,
                                )
                            )

            elif test_framework == "unittest":
                functions_to_search = [elem.test_function for elem in functions]
                test_suites = {elem.test_class for elem in functions}

                matching_names = test_suites & top_level_classes.keys()
                for matched_name in matching_names:
                    for def_name in all_defs:
                        if (
                            def_name.type == "function"
                            and def_name.full_name is not None
                            and f".{matched_name}." in def_name.full_name
                        ):
                            for function in functions_to_search:
                                (is_parameterized, new_function, parameters) = discover_parameters_unittest(function)

                                if is_parameterized and new_function == def_name.name:
                                    test_functions.add(
                                        TestFunction(
                                            function_name=def_name.name,
                                            test_class=matched_name,
                                            parameters=parameters,
                                            test_type=functions[0].test_type,
                                        )
                                    )
                                elif function == def_name.name:
                                    test_functions.add(
                                        TestFunction(
                                            function_name=def_name.name,
                                            test_class=matched_name,
                                            parameters=None,
                                            test_type=functions[0].test_type,
                                        )
                                    )

            test_functions_by_name = defaultdict(list)
            for func in test_functions:
                test_functions_by_name[func.function_name].append(func)

            test_function_names_set = set(test_functions_by_name.keys())
            relevant_names = []

            names_with_full_name = [name for name in all_names if name.full_name is not None]

            for name in names_with_full_name:
                match = FUNCTION_NAME_REGEX.search(name.full_name)
                if match and match.group(1) in test_function_names_set:
                    relevant_names.append((name, match.group(1)))

            for name, scope in relevant_names:
                try:
                    definition = name.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception as e:
                    logger.debug(str(e))
                    continue

                if not definition or definition[0].type != "function":
                    continue

                definition_obj = definition[0]
                definition_path = str(definition_obj.module_path)

                project_root_str = str(project_root_path)
                if (
                    definition_path.startswith(project_root_str + os.sep)
                    and definition_obj.module_name != name.module_name
                    and definition_obj.full_name is not None
                ):
                    # Pre-compute common values outside the inner loop
                    module_prefix = definition_obj.module_name + "."
                    full_name_without_module_prefix = definition_obj.full_name.replace(module_prefix, "", 1)
                    qualified_name_with_modules_from_root = f"{module_name_from_file_path(definition_obj.module_path, project_root_path)}.{full_name_without_module_prefix}"

                    for test_func in test_functions_by_name[scope]:
                        if test_func.parameters is not None:
                            if test_framework == "pytest":
                                scope_test_function = f"{test_func.function_name}[{test_func.parameters}]"
                            else:  # unittest
                                scope_test_function = f"{test_func.function_name}_{test_func.parameters}"
                        else:
                            scope_test_function = test_func.function_name

                        function_to_test_map[qualified_name_with_modules_from_root].add(
                            FunctionCalledInTest(
                                tests_in_file=TestsInFile(
                                    test_file=test_file,
                                    test_class=test_func.test_class,
                                    test_function=scope_test_function,
                                    test_type=test_func.test_type,
                                ),
                                position=CodePosition(line_no=name.line, col_no=name.column),
                            )
                        )
                        num_discovered_tests += 1

            progress.advance(task_id)

    return dict(function_to_test_map), num_discovered_tests
