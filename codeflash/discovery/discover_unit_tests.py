from __future__ import annotations

import os
import pickle
import re
import subprocess
import unittest
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import jedi
from pydantic.dataclasses import dataclass
from pytest import ExitCode

from codeflash.cli_cmds.console import console, logger, test_files_progress_bar
from codeflash.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
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


def discover_unit_tests(
    cfg: TestConfig, discover_only_these_tests: list[Path] | None = None
) -> dict[str, list[FunctionCalledInTest]]:
    framework_strategies: dict[str, Callable] = {
        "pytest": discover_tests_pytest,
        "unittest": discover_tests_unittest,
    }
    strategy = framework_strategies.get(cfg.test_framework, None)
    if not strategy:
        error_message = f"Unsupported test framework: {cfg.test_framework}"
        raise ValueError(error_message)

    return strategy(cfg, discover_only_these_tests)


def discover_tests_pytest(
    cfg: TestConfig, discover_only_these_tests: list[Path] | None = None
) -> dict[str, list[FunctionCalledInTest]]:
    tests_root = cfg.tests_root
    project_root = cfg.project_root_path

    tmp_pickle_path = get_run_tmp_file("collected_tests.pkl")
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
                f"Failed to collect tests. Pytest Exit code: {exitcode}={ExitCode(exitcode).name}\n {error_section}"
            )

        elif 0 <= exitcode <= 5:
            logger.warning(
                f"Failed to collect tests. Pytest Exit code: {exitcode}={ExitCode(exitcode).name}"
            )
        else:
            logger.warning(f"Failed to collect tests. Pytest Exit code: {exitcode}")
        console.rule()
    else:
        logger.debug(f"Pytest collection exit code: {exitcode}")
    if pytest_rootdir is not None:
        cfg.tests_project_rootdir = Path(pytest_rootdir)
    file_to_test_map = defaultdict(list)
    for test in tests:
        if (
            discover_only_these_tests
            and Path(test["test_file"]) not in discover_only_these_tests
        ):
            continue

        # Process discovered tests
        if "__replay_test" in test["test_file"]:
            test_type = TestType.REPLAY_TEST
        elif "test_concolic_coverage" in test["test_file"]:
            test_type = TestType.CONCOLIC_COVERAGE_TEST
        else:
            test_type = TestType.EXISTING_UNIT_TEST

        # Apply the string processing logic here before creating the TestsInFile object
        test_function = test["test_function"]

        # Process parameterized pytest tests
        if "[" in test_function:
            function_name = PYTEST_PARAMETERIZED_TEST_NAME_REGEX.split(test_function)[0]
            parameters = PYTEST_PARAMETERIZED_TEST_NAME_REGEX.split(test_function)[1]
            test_function = f"{function_name}[{parameters}]"
        # Process unittest parameterized tests
        elif UNITTEST_PARAMETERIZED_TEST_NAME_REGEX.match(test_function):
            base_name = UNITTEST_STRIP_NUMBERED_SUFFIX_REGEX.sub("", test_function)
            test_function = f"{base_name}[{test_function}]"

        test_obj = TestsInFile(
            test_file=Path(test["test_file"]),
            test_class=test["test_class"],
            test_function=test_function,
            test_type=test_type,
        )

        file_to_test_map[test_obj.test_file].append(test_obj)
    return process_test_files(file_to_test_map, cfg)


def discover_tests_unittest(
    cfg: TestConfig, discover_only_these_tests: list[str] | None = None
) -> dict[str, list[FunctionCalledInTest]]:
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
            discover_only_these_tests
            and str(_test_module_path) not in discover_only_these_tests
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
                            logger.warning(
                                f"Didn't find tests for {test_2}"
                            )  # it goes deeper?
                            continue
                        details = get_test_details(test_2)
                        if details is not None:
                            file_to_test_map[str(details.test_file)].append(details)
                else:
                    details = get_test_details(test)
                    if details is not None:
                        file_to_test_map[str(details.test_file)].append(details)
    return process_test_files(file_to_test_map, cfg)


def discover_parameters_unittest(function_name: str) -> tuple[bool, str, str | None]:
    function_name = function_name.split("_")
    if len(function_name) > 1 and function_name[-1].isdigit():
        return True, "_".join(function_name[:-1]), function_name[-1]

    return False, function_name, None


def process_test_files(
    file_to_test_map: dict[str, list[TestsInFile]], cfg: TestConfig
) -> dict[str, list[FunctionCalledInTest]]:
    project_root_path = cfg.project_root_path
    test_framework = cfg.test_framework
    function_to_test_map = defaultdict(set)
    jedi_project = jedi.Project(path=project_root_path)
    goto_cache = {}

    with test_files_progress_bar(
        total=len(file_to_test_map), description="Processing test files"
    ) as (progress, task_id):

        for test_file, test_functions in file_to_test_map.items():
            try:
                script = jedi.Script(path=test_file, project=jedi_project)
                function_to_tests_in_file = defaultdict(set)
                all_names = script.get_names(all_scopes=True, references=True)
                all_defs = script.get_names(all_scopes=True, definitions=True)

                top_level_functions = {
                    name.name: name for name in all_defs if name.type == "function"
                }
                top_level_classes = {
                    name.name: name for name in all_defs if name.type == "class"
                }
            except Exception as e:
                logger.debug(f"Failed to get jedi script for {test_file}: {e}")
                progress.advance(task_id)
                continue

            if test_framework == "pytest":
                for function in test_functions:
                    if function.test_function in top_level_functions:
                        function_to_tests_in_file[function.test_function].add(function)

            # Unittest will be deprecated soon
            elif test_framework == "unittest":
                functions_to_search = [elem.test_function for elem in test_functions]
                test_suites = {elem.test_class for elem in test_functions}

                matching_names = test_suites & top_level_classes.keys()
                for matched_name in matching_names:
                    for def_name in all_defs:
                        if (
                            def_name.type == "function"
                            and def_name.full_name is not None
                            and f".{matched_name}." in def_name.full_name
                        ):
                            for function in functions_to_search:
                                (is_parameterized, new_function, parameters) = (
                                    discover_parameters_unittest(function)
                                )
                                if is_parameterized and new_function == def_name.name:
                                    function_to_tests_in_file[def_name.name].add(TestsInFile(test_file=Path(test_file),
                                                                               test_class=matched_name,
                                                                               test_function=f"{def_name.name}_{parameters}",
                                                                               test_type=test_functions[0].test_type))
                                elif function == def_name.name:
                                    function_to_tests_in_file[def_name.name].add(TestsInFile(test_file=Path(test_file),
                                                                               test_class=matched_name,
                                                                               test_function=f"{def_name.name}",
                                                                               test_type=test_functions[0].test_type))

            for name in all_names:
                if name.full_name is None:
                    continue
                m = FUNCTION_NAME_REGEX.search(name.full_name)
                if not m:
                    continue

                scope = m.group(1)
                if scope not in function_to_tests_in_file:
                    continue

                cache_key = (name.full_name, name.module_name)
                try:
                    if cache_key in goto_cache:
                        definition = goto_cache[cache_key]
                    else:
                        definition = name.goto(
                            follow_imports=True, follow_builtin_imports=False
                        )
                        goto_cache[cache_key] = definition
                except Exception as e:
                    logger.debug(str(e))
                    continue

                if not definition or definition[0].type != "function":
                    continue

                definition_path = str(definition[0].module_path)
                if (
                    definition_path.startswith(str(project_root_path) + os.sep)
                    and definition[0].module_name != name.module_name
                    and definition[0].full_name is not None
                ):
                    for tests_in_file in function_to_tests_in_file[scope]:
                        full_name_without_module_prefix = definition[
                            0
                        ].full_name.replace(definition[0].module_name + ".", "", 1)
                        qualified_name_with_modules_from_root = f"{module_name_from_file_path(definition[0].module_path, project_root_path)}.{full_name_without_module_prefix}"

                        function_to_test_map[qualified_name_with_modules_from_root].add(
                            FunctionCalledInTest(
                                tests_in_file=tests_in_file,
                                position=CodePosition(
                                    line_no=name.line, col_no=name.column
                                ),
                            )
                        )

            progress.advance(task_id)
    return {function: list(tests) for function, tests in function_to_test_map.items()}
