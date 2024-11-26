from __future__ import annotations

import os
import pickle
import re
import subprocess
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import jedi
from pydantic.dataclasses import dataclass
from pytest import ExitCode

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path
from codeflash.models.models import CodePosition, FunctionCalledInTest, TestsInFile
from codeflash.verification.test_results import TestType

if TYPE_CHECKING:
    from codeflash.verification.verification_utils import TestConfig


@dataclass(frozen=True)
class TestFunction:
    function_name: str
    test_suite_name: Optional[str]
    parameters: Optional[str]
    test_type: TestType


def discover_unit_tests(
    cfg: TestConfig, discover_only_these_tests: list[Path] | None = None
) -> dict[str, list[FunctionCalledInTest]]:
    framework_strategies: dict[str, Callable] = {"pytest": discover_tests_pytest, "unittest": discover_tests_unittest}
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
    subprocess.run(
        [
            sys.executable,
            Path(__file__).parent / "pytest_new_process_discovery.py",
            str(project_root),
            str(tests_root),
            str(tmp_pickle_path),
        ],
        cwd=project_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with tmp_pickle_path.open(mode="rb") as f:
            exitcode, tests, pytest_rootdir = pickle.load(f)
    except Exception as e:
        logger.exception(f"Failed to discover tests: {e}")
        exitcode = -1
    finally:
        with Path.open(tmp_pickle_path, "w") as f:
            pass
    if exitcode != 0:
        if 0 <= exitcode <= 5:
            logger.warning(f"Failed to collect tests. Pytest Exit code: {exitcode}={ExitCode(exitcode).name}")
        else:
            logger.warning(f"Failed to collect tests. Pytest Exit code: {exitcode}")
    else:
        logger.debug(f"Pytest collection exit code: {exitcode}")
    if pytest_rootdir is not None:
        cfg.tests_project_rootdir = Path(pytest_rootdir)
    file_to_test_map = defaultdict(list)
    for test in tests:
        if "__replay_test" in test["test_file"]:
            test_type = TestType.REPLAY_TEST
        elif "test_concolic_coverage" in test["test_file"]:
            test_type = TestType.CONCOLIC_COVERAGE_TEST
        else:
            test_type = TestType.EXISTING_UNIT_TEST

        test_obj = TestsInFile(
            test_file=test["test_file"],
            test_class=test["test_class"],
            test_function=test["test_function"],
            test_suite=None,
            test_type=test_type,
        )
        if discover_only_these_tests and test_obj.test_file not in discover_only_these_tests:
            continue
        file_to_test_map[test_obj.test_file].append(test_obj)
    # Within these test files, find the project functions they are referring to and return their names/locations
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
            test_suite=_test_suite_name,
            test_function=_test_function,
            test_type=test_type,
            test_class=None,  # TODO: Validate if it is correct to set test_class to None
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
    function_to_test_map = defaultdict(list)
    jedi_project = jedi.Project(path=project_root_path)

    for test_file, functions in file_to_test_map.items():
        script = jedi.Script(path=test_file, project=jedi_project)
        test_functions = set()
        top_level_names = script.get_names()
        all_names = script.get_names(all_scopes=True, references=True)
        all_defs = script.get_names(all_scopes=True, definitions=True)

        for name in top_level_names:
            if test_framework == "pytest":
                functions_to_search = [elem.test_function for elem in functions]
                for i, function in enumerate(functions_to_search):
                    if "[" in function:
                        function_name = re.split(r"[\[\]]", function)[0]
                        parameters = re.split(r"[\[\]]", function)[1]
                        if name.name == function_name and name.type == "function":
                            test_functions.add(TestFunction(name.name, None, parameters, functions[i].test_type))
                    elif name.name == function and name.type == "function":
                        test_functions.add(TestFunction(name.name, None, None, functions[i].test_type))
                        break
            if test_framework == "unittest":
                functions_to_search = [elem.test_function for elem in functions]
                test_suites = [elem.test_suite for elem in functions]

                if name.name in test_suites and name.type == "class":
                    for def_name in all_defs:
                        if (
                            def_name.type == "function"
                            and def_name.full_name is not None
                            and f".{name.name}." in def_name.full_name
                        ):
                            for function in functions_to_search:
                                (is_parameterized, new_function, parameters) = discover_parameters_unittest(function)

                                if is_parameterized and new_function == def_name.name:
                                    test_functions.add(
                                        TestFunction(
                                            def_name.name, name.name, parameters, functions[0].test_type
                                        )  # A test file must not have more than one test type
                                    )
                                elif function == def_name.name:
                                    test_functions.add(
                                        TestFunction(def_name.name, name.name, None, functions[0].test_type)
                                    )

        test_functions_list = list(test_functions)
        test_functions_raw = [elem.function_name for elem in test_functions_list]

        for name in all_names:
            if name.full_name is None:
                continue
            m = re.search(r"([^.]+)\." + f"{name.name}$", name.full_name)
            if not m:
                continue
            scope = m.group(1)
            indices = [i for i, x in enumerate(test_functions_raw) if x == scope]
            for index in indices:
                scope_test_function = test_functions_list[index].function_name
                scope_test_suite = test_functions_list[index].test_suite_name
                scope_parameters = test_functions_list[index].parameters
                test_type = test_functions_list[index].test_type
                try:
                    definition = name.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception as e:
                    logger.exception(str(e))
                    continue
                if definition and definition[0].type == "function":
                    definition_path = str(definition[0].module_path)
                    # The definition is part of this project and not defined within the original function
                    if (
                        definition_path.startswith(str(project_root_path) + os.sep)
                        and definition[0].module_name != name.module_name
                        and definition[0].full_name is not None
                    ):
                        if scope_parameters is not None:
                            if test_framework == "pytest":
                                scope_test_function += "[" + scope_parameters + "]"
                            if test_framework == "unittest":
                                scope_test_function += "_" + scope_parameters
                        full_name_without_module_prefix = definition[0].full_name.replace(
                            definition[0].module_name + ".", "", 1
                        )
                        qualified_name_with_modules_from_root = f"{module_name_from_file_path(definition[0].module_path, project_root_path)}.{full_name_without_module_prefix}"
                        function_to_test_map[qualified_name_with_modules_from_root].append(
                            FunctionCalledInTest(
                                tests_in_file=TestsInFile(
                                    test_file=test_file,
                                    test_class=None,
                                    test_function=scope_test_function,
                                    test_suite=scope_test_suite,
                                    test_type=test_type,
                                ),
                                position=CodePosition(line_no=name.line, col_no=name.column),
                            )
                        )
    deduped_function_to_test_map = {}
    for function, tests in function_to_test_map.items():
        deduped_function_to_test_map[function] = list(set(tests))
    return deduped_function_to_test_map
