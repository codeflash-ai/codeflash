import logging
import os
import re
import shlex
import unittest
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple

import jedi
from pydantic.dataclasses import dataclass

from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.verification.test_results import TestType
from codeflash.verification.verification_utils import TestConfig


@dataclass(frozen=True)
class TestsInFile:
    test_file: str
    test_class: Optional[str]  # This might be unused...
    test_function: str
    test_suite: Optional[str]
    test_type: TestType


@dataclass(frozen=True)
class CodePosition:
    line_no: int
    col_no: int


@dataclass(frozen=True)
class FunctionCalledInTest:
    test_file: str
    test_class: Optional[str]  # This might be unused...
    test_function: str
    test_suite: Optional[str]
    test_type: TestType
    position: CodePosition


@dataclass(frozen=True)
class TestFunction:
    function_name: str
    test_suite_name: Optional[str]
    parameters: Optional[str]
    test_type: TestType


def discover_unit_tests(
    cfg: TestConfig,
    discover_only_these_tests: Optional[List[str]] = None,
) -> Dict[str, List[FunctionCalledInTest]]:
    test_frameworks = {
        "pytest": discover_tests_pytest,
        "unittest": discover_tests_unittest,
    }
    discover_tests = test_frameworks.get(cfg.test_framework)
    if discover_tests is None:
        raise ValueError(f"Unsupported test framework: {cfg.test_framework}")
    return discover_tests(cfg, discover_only_these_tests)


def run_pytest_discovery_new_process(queue: Queue, cwd: str, tests_root: str) -> Tuple[int, List]:
    import pytest

    os.chdir(cwd)
    collected_tests = []
    tests = []

    class PytestCollectionPlugin:
        def pytest_collection_finish(self, session):
            collected_tests.extend(session.items)

    try:
        exitcode = pytest.main(
            [tests_root, "--collect-only", "-pno:terminal", "-m", "not skip"],
            plugins=[PytestCollectionPlugin()],
        )
    except Exception as e:
        logging.exception(f"Failed to collect tests: {e!s}")
        exitcode = -1
        queue.put((exitcode, tests))
    tests = parse_pytest_collection_results(collected_tests)
    queue.put((exitcode, tests))


def parse_pytest_collection_results(
    pytest_tests: str,
) -> List[TestsInFile]:
    test_results = []
    for test in pytest_tests:
        test_class = None
        test_file_path = str(test.path)
        if test.cls:
            test_class = test.parent.name
        test_type = TestType.REPLAY_TEST if "__replay_test" in test_file_path else TestType.EXISTING_UNIT_TEST
        test_results.append(
            TestsInFile(
                test_file=str(test.path),
                test_class=test_class,
                test_function=test.name,
                test_suite=None,  # not used in pytest until now
                test_type=test_type,
            ),
        )
    return test_results


def discover_tests_pytest(
    cfg: TestConfig,
    discover_only_these_tests: Optional[List[str]] = None,
) -> Dict[str, List[FunctionCalledInTest]]:
    tests_root = cfg.tests_root
    project_root = cfg.project_root_path
    pytest_cmd_list = shlex.split(
        cfg.pytest_cmd,
        posix=os.name != "nt",
    )  # TODO: Do we need this for test collection?

    q = Queue()
    p = Process(target=run_pytest_discovery_new_process, args=(q, project_root, tests_root))
    p.start()
    exitcode, tests = q.get()
    p.join()
    logging.debug(f"Pytest collection exit code: {exitcode}")

    file_to_test_map = defaultdict(list)
    for test in tests:
        if discover_only_these_tests and test.test_file not in discover_only_these_tests:
            continue
        file_to_test_map[test.test_file].append(test)
    # Within these test files, find the project functions they are referring to and return their names/locations
    return process_test_files(file_to_test_map, cfg)


def discover_tests_unittest(
    cfg: TestConfig,
    discover_only_these_tests: Optional[List[str]] = None,
) -> Dict[str, List[FunctionCalledInTest]]:
    tests_root = cfg.tests_root
    loader = unittest.TestLoader()
    tests = loader.discover(str(tests_root))
    file_to_test_map = defaultdict(list)

    def get_test_details(_test) -> Optional[TestsInFile]:
        _test_function, _test_module, _test_suite_name = (
            _test._testMethodName,
            _test.__class__.__module__,
            _test.__class__.__qualname__,
        )

        _test_module_path = _test_module.replace(".", os.sep)
        _test_module_path = os.path.normpath(os.path.join(str(tests_root), _test_module_path) + ".py")
        if not os.path.exists(_test_module_path) or (
            discover_only_these_tests and _test_module_path not in discover_only_these_tests
        ):
            return None
        if "__replay_test" in _test_module_path:
            test_type = TestType.REPLAY_TEST
        else:
            test_type = TestType.EXISTING_UNIT_TEST
        return TestsInFile(
            test_file=_test_module_path,
            test_suite=_test_suite_name,
            test_function=_test_function,
            test_type=test_type,
            test_class=None,  # TODO: Validate if it is correct to set test_class to None
        )

    for _test_suite in tests._tests:
        for test_suite_2 in _test_suite._tests:
            if not hasattr(test_suite_2, "_tests"):
                logging.warning(f"Didn't find tests for {test_suite_2}")
                continue

            for test in test_suite_2._tests:
                # some test suites are nested, so we need to go deeper
                if not hasattr(test, "_testMethodName") and hasattr(test, "_tests"):
                    for test_2 in test._tests:
                        if not hasattr(test_2, "_testMethodName"):
                            logging.warning(
                                f"Didn't find tests for {test_2}",
                            )  # it goes deeper?
                            continue
                        details = get_test_details(test_2)
                        if details is not None:
                            file_to_test_map[details.test_file].append(details)
                else:
                    details = get_test_details(test)
                    if details is not None:
                        file_to_test_map[details.test_file].append(details)
    return process_test_files(file_to_test_map, cfg)


def discover_parameters_unittest(function_name: str):
    function_name = function_name.split("_")
    if len(function_name) > 1 and function_name[-1].isdigit():
        return True, "_".join(function_name[:-1]), function_name[-1]

    return False, function_name, None


def process_test_files(
    file_to_test_map: Dict[str, List[TestsInFile]],
    cfg: TestConfig,
) -> Dict[str, List[FunctionCalledInTest]]:
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
                        function_name = re.split(r"\[|\]", function)[0]
                        parameters = re.split(r"\[|\]", function)[1]
                        if name.name == function_name and name.type == "function":
                            test_functions.add(
                                TestFunction(name.name, None, parameters, functions[i].test_type),
                            )
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
                                (
                                    is_parameterized,
                                    new_function,
                                    parameters,
                                ) = discover_parameters_unittest(function)

                                if is_parameterized and new_function == def_name.name:
                                    test_functions.add(
                                        TestFunction(
                                            def_name.name,
                                            name.name,
                                            parameters,
                                            functions[0].test_type,
                                        ),  # A test file must not have more than one test type
                                    )
                                elif function == def_name.name:
                                    test_functions.add(
                                        TestFunction(def_name.name, name.name, None, functions[0].test_type),
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
                    definition = name.goto(
                        follow_imports=True,
                        follow_builtin_imports=False,
                    )
                except Exception as e:
                    logging.exception(str(e))
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
                            definition[0].module_name + ".",
                            "",
                            1,
                        )
                        qualified_name_with_modules_from_root = f"{module_name_from_file_path(definition[0].module_path, project_root_path)}.{full_name_without_module_prefix}"
                        function_to_test_map[qualified_name_with_modules_from_root].append(
                            FunctionCalledInTest(
                                test_file=test_file,
                                test_class=None,
                                test_function=scope_test_function,
                                test_suite=scope_test_suite,
                                test_type=test_type,
                                position=CodePosition(
                                    line_no=name.line,
                                    col_no=name.column,
                                ),
                            ),
                        )
    deduped_function_to_test_map = {}
    for function, tests in function_to_test_map.items():
        deduped_function_to_test_map[function] = list(set(tests))
    return deduped_function_to_test_map
