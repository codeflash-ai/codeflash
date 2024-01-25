import logging
import os
import re
import subprocess
import unittest
from collections import defaultdict
from typing import Dict, List, Optional

import jedi
from pydantic.dataclasses import dataclass

from codeflash.verification.verification_utils import TestConfig


@dataclass(frozen=True)
class TestsInFile:
    test_file: str
    test_class: Optional[str]
    test_function: str
    test_suite: Optional[str]

    @classmethod
    def from_pytest_stdout_line(cls, line: str, pytest_rootdir: str):
        parts = line.split("::")
        absolute_test_path = os.path.join(pytest_rootdir, parts[0])
        assert os.path.exists(
            absolute_test_path
        ), f"Test discovery failed - Test file does not exist {absolute_test_path}"
        if len(parts) == 3:
            return cls(
                test_file=absolute_test_path,
                test_class=parts[1],
                test_function=parts[2],
                test_suite=None,
            )
        elif len(parts) == 2:
            return cls(
                test_file=absolute_test_path,
                test_class=None,
                test_function=parts[1],
                test_suite=None,
            )
        else:
            raise ValueError(f"Unexpected pytest result format: {line}")


@dataclass(frozen=True)
class TestFunction:
    function_name: str
    test_suite_name: Optional[str]


def discover_unit_tests(cfg: TestConfig) -> Dict[str, List[TestsInFile]]:
    test_frameworks = {
        "pytest": discover_tests_pytest,
        "unittest": discover_tests_unittest,
    }
    discover_tests = test_frameworks.get(cfg.test_framework)
    if discover_tests is None:
        raise ValueError(f"Unsupported test framework: {cfg.test_framework}")
    return discover_tests(cfg)


def get_pytest_rootdir_only(pytest_cmd_list, test_root, project_root) -> str:
    # Ref - https://docs.pytest.org/en/stable/reference/customize.html#initialization-determining-rootdir-and-configfile
    # A very hacky solution that only runs the --co mode until we see the rootdir print and then it just kills the
    # pytest to save time. We should find better ways to just get the rootdir, one way is to not use the -q flag and
    # parse the --co output, but that could be more work.
    process = subprocess.Popen(
        pytest_cmd_list + [test_root, "--co"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=project_root,
    )
    rootdir_re = re.compile(r"^rootdir:\s?([^\s]*)")
    # Iterate over the output lines
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            if rootdir_re.search(output):
                process.kill()
                return rootdir_re.search(output).group(1)
    raise ValueError(f"Could not find rootdir in pytest output for {test_root}")


# TODO use output without -q, that way we also get the rootdir from the output
# then we can get rid of the above get_pytest_rootdir_only function
def discover_tests_pytest(cfg: TestConfig) -> Dict[str, List[TestsInFile]]:
    test_root = cfg.test_root
    project_root = cfg.project_root_path
    pytest_cmd_list = [chunk for chunk in cfg.pytest_cmd.split(" ") if chunk != ""]
    pytest_result = subprocess.run(
        pytest_cmd_list + [f"{test_root}", "--co", "-m", "not skip"],
        stdout=subprocess.PIPE,
        cwd=project_root,
    )
    pytest_stdout = pytest_result.stdout.decode("utf-8")
    rootdir_re = re.compile(r"^rootdir:\s?(\S*)", re.MULTILINE)
    pytest_rootdir_match = rootdir_re.search(pytest_stdout)
    if not pytest_rootdir_match:
        raise ValueError(f"Could not find rootdir in pytest output for {test_root}")
    pytest_rootdir = pytest_rootdir_match.group(1)

    tests = parse_pytest_stdout(pytest_result.stdout.decode("utf-8"), pytest_rootdir)
    file_to_test_map = defaultdict(list)

    for test in tests:
        file_to_test_map[test.test_file].append({"test_function": test.test_function})
    # Within these test files, find the project functions they are referring to and return their names/locations
    return process_test_files(file_to_test_map, cfg)


def discover_tests_unittest(cfg: TestConfig) -> Dict[str, List[TestsInFile]]:
    test_root = cfg.test_root
    project_root_path = cfg.project_root_path
    loader = unittest.TestLoader()
    tests = loader.discover(str(test_root))
    file_to_test_map = defaultdict(list)
    for _test_suite in tests._tests:
        for test_suite_2 in _test_suite._tests:
            if not hasattr(test_suite_2, "_tests"):
                logging.warning(f"Didn't find tests for {test_suite_2}")
                continue
            for test in test_suite_2._tests:
                test_function, test_module, test_suite_name = (
                    test._testMethodName,
                    test.__class__.__module__,
                    test.__class__.__qualname__,
                )

                test_module_path = test_module.replace(".", os.sep)
                test_module_path = os.path.join(str(test_root), test_module_path) + ".py"
                if not os.path.exists(test_module_path):
                    continue
                file_to_test_map[test_module_path].append(
                    {"test_function": test_function, "test_suite_name": test_suite_name}
                )
    return process_test_files(file_to_test_map, cfg)


def process_test_files(
    file_to_test_map: Dict[str, List[Dict[str, str]]], cfg: TestConfig
) -> Dict[str, List[TestsInFile]]:
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
                functions_to_search = [elem["test_function"] for elem in functions]
                if name.name in functions_to_search and name.type == "function":
                    test_functions.add(TestFunction(name.name, None))
            if test_framework == "unittest":
                functions_to_search = [elem["test_function"] for elem in functions]
                test_suites = [elem["test_suite_name"] for elem in functions]
                if name.name in test_suites and name.type == "class":
                    for def_name in all_defs:
                        if (
                            def_name.name in functions_to_search
                            and def_name.type == "function"
                            and def_name.full_name is not None
                            and f".{name.name}." in def_name.full_name
                        ):
                            test_functions.add(TestFunction(def_name.name, name.name))
        test_functions_list = list(test_functions)
        test_functions_raw = [elem.function_name for elem in test_functions_list]

        for name in all_names:
            if name.full_name is None:
                continue
            m = re.search(r"([^.]+)\." + f"{name.name}$", name.full_name)
            if not m:
                continue
            scope = m.group(1)
            index = test_functions_raw.index(scope) if scope in test_functions_raw else -1
            if index >= 0:
                scope_test_function = test_functions_list[index].function_name
                scope_test_suite = test_functions_list[index].test_suite_name
                try:
                    definition = script.goto(
                        line=name.line,
                        column=name.column,
                        follow_imports=True,
                        follow_builtin_imports=False,
                    )
                except Exception as e:
                    logging.error(str(e))
                    continue
                if definition and definition[0].type == "function":
                    definition_path = str(definition[0].module_path)
                    # The definition is part of this project and not defined within the original function
                    if (
                        definition_path.startswith(str(project_root_path) + os.sep)
                        and definition[0].module_name != name.module_name
                    ):
                        function_to_test_map[definition[0].full_name].append(
                            TestsInFile(test_file, None, scope_test_function, scope_test_suite)
                        )
    deduped_function_to_test_map = {}
    for function, tests in function_to_test_map.items():
        deduped_function_to_test_map[function] = list(set(tests))
    return deduped_function_to_test_map


def parse_pytest_stdout(pytest_stdout: str, pytest_rootdir) -> List[TestsInFile]:
    test_results = []
    for line in pytest_stdout.splitlines():
        if line.startswith("==") or line.startswith("\n") or line == "":
            break
        if "[" in line:
            # TODO: Handle parameterized tests later. Update - This is important
            continue
        try:
            test_result = TestsInFile.from_pytest_stdout_line(line, pytest_rootdir)
            test_results.append(test_result)
        except ValueError as e:
            logging.warning(str(e))
            continue
    return test_results
