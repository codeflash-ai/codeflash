import logging
import os
import pathlib
import re
import subprocess
import unittest
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional

import jedi
from pydantic.dataclasses import dataclass

from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.verification.verification_utils import TestConfig


class ParseType(Enum):
    CO = "co"
    Q = "q"


@dataclass(frozen=True)
class TestsInFile:
    test_file: str
    test_class: Optional[str]
    test_function: str
    test_suite: Optional[str]

    @classmethod
    def from_pytest_stdout_line_co(cls, module: str, function: str, directory: str):
        absolute_test_path = os.path.join(directory, module)
        assert os.path.exists(
            absolute_test_path,
        ), f"Test discovery failed - Test file does not exist {absolute_test_path}"
        return cls(
            test_file=absolute_test_path,
            test_class=None,
            test_function=function,
            test_suite=None,
        )

    @classmethod
    def from_pytest_stdout_line_q(cls, line: str, pytest_rootdir: str):
        parts = line.split("::")
        absolute_test_path = os.path.join(pytest_rootdir, parts[0])
        assert os.path.exists(
            absolute_test_path,
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
    parameters: Optional[str]


@dataclass(frozen=True)
class TestDetails:
    test_function: str
    test_module_path: str
    test_suite_name: str


def discover_unit_tests(cfg: TestConfig) -> Dict[str, List[TestsInFile]]:
    test_frameworks = {
        "pytest": discover_tests_pytest,
        "unittest": discover_tests_unittest,
    }
    discover_tests = test_frameworks.get(cfg.test_framework)
    if discover_tests is None:
        raise ValueError(f"Unsupported test framework: {cfg.test_framework}")
    return discover_tests(cfg)


def get_pytest_rootdir_only(pytest_cmd_list, tests_root, project_root) -> str:
    # Ref - https://docs.pytest.org/en/stable/reference/customize.html#initialization-determining-rootdir-and-configfile
    # A very hacky solution that only runs the --co mode until we see the rootdir print and then it just kills the
    # pytest to save time. We should find better ways to just get the rootdir, one way is to not use the -q flag and
    # parse the --co output, but that could be more work.
    process = subprocess.Popen(
        pytest_cmd_list + [tests_root, "--co"],
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
    raise ValueError(f"Could not find rootdir in pytest output for {tests_root}")


def discover_tests_pytest(cfg: TestConfig) -> Dict[str, List[TestsInFile]]:
    tests_root = cfg.tests_root
    project_root = cfg.project_root_path
    pytest_cmd_list = [chunk for chunk in cfg.pytest_cmd.split(" ") if chunk != ""]
    pytest_result = subprocess.run(
        pytest_cmd_list + [f"{tests_root}", "--co", "-q", "-m", "not skip"],
        stdout=subprocess.PIPE,
        cwd=project_root,
        check=False,
    )

    pytest_stdout = pytest_result.stdout.decode("utf-8")

    parse_type = ParseType.Q
    if "rootdir: " not in pytest_stdout:
        pytest_rootdir = get_pytest_rootdir_only(pytest_cmd_list, tests_root, project_root)
    else:
        rootdir_re = re.compile(r"^rootdir:\s?(\S*)", re.MULTILINE)
        pytest_rootdir_match = rootdir_re.search(pytest_stdout)
        if not pytest_rootdir_match:
            raise ValueError(f"Could not find rootdir in pytest output for {tests_root}")
        pytest_rootdir = pytest_rootdir_match.group(1)
        parse_type = ParseType.CO

    tests = parse_pytest_stdout(pytest_stdout, pytest_rootdir, tests_root, parse_type)
    file_to_test_map = defaultdict(list)

    for test in tests:
        file_to_test_map[test.test_file].append({"test_function": test.test_function})
    # Within these test files, find the project functions they are referring to and return their names/locations
    return process_test_files(file_to_test_map, cfg)


def discover_tests_unittest(cfg: TestConfig) -> Dict[str, List[TestsInFile]]:
    tests_root = cfg.tests_root
    project_root_path = cfg.project_root_path
    loader = unittest.TestLoader()
    tests = loader.discover(str(tests_root))
    file_to_test_map = defaultdict(list)

    def get_test_details(_test) -> Optional[TestDetails]:
        _test_function, _test_module, _test_suite_name = (
            _test._testMethodName,
            _test.__class__.__module__,
            _test.__class__.__qualname__,
        )

        _test_module_path = _test_module.replace(".", os.sep)
        _test_module_path = os.path.join(str(tests_root), _test_module_path) + ".py"
        if not os.path.exists(_test_module_path):
            return None
        return TestDetails(_test_function, _test_module_path, _test_suite_name)

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
                            logging.warning(f"Didn't find tests for {test_2}")  # it goes deeper?
                            continue
                        details = get_test_details(test_2)
                        if details is not None:
                            file_to_test_map[details.test_module_path].append(
                                {
                                    "test_function": details.test_function,
                                    "test_suite_name": details.test_suite_name,
                                },
                            )
                else:
                    details = get_test_details(test)
                    if details is not None:
                        file_to_test_map[details.test_module_path].append(
                            {
                                "test_function": details.test_function,
                                "test_suite_name": details.test_suite_name,
                            },
                        )
    return process_test_files(file_to_test_map, cfg)


def discover_parameters_unittest(function_name: str):
    function_name = function_name.split("_")
    if len(function_name) > 1 and function_name[-1].isdigit():
        return True, "_".join(function_name[:-1]), function_name[-1]

    return False, function_name, None


def process_test_files(
    file_to_test_map: Dict[str, List[Dict[str, str]]],
    cfg: TestConfig,
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
                for function in functions_to_search:
                    if "[" in function:
                        function_name = re.split(r"\[|\]", function)[0]
                        parameters = re.split(r"\[|\]", function)[1]
                        if name.name == function_name and name.type == "function":
                            test_functions.add(TestFunction(name.name, None, parameters))
                    elif name.name == function and name.type == "function":
                        test_functions.add(TestFunction(name.name, None, None))
                        break
            if test_framework == "unittest":
                functions_to_search = [elem["test_function"] for elem in functions]
                test_suites = [elem["test_suite_name"] for elem in functions]
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
                                        TestFunction(def_name.name, name.name, parameters),
                                    )
                                elif function == def_name.name:
                                    test_functions.add(TestFunction(def_name.name, name.name, None))

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
                try:
                    definition = script.goto(
                        line=name.line,
                        column=name.column,
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
                    ):
                        if scope_parameters is not None:
                            if test_framework == "pytest":
                                scope_test_function += "[" + scope_parameters + "]"
                            if test_framework == "unittest":
                                scope_test_function += "_" + scope_parameters
                        qualified_name_with_modules_from_root = f"{module_name_from_file_path(definition[0].module_path, project_root_path)}.{definition[0].name}"
                        function_to_test_map[qualified_name_with_modules_from_root].append(
                            TestsInFile(test_file, None, scope_test_function, scope_test_suite),
                        )
    deduped_function_to_test_map = {}
    for function, tests in function_to_test_map.items():
        deduped_function_to_test_map[function] = list(set(tests))
    return deduped_function_to_test_map


def parse_pytest_stdout(
    pytest_stdout: str,
    pytest_rootdir: str,
    tests_root: str,
    parse_type: ParseType,
) -> List[TestsInFile]:
    test_results = []
    if parse_type == ParseType.Q:
        for line in pytest_stdout.splitlines():
            if line.startswith("==") or line.startswith("\n") or line == "":
                break
            try:
                test_result = TestsInFile.from_pytest_stdout_line_q(line, pytest_rootdir)
                test_results.append(test_result)
            except ValueError as e:
                logging.warning(str(e))
                continue
        return test_results

    directory = tests_root
    for line in pytest_stdout.splitlines():
        if "<Dir " in line:
            new_dir = re.match(r"\s*<Dir (.+)>", line).group(1)
            new_directory = os.path.join(directory, new_dir)
            while not os.path.exists(new_directory):
                directory = os.path.dirname(directory)
                new_directory = os.path.join(directory, new_dir)

            directory = new_directory

        elif "<Package " in line:
            new_dir = re.match(r"\s*<Package (.+)>", line).group(1)
            new_directory = os.path.join(directory, new_dir)
            while len(new_directory) > 0 and not os.path.exists(new_directory):
                directory = os.path.dirname(directory)
                new_directory = os.path.join(directory, new_dir)

            if len(new_directory) == 0:
                return test_results

            directory = new_directory

        elif "<Module " in line:
            module = re.match(r"\s*<Module (.+)>", line).group(1)
            if ".py" not in module:
                module.append(".py")

            module_list = pathlib.Path(module).parts
            index = len(module_list) - 1
            if len(module_list) > 1:
                curr_dir = module
                while len(module_list) > 1 and curr_dir not in directory:
                    curr_dir = os.path.dirname(curr_dir)
                    module_list = module_list[:-1]
                    index -= 1

                module_list = pathlib.Path(module).parts
                if index < len(module_list) - 1:
                    index += 1
                    module_list = module_list[index:]
                    while not directory.endswith(curr_dir):
                        directory = os.path.dirname(directory)

                while len(module_list) > 1:
                    directory = os.path.join(directory, module_list[0])
                    module_list = module_list[1:]

                module = module_list[0]

            while len(directory) > 0 and not os.path.exists(os.path.join(directory, module)):
                directory = os.path.dirname(directory)

            if len(directory) == 0:
                return test_results

        elif "<Function " in line and module is not None:
            function = re.match(r"\s*<Function (.+)>", line)
            if function:
                function = function.group(1)
                try:
                    test_result = TestsInFile.from_pytest_stdout_line_co(
                        module,
                        function,
                        directory,
                    )
                    test_results.append(test_result)
                except ValueError as e:
                    logging.warning(str(e))

    return test_results
