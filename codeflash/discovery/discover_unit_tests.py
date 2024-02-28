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
    def from_pytest_stdout_line(cls, module: str, function: str, directory: str):
        absolute_test_path = os.path.join(directory, module)
        assert os.path.exists(
            absolute_test_path
        ), f"Test discovery failed - Test file does not exist {absolute_test_path}"
        return cls(
            test_file=absolute_test_path,
            test_class=None,
            test_function=function,
            test_suite=None,
        )


@dataclass(frozen=True)
class TestFunction:
    function_name: str
    test_suite_name: Optional[str]
    parameters: Optional[str]


def discover_unit_tests(cfg: TestConfig) -> Dict[str, List[TestsInFile]]:
    test_frameworks = {
        "pytest": discover_tests_pytest,
        "unittest": discover_tests_unittest,
    }
    discover_tests = test_frameworks.get(cfg.test_framework)
    if discover_tests is None:
        raise ValueError(f"Unsupported test framework: {cfg.test_framework}")
    return discover_tests(cfg)


def discover_tests_pytest(cfg: TestConfig) -> Dict[str, List[TestsInFile]]:
    tests_root = cfg.tests_root
    project_root = cfg.project_root_path
    pytest_cmd_list = [chunk for chunk in cfg.pytest_cmd.split(" ") if chunk != ""]
    pytest_result = subprocess.run(
        pytest_cmd_list + [f"{tests_root}", "--co", "-m", "not skip"],
        stdout=subprocess.PIPE,
        cwd=tests_root,
    )

    pytest_stdout = pytest_result.stdout.decode("utf-8")
    print(pytest_stdout)
    rootdir_re = re.compile(r"^rootdir:\s?(\S*)", re.MULTILINE)
    pytest_rootdir_match = rootdir_re.search(pytest_stdout)
    if not pytest_rootdir_match:
        raise ValueError(f"Could not find rootdir in pytest output for {tests_root}")
    pytest_rootdir = pytest_rootdir_match.group(1)

    tests = parse_pytest_stdout(pytest_stdout, pytest_rootdir, tests_root)
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
                test_module_path = os.path.join(str(tests_root), test_module_path) + ".py"
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
                for function in functions_to_search:
                    if "[" in function:
                        function_name = re.split(r"\[|\]", function)[0]
                        parameters = re.split(r"\[|\]", function)[1]
                        if name.name == function_name and name.type == "function":
                            test_functions.add(TestFunction(name.name, None, parameters))
                    else:
                        if name.name == function and name.type == "function":
                            test_functions.add(TestFunction(name.name, None, None))
                            break
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
                    logging.error(str(e))
                    continue
                if definition and definition[0].type == "function":
                    definition_path = str(definition[0].module_path)
                    # The definition is part of this project and not defined within the original function
                    if (
                        definition_path.startswith(str(project_root_path) + os.sep)
                        and definition[0].module_name != name.module_name
                    ):
                        if scope_parameters is not None:
                            scope_test_function += "[" + scope_parameters + "]"
                        function_to_test_map[definition[0].full_name].append(
                            TestsInFile(test_file, None, scope_test_function, scope_test_suite)
                        )
    deduped_function_to_test_map = {}
    for function, tests in function_to_test_map.items():
        deduped_function_to_test_map[function] = list(set(tests))
    return deduped_function_to_test_map


def parse_pytest_stdout(pytest_stdout: str, pytest_rootdir, tests_root) -> List[TestsInFile]:
    test_results = []
    module_line = None
    directory = tests_root
    print("directory: ", directory)
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
            print(directory, new_dir)
            print(new_dir, new_directory, directory)
            while len(new_directory) > 0 and not os.path.exists(new_directory):
                directory = os.path.dirname(directory)
                new_directory = os.path.join(directory, new_dir)

            if len(new_directory) == 0:
                return test_results

            directory = new_directory

        elif "<Module " in line:
            print("before: ", directory)
            module = re.match(r"\s*<Module (.+)>", line).group(1)
            # module = module.split("/")[-1]
            if ".py" not in module:
                module.append(".py")

            # print("module: ", module)

            module_list = module.split("/")
            index = len(module_list) - 1
            if len(module_list) > 1:
                curr_dir = module
                while len(module_list) > 1 and curr_dir not in directory:
                    curr_dir = os.path.dirname(curr_dir)
                    module_list = module_list[:-1]
                    index -= 1

                module_list = module.split("/")
                if index < len(module_list) - 1:
                    index += 1
                    module_list = module_list[index:]
                    while not directory.endswith(curr_dir):
                        directory = os.path.dirname(directory)

                while len(module_list) > 1:
                    directory = os.path.join(directory, module_list[0])
                    module_list = module_list[1:]

                module = module_list[0]

            print("module: ", module)

            while len(directory) > 0 and not os.path.exists(os.path.join(directory, module)):
                directory = os.path.dirname(directory)

            if len(directory) == 0:
                return test_results

            # module_line = line
            print("after: ", directory)

        elif "<Function " in line and module is not None:
            function = re.match(r"\s*<Function (.+)>", line)
            if function:
                function = function.group(1)
                try:
                    test_result = TestsInFile.from_pytest_stdout_line(module, function, directory)
                    print("Test result: ", test_result)
                    test_results.append(test_result)
                except ValueError as e:
                    logging.warning(str(e))

    return test_results
