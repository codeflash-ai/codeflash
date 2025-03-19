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

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from codeflash.models.models import CodePosition, FunctionCalledInTest, TestsInFile
from codeflash.verification.test_results import TestType

if TYPE_CHECKING:
    from codeflash.verification.verification_utils import TestConfig


@dataclass(frozen=True)
class TestFunction:
    function_name: str
    test_class: Optional[str]
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
            error_pattern = r"={3,}\s*ERRORS\s*={3,}\n([\s\S]*?)(?:={3,}|$)"
            match = re.search(error_pattern, result.stdout)
            error_section = match.group(1) if match else result.stdout

            logger.warning(
                f"Failed to collect tests. Pytest Exit code: {exitcode}={ExitCode(exitcode).name}\n {error_section}"
            )

        elif 0 <= exitcode <= 5:
            logger.warning(f"Failed to collect tests. Pytest Exit code: {exitcode}={ExitCode(exitcode).name}")
        else:
            logger.warning(f"Failed to collect tests. Pytest Exit code: {exitcode}")
        console.rule()
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
            test_file=Path(test["test_file"]),
            test_class=test["test_class"],
            test_function=test["test_function"],
            test_type=test_type,
        )
        if discover_only_these_tests and test_obj.test_file not in discover_only_these_tests:
            continue
        file_to_test_map[test_obj.test_file].append(test_obj)
    # Within these test files, find the project functions they are referring to and return their names/locations
    if len(file_to_test_map) < 25: #default to single-threaded if there aren't that many files
        return process_test_files_single_threaded(file_to_test_map, cfg)
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

    if len(file_to_test_map) < 25: #default to single-threaded if there aren't that many files
        return process_test_files_single_threaded(file_to_test_map, cfg)
    return process_test_files(file_to_test_map, cfg)


def discover_parameters_unittest(function_name: str) -> tuple[bool, str, str | None]:
    function_name = function_name.split("_")
    if len(function_name) > 1 and function_name[-1].isdigit():
        return True, "_".join(function_name[:-1]), function_name[-1]

    return False, function_name, None


# Add this worker function at the module level (outside any other function)
def process_file_worker(args_tuple):
    """Worker function for processing a single test file in a separate process.

    This must be at the module level (not nested) for multiprocessing to work.
    """
    import jedi
    import re
    import os
    from collections import defaultdict
    from pathlib import Path

    # Unpack the arguments
    test_file, functions, config = args_tuple

    try:
        # Each process creates its own Jedi project
        jedi_project = jedi.Project(path=config['project_root_path'])

        local_results = defaultdict(list)
        tests_found_in_file = 0

        # Convert test_file back to Path if necessary
        test_file_path = test_file if isinstance(test_file, Path) else Path(test_file)

        try:
            script = jedi.Script(path=test_file, project=jedi_project)
            all_names = script.get_names(all_scopes=True, references=True)
            all_defs = script.get_names(all_scopes=True, definitions=True)
            all_names_top = script.get_names(all_scopes=True)

            top_level_functions = {name.name: name for name in all_names_top if name.type == "function"}
            top_level_classes = {name.name: name for name in all_names_top if name.type == "class"}
        except Exception as e:
            return {
                'status': 'error',
                'error_type': 'jedi_script_error',
                'error_message': str(e),
                'test_file': test_file,
                'results': {}
            }

        test_functions = set()

        if config['test_framework'] == "pytest":
            for function in functions:
                if "[" in function.test_function:
                    function_name = re.split(r"[\[\]]", function.test_function)[0]
                    parameters = re.split(r"[\[\]]", function.test_function)[1]
                    if function_name in top_level_functions:
                        test_functions.add(
                            TestFunction(function_name, function.test_class, parameters, function.test_type)
                        )
                elif function.test_function in top_level_functions:
                    test_functions.add(
                        TestFunction(function.test_function, function.test_class, None, function.test_type)
                    )
                elif re.match(r"^test_\w+_\d+(?:_\w+)*", function.test_function):
                    # Try to match parameterized unittest functions here
                    base_name = re.sub(r"_\d+(?:_\w+)*$", "", function.test_function)
                    if base_name in top_level_functions:
                        test_functions.add(
                            TestFunction(
                                function_name=base_name,
                                test_class=function.test_class,
                                parameters=function.test_function,
                                test_type=function.test_type,
                            )
                        )

        elif config['test_framework'] == "unittest":
            functions_to_search = [elem.test_function for elem in functions]
            test_suites = {elem.test_class for elem in functions}

            matching_names = set(test_suites) & set(top_level_classes.keys())
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
                scope_test_class = test_functions_list[index].test_class
                scope_parameters = test_functions_list[index].parameters
                test_type = test_functions_list[index].test_type
                try:
                    definition = name.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception as e:
                    continue
                if definition and definition[0].type == "function":
                    definition_path = str(definition[0].module_path)
                    # The definition is part of this project and not defined within the original function
                    if (
                            definition_path.startswith(config['project_root_path'] + os.sep)
                            and definition[0].module_name != name.module_name
                            and definition[0].full_name is not None
                    ):
                        if scope_parameters is not None:
                            if config['test_framework'] == "pytest":
                                scope_test_function += "[" + scope_parameters + "]"
                            if config['test_framework'] == "unittest":
                                scope_test_function += "_" + scope_parameters

                        # Get module name relative to project root
                        module_name = module_name_from_file_path(definition[0].module_path, config['project_root_path'])

                        full_name_without_module_prefix = definition[0].full_name.replace(
                            definition[0].module_name + ".", "", 1
                        )
                        qualified_name_with_modules_from_root = f"{module_name}.{full_name_without_module_prefix}"

                        # Create a serializable representation of the result
                        result_entry = {
                            'test_file': str(test_file),
                            'test_class': scope_test_class,
                            'test_function': scope_test_function,
                            'test_type': test_type,
                            'line_no': name.line,
                            'col_no': name.column
                        }

                        # Add to local results
                        if qualified_name_with_modules_from_root not in local_results:
                            local_results[qualified_name_with_modules_from_root] = []
                        local_results[qualified_name_with_modules_from_root].append(result_entry)
                        tests_found_in_file += 1

        return {
            'status': 'success',
            'test_file': test_file,
            'tests_found': tests_found_in_file,
            'results': dict(local_results)  # Convert defaultdict to dict for serialization
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error_type': 'general_error',
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'test_file': test_file,
            'results': {}
        }

def process_test_files_single_threaded(
    file_to_test_map: dict[str, list[TestsInFile]], cfg: TestConfig
) -> dict[str, list[FunctionCalledInTest]]:
    project_root_path = cfg.project_root_path
    test_framework = cfg.test_framework
    function_to_test_map = defaultdict(list)
    jedi_project = jedi.Project(path=project_root_path)

    for test_file, functions in file_to_test_map.items():
        try:
            script = jedi.Script(path=test_file, project=jedi_project)
            test_functions = set()

            all_names = script.get_names(all_scopes=True, references=True)
            all_defs = script.get_names(all_scopes=True, definitions=True)
            all_names_top = script.get_names(all_scopes=True)

            top_level_functions = {name.name: name for name in all_names_top if name.type == "function"}
            top_level_classes = {name.name: name for name in all_names_top if name.type == "class"}
        except Exception as e:
            logger.debug(f"Failed to get jedi script for {test_file}: {e}")
            continue

        if test_framework == "pytest":
            for function in functions:
                if "[" in function.test_function:
                    function_name = re.split(r"[\[\]]", function.test_function)[0]
                    parameters = re.split(r"[\[\]]", function.test_function)[1]
                    if function_name in top_level_functions:
                        test_functions.add(
                            TestFunction(function_name, function.test_class, parameters, function.test_type)
                        )
                elif function.test_function in top_level_functions:
                    test_functions.add(
                        TestFunction(function.test_function, function.test_class, None, function.test_type)
                    )
                elif re.match(r"^test_\w+_\d+(?:_\w+)*", function.test_function):
                    # Try to match parameterized unittest functions here, although we can't get the parameters.
                    # Extract base name by removing the numbered suffix and any additional descriptions
                    base_name = re.sub(r"_\d+(?:_\w+)*$", "", function.test_function)
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
            test_suites = [elem.test_class for elem in functions]

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
                                    )  # A test file must not have more than one test type
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
                scope_test_class = test_functions_list[index].test_class
                scope_parameters = test_functions_list[index].parameters
                test_type = test_functions_list[index].test_type
                try:
                    definition = name.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception as e:
                    logger.debug(str(e))
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
                                    test_class=scope_test_class,
                                    test_function=scope_test_function,
                                    test_type=test_type,
                                ),
                                position=CodePosition(line_no=name.line, col_no=name.column),
                            )
                        )
    deduped_function_to_test_map = {}
    for function, tests in function_to_test_map.items():
        deduped_function_to_test_map[function] = list(set(tests))
    return deduped_function_to_test_map


def process_test_files(
        file_to_test_map: dict[str, list[TestsInFile]], cfg: TestConfig
) -> dict[str, list[FunctionCalledInTest]]:
    from multiprocessing import Pool, cpu_count
    import os
    import pickle

    project_root_path = cfg.project_root_path
    test_framework = cfg.test_framework

    logger.info(f"Starting to process {len(file_to_test_map)} test files with multiprocessing")

    # Create a configuration dictionary to pass to worker processes
    config_dict = {
        'project_root_path': str(project_root_path),
        'test_framework': test_framework
    }

    # Prepare data for processing - create a list of (test_file, functions, config) tuples
    process_inputs = []
    for test_file, functions in file_to_test_map.items():
        # Convert TestsInFile objects to serializable form if needed
        serializable_functions = []
        for func in functions:
            # Ensure test_file is a string (needed for pickling)
            if hasattr(func, 'test_file') and not isinstance(func.test_file, str):
                func_dict = func._asdict() if hasattr(func, '_asdict') else func.__dict__.copy()
                func_dict['test_file'] = str(func_dict['test_file'])
                serializable_functions.append(TestsInFile(**func_dict))
            else:
                serializable_functions.append(func)
        process_inputs.append((str(test_file), serializable_functions, config_dict))

    # Determine optimal number of processes
    max_processes = min(cpu_count() * 2, len(process_inputs), 32)
    logger.info(f"Using {max_processes} processes for parallel test file processing")

    # Create a Pool and process the files
    processed_files = 0
    error_count = 0
    function_to_test_map = defaultdict(list)

    # Use smaller chunk size for better load balancing
    chunk_size = max(1, len(process_inputs) // (max_processes * 4))

    with Pool(processes=max_processes) as pool:
        # Use imap_unordered for better performance (we don't care about order)
        for i, result in enumerate(pool.imap_unordered(process_file_worker, process_inputs, chunk_size)):
            processed_files += 1

            # Log progress
            if processed_files % 100 == 0 or processed_files == len(process_inputs):
                logger.info(f"Processed {processed_files}/{len(process_inputs)} files")

            if result['status'] == 'error':
                error_count += 1
                logger.warning(f"Error processing file {result['test_file']}: {result['error_message']}")
                if 'traceback' in result:
                    logger.debug(f"Traceback: {result['traceback']}")
                continue

            # Process results from this file
            for qualified_name, test_entries in result['results'].items():
                for entry in test_entries:
                    # Reconstruct FunctionCalledInTest from the serialized data
                    test_in_file = TestsInFile(
                        test_file=entry['test_file'],
                        test_class=entry['test_class'],
                        test_function=entry['test_function'],
                        test_type=entry['test_type']
                    )

                    position = CodePosition(line_no=entry['line_no'], col_no=entry['col_no'])

                    function_to_test_map[qualified_name].append(
                        FunctionCalledInTest(
                            tests_in_file=test_in_file,
                            position=position
                        )
                    )

    logger.info(f"Processing complete. Processed {processed_files}/{len(process_inputs)} files")
    logger.info(f"Files with errors: {error_count}")

    # Log metrics before deduplication
    total_tests_before_dedup = sum(len(tests) for tests in function_to_test_map.values())
    logger.info(
        f"Found {len(function_to_test_map)} unique functions with {total_tests_before_dedup} total tests before deduplication")

    # Deduplicate results
    deduped_function_to_test_map = {}
    for function, tests in function_to_test_map.items():
        # Convert to set and back to list to remove duplicates
        # We need to handle custom objects properly
        unique_tests = []
        seen = set()

        for test in tests:
            # Create a hashable representation of the test
            test_hash = (
                str(test.tests_in_file.test_file),
                test.tests_in_file.test_class,
                test.tests_in_file.test_function,
                test.tests_in_file.test_type,
                test.position.line_no,
                test.position.col_no
            )

            if test_hash not in seen:
                seen.add(test_hash)
                unique_tests.append(test)

        deduped_function_to_test_map[function] = unique_tests

    # Log metrics after deduplication
    total_tests_after_dedup = sum(len(tests) for tests in deduped_function_to_test_map.values())
    logger.info(
        f"After deduplication: {len(deduped_function_to_test_map)} unique functions with {total_tests_after_dedup} total tests")

    return deduped_function_to_test_map
