from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import isort
from code_to_optimize.bubble_sort_method import BubbleSorter
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent, TestFile, TestFiles, TestingMode
from codeflash.optimization.optimizer import Optimizer
from codeflash.verification.comparator import comparator
from codeflash.verification.equivalence import compare_test_results
from codeflash.verification.instrument_code import instrument_code
from codeflash.verification.test_results import TestType

# Used by aiservice instrumentation
behavior_logging_code = """from __future__ import annotations

import gc
import inspect
import os
import time

from pathlib import Path
from typing import Any, Callable, Optional

import dill as pickle

def codeflash_wrap(
    wrapped: Callable[..., Any],
    test_module_name: str,
    test_class_name: str | None,
    test_name: str,
    function_name: str,
    line_id: str,
    loop_index: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    test_id = f"{test_module_name}:{test_class_name}:{test_name}:{line_id}:{loop_index}"
    if not hasattr(codeflash_wrap, "index"):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f"{line_id}_{codeflash_test_index}"
    print(
        f"!######{test_module_name}:{(test_class_name + '.' if test_class_name else '')}{test_name}:{function_name}:{loop_index}:{invocation_id}######!"
    )
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = wrapped(*args, **kwargs)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    iteration = os.environ["CODEFLASH_TEST_ITERATION"]
    with Path(
        "{codeflash_run_tmp_dir_client_side}", f"test_return_values_{iteration}.bin"
    ).open("ab") as f:
        pickled_values = (
            pickle.dumps((args, kwargs, exception))
            if exception
            else pickle.dumps((args, kwargs, return_value))
        )
        _test_name = f"{test_module_name}:{(test_class_name + '.' if test_class_name else '')}{test_name}:{function_name}:{line_id}".encode(
            "ascii"
        )
        f.write(len(_test_name).to_bytes(4, byteorder="big"))
        f.write(_test_name)
        f.write(codeflash_duration.to_bytes(8, byteorder="big"))
        f.write(len(pickled_values).to_bytes(4, byteorder="big"))
        f.write(pickled_values)
        f.write(loop_index.to_bytes(8, byteorder="big"))
        f.write(len(invocation_id).to_bytes(4, byteorder="big"))
        f.write(invocation_id.encode("ascii"))
    if exception:
        raise exception
    return return_value
"""


def test_class_method_behavior_results() -> None:
    test_source = """import pytest
    from code_to_optimize.bubble_sort import BubbleSorter

    def test_single_element_list():
        # Test that a single element list returns the same single element
        obj = BubbleSorter()
        codeflash_output = obj.sorter([42])
        """
    instrumented_behavior_test_source = (
        behavior_logging_code
        + """
import pytest
from code_to_optimize.bubble_sort_method import BubbleSorter


def test_single_element_list():
    codeflash_loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])
    obj = BubbleSorter()
    _call__bound__arguments = inspect.signature(BubbleSorter.sorter).bind(obj,[42])
    _call__bound__arguments.apply_defaults()

    codeflash_return_value = codeflash_wrap(
        BubbleSorter.sorter,
        "code_to_optimize.tests.pytest.test_aiservice_behavior_results_temp",
        None,
        "test_single_element_list",
        "sorter",
        "1",
        codeflash_loop_index,
        **_call__bound__arguments.arguments,
    )
    """
    )
    instrumented_behavior_test_source = isort.code(
        instrumented_behavior_test_source, config=isort.Config(float_to_top=True)
    )

    # Init paths
    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_aiservice_behavior_results_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_aiservice_behavior_results_perf_temp.py"
    ).resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    run_cwd = Path(__file__).parent.parent.resolve()
    os.chdir(run_cwd)
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_method.py").resolve()
    original_code = fto_path.read_text("utf-8")

    try:
        temp_run_dir = get_run_tmp_file(Path()).as_posix()
        instrumented_behavior_test_source = instrumented_behavior_test_source.replace(
            "{codeflash_run_tmp_dir_client_side}", temp_run_dir
        )
        with test_path.open("w") as f:
            f.write(instrumented_behavior_test_source)

        opt = Optimizer(
            Namespace(
                project_root=project_root_path,
                disable_telemetry=True,
                tests_root=tests_root,
                test_framework="pytest",
                pytest_cmd="pytest",
                experiment_id=None,
                test_project_root=project_root_path,
            )
        )
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                )
            ]
        )
        a = BubbleSorter()
        test_results, coverage_data = opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.test_function_name == "test_single_element_list"
        assert test_results[0].did_pass
        assert test_results[0].return_value[1]["arr"] == [42]
        assert comparator(test_results[0].return_value[1]["self"], BubbleSorter())
        assert test_results[0].return_value[2] == [42]

        # Replace with optimized code that mutated instance attribute
        optimized_code_mutated_attr = """
class BubbleSorter:

    def __init__(self, x=1):
        self.x = x

    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr
                        """
        fto_path.write_text(optimized_code_mutated_attr, "utf-8")
        test_results_mutated_attr, coverage_data = opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results_mutated_attr[0].return_value[1]["self"].x == 1
        assert not compare_test_results(
            test_results, test_results_mutated_attr
        )  # The test should fail because the instance attribute was mutated
        # Replace with optimized code that did not mutate existing instance attribute, but added a new one
        optimized_code_new_attr = """
class BubbleSorter:
    def __init__(self, x=0):
        self.x = x
        self.y = 2

    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr
                        """
        fto_path.write_text(optimized_code_new_attr, "utf-8")
        test_results_new_attr, coverage_data = opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert not compare_test_results(
            test_results, test_results_new_attr
        )  # The test should pass because the instance attribute was not mutated, only a new one was added
    finally:
        fto_path.write_text(original_code, "utf-8")
        test_path.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)


def test_class_method_behavior_results_with_codeflash_capture() -> None:
    test_source = """import pytest
    from code_to_optimize.bubble_sort import BubbleSorter

    def test_single_element_list():
        # Test that a single element list returns the same single element
        obj = BubbleSorter()
        codeflash_output = obj.sorter([3,2,1])
        """
    instrumented_behavior_test_source = (
        behavior_logging_code
        + """
import pytest
from code_to_optimize.bubble_sort_method import BubbleSorter


def test_single_element_list():
    codeflash_loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])
    obj = BubbleSorter()
    _call__bound__arguments = inspect.signature(BubbleSorter.sorter).bind(obj,[3,2,1])
    _call__bound__arguments.apply_defaults()

    codeflash_return_value = codeflash_wrap(
        BubbleSorter.sorter,
        "code_to_optimize.tests.pytest.test_aiservice_behavior_results_temp",
        None,
        "test_single_element_list",
        "sorter",
        "1",
        codeflash_loop_index,
        **_call__bound__arguments.arguments,
    )
    """
    )
    instrumented_behavior_test_source = isort.code(
        instrumented_behavior_test_source, config=isort.Config(float_to_top=True)
    )

    # Init paths
    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_aiservice_behavior_results_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_aiservice_behavior_results_perf_temp.py"
    ).resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    run_cwd = Path(__file__).parent.parent.resolve()
    os.chdir(run_cwd)
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_method.py").resolve()
    original_code = fto_path.read_text("utf-8")
    function_to_optimize = FunctionToOptimize("sorter", fto_path, [FunctionParent("BubbleSorter", "ClassDef")])

    try:
        temp_run_dir = get_run_tmp_file(Path()).as_posix()
        instrumented_behavior_test_source = instrumented_behavior_test_source.replace(
            "{codeflash_run_tmp_dir_client_side}", temp_run_dir
        )
        with test_path.open("w") as f:
            f.write(instrumented_behavior_test_source)
        # Add codeflash capture decorator
        instrument_code(function_to_optimize, {})
        opt = Optimizer(
            Namespace(
                project_root=project_root_path,
                disable_telemetry=True,
                tests_root=tests_root,
                test_framework="pytest",
                pytest_cmd="pytest",
                experiment_id=None,
                test_project_root=project_root_path,
            )
        )
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                )
            ]
        )
        test_results, coverage_data = opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        # Verify instance_state result, which checks instance state right after __init__, using  codeflash_capture
        assert test_results[0].id.function_getting_tested == "BubbleSorter.__init__"
        assert test_results[0].id.test_function_name == "test_single_element_list"
        assert test_results[0].did_pass
        assert test_results[0].return_value[0] == {"x": 0}

        # Verify function_to_optimize result
        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.test_function_name == "test_single_element_list"
        assert test_results[1].did_pass

        # Checks input values to the function to see if they have mutated
        assert comparator(test_results[1].return_value[1]["self"], BubbleSorter())
        assert test_results[1].return_value[1]["arr"] == [1, 2, 3]

        # Check function return value
        assert test_results[1].return_value[2] == [1, 2, 3]

        # Replace with optimized code that mutated instance attribute
        optimized_code_mutated_attr = """
class BubbleSorter:

    def __init__(self, x=1):
        self.x = x

    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr
                        """
        fto_path.write_text(optimized_code_mutated_attr, "utf-8")
        instrument_code(function_to_optimize, {})
        test_results_mutated_attr, coverage_data = opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results_mutated_attr[1].return_value[1]["self"].x == 1
        assert not compare_test_results(
            test_results, test_results_mutated_attr
        )  # The test should fail because the instance attribute was mutated
        # Replace with optimized code that did not mutate existing instance attribute, but added a new one
        optimized_code_new_attr = """
class BubbleSorter:
    def __init__(self, x=0):
        self.x = x
        self.y = 2

    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr
                        """
        fto_path.write_text(optimized_code_new_attr, "utf-8")
        instrument_code(function_to_optimize, {})
        test_results_new_attr, coverage_data = opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results_new_attr[1].return_value[1]["self"].x == 0
        assert test_results_new_attr[1].return_value[1]["self"].y == 2
        assert not compare_test_results(
            test_results, test_results_new_attr
        )  # The test should pass because the instance attribute was not mutated, only a new one was added
    finally:
        fto_path.write_text(original_code, "utf-8")
        test_path.unlink(missing_ok=True)
