from __future__ import annotations

import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, TestFile, TestFiles, TestingMode
from codeflash.optimization.optimizer import Optimizer
from codeflash.verification.equivalence import compare_test_results
from codeflash.verification.test_results import TestType

# Used by cli instrumentation
codeflash_wrap_string = """def codeflash_wrap(wrapped, test_module_name, test_class_name, test_name, function_name, line_id, loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{{test_module_name}}:{{test_class_name}}:{{test_name}}:{{line_id}}:{{loop_index}}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {{}}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{{line_id}}_{{codeflash_test_index}}'
    print(f"!######{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}######!")
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
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (test_module_name, test_class_name, test_name, function_name, loop_index, invocation_id, codeflash_duration, pickled_return_value))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value
"""


def test_bubble_sort_behavior_results() -> None:
    code = """from code_to_optimize.bubble_sort import sorter


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    output = sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"""

    expected = (
        """import gc
import os
import sqlite3
import time

import dill as pickle

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_string
        + """
def test_sort():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB)')
    input = [5, 4, 3, 2, 1, 0]
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, input)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    codeflash_con.close()
"""
    )

    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_results_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_results_perf_temp.py"
    ).resolve()
    try:
        with test_path.open("w") as f:
            f.write(code)

        tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
        project_root_path = (Path(__file__).parent / "..").resolve()

        new_test = expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")

        with test_path.open("w") as f:
            f.write(new_test)

        # Overwrite old test with new instrumented test

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
        print(test_results)
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "1_0"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "4_0"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass

    finally:
        test_path.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)


def test_class_method_behavior_results() -> None:
    code = """from code_to_optimize.bubble_sort_method import BubbleSorter


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    sort_class = BubbleSorter()
    output = sort_class.sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    sort_class = BubbleSorter()
    output = sort_class.sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"""

    expected = """import gc
import os
import sqlite3
import time

import dill as pickle

from code_to_optimize.bubble_sort_method import BubbleSorter


def codeflash_wrap(wrapped, test_module_name, test_class_name, test_name, function_name, line_id, loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{{test_module_name}}:{{test_class_name}}:{{test_name}}:{{line_id}}:{{loop_index}}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {{}}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{{line_id}}_{{codeflash_test_index}}'
    """
    if sys.version_info < (3, 12):
        expected += """print(f"!######{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}######!")"""
    else:
        expected += """print(f'!######{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}######!')"""
    expected += """
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
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (test_module_name, test_class_name, test_name, function_name, loop_index, invocation_id, codeflash_duration, pickled_return_value))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value
"""
    expected += """
def test_sort():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB)')
    input = [5, 4, 3, 2, 1, 0]
    sort_class = BubbleSorter()
    output = codeflash_wrap(sort_class.sorter, '{module_path}', None, 'test_sort', 'sorter', '2', codeflash_loop_index, codeflash_cur, codeflash_con, input)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    sort_class = BubbleSorter()
    output = codeflash_wrap(sort_class.sorter, '{module_path}', None, 'test_sort', 'sorter', '6', codeflash_loop_index, codeflash_cur, codeflash_con, input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    codeflash_con.close()
"""

    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=Path("module.py"))
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name), [CodePosition(7, 13), CodePosition(12, 13)], func, Path(f.name).parent, "pytest"
        )
        os.chdir(original_cwd)
    assert success
    assert new_test == expected.format(
        module_path=Path(f.name).name, tmp_dir_path=get_run_tmp_file(Path("test_return_values"))
    )

    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_class_method_behavior_results_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_class_method_behavior_results_perf_temp.py"
    ).resolve()
    try:
        with test_path.open("w") as f:
            f.write(code)

        tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
        project_root_path = (Path(__file__).parent / "..").resolve()

        new_test = expected.format(
            module_path="code_to_optimize.tests.pytest.test_class_method_behavior_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")

        with test_path.open("w") as f:
            f.write(new_test)

        # Overwrite old test with new instrumented test

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

        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "2_0"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_class_method_behavior_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "6_0"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_class_method_behavior_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass

        # Replace with optimized code that mutated instance attribute
        optimized_code = """
class BubbleSorter:
    def __init__(self):
        self.x = 1

    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr

        """
        fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_method.py").resolve()
        original_code = fto_path.read_text("utf-8")
        fto_path.write_text(optimized_code, "utf-8")
        new_test_results, coverage_data = opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert new_test_results[0].id.function_getting_tested == "sorter"
        assert new_test_results[0].id.iteration_id == "2_0"
        assert new_test_results[0].id.test_class_name is None
        assert new_test_results[0].id.test_function_name == "test_sort"
        assert (
            new_test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_class_method_behavior_results_temp"
        )

        assert compare_test_results(test_results, new_test_results)
        fto_path.write_text(original_code, "utf-8")
    finally:
        # test_path.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)
