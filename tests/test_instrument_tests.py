from __future__ import annotations

import ast
import math
import os
import sys
import tempfile
from pathlib import Path
import pytest
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.instrument_existing_tests import (
    FunctionImportedAsVisitor,
    _create_device_sync_statements,
    create_wrapper_function,
    inject_profiling_into_existing_test,
)
from codeflash.code_utils.line_profile_utils import add_decorator_imports
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import (
    CodeOptimizationContext,
    CodePosition,
    FunctionParent,
    TestFile,
    TestFiles,
    TestingMode,
    TestsInFile,
    TestType,
)
import platform

from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

codeflash_wrap_string = """def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{{codeflash_test_module_name}}:{{codeflash_test_class_name}}:{{codeflash_test_name}}:{{codeflash_line_id}}:{{codeflash_loop_index}}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {{}}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{{codeflash_line_id}}_{{codeflash_test_index}}'
    test_stdout_tag = f"{{codeflash_test_module_name}}:{{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}}{{codeflash_test_name}}:{{codeflash_function_name}}:{{codeflash_loop_index}}:{{invocation_id}}"
    print(f"!$######{{test_stdout_tag}}######$!")
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f"!######{{test_stdout_tag}}######!")
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value
"""

codeflash_wrap_perfonly_string = """def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, *args, **kwargs):
    test_id = f'{{codeflash_test_module_name}}:{{codeflash_test_class_name}}:{{codeflash_test_name}}:{{codeflash_line_id}}:{{codeflash_loop_index}}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {{}}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{{codeflash_line_id}}_{{codeflash_test_index}}'
    test_stdout_tag = f"{{codeflash_test_module_name}}:{{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}}{{codeflash_test_name}}:{{codeflash_function_name}}:{{codeflash_loop_index}}:{{invocation_id}}"
    print(f"!$######{{test_stdout_tag}}######$!")
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f"!######{{test_stdout_tag}}:{{codeflash_duration}}######!")
    if exception:
        raise exception
    return return_value
"""


def build_expected_unittest_imports(extra_imports: str = "") -> str:
    imports = """import gc
import inspect
import os
import sqlite3
import time
import unittest

import dill as pickle"""
    # timeout_decorator no longer used since pytest handles timeouts
    if extra_imports:
        imports += "\n" + extra_imports
    return imports


def build_expected_pytest_imports(extra_imports: str = "") -> str:
    """Helper to build platform-aware imports for pytest tests."""
    imports = """import gc
import os
import time

import pytest"""
    if extra_imports:
        imports += "\n" + extra_imports
    return imports
# create a temporary directory for the test results
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

def test_perfinjector_bubble_sort(tmp_dir) -> None:
    code = """import unittest

from code_to_optimize.bubble_sort import sorter


class TestPigLatin(unittest.TestCase):
    def test_sort(self):
        input = [5, 4, 3, 2, 1, 0]
        output = sorter(input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])

        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = sorter(input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        input = list(reversed(range(5000)))
        self.assertEqual(sorter(input), list(range(5000)))
"""
    imports = """import gc
import inspect
import os
import sqlite3
import time
import unittest

import dill as pickle"""
    # timeout_decorator no longer used since pytest handles timeouts

    imports += "\n\nfrom code_to_optimize.bubble_sort import sorter"
    
    wrapper_func = codeflash_wrap_string
    
    test_class_header = "class TestPigLatin(unittest.TestCase):"
    test_decorator = ""  # pytest-timeout handles timeouts now, not timeout_decorator
    
    expected = imports + "\n\n\n" + wrapper_func + "\n" + test_class_header + "\n\n"
    if test_decorator:
        expected += test_decorator + "\n"
    expected += """    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        input = [5, 4, 3, 2, 1, 0]
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(5000)))
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        self.assertEqual(codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '7', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs), list(range(5000)))
        codeflash_con.close()
"""

    with (tmp_dir / "test_sort.py").open("w") as f:
        f.write(code)
        f.flush()
        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=Path(f.name))
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name),
            [CodePosition(9, 17), CodePosition(13, 17), CodePosition(17, 17)],
            func,
            Path(f.name).parent,
        )
        os.chdir(original_cwd)
    assert success
    assert new_test.replace('"', "'") == expected.format(
        module_path=Path(f.name).stem, tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
    ).replace('"', "'")


def test_perfinjector_only_replay_test(tmp_dir) -> None:
    code = """import dill as pickle
import pytest
from codeflash.tracing.replay_test import get_next_arg_and_return
from codeflash.validation.equivalence import compare_results
from packagename.ml.yolo.image_reshaping_utils import prepare_image_for_yolo as packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo
def test_prepare_image_for_yolo():
    for arg_val_pkl, return_val_pkl in get_next_arg_and_return('/home/saurabh/packagename/traces/first.trace', 3):
        args = pickle.loads(arg_val_pkl)
        return_val_1= pickle.loads(return_val_pkl)
        ret = packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo(**args)
        assert compare_results(return_val_1, ret)
"""
    expected = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import pytest
from packagename.ml.yolo.image_reshaping_utils import \\
    prepare_image_for_yolo as \\
    packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo

from codeflash.tracing.replay_test import get_next_arg_and_return
from codeflash.validation.equivalence import compare_results


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{{codeflash_test_module_name}}:{{codeflash_test_class_name}}:{{codeflash_test_name}}:{{codeflash_line_id}}:{{codeflash_loop_index}}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {{}}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{{codeflash_line_id}}_{{codeflash_test_index}}'
    """
    expected += """test_stdout_tag = f'{{codeflash_test_module_name}}:{{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}}{{codeflash_test_name}}:{{codeflash_function_name}}:{{codeflash_loop_index}}:{{invocation_id}}'
    """
    expected += """print(f'!$######{{test_stdout_tag}}######$!')
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{{test_stdout_tag}}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_prepare_image_for_yolo():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
"""
    if sys.version_info < (3, 11):
        expected += """    for (arg_val_pkl, return_val_pkl) in get_next_arg_and_return('/home/saurabh/packagename/traces/first.trace', 3):
"""
    else:
        expected += """    for arg_val_pkl, return_val_pkl in get_next_arg_and_return('/home/saurabh/packagename/traces/first.trace', 3):
"""
    expected += """        args = pickle.loads(arg_val_pkl)
        return_val_1 = pickle.loads(return_val_pkl)
        _call__bound__arguments = inspect.signature(packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo).bind(**args)
        _call__bound__arguments.apply_defaults()
        ret = codeflash_wrap(packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo, '{module_path}', None, 'test_prepare_image_for_yolo', 'packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo', '0_2', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        assert compare_results(return_val_1, ret)
    codeflash_con.close()
"""
    with (tmp_dir / "test_return_values.py").open("w") as f:
        f.write(code)
        f.flush()
        func = FunctionToOptimize(function_name="prepare_image_for_yolo", parents=[], file_path=Path("module.py"))
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name), [CodePosition(10, 14)], func, Path(f.name).parent
        )
        os.chdir(original_cwd)
    assert success
    assert new_test.replace('"', "'") == expected.format(
        module_path=Path(f.name).stem, tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
    ).replace('"', "'")


def test_perfinjector_bubble_sort_results() -> None:
    computed_fn_opt = False
    code = """from code_to_optimize.bubble_sort import sorter
import datetime


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    print(datetime.datetime.now().isoformat())
    output = sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"""

    expected = (
        """import datetime
import gc
import inspect
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
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    input = [5, 4, 3, 2, 1, 0]
    print(datetime.datetime.now().isoformat())
    _call__bound__arguments = inspect.signature(sorter).bind(input)
    _call__bound__arguments.apply_defaults()
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '2', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    _call__bound__arguments = inspect.signature(sorter).bind(input)
    _call__bound__arguments.apply_defaults()
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '5', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    codeflash_con.close()
"""
    )

    expected_perfonly = (
        """import datetime
import gc
import os
import time

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
def test_sort():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    input = [5, 4, 3, 2, 1, 0]
    print(datetime.datetime.now().isoformat())
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '2', codeflash_loop_index, input)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '5', codeflash_loop_index, input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
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
        code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
        tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
        project_root_path = (Path(__file__).parent / "..").resolve()
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            test_path,
            [CodePosition(8, 14), CodePosition(12, 14)],
            func,
            project_root_path,
            mode=TestingMode.BEHAVIOR,
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")

        success, new_perf_test = inject_profiling_into_existing_test(
            test_path,
            [CodePosition(8, 14), CodePosition(12, 14)],
            func,
            project_root_path,
            mode=TestingMode.PERFORMANCE,
        )
        assert success
        assert new_perf_test is not None
        assert new_perf_test.replace('"', "'") == expected_perfonly.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")

        with test_path.open("w") as f:
            f.write(new_test)

        # Overwrite old test with new instrumented test

        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
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
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
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
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "5_0"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass

        with test_path_perf.open("w") as f:
            f.write(new_perf_test)

        test_results_perf, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results_perf[0].id.function_getting_tested == "sorter"
        assert test_results_perf[0].id.iteration_id == "2_0"
        assert test_results_perf[0].id.test_class_name is None
        assert test_results_perf[0].id.test_function_name == "test_sort"
        assert (
            test_results_perf[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp"
        )
        assert test_results_perf[0].runtime > 0
        assert test_results_perf[0].did_pass
        assert test_results_perf[0].return_value is None
        assert (
            test_results_perf[0].stdout
            == """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""
        )

        assert test_results_perf[1].id.function_getting_tested == "sorter"
        assert test_results_perf[1].id.iteration_id == "5_0"
        assert test_results_perf[1].id.test_class_name is None
        assert test_results_perf[1].id.test_function_name == "test_sort"
        assert (
            test_results_perf[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp"
        )
        assert test_results_perf[1].runtime > 0
        assert test_results_perf[1].did_pass

        out_str = """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        assert test_results_perf[1].stdout == out_str
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        computed_fn_opt = True
        line_profiler_output_file = add_decorator_imports(func_optimizer.function_to_optimize, code_context)
        line_profile_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.LINE_PROFILE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
            line_profiler_output_file=line_profiler_output_file,
        )
        tmp_lpr = list(line_profile_results["timings"].keys())
        if sys.platform != "win32":
            assert len(tmp_lpr) == 1 and line_profile_results["timings"][tmp_lpr[0]][0][1] == 2
    finally:
        if computed_fn_opt:
            func_optimizer.write_code_and_helpers(
                func_optimizer.function_to_optimize_source_code,
                original_helper_code,
                func_optimizer.function_to_optimize.file_path,
            )
        test_path.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)


def test_perfinjector_bubble_sort_parametrized_results() -> None:
    computed_fn_opt = False
    code = """from code_to_optimize.bubble_sort import sorter
import pytest


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(50))), list(range(50))),
    ],
)
def test_sort_parametrized(input, expected_output):
    output = sorter(input)
    assert output == expected_output
"""
    expected = (
        """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import pytest

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_string
        + """
@pytest.mark.parametrize('input, expected_output', [([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
def test_sort_parametrized(input, expected_output):
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(sorter).bind(input)
    _call__bound__arguments.apply_defaults()
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort_parametrized', 'sorter', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert output == expected_output
    codeflash_con.close()
"""
    )

    expected_perfonly = (
        """import gc
import os
import time

import pytest

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
@pytest.mark.parametrize('input, expected_output', [([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
def test_sort_parametrized(input, expected_output):
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort_parametrized', 'sorter', '0', codeflash_loop_index, input)
    assert output == expected_output
"""
    )
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_parametrized_results_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_parametrized_results_temp_perf.py"
    ).resolve()
    try:
        with open(test_path, "w") as f:
            f.write(code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            test_path, [CodePosition(14, 13)], func, project_root_path, mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(14, 13)], func, project_root_path, mode=TestingMode.PERFORMANCE
        )

        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix(),
        ).replace('"', "'")
        assert new_test_perf.replace('"', "'") == expected_perfonly.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix(),
        ).replace('"', "'")
        #
        # Overwrite old test with new instrumented test

        with test_path.open("w") as f:
            f.write(new_test)
        with test_path_perf.open("w") as f:
            f.write(new_test_perf)
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
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
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "0_0"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_function_name == "test_sort_parametrized"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert (
            test_results[0].stdout
            == """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""
        )

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "0_1"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_function_name == "test_sort_parametrized"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass
        assert (
            test_results[1].stdout
            == """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        )

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "0_2"
        assert test_results[2].id.test_class_name is None
        assert test_results[2].id.test_function_name == "test_sort_parametrized"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass

        test_results_perf, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results_perf[0].id.function_getting_tested == "sorter"
        assert test_results_perf[0].id.iteration_id == "0_0"
        assert test_results_perf[0].id.test_class_name is None
        assert test_results_perf[0].id.test_function_name == "test_sort_parametrized"
        assert (
            test_results_perf[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp"
        )
        assert test_results_perf[0].runtime > 0
        assert test_results_perf[0].did_pass
        assert test_results_perf[0].return_value is None

        assert test_results_perf[1].id.function_getting_tested == "sorter"
        assert test_results_perf[1].id.iteration_id == "0_1"
        assert test_results_perf[1].id.test_class_name is None
        assert test_results_perf[1].id.test_function_name == "test_sort_parametrized"
        assert (
            test_results_perf[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp"
        )
        assert test_results_perf[1].runtime > 0
        assert test_results_perf[1].did_pass

        out_str = """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        assert out_str == test_results_perf[1].stdout

        assert test_results_perf[2].id.function_getting_tested == "sorter"
        assert test_results_perf[2].id.iteration_id == "0_2"
        assert test_results_perf[2].id.test_class_name is None
        assert test_results_perf[2].id.test_function_name == "test_sort_parametrized"
        assert (
            test_results_perf[2].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp"
        )
        assert test_results_perf[2].runtime > 0
        assert test_results_perf[2].did_pass
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        computed_fn_opt = True
        line_profiler_output_file = add_decorator_imports(func_optimizer.function_to_optimize, code_context)
        line_profile_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.LINE_PROFILE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
            line_profiler_output_file=line_profiler_output_file,
        )
        tmp_lpr = list(line_profile_results["timings"].keys())
        if sys.platform != "win32":
            assert len(tmp_lpr) == 1 and line_profile_results["timings"][tmp_lpr[0]][0][1] == 3
    finally:
        if computed_fn_opt:
            func_optimizer.write_code_and_helpers(
                func_optimizer.function_to_optimize_source_code,
                original_helper_code,
                func_optimizer.function_to_optimize.file_path,
            )
        test_path.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)


def test_perfinjector_bubble_sort_parametrized_loop_results() -> None:
    computed_fn_opt = False
    code = """from code_to_optimize.bubble_sort import sorter
import pytest


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(50))), list(range(50))),
    ],
)
def test_sort_parametrized_loop(input, expected_output):
    for i in range(2):
        output = sorter(input)
        assert output == expected_output
"""
    expected = (
        """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import pytest

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_string
        + """
@pytest.mark.parametrize('input, expected_output', [([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
def test_sort_parametrized_loop(input, expected_output):
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    for i in range(2):
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort_parametrized_loop', 'sorter', '0_0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        assert output == expected_output
    codeflash_con.close()
"""
    )
    expected_perf = (
        """import gc
import os
import time

import pytest

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
@pytest.mark.parametrize('input, expected_output', [([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
def test_sort_parametrized_loop(input, expected_output):
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    for i in range(2):
        output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort_parametrized_loop', 'sorter', '0_0', codeflash_loop_index, input)
        assert output == expected_output
"""
    )
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_parametrized_loop_results_temp.py"
    ).resolve()
    test_path_behavior = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_parametrized_loop_results_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_parametrized_loop_results_temp_perf.py"
    ).resolve()
    try:
        with open(test_path, "w") as f:
            f.write(code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            test_path, [CodePosition(15, 17)], func, project_root_path, mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(15, 17)], func, project_root_path, mode=TestingMode.PERFORMANCE
        )

        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")

        # Overwrite old test with new instrumented test
        with test_path_behavior.open("w") as f:
            f.write(new_test)

        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")

        # Overwrite old test with new instrumented test
        with test_path_perf.open("w") as f:
            f.write(new_test_perf)

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path_behavior,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                    tests_in_file=[
                        TestsInFile(
                            test_file=test_path,
                            test_class=None,
                            test_function="test_sort_parametrized_loop",
                            test_type=TestType.EXISTING_UNIT_TEST,
                        )
                    ],
                )
            ]
        )

        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "0_0_0"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""
        assert test_results[0].stdout == out_str
        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "0_0_1"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass
        assert test_results[1].stdout == out_str

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "0_0_2"
        assert test_results[2].id.test_class_name is None
        assert test_results[2].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        assert test_results[2].stdout == out_str

        assert test_results[3].id.function_getting_tested == "sorter"
        assert test_results[3].id.iteration_id == "0_0_3"
        assert test_results[3].id.test_class_name is None
        assert test_results[3].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[3].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[3].runtime > 0
        assert test_results[3].did_pass

        assert test_results[3].stdout == out_str

        assert test_results[4].id.function_getting_tested == "sorter"
        assert test_results[4].id.iteration_id == "0_0_4"
        assert test_results[4].id.test_class_name is None
        assert test_results[4].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[4].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[4].runtime > 0
        assert test_results[4].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
"""
        assert test_results[4].stdout == out_str

        assert test_results[5].id.function_getting_tested == "sorter"
        assert test_results[5].id.iteration_id == "0_0_5"
        assert test_results[5].id.test_class_name is None
        assert test_results[5].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[5].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[5].runtime > 0
        assert test_results[5].did_pass
        assert test_results[5].stdout == out_str

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "0_0_0"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value is None

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "0_0_1"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass
        assert test_results[1].return_value is None

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "0_0_2"
        assert test_results[2].id.test_class_name is None
        assert test_results[2].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        assert test_results[2].return_value is None

        assert test_results[3].id.function_getting_tested == "sorter"
        assert test_results[3].id.iteration_id == "0_0_3"
        assert test_results[3].id.test_class_name is None
        assert test_results[3].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[3].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[3].runtime > 0
        assert test_results[3].did_pass
        assert test_results[3].return_value is None

        assert test_results[4].id.function_getting_tested == "sorter"
        assert test_results[4].id.iteration_id == "0_0_4"
        assert test_results[4].id.test_class_name is None
        assert test_results[4].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[4].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[4].runtime > 0
        assert test_results[4].did_pass
        assert test_results[4].return_value is None

        assert test_results[5].id.function_getting_tested == "sorter"
        assert test_results[5].id.iteration_id == "0_0_5"
        assert test_results[5].id.test_class_name is None
        assert test_results[5].id.test_function_name == "test_sort_parametrized_loop"
        assert (
            test_results[5].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp"
        )
        assert test_results[5].runtime > 0
        assert test_results[5].did_pass
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        computed_fn_opt = True
        line_profiler_output_file = add_decorator_imports(func_optimizer.function_to_optimize, code_context)
        line_profile_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.LINE_PROFILE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
            line_profiler_output_file=line_profiler_output_file,
        )
        tmp_lpr = list(line_profile_results["timings"].keys())
        if sys.platform != "win32":
            assert len(tmp_lpr) == 1 and line_profile_results["timings"][tmp_lpr[0]][0][1] == 6
    finally:
        if computed_fn_opt:
            func_optimizer.write_code_and_helpers(
                func_optimizer.function_to_optimize_source_code,
                original_helper_code,
                func_optimizer.function_to_optimize.file_path,
            )
        test_path.unlink(missing_ok=True)
        test_path_behavior.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)


def test_perfinjector_bubble_sort_loop_results() -> None:
    computed_fn_opt = False
    code = """from code_to_optimize.bubble_sort import sorter


def test_sort():
    inputs = [[5, 4, 3, 2, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], list(reversed(range(50)))]
    expected_outputs = [[0, 1, 2, 3, 4, 5], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(50))]

    for i in range(3):
        input = inputs[i]
        expected_output = expected_outputs[i]
        output = sorter(input)
        assert output == expected_output"""

    expected = (
        """import gc
import inspect
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
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    inputs = [[5, 4, 3, 2, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], list(reversed(range(50)))]
    expected_outputs = [[0, 1, 2, 3, 4, 5], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(50))]
    for i in range(3):
        input = inputs[i]
        expected_output = expected_outputs[i]
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '2_2', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        assert output == expected_output
    codeflash_con.close()
"""
    )

    expected_perf = (
        """import gc
import os
import time

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
def test_sort():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    inputs = [[5, 4, 3, 2, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], list(reversed(range(50)))]
    expected_outputs = [[0, 1, 2, 3, 4, 5], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(50))]
    for i in range(3):
        input = inputs[i]
        expected_output = expected_outputs[i]
        output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '2_2', codeflash_loop_index, input)
        assert output == expected_output
"""
    )
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_loop_results_temp.py"
    ).resolve()
    test_path_behavior = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_loop_results_temp_behavior.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_loop_results_temp_perf.py"
    ).resolve()
    try:
        with test_path.open("w") as f:
            f.write(code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
        os.chdir(str(run_cwd))
        success, new_test_behavior = inject_profiling_into_existing_test(
            test_path, [CodePosition(11, 17)], func, project_root_path, mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(11, 17)], func, project_root_path, mode=TestingMode.PERFORMANCE
        )
        os.chdir(original_cwd)
        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")

        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")

        # Overwrite old test with new instrumented test

        with test_path_behavior.open("w") as f:
            f.write(new_test_behavior)
        with test_path_perf.open("w") as f:
            f.write(new_test_perf)
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path_behavior,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                    tests_in_file=[
                        TestsInFile(
                            test_file=test_path,
                            test_class=None,
                            test_function="test_sort",
                            test_type=TestType.EXISTING_UNIT_TEST,
                        )
                    ],
                )
            ]
        )

        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "2_2_0"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "2_2_1"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "2_2_2"
        assert test_results[2].id.test_class_name is None
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "2_2_0"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value is None
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""

        assert test_results[0].stdout == out_str
        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "2_2_1"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass
        out_str2 = """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        assert test_results[1].stdout == out_str2

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "2_2_2"
        assert test_results[2].id.test_class_name is None
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        out_str3 = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
"""
        assert test_results[2].stdout == out_str3
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        computed_fn_opt = True
        line_profiler_output_file = add_decorator_imports(func_optimizer.function_to_optimize, code_context)
        line_profile_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.LINE_PROFILE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
            line_profiler_output_file=line_profiler_output_file,
        )
        tmp_lpr = list(line_profile_results["timings"].keys())
        if sys.platform != "win32":
            assert len(tmp_lpr) == 1 and line_profile_results["timings"][tmp_lpr[0]][0][1] == 3
    finally:
        if computed_fn_opt is True:
            func_optimizer.write_code_and_helpers(
                func_optimizer.function_to_optimize_source_code,
                original_helper_code,
                func_optimizer.function_to_optimize.file_path,
            )
        test_path.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)
        test_path_behavior.unlink(missing_ok=True)


def test_perfinjector_bubble_sort_unittest_results() -> None:
    
    code = """import unittest

from code_to_optimize.bubble_sort import sorter


class TestPigLatin(unittest.TestCase):
    def test_sort(self):
        input = [5, 4, 3, 2, 1, 0]
        output = sorter(input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])

        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = sorter(input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        input = list(reversed(range(50)))
        output = sorter(input)
        self.assertEqual(output, list(range(50)))
"""

    is_windows = platform.system() == "Windows"
    
    if is_windows:
        expected = (
            """import gc
import inspect
import os
import sqlite3
import time
import unittest

import dill as pickle

from code_to_optimize.bubble_sort import sorter


"""
            + codeflash_wrap_string
            + """
class TestPigLatin(unittest.TestCase):

    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        input = [5, 4, 3, 2, 1, 0]
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(50)))
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '7', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, list(range(50)))
        codeflash_con.close()
"""
        )
        expected_perf = (
            """import gc
import os
import time
import unittest

from code_to_optimize.bubble_sort import sorter


"""
            + codeflash_wrap_perfonly_string
            + """
class TestPigLatin(unittest.TestCase):

    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        input = [5, 4, 3, 2, 1, 0]
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '1', codeflash_loop_index, input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '4', codeflash_loop_index, input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(50)))
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '7', codeflash_loop_index, input)
        self.assertEqual(output, list(range(50)))
"""
        )
    else:
        expected = (
            """import gc
import inspect
import os
import sqlite3
import time
import unittest

import dill as pickle

from code_to_optimize.bubble_sort import sorter


"""
            + codeflash_wrap_string
            + """
class TestPigLatin(unittest.TestCase):

    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        input = [5, 4, 3, 2, 1, 0]
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(50)))
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '7', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, list(range(50)))
        codeflash_con.close()
"""
        )
        expected_perf = (
            """import gc
import os
import time
import unittest

from code_to_optimize.bubble_sort import sorter


"""
            + codeflash_wrap_perfonly_string
            + """
class TestPigLatin(unittest.TestCase):

    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        input = [5, 4, 3, 2, 1, 0]
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '1', codeflash_loop_index, input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '4', codeflash_loop_index, input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(50)))
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '7', codeflash_loop_index, input)
        self.assertEqual(output, list(range(50)))
"""
        )
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_results_temp.py"
    ).resolve()
    test_path_behavior = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_results_temp_behavior.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_results_temp_perf.py"
    ).resolve()
    try:
        with test_path.open("w") as f:
            f.write(code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/unittest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
        os.chdir(run_cwd)
        success, new_test_behavior = inject_profiling_into_existing_test(
            test_path,
            [CodePosition(9, 17), CodePosition(13, 17), CodePosition(17, 17)],
            func,
            project_root_path,
            mode=TestingMode.BEHAVIOR,
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path,
            [CodePosition(9, 17), CodePosition(13, 17), CodePosition(17, 17)],
            func,
            project_root_path,
            mode=TestingMode.PERFORMANCE,
        )
        os.chdir(original_cwd)

        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")
        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")
        #
        # Overwrite old test with new instrumented test
        with test_path_behavior.open("w") as f:
            f.write(new_test_behavior)
        with test_path_perf.open("w") as f:
            f.write(new_test_perf)

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path_behavior,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                    tests_in_file=[
                        TestsInFile(
                            test_file=test_path,
                            test_class="TestPigLatin",
                            test_function="test_sort",
                            test_type=TestType.EXISTING_UNIT_TEST,
                        )
                    ],
                )
            ]
        )
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="unittest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "1_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""
        assert test_results[0].stdout == out_str

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "4_0"
        assert test_results[1].id.test_class_name == "TestPigLatin"
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        assert test_results[1].stdout == out_str

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "7_0"
        assert test_results[2].id.test_class_name == "TestPigLatin"
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "1_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value is None
        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "4_0"
        assert test_results[1].id.test_class_name == "TestPigLatin"
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "7_0"
        assert test_results[2].id.test_class_name == "TestPigLatin"
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
"""
        assert test_results[2].stdout == out_str
    finally:
        test_path.unlink(missing_ok=True)
        test_path_behavior.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)


def test_perfinjector_bubble_sort_unittest_parametrized_results() -> None:
    code = """import unittest
from parameterized import parameterized

from code_to_optimize.bubble_sort import sorter


class TestPigLatin(unittest.TestCase):
    @parameterized.expand(
        [
            ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
            ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            (list(reversed(range(50))), list(range(50))),
        ]
    )
    def test_sort(self, input, expected_output):
        output = sorter(input)
        self.assertEqual(output, expected_output)
"""

    # Build expected behavior output with platform-aware imports
    imports_behavior = build_expected_unittest_imports("from parameterized import parameterized")
    imports_behavior += "\n\nfrom code_to_optimize.bubble_sort import sorter"
    
    test_decorator_behavior = ""  # pytest-timeout handles timeouts now
    test_class_behavior = """class TestPigLatin(unittest.TestCase):

    @parameterized.expand([([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
"""
    if test_decorator_behavior:
        test_class_behavior += test_decorator_behavior + "\n"
    test_class_behavior += """    def test_sort(self, input, expected_output):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        self.assertEqual(output, expected_output)
        codeflash_con.close()
"""
    
    expected_behavior = imports_behavior + "\n\n\n" + codeflash_wrap_string + "\n" + test_class_behavior
    # Build expected perf output with platform-aware imports
    imports_perf = """import gc
import os
import time
import unittest
"""
    # pytest-timeout handles timeouts now, no timeout_decorator needed
    imports_perf += "\nfrom parameterized import parameterized\n\nfrom code_to_optimize.bubble_sort import sorter"
    
    test_decorator_perf = ""  # pytest-timeout handles timeouts now
    test_class_perf = """class TestPigLatin(unittest.TestCase):

    @parameterized.expand([([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
"""
    if test_decorator_perf:
        test_class_perf += test_decorator_perf + "\n"
    test_class_perf += """    def test_sort(self, input, expected_output):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '0', codeflash_loop_index, input)
        self.assertEqual(output, expected_output)
"""
    
    expected_perf = imports_perf + "\n\n\n" + codeflash_wrap_perfonly_string + "\n" + test_class_perf
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_parametrized_results_temp.py"
    ).resolve()
    test_path_behavior = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_parametrized_results_temp_behavior.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_parametrized_results_temp_perf.py"
    ).resolve()
    try:
        with test_path.open("w") as f:
            f.write(code)
        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/unittest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
        os.chdir(run_cwd)
        success, new_test_behavior = inject_profiling_into_existing_test(
            test_path, [CodePosition(16, 17)], func, project_root_path, mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(16, 17)], func, project_root_path, mode=TestingMode.PERFORMANCE
        )

        os.chdir(original_cwd)
        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected_behavior.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")

        assert new_test_perf is not None
        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")

        #
        # Overwrite old test with new instrumented test
        with test_path_behavior.open("w") as f:
            f.write(new_test_behavior)
        with test_path_perf.open("w") as f:
            f.write(new_test_perf)
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path_behavior,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                    tests_in_file=[
                        TestsInFile(
                            test_file=test_path,
                            test_class="TestPigLatin",
                            test_function="test_sort",
                            test_type=TestType.EXISTING_UNIT_TEST,
                        )
                    ],
                )
            ]
        )
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="unittest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "0_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""
        assert test_results[0].stdout == out_str

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "0_1"
        assert test_results[1].id.test_class_name == "TestPigLatin"
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        assert test_results[1].stdout == out_str

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "0_2"
        assert test_results[2].id.test_class_name == "TestPigLatin"
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
"""
        assert test_results[2].stdout == out_str

        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "0_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value is None

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "0_1"
        assert test_results[1].id.test_class_name == "TestPigLatin"
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "0_2"
        assert test_results[2].id.test_class_name == "TestPigLatin"
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass

    finally:
        test_path.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)
        test_path_behavior.unlink(missing_ok=True)


def test_perfinjector_bubble_sort_unittest_loop_results() -> None:
    code = """import unittest

from code_to_optimize.bubble_sort import sorter


class TestPigLatin(unittest.TestCase):
    def test_sort(self):
        inputs = [[5, 4, 3, 2, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], list(reversed(range(50)))]
        expected_outputs = [[0, 1, 2, 3, 4, 5], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(50))]

        for i in range(3):
            input = inputs[i]
            expected_output = expected_outputs[i]
            output = sorter(input)
            self.assertEqual(output, expected_output)"""

    # Build expected behavior output with platform-aware imports  
    imports_behavior = build_expected_unittest_imports()
    imports_behavior += "\n\nfrom code_to_optimize.bubble_sort import sorter"
    
    test_decorator_behavior = ""  # pytest-timeout handles timeouts now
    test_class_behavior = """class TestPigLatin(unittest.TestCase):

"""
    if test_decorator_behavior:
        test_class_behavior += test_decorator_behavior + "\n"
    test_class_behavior += """    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        inputs = [[5, 4, 3, 2, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], list(reversed(range(50)))]
        expected_outputs = [[0, 1, 2, 3, 4, 5], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(50))]
        for i in range(3):
            input = inputs[i]
            expected_output = expected_outputs[i]
            _call__bound__arguments = inspect.signature(sorter).bind(input)
            _call__bound__arguments.apply_defaults()
            output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '2_2', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
            self.assertEqual(output, expected_output)
        codeflash_con.close()
"""

    expected_behavior = imports_behavior + "\n\n\n" + codeflash_wrap_string + "\n" + test_class_behavior

    # Build expected perf output with platform-aware imports
    imports_perf = """import gc
import os
import time
import unittest
"""
    # pytest-timeout handles timeouts now, no timeout_decorator needed
    imports_perf += "\nfrom code_to_optimize.bubble_sort import sorter"
    
    test_decorator_perf = ""  # pytest-timeout handles timeouts now
    test_class_perf = """class TestPigLatin(unittest.TestCase):

"""
    if test_decorator_perf:
        test_class_perf += test_decorator_perf + "\n"
    test_class_perf += """    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        inputs = [[5, 4, 3, 2, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], list(reversed(range(50)))]
        expected_outputs = [[0, 1, 2, 3, 4, 5], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(50))]
        for i in range(3):
            input = inputs[i]
            expected_output = expected_outputs[i]
            output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '2_2', codeflash_loop_index, input)
            self.assertEqual(output, expected_output)
"""
    
    expected_perf = imports_perf + "\n\n\n" + codeflash_wrap_perfonly_string + "\n" + test_class_perf
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_loop_results_temp.py"
    ).resolve()
    test_path_behavior = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_loop_results_temp_behavior.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_loop_results_temp_perf.py"
    ).resolve()
    try:
        with test_path.open("w") as f:
            f.write(code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/unittest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
        os.chdir(run_cwd)
        success, new_test_behavior = inject_profiling_into_existing_test(
            test_path, [CodePosition(14, 21)], func, project_root_path, mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(14, 21)], func, project_root_path, mode=TestingMode.PERFORMANCE
        )
        os.chdir(original_cwd)
        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected_behavior.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")
        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")
        #
        # # Overwrite old test with new instrumented test
        with test_path_behavior.open("w") as f:
            f.write(new_test_behavior)
        with test_path_perf.open("w") as f:
            f.write(new_test_perf)
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path_behavior,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                    tests_in_file=[
                        TestsInFile(
                            test_file=test_path,
                            test_class="TestPigLatin",
                            test_function="test_sort",
                            test_type=TestType.EXISTING_UNIT_TEST,
                        )
                    ],
                )
            ]
        )
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="unittest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            test_env=test_env,
            testing_type=TestingMode.BEHAVIOR,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "2_2_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""
        assert test_results[0].stdout == out_str

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "2_2_1"
        assert test_results[1].id.test_class_name == "TestPigLatin"
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        assert test_results[1].stdout == out_str

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "2_2_2"
        assert test_results[2].id.test_class_name == "TestPigLatin"
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
"""
        assert test_results[2].stdout == out_str

        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            test_env=test_env,
            testing_type=TestingMode.PERFORMANCE,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "2_2_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value is None

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "2_2_1"
        assert test_results[1].id.test_class_name == "TestPigLatin"
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "2_2_2"
        assert test_results[2].id.test_class_name == "TestPigLatin"
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
    finally:
        test_path.unlink(missing_ok=True)
        test_path_behavior.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)


def test_perfinjector_bubble_sort_unittest_parametrized_loop_results() -> None:
    code = """import unittest
from parameterized import parameterized

from code_to_optimize.bubble_sort import sorter


class TestPigLatin(unittest.TestCase):
    @parameterized.expand(
        [
            ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
            ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            (list(reversed(range(50))), list(range(50))),
        ]
    )
    def test_sort(self, input, expected_output):
        for i in range(2):
            output = sorter(input)
            self.assertEqual(output, expected_output)
"""

    # Build expected behavior output with platform-aware imports
    imports_behavior = build_expected_unittest_imports("from parameterized import parameterized")
    imports_behavior += "\n\nfrom code_to_optimize.bubble_sort import sorter"
    
    test_decorator_behavior = ""  # pytest-timeout handles timeouts now
    test_class_behavior = """class TestPigLatin(unittest.TestCase):

    @parameterized.expand([([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
"""
    if test_decorator_behavior:
        test_class_behavior += test_decorator_behavior + "\n"
    test_class_behavior += """    def test_sort(self, input, expected_output):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        for i in range(2):
            _call__bound__arguments = inspect.signature(sorter).bind(input)
            _call__bound__arguments.apply_defaults()
            output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '0_0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
            self.assertEqual(output, expected_output)
        codeflash_con.close()
"""

    expected_behavior = imports_behavior + "\n\n\n" + codeflash_wrap_string + "\n" + test_class_behavior
    # Build expected perf output with platform-aware imports
    imports_perf = """import gc
import os
import time
import unittest
"""
    # pytest-timeout handles timeouts now, no timeout_decorator needed
    imports_perf += "\nfrom parameterized import parameterized\n\nfrom code_to_optimize.bubble_sort import sorter"
    
    test_decorator_perf = ""  # pytest-timeout handles timeouts now
    test_class_perf = """class TestPigLatin(unittest.TestCase):

    @parameterized.expand([([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
"""
    if test_decorator_perf:
        test_class_perf += test_decorator_perf + "\n"
    test_class_perf += """    def test_sort(self, input, expected_output):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        for i in range(2):
            output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '0_0', codeflash_loop_index, input)
            self.assertEqual(output, expected_output)
"""
    
    expected_perf = imports_perf + "\n\n\n" + codeflash_wrap_perfonly_string + "\n" + test_class_perf
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp.py"
    ).resolve()
    test_path_behavior = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp_behavior.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp_perf.py"
    ).resolve()
    try:
        with test_path.open("w") as _f:
            _f.write(code)
        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/unittest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()

        f = FunctionToOptimize(function_name="sorter", file_path=code_path, parents=[])
        os.chdir(run_cwd)
        success, new_test_behavior = inject_profiling_into_existing_test(
            test_path, [CodePosition(17, 21)], f, project_root_path, mode=TestingMode.BEHAVIOR
        )
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(17, 21)], f, project_root_path, mode=TestingMode.PERFORMANCE
        )
        os.chdir(original_cwd)
        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected_behavior.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")
        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")
        #
        # Overwrite old test with new instrumented test
        with test_path_behavior.open("w") as _f:
            _f.write(new_test_behavior)

        with test_path_perf.open("w") as _f:
            _f.write(new_test_perf)

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path_behavior,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                    tests_in_file=[
                        TestsInFile(
                            test_file=test_path,
                            test_class="TestPigLatin",
                            test_function="test_sort",
                            test_type=TestType.EXISTING_UNIT_TEST,
                        )
                    ],
                )
            ]
        )
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="unittest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=f, test_cfg=test_config)
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "0_0_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value == ([0, 1, 2, 3, 4, 5],)
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""
        assert test_results[0].stdout == out_str

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "0_0_1"
        assert test_results[1].id.test_class_name == "TestPigLatin"
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]
"""
        assert test_results[1].stdout == out_str

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "0_0_2"
        assert test_results[2].id.test_class_name == "TestPigLatin"
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
"""
        assert test_results[2].stdout == out_str

        assert test_results[3].id.function_getting_tested == "sorter"
        assert test_results[3].id.iteration_id == "0_0_3"
        assert test_results[3].id.test_class_name == "TestPigLatin"
        assert test_results[3].id.test_function_name == "test_sort"
        assert (
            test_results[3].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[3].runtime > 0
        assert test_results[3].did_pass

        assert test_results[4].id.function_getting_tested == "sorter"
        assert test_results[4].id.iteration_id == "0_0_4"
        assert test_results[4].id.test_class_name == "TestPigLatin"
        assert test_results[4].id.test_function_name == "test_sort"
        assert (
            test_results[4].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[4].runtime > 0
        assert test_results[4].did_pass

        assert test_results[5].id.function_getting_tested == "sorter"
        assert test_results[5].id.iteration_id == "0_0_5"
        assert test_results[5].id.test_class_name == "TestPigLatin"
        assert test_results[5].id.test_function_name == "test_sort"
        assert (
            test_results[5].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[5].runtime > 0
        assert test_results[5].did_pass
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert test_results[0].id.function_getting_tested == "sorter"
        assert test_results[0].id.iteration_id == "0_0_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sort"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[0].runtime > 0
        assert test_results[0].did_pass
        assert test_results[0].return_value is None

        assert test_results[1].id.function_getting_tested == "sorter"
        assert test_results[1].id.iteration_id == "0_0_1"
        assert test_results[1].id.test_class_name == "TestPigLatin"
        assert test_results[1].id.test_function_name == "test_sort"
        assert (
            test_results[1].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[1].runtime > 0
        assert test_results[1].did_pass

        assert test_results[2].id.function_getting_tested == "sorter"
        assert test_results[2].id.iteration_id == "0_0_2"
        assert test_results[2].id.test_class_name == "TestPigLatin"
        assert test_results[2].id.test_function_name == "test_sort"
        assert (
            test_results[2].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[2].runtime > 0
        assert test_results[2].did_pass

        assert test_results[3].id.function_getting_tested == "sorter"
        assert test_results[3].id.iteration_id == "0_0_3"
        assert test_results[3].id.test_class_name == "TestPigLatin"
        assert test_results[3].id.test_function_name == "test_sort"
        assert (
            test_results[3].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[3].runtime > 0
        assert test_results[3].did_pass

        assert test_results[4].id.function_getting_tested == "sorter"
        assert test_results[4].id.iteration_id == "0_0_4"
        assert test_results[4].id.test_class_name == "TestPigLatin"
        assert test_results[4].id.test_function_name == "test_sort"
        assert (
            test_results[4].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[4].runtime > 0
        assert test_results[4].did_pass

        assert test_results[5].id.function_getting_tested == "sorter"
        assert test_results[5].id.iteration_id == "0_0_5"
        assert test_results[5].id.test_class_name == "TestPigLatin"
        assert test_results[5].id.test_function_name == "test_sort"
        assert (
            test_results[5].id.test_module_path
            == "code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp"
        )
        assert test_results[5].runtime > 0
        assert test_results[5].did_pass
    finally:
        test_path.unlink(missing_ok=True)
        test_path_behavior.unlink(missing_ok=True)
        test_path_perf.unlink(missing_ok=True)


def test_class_method_imported_as() -> None:
    code = """import functionA
import moduleB as module_B
from module import functionB as function_B
import class_name_B
from nuitka.nodes.ImportNodes import ExpressionBuiltinImport as nuitka_nodes_ImportNodes_ExpressionBuiltinImport
"""
    f = FunctionToOptimize(function_name="functionA", file_path=Path("module.py"), parents=[])
    tree = ast.parse(code)
    visitor = FunctionImportedAsVisitor(f)
    visitor.visit(tree)
    assert visitor.imported_as.function_name == "functionA"

    f = FunctionToOptimize(function_name="functionB", file_path=Path("module.py"), parents=[])
    visitor = FunctionImportedAsVisitor(f)
    visitor.visit(tree)
    assert visitor.imported_as.function_name == "function_B"

    f = FunctionToOptimize(
        function_name="method_name",
        file_path=Path("module.py"),
        parents=[FunctionParent("ExpressionBuiltinImport", "ClassDef")],
    )
    visitor = FunctionImportedAsVisitor(f)
    visitor.visit(tree)
    assert visitor.imported_as.qualified_name == "nuitka_nodes_ImportNodes_ExpressionBuiltinImport.method_name"

    f = FunctionToOptimize(function_name="class_name_B", file_path=Path("module.py"), parents=[])
    visitor = FunctionImportedAsVisitor(f)
    visitor.visit(tree)
    assert visitor.imported_as.qualified_name == "class_name_B"


def test_class_function_instrumentation() -> None:
    code = """from module import class_name as class_name_A

def test_class_name_A_function_name():
    ret = class_name_A.function_name(**args)
"""

    expected = (
        """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
from module import class_name as class_name_A


"""
        + codeflash_wrap_string
        + """
def test_class_name_A_function_name():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(class_name_A.function_name).bind(**args)
    _call__bound__arguments.apply_defaults()
    ret = codeflash_wrap(class_name_A.function_name, '{module_path}', None, 'test_class_name_A_function_name', 'class_name_A.function_name', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    codeflash_con.close()
"""
    )

    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_class_function_instrumentation_temp.py"
    )
    try:
        with open(test_path, "w") as f:
            f.write(code)

        project_root_path = Path(__file__).parent.resolve() / "../code_to_optimize/"
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()
        func = FunctionToOptimize(
            function_name="function_name",
            file_path=project_root_path / "module.py",
            parents=[FunctionParent("class_name", "ClassDef")],
        )
        os.chdir(str(run_cwd))
        success, new_test = inject_profiling_into_existing_test(
            test_path, [CodePosition(4, 23)], func, project_root_path
        )
        os.chdir(original_cwd)
    finally:
        test_path.unlink(missing_ok=True)
    assert success
    assert new_test is not None
    assert new_test.replace('"', "'") == expected.format(
        tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix(),
        module_path="tests.pytest.test_class_function_instrumentation_temp",
    ).replace('"', "'")


def test_wrong_function_instrumentation() -> None:
    code = """from codeflash.result.common_tags import find_common_tags


def test_common_tags_1():
    articles_1 = [1, 2, 3]

    assert find_common_tags(articles_1) == set(1, 2)

    articles_2 = [1, 2]

    assert find_common_tags(articles_2) == set(1)
"""

    expected = (
        """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle

from codeflash.result.common_tags import find_common_tags


"""
        + codeflash_wrap_string
        + """
def test_common_tags_1():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    articles_1 = [1, 2, 3]
    _call__bound__arguments = inspect.signature(find_common_tags).bind(articles_1)
    _call__bound__arguments.apply_defaults()
    assert codeflash_wrap(find_common_tags, '{module_path}', None, 'test_common_tags_1', 'find_common_tags', '1', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs) == set(1, 2)
    articles_2 = [1, 2]
    _call__bound__arguments = inspect.signature(find_common_tags).bind(articles_2)
    _call__bound__arguments.apply_defaults()
    assert codeflash_wrap(find_common_tags, '{module_path}', None, 'test_common_tags_1', 'find_common_tags', '3', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs) == set(1)
    codeflash_con.close()
"""
    )

    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_wrong_function_instrumentation_temp.py"
    )
    try:
        with test_path.open("w") as f:
            f.write(code)

        tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
        project_root_path = Path(__file__).parent.resolve() / "../code_to_optimize/"
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()
        func = FunctionToOptimize(
            function_name="find_common_tags", file_path=project_root_path / "module.py", parents=[]
        )

        os.chdir(str(run_cwd))
        success, new_test = inject_profiling_into_existing_test(
            test_path, [CodePosition(7, 11), CodePosition(11, 11)], func, project_root_path
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="tests.pytest.test_wrong_function_instrumentation_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix(),
        ).replace('"', "'")
    finally:
        test_path.unlink(missing_ok=True)


def test_conditional_instrumentation() -> None:
    code = """from code_to_optimize.bubble_sort import sorter


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    if len(input) > 0:
        assert sorter(input) == [0, 1, 2, 3, 4, 5]"""

    expected = (
        """import gc
import inspect
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
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    input = [5, 4, 3, 2, 1, 0]
    if len(input) > 0:
        _call__bound__arguments = inspect.signature(sorter).bind(input)
        _call__bound__arguments.apply_defaults()
        assert codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '1_0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs) == [0, 1, 2, 3, 4, 5]
    codeflash_con.close()
"""
    )
    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_conditional_instrumentation_temp.py"
    )
    try:
        with open(test_path, "w") as f:
            f.write(code)

        tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
        project_root_path = Path(__file__).parent.resolve() / "../code_to_optimize/"
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()
        func = FunctionToOptimize(function_name="sorter", file_path=project_root_path / "module.py", parents=[])

        os.chdir(str(run_cwd))
        success, new_test = inject_profiling_into_existing_test(
            test_path, [CodePosition(7, 15)], func, project_root_path
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="tests.pytest.test_conditional_instrumentation_temp",
tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        ).replace('"', "'")
    finally:
        test_path.unlink(missing_ok=True)


def test_static_method_instrumentation():
    code = """from code_to_optimize.bubble_sort import BubbleSorter


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    output = BubbleSorter.sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = BubbleSorter.sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"""

    expected = (
        """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle

from code_to_optimize.bubble_sort import BubbleSorter


"""
        + codeflash_wrap_string
        + """
def test_sort():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    input = [5, 4, 3, 2, 1, 0]
    _call__bound__arguments = inspect.signature(BubbleSorter.sorter).bind(input)
    _call__bound__arguments.apply_defaults()
    output = codeflash_wrap(BubbleSorter.sorter, 'tests.pytest.test_perfinjector_bubble_sort_results_temp', None, 'test_sort', 'BubbleSorter.sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    _call__bound__arguments = inspect.signature(BubbleSorter.sorter).bind(input)
    _call__bound__arguments.apply_defaults()
    output = codeflash_wrap(BubbleSorter.sorter, '{module_path}', None, 'test_sort', 'BubbleSorter.sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    codeflash_con.close()
"""
    )

    function_to_optimize = FunctionToOptimize(
        function_name="sorter",
        file_path=Path("/Users/renaud/repos/codeflash/cli/code_to_optimize/bubble_sort.py"),
        parents=[FunctionParent("BubbleSorter", "ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_perfinjector_bubble_sort_results_temp.py"
    )
    try:
        with test_path.open("w") as f:
            f.write(code)
        tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
        project_root_path = Path(__file__).parent.resolve() / "../code_to_optimize/"
        run_cwd = Path(__file__).parent.parent.resolve()
        original_cwd = Path.cwd()

        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            test_path, [CodePosition(6, 26), CodePosition(10, 26)], function_to_optimize, project_root_path
        )
        os.chdir(original_cwd)
        assert success
        formatted_expected = expected.format(
            module_path="tests.pytest.test_perfinjector_bubble_sort_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
        )
        assert new_test is not None
        assert new_test.replace('"', "'") == formatted_expected.replace('"', "'")
    finally:
        test_path.unlink(missing_ok=True)


def test_class_method_instrumentation(tmp_path: Path) -> None:
    code = """from codeflash.optimization.optimizer import Optimizer
def test_code_replacement10() -> None:
    get_code_output = '''random code'''
    file_path = Path(__file__).resolve()
    opt = Optimizer(
        Namespace(
            project_root=str(file_path.parent.resolve()),
            disable_telemetry=True,
            tests_root="tests",
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
        ),
    )
    func_top_optimize = FunctionToOptimize(
        function_name="main_method",
        file_path=str(file_path),
        parents=[FunctionParent("MainClass", "ClassDef")],
    )
    with open(file_path) as f:
        original_code = f.read()
        code_context = opt.get_code_optimization_context(
            function_to_optimize=func_top_optimize,
            project_root=str(file_path.parent),
            original_source_code=original_code,
        ).unwrap()
        assert code_context.testgen_context_code == get_code_output
        code_context = opt.get_code_optimization_context(
            function_to_optimize=func_top_optimize,
            project_root=str(file_path.parent),
            original_source_code=original_code,
        )
        assert code_context.testgen_context_code == get_code_output
    """

    expected = (
        """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle

from codeflash.optimization.optimizer import Optimizer


"""
        + codeflash_wrap_string
        + """
def test_code_replacement10() -> None:
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    get_code_output = 'random code'
    file_path = Path(__file__).resolve()
    opt = Optimizer(Namespace(project_root=str(file_path.parent.resolve()), disable_telemetry=True, tests_root='tests', test_framework='pytest', pytest_cmd='pytest', experiment_id=None))
    func_top_optimize = FunctionToOptimize(function_name='main_method', file_path=str(file_path), parents=[FunctionParent('MainClass', 'ClassDef')])
    with open(file_path) as f:
        original_code = f.read()
        _call__bound__arguments = inspect.signature(opt.get_code_optimization_context).bind(function_to_optimize=func_top_optimize, project_root=str(file_path.parent), original_source_code=original_code)
        _call__bound__arguments.apply_defaults()
        code_context = codeflash_wrap(opt.get_code_optimization_context, '{module_path}', None, 'test_code_replacement10', 'Optimizer.get_code_optimization_context', '4_1', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs).unwrap()
        assert code_context.testgen_context_code == get_code_output
        _call__bound__arguments = inspect.signature(opt.get_code_optimization_context).bind(function_to_optimize=func_top_optimize, project_root=str(file_path.parent), original_source_code=original_code)
        _call__bound__arguments.apply_defaults()
        code_context = codeflash_wrap(opt.get_code_optimization_context, '{module_path}', None, 'test_code_replacement10', 'Optimizer.get_code_optimization_context', '4_3', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
        assert code_context.testgen_context_code == get_code_output
    codeflash_con.close()
"""
    )

    test_file_path = tmp_path / "test_class_method_instrumentation.py"
    test_file_path.write_text(code, encoding="utf-8")
    
    func = FunctionToOptimize(
        function_name="get_code_optimization_context",
        parents=[FunctionParent("Optimizer", "ClassDef")],
        file_path=test_file_path,
    )
    original_cwd = Path.cwd()
    run_cwd = Path(__file__).parent.parent.resolve()
    os.chdir(run_cwd)
    success, new_test = inject_profiling_into_existing_test(
        test_file_path, [CodePosition(22, 28), CodePosition(28, 28)], func, test_file_path.parent
    )
    os.chdir(original_cwd)
    assert success
    assert new_test.replace('"', "'") == expected.replace('"', "'").format(
        module_path=test_file_path.stem, tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix()
    )


def test_time_correction_instrumentation() -> None:
    code = """from code_to_optimize.sleeptime import accurate_sleepfunc
import pytest
@pytest.mark.parametrize("n, expected_total_sleep_time", [
    (0.01, 0.010),
    (0.02, 0.020),
])
def test_sleepfunc_sequence_short(n, expected_total_sleep_time):
    output = accurate_sleepfunc(n)
    assert output == expected_total_sleep_time

"""

    expected = (
        """import gc
import os
import time

import pytest

from code_to_optimize.sleeptime import accurate_sleepfunc


"""
        + codeflash_wrap_perfonly_string
        + """
@pytest.mark.parametrize('n, expected_total_sleep_time', [(0.01, 0.01), (0.02, 0.02)])
def test_sleepfunc_sequence_short(n, expected_total_sleep_time):
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    output = codeflash_wrap(accurate_sleepfunc, '{module_path}', None, 'test_sleepfunc_sequence_short', 'accurate_sleepfunc', '0', codeflash_loop_index, n)
    assert output == expected_total_sleep_time
"""
    )
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/sleeptime.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_time_correction_instrumentation_temp.py"
    ).resolve()
    try:
        with test_path.open("w") as f:
            f.write(code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        func = FunctionToOptimize(function_name="accurate_sleepfunc", parents=[], file_path=code_path)
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            test_path, [CodePosition(8, 13)], func, project_root_path, mode=TestingMode.PERFORMANCE
        )
        os.chdir(original_cwd)

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        assert success, "Test instrumentation failed"
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_time_correction_instrumentation_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix(),
        ).replace('"', "'")
        # Overwrite old test with new instrumented test
        with test_path.open("w") as f:
            f.write(new_test)

        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path,
                )
            ]
        )
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=2,
            pytest_max_loops=2,
            testing_time=0.1,
        )

        assert test_results[0].id.function_getting_tested == "accurate_sleepfunc"
        assert test_results[0].id.iteration_id == "0_0"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_function_name == "test_sleepfunc_sequence_short"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_time_correction_instrumentation_temp"
        )

        assert len(test_results) == 4
        for i, test_result in enumerate(test_results):
            assert test_result.did_pass
            assert math.isclose(test_result.runtime, ((i % 2) + 1) * 100_000_000, rel_tol=0.01)

    finally:
        test_path.unlink(missing_ok=True)


def test_time_correction_instrumentation_unittest() -> None:
    code = """import unittest
from parameterized import parameterized

from code_to_optimize.sleeptime import accurate_sleepfunc

class TestPigLatin(unittest.TestCase):
    @parameterized.expand([
        (0.01, 0.010),
        (0.02, 0.020),
    ])
    def test_sleepfunc_sequence_short(self, n, expected_total_sleep_time):
        output = accurate_sleepfunc(n)
"""

    # Build expected output with platform-aware imports
    imports = """import gc
import os
import time
import unittest
"""
    # pytest-timeout handles timeouts now, no timeout_decorator needed
    imports += "\nfrom parameterized import parameterized\n\nfrom code_to_optimize.sleeptime import accurate_sleepfunc"
    
    test_decorator = ""  # pytest-timeout handles timeouts now
    test_class = """class TestPigLatin(unittest.TestCase):

    @parameterized.expand([(0.01, 0.01), (0.02, 0.02)])
"""
    if test_decorator:
        test_class += test_decorator + "\n"
    test_class += """    def test_sleepfunc_sequence_short(self, n, expected_total_sleep_time):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        output = codeflash_wrap(accurate_sleepfunc, '{module_path}', 'TestPigLatin', 'test_sleepfunc_sequence_short', 'accurate_sleepfunc', '0', codeflash_loop_index, n)
"""
    
    expected = imports + "\n\n\n" + codeflash_wrap_perfonly_string + "\n" + test_class
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/sleeptime.py").resolve()
    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/unittest/test_time_correction_instrumentation_unittest_temp.py"
    ).resolve()
    try:
        with test_path.open("w") as f:
            f.write(code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/unittest/").resolve()
        project_root_path = (Path(__file__).parent.resolve() / "../").resolve()
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        func = FunctionToOptimize(function_name="accurate_sleepfunc", parents=[], file_path=code_path)
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            test_path, [CodePosition(12, 17)], func, project_root_path, mode=TestingMode.PERFORMANCE
        )
        os.chdir(original_cwd)

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        assert success, "Test instrumentation failed"
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.unittest.test_time_correction_instrumentation_unittest_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix(),
        ).replace('"', "'")
        # Overwrite old test with new instrumented test
        with test_path.open("w") as f:
            f.write(new_test)

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path,
                    tests_in_file=[
                        TestsInFile(
                            test_file=test_path,
                            test_class="TestPigLatin",
                            test_function="test_sleepfunc_sequence_short",
                            test_type=TestType.EXISTING_UNIT_TEST,
                        )
                    ],
                )
            ]
        )
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="unittest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert test_results[0].id.function_getting_tested == "accurate_sleepfunc"
        assert test_results[0].id.iteration_id == "0_0"
        assert test_results[0].id.test_class_name == "TestPigLatin"
        assert test_results[0].id.test_function_name == "test_sleepfunc_sequence_short"
        assert (
            test_results[0].id.test_module_path
            == "code_to_optimize.tests.unittest.test_time_correction_instrumentation_unittest_temp"
        )

        assert len(test_results) == 2
        for i, test_result in enumerate(test_results):
            assert test_result.did_pass
            assert math.isclose(test_result.runtime, ((i % 2) + 1) * 100_000_000, rel_tol=0.01)

    finally:
        test_path.unlink(missing_ok=True)


# ============================================================================
# Device Synchronization Tests
# ============================================================================


class TestDeviceSyncStatements:
    """Tests for _create_device_sync_statements function."""

    def test_no_frameworks_returns_empty_list(self):
        """When no frameworks are specified, should return empty list."""
        result = _create_device_sync_statements(None)
        assert result == []

        result = _create_device_sync_statements(set())
        assert result == []

    def test_torch_sync_generates_cuda_and_mps_checks(self):
        """Torch framework should generate CUDA and MPS synchronization checks."""
        result = _create_device_sync_statements({"torch"}, for_return_value=False)
        assert len(result) == 1

        # Unparse the AST to check the generated code
        code = ast.unparse(result[0])
        assert "codeflash_torch.cuda.is_available()" in code
        assert "codeflash_torch.cuda.is_initialized()" in code
        assert "codeflash_torch.cuda.synchronize()" in code
        assert "codeflash_torch.backends" in code
        assert "mps" in code
        assert "codeflash_torch.mps.synchronize()" in code

    def test_torch_sync_same_for_pre_and_post(self):
        """Torch sync should be the same before and after function call."""
        pre_sync = _create_device_sync_statements({"torch"}, for_return_value=False)
        post_sync = _create_device_sync_statements({"torch"}, for_return_value=True)

        # Both should have torch sync
        assert len(pre_sync) == 1
        assert len(post_sync) == 1

        pre_code = ast.unparse(pre_sync[0])
        post_code = ast.unparse(post_sync[0])
        assert "codeflash_torch.cuda.synchronize()" in pre_code
        assert "codeflash_torch.cuda.synchronize()" in post_code

    def test_jax_sync_only_for_return_value(self):
        """JAX sync should only be generated for post-function call (for_return_value=True)."""
        pre_sync = _create_device_sync_statements({"jax"}, for_return_value=False)
        post_sync = _create_device_sync_statements({"jax"}, for_return_value=True)

        # Pre-sync should be empty for JAX (no pre-sync needed)
        assert len(pre_sync) == 0

        # Post-sync should have JAX block_until_ready
        assert len(post_sync) == 1
        code = ast.unparse(post_sync[0])
        assert "block_until_ready" in code
        assert "return_value" in code

    def test_jax_sync_checks_hasattr(self):
        """JAX sync should check if return_value has block_until_ready method."""
        post_sync = _create_device_sync_statements({"jax"}, for_return_value=True)
        code = ast.unparse(post_sync[0])

        assert "hasattr(return_value, 'block_until_ready')" in code
        assert "return_value.block_until_ready()" in code
        # Fallback to jax.block_until_ready
        assert "codeflash_jax" in code

    def test_tensorflow_sync_uses_sync_devices(self):
        """TensorFlow sync should use tf.test.experimental.sync_devices()."""
        pre_sync = _create_device_sync_statements({"tensorflow"}, for_return_value=False)
        post_sync = _create_device_sync_statements({"tensorflow"}, for_return_value=True)

        # Both pre and post should have TF sync
        assert len(pre_sync) == 1
        assert len(post_sync) == 1

        pre_code = ast.unparse(pre_sync[0])
        post_code = ast.unparse(post_sync[0])

        assert "codeflash_tf.test.experimental" in pre_code
        assert "sync_devices" in pre_code
        assert "codeflash_tf.test.experimental" in post_code
        assert "sync_devices" in post_code

    def test_tensorflow_sync_checks_hasattr(self):
        """TensorFlow sync should check if sync_devices exists."""
        sync = _create_device_sync_statements({"tensorflow"}, for_return_value=False)
        code = ast.unparse(sync[0])

        assert "hasattr(codeflash_tf.test.experimental, 'sync_devices')" in code

    def test_multiple_frameworks_generates_all_syncs(self):
        """Multiple frameworks should generate sync for all of them."""
        post_sync = _create_device_sync_statements(
            {"torch", "jax", "tensorflow"}, for_return_value=True
        )

        # Should have 3 sync statements (torch, jax, tensorflow)
        assert len(post_sync) == 3

        all_code = " ".join(ast.unparse(stmt) for stmt in post_sync)
        assert "codeflash_torch" in all_code
        assert "codeflash_jax" in all_code or "block_until_ready" in all_code
        assert "codeflash_tf" in all_code

    def test_multiple_frameworks_pre_sync_excludes_jax(self):
        """Pre-sync should not include JAX (only post-sync for JAX)."""
        pre_sync = _create_device_sync_statements(
            {"torch", "jax", "tensorflow"}, for_return_value=False
        )

        # Should have 2 sync statements (torch, tensorflow) - no JAX for pre-sync
        assert len(pre_sync) == 2

        all_code = " ".join(ast.unparse(stmt) for stmt in pre_sync)
        assert "codeflash_torch" in all_code
        assert "codeflash_tf" in all_code
        # JAX should not be in pre-sync
        assert "codeflash_jax" not in all_code


class TestCreateWrapperFunctionWithDeviceSync:
    """Tests for create_wrapper_function with device synchronization."""

    def test_wrapper_without_frameworks_has_no_sync(self):
        """Wrapper without frameworks should not have any sync code."""
        wrapper = create_wrapper_function(TestingMode.PERFORMANCE, None)
        code = ast.unparse(wrapper)

        assert "codeflash_torch" not in code
        assert "codeflash_tf" not in code
        assert "codeflash_jax" not in code
        assert "synchronize" not in code
        assert "block_until_ready" not in code
        assert "sync_devices" not in code

    def test_wrapper_with_torch_has_sync_before_and_after(self):
        """Wrapper with torch should have sync before timer and after function call."""
        wrapper = create_wrapper_function(TestingMode.PERFORMANCE, {"torch"})
        code = ast.unparse(wrapper)

        # Should have torch sync code
        assert "codeflash_torch.cuda.synchronize()" in code
        assert "codeflash_torch.mps.synchronize()" in code

        # Check that sync appears twice (before timer and after function call)
        assert code.count("codeflash_torch.cuda.synchronize()") == 2

    def test_wrapper_with_jax_has_sync_after_function_call(self):
        """Wrapper with JAX should have block_until_ready after function call."""
        wrapper = create_wrapper_function(TestingMode.PERFORMANCE, {"jax"})
        code = ast.unparse(wrapper)

        assert "block_until_ready" in code
        # JAX sync should only appear once (after function call)
        assert code.count("block_until_ready") >= 1

    def test_wrapper_with_tensorflow_has_sync_devices(self):
        """Wrapper with TensorFlow should have sync_devices calls."""
        wrapper = create_wrapper_function(TestingMode.PERFORMANCE, {"tensorflow"})
        code = ast.unparse(wrapper)

        assert "codeflash_tf.test.experimental.sync_devices()" in code
        # Should appear twice (before timer and after function call)
        assert code.count("sync_devices()") == 2

    def test_wrapper_with_all_frameworks(self):
        """Wrapper with all frameworks should have all sync types."""
        wrapper = create_wrapper_function(
            TestingMode.PERFORMANCE, {"torch", "jax", "tensorflow"}
        )
        code = ast.unparse(wrapper)

        # Check all frameworks are present
        assert "codeflash_torch.cuda.synchronize()" in code
        assert "block_until_ready" in code
        assert "codeflash_tf.test.experimental.sync_devices()" in code

    def test_wrapper_behavior_mode_with_frameworks(self):
        """Wrapper in BEHAVIOR mode should also have sync code when frameworks specified."""
        wrapper = create_wrapper_function(TestingMode.BEHAVIOR, {"torch"})
        code = ast.unparse(wrapper)

        # Should have torch sync code even in behavior mode
        assert "codeflash_torch.cuda.synchronize()" in code
        # Behavior mode should also have the database code
        assert "codeflash_cur" in code
        assert "codeflash_con" in code

    def test_wrapper_sync_placement_is_correct(self):
        """Verify sync is placed correctly: before timer start and after function call."""
        wrapper = create_wrapper_function(TestingMode.PERFORMANCE, {"torch"})
        code = ast.unparse(wrapper)

        # Find positions of key elements
        first_sync_pos = code.find("codeflash_torch.cuda.synchronize()")
        counter_pos = code.find("counter = time.perf_counter_ns()")
        return_value_pos = code.find("return_value = codeflash_wrapped(")
        second_sync_pos = code.find("codeflash_torch.cuda.synchronize()", first_sync_pos + 1)
        duration_pos = code.find("codeflash_duration = time.perf_counter_ns()")

        # Verify order: first_sync < counter < return_value < second_sync < duration
        assert first_sync_pos < counter_pos, "First sync should be before counter"
        assert counter_pos < return_value_pos, "Counter should be before function call"
        assert return_value_pos < second_sync_pos, "Function call should be before second sync"
        assert second_sync_pos < duration_pos, "Second sync should be before duration calculation"


class TestInjectProfilingWithDeviceSync:
    """Tests for inject_profiling_into_existing_test with device synchronization."""

    def test_inject_profiling_with_torch_adds_import(self, tmp_dir):
        """inject_profiling_into_existing_test with torch should add torch import."""
        code = """import pytest

def sorter(items):
    return sorted(items)

def test_sort():
    result = sorter([3, 1, 2])
    assert result == [1, 2, 3]
"""
        test_file = tmp_dir / "test_sort.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=test_file)

        success, new_test = inject_profiling_into_existing_test(
            test_file,
            [CodePosition(7, 13)],
            func,
            tmp_dir,
            mode=TestingMode.PERFORMANCE,
            used_frameworks={"torch"},
        )

        assert success
        assert new_test is not None
        assert "import torch as codeflash_torch" in new_test
        assert "codeflash_torch.cuda.synchronize()" in new_test

    def test_inject_profiling_with_tensorflow_adds_import(self, tmp_dir):
        """inject_profiling_into_existing_test with tensorflow should add tf import."""
        code = """import pytest

def sorter(items):
    return sorted(items)

def test_sort():
    result = sorter([3, 1, 2])
    assert result == [1, 2, 3]
"""
        test_file = tmp_dir / "test_sort.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=test_file)

        success, new_test = inject_profiling_into_existing_test(
            test_file,
            [CodePosition(7, 13)],
            func,
            tmp_dir,
            mode=TestingMode.PERFORMANCE,
            used_frameworks={"tensorflow"},
        )

        assert success
        assert new_test is not None
        assert "import tensorflow as codeflash_tf" in new_test
        assert "codeflash_tf.test.experimental.sync_devices()" in new_test

    def test_inject_profiling_with_jax_adds_import(self, tmp_dir):
        """inject_profiling_into_existing_test with jax should add jax import."""
        code = """import pytest

def sorter(items):
    return sorted(items)

def test_sort():
    result = sorter([3, 1, 2])
    assert result == [1, 2, 3]
"""
        test_file = tmp_dir / "test_sort.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=test_file)

        success, new_test = inject_profiling_into_existing_test(
            test_file,
            [CodePosition(7, 13)],
            func,
            tmp_dir,
            mode=TestingMode.PERFORMANCE,
            used_frameworks={"jax"},
        )

        assert success
        assert new_test is not None
        assert "import jax as codeflash_jax" in new_test
        assert "block_until_ready" in new_test

    def test_inject_profiling_with_multiple_frameworks(self, tmp_dir):
        """inject_profiling_into_existing_test with multiple frameworks adds all imports."""
        code = """import pytest

def sorter(items):
    return sorted(items)

def test_sort():
    result = sorter([3, 1, 2])
    assert result == [1, 2, 3]
"""
        test_file = tmp_dir / "test_sort.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=test_file)

        success, new_test = inject_profiling_into_existing_test(
            test_file,
            [CodePosition(7, 13)],
            func,
            tmp_dir,
            mode=TestingMode.PERFORMANCE,
            used_frameworks={"torch", "jax", "tensorflow"},
        )

        assert success
        assert new_test is not None
        assert "import torch as codeflash_torch" in new_test
        assert "import tensorflow as codeflash_tf" in new_test
        assert "import jax as codeflash_jax" in new_test

    def test_inject_profiling_without_frameworks_no_sync(self, tmp_dir):
        """inject_profiling_into_existing_test without frameworks has no sync code."""
        code = """import pytest

def sorter(items):
    return sorted(items)

def test_sort():
    result = sorter([3, 1, 2])
    assert result == [1, 2, 3]
"""
        test_file = tmp_dir / "test_sort.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=test_file)

        success, new_test = inject_profiling_into_existing_test(
            test_file,
            [CodePosition(7, 13)],
            func,
            tmp_dir,
            mode=TestingMode.PERFORMANCE,
            used_frameworks=None,
        )

        assert success
        assert new_test is not None
        assert "codeflash_torch" not in new_test
        assert "codeflash_tf" not in new_test
        assert "codeflash_jax" not in new_test
        assert "synchronize" not in new_test
        assert "sync_devices" not in new_test

    def test_inject_profiling_behavior_mode_with_frameworks(self, tmp_dir):
        """inject_profiling_into_existing_test in BEHAVIOR mode with frameworks."""
        code = """import pytest

def sorter(items):
    return sorted(items)

def test_sort():
    result = sorter([3, 1, 2])
    assert result == [1, 2, 3]
"""
        test_file = tmp_dir / "test_sort.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="sorter", parents=[], file_path=test_file)

        success, new_test = inject_profiling_into_existing_test(
            test_file,
            [CodePosition(7, 13)],
            func,
            tmp_dir,
            mode=TestingMode.BEHAVIOR,
            used_frameworks={"torch"},
        )

        assert success
        assert new_test is not None
        # Should have torch sync in behavior mode too
        assert "import torch as codeflash_torch" in new_test
        assert "codeflash_torch.cuda.synchronize()" in new_test
        # Should also have behavior mode specifics
        assert "sqlite3" in new_test
