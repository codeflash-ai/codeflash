from __future__ import annotations

import ast
import math
import os
import sys
import tempfile
from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.instrument_existing_tests import (
    FunctionImportedAsVisitor,
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
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

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
    test_stdout_tag = f"{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}"
    print(f"!$######{test_stdout_tag}######$!")
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
    print(f"!######{test_stdout_tag}######!")
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (test_module_name, test_class_name, test_name, function_name, loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value
"""

codeflash_wrap_perfonly_string = """def codeflash_wrap(wrapped, test_module_name, test_class_name, test_name, function_name, line_id, loop_index, *args, **kwargs):
    test_id = f'{{test_module_name}}:{{test_class_name}}:{{test_name}}:{{line_id}}:{{loop_index}}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {{}}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{{line_id}}_{{codeflash_test_index}}'
    exception = None
    test_stdout_tag = f"{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}"
    print(f"!$######{{test_stdout_tag}}######$!")
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = wrapped(*args, **kwargs)
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


def test_perfinjector_bubble_sort() -> None:
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
    expected = """import gc
import os
import sqlite3
import time
import unittest

import dill as pickle
import timeout_decorator

from code_to_optimize.bubble_sort import sorter


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
    test_stdout_tag = f"{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}"
    """
    if sys.version_info < (3, 12):
        expected += """test_stdout_tag = f"{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}"
    print(f"!$######{{test_stdout_tag}}######$!")"""
    else:
        expected += """print(f'!######{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}######!')
    print(f'!$######{{test_stdout_tag}}######$!')"""
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
    print(f"!######{{test_stdout_tag}}######!")
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (test_module_name, test_class_name, test_name, function_name, loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

class TestPigLatin(unittest.TestCase):

    @timeout_decorator.timeout(15)
    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        input = [5, 4, 3, 2, 1, 0]
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(5000)))
        self.assertEqual(codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '7', codeflash_loop_index, codeflash_cur, codeflash_con, input), list(range(5000)))
        codeflash_con.close()
"""
    with tempfile.NamedTemporaryFile(mode="w") as f:
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
            "unittest",
        )
        os.chdir(original_cwd)
    assert success
    assert new_test == expected.format(
        module_path=Path(f.name).name, tmp_dir_path=get_run_tmp_file(Path("test_return_values"))
    )


def test_perfinjector_only_replay_test() -> None:
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
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (test_module_name, test_class_name, test_name, function_name, loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
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
        ret = codeflash_wrap(packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo, '{module_path}', None, 'test_prepare_image_for_yolo', 'packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo', '0_2', codeflash_loop_index, codeflash_cur, codeflash_con, **args)
        assert compare_results(return_val_1, ret)
    codeflash_con.close()
"""
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        func = FunctionToOptimize(function_name="prepare_image_for_yolo", parents=[], file_path=Path("module.py"))
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name), [CodePosition(10, 14)], func, Path(f.name).parent, "pytest"
        )
        os.chdir(original_cwd)
    assert success
    assert new_test == expected.format(
        module_path=Path(f.name).name, tmp_dir_path=get_run_tmp_file(Path("test_return_values"))
    )


def test_perfinjector_bubble_sort_results() -> None:
    computed_fn_opt = False
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
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    input = [5, 4, 3, 2, 1, 0]
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, input)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    codeflash_con.close()
"""
    )

    expected_perfonly = (
        """import gc
import os
import time

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
def test_sort():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    input = [5, 4, 3, 2, 1, 0]
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '1', codeflash_loop_index, input)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '4', codeflash_loop_index, input)
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
            [CodePosition(6, 13), CodePosition(10, 13)],
            func,
            project_root_path,
            "pytest",
            mode=TestingMode.BEHAVIOR,
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")

        success, new_perf_test = inject_profiling_into_existing_test(
            test_path,
            [CodePosition(6, 13), CodePosition(10, 13)],
            func,
            project_root_path,
            "pytest",
            mode=TestingMode.PERFORMANCE,
        )
        assert success
        assert new_perf_test is not None
        assert new_perf_test.replace('"', "'") == expected_perfonly.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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
        assert test_results_perf[0].id.iteration_id == "1_0"
        assert test_results_perf[0].id.test_class_name is None
        assert test_results_perf[0].id.test_function_name == "test_sort"
        assert (
            test_results_perf[0].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp"
        )
        assert test_results_perf[0].runtime > 0
        assert test_results_perf[0].did_pass
        assert test_results_perf[0].return_value is None

        assert test_results_perf[1].id.function_getting_tested == "sorter"
        assert test_results_perf[1].id.iteration_id == "4_0"
        assert test_results_perf[1].id.test_class_name is None
        assert test_results_perf[1].id.test_function_name == "test_sort"
        assert (
            test_results_perf[1].id.test_module_path
            == "code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_results_temp"
        )
        assert test_results_perf[1].runtime > 0
        assert test_results_perf[1].did_pass
        out_str = """codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5]

codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"""
        assert out_str == test_results_perf[1].stdout
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        computed_fn_opt = True
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        line_profile_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.LINE_PROFILE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
            line_profiler_output_file = line_profiler_output_file
        )
        tmp_lpr = list(line_profile_results["timings"].keys())
        assert len(tmp_lpr) == 1 and line_profile_results["timings"][tmp_lpr[0]][0][1]==2
    finally:
        if computed_fn_opt:
            func_optimizer.write_code_and_helpers(
                func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
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
    output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort_parametrized', 'sorter', '0', codeflash_loop_index, codeflash_cur, codeflash_con, input)
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
            test_path, [CodePosition(14, 13)], func, project_root_path, "pytest", mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(14, 13)], func, project_root_path, "pytest", mode=TestingMode.PERFORMANCE
        )

        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")
        assert new_test_perf.replace('"', "'") == expected_perfonly.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"""
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
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        line_profile_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.LINE_PROFILE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
            line_profiler_output_file = line_profiler_output_file
        )
        tmp_lpr = list(line_profile_results["timings"].keys())
        assert len(tmp_lpr) == 1 and line_profile_results["timings"][tmp_lpr[0]][0][1]==3
    finally:
        if computed_fn_opt:
            func_optimizer.write_code_and_helpers(
                func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
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
        output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort_parametrized_loop', 'sorter', '0_0', codeflash_loop_index, codeflash_cur, codeflash_con, input)
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
            test_path, [CodePosition(15, 17)], func, project_root_path, "pytest", mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(15, 17)], func, project_root_path, "pytest", mode=TestingMode.PERFORMANCE
        )

        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")

        # Overwrite old test with new instrumented test
        with test_path_behavior.open("w") as f:
            f.write(new_test)

        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_parametrized_loop_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        line_profile_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.LINE_PROFILE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
            line_profiler_output_file = line_profiler_output_file
        )
        tmp_lpr = list(line_profile_results["timings"].keys())
        assert len(tmp_lpr) == 1 and line_profile_results["timings"][tmp_lpr[0]][0][1]==6
    finally:
        if computed_fn_opt:
            func_optimizer.write_code_and_helpers(
                func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
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
        output = codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '2_2', codeflash_loop_index, codeflash_cur, codeflash_con, input)
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
            test_path, [CodePosition(11, 17)], func, project_root_path, "pytest", mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(11, 17)], func, project_root_path, "pytest", mode=TestingMode.PERFORMANCE
        )
        os.chdir(original_cwd)
        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")

        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.pytest.test_perfinjector_bubble_sort_loop_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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

codeflash stdout: Sorting list
result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

codeflash stdout: Sorting list
result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]"""
        assert test_results[1].stdout == out_str
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
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        computed_fn_opt = True
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        line_profile_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.LINE_PROFILE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
            line_profiler_output_file = line_profiler_output_file
        )
        tmp_lpr = list(line_profile_results["timings"].keys())
        assert len(tmp_lpr) == 1 and line_profile_results["timings"][tmp_lpr[0]][0][1]==3
    finally:
        if computed_fn_opt is True:
            func_optimizer.write_code_and_helpers(
                func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
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

    expected = (
        """import gc
import os
import sqlite3
import time
import unittest

import dill as pickle
import timeout_decorator

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_string
        + """
class TestPigLatin(unittest.TestCase):

    @timeout_decorator.timeout(15)
    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        input = [5, 4, 3, 2, 1, 0]
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(50)))
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '7', codeflash_loop_index, codeflash_cur, codeflash_con, input)
        self.assertEqual(output, list(range(50)))
        codeflash_con.close()
"""
    )
    expected_perf = (
        """import gc
import os
import time
import unittest

import timeout_decorator

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
class TestPigLatin(unittest.TestCase):

    @timeout_decorator.timeout(15)
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
            "unittest",
            mode=TestingMode.BEHAVIOR,
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path,
            [CodePosition(9, 17), CodePosition(13, 17), CodePosition(17, 17)],
            func,
            project_root_path,
            "unittest",
            mode=TestingMode.PERFORMANCE,
        )
        os.chdir(original_cwd)

        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")
        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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

    expected_behavior = (
        """import gc
import os
import sqlite3
import time
import unittest

import dill as pickle
import timeout_decorator
from parameterized import parameterized

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_string
        + """
class TestPigLatin(unittest.TestCase):

    @parameterized.expand([([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
    @timeout_decorator.timeout(15)
    def test_sort(self, input, expected_output):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '0', codeflash_loop_index, codeflash_cur, codeflash_con, input)
        self.assertEqual(output, expected_output)
        codeflash_con.close()
"""
    )
    expected_perf = (
        """import gc
import os
import time
import unittest

import timeout_decorator
from parameterized import parameterized

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
class TestPigLatin(unittest.TestCase):

    @parameterized.expand([([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
    @timeout_decorator.timeout(15)
    def test_sort(self, input, expected_output):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '0', codeflash_loop_index, input)
        self.assertEqual(output, expected_output)
"""
    )
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
            test_path, [CodePosition(16, 17)], func, project_root_path, "unittest", mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(16, 17)], func, project_root_path, "unittest", mode=TestingMode.PERFORMANCE
        )

        os.chdir(original_cwd)
        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected_behavior.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")

        assert new_test_perf is not None
        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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

    expected_behavior = (
        """import gc
import os
import sqlite3
import time
import unittest

import dill as pickle
import timeout_decorator

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_string
        + """
class TestPigLatin(unittest.TestCase):

    @timeout_decorator.timeout(15)
    def test_sort(self):
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
            output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '2_2', codeflash_loop_index, codeflash_cur, codeflash_con, input)
            self.assertEqual(output, expected_output)
        codeflash_con.close()
"""
    )

    expected_perf = (
        """import gc
import os
import time
import unittest

import timeout_decorator

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
class TestPigLatin(unittest.TestCase):

    @timeout_decorator.timeout(15)
    def test_sort(self):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        inputs = [[5, 4, 3, 2, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], list(reversed(range(50)))]
        expected_outputs = [[0, 1, 2, 3, 4, 5], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(50))]
        for i in range(3):
            input = inputs[i]
            expected_output = expected_outputs[i]
            output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '2_2', codeflash_loop_index, input)
            self.assertEqual(output, expected_output)
"""
    )
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
            test_path, [CodePosition(14, 21)], func, project_root_path, "unittest", mode=TestingMode.BEHAVIOR
        )
        assert success
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(14, 21)], func, project_root_path, "unittest", mode=TestingMode.PERFORMANCE
        )
        os.chdir(original_cwd)
        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected_behavior.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")
        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_loop_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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

    expected_behavior = (
        """import gc
import os
import sqlite3
import time
import unittest

import dill as pickle
import timeout_decorator
from parameterized import parameterized

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_string
        + """
class TestPigLatin(unittest.TestCase):

    @parameterized.expand([([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
    @timeout_decorator.timeout(15)
    def test_sort(self, input, expected_output):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
        for i in range(2):
            output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '0_0', codeflash_loop_index, codeflash_cur, codeflash_con, input)
            self.assertEqual(output, expected_output)
        codeflash_con.close()
"""
    )
    expected_perf = (
        """import gc
import os
import time
import unittest

import timeout_decorator
from parameterized import parameterized

from code_to_optimize.bubble_sort import sorter


"""
        + codeflash_wrap_perfonly_string
        + """
class TestPigLatin(unittest.TestCase):

    @parameterized.expand([([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]), ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (list(reversed(range(50))), list(range(50)))])
    @timeout_decorator.timeout(15)
    def test_sort(self, input, expected_output):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        for i in range(2):
            output = codeflash_wrap(sorter, '{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '0_0', codeflash_loop_index, input)
            self.assertEqual(output, expected_output)
"""
    )
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
            test_path, [CodePosition(17, 21)], f, project_root_path, "unittest", mode=TestingMode.BEHAVIOR
        )
        success, new_test_perf = inject_profiling_into_existing_test(
            test_path, [CodePosition(17, 21)], f, project_root_path, "unittest", mode=TestingMode.PERFORMANCE
        )
        os.chdir(original_cwd)
        assert success
        assert new_test_behavior is not None
        assert new_test_behavior.replace('"', "'") == expected_behavior.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
        ).replace('"', "'")
        assert new_test_perf.replace('"', "'") == expected_perf.format(
            module_path="code_to_optimize.tests.unittest.test_perfinjector_bubble_sort_unittest_parametrized_loop_results_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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
    ret = codeflash_wrap(class_name_A.function_name, '{module_path}', None, 'test_class_name_A_function_name', 'class_name_A.function_name', '0', codeflash_loop_index, codeflash_cur, codeflash_con, **args)
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
            test_path, [CodePosition(4, 23)], func, project_root_path, "pytest"
        )
        os.chdir(original_cwd)
    finally:
        test_path.unlink(missing_ok=True)
    assert success
    assert new_test is not None
    assert new_test.replace('"', "'") == expected.format(
        tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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
    assert codeflash_wrap(find_common_tags, '{module_path}', None, 'test_common_tags_1', 'find_common_tags', '1', codeflash_loop_index, codeflash_cur, codeflash_con, articles_1) == set(1, 2)
    articles_2 = [1, 2]
    assert codeflash_wrap(find_common_tags, '{module_path}', None, 'test_common_tags_1', 'find_common_tags', '3', codeflash_loop_index, codeflash_cur, codeflash_con, articles_2) == set(1)
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
            test_path, [CodePosition(7, 11), CodePosition(11, 11)], func, project_root_path, "pytest"
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="tests.pytest.test_wrong_function_instrumentation_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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
        assert codeflash_wrap(sorter, '{module_path}', None, 'test_sort', 'sorter', '1_0', codeflash_loop_index, codeflash_cur, codeflash_con, input) == [0, 1, 2, 3, 4, 5]
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
            test_path, [CodePosition(7, 15)], func, project_root_path, "pytest"
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        assert new_test.replace('"', "'") == expected.format(
            module_path="tests.pytest.test_conditional_instrumentation_temp",
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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
    output = codeflash_wrap(BubbleSorter.sorter, 'tests.pytest.test_perfinjector_bubble_sort_results_temp', None, 'test_sort', 'BubbleSorter.sorter', '1', codeflash_loop_index, codeflash_cur, codeflash_con, input)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = codeflash_wrap(BubbleSorter.sorter, '{module_path}', None, 'test_sort', 'BubbleSorter.sorter', '4', codeflash_loop_index, codeflash_cur, codeflash_con, input)
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
            test_path, [CodePosition(6, 26), CodePosition(10, 26)], function_to_optimize, project_root_path, "pytest"
        )
        os.chdir(original_cwd)
        assert success
        formatted_expected = expected.format(
            module_path="tests.pytest.test_perfinjector_bubble_sort_results_temp",
            tmp_dir_path=str(get_run_tmp_file(Path("test_return_values"))),
        )
        assert new_test is not None
        assert new_test.replace('"', "'") == formatted_expected.replace('"', "'")
    finally:
        test_path.unlink(missing_ok=True)


def test_class_method_instrumentation() -> None:
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

    expected = """import gc
import os
import sqlite3
import time

import dill as pickle

from codeflash.optimization.optimizer import Optimizer


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
        expected += """    print(f"!######{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}######!")"""
    else:
        expected += """    print(f'!######{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}######!')"""
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
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (test_module_name, test_class_name, test_name, function_name, loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

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
        code_context = codeflash_wrap(opt.get_code_optimization_context, '{module_path}', None, 'test_code_replacement10', 'Optimizer.get_code_optimization_context', '4_1', codeflash_loop_index, codeflash_cur, codeflash_con, function_to_optimize=func_top_optimize, project_root=str(file_path.parent), original_source_code=original_code).unwrap()
        assert code_context.testgen_context_code == get_code_output
        code_context = codeflash_wrap(opt.get_code_optimization_context, '{module_path}', None, 'test_code_replacement10', 'Optimizer.get_code_optimization_context', '4_3', codeflash_loop_index, codeflash_cur, codeflash_con, function_to_optimize=func_top_optimize, project_root=str(file_path.parent), original_source_code=original_code)
        assert code_context.testgen_context_code == get_code_output
    codeflash_con.close()
"""

    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        func = FunctionToOptimize(
            function_name="get_code_optimization_context",
            parents=[FunctionParent("Optimizer", "ClassDef")],
            file_path=Path(f.name),
        )
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name), [CodePosition(22, 28), CodePosition(28, 28)], func, Path(f.name).parent, "pytest"
        )
        os.chdir(original_cwd)
    assert success
    assert new_test == expected.format(
        module_path=Path(f.name).name, tmp_dir_path=get_run_tmp_file(Path("test_return_values"))
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
            test_path, [CodePosition(8, 13)], func, project_root_path, "pytest", mode=TestingMode.PERFORMANCE
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
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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

    expected = (
        """import gc
import os
import time
import unittest

import timeout_decorator
from parameterized import parameterized

from code_to_optimize.sleeptime import accurate_sleepfunc


"""
        + codeflash_wrap_perfonly_string
        + """
class TestPigLatin(unittest.TestCase):

    @parameterized.expand([(0.01, 0.01), (0.02, 0.02)])
    @timeout_decorator.timeout(15)
    def test_sleepfunc_sequence_short(self, n, expected_total_sleep_time):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        output = codeflash_wrap(accurate_sleepfunc, '{module_path}', 'TestPigLatin', 'test_sleepfunc_sequence_short', 'accurate_sleepfunc', '0', codeflash_loop_index, n)
"""
    )
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
            test_path, [CodePosition(12, 17)], func, project_root_path, "unittest", mode=TestingMode.PERFORMANCE
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
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")),
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
