from collections import defaultdict

import os
import tempfile
from pathlib import Path

from codeflash.models.models import TestFile, TestFiles
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.parse_test_output import parse_test_xml
from codeflash.verification.test_results import TestType
from codeflash.verification.test_runner import run_behavioral_tests, run_line_profile_tests
from codeflash.verification.verification_utils import TestConfig
from codeflash.code_utils.line_profile_utils import add_decorator_imports
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.either import is_successful
import pytest

def test_unittest_runner():
    code = """import time
import gc
import unittest
def sorter(arr):
    arr.sort()
    return arr
class TestUnittestRunnerSorter(unittest.TestCase):
    def test_sort(self):
        arr = [5, 4, 3, 2, 1, 0]
        gc.disable()
        counter = time.perf_counter_ns()
        output = sorter(arr)
        duration = time.perf_counter_ns() - counter
        gc.enable()
        print(f"#####test_sorter__unit_test_0:TestUnittestRunnerSorter.test_sort:sorter:0#####{duration}^^^^^")
"""
    cur_dir_path = Path(__file__).resolve().parent
    config = TestConfig(
        tests_root=cur_dir_path,
        project_root_path=cur_dir_path,
        test_framework="unittest",
        tests_project_rootdir=cur_dir_path.parent,
    )

    with tempfile.NamedTemporaryFile(prefix="test_xx", suffix=".py", dir=cur_dir_path) as fp:
        test_files = TestFiles(
            test_files=[TestFile(instrumented_behavior_file_path=Path(fp.name), test_type=TestType.EXISTING_UNIT_TEST)]
        )
        fp.write(code.encode("utf-8"))
        fp.flush()
        result_file, process, _, _ = run_behavioral_tests(
            test_files,
            test_framework=config.test_framework,
            cwd=Path(config.project_root_path),
            test_env=os.environ.copy(),
        )
        results = parse_test_xml(result_file, test_files, config, process)
    assert results[0].did_pass, "Test did not pass as expected"
    result_file.unlink(missing_ok=True)


def test_pytest_runner():
    code = """
def sorter(arr):
    arr.sort()
    return arr

def test_sort():
    arr = [5, 4, 3, 2, 1, 0]
    output = sorter(arr)
    assert output == [0, 1, 2, 3, 4, 5]
"""
    cur_dir_path = Path(__file__).resolve().parent
    config = TestConfig(
        tests_root=cur_dir_path,
        project_root_path=cur_dir_path,
        test_framework="pytest",
        tests_project_rootdir=cur_dir_path.parent,
    )

    test_env = os.environ.copy()
    test_env["CODEFLASH_TEST_ITERATION"] = "0"
    test_env["CODEFLASH_TRACER_DISABLE"] = "1"
    if "PYTHONPATH" not in test_env:
        test_env["PYTHONPATH"] = str(config.project_root_path)
    else:
        test_env["PYTHONPATH"] += os.pathsep + str(config.project_root_path)

    with tempfile.NamedTemporaryFile(prefix="test_xx", suffix=".py", dir=cur_dir_path) as fp:
        test_files = TestFiles(
            test_files=[TestFile(instrumented_behavior_file_path=Path(fp.name), test_type=TestType.EXISTING_UNIT_TEST)]
        )
        fp.write(code.encode("utf-8"))
        fp.flush()
        result_file, process, _, _ = run_behavioral_tests(
            test_files,
            test_framework=config.test_framework,
            cwd=Path(config.project_root_path),
            test_env=test_env,
            pytest_timeout=1,
            pytest_target_runtime_seconds=1,
        )
        results = parse_test_xml(
            test_xml_file_path=result_file, test_files=test_files, test_config=config, run_result=process
        )
    assert results[0].did_pass, "Test did not pass as expected"
    result_file.unlink(missing_ok=True)

def sorter(arr):
    arr.sort()
    return arr

def test_pytest_line_profile_runner():
    def get_fto_cc(file_path):
        function_to_optimize = FunctionToOptimize(
            function_name="sort_classmethod", file_path=file_path, parents=[], starting_line=None, ending_line=None
        )
        original_helper_code: dict[Path, str] = {}
        test_config = TestConfig(
            tests_root=file_path.parent / "tests",
            tests_project_rootdir=file_path.parent.resolve(),
            project_root_path=file_path.parent.parent.resolve(),
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=function_to_optimize, test_cfg=test_config)
        ctx_result = func_optimizer.get_code_optimization_context()
        if not is_successful(ctx_result):
            pytest.fail()
        code_context = ctx_result.unwrap()
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        return func_optimizer, code_context, original_helper_code
    file_path = (Path(__file__) / ".." / ".." / "code_to_optimize" / "bubble_sort_classmethod.py").resolve()
    func_optimizer, code_context, original_helper_code = get_fto_cc(file_path)
    #check if decorators are added properly or not
    file_path_to_helper_classes = defaultdict(set)
    for function_source in code_context.helper_functions:
        if (
                function_source.qualified_name != func_optimizer.function_to_optimize.qualified_name
                and "." in function_source.qualified_name
        ):
            file_path_to_helper_classes[function_source.file_path].add(function_source.qualified_name.split(".")[0])
    try:
        line_profiler_output_file = add_decorator_imports(func_optimizer.function_to_optimize, code_context)
    except Exception as e:
        print(e)
    finally:
        func_optimizer.write_code_and_helpers(func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path)
    #now check if lp runs properly or not
    # test_env = os.environ.copy()
    # test_env["CODEFLASH_TEST_ITERATION"] = "0"
    # test_env["CODEFLASH_TRACER_DISABLE"] = "1"
    # if "PYTHONPATH" not in test_env:
    #     test_env["PYTHONPATH"] = str(config.project_root_path)
    # else:
    #     test_env["PYTHONPATH"] += os.pathsep + str(config.project_root_path)
    #
    # with tempfile.NamedTemporaryFile(prefix="test_xx", suffix=".py", dir=cur_dir_path) as fp:
    #     test_files = TestFiles(
    #         test_files=[TestFile(instrumented_behavior_file_path=Path(fp.name), test_type=TestType.EXISTING_UNIT_TEST)]
    #     )
    #     fp.write(code.encode("utf-8"))
    #     fp.flush()
    #     result_file, process = run_line_profile_tests(
    #         test_files,
    #         cwd=cur_dir_path,
    #         test_env=test_env,
    #         test_framework="pytest",
    #         line_profiler_output_file=line_profiler_output_file,
    #         pytest_cmd="pytest",
    #     )
    # print(process.stdout)
    # result_file.unlink(missing_ok=True)

