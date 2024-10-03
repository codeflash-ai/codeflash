import os
import pathlib
import tempfile

from codeflash.verification.parse_test_output import parse_test_xml
from codeflash.verification.test_results import TestType
from codeflash.verification.test_runner import run_tests
from codeflash.verification.verification_utils import TestConfig


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
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    config = TestConfig(
        tests_root=cur_dir_path,
        project_root_path=cur_dir_path,
        test_framework="unittest",
    )

    with tempfile.NamedTemporaryFile(prefix="test_xx", suffix=".py", dir=cur_dir_path) as fp:
        fp.write(code.encode("utf-8"))
        fp.flush()
        result_file, process = run_tests(
            [fp.name],
            test_framework=config.test_framework,
            cwd=config.project_root_path,
        )
        results = parse_test_xml(result_file, [fp.name], [TestType.EXISTING_UNIT_TEST], config, process)
    assert results[0].did_pass, "Test did not pass as expected"
    pathlib.Path(result_file).unlink(missing_ok=True)


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
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    config = TestConfig(
        tests_root=cur_dir_path,
        project_root_path=cur_dir_path,
        test_framework="pytest",
    )
    with tempfile.NamedTemporaryFile(prefix="test_xx", suffix=".py", dir=cur_dir_path) as fp:
        fp.write(code.encode("utf-8"))
        fp.flush()
        test_env = os.environ.copy()
        result_file, process = run_tests(
            [fp.name],
            test_framework=config.test_framework,
            cwd=os.path.join(cur_dir_path),
            test_env=test_env,
            pytest_timeout=1,
            pytest_min_loops=1,
            pytest_max_loops=1,
            pytest_target_runtime_seconds=1,
        )
        results = parse_test_xml(
            test_xml_file_path=result_file,
            test_py_file_paths=[fp.name],
            test_type=TestType.EXISTING_UNIT_TEST,
            test_config=config,
            run_result=process,
        )
    assert results[0].did_pass, "Test did not pass as expected"
    pathlib.Path(result_file).unlink(missing_ok=True)
