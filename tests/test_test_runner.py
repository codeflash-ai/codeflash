import os
import tempfile
from pathlib import Path

import line_profiler
from codeflash.models.models import TestFile, TestFiles
from codeflash.verification.parse_test_output import parse_test_xml
from codeflash.verification.test_results import TestType
from codeflash.verification.test_runner import run_behavioral_tests
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

def test_pytest_runner_lprof_process():
    tmpdir = tempfile.TemporaryDirectory()
    prefix = "from line_profiler import profile\nprofile.enable(output_prefix=\""+tmpdir.name+os.sep+"baseline\""+")\n"
    code = prefix + """
@profile
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
            enable_lprofiler=True,
        )
        with open(tmpdir.name+os.sep+"/baseline.txt", "r") as f:
            output = f.read()
        expected_output = "Timer unit: 1e-09 s\n\nTotal time: 0 s\nFile: {}\nFunction: sorter at line 4\n\nLine #      Hits         Time  Per Hit   % Time  Line Contents\n==============================================================\n     4                                           @profile\n     5                                           def sorter(arr):\n     6         1          0.0      0.0               arr.sort()\n     7         1          0.0      0.0               return arr\n\n".format(fp.name)


    assert output == expected_output, "Test passed"
    result_file.unlink(missing_ok=True)

