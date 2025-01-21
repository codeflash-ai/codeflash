from __future__ import annotations

import os
import re
from argparse import Namespace
from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from codeflash.models.models import TestFile, TestFiles, TestingMode
from codeflash.optimization.optimizer import Optimizer
from codeflash.verification.test_results import TestType
from codeflash.verification.test_runner import execute_test_subprocess


# Tests for get_stack_info. Ensures that when a test is run via pytest, the correct test information is extracted
# from the stack for the codeflash_capture decorator. This information will be used in the test invocation id
def test_get_stack_info() -> None:
    test_code = """
from sample_code import MyClass
import unittest

def test_example_test():
    obj = MyClass()
    assert True

class TestExampleClass:
    def test_example_test_2(self):
        obj = MyClass()
        assert True

class TestUnittestExample(unittest.TestCase):
   def test_example_test_3(self):
       obj = MyClass()
       self.assertTrue(True)
"""
    sample_code = """
from codeflash.verification.codeflash_capture import get_test_info_from_stack
class MyClass:
    def __init__(self):
        self.x = 2
        print(f"TEST_INFO_START|{get_test_info_from_stack()}|TEST_INFO_END")
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    test_file_name = "test_stack_info_temp.py"

    test_path = test_dir / test_file_name
    sample_code_path = test_dir / "sample_code.py"
    try:
        with test_path.open("w") as f:
            f.write(test_code)
        with sample_code_path.open("w") as f:
            f.write(sample_code)
        result = execute_test_subprocess(
            cwd=test_dir, env={}, cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"]
        )
        assert not result.stderr
        assert result.returncode == 0
        pattern = r"TEST_INFO_START\|\((.*?)\)\|TEST_INFO_END"
        matches = re.finditer(pattern, result.stdout)
        if not matches:
            raise ValueError("Could not find test info in output")
        results = []
        for match in matches:
            values = [val.strip().strip("'") for val in match.group(1).split(",")]
            results.append(values)
            # Format is (test_module_name, test_class_name, test_name, line_id)

        # First test (test_example_test)
        assert results[0][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[0][1].strip() == "None"  # test_class_name
        assert results[0][2] == "test_example_test"  # test_name
        assert results[0][3] == "6"  # line_id

        # Second test (test_example_test_2 in TestExampleClass)
        assert results[1][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[1][1].strip() == "TestExampleClass"  # test_class_name
        assert results[1][2] == "test_example_test_2"  # test_name
        assert results[1][3] == "11"  # line_id

        # Third test (test_example_test_3 in TestUnittestExample)
        assert results[2][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[2][1].strip() == "TestUnittestExample"  # test_class_name
        assert results[2][2] == "test_example_test_3"  # test_name
        assert results[2][3] == "16"  # line_id

        # Verify we got exactly three results
        assert len(results) == 3

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_codeflash_capture_basic() -> None:
    test_code = """
from code_to_optimize.tests.pytest.sample_code import MyClass
import unittest

def test_example_test():
    obj = MyClass()
    assert True

class TestExampleClass:
    def test_example_test_2(self):
        obj = MyClass()
        assert True

class TestUnittestExample(unittest.TestCase):
   def test_example_test_3(self):
       obj = MyClass()
       self.assertTrue(True)
    """
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture
class MyClass:
    @codeflash_capture(function_name="some_function", tmp_dir_path="{get_run_tmp_file(Path("test_return_values"))}")
    def __init__(self, x=2):
        self.x = x
    """
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    test_file_name = "test_codeflash_capture_temp.py"

    test_path = test_dir / test_file_name
    test_path_perf = test_dir / "test_codeflash_capture_temp_perf.py"

    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    sample_code_path = test_dir / "sample_code.py"
    try:
        with test_path.open("w") as f:
            f.write(test_code)
        with sample_code_path.open("w") as f:
            f.write(sample_code)
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
        assert len(test_results) == 3
        assert test_results[0].did_pass
        assert test_results[0].return_value[0]["x"] == 2
        assert test_results[0].id.test_function_name == "test_example_test"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_module_path == "code_to_optimize.tests.pytest.test_codeflash_capture_temp"
        assert test_results[0].id.function_getting_tested == "some_function"
        assert test_results[0].id.iteration_id == "6_0"

        assert test_results[1].did_pass
        assert test_results[1].return_value[0]["x"] == 2
        assert test_results[1].id.test_function_name == "test_example_test_2"
        assert test_results[1].id.test_class_name == "TestExampleClass"
        assert test_results[1].id.test_module_path == "code_to_optimize.tests.pytest.test_codeflash_capture_temp"
        assert test_results[1].id.function_getting_tested == "some_function"
        assert test_results[1].id.iteration_id == "11_0"

        assert test_results[2].did_pass
        assert test_results[2].return_value[0]["x"] == 2
        assert test_results[2].id.test_function_name == "test_example_test_3"
        assert test_results[2].id.test_class_name == "TestUnittestExample"
        assert test_results[2].id.test_module_path == "code_to_optimize.tests.pytest.test_codeflash_capture_temp"
        assert test_results[2].id.function_getting_tested == "some_function"
        assert test_results[2].id.iteration_id == "16_0"

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_codeflash_capture_super_init() -> None:
    test_code = """
from code_to_optimize.tests.pytest.sample_code import MyClass
import unittest

def test_example_test():
    obj = MyClass()
    assert True

class TestExampleClass:
    def test_example_test_2(self):
        obj = MyClass()
        assert True

class TestUnittestExample(unittest.TestCase):
   def test_example_test_3(self):
       obj = MyClass()
       self.assertTrue(True)
    """
    # MyClass did not have an init function, we created the init function with the codeflash_capture decorator using instrumentation
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture
class ParentClass:
    def __init__(self):
        self.x = 2

class MyClass(ParentClass):
    @codeflash_capture(function_name="some_function", tmp_dir_path="{get_run_tmp_file(Path("test_return_values"))}")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    """
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    test_file_name = "test_codeflash_capture_temp.py"

    test_path = test_dir / test_file_name
    test_path_perf = test_dir / "test_codeflash_capture_temp_perf.py"

    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    sample_code_path = test_dir / "sample_code.py"
    try:
        with test_path.open("w") as f:
            f.write(test_code)
        with sample_code_path.open("w") as f:
            f.write(sample_code)
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
        assert len(test_results) == 3
        assert test_results[0].did_pass
        assert test_results[0].return_value[0]["x"] == 2
        assert test_results[0].id.test_function_name == "test_example_test"
        assert test_results[0].id.test_class_name is None
        assert test_results[0].id.test_module_path == "code_to_optimize.tests.pytest.test_codeflash_capture_temp"
        assert test_results[0].id.function_getting_tested == "some_function"
        assert test_results[0].id.iteration_id == "6_0"

        assert test_results[1].did_pass
        assert test_results[1].return_value[0]["x"] == 2
        assert test_results[1].id.test_function_name == "test_example_test_2"
        assert test_results[1].id.test_class_name == "TestExampleClass"
        assert test_results[1].id.test_module_path == "code_to_optimize.tests.pytest.test_codeflash_capture_temp"
        assert test_results[1].id.function_getting_tested == "some_function"
        assert test_results[1].id.iteration_id == "11_0"

        assert test_results[2].did_pass
        assert test_results[2].return_value[0]["x"] == 2
        assert test_results[2].id.test_function_name == "test_example_test_3"
        assert test_results[2].id.test_class_name == "TestUnittestExample"
        assert test_results[2].id.test_module_path == "code_to_optimize.tests.pytest.test_codeflash_capture_temp"
        assert test_results[2].id.function_getting_tested == "some_function"
        assert test_results[2].id.iteration_id == "16_0"

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)
