from __future__ import annotations

import os
import re
from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent, TestFile, TestFiles, TestingMode, TestType, VerificationType
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.equivalence import compare_test_results
from codeflash.verification.instrument_codeflash_capture import instrument_codeflash_capture
from codeflash.verification.test_runner import execute_test_subprocess
from codeflash.verification.verification_utils import TestConfig


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
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import get_test_info_from_stack
class MyClass:
    def __init__(self):
        self.x = 2
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir!s}')}}|TEST_INFO_END")
"""
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


def test_get_stack_info_2() -> None:
    test_code = """
from sample_code import MyClass
import unittest

obj = MyClass()
def test_example_test():
    assert obj.x == 2

class TestExampleClass:
    def test_example_test_2(self):
        assert obj.x == 2

class TestUnittestExample(unittest.TestCase):
   def test_example_test_3(self):
       self.assertEqual(obj.x, 2)
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import get_test_info_from_stack
class MyClass:
    def __init__(self):
        self.x = 2
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir!s}')}}|TEST_INFO_END")
"""
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
        assert len(results) == 1
        assert results[0][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[0][1].strip() == "None"  # test_class_name
        assert results[0][2].strip() == "None"  # test_name
        assert results[0][3] == "5"  # line_id

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_get_stack_info_3() -> None:
    test_code = """
from sample_code import MyClass
import unittest

def get_obj():
    return MyClass()

def test_example_test():
    result = get_obj().x
    assert result == 2

class TestExampleClass:
    def test_example_test_2(self):
        result = get_obj().x
        assert result == 2

class TestUnittestExample(unittest.TestCase):
    def test_example_test_3(self):
        result = get_obj().x
        self.assertEqual(result, 2)
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import get_test_info_from_stack
class MyClass:
    def __init__(self):
        self.x = 2
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir!s}')}}|TEST_INFO_END")
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
        assert len(results) == 3
        assert results[0][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[0][1].strip() == "None"  # test_class_name
        assert results[0][2].strip() == "test_example_test"  # test_name
        assert results[0][3] == "9"  # line_id

        assert results[1][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[1][1].strip() == "TestExampleClass"  # test_class_name
        assert results[1][2] == "test_example_test_2"  # test_name
        assert results[1][3] == "14"  # line_id

        assert results[2][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[2][1].strip() == "TestUnittestExample"  # test_class_name
        assert results[2][2] == "test_example_test_3"  # test_name
        assert results[2][3] == "19"  # line_id

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_get_stack_info_recursive() -> None:
    test_code = r"""
from sample_code import MyClass
import unittest

def recursive_call(n):
    if n <= 0:
        return
    MyClass()
    recursive_call(n - 1)

def test_example_test():
    # Calls MyClass() 3 times
    recursive_call(3)

class TestExampleClass:
    def test_example_test_2(self):
        # Calls MyClass() 2 times
        recursive_call(2)

class TestUnittestExample(unittest.TestCase):
    def test_example_test_3(self):
        # Calls MyClass() 1 time
        recursive_call(1)
"""
    # Make sure this directory aligns with your existing path structure.
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import get_test_info_from_stack
class MyClass:
    def __init__(self):
        self.x = 2
        # Print out the detected test info each time we instantiate MyClass
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir!s}')}}|TEST_INFO_END")
"""

    test_file_name = "test_stack_info_recursive_temp.py"
    test_path = test_dir / test_file_name
    sample_code_path = test_dir / "sample_code.py"

    try:
        # Write out our test code
        with test_path.open("w") as f:
            f.write(test_code)

        # Write out the sample_code (which includes MyClass and get_test_info_from_stack)
        with sample_code_path.open("w") as f:
            f.write(sample_code)

        # Run pytest as a subprocess
        result = execute_test_subprocess(
            cwd=test_dir, env={}, cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"]
        )

        # Check for errors
        assert not result.stderr
        assert result.returncode == 0

        # Extract the lines that contain the printed test info
        pattern = r"TEST_INFO_START\|\((.*?)\)\|TEST_INFO_END"
        matches = re.finditer(pattern, result.stdout)
        results = []
        for match in matches:
            # Each capture is something like: (module, class_name, test_name, line_id)
            values = [val.strip().strip("'") for val in match.group(1).split(",")]
            results.append(values)

        # We expect 3 calls from test_example_test, 2 from test_example_test_2, and 1 from test_example_test_3 = 6 total
        assert len(results) == 6

        # For the first 3 results, we expect them to come from `test_example_test`
        for i in range(3):
            assert results[i][0] == "code_to_optimize.tests.pytest.test_stack_info_recursive_temp"  # Module name
            assert results[i][1] == "None"  # No class
            assert results[i][2] == "test_example_test"  # Test name
            assert results[i][3] == "13"

        # Next 2 should come from the `TestExampleClass.test_example_test_2`
        for i in range(3, 5):
            assert results[i][0] == "code_to_optimize.tests.pytest.test_stack_info_recursive_temp"
            assert results[i][1] == "TestExampleClass"
            assert results[i][2] == "test_example_test_2"
            assert results[i][3] == "18"

        # Last call should come from the `TestUnittestExample.test_example_test_3`
        assert results[5][0] == "code_to_optimize.tests.pytest.test_stack_info_recursive_temp"
        assert results[5][1] == "TestUnittestExample"
        assert results[5][2] == "test_example_test_3"
        assert results[5][3] == "23"

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_get_stack_info_mixed() -> None:
    test_code = """
from sample_code import MyClass
import unittest

obj = MyClass()

def get_diff_obj():
    return MyClass()

def test_example_test():
    this_obj = MyClass()
    assert this_obj.x == get_diff_obj().x
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import get_test_info_from_stack
class MyClass:
    def __init__(self):
        self.x = 2
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir!s}')}}|TEST_INFO_END")
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

        assert results[0][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[0][1].strip() == "None"  # test_class_name
        assert results[0][2].strip() == "None"  # test_name
        assert results[0][3] == "5"  # line_id

        assert results[1][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[1][1].strip() == "None"  # test_class_name
        assert results[1][2].strip() == "test_example_test"  # test_name
        assert results[1][3] == "11"  # line_id

        assert results[2][0] == "code_to_optimize.tests.pytest.test_stack_info_temp"  # test_module_name
        assert results[2][1].strip() == "None"  # test_class_name
        assert results[2][2].strip() == "test_example_test"  # test_name
        assert results[2][3] == "12"  # line_id

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
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture
class MyClass:
    @codeflash_capture(function_name="some_function", tmp_dir_path="{get_run_tmp_file(Path("test_return_values"))}", tests_root="{test_dir!s}")
    def __init__(self, x=2):
        self.x = x
    """

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

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        fto = FunctionToOptimize(
            function_name="some_function",
            file_path=sample_code_path,
            parents=[FunctionParent(name="MyClass", type="ClassDef")],
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=fto, test_cfg=test_config)
        func_optimizer.test_files = TestFiles(
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
            test_files=func_optimizer.test_files,
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

        test_results2, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        assert compare_test_results(test_results, test_results2)

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
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    # MyClass did not have an init function, we created the init function with the codeflash_capture decorator using instrumentation
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture
class ParentClass:
    def __init__(self):
        self.x = 2

class MyClass(ParentClass):
    @codeflash_capture(function_name="some_function", tmp_dir_path="{get_run_tmp_file(Path("test_return_values"))}", tests_root="{test_dir!s}")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    """
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
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        fto = FunctionToOptimize(
            function_name="some_function",
            file_path=sample_code_path,
            parents=[FunctionParent(name="MyClass", type="ClassDef")],
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=fto, test_cfg=test_config)
        func_optimizer.test_files = TestFiles(
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
            test_files=func_optimizer.test_files,
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

        results2, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert compare_test_results(test_results, results2)

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_codeflash_capture_recursive() -> None:
    test_code = """
from code_to_optimize.tests.pytest.sample_code import MyClass
import unittest

def recursive_call(n):
    if n <= 0:
        return
    MyClass()
    recursive_call(n - 1)

def test_example_test():
    recursive_call(3)
    assert True

"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture

class MyClass:
    @codeflash_capture(
        function_name="some_function", 
        tmp_dir_path="{get_run_tmp_file(Path("test_return_values"))}", 
        tests_root="{test_dir!s}"
    )
    def __init__(self, x=2):
        self.x = x
"""

    test_file_name = "test_codeflash_capture_temp.py"

    test_path = test_dir / test_file_name
    test_path_perf = test_dir / "test_codeflash_capture_temp_perf.py"

    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    sample_code_path = test_dir / "sample_code.py"

    try:
        # Write out the test code
        with test_path.open("w") as f:
            f.write(test_code)
        # Write out the sample code
        with sample_code_path.open("w") as f:
            f.write(sample_code)

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.EXISTING_UNIT_TEST
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        fto = FunctionToOptimize(
            function_name="some_function",
            file_path=sample_code_path,
            parents=[FunctionParent(name="MyClass", type="ClassDef")],
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=fto, test_cfg=test_config)
        func_optimizer.test_files = TestFiles(
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
            test_files=func_optimizer.test_files,
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
        assert test_results[0].id.iteration_id == "12_0"

        assert test_results[1].did_pass
        assert test_results[1].return_value[0]["x"] == 2
        assert test_results[1].id.test_function_name == "test_example_test"
        assert test_results[1].id.test_class_name is None
        assert test_results[1].id.test_module_path == "code_to_optimize.tests.pytest.test_codeflash_capture_temp"
        assert test_results[1].id.function_getting_tested == "some_function"
        assert test_results[1].id.iteration_id == "12_1"

        assert test_results[2].did_pass
        assert test_results[2].return_value[0]["x"] == 2
        assert test_results[2].id.test_function_name == "test_example_test"
        assert test_results[2].id.test_class_name is None
        assert test_results[2].id.test_module_path == "code_to_optimize.tests.pytest.test_codeflash_capture_temp"
        assert test_results[2].id.function_getting_tested == "some_function"
        assert test_results[2].id.iteration_id == "12_2"  # Third call

        test_results2, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert compare_test_results(test_results, test_results2)
    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_codeflash_capture_multiple_helpers() -> None:
    test_code = """
from code_to_optimize.tests.pytest.fto_file import MyClass

def test_helper_classes():
    assert MyClass().target_function() == 6
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    original_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture
from code_to_optimize.tests.pytest.helper_file_1 import HelperClass1
from code_to_optimize.tests.pytest.helper_file_2 import HelperClass2, AnotherHelperClass

class MyClass:
    @codeflash_capture(function_name='MyClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values"))}', tests_root="{test_dir!s}" , is_fto=True)
    def __init__(self):
        self.x = 1

    def target_function(self):
        helper1 = HelperClass1().helper1()
        helper2 = HelperClass2().helper2()
        another = AnotherHelperClass().another_helper()
        return helper1 + helper2 + another
    """
    helper_code_1 = f"""
from codeflash.verification.codeflash_capture import codeflash_capture

class HelperClass1:
    @codeflash_capture(function_name='HelperClass1.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values"))}',  tests_root="{test_dir!s}", is_fto=False)
    def __init__(self):
        self.y = 1

    def helper1(self):
        return 1
    """

    helper_code_2 = f"""
from codeflash.verification.codeflash_capture import codeflash_capture

class HelperClass2:
    @codeflash_capture(function_name='HelperClass2.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values"))}', tests_root="{test_dir!s}", is_fto=False)
    def __init__(self):
        self.z = 2

    def helper2(self):
        return 2

class AnotherHelperClass:
    @codeflash_capture(function_name='AnotherHelperClass.__init__', tmp_dir_path='{get_run_tmp_file(Path("test_return_values"))}', tests_root="{test_dir!s}", is_fto=False)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def another_helper(self):
        return 3
    """

    test_file_name = "test_multiple_helpers.py"

    fto_file_name = "fto_file.py"
    helper_file_1 = "helper_file_1.py"
    helper_file_2 = "helper_file_2.py"

    test_path = test_dir / test_file_name
    test_path_perf = test_dir / "test_multiple_helpers_perf.py"
    helper_path_1 = test_dir / helper_file_1
    helper_path_2 = test_dir / helper_file_2
    fto_file_path = test_dir / fto_file_name

    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()

    try:
        with helper_path_1.open("w") as f:
            f.write(helper_code_1)
        with helper_path_2.open("w") as f:
            f.write(helper_code_2)
        with fto_file_path.open("w") as f:
            f.write(original_code)
        with test_path.open("w") as f:
            f.write(test_code)

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        test_type = TestType.EXISTING_UNIT_TEST
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        fto = FunctionToOptimize(
            function_name="target_function",
            file_path=fto_file_path,
            parents=[FunctionParent(name="MyClass", type="ClassDef")],
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=fto, test_cfg=test_config)
        func_optimizer.test_files = TestFiles(
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
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert len(test_results.test_results) == 4
        assert test_results[0].id.test_function_name == "test_helper_classes"
        assert test_results[0].id.function_getting_tested == "MyClass.__init__"
        assert test_results[0].verification_type == VerificationType.INIT_STATE_FTO
        assert test_results[1].id.function_getting_tested == "HelperClass1.__init__"
        assert test_results[1].verification_type == VerificationType.INIT_STATE_HELPER
        assert test_results[2].id.function_getting_tested == "HelperClass2.__init__"
        assert test_results[2].verification_type == VerificationType.INIT_STATE_HELPER
        assert test_results[3].id.function_getting_tested == "AnotherHelperClass.__init__"
        assert test_results[3].verification_type == VerificationType.INIT_STATE_HELPER

        results2, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert compare_test_results(test_results, results2)

    finally:
        test_path.unlink(missing_ok=True)
        fto_file_path.unlink(missing_ok=True)
        helper_path_1.unlink(missing_ok=True)
        helper_path_2.unlink(missing_ok=True)


def test_instrument_codeflash_capture_and_run_tests() -> None:
    # End to end run that instruments code and runs tests. Made to be similar to code used in the optimizer.py
    test_code = """
from code_to_optimize.tests.pytest.fto_file import MyClass

def test_helper_classes():
    assert MyClass().target_function() == 6
"""

    original_code = """
from code_to_optimize.tests.pytest.helper_file_1 import HelperClass1
from code_to_optimize.tests.pytest.helper_file_2 import HelperClass2, AnotherHelperClass

class MyClass:
    def __init__(self):
        self.x = 1

    def target_function(self):
        helper1 = HelperClass1().helper1()
        helper2 = HelperClass2().helper2()
        another = AnotherHelperClass().another_helper()
        return helper1 + helper2 + another
    """
    helper_code_1 = """
from codeflash.verification.codeflash_capture import codeflash_capture

class HelperClass1:
    def __init__(self):
        self.y = 1

    def helper1(self):
        return 1
    """

    helper_code_2 = """
from codeflash.verification.codeflash_capture import codeflash_capture

class HelperClass2:
    def __init__(self):
        self.z = 2

    def helper2(self):
        return 2

class AnotherHelperClass:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def another_helper(self):
        return 3
    """

    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    test_file_name = "test_multiple_helpers.py"

    fto_file_name = "fto_file.py"
    helper_file_1 = "helper_file_1.py"
    helper_file_2 = "helper_file_2.py"

    test_path = test_dir / test_file_name
    test_path_perf = test_dir / "test_multiple_helpers_perf.py"
    helper_path_1 = test_dir / helper_file_1
    helper_path_2 = test_dir / helper_file_2
    fto_file_path = test_dir / fto_file_name

    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()

    try:
        with helper_path_1.open("w") as f:
            f.write(helper_code_1)
        with helper_path_2.open("w") as f:
            f.write(helper_code_2)
        with fto_file_path.open("w") as f:
            f.write(original_code)
        with test_path.open("w") as f:
            f.write(test_code)

        fto = FunctionToOptimize("target_function", fto_file_path, parents=[FunctionParent("MyClass", "ClassDef")])
        file_path_to_helper_class = {
            helper_path_1: {"HelperClass1"},
            helper_path_2: {"HelperClass2", "AnotherHelperClass"},
        }
        instrument_codeflash_capture(fto, file_path_to_helper_class, tests_root)
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        test_type = TestType.EXISTING_UNIT_TEST
        test_config = TestConfig(
            tests_root=tests_root,
            tests_project_rootdir=project_root_path,
            project_root_path=project_root_path,
            test_framework="pytest",
            pytest_cmd="pytest",
        )
        func_optimizer = FunctionOptimizer(function_to_optimize=fto, test_cfg=test_config)
        func_optimizer.test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path_perf,
                )
            ]
        )
        # Code in optimizer.py
        # Instrument codeflash capture
        candidate_fto_code = Path(fto.file_path).read_text("utf-8")
        candidate_helper_code = {}
        for file_path in file_path_to_helper_class:
            candidate_helper_code[file_path] = Path(file_path).read_text("utf-8")
        file_path_to_helper_classes = {
            Path(helper_path_1): {"HelperClass1"},
            Path(helper_path_2): {"HelperClass2", "AnotherHelperClass"},
        }
        instrument_codeflash_capture(fto, file_path_to_helper_classes, tests_root)

        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)

        assert len(test_results.test_results) == 4
        assert test_results[0].id.test_function_name == "test_helper_classes"
        assert test_results[0].id.function_getting_tested == "MyClass.__init__"
        assert test_results[0].verification_type == VerificationType.INIT_STATE_FTO
        assert test_results[1].id.function_getting_tested == "HelperClass1.__init__"
        assert test_results[1].verification_type == VerificationType.INIT_STATE_HELPER
        assert test_results[2].id.function_getting_tested == "HelperClass2.__init__"
        assert test_results[2].verification_type == VerificationType.INIT_STATE_HELPER
        assert test_results[3].id.function_getting_tested == "AnotherHelperClass.__init__"
        assert test_results[3].verification_type == VerificationType.INIT_STATE_HELPER

        # Now, let's say we optimize the code and make changes.
        new_fto_code = """
from code_to_optimize.tests.pytest.helper_file_1 import HelperClass1
from code_to_optimize.tests.pytest.helper_file_2 import HelperClass2, AnotherHelperClass

class MyClass:
    def __init__(self):
        self.x = 1
        self.y = 3

    def target_function(self):
        helper1 = HelperClass1().helper1()
        helper2 = HelperClass2().helper2()
        another = AnotherHelperClass().another_helper()
        return helper1 + helper2 + another
    """
        with fto_file_path.open("w") as f:
            f.write(new_fto_code)
        # Instrument codeflash capture
        candidate_fto_code = Path(fto.file_path).read_text("utf-8")
        candidate_helper_code = {}
        for file_path in file_path_to_helper_class:
            candidate_helper_code[file_path] = Path(file_path).read_text("utf-8")
        file_path_to_helper_classes = {
            Path(helper_path_1): {"HelperClass1"},
            Path(helper_path_2): {"HelperClass2", "AnotherHelperClass"},
        }
        instrument_codeflash_capture(fto, file_path_to_helper_classes, tests_root)
        modified_test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)

        # Now, this fto_code mutates the instance so it should fail
        mutated_fto_code = """
from code_to_optimize.tests.pytest.helper_file_1 import HelperClass1
from code_to_optimize.tests.pytest.helper_file_2 import HelperClass2, AnotherHelperClass

class MyClass:
    def __init__(self):
        self.x = 2

    def target_function(self):
        helper1 = HelperClass1().helper1()
        helper2 = HelperClass2().helper2()
        another = AnotherHelperClass().another_helper()
        return helper1 + helper2 + another
            """
        with fto_file_path.open("w") as f:
            f.write(mutated_fto_code)
        # Instrument codeflash capture
        candidate_fto_code = Path(fto.file_path).read_text("utf-8")
        candidate_helper_code = {}
        for file_path in file_path_to_helper_class:
            candidate_helper_code[file_path] = Path(file_path).read_text("utf-8")
        file_path_to_helper_classes = {
            Path(helper_path_1): {"HelperClass1"},
            Path(helper_path_2): {"HelperClass2", "AnotherHelperClass"},
        }
        instrument_codeflash_capture(fto, file_path_to_helper_classes, tests_root)
        mutated_test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)
        assert not compare_test_results(test_results, mutated_test_results)

        # This fto code stopped using a helper class. it should still pass
        no_helper1_fto_code = """
from code_to_optimize.tests.pytest.helper_file_2 import HelperClass2, AnotherHelperClass

class MyClass:
    def __init__(self):
        self.x = 1

    def target_function(self):
        helper2 = HelperClass2().helper2()
        another = AnotherHelperClass().another_helper()
        return helper2 + another
                    """
        with fto_file_path.open("w") as f:
            f.write(no_helper1_fto_code)
        # Instrument codeflash capture
        candidate_fto_code = Path(fto.file_path).read_text("utf-8")
        candidate_helper_code = {}
        for file_path in file_path_to_helper_class:
            candidate_helper_code[file_path] = Path(file_path).read_text("utf-8")
        file_path_to_helper_classes = {
            Path(helper_path_1): {"HelperClass1"},
            Path(helper_path_2): {"HelperClass2", "AnotherHelperClass"},
        }
        instrument_codeflash_capture(fto, file_path_to_helper_classes, tests_root)
        no_helper1_test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )
        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)
        assert compare_test_results(test_results, no_helper1_test_results)

    finally:
        test_path.unlink(missing_ok=True)
        fto_file_path.unlink(missing_ok=True)
        helper_path_1.unlink(missing_ok=True)
        helper_path_2.unlink(missing_ok=True)
