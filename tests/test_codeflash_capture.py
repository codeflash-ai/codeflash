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
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir.as_posix()}')}}|TEST_INFO_END")
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
            cwd=test_dir,
            cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"],
            env=os.environ.copy(),
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
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir.as_posix()}')}}|TEST_INFO_END")
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
            cwd=test_dir,
            cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"],
            env=os.environ.copy(),
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
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir.as_posix()}')}}|TEST_INFO_END")
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
            cwd=test_dir,
            cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"],
            env=os.environ.copy(),
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
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir.as_posix()}')}}|TEST_INFO_END")
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
            cwd=test_dir,
            cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"],
            env=os.environ.copy(),
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
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir.as_posix()}')}}|TEST_INFO_END")
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
            cwd=test_dir,
            cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"],
            env=os.environ.copy(),
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
    tmp_dir_path = get_run_tmp_file(Path("test_return_values"))
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture
class MyClass:
    @codeflash_capture(function_name="some_function", tmp_dir_path="{tmp_dir_path.as_posix()}", tests_root="{test_dir.as_posix()}")
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
            min_outer_loops=1,
            max_outer_loops=1,
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
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )
        match, _ = compare_test_results(test_results, test_results2)
        assert match

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
    tmp_dir_path = get_run_tmp_file(Path("test_return_values"))
    # MyClass did not have an init function, we created the init function with the codeflash_capture decorator using instrumentation
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture
class ParentClass:
    def __init__(self):
        self.x = 2

class MyClass(ParentClass):
    @codeflash_capture(function_name="some_function", tmp_dir_path="{tmp_dir_path.as_posix()}", tests_root="{test_dir.as_posix()}")
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
            min_outer_loops=1,
            max_outer_loops=1,
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
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )

        match, _ = compare_test_results(test_results, results2)
        assert match

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
    tmp_dir_path = get_run_tmp_file(Path("test_return_values"))
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture

class MyClass:
    @codeflash_capture(
        function_name="some_function", 
        tmp_dir_path="{tmp_dir_path.as_posix()}", 
        tests_root="{test_dir.as_posix()}"
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
            min_outer_loops=1,
            max_outer_loops=1,
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
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )

        match, _ = compare_test_results(test_results, test_results2)
        assert match
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
    tmp_dir_path = get_run_tmp_file(Path("test_return_values"))
    original_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture
from code_to_optimize.tests.pytest.helper_file_1 import HelperClass1
from code_to_optimize.tests.pytest.helper_file_2 import HelperClass2, AnotherHelperClass

class MyClass:
    @codeflash_capture(function_name='MyClass.__init__', tmp_dir_path='{tmp_dir_path.as_posix()}', tests_root="{test_dir.as_posix()}" , is_fto=True)
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
    @codeflash_capture(function_name='HelperClass1.__init__', tmp_dir_path='{tmp_dir_path.as_posix()}',  tests_root="{test_dir.as_posix()}", is_fto=False)
    def __init__(self):
        self.y = 1

    def helper1(self):
        return 1
    """

    helper_code_2 = f"""
from codeflash.verification.codeflash_capture import codeflash_capture

class HelperClass2:
    @codeflash_capture(function_name='HelperClass2.__init__', tmp_dir_path='{tmp_dir_path.as_posix()}', tests_root="{test_dir.as_posix()}", is_fto=False)
    def __init__(self):
        self.z = 2

    def helper2(self):
        return 2

class AnotherHelperClass:
    @codeflash_capture(function_name='AnotherHelperClass.__init__', tmp_dir_path='{tmp_dir_path.as_posix()}', tests_root="{test_dir.as_posix()}", is_fto=False)
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
            min_outer_loops=1,
            max_outer_loops=1,
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
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )

        match, _ = compare_test_results(test_results, results2)
        assert match

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
            min_outer_loops=1,
            max_outer_loops=1,
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
            min_outer_loops=1,
            max_outer_loops=1,
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
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )
        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)
        match, _ = compare_test_results(test_results, mutated_test_results)
        assert not match

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
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )
        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)
        match, _ = compare_test_results(test_results, no_helper1_test_results)
        assert match

    finally:
        test_path.unlink(missing_ok=True)
        fto_file_path.unlink(missing_ok=True)
        helper_path_1.unlink(missing_ok=True)
        helper_path_2.unlink(missing_ok=True)


def test_get_stack_info_env_var_fallback() -> None:
    """Test that get_test_info_from_stack falls back to environment variables when stack walking fails to find test_name.

    At module level, stack walking finds test_module_name but NOT test_name.
    The env var fallback should fill in test_name from CODEFLASH_TEST_FUNCTION.
    """
    test_code = """
import os
from sample_code import MyClass

# Set environment variables before instantiation
os.environ["CODEFLASH_TEST_FUNCTION"] = "test_env_fallback_function"
os.environ["CODEFLASH_TEST_MODULE"] = "env_fallback_module"
os.environ["CODEFLASH_TEST_CLASS"] = "EnvFallbackClass"

# Instantiate at module level (stack walking won't find a test_ function name)
obj = MyClass()

def test_dummy():
    # This test exists just to make pytest run the file
    assert obj.x == 2
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import get_test_info_from_stack
class MyClass:
    def __init__(self):
        self.x = 2
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir.as_posix()}')}}|TEST_INFO_END")
"""
    test_file_name = "test_env_var_fallback_temp.py"

    test_path = test_dir / test_file_name
    sample_code_path = test_dir / "sample_code.py"
    try:
        with test_path.open("w") as f:
            f.write(test_code)
        with sample_code_path.open("w") as f:
            f.write(sample_code)

        # Make sure env vars are NOT set in the parent process (they should be set by the test file itself)
        test_env = os.environ.copy()
        test_env.pop("CODEFLASH_TEST_FUNCTION", None)
        test_env.pop("CODEFLASH_TEST_MODULE", None)
        test_env.pop("CODEFLASH_TEST_CLASS", None)

        result = execute_test_subprocess(
            cwd=test_dir, cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"], env=test_env
        )
        assert result.returncode == 0
        pattern = r"TEST_INFO_START\|\((.*?)\)\|TEST_INFO_END"
        matches = re.finditer(pattern, result.stdout)
        results = []
        for match in matches:
            values = [val.strip().strip("'") for val in match.group(1).split(",")]
            results.append(values)

        # Should have one result from the module-level instantiation
        assert len(results) == 1

        # test_name should come from env var (CODEFLASH_TEST_FUNCTION) since stack walking didn't find it
        assert results[0][2] == "test_env_fallback_function"  # test_name from env var
        # test_module_name is found via stack walking at module level, so env var doesn't override
        assert results[0][0] == "code_to_optimize.tests.pytest.test_env_var_fallback_temp"  # from stack
        # test_class_name should come from env var since stack walking didn't find a class
        assert results[0][1] == "EnvFallbackClass"  # test_class_name from env var

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_get_stack_info_env_var_fallback_partial() -> None:
    """Test that env var fallback only fills in missing values, not overwriting stack-found values."""
    test_code = """
import os
from sample_code import MyClass

# Set environment variables
os.environ["CODEFLASH_TEST_FUNCTION"] = "env_test_function"
os.environ["CODEFLASH_TEST_MODULE"] = "env_test_module"
os.environ["CODEFLASH_TEST_CLASS"] = "EnvTestClass"

def test_real_test_function():
    # Stack walking WILL find this test function
    obj = MyClass()
    assert obj.x == 2
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    sample_code = f"""
from codeflash.verification.codeflash_capture import get_test_info_from_stack
class MyClass:
    def __init__(self):
        self.x = 2
        print(f"TEST_INFO_START|{{get_test_info_from_stack('{test_dir.as_posix()}')}}|TEST_INFO_END")
"""
    test_file_name = "test_env_var_partial_temp.py"

    test_path = test_dir / test_file_name
    sample_code_path = test_dir / "sample_code.py"
    try:
        with test_path.open("w") as f:
            f.write(test_code)
        with sample_code_path.open("w") as f:
            f.write(sample_code)

        test_env = os.environ.copy()
        result = execute_test_subprocess(
            cwd=test_dir, cmd_list=[f"{SAFE_SYS_EXECUTABLE}", "-m", "pytest", test_file_name, "-s"], env=test_env
        )
        assert result.returncode == 0
        pattern = r"TEST_INFO_START\|\((.*?)\)\|TEST_INFO_END"
        matches = re.finditer(pattern, result.stdout)
        results = []
        for match in matches:
            values = [val.strip().strip("'") for val in match.group(1).split(",")]
            results.append(values)

        assert len(results) == 1

        # Stack walking should have found the test function, so env vars should NOT override
        assert results[0][2] == "test_real_test_function"  # test_name from stack, not env var
        assert results[0][0] == "code_to_optimize.tests.pytest.test_env_var_partial_temp"  # module from stack
        assert results[0][1].strip() == "None"  # no class in this test

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)


def test_instrument_codeflash_capture_and_run_tests_2() -> None:
    # End to end run that instruments code and runs tests. Made to be similar to code used in the optimizer.py
    test_code = """import math    
import pytest
from typing import List, Tuple, Optional
from code_to_optimize.tests.pytest.fto_file import calculate_portfolio_metrics

def test_calculate_portfolio_metrics():
    # Test case 1: Basic portfolio
    investments = [
        ('Stocks', 0.6, 0.12),
        ('Bonds', 0.3, 0.04),
        ('Cash', 0.1, 0.01)
    ]
    
    result = calculate_portfolio_metrics(investments)
    
    # Check weighted return calculation
    expected_return = 0.6*0.12 + 0.3*0.04 + 0.1*0.01
    assert abs(result['weighted_return'] - expected_return) < 1e-10
    
    # Check volatility calculation
    expected_vol = math.sqrt((0.6*0.12)**2 + (0.3*0.04)**2 + (0.1*0.01)**2)
    assert abs(result['volatility'] - expected_vol) < 1e-10
    
    # Check Sharpe ratio
    expected_sharpe = (expected_return - 0.02) / expected_vol
    assert abs(result['sharpe_ratio'] - expected_sharpe) < 1e-10
    
    # Check best/worst performers
    assert result['best_performing'][0] == 'Stocks'
    assert result['worst_performing'][0] == 'Cash'
    assert result['total_assets'] == 3

def test_empty_investments():
    with pytest.raises(ValueError, match="Investments list cannot be empty"):
        calculate_portfolio_metrics([])

def test_weights_not_sum_to_one():
    investments = [('Stock', 0.5, 0.1), ('Bond', 0.4, 0.05)]
    with pytest.raises(ValueError, match="Portfolio weights must sum to 1.0"):
        calculate_portfolio_metrics(investments)

def test_zero_volatility():
    investments = [('Cash', 1.0, 0.0)]
    result = calculate_portfolio_metrics(investments, risk_free_rate=0.0)
    assert result['sharpe_ratio'] == 0.0
    assert result['volatility'] == 0.0
"""

    original_code = """import math
from typing import List, Tuple, Optional

def calculate_portfolio_metrics(
    investments: List[Tuple[str, float, float]], 
    risk_free_rate: float = 0.02
) -> dict:
    if not investments:
        raise ValueError("Investments list cannot be empty")
    
    if abs(sum(weight for _, weight, _ in investments) - 1.0) > 1e-10:
        raise ValueError("Portfolio weights must sum to 1.0")
    
    # Calculate weighted return
    weighted_return = sum(weight * ret for _, weight, ret in investments)
    
    # Calculate portfolio volatility (simplified)
    volatility = math.sqrt(sum((weight * ret) ** 2 for _, weight, ret in investments))
    
    # Calculate Sharpe ratio
    if volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (weighted_return - risk_free_rate) / volatility
    
    # Find best and worst performing assets
    best_asset = max(investments, key=lambda x: x[2])
    worst_asset = min(investments, key=lambda x: x[2])
    
    return {
        'weighted_return': round(weighted_return, 6),
        'volatility': round(volatility, 6),
        'sharpe_ratio': round(sharpe_ratio, 6),
        'best_performing': (best_asset[0], round(best_asset[2], 6)),
        'worst_performing': (worst_asset[0], round(worst_asset[2], 6)),
        'total_assets': len(investments)
    }
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    test_file_name = "test_multiple_helpers.py"

    fto_file_name = "fto_file.py"

    test_path = test_dir / test_file_name
    test_path_perf = test_dir / "test_multiple_helpers_perf.py"
    fto_file_path = test_dir / fto_file_name

    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()

    try:
        with fto_file_path.open("w") as f:
            f.write(original_code)
        with test_path.open("w") as f:
            f.write(test_code)

        fto = FunctionToOptimize("calculate_portfolio_metrics", fto_file_path, parents=[])
        file_path_to_helper_class = {}
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
        file_path_to_helper_classes = {}
        instrument_codeflash_capture(fto, file_path_to_helper_classes, tests_root)

        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )

        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)

        # Now, let's say we optimize the code and make changes.
        new_fto_code = """import math
from typing import List, Tuple, Optional

def calculate_portfolio_metrics(
    investments: List[Tuple[str, float, float]], 
    risk_free_rate: float = 0.02
) -> dict:
    if not investments:
        raise ValueError("Investments list cannot be empty")

    total_weight = sum(w for _, w, _ in investments)
    if total_weight != 1.0:  # Should use tolerance check
        raise ValueError("Portfolio weights must sum to 1.0")

    weighted_return = 1.0
    for _, weight, ret in investments:
        weighted_return *= (1 + ret) ** weight
    weighted_return = weighted_return - 1.0  # Convert back from geometric

    returns = [r for _, _, r in investments]
    mean_return = sum(returns) / len(returns)
    volatility = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))

    # BUG 4: Sharpe ratio calculation is correct but uses wrong inputs
    if volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (weighted_return - risk_free_rate) / volatility

    def risk_adjusted_return(return_val, weight):
        return (return_val - risk_free_rate) / (weight * return_val) if weight * return_val != 0 else return_val
    
    best_asset = max(investments, key=lambda x: risk_adjusted_return(x[2], x[1]))
    worst_asset = min(investments, key=lambda x: risk_adjusted_return(x[2], x[1]))

    return {
        "weighted_return": round(weighted_return, 6),
        "volatility": 2, 
        "sharpe_ratio": round(sharpe_ratio, 6),
        "best_performing": (best_asset[0], round(best_asset[2], 6)),
        "worst_performing": (worst_asset[0], round(worst_asset[2], 6)),
        "total_assets": len(investments),
    }
"""
        with fto_file_path.open("w") as f:
            f.write(new_fto_code)
        # Instrument codeflash capture
        candidate_fto_code = Path(fto.file_path).read_text("utf-8")
        candidate_helper_code = {}
        for file_path in file_path_to_helper_class:
            candidate_helper_code[file_path] = Path(file_path).read_text("utf-8")
        file_path_to_helper_classes = {}
        instrument_codeflash_capture(fto, file_path_to_helper_classes, tests_root)
        modified_test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )
        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)
        matched, diffs = compare_test_results(test_results, modified_test_results)

        assert not matched

        new_fixed_code = """import math
from typing import List, Tuple, Optional

def calculate_portfolio_metrics(
    investments: List[Tuple[str, float, float]], 
    risk_free_rate: float = 0.02
) -> dict:
    if not investments:
        raise ValueError("Investments list cannot be empty")

    # Tolerant weight check (matches original)
    total_weight = sum(weight for _, weight, _ in investments)
    if abs(total_weight - 1.0) > 1e-10:
        raise ValueError("Portfolio weights must sum to 1.0")

    # Same weighted return as original
    weighted_return = sum(weight * ret for _, weight, ret in investments)

    # Same volatility formula as original
    volatility = math.sqrt(sum((weight * ret) ** 2 for _, weight, ret in investments))

    # Same Sharpe ratio logic
    if volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (weighted_return - risk_free_rate) / volatility

    # Same best/worst logic (based on return only)
    best_asset = max(investments, key=lambda x: x[2])
    worst_asset = min(investments, key=lambda x: x[2])

    return {
        "weighted_return": round(weighted_return, 6),
        "volatility": round(volatility, 6),
        "sharpe_ratio": round(sharpe_ratio, 6),
        "best_performing": (best_asset[0], round(best_asset[2], 6)),
        "worst_performing": (worst_asset[0], round(worst_asset[2], 6)),
        "total_assets": len(investments),
    }
"""
        with fto_file_path.open("w") as f:
            f.write(new_fixed_code)
        candidate_fto_code = Path(fto.file_path).read_text("utf-8")
        candidate_helper_code = {}
        for file_path in file_path_to_helper_class:
            candidate_helper_code[file_path] = Path(file_path).read_text("utf-8")
        file_path_to_helper_classes = {}
        instrument_codeflash_capture(fto, file_path_to_helper_classes, tests_root)
        modified_test_results_2, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )
        # Remove instrumentation
        FunctionOptimizer.write_code_and_helpers(candidate_fto_code, candidate_helper_code, fto.file_path)
        matched, diffs = compare_test_results(test_results, modified_test_results_2)
        # now the test should match and no diffs should be found
        assert len(diffs) == 0
        assert matched

    finally:
        test_path.unlink(missing_ok=True)
        fto_file_path.unlink(missing_ok=True)


def test_codeflash_capture_with_slots_class() -> None:
    """Test that codeflash_capture works with classes that use __slots__ instead of __dict__."""
    test_code = """
from code_to_optimize.tests.pytest.sample_code import SlotsClass
import unittest

def test_slots_class():
    obj = SlotsClass(10, "test")
    assert obj.x == 10
    assert obj.y == "test"
"""
    test_dir = (Path(__file__).parent.parent / "code_to_optimize" / "tests" / "pytest").resolve()
    tmp_dir_path = get_run_tmp_file(Path("test_return_values"))
    sample_code = f"""
from codeflash.verification.codeflash_capture import codeflash_capture

class SlotsClass:
    __slots__ = ('x', 'y')

    @codeflash_capture(function_name="SlotsClass.__init__", tmp_dir_path="{tmp_dir_path.as_posix()}", tests_root="{test_dir.as_posix()}")
    def __init__(self, x, y):
        self.x = x
        self.y = y
"""
    test_file_name = "test_slots_class_temp.py"
    test_path = test_dir / test_file_name
    test_path_perf = test_dir / "test_slots_class_temp_perf.py"

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
            function_name="__init__",
            file_path=sample_code_path,
            parents=[FunctionParent(name="SlotsClass", type="ClassDef")],
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
            min_outer_loops=1,
            max_outer_loops=1,
            testing_time=0.1,
        )

        # Test should pass and capture the slots values
        assert len(test_results) == 1
        assert test_results[0].did_pass
        # The return value should contain the slot values
        assert test_results[0].return_value[0]["x"] == 10
        assert test_results[0].return_value[0]["y"] == "test"

    finally:
        test_path.unlink(missing_ok=True)
        sample_code_path.unlink(missing_ok=True)
