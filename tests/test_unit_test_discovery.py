import os
import tempfile
from pathlib import Path

from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.verification.verification_utils import TestConfig


def test_unit_test_discovery_pytest():
    project_path = Path(__file__).parent.parent.resolve() / "code_to_optimize"
    tests_path = project_path / "tests" / "pytest"
    test_config = TestConfig(
        tests_root=tests_path,
        project_root_path=project_path,
        test_framework="pytest",
        tests_project_rootdir=tests_path.parent,
    )
    tests = discover_unit_tests(test_config)
    assert len(tests) > 0
    # print(tests)


def test_unit_test_discovery_unittest():
    project_path = Path(__file__).parent.parent.resolve() / "code_to_optimize"
    test_path = project_path / "tests" / "unittest"
    test_config = TestConfig(
        tests_root=project_path,
        project_root_path=project_path,
        test_framework="unittest",
        tests_project_rootdir=project_path.parent,
    )
    os.chdir(project_path)
    tests = discover_unit_tests(test_config)
    # assert len(tests) > 0
    # Unittest discovery within a pytest environment does not work

def test_benchmark_unit_test_discovery_pytest():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a dummy test file
        test_file_path = Path(tmpdirname) / "test_dummy.py"
        test_file_content = """
from bubble_sort import sorter

def test_benchmark_sort(benchmark):
     benchmark(sorter, [5, 4, 3, 2, 1, 0])

def test_normal_test():
    assert sorter(list(reversed(range(100)))) == list(range(100))

def test_normal_test2():
    assert sorter(list(reversed(range(100)))) == list(range(100))"""
        test_file_path.write_text(test_file_content)
        path_obj_tempdirname = Path(tmpdirname)

        # Create a file that the test file is testing
        code_file_path = path_obj_tempdirname / "bubble_sort.py"
        code_file_content = """
def sorter(arr):
    return sorted(arr)"""
        code_file_path.write_text(code_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tempdirname,
            project_root_path=path_obj_tempdirname,
            test_framework="pytest",
            tests_project_rootdir=path_obj_tempdirname.parent,
        )

        # Discover tests
        tests = discover_unit_tests(test_config)
        assert len(tests) == 1
        assert 'bubble_sort.sorter' in tests
        assert len(tests['bubble_sort.sorter']) == 2
        functions = [test.tests_in_file.test_function for test in tests['bubble_sort.sorter']]
        assert 'test_normal_test' in functions
        assert 'test_normal_test2' in functions
        assert 'test_benchmark_sort' not in functions

def test_discover_tests_pytest_with_temp_dir_root():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a dummy test file
        test_file_path = Path(tmpdirname) / "test_dummy.py"
        test_file_content = (
            "import pytest\n"
            "from dummy_code import dummy_function\n\n"
            "def test_dummy_function():\n"
            "    assert dummy_function() is True\n"
            "@pytest.mark.parametrize('param', [True])\n"
            "def test_dummy_parametrized_function(param):\n"
            "    assert dummy_function() is True\n"
        )
        test_file_path.write_text(test_file_content)
        path_obj_tempdirname = Path(tmpdirname)

        # Create a file that the test file is testing
        code_file_path = path_obj_tempdirname / "dummy_code.py"
        code_file_content = "def dummy_function():\n    return True\n"
        code_file_path.write_text(code_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tempdirname,
            project_root_path=path_obj_tempdirname,
            test_framework="pytest",
            tests_project_rootdir=path_obj_tempdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the dummy test file is discovered
        assert len(discovered_tests) == 1
        assert len(discovered_tests["dummy_code.dummy_function"]) == 2
        assert discovered_tests["dummy_code.dummy_function"][0].tests_in_file.test_file == test_file_path
        assert discovered_tests["dummy_code.dummy_function"][1].tests_in_file.test_file == test_file_path
        assert {
            discovered_tests["dummy_code.dummy_function"][0].tests_in_file.test_function,
            discovered_tests["dummy_code.dummy_function"][1].tests_in_file.test_function,
        } == {"test_dummy_parametrized_function[True]", "test_dummy_function"}


def test_discover_tests_pytest_with_multi_level_dirs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create multi-level directories
        level1_dir = path_obj_tmpdirname / "level1"
        level2_dir = level1_dir / "level2"
        level2_dir.mkdir(parents=True)

        # Create code files at each level
        root_code_file_path = path_obj_tmpdirname / "root_code.py"
        root_code_file_content = "def root_function():\n    return True\n"
        root_code_file_path.write_text(root_code_file_content)

        level1_code_file_path = level1_dir / "level1_code.py"
        level1_code_file_content = "def level1_function():\n    return True\n"
        level1_code_file_path.write_text(level1_code_file_content)

        level2_code_file_path = level2_dir / "level2_code.py"
        level2_code_file_content = "def level2_function():\n    return True\n"
        level2_code_file_path.write_text(level2_code_file_content)

        # Create a test file at the root level
        root_test_file_path = path_obj_tmpdirname / "test_root.py"
        root_test_file_content = (
            "from root_code import root_function\n\n"
            "def test_root_function():\n"
            "    assert True\n"
            "    assert root_function() is True\n"
        )
        root_test_file_path.write_text(root_test_file_content)

        # Create a test file at level 1
        level1_test_file_path = level1_dir / "test_level1.py"
        level1_test_file_content = (
            "from level1_code import level1_function\n\n"
            "def test_level1_function():\n"
            "    assert True\n"
            "    assert level1_function() is True\n"
        )
        level1_test_file_path.write_text(level1_test_file_content)

        # Create a test file at level 2
        level2_test_file_path = level2_dir / "test_level2.py"
        level2_test_file_content = (
            "from level2_code import level2_function\n\n"
            "def test_level2_function():\n"
            "    assert True\n"
            "    assert level2_function() is True\n"
        )
        level2_test_file_path.write_text(level2_test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test files at all levels are discovered
        assert len(discovered_tests) == 3
        assert discovered_tests["root_code.root_function"][0].tests_in_file.test_file == root_test_file_path
        assert (
            discovered_tests["level1.level1_code.level1_function"][0].tests_in_file.test_file == level1_test_file_path
        )

        assert (
            discovered_tests["level1.level2.level2_code.level2_function"][0].tests_in_file.test_file
            == level2_test_file_path
        )


def test_discover_tests_pytest_dirs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create multi-level directories
        level1_dir = Path(tmpdirname) / "level1"
        level2_dir = level1_dir / "level2"
        level2_dir.mkdir(parents=True)
        level3_dir = level1_dir / "level3"
        level3_dir.mkdir(parents=True)

        # Create code files at each level
        root_code_file_path = path_obj_tmpdirname / "root_code.py"
        root_code_file_content = "def root_function():\n    return True\n"
        root_code_file_path.write_text(root_code_file_content)

        level1_code_file_path = level1_dir / "level1_code.py"
        level1_code_file_content = "def level1_function():\n    return True\n"
        level1_code_file_path.write_text(level1_code_file_content)

        level2_code_file_path = level2_dir / "level2_code.py"
        level2_code_file_content = "def level2_function():\n    return True\n"
        level2_code_file_path.write_text(level2_code_file_content)

        level3_code_file_path = level3_dir / "level3_code.py"
        level3_code_file_content = "def level3_function():\n    return True\n"
        level3_code_file_path.write_text(level3_code_file_content)

        # Create a test file at the root level
        root_test_file_path = path_obj_tmpdirname / "test_root.py"
        root_test_file_content = (
            "from root_code import root_function\n\n"
            "def test_root_function():\n"
            "    assert True\n"
            "    assert root_function() is True\n"
        )
        root_test_file_path.write_text(root_test_file_content)

        # Create a test file at level 1
        level1_test_file_path = level1_dir / "test_level1.py"
        level1_test_file_content = (
            "from level1_code import level1_function\n\n"
            "def test_level1_function():\n"
            "    assert True\n"
            "    assert level1_function() is True\n"
        )
        level1_test_file_path.write_text(level1_test_file_content)

        # Create a test file at level 2
        level2_test_file_path = level2_dir / "test_level2.py"
        level2_test_file_content = (
            "from level2_code import level2_function\n\n"
            "def test_level2_function():\n"
            "    assert True\n"
            "    assert level2_function() is True\n"
        )
        level2_test_file_path.write_text(level2_test_file_content)

        level3_test_file_path = level3_dir / "test_level3.py"
        level3_test_file_content = (
            "from level3_code import level3_function\n\n"
            "def test_level3_function():\n"
            "    assert True\n"
            "    assert level3_function() is True\n"
        )
        level3_test_file_path.write_text(level3_test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test files at all levels are discovered
        assert len(discovered_tests) == 4
        assert discovered_tests["root_code.root_function"][0].tests_in_file.test_file == root_test_file_path
        assert (
            discovered_tests["level1.level1_code.level1_function"][0].tests_in_file.test_file == level1_test_file_path
        )
        assert (
            discovered_tests["level1.level2.level2_code.level2_function"][0].tests_in_file.test_file
            == level2_test_file_path
        )

        assert (
            discovered_tests["level1.level3.level3_code.level3_function"][0].tests_in_file.test_file
            == level3_test_file_path
        )


def test_discover_tests_pytest_with_class():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create a code file with a class
        code_file_path = path_obj_tmpdirname / "some_class_code.py"
        code_file_content = "class SomeClass:\n    def some_method(self):\n        return True\n"
        code_file_path.write_text(code_file_content)

        # Create a test file with a test class and a test method
        test_file_path = path_obj_tmpdirname / "test_some_class.py"
        test_file_content = (
            "from some_class_code import SomeClass\n\n"
            "def test_some_method():\n"
            "    instance = SomeClass()\n"
            "    assert instance.some_method() is True\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test class and method are discovered
        assert len(discovered_tests) == 1
        assert discovered_tests["some_class_code.SomeClass.some_method"][0].tests_in_file.test_file == test_file_path


def test_discover_tests_pytest_with_double_nested_directories():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create nested directories
        nested_dir = path_obj_tmpdirname / "nested" / "more_nested"
        nested_dir.mkdir(parents=True)

        # Create a code file with a class in the nested directory
        code_file_path = nested_dir / "nested_class_code.py"
        code_file_content = "class NestedClass:\n    def nested_method(self):\n        return True\n"
        code_file_path.write_text(code_file_content)

        # Create a test file with a test class and a test method in the nested directory
        test_file_path = nested_dir / "test_nested_class.py"
        test_file_content = (
            "from nested_class_code import NestedClass\n\n"
            "def test_nested_method():\n"
            "    instance = NestedClass()\n"
            "    assert instance.nested_method() is True\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test class and method are discovered
        assert len(discovered_tests) == 1
        assert (
            discovered_tests["nested.more_nested.nested_class_code.NestedClass.nested_method"][
                0
            ].tests_in_file.test_file
            == test_file_path
        )


def test_discover_tests_with_code_in_dir_and_test_in_subdir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create a directory for the code file
        code_dir = path_obj_tmpdirname / "code"
        code_dir.mkdir()

        # Create a code file in the code directory
        code_file_path = code_dir / "some_code.py"
        code_file_content = "def some_function():\n    return True\n"
        code_file_path.write_text(code_file_content)

        # Create a subdirectory for the test file within the code directory
        test_subdir = code_dir / "tests"
        test_subdir.mkdir()

        # Create a test file in the test subdirectory
        test_file_path = test_subdir / "test_some_code.py"
        test_file_content = (
            "import sys\n"
            "import os\n"
            # I am suspicious of this line, we should not need to insert the code directory into the path
            "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n"
            "from some_code import some_function\n\n"
            "def test_some_function():\n"
            "    assert some_function() is True\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the code directory as the root
        test_config = TestConfig(
            tests_root=test_subdir,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
            tests_project_rootdir=test_subdir.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test file is discovered and associated with the code file
        assert len(discovered_tests) == 1
        assert discovered_tests["code.some_code.some_function"][0].tests_in_file.test_file == test_file_path


def test_discover_tests_pytest_with_nested_class():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create a code file with a nested class
        code_file_path = path_obj_tmpdirname / "nested_class_code.py"
        code_file_content = (
            "class OuterClass:\n"
            "    class InnerClass:\n"
            "        def inner_method(self):\n"
            "            return True\n"
        )
        code_file_path.write_text(code_file_content)

        # Create a test file with a test for the nested class method
        test_file_path = path_obj_tmpdirname / "test_nested_class.py"
        test_file_content = (
            "from nested_class_code import OuterClass\n\n"
            "def test_inner_method():\n"
            "    instance = OuterClass.InnerClass()\n"
            "    assert instance.inner_method() is True\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test for the nested class method is discovered
        assert len(discovered_tests) == 1
        assert (
            discovered_tests["nested_class_code.OuterClass.InnerClass.inner_method"][0].tests_in_file.test_file
            == test_file_path
        )


def test_discover_tests_pytest_separate_moduledir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        rootdir = Path(tmpdirname)
        # Create a code file with a nested class
        codedir = rootdir / "src" / "mypackage"
        codedir.mkdir(parents=True)
        code_file_path = codedir / "code.py"
        code_file_content = "def find_common_tags(articles):\n    if not articles:\n        return set()\n"
        code_file_path.write_text(code_file_content)

        # Create a test file with a test for the nested class method
        testdir = rootdir / "tests"
        testdir.mkdir()
        test_file_path = testdir / "test_code.py"
        test_file_content = (
            "from mypackage.code import find_common_tags\n\n"
            "def test_common_tags():\n"
            "    assert find_common_tags(None) == set()\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=testdir,
            project_root_path=codedir.parent.resolve(),
            test_framework="pytest",
            tests_project_rootdir=testdir.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test for the nested class method is discovered
        assert len(discovered_tests) == 1
        assert discovered_tests["mypackage.code.find_common_tags"][0].tests_in_file.test_file == test_file_path


def test_unittest_discovery_with_pytest():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)

        # Create a simple code file
        code_file_path = path_obj_tmpdirname / "calculator.py"
        code_file_content = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        code_file_path.write_text(code_file_content)

        # Create a unittest test file
        test_file_path = path_obj_tmpdirname / "test_calculator.py"
        test_file_content = """
import unittest
from calculator import Calculator

class TestCalculator(unittest.TestCase):
    def test_add(self):
        calc = Calculator()
        self.assertEqual(calc.add(2, 2), 4)
"""
        test_file_path.write_text(test_file_content)

        # Configure test discovery
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",  # Using pytest framework to discover unittest tests
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Verify the unittest was discovered
        assert len(discovered_tests) == 1
        assert "calculator.Calculator.add" in discovered_tests
        assert len(discovered_tests["calculator.Calculator.add"]) == 1
        assert discovered_tests["calculator.Calculator.add"][0].tests_in_file.test_file == test_file_path
        assert discovered_tests["calculator.Calculator.add"][0].tests_in_file.test_function == "test_add"


def test_unittest_discovery_with_pytest_parent_class():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)

        # Create a simple code file
        code_file_path = path_obj_tmpdirname / "calculator.py"
        code_file_content = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        code_file_path.write_text(code_file_content)

        # Create a base test class file
        base_test_file_path = path_obj_tmpdirname / "base_test.py"
        base_test_content = """
import unittest

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.setup_called = True

    def tearDown(self):
        self.setup_called = False

    def assert_setup_called(self):
        self.assertTrue(self.setup_called, "Setup was not called")
"""
        base_test_file_path.write_text(base_test_content)

        # Create a unittest test file that extends the base test
        test_file_path = path_obj_tmpdirname / "test_calculator.py"
        test_file_content = """
from base_test import BaseTestCase
from calculator import Calculator

class ExtendedTestCase(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.calc = Calculator()

class TestCalculator(ExtendedTestCase):
    def test_add(self):
        self.assert_setup_called()
        self.assertEqual(self.calc.add(2, 2), 4)
"""
        test_file_path.write_text(test_file_content)

        # Configure test discovery
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",  # Using pytest framework to discover unittest tests
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Verify the unittest was discovered
        assert len(discovered_tests) == 2
        assert "calculator.Calculator.add" in discovered_tests
        assert len(discovered_tests["calculator.Calculator.add"]) == 1
        assert discovered_tests["calculator.Calculator.add"][0].tests_in_file.test_file == test_file_path
        assert discovered_tests["calculator.Calculator.add"][0].tests_in_file.test_function == "test_add"


def test_unittest_discovery_with_pytest_private():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)

        # Create a simple code file
        code_file_path = path_obj_tmpdirname / "calculator.py"
        code_file_content = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        code_file_path.write_text(code_file_content)

        # Create a unittest test file with a private test method (prefixed with _)
        test_file_path = path_obj_tmpdirname / "test_calculator.py"
        test_file_content = """
import unittest
from calculator import Calculator

class TestCalculator(unittest.TestCase):
    def _test_add(self):  # Private test method should not be discovered
        calc = Calculator()
        self.assertEqual(calc.add(2, 2), 4)
"""
        test_file_path.write_text(test_file_content)

        # Configure test discovery
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",  # Using pytest framework to discover unittest tests
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Verify no tests were discovered
        assert len(discovered_tests) == 0
        assert "calculator.Calculator.add" not in discovered_tests


def test_unittest_discovery_with_pytest_subtest():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)

        # Create a simple code file
        code_file_path = path_obj_tmpdirname / "calculator.py"
        code_file_content = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        code_file_path.write_text(code_file_content)

        # Create a unittest test file with parameterized tests
        test_file_path = path_obj_tmpdirname / "test_calculator.py"
        test_file_content = """
import unittest
from calculator import Calculator

class TestCalculator(unittest.TestCase):
    def test_add_with_parameters(self):
        calc = Calculator()
        test_cases = [
            {"a": 2, "b": 2, "expected": 4},
            {"a": 0, "b": 0, "expected": 0},
            {"a": -1, "b": 1, "expected": 0},
            {"a": 10, "b": -5, "expected": 5}
        ]

        for case in test_cases:
            with self.subTest(a=case["a"], b=case["b"]):
                result = calc.add(case["a"], case["b"])
                self.assertEqual(result, case["expected"])
"""
        test_file_path.write_text(test_file_content)

        # Configure test discovery
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",  # Using pytest framework to discover unittest tests
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Verify the unittest was discovered
        assert len(discovered_tests) == 1
        assert "calculator.Calculator.add" in discovered_tests
        assert len(discovered_tests["calculator.Calculator.add"]) == 1
        assert discovered_tests["calculator.Calculator.add"][0].tests_in_file.test_file == test_file_path
        assert (
            discovered_tests["calculator.Calculator.add"][0].tests_in_file.test_function == "test_add_with_parameters"
        )


def test_unittest_discovery_with_pytest_parameterized():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)

        # Create a simple code file
        code_file_path = path_obj_tmpdirname / "calculator.py"
        code_file_content = """
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
"""
        code_file_path.write_text(code_file_content)

        # Create a unittest test file with different parameterized patterns
        test_file_path = path_obj_tmpdirname / "test_calculator.py"
        test_file_content = """
import unittest
from parameterized import parameterized
from calculator import Calculator

class TestCalculator(unittest.TestCase):
    # Test with named parameters
    @parameterized.expand([
        ("positive_numbers", 2, 2, 4),
        ("zeros", 0, 0, 0),
        ("negative_and_positive", -1, 1, 0),
        ("negative_result", 10, -15, -5),
    ])
    def test_add(self, name, a, b, expected):
        calc = Calculator()
        result = calc.add(a, b)
        self.assertEqual(result, expected)

    # Test with unnamed parameters
    @parameterized.expand([
        (2, 3, 6),
        (0, 5, 0),
        (-2, 3, -6),
    ])
    def test_multiply(self, a, b, expected):
        calc = Calculator()
        result = calc.multiply(a, b)
        self.assertEqual(result, expected)

    # Test with mixed naming patterns
    @parameterized.expand([
        ("test with spaces", 1, 1, 2),
        ("test_with_underscores", 2, 2, 4),
        ("test.with.dots", 3, 3, 6),
        ("test-with-hyphens", 4, 4, 8),
    ])
    def test_add_mixed(self, name, a, b, expected):
        calc = Calculator()
        result = calc.add(a, b)
        self.assertEqual(result, expected)
"""
        test_file_path.write_text(test_file_content)

        # Configure test discovery
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
            tests_project_rootdir=path_obj_tmpdirname.parent,
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Verify the basic structure
        assert len(discovered_tests) == 2  # Should have tests for both add and multiply
        assert "calculator.Calculator.add" in discovered_tests
        assert "calculator.Calculator.multiply" in discovered_tests
