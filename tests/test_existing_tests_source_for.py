from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import pytest

from codeflash.result.create_pr import existing_tests_source_for


class MockInvocationId(NamedTuple):
    test_module_path: str
    test_class_name: str | None
    test_function_name: str


class MockTestsInFile(NamedTuple):
    test_file: str


class MockFunctionCalledInTest(NamedTuple):
    tests_in_file: MockTestsInFile


class MockTestConfig(NamedTuple):
    tests_root: Path
    project_root_path: Path


class TestExistingTestsSourceFor:
    """Test cases for existing_tests_source_for function."""

    def test_no_test_files_found(self):
        """Test when no test files are found for the function."""
        function_qualified_name = "module.function_name"
        function_to_tests = {}
        test_cfg = MockTestConfig(
            tests_root=Path("/project/tests"),
            project_root_path=Path("/project")
        )
        original_runtimes = {}
        optimized_runtimes = {}

        result = existing_tests_source_for(
            function_qualified_name,
            function_to_tests,
            test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        assert result == ""

    def test_single_test_file_with_function_test(self):
        """Test with a single test file containing one test function."""
        function_qualified_name = "module.function_name"
        test_file_path = "/project/tests/test_module.py"

        function_to_tests = {
            function_qualified_name: {
                MockFunctionCalledInTest(
                    tests_in_file=MockTestsInFile(test_file=test_file_path)
                )
            }
        }

        test_cfg = MockTestConfig(
            tests_root=Path("/project/tests"),
            project_root_path=Path("/project")
        )

        invocation_id = MockInvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_function"
        )

        original_runtimes = {invocation_id: [1000000, 1100000, 900000]}  # 1ms, 1.1ms, 0.9ms
        optimized_runtimes = {invocation_id: [500000, 600000, 400000]}   # 0.5ms, 0.6ms, 0.4ms

        result = existing_tests_source_for(
            function_qualified_name,
            function_to_tests,
            test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """- test_module.py
    - test_function: 900μs -> 400μs $\\color{green}(55.56\\%)$

"""
        assert result == expected

    def test_single_test_file_with_class_test(self):
        """Test with a single test file containing a test method in a class."""
        function_qualified_name = "module.function_name"
        test_file_path = "/project/tests/test_module.py"

        function_to_tests = {
            function_qualified_name: {
                MockFunctionCalledInTest(
                    tests_in_file=MockTestsInFile(test_file=test_file_path)
                )
            }
        }

        test_cfg = MockTestConfig(
            tests_root=Path("/project/tests"),
            project_root_path=Path("/project")
        )

        invocation_id = MockInvocationId(
            test_module_path="tests.test_module",
            test_class_name="TestClass",
            test_function_name="test_method"
        )

        original_runtimes = {invocation_id: [2000000]}  # 2ms
        optimized_runtimes = {invocation_id: [3000000]}  # 3ms (slower)

        result = existing_tests_source_for(
            function_qualified_name,
            function_to_tests,
            test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """- test_module.py
    - TestClass.test_method: 2.00ms -> 3.00ms $\\color{red}(-50.00\\%)$

"""
        assert result == expected

    def test_multiple_test_files_and_methods(self):
        """Test with multiple test files and multiple test methods."""
        function_qualified_name = "module.function_name"
        test_file_path1 = "/project/tests/test_module1.py"
        test_file_path2 = "/project/tests/test_module2.py"

        function_to_tests = {
            function_qualified_name: {
                MockFunctionCalledInTest(
                    tests_in_file=MockTestsInFile(test_file=test_file_path1)
                ),
                MockFunctionCalledInTest(
                    tests_in_file=MockTestsInFile(test_file=test_file_path2)
                )
            }
        }

        test_cfg = MockTestConfig(
            tests_root=Path("/project/tests"),
            project_root_path=Path("/project")
        )

        invocation_id1 = MockInvocationId(
            test_module_path="tests.test_module1",
            test_class_name=None,
            test_function_name="test_function1"
        )

        invocation_id2 = MockInvocationId(
            test_module_path="tests.test_module1",
            test_class_name="TestClass",
            test_function_name="test_method1"
        )

        invocation_id3 = MockInvocationId(
            test_module_path="tests.test_module2",
            test_class_name=None,
            test_function_name="test_function2"
        )

        original_runtimes = {
            invocation_id1: [1000000],  # 1ms
            invocation_id2: [2000000],  # 2ms
            invocation_id3: [500000]    # 0.5ms
        }
        optimized_runtimes = {
            invocation_id1: [800000],   # 0.8ms
            invocation_id2: [1500000],  # 1.5ms
            invocation_id3: [400000]    # 0.4ms
        }

        result = existing_tests_source_for(
            function_qualified_name,
            function_to_tests,
            test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """- test_module1.py
    - TestClass.test_method1: 2.00ms -> 1.50ms $\\color{green}(25.00\\%)$
    - test_function1: 1.00ms -> 800μs $\\color{green}(20.00\\%)$

- test_module2.py
    - test_function2: 500μs -> 400μs $\\color{green}(20.00\\%)$

"""
        assert result == expected

    def test_missing_runtime_data(self):
        """Test when runtime data is missing for some tests."""
        function_qualified_name = "module.function_name"
        test_file_path = "/project/tests/test_module.py"

        function_to_tests = {
            function_qualified_name: {
                MockFunctionCalledInTest(
                    tests_in_file=MockTestsInFile(test_file=test_file_path)
                )
            }
        }

        test_cfg = MockTestConfig(
            tests_root=Path("/project/tests"),
            project_root_path=Path("/project")
        )

        invocation_id1 = MockInvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_with_original_only"
        )

        invocation_id2 = MockInvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_with_optimized_only"
        )

        original_runtimes = {invocation_id1: [1000000]}  # Only original
        optimized_runtimes = {invocation_id2: [500000]}  # Only optimized

        result = existing_tests_source_for(
            function_qualified_name,
            function_to_tests,
            test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """- test_module.py
    - test_with_optimized_only: NaN -> 500μs
    - test_with_original_only: 1.00ms -> NaN

"""
        assert result == expected

    def test_nested_test_directory(self):
        """Test with nested test directories."""
        function_qualified_name = "module.function_name"
        test_file_path = "/project/tests/unit/test_module.py"

        function_to_tests = {
            function_qualified_name: {
                MockFunctionCalledInTest(
                    tests_in_file=MockTestsInFile(test_file=test_file_path)
                )
            }
        }

        test_cfg = MockTestConfig(
            tests_root=Path("/project/tests"),
            project_root_path=Path("/project")
        )

        invocation_id = MockInvocationId(
            test_module_path="tests.unit.test_module",
            test_class_name=None,
            test_function_name="test_function"
        )

        original_runtimes = {invocation_id: [1000000]}
        optimized_runtimes = {invocation_id: [800000]}

        result = existing_tests_source_for(
            function_qualified_name,
            function_to_tests,
            test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """- unit/test_module.py
    - test_function: 1.00ms -> 800μs $\\color{green}(20.00\\%)$

"""
        assert result == expected

    def test_multiple_invocations_same_test(self):
        """Test when the same test has multiple invocations (runtimes are summed)."""
        function_qualified_name = "module.function_name"
        test_file_path = "/project/tests/test_module.py"

        function_to_tests = {
            function_qualified_name: {
                MockFunctionCalledInTest(
                    tests_in_file=MockTestsInFile(test_file=test_file_path)
                )
            }
        }

        test_cfg = MockTestConfig(
            tests_root=Path("/project/tests"),
            project_root_path=Path("/project")
        )

        # Same test function with multiple invocations
        invocation_id1 = MockInvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_function"
        )

        invocation_id2 = MockInvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_function"
        )

        original_runtimes = {
            invocation_id1: [1000000, 1200000],  # min: 1ms
            invocation_id2: [800000, 900000]     # min: 0.8ms
        }
        optimized_runtimes = {
            invocation_id1: [600000, 700000],    # min: 0.6ms
            invocation_id2: [400000, 500000]     # min: 0.4ms
        }

        result = existing_tests_source_for(
            function_qualified_name,
            function_to_tests,
            test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        # Total original: 1ms + 0.8ms = 1.8ms
        # Total optimized: 0.6ms + 0.4ms = 1ms
        expected = """- test_module.py
    - test_function: 1.80ms -> 1.00ms $\\color{green}(44.44\\%)$

"""
        assert result == expected

    def test_zero_runtime_values(self):
        """Test handling of zero runtime values."""
        function_qualified_name = "module.function_name"
        test_file_path = "/project/tests/test_module.py"

        function_to_tests = {
            function_qualified_name: {
                MockFunctionCalledInTest(
                    tests_in_file=MockTestsInFile(test_file=test_file_path)
                )
            }
        }

        test_cfg = MockTestConfig(
            tests_root=Path("/project/tests"),
            project_root_path=Path("/project")
        )

        invocation_id = MockInvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_function"
        )

        original_runtimes = {invocation_id: [0]}
        optimized_runtimes = {invocation_id: [0]}

        result = existing_tests_source_for(
            function_qualified_name,
            function_to_tests,
            test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """- test_module.py
    - test_function: NaN -> NaN

"""
        assert result == expected
