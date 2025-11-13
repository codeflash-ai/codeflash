"""Comprehensive unit tests for TestResults.file_to_no_of_tests method."""

from collections import Counter
from pathlib import Path

import pytest

from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults, TestType


class TestFileToNoOfTests:
    """Test suite for TestResults.file_to_no_of_tests method."""

    def test_empty_test_results(self):
        """Test with empty test results."""
        test_results = TestResults()
        counter = test_results.file_to_no_of_tests([])
        assert counter == Counter()
        assert len(counter) == 0

    def test_empty_test_functions_to_remove(self):
        """Test with empty list of test functions to remove."""
        test_results = TestResults()
        test_results.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test.module",
                    test_class_name="TestClass",
                    test_function_name="test_function",
                    function_getting_tested="target_func",
                    iteration_id="1",
                ),
                file_name=Path("/tmp/test_file.py"),
                did_pass=True,
                runtime=100,
                test_framework="pytest",
                test_type=TestType.GENERATED_REGRESSION,
                return_value=None,
                timed_out=False,
                loop_index=1,
            )
        )
        counter = test_results.file_to_no_of_tests([])
        assert counter == Counter({Path("/tmp/test_file.py"): 1})

    def test_single_test_not_removed(self):
        """Test with a single test that should not be removed."""
        test_results = TestResults()
        test_results.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test.module",
                    test_class_name="TestClass",
                    test_function_name="test_keep",
                    function_getting_tested="target_func",
                    iteration_id="1",
                ),
                file_name=Path("/tmp/test_file.py"),
                did_pass=True,
                runtime=100,
                test_framework="pytest",
                test_type=TestType.GENERATED_REGRESSION,
                return_value=None,
                timed_out=False,
                loop_index=1,
            )
        )
        counter = test_results.file_to_no_of_tests(["test_remove"])
        assert counter == Counter({Path("/tmp/test_file.py"): 1})

    def test_single_test_removed(self):
        """Test with a single test that should be removed."""
        test_results = TestResults()
        test_results.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test.module",
                    test_class_name="TestClass",
                    test_function_name="test_remove",
                    function_getting_tested="target_func",
                    iteration_id="1",
                ),
                file_name=Path("/tmp/test_file.py"),
                did_pass=True,
                runtime=100,
                test_framework="pytest",
                test_type=TestType.GENERATED_REGRESSION,
                return_value=None,
                timed_out=False,
                loop_index=1,
            )
        )
        counter = test_results.file_to_no_of_tests(["test_remove"])
        assert counter == Counter()

    def test_multiple_tests_same_file(self):
        """Test with multiple tests in the same file."""
        test_results = TestResults()
        for i in range(5):
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path="test.module",
                        test_class_name="TestClass",
                        test_function_name=f"test_func_{i}",
                        function_getting_tested="target_func",
                        iteration_id=str(i),
                    ),
                    file_name=Path("/tmp/test_file.py"),
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=TestType.GENERATED_REGRESSION,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )
        counter = test_results.file_to_no_of_tests([])
        assert counter == Counter({Path("/tmp/test_file.py"): 5})

    def test_multiple_tests_different_files(self):
        """Test with multiple tests in different files."""
        test_results = TestResults()
        files = [Path(f"/tmp/test_file_{i}.py") for i in range(3)]
        for i, file_path in enumerate(files):
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path=f"test.module{i}",
                        test_class_name="TestClass",
                        test_function_name=f"test_func_{i}",
                        function_getting_tested="target_func",
                        iteration_id=str(i),
                    ),
                    file_name=file_path,
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=TestType.GENERATED_REGRESSION,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )
        counter = test_results.file_to_no_of_tests([])
        expected = Counter({files[0]: 1, files[1]: 1, files[2]: 1})
        assert counter == expected

    def test_mixed_test_types(self):
        """Test with different test types - only GENERATED_REGRESSION should be counted."""
        test_results = TestResults()
        test_types = [
            TestType.EXISTING_UNIT_TEST,
            TestType.INSPIRED_REGRESSION,
            TestType.GENERATED_REGRESSION,
            TestType.REPLAY_TEST,
            TestType.CONCOLIC_COVERAGE_TEST,
            TestType.INIT_STATE_TEST,
        ]

        for i, test_type in enumerate(test_types):
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path="test.module",
                        test_class_name="TestClass",
                        test_function_name=f"test_func_{i}",
                        function_getting_tested="target_func",
                        iteration_id=str(i),
                    ),
                    file_name=Path(f"/tmp/test_file_{i}.py"),
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=test_type,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )

        counter = test_results.file_to_no_of_tests([])
        # Only the GENERATED_REGRESSION test should be counted
        assert counter == Counter({Path("/tmp/test_file_2.py"): 1})

    def test_partial_removal(self):
        """Test removing some but not all tests from a file."""
        test_results = TestResults()
        test_names = ["test_keep_1", "test_remove_1", "test_keep_2", "test_remove_2"]

        for name in test_names:
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path="test.module",
                        test_class_name="TestClass",
                        test_function_name=name,
                        function_getting_tested="target_func",
                        iteration_id=name,
                    ),
                    file_name=Path("/tmp/test_file.py"),
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=TestType.GENERATED_REGRESSION,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )

        counter = test_results.file_to_no_of_tests(["test_remove_1", "test_remove_2"])
        assert counter == Counter({Path("/tmp/test_file.py"): 2})  # Only test_keep_1 and test_keep_2

    def test_none_test_function_name(self):
        """Test with None test_function_name."""
        test_results = TestResults()
        test_results.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test.module",
                    test_class_name="TestClass",
                    test_function_name=None,
                    function_getting_tested="target_func",
                    iteration_id="1",
                ),
                file_name=Path("/tmp/test_file.py"),
                did_pass=True,
                runtime=100,
                test_framework="pytest",
                test_type=TestType.GENERATED_REGRESSION,
                return_value=None,
                timed_out=False,
                loop_index=1,
            )
        )
        # None should not match any string in test_functions_to_remove
        counter = test_results.file_to_no_of_tests(["test_remove"])
        assert counter == Counter({Path("/tmp/test_file.py"): 1})

    def test_duplicate_file_paths(self):
        """Test counting with duplicate file paths across multiple tests."""
        test_results = TestResults()
        file_path = Path("/tmp/test_file.py")

        # Add multiple tests with the same file path
        for i in range(3):
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path="test.module",
                        test_class_name="TestClass",
                        test_function_name=f"test_func_{i}",
                        function_getting_tested="target_func",
                        iteration_id=str(i),
                    ),
                    file_name=file_path,
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=TestType.GENERATED_REGRESSION,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )

        counter = test_results.file_to_no_of_tests([])
        assert counter == Counter({file_path: 3})

    def test_complex_scenario(self):
        """Test complex scenario with mixed conditions."""
        test_results = TestResults()

        # File 1: Mix of test types
        for i, test_type in enumerate([TestType.GENERATED_REGRESSION, TestType.EXISTING_UNIT_TEST]):
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path="test.module1",
                        test_class_name="TestClass",
                        test_function_name=f"test_file1_{i}",
                        function_getting_tested="target_func",
                        iteration_id=str(i),
                    ),
                    file_name=Path("/tmp/file1.py"),
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=test_type,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )

        # File 2: Tests to be removed and kept
        for name in ["test_keep", "test_remove"]:
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path="test.module2",
                        test_class_name="TestClass",
                        test_function_name=name,
                        function_getting_tested="target_func",
                        iteration_id=name,
                    ),
                    file_name=Path("/tmp/file2.py"),
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=TestType.GENERATED_REGRESSION,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )

        # File 3: All GENERATED_REGRESSION tests
        for i in range(3):
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path="test.module3",
                        test_class_name="TestClass",
                        test_function_name=f"test_file3_{i}",
                        function_getting_tested="target_func",
                        iteration_id=str(i),
                    ),
                    file_name=Path("/tmp/file3.py"),
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=TestType.GENERATED_REGRESSION,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )

        counter = test_results.file_to_no_of_tests(["test_remove"])
        expected = Counter({
            Path("/tmp/file1.py"): 1,  # Only 1 GENERATED_REGRESSION test
            Path("/tmp/file2.py"): 1,  # Only test_keep (test_remove is excluded)
            Path("/tmp/file3.py"): 3,  # All 3 tests
        })
        assert counter == expected

    def test_case_sensitivity(self):
        """Test that function name matching is case-sensitive."""
        test_results = TestResults()
        test_results.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test.module",
                    test_class_name="TestClass",
                    test_function_name="Test_Function",
                    function_getting_tested="target_func",
                    iteration_id="1",
                ),
                file_name=Path("/tmp/test_file.py"),
                did_pass=True,
                runtime=100,
                test_framework="pytest",
                test_type=TestType.GENERATED_REGRESSION,
                return_value=None,
                timed_out=False,
                loop_index=1,
            )
        )

        # Should not remove because case doesn't match
        counter = test_results.file_to_no_of_tests(["test_function"])
        assert counter == Counter({Path("/tmp/test_file.py"): 1})

        # Should remove with correct case
        counter = test_results.file_to_no_of_tests(["Test_Function"])
        assert counter == Counter()

    def test_windows_paths(self):
        """Test with Windows-style paths."""
        test_results = TestResults()
        windows_path = Path("C:\\Users\\test\\test_file.py")

        test_results.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test.module",
                    test_class_name="TestClass",
                    test_function_name="test_func",
                    function_getting_tested="target_func",
                    iteration_id="1",
                ),
                file_name=windows_path,
                did_pass=True,
                runtime=100,
                test_framework="pytest",
                test_type=TestType.GENERATED_REGRESSION,
                return_value=None,
                timed_out=False,
                loop_index=1,
            )
        )

        counter = test_results.file_to_no_of_tests([])
        assert counter == Counter({windows_path: 1})

    def test_relative_and_absolute_paths(self):
        """Test with both relative and absolute paths."""
        test_results = TestResults()
        paths = [
            Path("/absolute/path/test.py"),
            Path("relative/path/test.py"),
            Path("./current/dir/test.py"),
            Path("../parent/dir/test.py"),
        ]

        for i, path in enumerate(paths):
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path=f"test.module{i}",
                        test_class_name="TestClass",
                        test_function_name=f"test_func_{i}",
                        function_getting_tested="target_func",
                        iteration_id=str(i),
                    ),
                    file_name=path,
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=TestType.GENERATED_REGRESSION,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )

        counter = test_results.file_to_no_of_tests([])
        expected = Counter({path: 1 for path in paths})
        assert counter == expected

    def test_large_removal_list(self):
        """Test with a large list of functions to remove."""
        test_results = TestResults()
        num_tests = 100
        removal_list = [f"test_remove_{i}" for i in range(50)]

        for i in range(num_tests):
            test_name = f"test_remove_{i}" if i < 50 else f"test_keep_{i}"
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path="test.module",
                        test_class_name="TestClass",
                        test_function_name=test_name,
                        function_getting_tested="target_func",
                        iteration_id=str(i),
                    ),
                    file_name=Path("/tmp/test_file.py"),
                    did_pass=True,
                    runtime=100,
                    test_framework="pytest",
                    test_type=TestType.GENERATED_REGRESSION,
                    return_value=None,
                    timed_out=False,
                    loop_index=1,
                )
            )

        counter = test_results.file_to_no_of_tests(removal_list)
        assert counter == Counter({Path("/tmp/test_file.py"): 50})  # 50 kept, 50 removed