"""Comprehensive unit tests for TestResults.file_to_no_of_tests method."""

from pathlib import Path
from collections import Counter
import pytest

from codeflash.models.models import (
    FunctionTestInvocation,
    InvocationId,
    TestResults,
    TestType,
)


class TestFileToNoOfTests:
    """Test suite for the file_to_no_of_tests method of TestResults class."""

    _invocation_counter = 0  # Class variable to ensure unique IDs

    def create_test_invocation(
        self,
        file_name: str,
        test_function_name: str = "test_example",
        test_module_path: str = "test.module",
        function_getting_tested: str = "example_func",
        loop_index: int = 1,
        test_type: TestType = TestType.GENERATED_REGRESSION,
    ) -> FunctionTestInvocation:
        """Helper method to create a FunctionTestInvocation."""
        # Increment counter to ensure unique iteration_id
        TestFileToNoOfTests._invocation_counter += 1
        return FunctionTestInvocation(
            id=InvocationId(
                test_module_path=test_module_path,
                test_class_name=None,
                test_function_name=test_function_name,
                function_getting_tested=function_getting_tested,
                iteration_id=str(TestFileToNoOfTests._invocation_counter),
            ),
            file_name=Path(file_name),
            did_pass=True,
            runtime=1000,
            test_framework="pytest",
            test_type=test_type,
            return_value=None,
            timed_out=False,
            loop_index=loop_index,
        )

    def test_basic_functionality(self):
        """Test basic counting of test files with __unit_test_ in the name."""
        test_results = TestResults()

        # Add test invocations with __unit_test_ in the file name
        test_results.add(self.create_test_invocation("/path/to/test__unit_test_0.py"))
        test_results.add(self.create_test_invocation("/path/to/test__unit_test_1.py"))
        test_results.add(self.create_test_invocation("/path/to/test__unit_test_2.py"))

        result = test_results.file_to_no_of_tests([])

        assert isinstance(result, Counter)
        assert len(result) == 3
        assert result[Path("/path/to/test__unit_test_0.py")] == 1
        assert result[Path("/path/to/test__unit_test_1.py")] == 1
        assert result[Path("/path/to/test__unit_test_2.py")] == 1

    def test_multiple_tests_same_file(self):
        """Test counting multiple tests from the same file."""
        test_results = TestResults()

        # Add multiple test invocations from the same file
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_one"
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_two"
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_three"
            )
        )

        result = test_results.file_to_no_of_tests([])

        assert len(result) == 1
        assert result[Path("/path/to/test__unit_test_0.py")] == 3

    def test_exclude_non_unit_test_files(self):
        """Test that files without __unit_test_ in the name are excluded."""
        test_results = TestResults()

        # Add test invocations with and without __unit_test_
        test_results.add(self.create_test_invocation("/path/to/test__unit_test_0.py"))
        test_results.add(self.create_test_invocation("/path/to/regular_test.py"))
        test_results.add(self.create_test_invocation("/path/to/test_file.py"))
        test_results.add(self.create_test_invocation("/path/to/another__unit_test_file.py"))

        result = test_results.file_to_no_of_tests([])

        assert len(result) == 2
        assert result[Path("/path/to/test__unit_test_0.py")] == 1
        assert result[Path("/path/to/another__unit_test_file.py")] == 1
        assert Path("/path/to/regular_test.py") not in result
        assert Path("/path/to/test_file.py") not in result

    def test_test_functions_to_remove(self):
        """Test filtering out specific test functions."""
        test_results = TestResults()

        # Add test invocations with different function names
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_keep_me"
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_remove_me"
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_1.py",
                test_function_name="test_also_remove"
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_1.py",
                test_function_name="test_keep_this"
            )
        )

        result = test_results.file_to_no_of_tests(["test_remove_me", "test_also_remove"])

        assert result[Path("/path/to/test__unit_test_0.py")] == 1
        assert result[Path("/path/to/test__unit_test_1.py")] == 1

    def test_empty_test_results(self):
        """Test with empty test results."""
        test_results = TestResults()

        result = test_results.file_to_no_of_tests([])

        assert isinstance(result, Counter)
        assert len(result) == 0

    def test_all_tests_removed(self):
        """Test when all tests are in the removal list."""
        test_results = TestResults()

        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_remove_me"
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_1.py",
                test_function_name="test_also_remove"
            )
        )

        result = test_results.file_to_no_of_tests(["test_remove_me", "test_also_remove"])

        assert len(result) == 0

    def test_case_sensitivity(self):
        """Test that __unit_test_ matching is case-sensitive."""
        test_results = TestResults()

        test_results.add(self.create_test_invocation("/path/to/test__unit_test_0.py"))
        test_results.add(self.create_test_invocation("/path/to/test__UNIT_TEST_1.py"))
        test_results.add(self.create_test_invocation("/path/to/test__Unit_Test_2.py"))

        result = test_results.file_to_no_of_tests([])

        # Only lowercase __unit_test_ should be counted
        assert len(result) == 1
        assert result[Path("/path/to/test__unit_test_0.py")] == 1

    def test_unit_test_in_middle_of_path(self):
        """Test that __unit_test_ can appear anywhere in the file path."""
        test_results = TestResults()

        test_results.add(self.create_test_invocation("/path/__unit_test_/test_file.py"))
        test_results.add(self.create_test_invocation("/path/to/prefix__unit_test_suffix.py"))
        test_results.add(self.create_test_invocation("__unit_test_test.py"))

        result = test_results.file_to_no_of_tests([])

        # All files with __unit_test_ anywhere in the path should be counted
        assert len(result) == 3
        assert result[Path("/path/__unit_test_/test_file.py")] == 1
        assert result[Path("/path/to/prefix__unit_test_suffix.py")] == 1
        assert result[Path("__unit_test_test.py")] == 1

    def test_windows_path_handling(self):
        """Test handling of Windows-style paths."""
        test_results = TestResults()

        test_results.add(self.create_test_invocation("C:\\path\\to\\test__unit_test_0.py"))
        test_results.add(self.create_test_invocation("D:\\another\\path\\test__unit_test_1.py"))

        result = test_results.file_to_no_of_tests([])

        # Should handle Windows paths correctly
        assert len(result) == 2

    def test_duplicate_entries_same_test(self):
        """Test handling of duplicate test entries (same file and function name)."""
        test_results = TestResults()

        # Add the same test multiple times (different loop indices)
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_duplicate",
                loop_index=1
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_duplicate",
                loop_index=2
            )
        )

        result = test_results.file_to_no_of_tests([])

        # Should count both instances
        assert result[Path("/path/to/test__unit_test_0.py")] == 2

    def test_mixed_test_types(self):
        """Test with different test types."""
        test_results = TestResults()

        # Add tests with different TestType values
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_type=TestType.GENERATED_REGRESSION
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_1.py",
                test_type=TestType.EXISTING_UNIT_TEST
            )
        )
        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_2.py",
                test_type=TestType.INSPIRED_REGRESSION
            )
        )

        result = test_results.file_to_no_of_tests([])

        # All should be counted regardless of test type
        assert len(result) == 3

    def test_empty_removal_list(self):
        """Test with empty removal list."""
        test_results = TestResults()

        test_results.add(
            self.create_test_invocation(
                "/path/to/test__unit_test_0.py",
                test_function_name="test_function"
            )
        )

        result = test_results.file_to_no_of_tests([])

        assert result[Path("/path/to/test__unit_test_0.py")] == 1

    def test_none_test_function_name(self):
        """Test handling of None test function names."""
        test_results = TestResults()

        # Create an invocation with None test function name
        TestFileToNoOfTests._invocation_counter += 1
        invocation = FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test.module",
                test_class_name=None,
                test_function_name=None,  # None function name
                function_getting_tested="example_func",
                iteration_id=str(TestFileToNoOfTests._invocation_counter),
            ),
            file_name=Path("/path/to/test__unit_test_0.py"),
            did_pass=True,
            runtime=1000,
            test_framework="pytest",
            test_type=TestType.GENERATED_REGRESSION,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
        test_results.add(invocation)

        # Should not crash and should count the test
        result = test_results.file_to_no_of_tests([])
        assert result[Path("/path/to/test__unit_test_0.py")] == 1

        # Should not be affected by removal list since None won't match any string
        result = test_results.file_to_no_of_tests(["test_function"])
        assert result[Path("/path/to/test__unit_test_0.py")] == 1

    def test_complex_removal_scenarios(self):
        """Test complex scenarios with multiple files and removal patterns."""
        test_results = TestResults()

        # Add various test combinations
        for i in range(3):
            for test_name in ["test_keep", "test_remove", "test_another_keep"]:
                test_results.add(
                    self.create_test_invocation(
                        f"/path/to/test__unit_test_{i}.py",
                        test_function_name=test_name
                    )
                )

        # Remove only "test_remove"
        result = test_results.file_to_no_of_tests(["test_remove"])

        # Each file should have 2 tests remaining (test_keep and test_another_keep)
        assert len(result) == 3
        for i in range(3):
            assert result[Path(f"/path/to/test__unit_test_{i}.py")] == 2

    def test_special_characters_in_path(self):
        """Test handling of special characters in file paths."""
        test_results = TestResults()

        # Paths with special characters
        test_results.add(self.create_test_invocation("/path/with spaces/test__unit_test_0.py"))
        test_results.add(self.create_test_invocation("/path/with-dashes/test__unit_test_1.py"))
        test_results.add(self.create_test_invocation("/path/with.dots/test__unit_test_2.py"))

        result = test_results.file_to_no_of_tests([])

        assert len(result) == 3
        assert result[Path("/path/with spaces/test__unit_test_0.py")] == 1
        assert result[Path("/path/with-dashes/test__unit_test_1.py")] == 1
        assert result[Path("/path/with.dots/test__unit_test_2.py")] == 1

    def test_return_type_is_counter(self):
        """Verify that the return type is always a Counter object."""
        test_results = TestResults()

        # Test with no data
        result = test_results.file_to_no_of_tests([])
        assert isinstance(result, Counter)

        # Test with data
        test_results.add(self.create_test_invocation("/path/to/test__unit_test_0.py"))
        result = test_results.file_to_no_of_tests([])
        assert isinstance(result, Counter)

        # Counter specific operations should work
        result.update({Path("/another/path"): 5})
        assert result[Path("/another/path")] == 5