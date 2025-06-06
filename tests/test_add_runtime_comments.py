"""Tests for the add_runtime_comments_to_generated_tests functionality."""

from pathlib import Path

from codeflash.code_utils.edit_generated_tests import add_runtime_comments_to_generated_tests
from codeflash.models.models import (
    FunctionTestInvocation,
    GeneratedTests,
    GeneratedTestsList,
    InvocationId,
    TestResults,
    TestType,
    VerificationType,
)


class TestAddRuntimeComments:
    """Test cases for add_runtime_comments_to_generated_tests method."""

    def create_test_invocation(
        self, test_function_name: str, runtime: int, loop_index: int = 1, iteration_id: str = "1", did_pass: bool = True
    ) -> FunctionTestInvocation:
        """Helper to create test invocation objects."""
        return FunctionTestInvocation(
            loop_index=loop_index,
            id=InvocationId(
                test_module_path="test_module",
                test_class_name=None,
                test_function_name=test_function_name,
                function_getting_tested="test_function",
                iteration_id=iteration_id,
            ),
            file_name=Path("test.py"),
            did_pass=did_pass,
            runtime=runtime,
            test_framework="pytest",
            test_type=TestType.GENERATED_REGRESSION,
            return_value=None,
            timed_out=False,
            verification_type=VerificationType.FUNCTION_CALL,
        )

    def test_basic_runtime_comment_addition(self):
        """Test basic functionality of adding runtime comments."""
        # Create test source code
        test_source = """def test_bubble_sort():
    codeflash_output = bubble_sort([3, 1, 2])
    assert codeflash_output == [1, 2, 3]
"""

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior.py"),
            perf_file_path=Path("test_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create test results
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        # Add test invocations with different runtimes
        original_invocation = self.create_test_invocation("test_bubble_sort", 500_000)  # 500μs
        optimized_invocation = self.create_test_invocation("test_bubble_sort", 300_000)  # 300μs

        original_test_results.add(original_invocation)
        optimized_test_results.add(optimized_invocation)

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that comments were added
        modified_source = result.generated_tests[0].generated_original_test_source
        assert "# 500μs -> 300μs" in modified_source
        assert "codeflash_output = bubble_sort([3, 1, 2]) # 500μs -> 300μs" in modified_source

    def test_multiple_test_functions(self):
        """Test handling multiple test functions in the same file."""
        test_source = """def test_bubble_sort():
    codeflash_output = bubble_sort([3, 1, 2])
    assert codeflash_output == [1, 2, 3]

def test_quick_sort():
    codeflash_output = quick_sort([5, 2, 8])
    assert codeflash_output == [2, 5, 8]

def helper_function():
    return "not a test"
"""

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior.py"),
            perf_file_path=Path("test_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create test results for both functions
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        # Add test invocations for both test functions
        original_test_results.add(self.create_test_invocation("test_bubble_sort", 500_000))
        original_test_results.add(self.create_test_invocation("test_quick_sort", 800_000))

        optimized_test_results.add(self.create_test_invocation("test_bubble_sort", 300_000))
        optimized_test_results.add(self.create_test_invocation("test_quick_sort", 600_000))

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        modified_source = result.generated_tests[0].generated_original_test_source

        # Check that comments were added to both test functions
        assert "# 500μs -> 300μs" in modified_source
        assert "# 800μs -> 600μs" in modified_source
        # Helper function should not have comments
        assert (
            "helper_function():" in modified_source
            and "# " not in modified_source.split("helper_function():")[1].split("\n")[0]
        )

    def test_different_time_formats(self):
        """Test that different time ranges are formatted correctly with new precision rules."""
        test_cases = [
            (999, 500, "999ns -> 500ns"),  # nanoseconds
            (25_000, 18_000, "25.0μs -> 18.0μs"),  # microseconds with precision
            (500_000, 300_000, "500μs -> 300μs"),  # microseconds full integers
            (1_500_000, 800_000, "1.50ms -> 800μs"),  # milliseconds with precision
            (365_000_000, 290_000_000, "365ms -> 290ms"),  # milliseconds full integers
            (2_000_000_000, 1_500_000_000, "2.00s -> 1.50s"),  # seconds with precision
        ]

        for original_time, optimized_time, expected_comment in test_cases:
            test_source = """def test_function():
    codeflash_output = some_function()
    assert codeflash_output is not None
"""

            generated_test = GeneratedTests(
                generated_original_test_source=test_source,
                instrumented_behavior_test_source="",
                instrumented_perf_test_source="",
                behavior_file_path=Path("test_behavior.py"),
                perf_file_path=Path("test_perf.py"),
            )

            generated_tests = GeneratedTestsList(generated_tests=[generated_test])

            # Create test results
            original_test_results = TestResults()
            optimized_test_results = TestResults()

            original_test_results.add(self.create_test_invocation("test_function", original_time))
            optimized_test_results.add(self.create_test_invocation("test_function", optimized_time))

            # Test the functionality
            result = add_runtime_comments_to_generated_tests(
                generated_tests, original_test_results, optimized_test_results
            )

            modified_source = result.generated_tests[0].generated_original_test_source
            assert f"# {expected_comment}" in modified_source

    def test_missing_test_results(self):
        """Test behavior when test results are missing for a test function."""
        test_source = """def test_bubble_sort():
    codeflash_output = bubble_sort([3, 1, 2])
    assert codeflash_output == [1, 2, 3]
"""

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior.py"),
            perf_file_path=Path("test_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create empty test results
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that no comments were added
        modified_source = result.generated_tests[0].generated_original_test_source
        assert modified_source == test_source  # Should be unchanged

    def test_partial_test_results(self):
        """Test behavior when only one set of test results is available."""
        test_source = """def test_bubble_sort():
    codeflash_output = bubble_sort([3, 1, 2])
    assert codeflash_output == [1, 2, 3]
"""

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior.py"),
            perf_file_path=Path("test_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create test results with only original data
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        original_test_results.add(self.create_test_invocation("test_bubble_sort", 500_000))
        # No optimized results

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that no comments were added
        modified_source = result.generated_tests[0].generated_original_test_source
        assert modified_source == test_source  # Should be unchanged

    def test_multiple_runtimes_uses_minimum(self):
        """Test that when multiple runtimes exist, the minimum is used."""
        test_source = """def test_bubble_sort():
    codeflash_output = bubble_sort([3, 1, 2])
    assert codeflash_output == [1, 2, 3]
"""

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior.py"),
            perf_file_path=Path("test_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create test results with multiple loop iterations
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        # Add multiple runs with different runtimes
        original_test_results.add(self.create_test_invocation("test_bubble_sort", 600_000, loop_index=1))
        original_test_results.add(self.create_test_invocation("test_bubble_sort", 500_000, loop_index=2))
        original_test_results.add(self.create_test_invocation("test_bubble_sort", 550_000, loop_index=3))

        optimized_test_results.add(self.create_test_invocation("test_bubble_sort", 350_000, loop_index=1))
        optimized_test_results.add(self.create_test_invocation("test_bubble_sort", 300_000, loop_index=2))
        optimized_test_results.add(self.create_test_invocation("test_bubble_sort", 320_000, loop_index=3))

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that minimum times were used (500μs -> 300μs)
        modified_source = result.generated_tests[0].generated_original_test_source
        assert "# 500μs -> 300μs" in modified_source

    def test_no_codeflash_output_assignment(self):
        """Test behavior when test doesn't have codeflash_output assignment."""
        test_source = """def test_bubble_sort():
    result = bubble_sort([3, 1, 2])
    assert result == [1, 2, 3]
"""

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior.py"),
            perf_file_path=Path("test_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create test results
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        original_test_results.add(self.create_test_invocation("test_bubble_sort", 500_000))
        optimized_test_results.add(self.create_test_invocation("test_bubble_sort", 300_000))

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that no comments were added (no codeflash_output assignment)
        modified_source = result.generated_tests[0].generated_original_test_source
        assert modified_source == test_source  # Should be unchanged

    def test_invalid_python_code_handling(self):
        """Test behavior when test source code is invalid Python."""
        test_source = """def test_bubble_sort(:
    codeflash_output = bubble_sort([3, 1, 2])
    assert codeflash_output == [1, 2, 3]
"""  # Invalid syntax: extra colon

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior.py"),
            perf_file_path=Path("test_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create test results
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        original_test_results.add(self.create_test_invocation("test_bubble_sort", 500_000))
        optimized_test_results.add(self.create_test_invocation("test_bubble_sort", 300_000))

        # Test the functionality - should handle parse error gracefully
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that original test is preserved when parsing fails
        modified_source = result.generated_tests[0].generated_original_test_source
        assert modified_source == test_source  # Should be unchanged due to parse error

    def test_multiple_generated_tests(self):
        """Test handling multiple generated test objects."""
        test_source_1 = """def test_bubble_sort():
    codeflash_output = bubble_sort([3, 1, 2])
    assert codeflash_output == [1, 2, 3]
"""

        test_source_2 = """def test_quick_sort():
    codeflash_output = quick_sort([5, 2, 8])
    assert codeflash_output == [2, 5, 8]
"""

        generated_test_1 = GeneratedTests(
            generated_original_test_source=test_source_1,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior_1.py"),
            perf_file_path=Path("test_perf_1.py"),
        )

        generated_test_2 = GeneratedTests(
            generated_original_test_source=test_source_2,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior_2.py"),
            perf_file_path=Path("test_perf_2.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test_1, generated_test_2])

        # Create test results
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        original_test_results.add(self.create_test_invocation("test_bubble_sort", 500_000))
        original_test_results.add(self.create_test_invocation("test_quick_sort", 800_000))

        optimized_test_results.add(self.create_test_invocation("test_bubble_sort", 300_000))
        optimized_test_results.add(self.create_test_invocation("test_quick_sort", 600_000))

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that comments were added to both test files
        modified_source_1 = result.generated_tests[0].generated_original_test_source
        modified_source_2 = result.generated_tests[1].generated_original_test_source

        assert "# 500μs -> 300μs" in modified_source_1
        assert "# 800μs -> 600μs" in modified_source_2

    def test_preserved_test_attributes(self):
        """Test that other test attributes are preserved during modification."""
        test_source = """def test_bubble_sort():
    codeflash_output = bubble_sort([3, 1, 2])
    assert codeflash_output == [1, 2, 3]
"""

        original_behavior_source = "behavior test source"
        original_perf_source = "perf test source"
        original_behavior_path = Path("test_behavior.py")
        original_perf_path = Path("test_perf.py")

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source=original_behavior_source,
            instrumented_perf_test_source=original_perf_source,
            behavior_file_path=original_behavior_path,
            perf_file_path=original_perf_path,
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create test results
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        original_test_results.add(self.create_test_invocation("test_bubble_sort", 500_000))
        optimized_test_results.add(self.create_test_invocation("test_bubble_sort", 300_000))

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that other attributes are preserved
        modified_test = result.generated_tests[0]
        assert modified_test.instrumented_behavior_test_source == original_behavior_source
        assert modified_test.instrumented_perf_test_source == original_perf_source
        assert modified_test.behavior_file_path == original_behavior_path
        assert modified_test.perf_file_path == original_perf_path

        # Check that only the generated_original_test_source was modified
        assert "# 500μs -> 300μs" in modified_test.generated_original_test_source

    def test_multistatement_line_handling(self):
        """Test that runtime comments work correctly with multiple statements on one line."""
        test_source = """def test_mutation_of_input():
    # Test that the input list is mutated in-place and returned
    arr = [3, 1, 2]
    codeflash_output = sorter(arr); result = codeflash_output
    assert result == [1, 2, 3]
    assert arr == [1, 2, 3]  # Input should be mutated
"""

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("test_behavior.py"),
            perf_file_path=Path("test_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Create test results
        original_test_results = TestResults()
        optimized_test_results = TestResults()

        original_test_results.add(self.create_test_invocation("test_mutation_of_input", 19_000))  # 19μs
        optimized_test_results.add(self.create_test_invocation("test_mutation_of_input", 14_000))  # 14μs

        # Test the functionality
        result = add_runtime_comments_to_generated_tests(generated_tests, original_test_results, optimized_test_results)

        # Check that comments were added to the correct line
        modified_source = result.generated_tests[0].generated_original_test_source
        assert "# 19.0μs -> 14.0μs" in modified_source

        # Verify the comment is on the line with codeflash_output assignment
        lines = modified_source.split("\n")
        codeflash_line = None
        for line in lines:
            if "codeflash_output = sorter(arr)" in line:
                codeflash_line = line
                break

        assert codeflash_line is not None, "Could not find codeflash_output assignment line"
        assert "# 19.0μs -> 14.0μs" in codeflash_line, f"Comment not found in the correct line: {codeflash_line}"
