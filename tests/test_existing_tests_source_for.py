"""Tests for the existing_tests_source_for function in result/create_pr.py."""

from pathlib import Path
from unittest.mock import patch

import pytest
from codeflash.models.models import (
    CodePosition,
    FunctionCalledInTest,
    FunctionTestInvocation,
    InvocationId,
    TestResults,
    TestsInFile,
    TestType, VerificationType,
)
from codeflash.result.create_pr import existing_tests_source_for


@pytest.fixture
def sample_tests_root(tmp_path: Path) -> Path:
    """Create a temporary test root directory."""
    return tmp_path / "tests"


@pytest.fixture
def sample_function_to_tests(sample_tests_root: Path) -> dict[str, set[FunctionCalledInTest]]:
    """Create sample function to tests mapping."""
    test_file_1 = sample_tests_root / "test_module1.py"
    test_file_2 = sample_tests_root / "test_module2.py"

    return {
        "my_module.my_function": {
            FunctionCalledInTest(
                tests_in_file=TestsInFile(
                    test_file=test_file_1,
                    test_class=None,
                    test_function="test_basic_functionality",
                    test_type=TestType.EXISTING_UNIT_TEST,
                ),
                position=CodePosition(line_no=10, col_no=4),
            ),
            FunctionCalledInTest(
                tests_in_file=TestsInFile(
                    test_file=test_file_1,
                    test_class="TestMyFunction",
                    test_function="test_edge_cases",
                    test_type=TestType.EXISTING_UNIT_TEST,
                ),
                position=CodePosition(line_no=20, col_no=8),
            ),
            FunctionCalledInTest(
                tests_in_file=TestsInFile(
                    test_file=test_file_2,
                    test_class=None,
                    test_function="test_performance",
                    test_type=TestType.EXISTING_UNIT_TEST,
                ),
                position=CodePosition(line_no=15, col_no=4),
            ),
        }
    }


@pytest.fixture
def sample_original_test_results() -> TestResults:
    """Create sample original test results with timing information."""
    results = TestResults()

    # Test case 1: test_basic_functionality with multiple function calls
    results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name=None,
                test_function_name="test_basic_functionality",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=1000,  # 1000 ns
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name=None,
                test_function_name="test_basic_functionality",
                function_getting_tested="my_function",
                iteration_id="2",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=500,  # 500 ns
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    # Test case 2: test_edge_cases
    results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name="TestMyFunction",
                test_function_name="test_edge_cases",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=2000,  # 2000 ns
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    # Test case 3: test_performance
    results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module2",
                test_class_name=None,
                test_function_name="test_performance",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module2.py"),
            did_pass=True,
            runtime=3000,  # 3000 ns
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    return results


@pytest.fixture
def sample_optimized_test_results() -> TestResults:
    """Create sample optimized test results with improved timing information."""
    results = TestResults()

    # Test case 1: test_basic_functionality with multiple function calls (improved)
    results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name=None,
                test_function_name="test_basic_functionality",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=800,  # 800 ns (improved from 1000 ns)
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name=None,
                test_function_name="test_basic_functionality",
                function_getting_tested="my_function",
                iteration_id="2",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=400,  # 400 ns (improved from 500 ns)
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    # Test case 2: test_edge_cases (improved)
    results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name="TestMyFunction",
                test_function_name="test_edge_cases",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=1500,  # 1500 ns (improved from 2000 ns)
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    # Test case 3: test_performance (improved)
    results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module2",
                test_class_name=None,
                test_function_name="test_performance",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module2.py"),
            did_pass=True,
            runtime=2100,  # 2100 ns (improved from 3000 ns)
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    return results


def test_existing_tests_source_for_without_timing_info(
    sample_function_to_tests: dict[str, set[FunctionCalledInTest]], sample_tests_root: Path
):
    """Test the function works without timing information (backward compatibility)."""
    result = existing_tests_source_for("my_module.my_function", sample_function_to_tests, sample_tests_root)

    expected_lines = ["- test_module1.py", "- test_module2.py"]

    for line in expected_lines:
        assert line in result

    # Should not contain any timing information
    assert "->" not in result
    assert "ns" not in result


def test_existing_tests_source_for_with_timing_info(
    sample_function_to_tests: dict[str, set[FunctionCalledInTest]],
    sample_tests_root: Path,
    sample_original_test_results: TestResults,
    sample_optimized_test_results: TestResults,
):
    """Test the function includes timing information when provided."""
    with patch("codeflash.code_utils.time_utils.format_time") as mock_format_time:
        # Mock format_time to return predictable values
        mock_format_time.side_effect = lambda x: f"{x} ns"

        result = existing_tests_source_for(
            "my_module.my_function",
            sample_function_to_tests,
            sample_tests_root,
            sample_original_test_results,
            sample_optimized_test_results,
        )

    # Should contain file names
    assert "- test_module1.py" in result
    assert "- test_module2.py" in result

    # Should contain test function names with timing (using min values now)
    assert "test_basic_functionality: 500 ns -> 400 ns" in result  # min(1000,500) -> min(800,400)
    assert "test_edge_cases: 2000 ns -> 1500 ns" in result
    assert "test_performance: 3000 ns -> 2100 ns" in result


def test_existing_tests_source_for_aggregates_multiple_function_calls(
    sample_function_to_tests: dict[str, set[FunctionCalledInTest]],
    sample_tests_root: Path,
    sample_original_test_results: TestResults,
    sample_optimized_test_results: TestResults,
):
    """Test that multiple function calls within a test case use minimum timing."""
    with patch("codeflash.code_utils.time_utils.format_time") as mock_format_time:
        mock_format_time.side_effect = lambda x: f"{x} ns"

        result = existing_tests_source_for(
            "my_module.my_function",
            sample_function_to_tests,
            sample_tests_root,
            sample_original_test_results,
            sample_optimized_test_results,
        )

    # test_basic_functionality should show minimum timing: min(1000,500) -> min(800,400)
    assert "test_basic_functionality: 500 ns -> 400 ns" in result


def test_existing_tests_source_for_only_includes_passing_tests(
    sample_function_to_tests: dict[str, set[FunctionCalledInTest]], sample_tests_root: Path
):
    """Test that only passing tests with runtime data are included in timing report."""
    original_results = TestResults()
    optimized_results = TestResults()

    # Add a passing test with runtime
    original_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name=None,
                test_function_name="test_basic_functionality",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=1000,
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    optimized_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name=None,
                test_function_name="test_basic_functionality",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=800,
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    # Add a failing test (should be excluded)
    original_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name="TestMyFunction",
                test_function_name="test_edge_cases",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=False,  # Failing test
            runtime=2000,
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    # Add a test without runtime (should be excluded)
    original_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module2",
                test_class_name=None,
                test_function_name="test_performance",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module2.py"),
            did_pass=True,
            runtime=None,  # No runtime data
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    with patch("codeflash.code_utils.time_utils.format_time") as mock_format_time:
        mock_format_time.side_effect = lambda x: f"{x} ns"

        result = existing_tests_source_for(
            "my_module.my_function", sample_function_to_tests, sample_tests_root, original_results, optimized_results
        )

    # Should only include the passing test with runtime data
    assert "test_basic_functionality: 1000 ns -> 800 ns" in result
    # Should not include failing test or test without runtime
    assert "test_edge_cases" not in result
    assert "test_performance" not in result


def test_existing_tests_source_for_with_empty_test_mapping(sample_tests_root: Path):
    """Test behavior when there are no tests for the function."""
    result = existing_tests_source_for("nonexistent.function", {}, sample_tests_root)

    assert result == ""


def test_existing_tests_source_for_missing_optimized_results(
    sample_function_to_tests: dict[str, set[FunctionCalledInTest]],
    sample_tests_root: Path,
    sample_original_test_results: TestResults,
):
    """Test behavior when optimized results are missing for some test cases."""
    # Create optimized results that are missing some test cases
    optimized_results = TestResults()
    optimized_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.test_module1",
                test_class_name=None,
                test_function_name="test_basic_functionality",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/test_module1.py"),
            did_pass=True,
            runtime=800,
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )
    # Note: Missing test_edge_cases and test_performance optimized results

    with patch("codeflash.code_utils.time_utils.format_time") as mock_format_time:
        mock_format_time.side_effect = lambda x: f"{x} ns"

        result = existing_tests_source_for(
            "my_module.my_function",
            sample_function_to_tests,
            sample_tests_root,
            sample_original_test_results,
            optimized_results,
        )

    # Should not include test cases without both original and optimized results
    assert "test_basic_functionality" not in result  # Missing second function call
    assert "test_edge_cases" not in result
    assert "test_performance" not in result

    # Should still show file names
    assert "- test_module1.py" in result
    assert "- test_module2.py" in result


def test_existing_tests_source_for_sorted_output(sample_tests_root: Path):
    """Test that output is properly sorted by file name and test function name."""
    # Create a more complex test mapping with multiple files and functions
    test_file_a = sample_tests_root / "a_test_module.py"
    test_file_z = sample_tests_root / "z_test_module.py"

    function_to_tests = {
        "my_module.my_function": {
            FunctionCalledInTest(
                tests_in_file=TestsInFile(
                    test_file=test_file_z,
                    test_class=None,
                    test_function="z_test_function",
                    test_type=TestType.EXISTING_UNIT_TEST,
                ),
                position=CodePosition(line_no=10, col_no=4),
            ),
            FunctionCalledInTest(
                tests_in_file=TestsInFile(
                    test_file=test_file_a,
                    test_class=None,
                    test_function="a_test_function",
                    test_type=TestType.EXISTING_UNIT_TEST,
                ),
                position=CodePosition(line_no=20, col_no=8),
            ),
            FunctionCalledInTest(
                tests_in_file=TestsInFile(
                    test_file=test_file_a,
                    test_class=None,
                    test_function="b_test_function",
                    test_type=TestType.EXISTING_UNIT_TEST,
                ),
                position=CodePosition(line_no=30, col_no=8),
            ),
        }
    }

    original_results = TestResults()
    optimized_results = TestResults()

    # Add test results for all functions
    for test_func in ["a_test_function", "b_test_function"]:
        original_results.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="tests.a_test_module",
                    test_class_name=None,
                    test_function_name=test_func,
                    function_getting_tested="my_function",
                    iteration_id="1",
                ),
                file_name=Path("/tmp/tests/a_test_module.py"),
                did_pass=True,
                runtime=1000,
                test_framework="pytest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=None,
                timed_out=False,
                loop_index=1,
            )
        )

        optimized_results.add(
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="tests.a_test_module",
                    test_class_name=None,
                    test_function_name=test_func,
                    function_getting_tested="my_function",
                    iteration_id="1",
                ),
                file_name=Path("/tmp/tests/a_test_module.py"),
                did_pass=True,
                runtime=800,
                test_framework="pytest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=None,
                timed_out=False,
                loop_index=1,
            )
        )

    original_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.z_test_module",
                test_class_name=None,
                test_function_name="z_test_function",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/z_test_module.py"),
            did_pass=True,
            runtime=1000,
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    optimized_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="tests.z_test_module",
                test_class_name=None,
                test_function_name="z_test_function",
                function_getting_tested="my_function",
                iteration_id="1",
            ),
            file_name=Path("/tmp/tests/z_test_module.py"),
            did_pass=True,
            runtime=800,
            test_framework="pytest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
            loop_index=1,
        )
    )

    with patch("codeflash.code_utils.time_utils.format_time") as mock_format_time:
        mock_format_time.side_effect = lambda x: f"{x} ns"

        result = existing_tests_source_for(
            "my_module.my_function", function_to_tests, sample_tests_root, original_results, optimized_results
        )

    lines = result.split("\n")

    # Files should be sorted alphabetically
    a_file_index = next(i for i, line in enumerate(lines) if "a_test_module.py" in line)
    z_file_index = next(i for i, line in enumerate(lines) if "z_test_module.py" in line)
    assert a_file_index < z_file_index

    # Test functions within a file should be sorted alphabetically
    a_func_index = next(i for i, line in enumerate(lines) if "a_test_function" in line)
    b_func_index = next(i for i, line in enumerate(lines) if "b_test_function" in line)
    assert a_func_index < b_func_index



def test_existing_tests_source_for_format_time_called_correctly(
    sample_function_to_tests: dict[str, set[FunctionCalledInTest]],
    sample_tests_root: Path,
    sample_original_test_results: TestResults,
    sample_optimized_test_results: TestResults,
):
    """Test that format_time is called with correct values (min of runtime lists)."""
    with patch("codeflash.code_utils.time_utils.format_time") as mock_format_time:
        mock_format_time.side_effect = lambda x: f"{x} ns"

        existing_tests_source_for(
            "my_module.my_function",
            sample_function_to_tests,
            sample_tests_root,
            sample_original_test_results,
            sample_optimized_test_results,
        )

        # Check that format_time was called with the minimum values
        call_args = [call[0][0] for call in mock_format_time.call_args_list]

        # Should include minimum values (not aggregated)
        assert 500 in call_args  # test_basic_functionality original: min(1000, 500)
        assert 400 in call_args  # test_basic_functionality optimized: min(800, 400)
        assert 2000 in call_args  # test_edge_cases original
        assert 1500 in call_args  # test_edge_cases optimized
        assert 3000 in call_args  # test_performance original
        assert 2100 in call_args  # test_performance optimized