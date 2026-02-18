from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from codeflash.code_utils.instrument_existing_tests import (
    add_async_decorator_to_function,
    get_async_inline_code,
    inject_profiling_into_existing_test,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, FunctionParent, TestFile, TestFiles, TestingMode, TestType
from codeflash.optimization.optimizer import Optimizer
from codeflash.verification.instrument_codeflash_capture import instrument_codeflash_capture


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_bubble_sort_behavior_results() -> None:
    test_code = """import asyncio
import pytest
from code_to_optimize.async_bubble_sort import async_sorter


@pytest.mark.asyncio
async def test_async_sort():
    input = [5, 4, 3, 2, 1, 0]
    output = await async_sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]  
    output = await async_sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"""

    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_bubble_sort_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_bubble_sort_perf_temp.py"
    ).resolve()
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/async_bubble_sort.py").resolve()
    original_code = fto_path.read_text("utf-8")

    try:
        # Write test file
        with test_path.open("w") as f:
            f.write(test_code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()

        # Create async function to optimize
        func = FunctionToOptimize(function_name="async_sorter", parents=[], file_path=Path(fto_path), is_async=True)

        # For async functions, instrument the source module directly with decorators
        source_success = add_async_decorator_to_function(fto_path, func, TestingMode.BEHAVIOR)

        assert source_success

        # Verify the file was modified with exact expected output
        instrumented_source = fto_path.read_text("utf-8")
        from codeflash.code_utils.formatter import sort_imports

        inline_code = get_async_inline_code(TestingMode.BEHAVIOR)
        decorated_original = original_code.replace(
            "async def async_sorter", "@codeflash_behavior_async\nasync def async_sorter"
        )
        expected = sort_imports(code=inline_code + decorated_original, float_to_top=True)
        assert instrumented_source.strip() == expected.strip()

        # Add codeflash capture
        instrument_codeflash_capture(func, {}, tests_root)

        # Create optimizer
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
        test_env["CODEFLASH_TEST_MODULE"] = "code_to_optimize.tests.pytest.test_async_bubble_sort_temp"
        test_env["CODEFLASH_TEST_CLASS"] = ""
        test_env["CODEFLASH_TEST_FUNCTION"] = "test_async_sort"
        test_env["CODEFLASH_CURRENT_LINE_ID"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST

        # Create function optimizer and set up test files
        func_optimizer = opt.create_function_optimizer(func)
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

        assert test_results is not None
        assert test_results.test_results is not None

        results_list = test_results.test_results
        assert results_list[0].id.function_getting_tested == "async_sorter"
        assert results_list[0].id.test_class_name is None
        assert results_list[0].id.test_function_name == "test_async_sort"
        assert results_list[0].did_pass
        assert results_list[0].runtime is None or results_list[0].runtime >= 0

        expected_stdout = "codeflash stdout: Async sorting list\nresult: [0, 1, 2, 3, 4, 5]\n"
        assert expected_stdout == results_list[0].stdout

        assert results_list[1].id.function_getting_tested == "async_sorter"
        assert results_list[1].id.test_function_name == "test_async_sort"
        assert results_list[1].did_pass

        expected_stdout2 = "codeflash stdout: Async sorting list\nresult: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]\n"
        assert expected_stdout2 == results_list[1].stdout

    finally:
        # Restore original code
        fto_path.write_text(original_code, "utf-8")
        # Clean up test files
        if test_path.exists():
            test_path.unlink()
        if test_path_perf.exists():
            test_path_perf.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_class_method_behavior_results() -> None:
    """Test async class method behavior with run_and_parse_tests."""
    test_code = """import asyncio
import pytest
from code_to_optimize.async_bubble_sort import AsyncBubbleSorter


@pytest.mark.asyncio
async def test_async_class_sort():
    sorter = AsyncBubbleSorter()
    input = [3, 1, 4, 1, 5]
    output = await sorter.sorter(input)
    assert output == [1, 1, 3, 4, 5]"""

    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_class_bubble_sort_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_class_bubble_sort_perf_temp.py"
    ).resolve()
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/async_bubble_sort.py").resolve()
    original_code = fto_path.read_text("utf-8")

    try:
        with test_path.open("w") as f:
            f.write(test_code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()

        func = FunctionToOptimize(
            function_name="sorter",
            parents=[FunctionParent("AsyncBubbleSorter", "ClassDef")],
            file_path=Path(fto_path),
            is_async=True,
        )

        source_success = add_async_decorator_to_function(fto_path, func, TestingMode.BEHAVIOR)

        assert source_success

        # Verify the file was modified
        instrumented_source = fto_path.read_text("utf-8")
        assert "@codeflash_behavior_async" in instrumented_source

        instrument_codeflash_capture(func, {}, tests_root)

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
        test_env["CODEFLASH_TEST_MODULE"] = "code_to_optimize.tests.pytest.test_async_class_bubble_sort_temp"
        test_env["CODEFLASH_TEST_CLASS"] = ""
        test_env["CODEFLASH_TEST_FUNCTION"] = "test_async_class_sort"
        test_env["CODEFLASH_CURRENT_LINE_ID"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST

        func_optimizer = opt.create_function_optimizer(func)
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

        assert test_results is not None
        assert test_results.test_results is not None

        results_list = test_results.test_results
        assert len(results_list) == 2, (
            f"Expected 2 results but got {len(results_list)}: {[r.id.function_getting_tested for r in results_list]}"
        )

        init_result = results_list[0]
        sorter_result = results_list[1]

        assert sorter_result.id.function_getting_tested == "sorter"
        assert sorter_result.id.test_class_name is None
        assert sorter_result.id.test_function_name == "test_async_class_sort"
        assert sorter_result.did_pass
        assert sorter_result.runtime is None or sorter_result.runtime >= 0

        expected_stdout = "codeflash stdout: AsyncBubbleSorter.sorter() called\n"
        assert expected_stdout == sorter_result.stdout

        assert ".__init__" in init_result.id.function_getting_tested
        assert init_result.did_pass

    finally:
        fto_path.write_text(original_code, "utf-8")
        if test_path.exists():
            test_path.unlink()
        if test_path_perf.exists():
            test_path_perf.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_function_performance_mode() -> None:
    test_code = """import asyncio
import pytest
from code_to_optimize.async_bubble_sort import async_sorter


@pytest.mark.asyncio 
async def test_async_perf():
    input = [8, 7, 6, 5, 4, 3, 2, 1]
    output = await async_sorter(input)
    assert output == [1, 2, 3, 4, 5, 6, 7, 8]"""

    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_perf_temp.py").resolve()
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/async_bubble_sort.py").resolve()
    original_code = fto_path.read_text("utf-8")

    try:
        with test_path.open("w") as f:
            f.write(test_code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()

        # Create async function to optimize
        func = FunctionToOptimize(function_name="async_sorter", parents=[], file_path=Path(fto_path), is_async=True)

        # Instrument the source module with async performance decorators
        source_success = add_async_decorator_to_function(fto_path, func, TestingMode.PERFORMANCE)

        assert source_success

        # Verify the file was modified
        instrumented_source = fto_path.read_text("utf-8")
        from codeflash.code_utils.formatter import sort_imports

        inline_code = get_async_inline_code(TestingMode.PERFORMANCE)
        decorated_original = original_code.replace(
            "async def async_sorter", "@codeflash_performance_async\nasync def async_sorter"
        )
        expected = sort_imports(code=inline_code + decorated_original, float_to_top=True)
        assert instrumented_source.strip() == expected.strip()

        instrument_codeflash_capture(func, {}, tests_root)

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
        test_env["CODEFLASH_TEST_MODULE"] = "code_to_optimize.tests.pytest.test_async_perf_temp"
        test_env["CODEFLASH_TEST_CLASS"] = ""
        test_env["CODEFLASH_TEST_FUNCTION"] = "test_async_perf"
        test_env["CODEFLASH_CURRENT_LINE_ID"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST

        func_optimizer = opt.create_function_optimizer(func)
        func_optimizer.test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_path,
                    test_type=test_type,
                    original_file_path=test_path,
                    benchmarking_file_path=test_path,  # Same file for perf
                )
            ]
        )

        test_results, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert test_results is not None
        assert test_results.test_results is not None

    finally:
        # Restore original code
        fto_path.write_text(original_code, "utf-8")
        # Clean up test files
        if test_path.exists():
            test_path.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_function_error_handling() -> None:
    test_code = """import asyncio
import pytest
from code_to_optimize.async_bubble_sort import async_error_function


@pytest.mark.asyncio
async def test_async_error():
    with pytest.raises(ValueError, match="Test error"):
        await async_error_function([1, 2, 3])"""

    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_error_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_error_perf_temp.py"
    ).resolve()
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/async_bubble_sort.py").resolve()
    original_code = fto_path.read_text("utf-8")

    try:
        error_func_code = """

async def async_error_function(lst):
    \"\"\"Async function that raises an error for testing.\"\"\"
    await asyncio.sleep(0.001)  # Small delay
    raise ValueError("Test error")
"""

        modified_code = original_code + error_func_code
        fto_path.write_text(modified_code, "utf-8")

        with test_path.open("w") as f:
            f.write(test_code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()

        func = FunctionToOptimize(
            function_name="async_error_function", parents=[], file_path=Path(fto_path), is_async=True
        )

        source_success = add_async_decorator_to_function(fto_path, func, TestingMode.BEHAVIOR)

        assert source_success

        # Verify the file was modified
        instrumented_source = fto_path.read_text("utf-8")

        from codeflash.code_utils.formatter import sort_imports

        inline_code = get_async_inline_code(TestingMode.BEHAVIOR)
        decorated_modified = modified_code.replace(
            "async def async_error_function", "@codeflash_behavior_async\nasync def async_error_function"
        )
        expected = sort_imports(code=inline_code + decorated_modified, float_to_top=True)
        assert instrumented_source.strip() == expected.strip()
        instrument_codeflash_capture(func, {}, tests_root)

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
        test_env["CODEFLASH_TEST_MODULE"] = "code_to_optimize.tests.pytest.test_async_error_temp"
        test_env["CODEFLASH_TEST_CLASS"] = ""
        test_env["CODEFLASH_TEST_FUNCTION"] = "test_async_error"
        test_env["CODEFLASH_CURRENT_LINE_ID"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST

        func_optimizer = opt.create_function_optimizer(func)
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

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert test_results is not None
        assert test_results.test_results is not None
        assert len(test_results.test_results) >= 1

        result = test_results.test_results[0]
        assert result.id.function_getting_tested == "async_error_function"
        assert result.did_pass
        assert result.runtime is None or result.runtime >= 0

    finally:
        fto_path.write_text(original_code, "utf-8")
        if test_path.exists():
            test_path.unlink()
        if test_path_perf.exists():
            test_path_perf.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_multiple_iterations() -> None:
    test_code = """import asyncio
import pytest
from code_to_optimize.async_bubble_sort import async_sorter


@pytest.mark.asyncio
async def test_async_multi():
    input1 = [5, 4, 3]
    output1 = await async_sorter(input1)
    assert output1 == [3, 4, 5]
    
    input2 = [9, 7]
    output2 = await async_sorter(input2)
    assert output2 == [7, 9]"""

    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_multi_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_multi_perf_temp.py"
    ).resolve()
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/async_bubble_sort.py").resolve()
    original_code = fto_path.read_text("utf-8")

    try:
        with test_path.open("w") as f:
            f.write(test_code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()

        func = FunctionToOptimize(function_name="async_sorter", parents=[], file_path=Path(fto_path), is_async=True)

        source_success = add_async_decorator_to_function(fto_path, func, TestingMode.BEHAVIOR)

        assert source_success
        instrument_codeflash_capture(func, {}, tests_root)

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
        test_env["CODEFLASH_LOOP_INDEX"] = "3"
        test_env["CODEFLASH_TEST_MODULE"] = "code_to_optimize.tests.pytest.test_async_multi_temp"
        test_env["CODEFLASH_TEST_CLASS"] = ""
        test_env["CODEFLASH_TEST_FUNCTION"] = "test_async_multi"
        test_env["CODEFLASH_CURRENT_LINE_ID"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST

        func_optimizer = opt.create_function_optimizer(func)
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

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=2,
            pytest_max_loops=5,
            testing_time=0.2,
        )

        assert test_results is not None
        assert test_results.test_results is not None
        assert len(test_results.test_results) >= 2

        results_list = test_results.test_results
        function_calls = [r for r in results_list if r.id.function_getting_tested == "async_sorter"]
        assert len(function_calls) == 2

        first_call = function_calls[0]
        second_call = function_calls[1]

        assert first_call.stdout == "codeflash stdout: Async sorting list\nresult: [3, 4, 5]\n"
        assert second_call.stdout == "codeflash stdout: Async sorting list\nresult: [7, 9]\n"

        assert first_call.did_pass
        assert second_call.did_pass
        assert first_call.runtime is None or first_call.runtime >= 0
        assert second_call.runtime is None or second_call.runtime >= 0

    finally:
        fto_path.write_text(original_code, "utf-8")
        if test_path.exists():
            test_path.unlink()
        if test_path_perf.exists():
            test_path_perf.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_empty_input_edge_cases() -> None:
    test_code = """import asyncio
import pytest
from code_to_optimize.async_bubble_sort import async_sorter


@pytest.mark.asyncio
async def test_async_edge_cases():
    # Empty list
    empty = []
    result_empty = await async_sorter(empty)
    assert result_empty == []
    
    # Single item
    single = [42]
    result_single = await async_sorter(single)
    assert result_single == [42]
    
    # Already sorted
    sorted_list = [1, 2, 3, 4]
    result_sorted = await async_sorter(sorted_list)
    assert result_sorted == [1, 2, 3, 4]"""

    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_edge_temp.py").resolve()
    test_path_perf = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_async_edge_perf_temp.py"
    ).resolve()
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/async_bubble_sort.py").resolve()
    original_code = fto_path.read_text("utf-8")

    try:
        with test_path.open("w") as f:
            f.write(test_code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()

        func = FunctionToOptimize(function_name="async_sorter", parents=[], file_path=Path(fto_path), is_async=True)

        source_success = add_async_decorator_to_function(fto_path, func, TestingMode.BEHAVIOR)

        assert source_success
        instrument_codeflash_capture(func, {}, tests_root)

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
        test_env["CODEFLASH_TEST_MODULE"] = "code_to_optimize.tests.pytest.test_async_edge_temp"
        test_env["CODEFLASH_TEST_CLASS"] = ""
        test_env["CODEFLASH_TEST_FUNCTION"] = "test_async_edge_cases"
        test_env["CODEFLASH_CURRENT_LINE_ID"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST

        func_optimizer = opt.create_function_optimizer(func)
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

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert test_results is not None
        assert test_results.test_results is not None
        assert len(test_results.test_results) >= 3  # 3 function calls for edge cases

        results_list = test_results.test_results
        function_calls = [r for r in results_list if r.id.function_getting_tested == "async_sorter"]
        assert len(function_calls) == 3

        # Verify all calls passed
        for call in function_calls:
            assert call.did_pass
            assert call.runtime is None or call.runtime >= 0

        empty_call = function_calls[0]
        single_call = function_calls[1]
        sorted_call = function_calls[2]

        assert empty_call.stdout == "codeflash stdout: Async sorting list\nresult: []\n"
        assert single_call.stdout == "codeflash stdout: Async sorting list\nresult: [42]\n"
        assert sorted_call.stdout == "codeflash stdout: Async sorting list\nresult: [1, 2, 3, 4]\n"

    finally:
        fto_path.write_text(original_code, "utf-8")
        if test_path.exists():
            test_path.unlink()
        if test_path_perf.exists():
            test_path_perf.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_sync_function_behavior_in_async_test_environment() -> None:
    sync_sorter_code = """def sync_sorter(lst):
    \"\"\"Synchronous bubble sort for comparison.\"\"\"
    print("codeflash stdout: Sync sorting list")
    n = len(lst)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    result = lst.copy()
    print(f"result: {result}")
    return result
"""

    test_code = """from code_to_optimize.sync_bubble_sort import sync_sorter


def test_sync_sort():
    input = [5, 4, 3, 2, 1, 0]
    output = sync_sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]  
    output = sync_sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"""

    test_path = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_sync_in_async_temp.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_sync_in_async_perf_temp.py"
    ).resolve()
    sync_fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/sync_bubble_sort.py").resolve()

    try:
        with sync_fto_path.open("w") as f:
            f.write(sync_sorter_code)

        with test_path.open("w") as f:
            f.write(test_code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()

        func = FunctionToOptimize(
            function_name="sync_sorter", parents=[], file_path=Path(sync_fto_path), is_async=False
        )

        original_cwd = os.getcwd()
        run_cwd = project_root_path
        os.chdir(run_cwd)

        success, instrumented_test = inject_profiling_into_existing_test(
            test_path,
            [CodePosition(6, 13), CodePosition(10, 13)],  # Lines where sync_sorter is called
            func,
            project_root_path,
            mode=TestingMode.BEHAVIOR,
        )
        os.chdir(original_cwd)

        assert success
        assert instrumented_test is not None

        with test_path.open("w") as f:
            f.write(instrumented_test)

        instrument_codeflash_capture(func, {}, tests_root)

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
        test_env["CODEFLASH_TEST_MODULE"] = "code_to_optimize.tests.pytest.test_sync_in_async_temp"
        test_env["CODEFLASH_TEST_CLASS"] = ""
        test_env["CODEFLASH_TEST_FUNCTION"] = "test_sync_sort"
        test_env["CODEFLASH_CURRENT_LINE_ID"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST

        func_optimizer = opt.create_function_optimizer(func)
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

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert test_results is not None
        assert test_results.test_results is not None

        results_list = test_results.test_results
        assert results_list[0].id.function_getting_tested == "sync_sorter"
        assert results_list[0].id.iteration_id == "1_0"
        assert results_list[0].id.test_class_name is None
        assert results_list[0].id.test_function_name == "test_sync_sort"
        assert results_list[0].did_pass
        assert results_list[0].runtime > 0

        expected_stdout = "codeflash stdout: Sync sorting list\nresult: [0, 1, 2, 3, 4, 5]\n"
        assert expected_stdout == results_list[0].stdout

        if len(results_list) > 1:
            assert results_list[1].id.function_getting_tested == "sync_sorter"
            assert results_list[1].id.iteration_id == "4_0"
            assert results_list[1].id.test_function_name == "test_sync_sort"
            assert results_list[1].did_pass

            expected_stdout2 = "codeflash stdout: Sync sorting list\nresult: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]\n"
            assert expected_stdout2 == results_list[1].stdout

    finally:
        if sync_fto_path.exists():
            sync_fto_path.unlink()
        if test_path.exists():
            test_path.unlink()
        if test_path_perf.exists():
            test_path_perf.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_mixed_async_sync_function_calls() -> None:
    mixed_module_code = """import asyncio
from typing import List, Union


def sync_quick_sort(lst: List[Union[int, float]]) -> List[Union[int, float]]:
    \"\"\"Synchronous quick sort.\"\"\"
    print("codeflash stdout: Sync quick sort")
    if len(lst) <= 1:
        return lst.copy()
    pivot = lst[len(lst) // 2]
    left = [x for x in lst if x < pivot]
    middle = [x for x in lst if x == pivot]
    right = [x for x in lst if x > pivot]
    result = sync_quick_sort(left) + middle + sync_quick_sort(right)
    print(f"result: {result}")
    return result


async def async_merge_sort(lst: List[Union[int, float]]) -> List[Union[int, float]]:
    \"\"\"Asynchronous merge sort.\"\"\"
    print("codeflash stdout: Async merge sort")
    await asyncio.sleep(0.001)  # Small delay
    
    if len(lst) <= 1:
        return lst.copy()
    
    mid = len(lst) // 2
    left = await async_merge_sort(lst[:mid])
    right = await async_merge_sort(lst[mid:])
    
    # Merge
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    
    print(f"result: {result}")
    return result

"""

    test_code = """import asyncio
import pytest
from code_to_optimize.mixed_sort import sync_quick_sort, async_merge_sort


@pytest.mark.asyncio
async def test_mixed_sorting():
    # Test sync function
    sync_input = [3, 1, 4, 1, 5]
    sync_output = sync_quick_sort(sync_input)
    assert sync_output == [1, 1, 3, 4, 5]
    
    # Test async function
    async_input = [9, 2, 6, 5, 3]
    async_output = await async_merge_sort(async_input)
    assert async_output == [2, 3, 5, 6, 9]"""

    test_path = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_mixed_sort_temp.py").resolve()
    test_path_perf = (
        Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/test_mixed_sort_perf_temp.py"
    ).resolve()
    mixed_fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/mixed_sort.py").resolve()

    try:
        with mixed_fto_path.open("w") as f:
            f.write(mixed_module_code)

        with test_path.open("w") as f:
            f.write(test_code)

        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()

        async_func = FunctionToOptimize(
            function_name="async_merge_sort", parents=[], file_path=Path(mixed_fto_path), is_async=True
        )

        source_success = add_async_decorator_to_function(mixed_fto_path, async_func, TestingMode.BEHAVIOR)

        assert source_success

        # Verify the file was modified
        instrumented_source = mixed_fto_path.read_text("utf-8")
        assert "@codeflash_behavior_async" in instrumented_source
        assert "async def async_merge_sort" in instrumented_source
        assert "def sync_quick_sort" in instrumented_source  # Should preserve sync function
        instrument_codeflash_capture(async_func, {}, tests_root)

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
        test_env["CODEFLASH_TEST_MODULE"] = "code_to_optimize.tests.pytest.test_mixed_sort_temp"
        test_env["CODEFLASH_TEST_CLASS"] = ""
        test_env["CODEFLASH_TEST_FUNCTION"] = "test_mixed_sorting"
        test_env["CODEFLASH_CURRENT_LINE_ID"] = "0"
        test_type = TestType.EXISTING_UNIT_TEST

        func_optimizer = opt.create_function_optimizer(async_func)
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

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        assert test_results is not None
        assert test_results.test_results is not None

        results_list = test_results.test_results
        async_calls = [r for r in results_list if r.id.function_getting_tested == "async_merge_sort"]
        assert len(async_calls) >= 1

        for call in async_calls:
            assert call.did_pass
            assert call.runtime is None or call.runtime >= 0
            assert "codeflash stdout: Async merge sort" in call.stdout

    finally:
        if mixed_fto_path.exists():
            mixed_fto_path.unlink()
        if test_path.exists():
            test_path.unlink()
        if test_path_perf.exists():
            test_path_perf.unlink()
