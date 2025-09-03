from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

from codeflash.code_utils.instrument_existing_tests import instrument_source_module_with_async_decorators
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent, TestFile, TestFiles, TestingMode, TestType
from codeflash.optimization.optimizer import Optimizer
from codeflash.verification.instrument_codeflash_capture import instrument_codeflash_capture


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
        source_success, instrumented_source = instrument_source_module_with_async_decorators(
            fto_path, func, TestingMode.BEHAVIOR
        )

        assert source_success
        assert instrumented_source is not None
        assert '''import asyncio\nfrom typing import List, Union\n\nfrom codeflash.code_utils.codeflash_wrap_decorator import \\\n    codeflash_behavior_async\n\n\n@codeflash_behavior_async\nasync def async_sorter(lst: List[Union[int, float]]) -> List[Union[int, float]]:\n    """\n    Async bubble sort implementation for testing.\n    """\n    print("codeflash stdout: Async sorting list")\n    \n    # Add some async delay to simulate async work\n    await asyncio.sleep(0.01)\n    \n    n = len(lst)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if lst[j] > lst[j + 1]:\n                lst[j], lst[j + 1] = lst[j + 1], lst[j]\n    \n    result = lst.copy()\n    print(f"result: {result}")\n    return result\n\n\nclass AsyncBubbleSorter:\n    """Class with async sorting method for testing."""\n    \n    async def sorter(self, lst: List[Union[int, float]]) -> List[Union[int, float]]:\n        """\n        Async bubble sort implementation within a class.\n        """\n        print("codeflash stdout: AsyncBubbleSorter.sorter() called")\n        \n        # Add some async delay\n        await asyncio.sleep(0.005)\n        \n        n = len(lst)\n        for i in range(n):\n            for j in range(0, n - i - 1):\n                if lst[j] > lst[j + 1]:\n                    lst[j], lst[j + 1] = lst[j + 1], lst[j]\n        \n        result = lst.copy()\n        return result\n''' in instrumented_source

        # Write the instrumented source back
        fto_path.write_text(instrumented_source, "utf-8")

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
        assert results_list[0].id.test_class_name == "PytestPluginManager" 
        assert results_list[0].id.test_function_name == "test_async_sort"
        assert results_list[0].did_pass
        assert results_list[0].runtime is None or results_list[0].runtime >= 0

        expected_stdout = "codeflash stdout: Async sorting list\nresult: [0, 1, 2, 3, 4, 5]\n"
        assert expected_stdout == results_list[0].stdout


        if len(results_list) > 1:
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

        source_success, instrumented_source = instrument_source_module_with_async_decorators(
            fto_path, func, TestingMode.BEHAVIOR
        )

        assert source_success
        assert instrumented_source is not None
        assert "@codeflash_behavior_async" in instrumented_source

        fto_path.write_text(instrumented_source, "utf-8")

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
        assert len(results_list) == 2, f"Expected 2 results but got {len(results_list)}: {[r.id.function_getting_tested for r in results_list]}"

        init_result = results_list[0]
        sorter_result = results_list[1]


        assert sorter_result.id.function_getting_tested == "sorter"
        assert sorter_result.id.test_class_name == "PytestPluginManager"
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
        source_success, instrumented_source = instrument_source_module_with_async_decorators(
            fto_path, func, TestingMode.PERFORMANCE
        )

        assert source_success
        assert instrumented_source is not None
        assert '''import asyncio\nfrom typing import List, Union\n\nfrom codeflash.code_utils.codeflash_wrap_decorator import \\\n    codeflash_performance_async\n\n\n@codeflash_performance_async\nasync def async_sorter(lst: List[Union[int, float]]) -> List[Union[int, float]]:\n    """\n    Async bubble sort implementation for testing.\n    """\n    print("codeflash stdout: Async sorting list")\n    \n    # Add some async delay to simulate async work\n    await asyncio.sleep(0.01)\n    \n    n = len(lst)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if lst[j] > lst[j + 1]:\n                lst[j], lst[j + 1] = lst[j + 1], lst[j]\n    \n    result = lst.copy()\n    print(f"result: {result}")\n    return result\n\n\nclass AsyncBubbleSorter:\n    """Class with async sorting method for testing."""\n    \n    async def sorter(self, lst: List[Union[int, float]]) -> List[Union[int, float]]:\n        """\n        Async bubble sort implementation within a class.\n        """\n        print("codeflash stdout: AsyncBubbleSorter.sorter() called")\n        \n        # Add some async delay\n        await asyncio.sleep(0.005)\n        \n        n = len(lst)\n        for i in range(n):\n            for j in range(0, n - i - 1):\n                if lst[j] > lst[j + 1]:\n                    lst[j], lst[j + 1] = lst[j + 1], lst[j]\n        \n        result = lst.copy()\n        return result\n''' == instrumented_source

        fto_path.write_text(instrumented_source, "utf-8")

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
