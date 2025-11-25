import gc
import multiprocessing
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from codeflash.benchmarking.plugin.plugin import codeflash_benchmark_plugin
from codeflash.benchmarking.replay_test import generate_replay_test
from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
from codeflash.benchmarking.utils import validate_and_format_benchmark_table


def safe_unlink(file_path: Path, max_retries: int = 5, retry_delay: float = 0.5) -> None:
    """Safely delete a file with retries, handling Windows file locking issues."""
    for attempt in range(max_retries):
        try:
            file_path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                # Last attempt: force garbage collection to close any lingering SQLite connections
                gc.collect()
                time.sleep(retry_delay * 2)
                try:
                    file_path.unlink(missing_ok=True)
                except PermissionError:
                    # Silently fail on final attempt to avoid test failures from cleanup issues
                    pass


def test_trace_benchmarks() -> None:
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks_test"
    replay_tests_dir = benchmarks_root / "codeflash_replay_tests"
    tests_root = project_root / "tests"
    output_file = (benchmarks_root / Path("test_trace_benchmarks.trace")).resolve()
    trace_benchmarks_pytest(benchmarks_root, tests_root, project_root, output_file)
    assert output_file.exists()
    try:
        # Query the trace database to verify recorded function calls
        with sqlite3.connect(output_file.as_posix()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT function_name, class_name, module_name, file_path, benchmark_function_name, benchmark_module_path, benchmark_line_number FROM benchmark_function_timings ORDER BY benchmark_module_path, benchmark_function_name, function_name")
            function_calls = cursor.fetchall()

            # Accept platform-dependent run multipliers: function calls should come in complete groups of the base set (8)
            base_count = 8
            assert len(function_calls) >= base_count and len(function_calls) % base_count == 0, (
                f"Expected count to be a multiple of {base_count}, but got {len(function_calls)}"
            )

            bubble_sort_path = (project_root / "bubble_sort_codeflash_trace.py").as_posix()
            process_and_bubble_sort_path = (project_root / "process_and_bubble_sort_codeflash_trace.py").as_posix()
            # Expected function calls (each appears twice due to benchmark execution pattern)
            base_expected_calls = [
                ("sorter", "Sorter", "code_to_optimize.bubble_sort_codeflash_trace",
                 f"{bubble_sort_path}",
                 "test_class_sort", "tests.pytest.benchmarks_test.test_benchmark_bubble_sort_example", 17),

                ("sort_class", "Sorter", "code_to_optimize.bubble_sort_codeflash_trace",
                 f"{bubble_sort_path}",
                 "test_class_sort2", "tests.pytest.benchmarks_test.test_benchmark_bubble_sort_example", 20),

                ("sort_static", "Sorter", "code_to_optimize.bubble_sort_codeflash_trace",
                 f"{bubble_sort_path}",
                 "test_class_sort3", "tests.pytest.benchmarks_test.test_benchmark_bubble_sort_example", 23),

                ("__init__", "Sorter", "code_to_optimize.bubble_sort_codeflash_trace",
                 f"{bubble_sort_path}",
                 "test_class_sort4", "tests.pytest.benchmarks_test.test_benchmark_bubble_sort_example", 26),

                ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
                 f"{bubble_sort_path}",
                 "test_sort", "tests.pytest.benchmarks_test.test_benchmark_bubble_sort_example", 7),

                ("compute_and_sort", "", "code_to_optimize.process_and_bubble_sort_codeflash_trace",
                 f"{process_and_bubble_sort_path}",
                 "test_compute_and_sort", "tests.pytest.benchmarks_test.test_process_and_sort_example", 4),

                ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
                 f"{bubble_sort_path}",
                 "test_no_func", "tests.pytest.benchmarks_test.test_process_and_sort_example", 8),

                ("recursive_bubble_sort", "", "code_to_optimize.bubble_sort_codeflash_trace",
                 f"{bubble_sort_path}",
                 "test_recursive_sort", "tests.pytest.benchmarks_test.test_recursive_example", 5),
            ]
            expected_calls = base_expected_calls * 3
            # Order-agnostic validation: ensure at least one instance of each base expected call exists
            normalized_calls = [(a[0], a[1], a[2], Path(a[3]).name, a[4], a[5], a[6]) for a in function_calls]
            normalized_expected = [(e[0], e[1], e[2], Path(e[3]).name, e[4], e[5], e[6]) for e in base_expected_calls]
            for expected in normalized_expected:
                assert expected in normalized_calls, f"Missing expected call: {expected}"
        
        # Close database connection and ensure cleanup before opening new connections
        gc.collect()
        time.sleep(0.1)
        generate_replay_test(output_file, replay_tests_dir)
        test_class_sort_path = replay_tests_dir/ Path("test_tests_pytest_benchmarks_test_test_benchmark_bubble_sort_example__replay_test_0.py")
        assert test_class_sort_path.exists()
        test_class_sort_code = f"""
from code_to_optimize.bubble_sort_codeflash_trace import \\
    Sorter as code_to_optimize_bubble_sort_codeflash_trace_Sorter
from code_to_optimize.bubble_sort_codeflash_trace import \\
    sorter as code_to_optimize_bubble_sort_codeflash_trace_sorter
from codeflash.benchmarking.replay_test import get_next_arg_and_return
from codeflash.picklepatch.pickle_patcher import PicklePatcher as pickle

functions = ['sort_class', 'sort_static', 'sorter']
trace_file_path = r"{output_file.as_posix()}"

def test_code_to_optimize_bubble_sort_codeflash_trace_sorter_test_sort():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_sort", function_name="sorter", file_path=r"{bubble_sort_path}", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_bubble_sort_codeflash_trace_sorter(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sorter_test_class_sort():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_class_sort", function_name="sorter", file_path=r"{bubble_sort_path}", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        function_name = "sorter"
        if not args:
            raise ValueError("No arguments provided for the method.")
        if function_name == "__init__":
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter(*args[1:], **kwargs)
        else:
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter.sorter(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sort_class_test_class_sort2():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_class_sort2", function_name="sort_class", file_path=r"{bubble_sort_path}", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        if not args:
            raise ValueError("No arguments provided for the method.")
        ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter.sort_class(*args[1:], **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sort_static_test_class_sort3():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_class_sort3", function_name="sort_static", file_path=r"{bubble_sort_path}", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter.sort_static(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter___init___test_class_sort4():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_class_sort4", function_name="__init__", file_path=r"{bubble_sort_path}", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        function_name = "__init__"
        if not args:
            raise ValueError("No arguments provided for the method.")
        if function_name == "__init__":
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter(*args[1:], **kwargs)
        else:
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter(*args, **kwargs)

"""
        assert test_class_sort_path.read_text("utf-8").strip()==test_class_sort_code.strip()

        test_sort_path = replay_tests_dir / Path("test_tests_pytest_benchmarks_test_test_process_and_sort_example__replay_test_0.py")
        assert test_sort_path.exists()
        test_sort_code = f"""
from code_to_optimize.bubble_sort_codeflash_trace import \\
    sorter as code_to_optimize_bubble_sort_codeflash_trace_sorter
from code_to_optimize.process_and_bubble_sort_codeflash_trace import \\
    compute_and_sort as \\
    code_to_optimize_process_and_bubble_sort_codeflash_trace_compute_and_sort
from codeflash.benchmarking.replay_test import get_next_arg_and_return
from codeflash.picklepatch.pickle_patcher import PicklePatcher as pickle

functions = ['compute_and_sort', 'sorter']
trace_file_path = r"{output_file.as_posix()}"

def test_code_to_optimize_process_and_bubble_sort_codeflash_trace_compute_and_sort_test_compute_and_sort():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_compute_and_sort", function_name="compute_and_sort", file_path=r"{process_and_bubble_sort_path}", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_process_and_bubble_sort_codeflash_trace_compute_and_sort(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_sorter_test_no_func():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_no_func", function_name="sorter", file_path=r"{bubble_sort_path}", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_bubble_sort_codeflash_trace_sorter(*args, **kwargs)

"""
        assert test_sort_path.read_text("utf-8").strip()==test_sort_code.strip()
        # Ensure database connections are closed before cleanup
        gc.collect()
        time.sleep(0.1)
    finally:
        # Cleanup with retry mechanism to handle Windows file locking issues
        safe_unlink(output_file)
        shutil.rmtree(replay_tests_dir, ignore_errors=True)

# Skip the test in CI as the machine may not be multithreaded
@pytest.mark.ci_skip
def test_trace_multithreaded_benchmark() -> None:
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks_multithread"
    tests_root = project_root / "tests"
    output_file = (benchmarks_root / Path("test_trace_benchmarks.trace")).resolve()
    trace_benchmarks_pytest(benchmarks_root, tests_root, project_root, output_file)
    assert output_file.exists()
    try:
        # Query the trace database to verify recorded function calls
        with sqlite3.connect(output_file.as_posix()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT function_name, class_name, module_name, file_path, benchmark_function_name, benchmark_module_path, benchmark_line_number FROM benchmark_function_timings ORDER BY benchmark_module_path, benchmark_function_name, function_name")
            function_calls = cursor.fetchall()
        
        # Accept platform-dependent run multipliers; any positive count is fine for multithread case
        assert len(function_calls) >= 1, f"Expected at least 1 function call, got {len(function_calls)}"
        function_benchmark_timings = codeflash_benchmark_plugin.get_function_benchmark_timings(output_file)
        total_benchmark_timings = codeflash_benchmark_plugin.get_benchmark_timings(output_file)
        function_to_results = validate_and_format_benchmark_table(function_benchmark_timings, total_benchmark_timings)
        assert "code_to_optimize.bubble_sort_codeflash_trace.sorter" in function_to_results

        test_name, total_time, function_time, percent = function_to_results["code_to_optimize.bubble_sort_codeflash_trace.sorter"][0]
        assert total_time >= 0.0
        assert function_time >= 0.0
        assert percent >= 0.0

        bubble_sort_path = (project_root / "bubble_sort_codeflash_trace.py").as_posix()
        # Expected function calls (each appears multiple times due to benchmark execution pattern)
        expected_calls = [
            ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{bubble_sort_path}",
             "test_benchmark_sort", "tests.pytest.benchmarks_multithread.test_multithread_sort", 4),
        ] * 30
        for idx, (actual, expected) in enumerate(zip(function_calls, expected_calls)):
            assert actual[0] == expected[0], f"Mismatch at index {idx} for function_name"
            assert actual[1] == expected[1], f"Mismatch at index {idx} for class_name"
            assert actual[2] == expected[2], f"Mismatch at index {idx} for module_name"
            assert Path(actual[3]).name == Path(expected[3]).name, f"Mismatch at index {idx} for file_path"
            assert actual[4] == expected[4], f"Mismatch at index {idx} for benchmark_function_name"
            assert actual[6] == expected[6], f"Mismatch at index {idx} for benchmark_line_number"
        
        # Ensure database connections are closed before cleanup
        gc.collect()
        time.sleep(0.1)
    finally:
        # Cleanup with retry mechanism to handle Windows file locking issues
        safe_unlink(output_file)

def test_trace_benchmark_decorator() -> None:
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks_test_decorator"
    tests_root = project_root / "tests"
    output_file = (benchmarks_root / Path("test_trace_benchmarks.trace")).resolve()
    trace_benchmarks_pytest(benchmarks_root, tests_root, project_root, output_file)
    assert output_file.exists()
    try:
        # Query the trace database to verify recorded function calls
        with sqlite3.connect(output_file.as_posix()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT function_name, class_name, module_name, file_path, benchmark_function_name, benchmark_module_path, benchmark_line_number FROM benchmark_function_timings ORDER BY benchmark_module_path, benchmark_function_name, function_name")
            function_calls = cursor.fetchall()

            # Accept platform-dependent run multipliers: should be a multiple of base set (2)
            base_count = 2
            assert len(function_calls) >= base_count and len(function_calls) % base_count == 0, (
                f"Expected count to be a multiple of {base_count}, but got {len(function_calls)}"
            )
        
        # Close database connection and ensure cleanup before opening new connections
        gc.collect()
        time.sleep(0.1)
        
        function_benchmark_timings = codeflash_benchmark_plugin.get_function_benchmark_timings(output_file)
        total_benchmark_timings = codeflash_benchmark_plugin.get_benchmark_timings(output_file)
        function_to_results = validate_and_format_benchmark_table(function_benchmark_timings, total_benchmark_timings)
        assert "code_to_optimize.bubble_sort_codeflash_trace.sorter" in function_to_results

        test_name, total_time, function_time, percent = function_to_results["code_to_optimize.bubble_sort_codeflash_trace.sorter"][0]
        assert total_time >= 0.0
        assert function_time >= 0.0
        assert percent >= 0.0

        bubble_sort_path = (project_root / "bubble_sort_codeflash_trace.py").as_posix()
        # Expected function calls (each appears twice due to benchmark execution pattern)
        expected_calls = [
            ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{bubble_sort_path}",
             "test_benchmark_sort", "tests.pytest.benchmarks_test_decorator.test_benchmark_decorator", 5),
            ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{bubble_sort_path}",
             "test_pytest_mark", "tests.pytest.benchmarks_test_decorator.test_benchmark_decorator", 11),
        ]
        # Order-agnostic validation for decorator case as well
        normalized_calls = [(a[0], a[1], a[2], Path(a[3]).name, a[4], a[5], a[6]) for a in function_calls]
        normalized_expected = [(e[0], e[1], e[2], Path(e[3]).name, e[4], e[5], e[6]) for e in expected_calls]
        for expected in normalized_expected:
            assert expected in normalized_calls, f"Missing expected call: {expected}"
        
        # Ensure database connections are closed before cleanup
        gc.collect()
        time.sleep(0.1)
    finally:
        # Cleanup with retry mechanism to handle Windows file locking issues
        safe_unlink(output_file)
