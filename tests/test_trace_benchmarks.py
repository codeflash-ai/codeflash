import multiprocessing
import shutil
import sqlite3
from pathlib import Path

import pytest

from codeflash.benchmarking.plugin.plugin import codeflash_benchmark_plugin
from codeflash.benchmarking.replay_test import generate_replay_test
from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
from codeflash.benchmarking.utils import validate_and_format_benchmark_table


def test_trace_benchmarks() -> None:
    # Test the trace_benchmarks function
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks_test"
    replay_tests_dir = benchmarks_root / "codeflash_replay_tests"
    tests_root = project_root / "tests"
    output_file = (benchmarks_root / Path("test_trace_benchmarks.trace")).resolve()
    trace_benchmarks_pytest(benchmarks_root, tests_root, project_root, output_file)
    assert output_file.exists()
    try:
        # check contents of trace file
        # connect to database
        conn = sqlite3.connect(output_file.as_posix())
        cursor = conn.cursor()

        # Get the count of records
        # Get all records
        cursor.execute(
            "SELECT function_name, class_name, module_name, file_path, benchmark_function_name, benchmark_module_path, benchmark_line_number FROM benchmark_function_timings ORDER BY benchmark_module_path, benchmark_function_name, function_name")
        function_calls = cursor.fetchall()

        # Assert the length of function calls
        assert len(function_calls) == 8, f"Expected 8 function calls, but got {len(function_calls)}"

        bubble_sort_path = (project_root / "bubble_sort_codeflash_trace.py").as_posix()
        process_and_bubble_sort_path = (project_root / "process_and_bubble_sort_codeflash_trace.py").as_posix()
        # Expected function calls
        expected_calls = [
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
        for idx, (actual, expected) in enumerate(zip(function_calls, expected_calls)):
            assert actual[0] == expected[0], f"Mismatch at index {idx} for function_name"
            assert actual[1] == expected[1], f"Mismatch at index {idx} for class_name"
            assert actual[2] == expected[2], f"Mismatch at index {idx} for module_name"
            assert Path(actual[3]).name == Path(expected[3]).name, f"Mismatch at index {idx} for file_path"
            assert actual[4] == expected[4], f"Mismatch at index {idx} for benchmark_function_name"
            assert actual[5] == expected[5], f"Mismatch at index {idx} for benchmark_module_path"
            assert actual[6] == expected[6], f"Mismatch at index {idx} for benchmark_line_number"
        # Close connection
        conn.close()
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

def test_code_to_optimize_bubble_sort_codeflash_trace_sorter():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_sort", function_name="sorter", file_path=r"{bubble_sort_path}", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_bubble_sort_codeflash_trace_sorter(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sorter():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_class_sort", function_name="sorter", file_path=r"{bubble_sort_path}", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        function_name = "sorter"
        if not args:
            raise ValueError("No arguments provided for the method.")
        if function_name == "__init__":
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter(*args[1:], **kwargs)
        else:
            instance = args[0] # self
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter.sorter(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sort_class():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_class_sort2", function_name="sort_class", file_path=r"{bubble_sort_path}", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        if not args:
            raise ValueError("No arguments provided for the method.")
        ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter.sort_class(*args[1:], **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sort_static():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_class_sort3", function_name="sort_static", file_path=r"{bubble_sort_path}", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter.sort_static(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter___init__():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_class_sort4", function_name="__init__", file_path=r"{bubble_sort_path}", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        function_name = "__init__"
        if not args:
            raise ValueError("No arguments provided for the method.")
        if function_name == "__init__":
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter(*args[1:], **kwargs)
        else:
            instance = args[0] # self
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
trace_file_path = r"{output_file}"

def test_code_to_optimize_process_and_bubble_sort_codeflash_trace_compute_and_sort():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_compute_and_sort", function_name="compute_and_sort", file_path=r"{process_and_bubble_sort_path}", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_process_and_bubble_sort_codeflash_trace_compute_and_sort(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_sorter():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="test_no_func", function_name="sorter", file_path=r"{bubble_sort_path}", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_bubble_sort_codeflash_trace_sorter(*args, **kwargs)

"""
        assert test_sort_path.read_text("utf-8").strip()==test_sort_code.strip()
    finally:
        # cleanup
        output_file.unlink(missing_ok=True)
        shutil.rmtree(replay_tests_dir)

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
        # check contents of trace file
        # connect to database
        conn = sqlite3.connect(output_file.as_posix())
        cursor = conn.cursor()

        # Get the count of records
        # Get all records
        cursor.execute(
            "SELECT function_name, class_name, module_name, file_path, benchmark_function_name, benchmark_module_path, benchmark_line_number FROM benchmark_function_timings ORDER BY benchmark_module_path, benchmark_function_name, function_name")
        function_calls = cursor.fetchall()

        # Assert the length of function calls
        assert len(function_calls) == 10, f"Expected 10 function calls, but got {len(function_calls)}"
        function_benchmark_timings = codeflash_benchmark_plugin.get_function_benchmark_timings(output_file)
        total_benchmark_timings = codeflash_benchmark_plugin.get_benchmark_timings(output_file)
        function_to_results = validate_and_format_benchmark_table(function_benchmark_timings, total_benchmark_timings)
        assert "code_to_optimize.bubble_sort_codeflash_trace.sorter" in function_to_results

        test_name, total_time, function_time, percent = function_to_results["code_to_optimize.bubble_sort_codeflash_trace.sorter"][0]
        assert total_time > 0.0
        assert function_time > 0.0
        assert percent > 0.0

        bubble_sort_path = (project_root / "bubble_sort_codeflash_trace.py").as_posix()
        # Expected function calls
        expected_calls = [
            ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{bubble_sort_path}",
             "test_benchmark_sort", "tests.pytest.benchmarks_multithread.test_multithread_sort", 4),
        ]
        for idx, (actual, expected) in enumerate(zip(function_calls, expected_calls)):
            assert actual[0] == expected[0], f"Mismatch at index {idx} for function_name"
            assert actual[1] == expected[1], f"Mismatch at index {idx} for class_name"
            assert actual[2] == expected[2], f"Mismatch at index {idx} for module_name"
            assert Path(actual[3]).name == Path(expected[3]).name, f"Mismatch at index {idx} for file_path"
            assert actual[4] == expected[4], f"Mismatch at index {idx} for benchmark_function_name"
            assert actual[5] == expected[5], f"Mismatch at index {idx} for benchmark_module_path"
            assert actual[6] == expected[6], f"Mismatch at index {idx} for benchmark_line_number"
        # Close connection
        conn.close()

    finally:
        # cleanup
        output_file.unlink(missing_ok=True)

def test_trace_benchmark_decorator() -> None:
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks_test_decorator"
    tests_root = project_root / "tests"
    output_file = (benchmarks_root / Path("test_trace_benchmarks.trace")).resolve()
    trace_benchmarks_pytest(benchmarks_root, tests_root, project_root, output_file)
    assert output_file.exists()
    try:
        # check contents of trace file
        # connect to database
        conn = sqlite3.connect(output_file.as_posix())
        cursor = conn.cursor()

        # Get the count of records
        # Get all records
        cursor.execute(
            "SELECT function_name, class_name, module_name, file_path, benchmark_function_name, benchmark_module_path, benchmark_line_number FROM benchmark_function_timings ORDER BY benchmark_module_path, benchmark_function_name, function_name")
        function_calls = cursor.fetchall()

        # Assert the length of function calls
        assert len(function_calls) == 2, f"Expected 2 function calls, but got {len(function_calls)}"
        function_benchmark_timings = codeflash_benchmark_plugin.get_function_benchmark_timings(output_file)
        total_benchmark_timings = codeflash_benchmark_plugin.get_benchmark_timings(output_file)
        function_to_results = validate_and_format_benchmark_table(function_benchmark_timings, total_benchmark_timings)
        assert "code_to_optimize.bubble_sort_codeflash_trace.sorter" in function_to_results

        test_name, total_time, function_time, percent = function_to_results["code_to_optimize.bubble_sort_codeflash_trace.sorter"][0]
        assert total_time > 0.0
        assert function_time > 0.0
        assert percent > 0.0

        bubble_sort_path = (project_root / "bubble_sort_codeflash_trace.py").as_posix()
        # Expected function calls
        expected_calls = [
            ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{bubble_sort_path}",
             "test_benchmark_sort", "tests.pytest.benchmarks_test_decorator.test_benchmark_decorator", 5),
            ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{bubble_sort_path}",
             "test_pytest_mark", "tests.pytest.benchmarks_test_decorator.test_benchmark_decorator", 11),
        ]
        for idx, (actual, expected) in enumerate(zip(function_calls, expected_calls)):
            assert actual[0] == expected[0], f"Mismatch at index {idx} for function_name"
            assert actual[1] == expected[1], f"Mismatch at index {idx} for class_name"
            assert actual[2] == expected[2], f"Mismatch at index {idx} for module_name"
            assert Path(actual[3]).name == Path(expected[3]).name, f"Mismatch at index {idx} for file_path"
            assert actual[4] == expected[4], f"Mismatch at index {idx} for benchmark_function_name"
            assert actual[5] == expected[5], f"Mismatch at index {idx} for benchmark_module_path"
        # Close connection
        conn.close()

    finally:
        # cleanup
        output_file.unlink(missing_ok=True)
