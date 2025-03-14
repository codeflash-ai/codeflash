import sqlite3

from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
from codeflash.benchmarking.replay_test import generate_replay_test
from pathlib import Path
from codeflash.code_utils.code_utils import get_run_tmp_file
import shutil


def test_trace_benchmarks():
    # Test the trace_benchmarks function
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks"
    tests_root = project_root / "tests" / "test_trace_benchmarks"
    tests_root.mkdir(parents=False, exist_ok=False)
    output_file = (tests_root / Path("test_trace_benchmarks.trace")).resolve()
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
            "SELECT function_name, class_name, module_name, file_name, benchmark_function_name, benchmark_file_name, benchmark_line_number FROM function_calls ORDER BY benchmark_file_name, benchmark_function_name, function_name")
        function_calls = cursor.fetchall()

        # Assert the length of function calls
        assert len(function_calls) == 7, f"Expected 6 function calls, but got {len(function_calls)}"

        # Expected function calls
        expected_calls = [
            ("__init__", "Sorter", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{project_root / 'code_to_optimize/bubble_sort_codeflash_trace.py'}",
             "test_class_sort", "test_benchmark_bubble_sort.py", 20),

            ("sort_class", "Sorter", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{project_root / 'code_to_optimize/bubble_sort_codeflash_trace.py'}",
             "test_class_sort", "test_benchmark_bubble_sort.py", 18),

            ("sort_static", "Sorter", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{project_root / 'code_to_optimize/bubble_sort_codeflash_trace.py'}",
             "test_class_sort", "test_benchmark_bubble_sort.py", 19),

            ("sorter", "Sorter", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{project_root / 'code_to_optimize/bubble_sort_codeflash_trace.py'}",
             "test_class_sort", "test_benchmark_bubble_sort.py", 17),

            ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{project_root / 'code_to_optimize/bubble_sort_codeflash_trace.py'}",
             "test_sort", "test_benchmark_bubble_sort.py", 7),

            ("compute_and_sort", "", "code_to_optimize.process_and_bubble_sort_codeflash_trace",
             f"{project_root / 'code_to_optimize/process_and_bubble_sort_codeflash_trace.py'}",
             "test_compute_and_sort", "test_process_and_sort.py", 4),

            ("sorter", "", "code_to_optimize.bubble_sort_codeflash_trace",
             f"{project_root / 'code_to_optimize/bubble_sort_codeflash_trace.py'}",
             "test_no_func", "test_process_and_sort.py", 8),
        ]
        for idx, (actual, expected) in enumerate(zip(function_calls, expected_calls)):
            assert actual[0] == expected[0], f"Mismatch at index {idx} for function_name"
            assert actual[1] == expected[1], f"Mismatch at index {idx} for class_name"
            assert actual[2] == expected[2], f"Mismatch at index {idx} for module_name"
            assert Path(actual[3]).name == Path(expected[3]).name, f"Mismatch at index {idx} for file_name"
            assert actual[4] == expected[4], f"Mismatch at index {idx} for benchmark_function_name"
            assert actual[5] == expected[5], f"Mismatch at index {idx} for benchmark_file_name"
            assert actual[6] == expected[6], f"Mismatch at index {idx} for benchmark_line_number"
        # Close connection
        conn.close()
        generate_replay_test(output_file, tests_root)
        test_class_sort_path = tests_root / Path("test_benchmark_bubble_sort_py_test_class_sort__replay_test_0.py")
        assert test_class_sort_path.exists()
        test_class_sort_code = f"""
import dill as pickle

from code_to_optimize.bubble_sort_codeflash_trace import \\
    Sorter as code_to_optimize_bubble_sort_codeflash_trace_Sorter
from codeflash.benchmarking.replay_test import get_next_arg_and_return

functions = ['sorter', 'sort_class', 'sort_static']
trace_file_path = r"{output_file.as_posix()}"

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sorter():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="sorter", file_name=r"/Users/alvinryanputra/cf/codeflash/code_to_optimize/bubble_sort_codeflash_trace.py", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        function_name = "sorter"
        if not args:
            raise ValueError("No arguments provided for the method.")
        if function_name == "__init__":
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter(*args[1:], **kwargs)
        else:
            instance = args[0] # self
            ret = instance.sorter(*args[1:], **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sort_class():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="sort_class", file_name=r"/Users/alvinryanputra/cf/codeflash/code_to_optimize/bubble_sort_codeflash_trace.py", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        if not args:
            raise ValueError("No arguments provided for the method.")
        ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter.sort_class(*args[1:], **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter_sort_static():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="sort_static", file_name=r"/Users/alvinryanputra/cf/codeflash/code_to_optimize/bubble_sort_codeflash_trace.py", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter.sort_static(*args, **kwargs)

def test_code_to_optimize_bubble_sort_codeflash_trace_Sorter___init__():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="__init__", file_name=r"/Users/alvinryanputra/cf/codeflash/code_to_optimize/bubble_sort_codeflash_trace.py", class_name="Sorter", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        function_name = "__init__"
        if not args:
            raise ValueError("No arguments provided for the method.")
        if function_name == "__init__":
            ret = code_to_optimize_bubble_sort_codeflash_trace_Sorter(*args[1:], **kwargs)
        else:
            instance = args[0] # self
            ret = instance(*args[1:], **kwargs)

"""
        assert test_class_sort_path.read_text("utf-8").strip()==test_class_sort_code.strip()

        test_sort_path = tests_root / Path("test_benchmark_bubble_sort_py_test_sort__replay_test_0.py")
        assert test_sort_path.exists()
        test_sort_code = f"""
import dill as pickle

from code_to_optimize.bubble_sort_codeflash_trace import \\
    sorter as code_to_optimize_bubble_sort_codeflash_trace_sorter
from codeflash.benchmarking.replay_test import get_next_arg_and_return

functions = ['sorter']
trace_file_path = r"{output_file}"

def test_code_to_optimize_bubble_sort_codeflash_trace_sorter():
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="sorter", file_name=r"/Users/alvinryanputra/cf/codeflash/code_to_optimize/bubble_sort_codeflash_trace.py", num_to_get=100):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = code_to_optimize_bubble_sort_codeflash_trace_sorter(*args, **kwargs)

"""
        assert test_sort_path.read_text("utf-8").strip()==test_sort_code.strip()
    finally:
        # cleanup
        shutil.rmtree(tests_root)
        pass