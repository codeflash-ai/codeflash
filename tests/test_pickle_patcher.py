import os
import pickle
import shutil
import socket
import sqlite3
from argparse import Namespace
from pathlib import Path

import dill
import pytest

from codeflash.benchmarking.plugin.plugin import codeflash_benchmark_plugin
from codeflash.benchmarking.replay_test import generate_replay_test
from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
from codeflash.benchmarking.utils import validate_and_format_benchmark_table
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, TestFile, TestFiles, TestingMode, TestsInFile, TestType
from codeflash.optimization.optimizer import Optimizer
from codeflash.verification.equivalence import compare_test_results

try:
    import sqlalchemy
    from sqlalchemy import Column, Integer, String, create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import Session

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

from codeflash.picklepatch.pickle_patcher import PicklePatcher
from codeflash.picklepatch.pickle_placeholder import PicklePlaceholder, PicklePlaceholderAccessError


def test_picklepatch_simple_nested():
    """Test that a simple nested data structure pickles and unpickles correctly.
    """
    original_data = {
        "numbers": [1, 2, 3],
        "nested_dict": {"key": "value", "another": 42},
    }

    dumped = PicklePatcher.dumps(original_data)
    reloaded = PicklePatcher.loads(dumped)

    assert reloaded == original_data
    # Everything was pickleable, so no placeholders should appear.


def test_picklepatch_with_socket():
    """Test that a data structure containing a raw socket is replaced by
    PicklePlaceholder rather than raising an error.
    """
    # Create a pair of connected sockets instead of a single socket
    sock1, sock2 = socket.socketpair()

    data_with_socket = {
        "safe_value": 123,
        "raw_socket": sock1,
    }

    # Send a message through sock1, which can be received by sock2
    sock1.send(b"Hello, world!")
    received = sock2.recv(1024)
    assert received == b"Hello, world!"
    # Pickle the data structure containing the socket
    dumped = PicklePatcher.dumps(data_with_socket)
    reloaded = PicklePatcher.loads(dumped)

    # We expect "raw_socket" to be replaced by a placeholder
    assert isinstance(reloaded, dict)
    assert reloaded["safe_value"] == 123
    assert isinstance(reloaded["raw_socket"], PicklePlaceholder)

    # Attempting to use or access attributes => AttributeError
    # (not RuntimeError as in original tests, our implementation uses AttributeError)
    with pytest.raises(PicklePlaceholderAccessError):
        reloaded["raw_socket"].recv(1024)

    # Clean up by closing both sockets
    sock1.close()
    sock2.close()


def test_picklepatch_deeply_nested():
    """Test that deep nesting with unpicklable objects works correctly.
    """
    # Create a deeply nested structure with an unpicklable object
    deep_nested = {
        "level1": {
            "level2": {
                "level3": {
                    "normal": "value",
                    "socket": socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                }
            }
        }
    }

    dumped = PicklePatcher.dumps(deep_nested)
    reloaded = PicklePatcher.loads(dumped)

    # We should be able to access the normal value
    assert reloaded["level1"]["level2"]["level3"]["normal"] == "value"

    # The socket should be replaced with a placeholder
    assert isinstance(reloaded["level1"]["level2"]["level3"]["socket"], PicklePlaceholder)

def test_picklepatch_class_with_unpicklable_attr():
    """Test that a class with an unpicklable attribute works correctly.
    """
    class TestClass:
        def __init__(self):
            self.normal = "normal value"
            self.unpicklable = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    obj = TestClass()

    dumped = PicklePatcher.dumps(obj)
    reloaded = PicklePatcher.loads(dumped)

    # Normal attribute should be preserved
    assert reloaded.normal == "normal value"

    # Unpicklable attribute should be replaced with a placeholder
    assert isinstance(reloaded.unpicklable, PicklePlaceholder)




def test_picklepatch_with_database_connection():
    """Test that a data structure containing a database connection is replaced
    by PicklePlaceholder rather than raising an error.
    """
    # SQLite connection - not pickleable
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    data_with_db = {
        "description": "Database connection",
        "connection": conn,
        "cursor": cursor,
    }

    dumped = PicklePatcher.dumps(data_with_db)
    reloaded = PicklePatcher.loads(dumped)

    # Both connection and cursor should become placeholders
    assert isinstance(reloaded, dict)
    assert reloaded["description"] == "Database connection"
    assert isinstance(reloaded["connection"], PicklePlaceholder)
    assert isinstance(reloaded["cursor"], PicklePlaceholder)

    # Attempting to use attributes => AttributeError
    with pytest.raises(PicklePlaceholderAccessError):
        reloaded["connection"].execute("SELECT 1")


def test_picklepatch_with_generator():
    """Test that a data structure containing a generator is replaced by
    PicklePlaceholder rather than raising an error.
    """

    def simple_generator():
        yield 1
        yield 2
        yield 3

    # Create a generator
    gen = simple_generator()

    # Put it in a data structure
    data_with_generator = {
        "description": "Contains a generator",
        "generator": gen,
        "normal_list": [1, 2, 3]
    }

    dumped = PicklePatcher.dumps(data_with_generator)
    reloaded = PicklePatcher.loads(dumped)

    # Generator should be replaced with a placeholder
    assert isinstance(reloaded, dict)
    assert reloaded["description"] == "Contains a generator"
    assert reloaded["normal_list"] == [1, 2, 3]
    assert isinstance(reloaded["generator"], PicklePlaceholder)

    # Attempting to use the generator => AttributeError
    with pytest.raises(TypeError):
        next(reloaded["generator"])

    # Attempting to call methods on the generator => AttributeError
    with pytest.raises(PicklePlaceholderAccessError):
        reloaded["generator"].send(None)


def test_picklepatch_loads_standard_pickle():
    """Test that PicklePatcher.loads can correctly load data that was pickled
    using the standard pickle module.
    """
    # Create a simple data structure
    original_data = {
        "numbers": [1, 2, 3],
        "nested_dict": {"key": "value", "another": 42},
        "tuple": (1, "two", 3.0),
    }

    # Pickle it with standard pickle
    pickled_data = pickle.dumps(original_data)

    # Load with PicklePatcher
    reloaded = PicklePatcher.loads(pickled_data)

    # Verify the data is correctly loaded
    assert reloaded == original_data
    assert isinstance(reloaded, dict)
    assert reloaded["numbers"] == [1, 2, 3]
    assert reloaded["nested_dict"]["key"] == "value"
    assert reloaded["tuple"] == (1, "two", 3.0)


def test_picklepatch_loads_dill_pickle():
    """Test that PicklePatcher.loads can correctly load data that was pickled
    using the dill module, which can pickle more complex objects than the
    standard pickle module.
    """
    # Create a more complex data structure that includes a lambda function
    # which dill can handle but standard pickle cannot
    original_data = {
        "numbers": [1, 2, 3],
        "function": lambda x: x * 2,
        "nested": {
            "another_function": lambda y: y ** 2
        }
    }

    # Pickle it with dill
    dilled_data = dill.dumps(original_data)

    # Load with PicklePatcher
    reloaded = PicklePatcher.loads(dilled_data)

    # Verify the data structure
    assert isinstance(reloaded, dict)
    assert reloaded["numbers"] == [1, 2, 3]

    # Test that the functions actually work
    assert reloaded["function"](5) == 10
    assert reloaded["nested"]["another_function"](4) == 16

def test_run_and_parse_picklepatch() -> None:
    """Test the end to end functionality of picklepatch, from tracing benchmarks to running the replay tests.

    The first example has an argument (an object containing a socket) that is not pickleable  However, the socket attributs is not used, so we are able to compare the test results with the optimized test results.
    Here, we are simply 'ignoring' the unused unpickleable object.

    The second example also has an argument (an object containing socket) that is not pickleable. The socket attribute is used, which results in an error thrown by the PicklePlaceholder object.
    Both the original and optimized results should error out in this case, but this should be flagged as incorrect behavior when comparing test results,
    since we were not able to reuse the unpickleable object in the replay test.
    """
    # Init paths
    project_root = Path(__file__).parent.parent.resolve()
    tests_root = project_root / "code_to_optimize" / "tests" / "pytest"
    benchmarks_root = project_root / "code_to_optimize" / "tests" / "pytest" / "benchmarks_socket_test"
    replay_tests_dir = benchmarks_root / "codeflash_replay_tests"
    output_file = (benchmarks_root / Path("test_trace_benchmarks.trace")).resolve()
    fto_unused_socket_path = (project_root / "code_to_optimize" / "bubble_sort_picklepatch_test_unused_socket.py").resolve()
    fto_used_socket_path = (project_root / "code_to_optimize" / "bubble_sort_picklepatch_test_used_socket.py").resolve()
    original_fto_unused_socket_code = fto_unused_socket_path.read_text("utf-8")
    original_fto_used_socket_code = fto_used_socket_path.read_text("utf-8")
    # Trace benchmarks
    trace_benchmarks_pytest(benchmarks_root, tests_root, project_root, output_file)
    assert output_file.exists()
    try:
        # Check contents
        conn = sqlite3.connect(output_file.as_posix())
        cursor = conn.cursor()

        cursor.execute(
            "SELECT function_name, class_name, module_name, file_path, benchmark_function_name, benchmark_module_path, benchmark_line_number FROM benchmark_function_timings ORDER BY benchmark_module_path, benchmark_function_name, function_name")
        function_calls = cursor.fetchall()

        # Assert the length of function calls
        assert len(function_calls) == 2, f"Expected 2 function calls, but got {len(function_calls)}"
        function_benchmark_timings = codeflash_benchmark_plugin.get_function_benchmark_timings(output_file)
        total_benchmark_timings = codeflash_benchmark_plugin.get_benchmark_timings(output_file)
        function_to_results = validate_and_format_benchmark_table(function_benchmark_timings, total_benchmark_timings)
        assert "code_to_optimize.bubble_sort_picklepatch_test_unused_socket.bubble_sort_with_unused_socket" in function_to_results

        test_name, total_time, function_time, percent = function_to_results["code_to_optimize.bubble_sort_picklepatch_test_unused_socket.bubble_sort_with_unused_socket"][0]
        assert total_time > 0.0
        assert function_time > 0.0
        assert percent > 0.0

        test_name, total_time, function_time, percent = \
        function_to_results["code_to_optimize.bubble_sort_picklepatch_test_unused_socket.bubble_sort_with_unused_socket"][0]
        assert total_time > 0.0
        assert function_time > 0.0
        assert percent > 0.0

        bubble_sort_unused_socket_path = (project_root / "code_to_optimize"/ "bubble_sort_picklepatch_test_unused_socket.py").as_posix()
        bubble_sort_used_socket_path = (project_root / "code_to_optimize" / "bubble_sort_picklepatch_test_used_socket.py").as_posix()
        # Expected function calls
        expected_calls = [
            ("bubble_sort_with_unused_socket", "", "code_to_optimize.bubble_sort_picklepatch_test_unused_socket",
             f"{bubble_sort_unused_socket_path}",
             "test_socket_picklepatch", "code_to_optimize.tests.pytest.benchmarks_socket_test.test_socket", 12),
            ("bubble_sort_with_used_socket", "", "code_to_optimize.bubble_sort_picklepatch_test_used_socket",
             f"{bubble_sort_used_socket_path}",
             "test_used_socket_picklepatch", "code_to_optimize.tests.pytest.benchmarks_socket_test.test_socket", 20)
        ]
        for idx, (actual, expected) in enumerate(zip(function_calls, expected_calls)):
            assert actual[0] == expected[0], f"Mismatch at index {idx} for function_name"
            assert actual[1] == expected[1], f"Mismatch at index {idx} for class_name"
            assert actual[2] == expected[2], f"Mismatch at index {idx} for module_name"
            assert Path(actual[3]).name == Path(expected[3]).name, f"Mismatch at index {idx} for file_path"
            assert actual[4] == expected[4], f"Mismatch at index {idx} for benchmark_function_name"
            assert actual[5] == expected[5], f"Mismatch at index {idx} for benchmark_module_path"
            assert actual[6] == expected[6], f"Mismatch at index {idx} for benchmark_line_number"
        conn.close()

        # Generate replay test
        generate_replay_test(output_file, replay_tests_dir)
        replay_test_path = replay_tests_dir / Path(
            "test_code_to_optimize_tests_pytest_benchmarks_socket_test_test_socket__replay_test_0.py")
        replay_test_perf_path = replay_tests_dir / Path(
            "test_code_to_optimize_tests_pytest_benchmarks_socket_test_test_socket__replay_test_0_perf.py")
        assert replay_test_path.exists()
        original_replay_test_code = replay_test_path.read_text("utf-8")

        # Instrument the replay test
        func = FunctionToOptimize(function_name="bubble_sort_with_unused_socket", parents=[], file_path=Path(fto_unused_socket_path))
        original_cwd = Path.cwd()
        run_cwd = project_root
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            replay_test_path,
            [CodePosition(17, 15)],
            func,
            project_root,
            "pytest",
            mode=TestingMode.BEHAVIOR,
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        replay_test_path.write_text(new_test)

        opt = Optimizer(
            Namespace(
                project_root=project_root,
                disable_telemetry=True,
                tests_root=tests_root,
                test_framework="pytest",
                pytest_cmd="pytest",
                experiment_id=None,
                test_project_root=project_root,
            )
        )

        # Run the replay test for the original code that does not use the socket
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.REPLAY_TEST
        replay_test_function = "test_code_to_optimize_bubble_sort_picklepatch_test_unused_socket_bubble_sort_with_unused_socket_test_socket_picklepatch"
        func_optimizer = opt.create_function_optimizer(func)
        func_optimizer.test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=replay_test_path,
                    test_type=test_type,
                    original_file_path=replay_test_path,
                    benchmarking_file_path=replay_test_perf_path,
                    tests_in_file=[TestsInFile(test_file=replay_test_path, test_class=None, test_function=replay_test_function, test_type=test_type)],
                )
            ]
        )
        test_results_unused_socket, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=1.0,
        )
        assert len(test_results_unused_socket) == 1
        assert test_results_unused_socket.test_results[0].id.test_module_path == "code_to_optimize.tests.pytest.benchmarks_socket_test.codeflash_replay_tests.test_code_to_optimize_tests_pytest_benchmarks_socket_test_test_socket__replay_test_0"
        assert test_results_unused_socket.test_results[0].id.test_function_name == "test_code_to_optimize_bubble_sort_picklepatch_test_unused_socket_bubble_sort_with_unused_socket_test_socket_picklepatch"
        assert test_results_unused_socket.test_results[0].did_pass == True

        # Replace with optimized candidate
        fto_unused_socket_path.write_text("""
from codeflash.benchmarking.codeflash_trace import codeflash_trace

@codeflash_trace
def bubble_sort_with_unused_socket(data_container):
    # Extract the list to sort, leaving the socket untouched
    numbers = data_container.get('numbers', []).copy()
    return sorted(numbers)
""")
        # Run optimized code for unused socket
        optimized_test_results_unused_socket, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=1.0,
        )
        assert len(optimized_test_results_unused_socket) == 1
        verification_result = compare_test_results(test_results_unused_socket, optimized_test_results_unused_socket)
        assert verification_result is True

        # Remove the previous instrumentation
        replay_test_path.write_text(original_replay_test_code)
        # Instrument the replay test
        func = FunctionToOptimize(function_name="bubble_sort_with_used_socket", parents=[], file_path=Path(fto_used_socket_path))
        success, new_test = inject_profiling_into_existing_test(
            replay_test_path,
            [CodePosition(23,15)],
            func,
            project_root,
            "pytest",
            mode=TestingMode.BEHAVIOR,
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None
        replay_test_path.write_text(new_test)

        # Run test for original function code that uses the socket. This should fail, as the PicklePlaceholder is accessed.
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_type = TestType.REPLAY_TEST
        func = FunctionToOptimize(function_name="bubble_sort_with_used_socket", parents=[],
                                  file_path=Path(fto_used_socket_path))
        replay_test_function = "test_code_to_optimize_bubble_sort_picklepatch_test_used_socket_bubble_sort_with_used_socket_test_used_socket_picklepatch"
        func_optimizer = opt.create_function_optimizer(func)
        func_optimizer.test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=replay_test_path,
                    test_type=test_type,
                    original_file_path=replay_test_path,
                    benchmarking_file_path=replay_test_perf_path,
                    tests_in_file=[
                        TestsInFile(test_file=replay_test_path, test_class=None, test_function=replay_test_function,
                                    test_type=test_type)],
                )
            ]
        )
        test_results_used_socket, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=1.0,
        )
        assert len(test_results_used_socket) == 1
        assert test_results_used_socket.test_results[
                   0].id.test_module_path == "code_to_optimize.tests.pytest.benchmarks_socket_test.codeflash_replay_tests.test_code_to_optimize_tests_pytest_benchmarks_socket_test_test_socket__replay_test_0"
        assert test_results_used_socket.test_results[
                   0].id.test_function_name == "test_code_to_optimize_bubble_sort_picklepatch_test_used_socket_bubble_sort_with_used_socket_test_used_socket_picklepatch"
        assert test_results_used_socket.test_results[0].did_pass is False
        print("test results used socket")
        print(test_results_used_socket)
        # Replace with optimized candidate
        fto_used_socket_path.write_text("""
from codeflash.benchmarking.codeflash_trace import codeflash_trace

@codeflash_trace
def bubble_sort_with_used_socket(data_container):
    # Extract the list to sort, leaving the socket untouched
    numbers = data_container.get('numbers', []).copy()
    socket = data_container.get('socket')
    socket.send("Hello from the optimized function!")
    return sorted(numbers)
        """)

        # Run test for optimized function code that uses the socket. This should fail, as the PicklePlaceholder is accessed.
        optimized_test_results_used_socket, coverage_data = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=1.0,
        )
        assert len(test_results_used_socket) == 1
        assert test_results_used_socket.test_results[
                   0].id.test_module_path == "code_to_optimize.tests.pytest.benchmarks_socket_test.codeflash_replay_tests.test_code_to_optimize_tests_pytest_benchmarks_socket_test_test_socket__replay_test_0"
        assert test_results_used_socket.test_results[
                   0].id.test_function_name == "test_code_to_optimize_bubble_sort_picklepatch_test_used_socket_bubble_sort_with_used_socket_test_used_socket_picklepatch"
        assert test_results_used_socket.test_results[0].did_pass is False

        # Even though tests threw the same error, we reject this as the behavior of the unpickleable object could not be determined.
        assert compare_test_results(test_results_used_socket, optimized_test_results_used_socket) is False

    finally:
        # cleanup
        output_file.unlink(missing_ok=True)
        shutil.rmtree(replay_tests_dir, ignore_errors=True)
        fto_unused_socket_path.write_text(original_fto_unused_socket_code)
        fto_used_socket_path.write_text(original_fto_used_socket_code)

