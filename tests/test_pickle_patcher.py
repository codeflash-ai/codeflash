import os
import pickle
import socket
from argparse import Namespace
from pathlib import Path

import dill
import pytest
import requests
import sqlite3

from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, TestingMode, TestType, TestFiles, TestFile
from codeflash.optimization.optimizer import Optimizer

try:
    import sqlalchemy
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine, Column, Integer, String
    from sqlalchemy.ext.declarative import declarative_base

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

from codeflash.picklepatch.pickle_patcher import PicklePatcher
from codeflash.picklepatch.pickle_placeholder import PicklePlaceholder
def test_picklepatch_simple_nested():
    """
    Test that a simple nested data structure pickles and unpickles correctly.
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
    """
    Test that a data structure containing a raw socket is replaced by
    PicklePlaceholder rather than raising an error.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data_with_socket = {
        "safe_value": 123,
        "raw_socket": s,
    }

    dumped = PicklePatcher.dumps(data_with_socket)
    reloaded = PicklePatcher.loads(dumped)

    # We expect "raw_socket" to be replaced by a placeholder
    assert isinstance(reloaded, dict)
    assert reloaded["safe_value"] == 123
    assert isinstance(reloaded["raw_socket"], PicklePlaceholder)

    # Attempting to use or access attributes => AttributeError 
    # (not RuntimeError as in original tests, our implementation uses AttributeError)
    with pytest.raises(AttributeError) :
        reloaded["raw_socket"].recv(1024)


def test_picklepatch_deeply_nested():
    """
    Test that deep nesting with unpicklable objects works correctly.
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
    """
    Test that a class with an unpicklable attribute works correctly.
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
    """
    Test that a data structure containing a database connection is replaced
    by PicklePlaceholder rather than raising an error.
    """
    # SQLite connection - not pickleable
    conn = sqlite3.connect(':memory:')
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
    with pytest.raises(AttributeError):
        reloaded["connection"].execute("SELECT 1")


def test_picklepatch_with_generator():
    """
    Test that a data structure containing a generator is replaced by
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
    with pytest.raises(AttributeError):
        reloaded["generator"].send(None)


def test_picklepatch_loads_standard_pickle():
    """
    Test that PicklePatcher.loads can correctly load data that was pickled
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
    """
    Test that PicklePatcher.loads can correctly load data that was pickled
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

    test_path = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_bubble_sort_picklepatch.py"
    ).resolve()
    test_path_perf = (
        Path(__file__).parent.resolve()
        / "../code_to_optimize/tests/pytest/test_bubble_sort_picklepatch_perf.py"
    ).resolve()
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_picklepatch.py").resolve()
    original_test =test_path.read_text("utf-8")
    try:
        tests_root = (Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/").resolve()
        project_root_path = (Path(__file__).parent / "..").resolve()
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        func = FunctionToOptimize(function_name="bubble_sort_with_unused_socket", parents=[], file_path=Path(fto_path))
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            test_path,
            [CodePosition(13,14), CodePosition(31,14)],
            func,
            project_root_path,
            "pytest",
            mode=TestingMode.BEHAVIOR,
        )
        os.chdir(original_cwd)
        assert success
        assert new_test is not None

        with test_path.open("w") as f:
            f.write(new_test)

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
        assert test_results.test_results[0].id.test_function_name =="test_bubble_sort_with_unused_socket"
        assert test_results.test_results[0].did_pass ==True
        assert test_results.test_results[1].id.test_function_name =="test_bubble_sort_with_used_socket"
        assert test_results.test_results[1].did_pass ==False
        # assert pickle placeholder problem
        print(test_results)
    finally:
        test_path.write_text(original_test)