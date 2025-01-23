from __future__ import annotations

import functools
import gc
import inspect
import os
import sqlite3
import time

import dill as pickle

from codeflash.verification.test_results import VerificationType


def get_test_info_from_stack() -> tuple[str, str | None, str, str]:
    """Extract test information from the call stack."""
    stack = inspect.stack()

    # Default values
    test_module_name = ""
    test_class_name = None
    test_name = None
    line_id = ""  # Note that the way this line_id is defined is from the line_id called in instrumentation

    # Search through stack for test information
    for frame in stack:
        if frame.function.startswith("test_"):  # May need a more robust way to find the test file
            test_name = frame.function
            test_module_name = inspect.getmodule(frame[0]).__name__
            line_id = str(frame.lineno)
            # Check if it's a method in a class
            if "self" in frame.frame.f_locals:
                test_class_name = frame.frame.f_locals["self"].__class__.__name__
            break
        # Check if module name starts with test
        module_name = frame.frame.f_globals["__name__"]
        if module_name and module_name.split(".")[-1].startswith("test_"):
            test_module_name = module_name
            line_id = str(frame.lineno)
            if frame.function != "<module>":
                test_name = frame.function  # Technically not a test, but save the info since there is no test function
            # Check if it's in a class
            if "self" in frame.frame.f_locals:
                test_class_name = frame.frame.f_locals["self"].__class__.__name__
            break

    return test_module_name, test_class_name, test_name, line_id


def codeflash_capture(function_name: str, tmp_dir_path: str, is_fto: bool = False):
    """Defines decorator to be instrumented onto the init function in the code. Collects info of the test that called this, and captures the state of the instance."""

    def decorator(wrapped):
        @functools.wraps(wrapped)
        def wrapper(*args, **kwargs):
            # Dynamic information retrieved from stack
            test_module_name, test_class_name, test_name, line_id = get_test_info_from_stack()

            # Get env variables
            loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])
            codeflash_iteration = os.environ["CODEFLASH_TEST_ITERATION"]

            # Create test_id
            test_id = f"{test_module_name}:{test_class_name}:{test_name}:{line_id}:{loop_index}"

            # Initialize index tracking if needed, handles multiple instances created in the same test line number
            if not hasattr(wrapper, "index"):
                wrapper.index = {}

            # Update index for this test
            if test_id in wrapper.index:
                wrapper.index[test_id] += 1
            else:
                wrapper.index[test_id] = 0

            codeflash_test_index = wrapper.index[test_id]

            # Generate invocation id
            invocation_id = f"{line_id}_{codeflash_test_index}"
            print(
                f"!######{test_module_name}:{(test_class_name + '.' if test_class_name else '')}{test_name}:{function_name}:{loop_index}:{invocation_id}######!"
            )
            # Connect to sqlite
            codeflash_con = sqlite3.connect(f"{tmp_dir_path}_{codeflash_iteration}.sqlite")
            codeflash_cur = codeflash_con.cursor()

            # Record timing information
            exception = None
            gc.disable()
            try:
                counter = time.perf_counter_ns()
                wrapped(*args, **kwargs)
                codeflash_duration = time.perf_counter_ns() - counter
            except Exception as e:
                codeflash_duration = time.perf_counter_ns() - counter
                exception = e
            finally:
                gc.enable()

            # Capture instance state after initialization
            instance_state = args[0].__dict__  # self is always the first argument
            codeflash_cur.execute(
                "CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)"
            )

            # Write to sqlite
            pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(instance_state)
            codeflash_cur.execute(
                "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    test_module_name,
                    test_class_name,
                    test_name,
                    function_name,
                    loop_index,
                    invocation_id,
                    codeflash_duration,
                    pickled_return_value,
                    VerificationType.INSTANCE_STATE_FTO if is_fto else VerificationType.INSTANCE_STATE_HELPER,
                ),
            )
            codeflash_con.commit()
            if exception:
                raise exception

        return wrapper

    return decorator
