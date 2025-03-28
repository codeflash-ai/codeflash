from __future__ import annotations

import functools
import gc
import inspect
import os
import sqlite3
import time
from pathlib import Path

import dill as pickle

from codeflash.models.models import VerificationType


def get_test_info_from_stack(tests_root: str) -> tuple[str, str | None, str, str]:
    """Extract test information by walking the call stack from the current frame."""
    test_module_name = ""
    test_class_name: str | None = None
    test_name: str | None = None
    line_id = ""

    # Get current frame and skip our own function's frame
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back

    # Walk the stack
    while frame is not None:
        function_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        # Check if function name indicates a test (e.g., starts with "test_")
        if function_name.startswith("test_"):
            test_name = function_name
            test_module = inspect.getmodule(frame)
            if hasattr(test_module, "__name__"):
                test_module_name = test_module.__name__
            line_id = str(lineno)

            # Check if it's a method in a class
            if (
                "self" in frame.f_locals
                and hasattr(frame.f_locals["self"], "__class__")
                and hasattr(frame.f_locals["self"].__class__, "__name__")
            ):
                test_class_name = frame.f_locals["self"].__class__.__name__
            break

        # Check for instantiation on the module level
        if (
            "__name__" in frame.f_globals
            and frame.f_globals["__name__"].split(".")[-1].startswith("test_")
            and Path(filename).resolve().is_relative_to(Path(tests_root))
            and function_name == "<module>"
        ):
            test_module_name = frame.f_globals["__name__"]
            line_id = str(lineno)

            #     # Check if it's a method in a class
            if (
                "self" in frame.f_locals
                and hasattr(frame.f_locals["self"], "__class__")
                and hasattr(frame.f_locals["self"].__class__, "__name__")
            ):
                test_class_name = frame.f_locals["self"].__class__.__name__
            break

        # Go to the previous frame
        frame = frame.f_back

    return test_module_name, test_class_name, test_name, line_id


def codeflash_capture(function_name: str, tmp_dir_path: str, tests_root: str, is_fto: bool = False):
    """Defines decorator to be instrumented onto the init function in the code. Collects info of the test that called this, and captures the state of the instance."""

    def decorator(wrapped):
        @functools.wraps(wrapped)
        def wrapper(*args, **kwargs):
            # Dynamic information retrieved from stack
            test_module_name, test_class_name, test_name, line_id = get_test_info_from_stack(tests_root)

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
            if hasattr(args[0], "__dict__"):
                instance_state = args[
                    0
                ].__dict__  # self is always the first argument, this is ensured during instrumentation
            else:
                raise ValueError("Instance state could not be captured.")
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
                    VerificationType.INIT_STATE_FTO if is_fto else VerificationType.INIT_STATE_HELPER,
                ),
            )
            codeflash_con.commit()
            if exception:
                raise exception

        return wrapper

    return decorator
