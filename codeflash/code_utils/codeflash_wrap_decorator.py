from __future__ import annotations

import contextlib
import gc
import inspect
import os
import sqlite3
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import dill as pickle

if TYPE_CHECKING:
    from types import FrameType


class VerificationType(str, Enum):  # moved from codeflash/verification/codeflash_capture.py
    FUNCTION_CALL = (
        "function_call"  # Correctness verification for a test function, checks input values and output values)
    )
    INIT_STATE_FTO = "init_state_fto"  # Correctness verification for fto class instance attributes after init
    INIT_STATE_HELPER = "init_state_helper"  # Correctness verification for helper class instance attributes after init


F = TypeVar("F", bound=Callable[..., Any])


def _extract_class_name_tracer(frame_locals: dict[str, Any]) -> str | None:
    try:
        self_arg = frame_locals.get("self")
        if self_arg is not None:
            try:
                return self_arg.__class__.__name__
            except AttributeError:
                cls_arg = frame_locals.get("cls")
                if cls_arg is not None:
                    with contextlib.suppress(AttributeError):
                        return cls_arg.__name__
        else:
            cls_arg = frame_locals.get("cls")
            if cls_arg is not None:
                with contextlib.suppress(AttributeError):
                    return cls_arg.__name__
    except:  # noqa: E722
        # Handle cases where getattr is overridden and raises exceptions (e.g., wrapt)
        return None
    return None


def _get_module_name_cf_tracer(frame: FrameType | None) -> str:
    with contextlib.suppress(Exception):
        test_module = inspect.getmodule(frame)
        if test_module and hasattr(test_module, "__name__"):
            return test_module.__name__

    if frame is not None:
        return frame.f_globals.get("__name__", "unknown_module")
    return "unknown_module"


def extract_test_context_from_frame() -> tuple[str, str | None, str]:
    frame = inspect.currentframe()
    # optimize?
    try:
        potential_tests = []

        if frame is not None:
            frame = frame.f_back

        while frame is not None:
            try:
                function_name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                filename_path = Path(filename)
                frame_locals = frame.f_locals
                if function_name.startswith("test_"):
                    test_name = function_name
                    test_module_name = _get_module_name_cf_tracer(frame)
                    test_class_name = _extract_class_name_tracer(frame_locals)
                    return test_module_name, test_class_name, test_name

                if (
                    frame.f_globals.get("__name__", "").startswith("test_")
                    or filename_path.stem.startswith("test_")
                    or "test" in filename_path.parts
                ):
                    test_module_name = _get_module_name_cf_tracer(frame)
                    class_name = _extract_class_name_tracer(frame_locals)

                    if class_name:
                        if class_name.startswith("Test") or class_name.endswith("Test") or "test" in class_name.lower():
                            potential_tests.append((test_module_name, class_name, function_name))
                    elif "test" in test_module_name or filename_path.stem.startswith("test_"):
                        potential_tests.append((test_module_name, None, function_name))

                if (
                    function_name in ["runTest", "_runTest", "run", "_testMethodName"]
                    or "pytest" in str(frame.f_globals.get("__file__", ""))
                    or "unittest" in str(frame.f_globals.get("__file__", ""))
                ):
                    test_module_name = _get_module_name_cf_tracer(frame)
                    class_name = _extract_class_name_tracer(frame_locals)

                    if class_name and (class_name.startswith("Test") or "test" in class_name.lower()):
                        test_method = function_name
                        if "self" in frame_locals:
                            with contextlib.suppress(AttributeError, TypeError):
                                test_method = getattr(frame_locals["self"], "_testMethodName", function_name)
                        potential_tests.append((test_module_name, class_name, test_method))

                frame = frame.f_back

            except Exception:
                frame = frame.f_back
                continue

        if potential_tests:
            for test_module, test_class, test_func in potential_tests:
                if test_func.startswith("test_"):
                    return test_module, test_class, test_func
            return potential_tests[0]

        raise RuntimeError("No test function found in call stack")
    finally:
        del frame


def codeflash_behavior_async(func: F) -> F:
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        function_name = func.__name__
        line_id = f"{func.__name__}_{func.__code__.co_firstlineno}"
        loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])
        test_module_name, test_class_name, test_name = extract_test_context_from_frame()

        test_id = f"{test_module_name}:{test_class_name}:{test_name}:{line_id}:{loop_index}"

        if not hasattr(async_wrapper, "index"):
            async_wrapper.index = {}
        if test_id in async_wrapper.index:
            async_wrapper.index[test_id] += 1
        else:
            async_wrapper.index[test_id] = 0

        codeflash_test_index = async_wrapper.index[test_id]
        invocation_id = f"{line_id}_{codeflash_test_index}"
        test_stdout_tag = f"{test_module_name}:{(test_class_name + '.' if test_class_name else '')}{test_name}:{function_name}:{loop_index}:{invocation_id}"

        print(f"!$######{test_stdout_tag}######$!")

        iteration = os.environ.get("CODEFLASH_TEST_ITERATION", "0")
        db_path = Path.cwd() / f"codeflash_test_results_{iteration}.sqlite"
        codeflash_con = sqlite3.connect(db_path)
        codeflash_cur = codeflash_con.cursor()

        codeflash_cur.execute(
            "CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, "
            "test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, "
            "runtime INTEGER, return_value BLOB, verification_type TEXT)"
        )

        exception = None
        gc.disable()
        try:
            ret = func(*args, **kwargs)  # coroutine creation has some overhead, though it is very small
            counter = time.perf_counter_ns()
            return_value = await ret  # let's measure the actual execution time of the code
            codeflash_duration = time.perf_counter_ns() - counter
        except Exception as e:
            codeflash_duration = time.perf_counter_ns() - counter
            exception = e
        finally:
            gc.enable()

        print(f"!######{test_stdout_tag}######!")

        pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps((args, kwargs, return_value))
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
                VerificationType.FUNCTION_CALL.value,
            ),
        )
        codeflash_con.commit()
        codeflash_con.close()

        if exception:
            raise exception
        return return_value

    return async_wrapper


def codeflash_performance_async(func: F) -> F:
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        function_name = func.__name__
        line_id = f"{func.__name__}_{func.__code__.co_firstlineno}"
        loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])

        test_module_name, test_class_name, test_name = extract_test_context_from_frame()

        test_id = f"{test_module_name}:{test_class_name}:{test_name}:{line_id}:{loop_index}"

        if not hasattr(async_wrapper, "index"):
            async_wrapper.index = {}
        if test_id in async_wrapper.index:
            async_wrapper.index[test_id] += 1
        else:
            async_wrapper.index[test_id] = 0

        codeflash_test_index = async_wrapper.index[test_id]
        invocation_id = f"{line_id}_{codeflash_test_index}"
        test_stdout_tag = f"{test_module_name}:{(test_class_name + '.' if test_class_name else '')}{test_name}:{function_name}:{loop_index}:{invocation_id}"

        print(f"!$######{test_stdout_tag}######$!")

        exception = None
        gc.disable()
        try:
            ret = func(*args, **kwargs)
            counter = time.perf_counter_ns()
            return_value = await ret
            codeflash_duration = time.perf_counter_ns() - counter
        except Exception as e:
            codeflash_duration = time.perf_counter_ns() - counter
            exception = e
        finally:
            gc.enable()

        print(f"!######{test_stdout_tag}:{codeflash_duration}######!")

        if exception:
            raise exception
        return return_value

    return async_wrapper
