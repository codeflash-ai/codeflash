from __future__ import annotations

import gc
import inspect
import os
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from codeflash.code_utils.code_utils import get_run_tmp_file

F = TypeVar("F", bound=Callable[..., Any])


def extract_test_context_from_frame() -> tuple[str, str | None, str]:
    frame = inspect.currentframe()
    try:
        while frame:
            frame = frame.f_back
            if frame and frame.f_code.co_name.startswith("test_"):
                test_name = frame.f_code.co_name
                test_module_name = frame.f_globals.get("__name__", "unknown_module")
                test_class_name = None
                if "self" in frame.f_locals:
                    test_class_name = frame.f_locals["self"].__class__.__name__

                return test_module_name, test_class_name, test_name
        raise RuntimeError("No test function found in call stack")
    finally:
        del frame


def codeflash_behavior_async(func: F) -> F:
    function_name = func.__name__
    line_id = f"{func.__name__}_{func.__code__.co_firstlineno}"
    loop_index = int(os.environ.get("CODEFLASH_LOOP_INDEX", "1"))

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
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
            counter = time.perf_counter_ns()
            ret = func(*args, **kwargs)

            if inspect.isawaitable(ret):
                counter = time.perf_counter_ns()
                return_value = await ret
            else:
                return_value = ret

            codeflash_duration = time.perf_counter_ns() - counter
        except Exception as e:
            codeflash_duration = time.perf_counter_ns() - counter
            exception = e
        finally:
            gc.enable()

        print(f"!######{test_stdout_tag}######!")

        iteration = os.environ.get("CODEFLASH_TEST_ITERATION", "0")

        codeflash_run_tmp_dir = get_run_tmp_file(Path()).as_posix()

        output_file = Path(codeflash_run_tmp_dir) / f"test_return_values_{iteration}.bin"

        with output_file.open("ab") as f:
            pickled_values = (
                pickle.dumps((args, kwargs, exception)) if exception else pickle.dumps((args, kwargs, return_value))
            )
            _test_name = f"{test_module_name}:{(test_class_name + '.' if test_class_name else '')}{test_name}:{function_name}:{line_id}".encode(
                "ascii"
            )

            f.write(len(_test_name).to_bytes(4, byteorder="big"))
            f.write(_test_name)
            f.write(codeflash_duration.to_bytes(8, byteorder="big"))
            f.write(len(pickled_values).to_bytes(4, byteorder="big"))
            f.write(pickled_values)
            f.write(loop_index.to_bytes(8, byteorder="big"))
            f.write(len(invocation_id).to_bytes(4, byteorder="big"))
            f.write(invocation_id.encode("ascii"))

        if exception:
            raise exception
        return return_value

    return async_wrapper


def codeflash_performance_async(func: F) -> F:
    function_name = func.__name__
    line_id = f"{func.__name__}_{func.__code__.co_firstlineno}"
    loop_index = int(os.environ.get("CODEFLASH_LOOP_INDEX", "1"))

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
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
            counter = time.perf_counter_ns()
            ret = func(*args, **kwargs)

            if inspect.isawaitable(ret):
                counter = time.perf_counter_ns()
                return_value = await ret
            else:
                return_value = ret

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
