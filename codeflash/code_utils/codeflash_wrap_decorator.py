from __future__ import annotations

import asyncio
import gc
import os
import sqlite3
from enum import Enum
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, TypeVar

import dill as pickle


class VerificationType(str, Enum):  # moved from codeflash/verification/codeflash_capture.py
    FUNCTION_CALL = (
        "function_call"  # Correctness verification for a test function, checks input values and output values)
    )
    INIT_STATE_FTO = "init_state_fto"  # Correctness verification for fto class instance attributes after init
    INIT_STATE_HELPER = "init_state_helper"  # Correctness verification for helper class instance attributes after init


F = TypeVar("F", bound=Callable[..., Any])


def get_run_tmp_file(file_path: Path) -> Path:  # moved from codeflash/code_utils/code_utils.py
    if not hasattr(get_run_tmp_file, "tmpdir"):
        get_run_tmp_file.tmpdir = TemporaryDirectory(prefix="codeflash_")
    return Path(get_run_tmp_file.tmpdir.name) / file_path


def extract_test_context_from_env() -> tuple[str, str | None, str]:
    test_module = os.environ["CODEFLASH_TEST_MODULE"]
    test_class = os.environ.get("CODEFLASH_TEST_CLASS", None)
    test_function = os.environ["CODEFLASH_TEST_FUNCTION"]

    if test_module and test_function:
        return (test_module, test_class if test_class else None, test_function)

    raise RuntimeError(
        "Test context environment variables not set - ensure tests are run through codeflash test runner"
    )


def codeflash_behavior_async(func: F) -> F:
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        loop = asyncio.get_running_loop()
        function_name = func.__name__
        line_id = os.environ["CODEFLASH_CURRENT_LINE_ID"]
        loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])
        test_module_name, test_class_name, test_name = extract_test_context_from_env()

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
        db_path = get_run_tmp_file(Path(f"test_return_values_{iteration}.sqlite"))
        codeflash_con = sqlite3.connect(db_path)
        codeflash_cur = codeflash_con.cursor()

        codeflash_cur.execute(
            "CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, "
            "test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, "
            "runtime INTEGER, return_value BLOB, verification_type TEXT)"
        )

        exception = None
        counter = loop.time()
        gc.disable()
        try:
            ret = func(*args, **kwargs)  # coroutine creation has some overhead, though it is very small
            counter = loop.time()
            return_value = await ret  # let's measure the actual execution time of the code
            codeflash_duration = int((loop.time() - counter) * 1_000_000_000)
        except Exception as e:
            codeflash_duration = int((loop.time() - counter) * 1_000_000_000)
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
        loop = asyncio.get_running_loop()
        function_name = func.__name__
        line_id = os.environ["CODEFLASH_CURRENT_LINE_ID"]
        loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])

        test_module_name, test_class_name, test_name = extract_test_context_from_env()

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
        counter = loop.time()
        gc.disable()
        try:
            ret = func(*args, **kwargs)
            counter = loop.time()
            return_value = await ret
            codeflash_duration = int((loop.time() - counter) * 1_000_000_000)
        except Exception as e:
            codeflash_duration = int((loop.time() - counter) * 1_000_000_000)
            exception = e
        finally:
            gc.enable()

        print(f"!######{test_stdout_tag}:{codeflash_duration}######!")

        if exception:
            raise exception
        return return_value

    return async_wrapper
