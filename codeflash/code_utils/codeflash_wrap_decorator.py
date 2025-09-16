from __future__ import annotations

import asyncio
import contextlib
import gc
import inspect
import os
import sqlite3
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from tempfile import TemporaryDirectory
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


def get_run_tmp_file(file_path: Path) -> Path:  # moved from codeflash/code_utils/code_utils.py
    if not hasattr(get_run_tmp_file, "tmpdir"):
        get_run_tmp_file.tmpdir = TemporaryDirectory(prefix="codeflash_")
    return Path(get_run_tmp_file.tmpdir.name) / file_path


def module_name_from_file_path(
    file_path: Path, project_root_path: Path, *, traverse_up: bool = False
) -> str:  # moved from codeflash/code_utils/code_utils.py
    try:
        relative_path = file_path.relative_to(project_root_path)
        return relative_path.with_suffix("").as_posix().replace("/", ".")
    except ValueError:
        if traverse_up:
            parent = file_path.parent
            while parent not in (project_root_path, parent.parent):
                try:
                    relative_path = file_path.relative_to(parent)
                    return relative_path.with_suffix("").as_posix().replace("/", ".")
                except ValueError:
                    parent = parent.parent
        msg = f"File {file_path} is not within the project root {project_root_path}."
        raise ValueError(msg)  # noqa: B904


def _extract_class_name_tracer(frame_locals: dict[str, Any]) -> str | None:
    try:
        self_arg = frame_locals.get("self")
        if self_arg is not None:
            try:
                return self_arg.__class__.__name__
            except (AttributeError, Exception):
                cls_arg = frame_locals.get("cls")
                if cls_arg is not None:
                    with contextlib.suppress(AttributeError, Exception):
                        return cls_arg.__name__
        else:
            cls_arg = frame_locals.get("cls")
            if cls_arg is not None:
                with contextlib.suppress(AttributeError, Exception):
                    return cls_arg.__name__
    except Exception:
        return None
    return None


def _get_module_name_cf_tracer(frame: FrameType | None) -> str:
    try:
        test_module = inspect.getmodule(frame)
    except Exception:
        test_module = None

    if test_module is not None:
        module_name = getattr(test_module, "__name__", None)
        if module_name is not None:
            return module_name

    if frame is not None:
        return frame.f_globals.get("__name__", "unknown_module")
    return "unknown_module"


@lru_cache(maxsize=32)
def extract_test_context_from_frame(tests_project_root: Path) -> tuple[str, str, str]:
    frame = inspect.currentframe()
    try:
        frames_info = []
        potential_tests = []

        # First pass: collect all frame information
        if frame is not None:
            frame = frame.f_back

        while frame is not None:
            try:
                function_name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                filename_path = Path(filename)
                frame_locals = frame.f_locals
                test_module_name = module_name_from_file_path(filename_path, tests_project_root)
                class_name = _extract_class_name_tracer(frame_locals)

                frames_info.append(
                    {
                        "function_name": function_name,
                        "filename_path": filename_path,
                        "frame_locals": frame_locals,
                        "test_module_name": test_module_name,
                        "class_name": class_name,
                        "frame": frame,
                    }
                )

            except Exception:  # noqa: S112
                continue

            frame = frame.f_back
        # Second pass: analyze frames with full context
        test_class_candidates = []
        for frame_info in frames_info:
            function_name = frame_info["function_name"]
            filename_path = frame_info["filename_path"]
            frame_locals = frame_info["frame_locals"]
            test_module_name = frame_info["test_module_name"]
            class_name = frame_info["class_name"]
            frame_obj = frame_info["frame"]

            # Keep track of test classes
            if class_name and (
                class_name.startswith("Test") or class_name.endswith("Test") or "test" in class_name.lower()
            ) and not class_name.startswith(("Pytest", "_Pytest")):
                test_class_candidates.append((class_name, test_module_name))

        # Now process frames again looking for test functions with full candidates list
        # Collect all test functions to prioritize outer ones over nested ones
        test_functions = []
        for frame_info in frames_info:
            function_name = frame_info["function_name"]
            filename_path = frame_info["filename_path"]
            frame_locals = frame_info["frame_locals"]
            test_module_name = frame_info["test_module_name"]
            class_name = frame_info["class_name"]
            frame_obj = frame_info["frame"]

            # Collect test functions
            if function_name.startswith("test_"):
                test_class_name = class_name or None
                if not test_class_name and test_class_candidates:
                    test_class_name = test_class_candidates[0][0]
                test_functions.append((test_module_name, test_class_name, function_name))

        if test_functions:
            for test_func in test_functions:
                if test_func[1]:  # has non-empty class_name
                    return test_func
            return test_functions[-1]

        # If no direct test functions found, look for other test patterns
        for frame_info in frames_info:
            function_name = frame_info["function_name"]
            filename_path = frame_info["filename_path"]
            frame_locals = frame_info["frame_locals"]
            test_module_name = frame_info["test_module_name"]
            class_name = frame_info["class_name"]
            frame_obj = frame_info["frame"]

            # Test file/module detection
            if (
                frame_obj.f_globals.get("__name__", "").startswith("test_")
                or filename_path.stem.startswith("test_")
                or "test" in filename_path.parts
            ):
                if class_name and (
                    class_name.startswith("Test") or class_name.endswith("Test") or "test" in class_name.lower()
                ) and not class_name.startswith(("Pytest", "_Pytest")):
                    potential_tests.append((test_module_name, class_name, function_name))
                elif "test" in test_module_name or filename_path.stem.startswith("test_"):
                    best_class = test_class_candidates[0][0] if test_class_candidates else None
                    potential_tests.append((test_module_name, best_class, function_name))

            if (
                (
                    function_name in ["runTest", "_runTest", "run", "_testMethodName"]
                    or "pytest" in str(frame_obj.f_globals.get("__file__", ""))
                    or "unittest" in str(frame_obj.f_globals.get("__file__", ""))
                )
                and class_name
                and (class_name.startswith("Test") or "test" in class_name.lower())
                and not class_name.startswith(("Pytest", "_Pytest"))
            ):
                test_method = function_name
                if "self" in frame_locals:
                    with contextlib.suppress(AttributeError, TypeError):
                        test_method = getattr(frame_locals["self"], "_testMethodName", function_name)
                potential_tests.append((test_module_name, class_name, test_method))

        if potential_tests:
            for test_module, test_class, test_func in potential_tests:
                if test_func.startswith("test_"):
                    return test_module, test_class, test_func
            return potential_tests[0]

        raise RuntimeError("No test function found in call stack")
    finally:
        del frame


def codeflash_behavior_async(*, tests_project_root: Path) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            loop = asyncio.get_running_loop()
            function_name = func.__name__
            line_id = f"{func.__name__}_{func.__code__.co_firstlineno}"
            loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])

            test_module_name, test_class_name, test_name = extract_test_context_from_frame(tests_project_root)

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

    return decorator


def codeflash_performance_async(*, tests_project_root: Path) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            loop = asyncio.get_running_loop()
            function_name = func.__name__
            line_id = f"{func.__name__}_{func.__code__.co_firstlineno}"
            loop_index = int(os.environ["CODEFLASH_LOOP_INDEX"])

            test_module_name, test_class_name, test_name = extract_test_context_from_frame(tests_project_root)

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

    return decorator
