from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import libcst as cst

from codeflash.models.models import TestingMode
from codeflash_python.code_utils.formatter import sort_imports

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_core.models import FunctionToOptimize

logger = logging.getLogger("codeflash_python")


class AsyncDecoratorAdder(cst.CSTTransformer):
    """Transformer that adds async decorator to async function definitions."""

    def __init__(self, function: FunctionToOptimize, mode: TestingMode = TestingMode.BEHAVIOR) -> None:
        """Initialize the transformer.

        Args:
        ----
            function: The FunctionToOptimize object representing the target async function.
            mode: The testing mode to determine which decorator to apply.

        """
        super().__init__()
        self.function = function
        self.mode = mode
        self.qualified_name_parts = function.qualified_name.split(".")
        self.context_stack = []
        self.added_decorator = False

        # Choose decorator based on mode
        if mode == TestingMode.BEHAVIOR:
            self.decorator_name = "codeflash_behavior_async"
        elif mode == TestingMode.CONCURRENCY:
            self.decorator_name = "codeflash_concurrency_async"
        else:
            self.decorator_name = "codeflash_performance_async"

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        # Track when we enter a class
        self.context_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        # Pop the context when we leave a class
        self.context_stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        # Track when we enter a function
        self.context_stack.append(node.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Check if this is an async function and matches our target
        if original_node.asynchronous is not None and self.context_stack == self.qualified_name_parts:
            # Check if the decorator is already present
            has_decorator = any(self.is_target_decorator(decorator.decorator) for decorator in original_node.decorators)  # type: ignore[invalid-argument-type]

            # Only add the decorator if it's not already there
            if not has_decorator:
                new_decorator = cst.Decorator(decorator=cst.Name(value=self.decorator_name))

                # Add our new decorator to the existing decorators
                updated_decorators = [new_decorator, *list(updated_node.decorators)]
                updated_node = updated_node.with_changes(decorators=tuple(updated_decorators))
                self.added_decorator = True

        # Pop the context when we leave a function
        self.context_stack.pop()
        return updated_node

    def is_target_decorator(self, decorator_node: cst.Name | cst.Attribute | cst.Call) -> bool:
        """Check if a decorator matches our target decorator name."""
        if isinstance(decorator_node, cst.Name):
            return decorator_node.value in {
                "codeflash_trace_async",
                "codeflash_behavior_async",
                "codeflash_performance_async",
                "codeflash_concurrency_async",
            }
        if isinstance(decorator_node, cst.Call) and isinstance(decorator_node.func, cst.Name):
            return decorator_node.func.value in {
                "codeflash_trace_async",
                "codeflash_behavior_async",
                "codeflash_performance_async",
                "codeflash_concurrency_async",
            }
        return False


ASYNC_HELPER_INLINE_CODE = """import asyncio
import gc
import os
import sqlite3
import time
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory

import dill as pickle


def get_run_tmp_file(file_path):
    if not hasattr(get_run_tmp_file, "tmpdir"):
        get_run_tmp_file.tmpdir = TemporaryDirectory(prefix="codeflash_")
    return Path(get_run_tmp_file.tmpdir.name) / file_path


def extract_test_context_from_env():
    test_module = os.environ["CODEFLASH_TEST_MODULE"]
    test_class = os.environ.get("CODEFLASH_TEST_CLASS", None)
    test_function = os.environ["CODEFLASH_TEST_FUNCTION"]
    if test_module and test_function:
        return (test_module, test_class if test_class else None, test_function)
    raise RuntimeError(
        "Test context environment variables not set - ensure tests are run through codeflash test runner"
    )


def codeflash_behavior_async(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
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
        class_prefix = (test_class_name + ".") if test_class_name else ""
        test_stdout_tag = f"{test_module_name}:{class_prefix}{test_name}:{function_name}:{loop_index}:{invocation_id}"
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
            ret = func(*args, **kwargs)
            counter = loop.time()
            return_value = await ret
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
                "function_call",
            ),
        )
        codeflash_con.commit()
        codeflash_con.close()
        if exception:
            raise exception
        return return_value
    return async_wrapper


def codeflash_performance_async(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
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
        class_prefix = (test_class_name + ".") if test_class_name else ""
        test_stdout_tag = f"{test_module_name}:{class_prefix}{test_name}:{function_name}:{loop_index}:{invocation_id}"
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


def codeflash_concurrency_async(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        function_name = func.__name__
        concurrency_factor = int(os.environ.get("CODEFLASH_CONCURRENCY_FACTOR", "10"))
        test_module_name = os.environ.get("CODEFLASH_TEST_MODULE", "")
        test_class_name = os.environ.get("CODEFLASH_TEST_CLASS", "")
        test_function = os.environ.get("CODEFLASH_TEST_FUNCTION", "")
        loop_index = os.environ.get("CODEFLASH_LOOP_INDEX", "0")
        gc.disable()
        try:
            seq_start = time.perf_counter_ns()
            for _ in range(concurrency_factor):
                result = await func(*args, **kwargs)
            sequential_time = time.perf_counter_ns() - seq_start
        finally:
            gc.enable()
        gc.disable()
        try:
            conc_start = time.perf_counter_ns()
            tasks = [func(*args, **kwargs) for _ in range(concurrency_factor)]
            await asyncio.gather(*tasks)
            concurrent_time = time.perf_counter_ns() - conc_start
        finally:
            gc.enable()
        tag = f"{test_module_name}:{test_class_name}:{test_function}:{function_name}:{loop_index}"
        print(f"!@######CONC:{tag}:{sequential_time}:{concurrent_time}:{concurrency_factor}######@!")
        return result
    return async_wrapper
"""

ASYNC_HELPER_FILENAME = "codeflash_async_wrapper.py"


def get_decorator_name_for_mode(mode: TestingMode) -> str:
    if mode == TestingMode.BEHAVIOR:
        return "codeflash_behavior_async"
    if mode == TestingMode.CONCURRENCY:
        return "codeflash_concurrency_async"
    return "codeflash_performance_async"


def write_async_helper_file(target_dir: Path) -> Path:
    """Write the async decorator helper file to the target directory."""
    helper_path = target_dir / ASYNC_HELPER_FILENAME
    if not helper_path.exists():
        helper_path.write_text(ASYNC_HELPER_INLINE_CODE, "utf-8")
    return helper_path


def add_async_decorator_to_function(
    source_path: Path,
    function: FunctionToOptimize,
    mode: TestingMode = TestingMode.BEHAVIOR,
    project_root: Path | None = None,
) -> bool:
    """Add async decorator to an async function definition and write back to file.

    Writes a helper file containing the decorator implementation to project_root (or source directory
    as fallback) and adds a standard import + decorator to the source file.

    """
    if not function.is_async:
        return False

    try:
        with source_path.open(encoding="utf8") as f:
            source_code = f.read()

        module = cst.parse_module(source_code)

        # Add the decorator to the function
        decorator_transformer = AsyncDecoratorAdder(function, mode)
        module = module.visit(decorator_transformer)

        if decorator_transformer.added_decorator:
            # Write the helper file to project_root (on sys.path) or source dir as fallback
            helper_dir = project_root if project_root is not None else source_path.parent
            write_async_helper_file(helper_dir)
            # Add the import via CST so sort_imports can place it correctly
            decorator_name = get_decorator_name_for_mode(mode)
            import_node = cst.parse_statement(f"from codeflash_async_wrapper import {decorator_name}")
            module = module.with_changes(body=[import_node, *list(module.body)])

        modified_code = sort_imports(code=module.code, float_to_top=True)
    except Exception as e:
        logger.exception("Error adding async decorator to function %s: %s", function.qualified_name, e)
        return False
    else:
        if decorator_transformer.added_decorator:
            with source_path.open("w", encoding="utf8") as f:
                f.write(modified_code)
            logger.debug("Applied async %s instrumentation to %s", mode.value, source_path)
            return True
        return False
