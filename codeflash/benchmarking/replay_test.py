from __future__ import annotations

import re
import sqlite3
import textwrap
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import isort

from codeflash.cli_cmds.console import logger
from codeflash.discovery.functions_to_optimize import inspect_top_level_functions_or_methods
from codeflash.verification.verification_utils import get_test_file_path

if TYPE_CHECKING:
    from collections.abc import Generator

benchmark_context_cleaner = re.compile(r"[^a-zA-Z0-9_]+")


def get_next_arg_and_return(
    trace_file: str,
    benchmark_function_name: str,
    function_name: str,
    file_path: str,
    class_name: str | None = None,
    num_to_get: int = 256,
) -> Generator[Any]:
    db = sqlite3.connect(trace_file)
    cur = db.cursor()
    limit = num_to_get

    if class_name is not None:
        cursor = cur.execute(
            "SELECT * FROM benchmark_function_timings WHERE benchmark_function_name = ? AND function_name = ? AND file_path = ? AND class_name = ? LIMIT ?",
            (benchmark_function_name, function_name, file_path, class_name, limit),
        )
    else:
        cursor = cur.execute(
            "SELECT * FROM benchmark_function_timings WHERE benchmark_function_name = ? AND function_name = ? AND file_path = ? AND class_name = '' LIMIT ?",
            (benchmark_function_name, function_name, file_path, limit),
        )

    while (val := cursor.fetchone()) is not None:
        yield val[9], val[10]  # pickled_args, pickled_kwargs


def get_function_alias(module: str, function_name: str) -> str:
    return "_".join(module.split(".")) + "_" + function_name


def get_unique_test_name(module: str, function_name: str, benchmark_name: str, class_name: str | None = None) -> str:
    clean_benchmark = benchmark_context_cleaner.sub("_", benchmark_name).strip("_")

    base_alias = get_function_alias(module, function_name)
    if class_name:
        class_alias = get_function_alias(module, class_name)
        return f"{class_alias}_{function_name}_{clean_benchmark}"
    return f"{base_alias}_{clean_benchmark}"


def create_trace_replay_test_code(
    trace_file: str,
    functions_data: list[dict[str, Any]],
    test_framework: str = "pytest",
    max_run_count=256,  # noqa: ANN001
) -> str:
    """Create a replay test for functions based on trace data.

    Args:
    ----
        trace_file: Path to the SQLite database file
        functions_data: List of dictionaries with function info extracted from DB
        test_framework: 'pytest' or 'unittest'
        max_run_count: Maximum number of runs to include in the test

    Returns:
    -------
        A string containing the test code

    """
    assert test_framework in ("pytest", "unittest")
    # Build imports
    imports = [
        "from codeflash.picklepatch.pickle_patcher import PicklePatcher as pickle",
        "import unittest" if test_framework == "unittest" else "",
        "from codeflash.benchmarking.replay_test import get_next_arg_and_return",
    ]

    function_imports = []
    get_alias = get_function_alias  # avoid attribute lookup in loop

    append_func_import = function_imports.append

    # BUILD function imports (string join at the end!)
    for func in functions_data:
        module_name = func["module_name"]
        function_name = func["function_name"]
        class_name = func.get("class_name", "")
        if class_name:
            # Only alias imports once per unique combo (rely on LRU cache)
            append_func_import(f"from {module_name} import {class_name} as {get_alias(module_name, class_name)}")
        else:
            append_func_import(f"from {module_name} import {function_name} as {get_alias(module_name, function_name)}")

    imports.append("\n".join(function_imports))

    # Build sorted functions_to_optimize (skip __init__)
    functions_to_optimize = sorted(
        {func["function_name"] for func in functions_data if func["function_name"] != "__init__"}
    )
    metadata = f'functions = {functions_to_optimize}\ntrace_file_path = r"{trace_file}"\n'

    # Pointer to templates
    templates = {
        "function": _test_function_body,
        "method": _test_method_body,
        "classmethod": _test_class_method_body,
        "staticmethod": _test_static_method_body,
    }

    # Setup for main test generation
    tests = []

    if test_framework == "unittest":
        test_class_header = "\nclass TestTracedFunctions(unittest.TestCase):\n"
        tests.append(test_class_header)
        indent = "        "
        self_arg = "self"
    else:
        indent = "    "
        self_arg = ""

    # Inline access for performance
    get_unique_name = get_unique_test_name

    for func in functions_data:
        module_name = func["module_name"]
        function_name = func["function_name"]
        class_name = func.get("class_name")
        file_path = func["file_path"]
        benchmark_function_name = func["benchmark_function_name"]
        function_properties = func["function_properties"]

        if not class_name:
            alias = get_alias(module_name, function_name)
            test_body = templates["function"].format(
                benchmark_function_name=benchmark_function_name,
                orig_function_name=function_name,
                function_name=alias,
                file_path=file_path,
                max_run_count=max_run_count,
            )
        else:
            class_name_alias = get_alias(module_name, class_name)
            alias = get_alias(module_name, class_name + "_" + function_name)
            filter_variables = ""
            method_name = "." + function_name if function_name != "__init__" else ""
            if getattr(function_properties, "is_classmethod", False):
                test_body = templates["classmethod"].format(
                    benchmark_function_name=benchmark_function_name,
                    orig_function_name=function_name,
                    file_path=file_path,
                    class_name=class_name,
                    class_name_alias=class_name_alias,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )
            elif getattr(function_properties, "is_staticmethod", False):
                test_body = templates["staticmethod"].format(
                    benchmark_function_name=benchmark_function_name,
                    orig_function_name=function_name,
                    file_path=file_path,
                    class_name=class_name,
                    class_name_alias=class_name_alias,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )
            else:
                test_body = templates["method"].format(
                    benchmark_function_name=benchmark_function_name,
                    orig_function_name=function_name,
                    file_path=file_path,
                    class_name=class_name,
                    class_name_alias=class_name_alias,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )
        # Indent the block only once
        formatted_test_body = textwrap.indent(test_body, indent)
        unique_test_name = get_unique_name(module_name, function_name, benchmark_function_name, class_name)
        # Compose function definition
        if test_framework == "unittest":
            tests.append(f"    def test_{unique_test_name}({self_arg}):\n{formatted_test_body}\n")
        else:
            tests.append(f"def test_{unique_test_name}({self_arg}):\n{formatted_test_body}\n")

    # Final string build (list join for speed)
    return "\n".join(imports) + "\n" + metadata + "\n" + "".join(tests)


def generate_replay_test(
    trace_file_path: Path, output_dir: Path, test_framework: str = "pytest", max_run_count: int = 100
) -> int:
    """Generate multiple replay tests from the traced function calls, grouped by benchmark.

    Args:
    ----
        trace_file_path: Path to the SQLite database file
        output_dir: Directory to write the generated tests (if None, only returns the code)
        test_framework: 'pytest' or 'unittest'
        max_run_count: Maximum number of runs to include per function

    Returns:
    -------
        Dictionary mapping benchmark names to generated test code

    """
    count = 0
    try:
        # Connect to the database
        conn = sqlite3.connect(trace_file_path.as_posix())
        cursor = conn.cursor()

        # Get distinct benchmark file paths
        cursor.execute("SELECT DISTINCT benchmark_module_path FROM benchmark_function_timings")
        benchmark_files = cursor.fetchall()

        # Generate a test for each benchmark file
        for benchmark_file in benchmark_files:
            benchmark_module_path = benchmark_file[0]
            # Get all benchmarks and functions associated with this file path
            cursor.execute(
                "SELECT DISTINCT benchmark_function_name, function_name, class_name, module_name, file_path, benchmark_line_number FROM benchmark_function_timings "
                "WHERE benchmark_module_path = ?",
                (benchmark_module_path,),
            )

            functions_data = []
            for row in cursor.fetchall():
                benchmark_function_name, function_name, class_name, module_name, file_path, benchmark_line_number = row
                # Add this function to our list
                functions_data.append(
                    {
                        "function_name": function_name,
                        "class_name": class_name,
                        "file_path": file_path,
                        "module_name": module_name,
                        "benchmark_function_name": benchmark_function_name,
                        "benchmark_module_path": benchmark_module_path,
                        "benchmark_line_number": benchmark_line_number,
                        "function_properties": inspect_top_level_functions_or_methods(
                            file_name=Path(file_path), function_or_method_name=function_name, class_name=class_name
                        ),
                    }
                )

            if not functions_data:
                logger.info(f"No benchmark test functions found in {benchmark_module_path}")
                continue
            # Generate the test code for this benchmark
            test_code = create_trace_replay_test_code(
                trace_file=trace_file_path.as_posix(),
                functions_data=functions_data,
                test_framework=test_framework,
                max_run_count=max_run_count,
            )
            test_code = isort.code(test_code)
            output_file = get_test_file_path(
                test_dir=Path(output_dir), function_name=benchmark_module_path, test_type="replay"
            )
            # Write test code to file, parents = true
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file.write_text(test_code, "utf-8")
            count += 1

        conn.close()
    except Exception as e:
        logger.info(f"Error generating replay tests: {e}")

    return count


_test_function_body = textwrap.dedent(
    """\
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="{benchmark_function_name}", function_name="{orig_function_name}", file_path=r"{file_path}", num_to_get={max_run_count}):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl)
        ret = {function_name}(*args, **kwargs)
    """
)

_test_method_body = textwrap.dedent(
    """\
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="{benchmark_function_name}", function_name="{orig_function_name}", file_path=r"{file_path}", class_name="{class_name}", num_to_get={max_run_count}):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl){filter_variables}
        function_name = "{orig_function_name}"
        if not args:
            raise ValueError("No arguments provided for the method.")
        if function_name == "__init__":
            ret = {class_name_alias}(*args[1:], **kwargs)
        else:
            ret = {class_name_alias}{method_name}(*args, **kwargs)
    """
)

_test_class_method_body = textwrap.dedent(
    """\
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="{benchmark_function_name}", function_name="{orig_function_name}", file_path=r"{file_path}", class_name="{class_name}", num_to_get={max_run_count}):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl){filter_variables}
        if not args:
            raise ValueError("No arguments provided for the method.")
        ret = {class_name_alias}{method_name}(*args[1:], **kwargs)
    """
)

_test_static_method_body = textwrap.dedent(
    """\
    for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, benchmark_function_name="{benchmark_function_name}", function_name="{orig_function_name}", file_path=r"{file_path}", class_name="{class_name}", num_to_get={max_run_count}):
        args = pickle.loads(args_pkl)
        kwargs = pickle.loads(kwargs_pkl){filter_variables}
        ret = {class_name_alias}{method_name}(*args, **kwargs)
    """
)
