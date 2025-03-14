from __future__ import annotations

import sqlite3
import textwrap
from collections.abc import Generator
from typing import Any, Dict

import isort

from codeflash.cli_cmds.console import logger
from codeflash.discovery.functions_to_optimize import inspect_top_level_functions_or_methods
from codeflash.verification.verification_utils import get_test_file_path
from pathlib import Path

def get_next_arg_and_return(
        trace_file: str, function_name: str, file_name: str, class_name: str | None = None, num_to_get: int = 25
) -> Generator[Any]:
    db = sqlite3.connect(trace_file)
    cur = db.cursor()
    limit = num_to_get

    if class_name is not None:
        cursor = cur.execute(
            "SELECT * FROM function_calls WHERE function_name = ? AND file_name = ? AND class_name = ? ORDER BY time_ns ASC LIMIT ?",
            (function_name, file_name, class_name, limit),
        )
    else:
        cursor = cur.execute(
            "SELECT * FROM function_calls WHERE function_name = ? AND file_name = ? AND class_name = '' ORDER BY time_ns ASC LIMIT ?",
            (function_name, file_name, limit),
        )

    while (val := cursor.fetchone()) is not None:
        yield val[9], val[10]  # args and kwargs are at indices 7 and 8


def get_function_alias(module: str, function_name: str) -> str:
    return "_".join(module.split(".")) + "_" + function_name


def create_trace_replay_test_code(
        trace_file: str,
        functions_data: list[dict[str, Any]],
        test_framework: str = "pytest",
        max_run_count=100
) -> str:
    """Create a replay test for functions based on trace data.

    Args:
        trace_file: Path to the SQLite database file
        functions_data: List of dictionaries with function info extracted from DB
        test_framework: 'pytest' or 'unittest'
        max_run_count: Maximum number of runs to include in the test

    Returns:
        A string containing the test code

    """
    assert test_framework in ["pytest", "unittest"]

    imports = f"""import dill as pickle 
{"import unittest" if test_framework == "unittest" else ""}
from codeflash.benchmarking.replay_test import get_next_arg_and_return
"""

    function_imports = []
    for func in functions_data:
        module_name = func.get("module_name")
        function_name = func.get("function_name")
        class_name = func.get("class_name", "")
        if class_name:
            function_imports.append(
                f"from {module_name} import {class_name} as {get_function_alias(module_name, class_name)}"
            )
        else:
            function_imports.append(
                f"from {module_name} import {function_name} as {get_function_alias(module_name, function_name)}"
            )

    imports += "\n".join(function_imports)

    functions_to_optimize = [func.get("function_name") for func in functions_data
                             if func.get("function_name") != "__init__"]
    metadata = f"""functions = {functions_to_optimize}
trace_file_path = r"{trace_file}"
"""

    # Templates for different types of tests
    test_function_body = textwrap.dedent(
        """\
        for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="{orig_function_name}", file_name=r"{file_name}", num_to_get={max_run_count}):
            args = pickle.loads(args_pkl)
            kwargs = pickle.loads(kwargs_pkl)
            ret = {function_name}(*args, **kwargs)
            """
    )

    test_method_body = textwrap.dedent(
        """\
        for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="{orig_function_name}", file_name=r"{file_name}", class_name="{class_name}", num_to_get={max_run_count}):
            args = pickle.loads(args_pkl)
            kwargs = pickle.loads(kwargs_pkl){filter_variables}
            function_name = "{orig_function_name}"
            if not args:
                raise ValueError("No arguments provided for the method.")
            if function_name == "__init__":
                ret = {class_name_alias}(*args[1:], **kwargs)
            else:
                instance = args[0] # self
                ret = instance{method_name}(*args[1:], **kwargs)
            """)

    test_class_method_body = textwrap.dedent(
        """\
        for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="{orig_function_name}", file_name=r"{file_name}", class_name="{class_name}", num_to_get={max_run_count}):
            args = pickle.loads(args_pkl)
            kwargs = pickle.loads(kwargs_pkl){filter_variables}
            if not args:
                raise ValueError("No arguments provided for the method.")
            ret = {class_name_alias}{method_name}(*args[1:], **kwargs)
            """
    )
    test_static_method_body = textwrap.dedent(
        """\
        for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="{orig_function_name}", file_name=r"{file_name}", class_name="{class_name}", num_to_get={max_run_count}):
            args = pickle.loads(args_pkl)
            kwargs = pickle.loads(kwargs_pkl){filter_variables}
            ret = {class_name_alias}{method_name}(*args, **kwargs)
            """
    )

    if test_framework == "unittest":
        self = "self"
        test_template = "\nclass TestTracedFunctions(unittest.TestCase):\n"
    else:
        test_template = ""
        self = ""

    for func in functions_data:
        module_name = func.get("module_name")
        function_name = func.get("function_name")
        class_name = func.get("class_name")
        file_name = func.get("file_name")
        function_properties = func.get("function_properties")
        print(f"Class: {class_name}, Function: {function_name}")
        print(function_properties)
        if not class_name:
            alias = get_function_alias(module_name, function_name)
            test_body = test_function_body.format(
                function_name=alias,
                file_name=file_name,
                orig_function_name=function_name,
                max_run_count=max_run_count,
            )
        else:
            class_name_alias = get_function_alias(module_name, class_name)
            alias = get_function_alias(module_name, class_name + "_" + function_name)

            filter_variables = ""
            # filter_variables = '\n    args.pop("cls", None)'
            method_name = "." + function_name if function_name != "__init__" else ""
            if function_properties.is_classmethod:
                test_body = test_class_method_body.format(
                    orig_function_name=function_name,
                    file_name=file_name,
                    class_name_alias=class_name_alias,
                    class_name=class_name,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )
            elif function_properties.is_staticmethod:
                test_body = test_static_method_body.format(
                    orig_function_name=function_name,
                    file_name=file_name,
                    class_name_alias=class_name_alias,
                    class_name=class_name,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )
            else:
                test_body = test_method_body.format(
                    orig_function_name=function_name,
                    file_name=file_name,
                    class_name_alias=class_name_alias,
                    class_name=class_name,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )

        formatted_test_body = textwrap.indent(test_body, "        " if test_framework == "unittest" else "    ")

        test_template += "    " if test_framework == "unittest" else ""
        test_template += f"def test_{alias}({self}):\n{formatted_test_body}\n"

    return imports + "\n" + metadata + "\n" + test_template

def generate_replay_test(trace_file_path: Path, output_dir: Path, test_framework: str = "pytest", max_run_count: int = 100) -> None:
    """Generate multiple replay tests from the traced function calls, grouping by benchmark name.

    Args:
        trace_file_path: Path to the SQLite database file
        output_dir: Directory to write the generated tests (if None, only returns the code)
        project_root: Root directory of the project for module imports
        test_framework: 'pytest' or 'unittest'
        max_run_count: Maximum number of runs to include per function

    Returns:
        Dictionary mapping benchmark names to generated test code

    """
    try:
        # Connect to the database
        conn = sqlite3.connect(trace_file_path.as_posix())
        cursor = conn.cursor()

        # Get distinct benchmark names
        cursor.execute(
            "SELECT DISTINCT benchmark_function_name, benchmark_file_name FROM function_calls"
        )
        benchmarks = cursor.fetchall()

        # Generate a test for each benchmark
        for benchmark in benchmarks:
            benchmark_function_name, benchmark_file_name = benchmark
            # Get functions associated with this benchmark
            cursor.execute(
                "SELECT DISTINCT function_name, class_name, module_name, file_name, benchmark_line_number FROM function_calls "
                "WHERE benchmark_function_name = ? AND benchmark_file_name = ?",
                (benchmark_function_name, benchmark_file_name)
            )

            functions_data = []
            for func_row in cursor.fetchall():
                function_name, class_name, module_name, file_name, benchmark_line_number = func_row

                # Add this function to our list
                functions_data.append({
                    "function_name": function_name,
                    "class_name": class_name,
                    "file_name": file_name,
                    "module_name": module_name,
                    "benchmark_function_name": benchmark_function_name,
                    "benchmark_file_name": benchmark_file_name,
                    "benchmark_line_number": benchmark_line_number,
                    "function_properties": inspect_top_level_functions_or_methods(
                            file_name=file_name,
                            function_or_method_name=function_name,
                            class_name=class_name,
                        )
                })

            if not functions_data:
                print(f"No functions found for benchmark {benchmark_function_name} in {benchmark_file_name}")
                continue

            # Generate the test code for this benchmark
            test_code = create_trace_replay_test_code(
                trace_file=trace_file_path.as_posix(),
                functions_data=functions_data,
                test_framework=test_framework,
                max_run_count=max_run_count,
            )
            test_code = isort.code(test_code)

            # Write to file if requested
            if output_dir:
                output_file = get_test_file_path(
                    test_dir=Path(output_dir), function_name=f"{benchmark_file_name[5:]}_{benchmark_function_name}", test_type="replay"
                )
                with open(output_file, 'w') as f:
                    f.write(test_code)
                print(f"Replay test for benchmark `{benchmark_function_name}` in {benchmark_file_name} written to {output_file}")

        conn.close()

    except Exception as e:
        print(f"Error generating replay tests: {e}")
