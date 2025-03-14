from __future__ import annotations

import sqlite3
import textwrap
from collections.abc import Generator
from typing import Any, Dict


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
            "SELECT * FROM function_calls WHERE function_name = ? AND file_name = ? ORDER BY time_ns ASC LIMIT ?",
            (function_name, file_name, limit),
        )

    while (val := cursor.fetchone()) is not None:
        yield val[8], val[9]  # args and kwargs are at indices 7 and 8


def get_function_alias(module: str, function_name: str) -> str:
    return "_".join(module.split(".")) + "_" + function_name


def create_trace_replay_test(
        trace_file: str,
        functions_data: list[Dict[str, Any]],
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

    test_class_method_body = textwrap.dedent(
        """\
        for args_pkl, kwargs_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="{orig_function_name}", file_name=r"{file_name}", class_name="{class_name}", num_to_get={max_run_count}):
            args = pickle.loads(args_pkl)
            kwargs = pickle.loads(kwargs_pkl){filter_variables}
            ret = {class_name_alias}{method_name}(**args, **kwargs)
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
            method_name = "." + function_name if function_name != "__init__" else ""
            test_body = test_class_method_body.format(
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
