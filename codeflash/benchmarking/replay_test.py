from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import isort

from codeflash.cli_cmds.console import logger
from codeflash.discovery.functions_to_optimize import inspect_top_level_functions_or_methods
from codeflash.verification.verification_utils import get_test_file_path

if TYPE_CHECKING:
    from collections.abc import Generator


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
    assert test_framework in ["pytest", "unittest"]

    # Precompute all needed values up-front for efficiency
    unittest_import = "import unittest" if test_framework == "unittest" else ""
    imports = (
        "from codeflash.picklepatch.pickle_patcher import PicklePatcher as pickle\n"
        f"{unittest_import}\n"
        "from codeflash.benchmarking.replay_test import get_next_arg_and_return\n"
    )

    function_imports = []
    functions_to_optimize = set()

    # Collect imports and test function names in one pass:
    for func in functions_data:
        module_name = func["module_name"]
        function_name = func["function_name"]
        class_name = func.get("class_name")
        if class_name:
            alias = get_function_alias(module_name, class_name)
            function_imports.append(f"from {module_name} import {class_name} as {alias}")
        else:
            alias = get_function_alias(module_name, function_name)
            function_imports.append(f"from {module_name} import {function_name} as {alias}")
        if function_name != "__init__":
            functions_to_optimize.add(function_name)
    imports += "\n".join(function_imports)

    metadata = f'functions = {sorted(functions_to_optimize)}\ntrace_file_path = r"{trace_file}"\n'

    # Templates, dedented once for speed
    test_function_body = (
        "for args_pkl, kwargs_pkl in get_next_arg_and_return("
        'trace_file=trace_file_path, benchmark_function_name="{benchmark_function_name}", '
        'function_name="{orig_function_name}", file_path=r"{file_path}", num_to_get={max_run_count}):\n'
        "    args = pickle.loads(args_pkl)\n"
        "    kwargs = pickle.loads(kwargs_pkl)\n"
        "    ret = {function_name}(*args, **kwargs)\n"
    )
    test_method_body = (
        "for args_pkl, kwargs_pkl in get_next_arg_and_return("
        'trace_file=trace_file_path, benchmark_function_name="{benchmark_function_name}", '
        'function_name="{orig_function_name}", file_path=r"{file_path}", class_name="{class_name}", num_to_get={max_run_count}):\n'
        "    args = pickle.loads(args_pkl)\n"
        "    kwargs = pickle.loads(kwargs_pkl){filter_variables}\n"
        '    function_name = "{orig_function_name}"\n'
        "    if not args:\n"
        '        raise ValueError("No arguments provided for the method.")\n'
        '    if function_name == "__init__":\n'
        "        ret = {class_name_alias}(*args[1:], **kwargs)\n"
        "    else:\n"
        "        ret = {class_name_alias}{method_name}(*args, **kwargs)\n"
    )
    test_class_method_body = (
        "for args_pkl, kwargs_pkl in get_next_arg_and_return("
        'trace_file=trace_file_path, benchmark_function_name="{benchmark_function_name}", '
        'function_name="{orig_function_name}", file_path=r"{file_path}", class_name="{class_name}", num_to_get={max_run_count}):\n'
        "    args = pickle.loads(args_pkl)\n"
        "    kwargs = pickle.loads(kwargs_pkl){filter_variables}\n"
        "    if not args:\n"
        '        raise ValueError("No arguments provided for the method.")\n'
        "    ret = {class_name_alias}{method_name}(*args[1:], **kwargs)\n"
    )
    test_static_method_body = (
        "for args_pkl, kwargs_pkl in get_next_arg_and_return("
        'trace_file=trace_file_path, benchmark_function_name="{benchmark_function_name}", '
        'function_name="{orig_function_name}", file_path=r"{file_path}", class_name="{class_name}", num_to_get={max_run_count}):\n'
        "    args = pickle.loads(args_pkl)\n"
        "    kwargs = pickle.loads(kwargs_pkl){filter_variables}\n"
        "    ret = {class_name_alias}{method_name}(*args, **kwargs)\n"
    )

    if test_framework == "unittest":
        self_arg = "self"
        test_header = "\nclass TestTracedFunctions(unittest.TestCase):\n"
        def_indent = "    "
        body_indent = "        "
    else:
        self_arg = ""
        test_header = ""
        def_indent = ""
        body_indent = "    "

    # String builder technique for fast test template construction
    test_template_lines = [test_header]
    append = test_template_lines.append  # local variable for speed

    for func in functions_data:
        module_name = func["module_name"]
        function_name = func["function_name"]
        class_name = func.get("class_name")
        file_path = func["file_path"]
        benchmark_function_name = func["benchmark_function_name"]
        function_properties = func["function_properties"]

        if not class_name:
            alias = get_function_alias(module_name, function_name)
            test_body = test_function_body.format(
                benchmark_function_name=benchmark_function_name,
                orig_function_name=function_name,
                function_name=alias,
                file_path=file_path,
                max_run_count=max_run_count,
            )
        else:
            class_name_alias = get_function_alias(module_name, class_name)
            alias = get_function_alias(module_name, class_name + "_" + function_name)
            filter_variables = ""
            method_name = "." + function_name if function_name != "__init__" else ""
            if function_properties.is_classmethod:
                test_body = test_class_method_body.format(
                    benchmark_function_name=benchmark_function_name,
                    orig_function_name=function_name,
                    file_path=file_path,
                    class_name_alias=class_name_alias,
                    class_name=class_name,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )
            elif function_properties.is_staticmethod:
                test_body = test_static_method_body.format(
                    benchmark_function_name=benchmark_function_name,
                    orig_function_name=function_name,
                    file_path=file_path,
                    class_name_alias=class_name_alias,
                    class_name=class_name,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )
            else:
                test_body = test_method_body.format(
                    benchmark_function_name=benchmark_function_name,
                    orig_function_name=function_name,
                    file_path=file_path,
                    class_name_alias=class_name_alias,
                    class_name=class_name,
                    method_name=method_name,
                    max_run_count=max_run_count,
                    filter_variables=filter_variables,
                )

        # Manually indent for speed (no textwrap.indent)
        test_body_indented = "".join(
            body_indent + ln if ln else body_indent for ln in test_body.splitlines(keepends=True)
        )
        append(f"{def_indent}def test_{alias}({self_arg}):\n{test_body_indented}\n")

    # Final string concatenation
    return f"{imports}\n{metadata}\n{''.join(test_template_lines)}"


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
