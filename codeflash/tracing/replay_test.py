from __future__ import annotations

import sqlite3
import textwrap
from collections.abc import Generator
from typing import Any, Optional

from codeflash.discovery.functions_to_optimize import FunctionProperties, inspect_top_level_functions_or_methods
from codeflash.tracing.tracing_utils import FunctionModules


def get_next_arg_and_return(
    trace_file: str, function_name: str, file_name: str, class_name: Optional[str] = None, num_to_get: int = 25
) -> Generator[Any]:
    db = sqlite3.connect(trace_file)
    cur = db.cursor()
    limit = num_to_get
    if class_name is not None:
        cursor = cur.execute(
            "SELECT * FROM function_calls WHERE function = ? AND filename = ? AND classname = ? ORDER BY time_ns ASC LIMIT ?",
            (function_name, file_name, class_name, limit),
        )
    else:
        cursor = cur.execute(
            "SELECT * FROM function_calls WHERE function = ? AND filename = ? ORDER BY time_ns ASC LIMIT ?",
            (function_name, file_name, limit),
        )

    while (val := cursor.fetchone()) is not None:
        event_type = val[0]
        if event_type == "call":
            yield val[7]
        else:
            msg = "Invalid Trace event type"
            raise ValueError(msg)


def get_function_alias(module: str, function_name: str) -> str:
    return "_".join(module.split(".")) + "_" + function_name


def create_trace_replay_test(
    trace_file: str, functions: list[FunctionModules], test_framework: str = "pytest", max_run_count=100
) -> str:
    assert test_framework in ["pytest", "unittest"]

    imports = f"""import dill as pickle
{"import unittest" if test_framework == "unittest" else ""}
from codeflash.tracing.replay_test import get_next_arg_and_return
"""

    # TODO: Module can have "-" character if the module-root is ".". Need to handle that case
    function_properties: list[FunctionProperties] = [
        inspect_top_level_functions_or_methods(
            file_name=function.file_name,
            function_or_method_name=function.function_name,
            class_name=function.class_name,
            line_no=function.line_no,
        )
        for function in functions
    ]
    function_imports = []
    for function, function_property in zip(functions, function_properties):
        if not function_property.is_top_level:
            # can't be imported and run in the replay test
            continue
        if function_property.is_staticmethod:
            function_imports.append(
                f"from {function.module_name} import {function_property.staticmethod_class_name} as {get_function_alias(function.module_name, function_property.staticmethod_class_name)}"
            )
        elif function.class_name:
            function_imports.append(
                f"from {function.module_name} import {function.class_name} as {get_function_alias(function.module_name, function.class_name)}"
            )
        else:
            function_imports.append(
                f"from {function.module_name} import {function.function_name} as {get_function_alias(function.module_name, function.function_name)}"
            )

    imports += "\n".join(function_imports)
    functions_to_optimize = [function.function_name for function in functions if function.function_name != "__init__"]
    metadata = f"""functions = {functions_to_optimize}
trace_file_path = r"{trace_file}"
"""  # trace_file_path path is parsed with regex later, format is important
    test_function_body = textwrap.dedent(
        """\
        for arg_val_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="{orig_function_name}", file_name=r"{file_name}", num_to_get={max_run_count}):
            args = pickle.loads(arg_val_pkl)
            ret = {function_name}({args})
            """
    )
    test_class_method_body = textwrap.dedent(
        """\
        for arg_val_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="{orig_function_name}", file_name=r"{file_name}", class_name="{class_name}", num_to_get={max_run_count}):
            args = pickle.loads(arg_val_pkl){filter_variables}
            ret = {class_name_alias}{method_name}(**args)
            """
    )
    test_class_staticmethod_body = textwrap.dedent(
        """\
        for arg_val_pkl in get_next_arg_and_return(trace_file=trace_file_path, function_name="{orig_function_name}", file_name=r"{file_name}", num_to_get={max_run_count}):
            args = pickle.loads(arg_val_pkl){filter_variables}
            ret = {class_name_alias}{method_name}(**args)
            """
    )
    if test_framework == "unittest":
        self = "self"
        test_template = "\nclass TestTracedFunctions(unittest.TestCase):\n"
    else:
        test_template = ""
        self = ""
    for func, func_property in zip(functions, function_properties):
        if not func_property.is_top_level:
            # can't be imported and run in the replay test
            continue
        if func.class_name is None and not func_property.is_staticmethod:
            alias = get_function_alias(func.module_name, func.function_name)
            test_body = test_function_body.format(
                function_name=alias,
                file_name=func.file_name,
                orig_function_name=func.function_name,
                max_run_count=max_run_count,
                args="**args" if func_property.has_args else "",
            )
        elif func_property.is_staticmethod:
            class_name_alias = get_function_alias(func.module_name, func_property.staticmethod_class_name)
            alias = get_function_alias(
                func.module_name, func_property.staticmethod_class_name + "_" + func.function_name
            )
            method_name = "." + func.function_name if func.function_name != "__init__" else ""
            test_body = test_class_staticmethod_body.format(
                orig_function_name=func.function_name,
                file_name=func.file_name,
                class_name_alias=class_name_alias,
                method_name=method_name,
                max_run_count=max_run_count,
                filter_variables="",
            )
        else:
            class_name_alias = get_function_alias(func.module_name, func.class_name)
            alias = get_function_alias(func.module_name, func.class_name + "_" + func.function_name)

            if func_property.is_classmethod:
                filter_variables = '\n    args.pop("cls", None)'
            elif func.function_name == "__init__":
                filter_variables = '\n    args.pop("__class__", None)'
            else:
                filter_variables = ""
            method_name = "." + func.function_name if func.function_name != "__init__" else ""
            test_body = test_class_method_body.format(
                orig_function_name=func.function_name,
                file_name=func.file_name,
                class_name_alias=class_name_alias,
                class_name=func.class_name,
                method_name=method_name,
                max_run_count=max_run_count,
                filter_variables=filter_variables,
            )
        formatted_test_body = textwrap.indent(test_body, "        " if test_framework == "unittest" else "    ")

        test_template += "    " if test_framework == "unittest" else ""
        test_template += f"def test_{alias}({self}):\n{formatted_test_body}\n"

    return imports + "\n" + metadata + "\n" + test_template
