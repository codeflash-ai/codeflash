import logging
import sqlite3
import textwrap
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple

from codeflash.discovery.functions_to_optimize import is_function_or_method_top_level
from codeflash.tracing.tracing_utils import FunctionModules


def get_next_arg_and_return(
    trace_file: str,
    function_name: str,
    file_name: str,
    class_name: Optional[str] = None,
    num_to_get: int = 3,
) -> Generator[Tuple[Any, Any], None, None]:
    db = sqlite3.connect(trace_file)
    cur = db.cursor()
    limit = num_to_get * 2 + 100  # we may have to get more than num_to_get*2 to get num_to_get valid pairs
    if class_name is not None:
        data = cur.execute(
            "SELECT * FROM events WHERE function = ? AND filename = ? AND classname = ? ORDER BY time_ns ASC LIMIT ?",
            (function_name, file_name, class_name, limit),
        ).fetchall()
    else:
        data = cur.execute(
            "SELECT * FROM events WHERE function = ? AND filename = ? ORDER BY time_ns ASC LIMIT ?",
            (function_name, file_name, limit),
        ).fetchall()

    counts = 0
    matched_arg_return: Dict[int, List[Any]] = defaultdict(list)
    for val in data:
        if counts >= num_to_get:
            break

        event_type, frame_address = val[0], val[5]
        if event_type == "call":
            matched_arg_return[frame_address].append(val[8])
            if len(matched_arg_return[frame_address]) > 1:
                logging.warning(
                    f"Pre-existing call to the function {function_name} with same frame address.",
                )
        elif event_type == "return":
            matched_arg_return[frame_address].append(val[7])
            arg_return_length = len(matched_arg_return[frame_address])
            if arg_return_length > 2:
                logging.warning(
                    f"Pre-existing return to the function {function_name} with same frame address.",
                )
            elif arg_return_length == 1:
                logging.warning(f"No call before return for {function_name}!")
            elif arg_return_length == 2:
                yield matched_arg_return[frame_address]
                counts += 1
                del matched_arg_return[frame_address]
        else:
            raise ValueError("Invalid Trace event type")


def get_function_alias(module: str, function_name: str) -> str:
    return "_".join(module.split(".")) + "_" + function_name


def create_trace_replay_test(
    trace_file: str,
    functions: List[FunctionModules],
    test_framework: str = "pytest",
    max_run_count=100,
) -> str:
    assert test_framework in ["pytest", "unittest"]

    imports = f"""import dill as pickle
import {test_framework}
from codeflash.tracing.replay_test import get_next_arg_and_return
from codeflash.verification.comparator import comparator
"""

    # TODO: Module can have "-" character if the module-root is ".". Need to handle that case
    function_imports = []
    for function in functions:
        if not is_function_or_method_top_level(
            file_name=function.file_name,
            function_or_method_name=function.function_name,
            class_name=function.class_name,
        ):
            # can't be imported and run in the replay test
            continue
        if function.class_name:
            function_imports.append(
                f"from {function.module_name} import {function.class_name} as {get_function_alias(function.module_name, function.class_name)}",
            )
        else:
            function_imports.append(
                f"from {function.module_name} import {function.function_name} as {get_function_alias(function.module_name, function.function_name)}",
            )

    imports += "\n".join(function_imports)
    test_function_body = textwrap.dedent(
        """\
        for arg_val_pkl, return_val_pkl in get_next_arg_and_return(trace_file=r'{trace_file}', function_name='{orig_function_name}', file_name=r'{file_name}', num_to_get={max_run_count}):
            args = pickle.loads(arg_val_pkl)
            traced_return_val = pickle.loads(return_val_pkl)
            ret = {function_name}(**args)
            """
        + (
            """self.assertTrue(comparator(traced_return_val, ret))
        """
            if test_framework == "unittest"
            else """assert comparator(traced_return_val, ret)
        """
        ),
    )
    test_class_method_body = textwrap.dedent(
        """\
        for arg_val_pkl, return_val_pkl in get_next_arg_and_return(trace_file=r'{trace_file}', function_name='{orig_function_name}', file_name=r'{file_name}', class_name='{class_name}', num_to_get={max_run_count}):
            args = pickle.loads(arg_val_pkl)
            traced_return_val = pickle.loads(return_val_pkl){filter_variables}
            ret = {class_name_alias}.{method_name}(**args)
            """
        + (
            """self.assertTrue(comparator(traced_return_val, ret))
        """
            if test_framework == "unittest"
            else """assert comparator(traced_return_val, ret)
        """
        ),
    )
    if test_framework == "unittest":
        self = "self"
        test_template = "\nclass TestTracedFunctions(unittest.TestCase):\n"
    else:
        test_template = ""
        self = ""
    for func in functions:
        if not is_function_or_method_top_level(
            file_name=func.file_name,
            function_or_method_name=func.function_name,
            class_name=func.class_name,
        ):
            # can't be imported and run in the replay test
            continue
        if func.class_name is None:
            alias = get_function_alias(func.module_name, func.function_name)
            test_body = test_function_body.format(
                trace_file=trace_file,
                function_name=alias,
                file_name=func.file_name,
                orig_function_name=func.function_name,
                max_run_count=max_run_count,
            )
        else:
            class_name_alias = get_function_alias(func.module_name, func.class_name)
            alias = get_function_alias(func.module_name, func.class_name + "_" + func.function_name)
            if func.function_name == "__init__":
                filter_variables = "\n    args.pop('__class__', None)"
            else:
                filter_variables = ""
            test_body = test_class_method_body.format(
                trace_file=trace_file,
                orig_function_name=func.function_name,
                file_name=func.file_name,
                class_name_alias=class_name_alias,
                class_name=func.class_name,
                method_name=func.function_name,
                max_run_count=max_run_count,
                filter_variables=filter_variables,
            )
        formatted_test_body = textwrap.indent(
            test_body,
            "        " if test_framework == "unittest" else "    ",
        )

        test_template += "    " if test_framework == "unittest" else ""
        test_template += f"def test_{alias}({self}):\n{formatted_test_body}\n"

    return imports + "\n" + test_template
