import logging
import sqlite3
import textwrap
from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple

from codeflash.tracing.tracing_utils import FunctionModules


def get_next_arg_and_return(
    trace_file: str, function_name: str, file_name: str, num_to_get: int = 3
) -> Generator[Tuple[Any, Any], None, None]:
    db = sqlite3.connect(trace_file)
    cur = db.cursor()
    limit = num_to_get * 2 + 100
    data = cur.execute(
        "SELECT * FROM events WHERE function = ? AND filename = ? ORDER BY time_ns ASC LIMIT ?",
        (function_name, file_name, limit),
    ).fetchall()

    counts = 0
    matched_arg_return: Dict[int, List[Any]] = defaultdict(list)
    for val in data:
        if counts >= num_to_get:
            break

        event_type, frame_address = val[0], val[4]
        if event_type == "call":
            matched_arg_return[frame_address].append(val[7])
            if len(matched_arg_return[frame_address]) > 1:
                logging.warning(
                    f"Pre-existing call to the function {function_name} with same frame address."
                )
        elif event_type == "return":
            matched_arg_return[frame_address].append(val[6])
            arg_return_length = len(matched_arg_return[frame_address])
            if arg_return_length > 2:
                logging.warning(
                    f"Pre-existing return to the function {function_name} with same frame address."
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
    max_run_count=30,
) -> str:
    assert test_framework in ["pytest", "unittest"]

    imports = f"""import pickle
import {test_framework}
from codeflash.tracing.replay_test import get_next_arg_and_return
from codeflash.verification.comparator import comparator
"""

    # TODO: Module can have "-" character if the module-root is ".". Need to handle that case
    function_imports = []
    for function in functions:
        if function.class_name:
            function_imports.append(
                f"from {function.module_name} import {function.class_name} as {get_function_alias(function.module_name, function.class_name)}"
            )
        else:
            function_imports.append(
                f"from {function.module_name} import {function.function_name} as {get_function_alias(function.module_name, function.function_name)}"
            )

    imports += "\n".join(function_imports)

    if test_framework == "unittest":
        return imports + _create_unittest_trace_replay_test(
            trace_file, functions, max_run_count=max_run_count
        )
    elif test_framework == "pytest":
        return imports + _create_pytest_trace_replay_test(
            trace_file, functions, max_run_count=max_run_count
        )
    else:
        raise ValueError("Invalid test framework")


def _create_unittest_trace_replay_test(
    trace_file: str, functions: List[FunctionModules], max_run_count
) -> str:
    test_function_body = textwrap.dedent(
        """\
        for arg_val_pkl, return_val_pkl in get_next_arg_and_return(r'{trace_file}', '{orig_function_name}', '{file_name}', {max_run_count}):
            args = pickle.loads(arg_val_pkl)
            return_val = pickle.loads(return_val_pkl)
            ret = {function_name}(**args)
            self.assertTrue(comparator(return_val, ret))
    """
    )

    test_template = "\nclass TestTracedFunctions(unittest.TestCase):\n"
    for func in functions:
        function_name_alias = get_function_alias(func.module_name, func.function_name)
        formatted_test_body = textwrap.indent(
            test_function_body.format(
                trace_file=trace_file,
                function_name=function_name_alias,
                file_name=func.file_name,
                orig_function_name=func.function_name,
                max_run_count=max_run_count,
            ),
            "        ",
        )
        test_template += f"    def test_{function_name_alias}(self):\n{formatted_test_body}\n"

    return test_template


def _create_pytest_trace_replay_test(
    trace_file: str, functions: List[FunctionModules], max_run_count
) -> str:
    test_function_body = textwrap.dedent(
        """\
        for arg_val_pkl, return_val_pkl in get_next_arg_and_return(r'{trace_file}', '{orig_function_name}', '{file_name}', {max_run_count}):
            args = pickle.loads(arg_val_pkl)
            return_val = pickle.loads(return_val_pkl)
            ret = {function_name}(**args)
            assert comparator(return_val, ret)
    """
    )

    test_template = ""
    for func in functions:
        function_name_alias = get_function_alias(func.module_name, func.function_name)
        formatted_test_body = textwrap.indent(
            test_function_body.format(
                trace_file=trace_file,
                function_name=function_name_alias,
                orig_function_name=func.function_name,
                file_name=func.file_name,
                max_run_count=max_run_count,
            ),
            "    ",
        )
        test_template += f"\ndef test_{function_name_alias}():\n{formatted_test_body}\n"

    return test_template
