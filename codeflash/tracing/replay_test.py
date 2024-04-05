import logging
import sqlite3
import textwrap
from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple


def get_next_arg_and_return(
    trace_file: str, function_name: str, num_to_get: int = 3
) -> Generator[Tuple[Any, Any], None, None]:
    db = sqlite3.connect(trace_file)
    cur = db.cursor()
    limit = num_to_get * 2 + 100
    data = cur.execute(
        "SELECT * FROM events WHERE function = ? ORDER BY time_ns ASC LIMIT ?",
        (function_name, limit),
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
    trace_file: str, functions: List[Tuple[str, str]], test_framework: str = "pytest"
) -> str:
    assert test_framework in ["pytest", "unittest"]

    imports = f"""import pickle
import {test_framework}
from codeflash.tracing.replay_test import get_next_arg_and_return
from codeflash.verification.comparator import comparator
"""

    # TODO: Module can have "-" character if the module-root is ".". Need to handle that case
    function_imports = [
        f"from {module} import {function_name} as {get_function_alias(module, function_name)}"
        for module, function_name in functions
    ]
    imports += "\n".join(function_imports)

    if test_framework == "unittest":
        return imports + _create_unittest_trace_replay_test(trace_file, functions)
    elif test_framework == "pytest":
        return imports + _create_pytest_trace_replay_test(trace_file, functions)
    else:
        raise ValueError("Invalid test framework")


def _create_unittest_trace_replay_test(trace_file: str, functions: List[Tuple[str, str]]) -> str:
    test_function_body = textwrap.dedent(
        """\
        for arg_val_pkl, return_val_pkl in get_next_arg_and_return('{trace_file}', '{orig_function_name}', 3):
            args = pickle.loads(arg_val_pkl)
            return_val = pickle.loads(return_val_pkl)
            ret = {function_name}(**args)
            self.assertTrue(comparator(return_val, ret))
    """
    )

    test_template = "\nclass TestTracedFunctions(unittest.TestCase):\n"
    for module, function_name in functions:
        function_name_alias = get_function_alias(module, function_name)
        formatted_test_body = textwrap.indent(
            test_function_body.format(
                trace_file=trace_file,
                function_name=function_name_alias,
                orig_function_name=function_name,
            ),
            "        ",
        )
        test_template += f"    def test_{function_name_alias}(self):\n{formatted_test_body}\n"

    return test_template


def _create_pytest_trace_replay_test(trace_file: str, functions: List[Tuple[str, str]]) -> str:
    test_function_body = textwrap.dedent(
        """\
        for arg_val_pkl, return_val_pkl in get_next_arg_and_return('{trace_file}', '{orig_function_name}', 3):
            args = pickle.loads(arg_val_pkl)
            return_val = pickle.loads(return_val_pkl)
            ret = {function_name}(**args)
            assert comparator(return_val, ret)
    """
    )

    test_template = ""
    for module, function_name in functions:
        function_name_alias = get_function_alias(module, function_name)
        formatted_test_body = textwrap.indent(
            test_function_body.format(
                trace_file=trace_file,
                function_name=function_name_alias,
                orig_function_name=function_name,
            ),
            "    ",
        )
        test_template += f"\ndef test_{function_name_alias}():\n{formatted_test_body}\n"

    return test_template
