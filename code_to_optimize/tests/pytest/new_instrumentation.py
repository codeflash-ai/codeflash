import time
import gc
import os
import sqlite3
import pickle


def codeflash_wrap(
    wrapped,
    test_module_name,
    test_class_name,
    test_name,
    function_name,
    line_id,
    codeflash_cur,
    codeflash_con,
    *args,
    **kwargs,
):
    test_id = f"{test_module_name}:{test_class_name}:{test_name}:{line_id}"
    if not hasattr(codeflash_wrap, "index"):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f"{line_id}_{codeflash_test_index}"
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = wrapped(*args, **kwargs)
    codeflash_duration = time.perf_counter_ns() - counter
    gc.enable()
    codeflash_cur.execute(
        "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            test_module_name,
            test_class_name,
            test_name,
            function_name,
            invocation_id,
            codeflash_duration,
            pickle.dumps(return_value),
        ),
    )
    codeflash_con.commit()
    return return_value


from code_to_optimize.bubble_sort import sorter
import pytest


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(5000))), list(range(5000))),
    ],
)
def test_sort_parametrized(input, expected_output):
    codeflash_iteration = os.environ["CODEFLASH_TEST_ITERATION"]
    codeflash_con = sqlite3.connect(
        f"/var/folders/gq/15b1g3r95zj1y8p0c737z7sh0000gn/T/codeflash__ey38owb/test_return_values_{codeflash_iteration}.sqlite"
    )
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute(
        "CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, iteration_id TEXT, runtime INTEGER, return_value BLOB)"
    )
    output = codeflash_wrap(
        sorter,
        "code_to_optimize.tests.pytest.test_bubble_sort_parametrized",
        None,
        "test_sort_parametrized",
        "sorter",
        "4",
        codeflash_cur,
        codeflash_con,
        input,
    )
    assert output == expected_output
    codeflash_con.close()
