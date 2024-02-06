import time
import gc
import os
import sqlite3
import pickle
from code_to_optimize.bubble_sort import sorter
import pytest


def wrap(wrapped, function_name, test_id, *args):
    if not hasattr(wrap, "index"):
        wrap.index = {}

    if function_name in wrap.index:
        wrap.index[function_name] += 1
    else:
        wrap.index[function_name] = 0

    codeflash_test_index = wrap.index[function_name]
    test_id = f"{test_id}_{codeflash_test_index}"
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = wrapped(*args)
    codeflash_duration = time.perf_counter_ns() - counter
    gc.enable()
    return return_value, codeflash_duration, test_id


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
        f"/var/folders/gq/15b1g3r95zj1y8p0c737z7sh0000gn/T/codeflash_0s778n18/test_return_values_{codeflash_iteration}.sqlite"
    )
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute(
        "CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, iteration_id TEXT, runtime INTEGER, return_value BLOB)"
    )
    # gc.disable()
    # counter = time.perf_counter_ns()
    # return_value = wrap(sorter(input))
    # codeflash_duration = time.perf_counter_ns() - counter
    # gc.enable()
    output, codeflash_duration, codeflash_test_id = wrap(
        sorter, "test_sort_parametrized", "4", input
    )

    codeflash_cur.execute(
        "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "code_to_optimize.tests.pytest.test_bubble_sort_parametrized",
            None,
            "test_sort_parametrized",
            "sorter",
            f"{codeflash_test_id}",
            codeflash_duration,
            pickle.dumps(output),
        ),
    )
    codeflash_con.commit()
    print(
        f"#####code_to_optimize.tests.pytest.test_bubble_sort_parametrized:test_sort_parametrized:sorter:{codeflash_test_id}#####{codeflash_duration}^^^^^"
    )
    # output = return_value
    assert output == expected_output
    codeflash_con.close()


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(5000))), list(range(5000))),
    ],
)
def test_sort_parametrized2(input, expected_output):
    codeflash_iteration = os.environ["CODEFLASH_TEST_ITERATION"]
    codeflash_con = sqlite3.connect(
        f"/var/folders/gq/15b1g3r95zj1y8p0c737z7sh0000gn/T/codeflash_0s778n18/test_return_values_{codeflash_iteration}.sqlite"
    )
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute(
        "CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, iteration_id TEXT, runtime INTEGER, return_value BLOB)"
    )
    # gc.disable()
    # counter = time.perf_counter_ns()
    # return_value = sorter(input)
    # codeflash_duration = time.perf_counter_ns() - counter
    # gc.enable()
    output, codeflash_duration, codeflash_test_id = wrap(
        sorter, "test_sort_parametrized2", "5", input
    )
    codeflash_cur.execute(
        "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "code_to_optimize.tests.pytest.test_bubble_sort_parametrized",
            None,
            "test_sort_parametrized",
            "sorter",
            f"{codeflash_test_id}",
            codeflash_duration,
            pickle.dumps(output),
        ),
    )
    codeflash_con.commit()
    print(
        f"#####code_to_optimize.tests.pytest.test_bubble_sort_parametrized:test_sort_parametrized:sorter:{codeflash_test_id}#####{codeflash_duration}^^^^^"
    )
    # output = return_value
    assert output == expected_output
    codeflash_con.close()
