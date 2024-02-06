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
    return_value = wrap(sorter, "test_sort_parametrized", "4", input)
    output = return_value[0]
    codeflash_duration = return_value[1]
    test_id = return_value[2]
    assert output == expected_output


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(5000))), list(range(5000))),
    ],
)
def test_sort_parametrized2(input, expected_output):
    return_value = wrap(sorter, "test_sort_parametrized2", "5", input)
    output = return_value[0]
    codeflash_duration = return_value[1]
    test_id = return_value[2]
    assert output == expected_output
