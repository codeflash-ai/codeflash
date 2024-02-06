import time
import gc
import os
import sqlite3
import pickle
from code_to_optimize.bubble_sort import sorter


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


def test_sort():
    inputs = [[5, 4, 3, 2, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], list(reversed(range(5000)))]
    expected_outputs = [[0, 1, 2, 3, 4, 5], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(5000))]

    for input, expected_output in zip(inputs, expected_outputs):
        return_value = wrap(sorter, "test_sort", "4", input)
        output = return_value[0]
        codeflash_duration = output[1]
        test_id = return_value[2]
        assert output == expected_output
