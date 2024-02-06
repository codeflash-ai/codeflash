import time
import gc
import os
import sqlite3
import pickle
from code_to_optimize.bubble_sort import sorter


def codeflash_wrap(wrapped, function_name, test_id, *args, **kwargs):
    if not hasattr(codeflash_wrap, "index"):
        codeflash_wrap.index = {}

    if function_name in codeflash_wrap.index:
        codeflash_wrap.index[function_name] += 1
    else:
        codeflash_wrap.index[function_name] = 0

    codeflash_test_index = codeflash_wrap.index[function_name]
    test_id = f"{test_id}_{codeflash_test_index}"
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = wrapped(*args, **kwargs)
    codeflash_duration = time.perf_counter_ns() - counter
    gc.enable()
    return return_value


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    input_2 = 4
    return_value = codeflash_wrap(sorter, "test_sort", "3", input)
    output = return_value[0]
    codeflash_duration = output[1]
    test_id = return_value[2]
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    return_value = codeflash_wrap(sorter, "test_sort", "4", input)
    output = return_value[0]
    codeflash_duration = output[1]
    test_id = return_value[2]
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    input = list(reversed(range(5000)))
    return_value = codeflash_wrap(sorter, "test_sort", "5", input)
    output = return_value[0]
    codeflash_duration = output[1]
    test_id = return_value[2]
    assert output == list(range(5000))
