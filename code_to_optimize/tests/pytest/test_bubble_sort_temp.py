import time
import gc
import os
import sqlite3
import pickle
from code_to_optimize.bubble_sort import sorter


def codeflash_wrap(wrapped, test_name, test_id, codeflash_cur, codeflash_con, *args, **kwargs):
    if not hasattr(codeflash_wrap, "index"):
        codeflash_wrap.index = {}

    if test_name in codeflash_wrap.index:
        codeflash_wrap.index[test_name] += 1
    else:
        codeflash_wrap.index[test_name] = 0

    codeflash_test_index = codeflash_wrap.index[test_name]
    test_id = f"{test_id}_{codeflash_test_index}"
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = wrapped(*args, **kwargs)
    codeflash_duration = time.perf_counter_ns() - counter
    gc.enable()
    codeflash_cur.execute(
        "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "code_to_optimize.tests.pytest.test_bubble_sort",
            None,
            "test_sort",
            "sorter",
            f"{test_id}",
            codeflash_duration,
            pickle.dumps(return_value),
        ),
    )
    codeflash_con.commit()
    print(
        f"#####code_to_optimize.tests.pytest.test_bubble_sort:test_sort:sorter:{test_id}#####{codeflash_duration}^^^^^"
    )
    return return_value


def test_sort():
    codeflash_iteration = os.environ["CODEFLASH_TEST_ITERATION"]
    codeflash_con = sqlite3.connect(
        f"/var/folders/gq/15b1g3r95zj1y8p0c737z7sh0000gn/T/codeflash_0s778n18/test_return_values_{codeflash_iteration}.sqlite"
    )
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute(
        "CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, iteration_id TEXT, runtime INTEGER, return_value BLOB)"
    )
    input = [5, 4, 3, 2, 1, 0]
    # gc.disable()
    # counter = time.perf_counter_ns()
    # return_value = sorter(input)
    # codeflash_duration = time.perf_counter_ns() - counter
    # gc.enable()
    output = codeflash_wrap(sorter, "test_sort", "5", codeflash_cur, codeflash_con, input)

    # output = return_value
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    input_2 = 5.0
    # gc.disable()
    # counter = time.perf_counter_ns()
    # return_value = sorter(input, input_2, input_3=5)
    # codeflash_duration = time.perf_counter_ns() - counter
    # gc.enable()
    output, codeflash_duration, codeflash_test_id = codeflash_wrap(
        sorter, "test_sort", "8", codeflash_cur, codeflash_con, input, input_2, input_3=5
    )

    # output = return_value
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    input = list(reversed(range(5000)))
    # gc.disable()
    # counter = time.perf_counter_ns()
    # return_value = sorter(input)
    # codeflash_duration = time.perf_counter_ns() - counter
    # gc.enable()
    output, codeflash_duration, codeflash_test_id = codeflash_wrap(
        sorter, "test_sort", "11", codeflash_cur, codeflash_con, input
    )
    codeflash_cur.execute(
        "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "code_to_optimize.tests.pytest.test_bubble_sort",
            None,
            "test_sort",
            "sorter",
            f"{codeflash_test_id}",
            codeflash_duration,
            pickle.dumps(output),
        ),
    )
    codeflash_con.commit()
    print(
        f"#####code_to_optimize.tests.pytest.test_bubble_sort:test_sort:sorter:{codeflash_test_id}#####{codeflash_duration}^^^^^"
    )
    # output = return_value
    assert output == list(range(5000))
    codeflash_con.close()
