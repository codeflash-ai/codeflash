import gc
import inspect
import os
import sqlite3
import time

import dill as pickle

from code_to_optimize.bubble_sort_method import BubbleSorter


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f"{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}"
    print(f"!$######{test_stdout_tag}######$!")
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f"!######{test_stdout_tag}######!")
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_sort():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'/var/folders/mg/k_c0twcj37q_gph3cfy3zlt80000gn/T/codeflash_ec8xrcji/test_return_values_{codeflash_iteration}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    input = [5, 4, 3, 2, 1, 0]
    _call__bound__arguments = inspect.signature(BubbleSorter.sorter_classmethod).bind(input)
    _call__bound__arguments.apply_defaults()
    output = codeflash_wrap(BubbleSorter.sorter_classmethod, 'code_to_optimize.tests.pytest.test_classmethod_behavior_results_temp', None, 'test_sort', 'BubbleSorter.sorter_classmethod', '1', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert output == [0, 1, 2, 3, 4, 5]
    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    _call__bound__arguments = inspect.signature(BubbleSorter.sorter_classmethod).bind(input)
    _call__bound__arguments.apply_defaults()
    output = codeflash_wrap(BubbleSorter.sorter_classmethod, 'code_to_optimize.tests.pytest.test_classmethod_behavior_results_temp', None, 'test_sort', 'BubbleSorter.sorter_classmethod', '4', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    codeflash_con.close()
