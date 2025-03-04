from code_to_optimize.bubble_sort_dep1_helper import dep1_comparer
from code_to_optimize.bubble_sort_dep2_swap import dep2_swap


def sorter_deps(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if dep1_comparer(arr, j):
                dep2_swap(arr, j)
    return arr


CACHED_TESTS = """import dill as pickle
import os
def _log__test__values(values, duration, test_name):
    iteration = os.environ["CODEFLASH_TEST_ITERATION"]
    with open(os.path.join(
    '/var/folders/ms/1tz2l1q55w5b7pp4wpdkbjq80000gn/T/codeflash_jk4pzz3w/', 
    f'test_return_values_{iteration}.bin'), 'ab') as f:
        return_bytes = pickle.dumps(values)
        _test_name = f"{test_name}".encode("ascii")
        f.write(len(_test_name).to_bytes(4, byteorder='big'))
        f.write(_test_name)
        f.write(duration.to_bytes(8, byteorder='big'))
        f.write(len(return_bytes).to_bytes(4, byteorder='big'))
        f.write(return_bytes)
import time
import gc
from code_to_optimize.bubble_sort_deps import sorter_deps
import timeout_decorator
import unittest

def dep1_comparer(arr, j: int) -> bool:
    return arr[j] > arr[j + 1]

def dep2_swap(arr, j):
    temp = arr[j]
    arr[j] = arr[j + 1]
    arr[j + 1] = temp

class TestSorterDeps(unittest.TestCase):

    @timeout_decorator.timeout(15, use_signals=True)
    def test_integers(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([5, 3, 2, 4, 1])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(
        return_value, duration, 
        'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_integers:sorter_deps:0')
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([10, -3, 0, 2, 7])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(
        return_value, duration, 
        ('code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:'
        'TestSorterDeps.test_integers:sorter_deps:1'))

    @timeout_decorator.timeout(15, use_signals=True)
    def test_floats(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([3.2, 1.5, 2.7, 4.1, 1.0])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 
        'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_floats:sorter_deps:0')
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([-1.1, 0.0, 3.14, 2.71, -0.5])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 
        'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_floats:sorter_deps:1')

    @timeout_decorator.timeout(15, use_signals=True)
    def test_identical_elements(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([1, 1, 1, 1, 1])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 
        ('code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:'
        'TestSorterDeps.test_identical_elements:sorter_deps:0'))
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([3.14, 3.14, 3.14])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 
        ('code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:'
        'TestSorterDeps.test_identical_elements:sorter_deps:1'))

    @timeout_decorator.timeout(15, use_signals=True)
    def test_single_element(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([5])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_single_element:sorter_deps:0')
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([-3.2])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_single_element:sorter_deps:1')

    @timeout_decorator.timeout(15, use_signals=True)
    def test_empty_array(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps([])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_empty_array:sorter_deps:0')

    @timeout_decorator.timeout(15, use_signals=True)
    def test_strings(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps(['apple', 'banana', 'cherry', 'date'])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_strings:sorter_deps:0')
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter_deps(['dog', 'cat', 'elephant', 'ant'])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_strings:sorter_deps:1')

    @timeout_decorator.timeout(15, use_signals=True)
    def test_mixed_types(self):
        with self.assertRaises(TypeError):
            gc.disable()
            counter = time.perf_counter_ns()
            return_value = sorter_deps([1, 'two', 3.0, 'four'])
            duration = time.perf_counter_ns() - counter
            gc.enable()
            _log__test__values(return_value, duration, 'code_to_optimize.tests.unittest.test_sorter_deps__unit_test_0:TestSorterDeps.test_mixed_types:sorter_deps:0_0')
if __name__ == '__main__':
    unittest.main()

"""
