import ast
import os
import sys
import tempfile

from codeflash.code_utils.ast_unparser import ast_unparse
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.discovery.functions_to_optimize import FunctionToOptimize, FunctionParent
from codeflash.instrumentation.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash.instrumentation.instrument_new_tests import InjectPerfAndLogging

os.environ["CODEFLASH_API_KEY"] = "test-key"


def test_InjectPerfAndLogging_with():
    code = """def test_relative_validity_no_tree():
    hdbscan = HDBSCAN()
    with pytest.raises(AttributeError):
        hdbscan.relative_validity_()"""
    auxillary_functions = []
    module_node = ast.parse(code)
    test_module_path = "code_to_optimize_path"
    function_to_optimize = FunctionToOptimize(
        function_name="relative_validity_",
        file_path="/tmp/path",
        parents=[FunctionParent(name="HDBSCAN", type="ClassDef")],
    )
    new_module_node = InjectPerfAndLogging(
        function_to_optimize, auxillary_functions, test_module_path
    ).visit(module_node)

    expected = """def test_relative_validity_no_tree():
    hdbscan = HDBSCAN()
    with pytest.raises(AttributeError):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = hdbscan.relative_validity_()\n"""
    expected += (
        """        duration = (time.perf_counter_ns() - counter)\n"""
        if sys.version_info < (3, 9, 0)
        else """        duration = time.perf_counter_ns() - counter\n"""
    )
    expected += """        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:test_relative_validity_no_tree:relative_validity_:1_0')"""
    assert ast_unparse(new_module_node).strip("\n") == expected


def test_InjectPerfAndLogging():
    code = """def test_relative_validity_no_tree():
    hdbscan = HDBSCAN()
    hdbscan.relative_validity_()"""
    auxillary_functions = []
    module_node = ast.parse(code)
    test_module_path = "code_to_optimize_path"
    function_to_optimize = FunctionToOptimize(
        function_name="relative_validity_",
        file_path="/tmp/path",
        parents=[FunctionParent(name="HDBSCAN", type="ClassDef")],
    )
    new_module_node = InjectPerfAndLogging(
        function_to_optimize, auxillary_functions, test_module_path
    ).visit(module_node)
    expected = """def test_relative_validity_no_tree():
    hdbscan = HDBSCAN()
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = hdbscan.relative_validity_()\n"""
    expected += (
        """    duration = (time.perf_counter_ns() - counter)\n"""
        if sys.version_info < (3, 9, 0)
        else """    duration = time.perf_counter_ns() - counter\n"""
    )
    expected += """    gc.enable()
    _log__test__values(return_value, duration, 'code_to_optimize_path:test_relative_validity_no_tree:relative_validity_:1')"""
    assert ast_unparse(new_module_node).strip("\n") == expected


def test_perfinjector_only_replay_test():
    code = """import pickle
import pytest
from codeflash.tracing.replay_test import get_next_arg_and_return
from codeflash.validation.equivalence import compare_results
from velo.ml.yolo.image_reshaping_utils import prepare_image_for_yolo as velo_ml_yolo_image_reshaping_utils_prepare_image_for_yolo
def test_prepare_image_for_yolo():
    for arg_val_pkl, return_val_pkl in get_next_arg_and_return('/home/saurabh/velo/traces/first.trace', 3):
        args = pickle.loads(arg_val_pkl)
        return_val_1= pickle.loads(return_val_pkl)
        ret = velo_ml_yolo_image_reshaping_utils_prepare_image_for_yolo(**args)
        assert compare_results(return_val_1, ret)
"""
    expected = """import time
import gc
import os
import sqlite3
import pickle
import pickle
import pytest
from codeflash.tracing.replay_test import get_next_arg_and_return
from codeflash.validation.equivalence import compare_results
from velo.ml.yolo.image_reshaping_utils import prepare_image_for_yolo as velo_ml_yolo_image_reshaping_utils_prepare_image_for_yolo

def test_prepare_image_for_yolo():
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, iteration_id TEXT, runtime INTEGER, return_value BLOB)')
    for arg_val_pkl, return_val_pkl in get_next_arg_and_return('/home/saurabh/velo/traces/first.trace', 3):
        args = pickle.loads(arg_val_pkl)
        return_val_1 = pickle.loads(return_val_pkl)
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = velo_ml_yolo_image_reshaping_utils_prepare_image_for_yolo(**args)
        codeflash_duration = time.perf_counter_ns() - counter
        gc.enable()
        codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)', ('{module_path}', None, 'test_prepare_image_for_yolo', 'velo_ml_yolo_image_reshaping_utils_prepare_image_for_yolo', '4_2', codeflash_duration, pickle.dumps(return_value)))
        codeflash_con.commit()
        print(f'#####{module_path}:test_prepare_image_for_yolo:velo_ml_yolo_image_reshaping_utils_prepare_image_for_yolo:4_2#####{{codeflash_duration}}^^^^^')
        ret = return_value
        assert compare_results(return_val_1, ret)
    codeflash_con.close()"""
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        new_test = inject_profiling_into_existing_test(
            f.name, "prepare_image_for_yolo", os.path.dirname(f.name)
        )
        assert new_test == expected.format(
            module_path=os.path.basename(f.name),
            tmp_dir_path=get_run_tmp_file("test_return_values"),
        )


def test_remove_bad_assert():
    code = """def test_relative_validity_no_tree():
    hdbscan = HDBSCAN()
    assert hdbscan.relative_validity_() == 0.5
    result = 5
    assert result == 5"""
    auxillary_functions = []
    module_node = ast.parse(code)
    test_module_path = "code_to_optimize_path"
    function_to_optimize = FunctionToOptimize(
        function_name="relative_validity_",
        file_path="/tmp/path",
        parents=[FunctionParent(name="HDBSCAN", type="ClassDef")],
    )
    new_module_node = InjectPerfAndLogging(
        function_to_optimize, auxillary_functions, test_module_path
    ).visit(module_node)
    expected = """def test_relative_validity_no_tree():
    hdbscan = HDBSCAN()
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = hdbscan.relative_validity_()
"""
    if sys.version_info < (3, 9, 0):
        expected += """    duration = (time.perf_counter_ns() - counter)\n"""
    else:
        expected += """    duration = time.perf_counter_ns() - counter\n"""
    expected += """    gc.enable()
    _log__test__values(return_value, duration, 'code_to_optimize_path:test_relative_validity_no_tree:relative_validity_:1')
    result = 5"""
    assert ast_unparse(new_module_node).strip("\n") == expected

    code = """def test_translate_word_starting_with_vowel():
    assert 1 == True
    assert translate('apple') == 'appleway'
def test_translate_word_starting_with_single_consonant():
    assert translate('banana') == 'ananabay'"""

    function_to_optimize = FunctionToOptimize(
        function_name="translate", file_path="/tmp/path", parents=[]
    )
    auxillary_functions = []
    module_node = ast.parse(code)
    test_module_path = "code_to_optimize_path"
    new_module_node = InjectPerfAndLogging(
        function_to_optimize, auxillary_functions, test_module_path
    ).visit(module_node)
    expected = """def test_translate_word_starting_with_vowel():
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = translate('apple')
"""
    expected += (
        """    duration = (time.perf_counter_ns() - counter)\n"""
        if sys.version_info < (3, 9, 0)
        else """    duration = time.perf_counter_ns() - counter\n"""
    )
    # duration = time.perf_counter_ns() - counter

    expected += """    gc.enable()
    _log__test__values(return_value, duration, 'code_to_optimize_path:test_translate_word_starting_with_vowel:translate:1')

def test_translate_word_starting_with_single_consonant():
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = translate('banana')\n"""
    expected += (
        """    duration = (time.perf_counter_ns() - counter)\n"""
        if sys.version_info < (3, 9, 0)
        else """    duration = time.perf_counter_ns() - counter\n"""
    )
    expected += """    gc.enable()
    _log__test__values(return_value, duration, 'code_to_optimize_path:test_translate_word_starting_with_single_consonant:translate:0')"""
    assert ast_unparse(new_module_node).strip("\n") == expected


def test_perfinjector_bubble_sort():
    code = """import unittest

from code_to_optimize.bubble_sort import sorter


class TestPigLatin(unittest.TestCase):
    def test_sort(self):
        input = [5, 4, 3, 2, 1, 0]
        output = sorter(input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])

        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = sorter(input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        input = list(reversed(range(5000)))
        output = sorter(input)
        self.assertEqual(output, list(range(5000)))
"""
    expected = """import time
import gc
import os
import sqlite3
import pickle
import unittest
from code_to_optimize.bubble_sort import sorter

class TestPigLatin(unittest.TestCase):

    def test_sort(self):
        codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
        codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
        codeflash_cur = codeflash_con.cursor()
        codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, iteration_id TEXT, runtime INTEGER, return_value BLOB)')
        input = [5, 4, 3, 2, 1, 0]
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter(input)
        codeflash_duration = time.perf_counter_ns() - counter
        gc.enable()
        codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)', ('{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '5', codeflash_duration, pickle.dumps(return_value)))
        codeflash_con.commit()
        print(f'#####{module_path}:TestPigLatin.test_sort:sorter:5#####{{codeflash_duration}}^^^^^')
        output = return_value
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter(input)
        codeflash_duration = time.perf_counter_ns() - counter
        gc.enable()
        codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)', ('{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '8', codeflash_duration, pickle.dumps(return_value)))
        codeflash_con.commit()
        print(f'#####{module_path}:TestPigLatin.test_sort:sorter:8#####{{codeflash_duration}}^^^^^')
        output = return_value
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(5000)))
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter(input)
        codeflash_duration = time.perf_counter_ns() - counter
        gc.enable()
        codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)', ('{module_path}', 'TestPigLatin', 'test_sort', 'sorter', '11', codeflash_duration, pickle.dumps(return_value)))
        codeflash_con.commit()
        print(f'#####{module_path}:TestPigLatin.test_sort:sorter:11#####{{codeflash_duration}}^^^^^')
        output = return_value
        self.assertEqual(output, list(range(5000)))
        codeflash_con.close()"""
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        new_test = inject_profiling_into_existing_test(f.name, "sorter", os.path.dirname(f.name))
        assert new_test == expected.format(
            module_path=os.path.basename(f.name),
            tmp_dir_path=get_run_tmp_file("test_return_values"),
        )


def test_unittest_generated_tests_bubble_sort():
    code = """import unittest
def sorter(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    return arr
class SorterTestCase(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(sorter([]), [])
    def test_single_element_list(self):
        self.assertEqual(sorter([5]), [5])
    def test_ascending_order_list(self):
        self.assertEqual(sorter([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])
    def test_descending_order_list(self):
        self.assertEqual(sorter([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])
    def test_random_order_list(self):
        self.assertEqual(sorter([3, 1, 4, 2, 5]), [1, 2, 3, 4, 5])
    def test_duplicate_elements_list(self):
        self.assertEqual(sorter([3, 1, 4, 2, 2, 5, 1]), [1, 1, 2, 2, 3, 4, 5])
    def test_negative_numbers_list(self):
        self.assertEqual(sorter([-5, -2, -8, -1, -3]), [-8, -5, -3, -2, -1])
    def test_mixed_data_types_list(self):
        self.assertEqual(sorter(['apple', 2, 'banana', 1, 'cherry']), [1, 2, 'apple', 'banana', 'cherry'])
    def test_large_input_list(self):
        self.assertEqual(sorter(list(range(1000, 0, -1))), list(range(1, 1001)))
    def test_list_with_none_values(self):
        self.assertEqual(sorter([None, 2, None, 1, None]), [None, None, None, 1, 2])
    def test_list_with_nan_values(self):
        self.assertEqual(sorter([float('nan'), 2, float('nan'), 1, float('nan')]), [1, 2, float('nan'), float('nan'), float('nan')])
    def test_list_with_complex_numbers(self):
        self.assertEqual(sorter([3 + 2j, 1 + 1j, 4 + 3j, 2 + 1j, 5 + 4j]), [1 + 1j, 2 + 1j, 3 + 2j, 4 + 3j, 5 + 4j])
    def test_list_with_custom_class_objects(self):
        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age
            def __repr__(self):
                return f\"Person('{self.name}', {self.age})\"
        input_list = [Person('Alice', 25), Person('Bob', 30), Person('Charlie', 20)]
        expected_output = [Person('Charlie', 20), Person('Alice', 25), Person('Bob', 30)]
        self.assertEqual(sorter(input_list), expected_output)
    def test_list_with_uncomparable_elements(self):
        with self.assertRaises(TypeError):
            sorter([5, 'apple', 3, [1, 2, 3], 2])
    def test_list_with_custom_comparison_function(self):
        input_list = [5, 4, 3, 2, 1]
        expected_output = [5, 4, 3, 2, 1]
        self.assertEqual(sorter(input_list, reverse=True), expected_output)
if __name__ == '__main__':
    unittest.main()
"""

    expected = """import unittest

class SorterTestCase(unittest.TestCase):

    def test_empty_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_empty_list:sorter:0')

    def test_single_element_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([5])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_single_element_list:sorter:0')

    def test_ascending_order_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([1, 2, 3, 4, 5])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_ascending_order_list:sorter:0')

    def test_descending_order_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([5, 4, 3, 2, 1])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_descending_order_list:sorter:0')

    def test_random_order_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([3, 1, 4, 2, 5])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_random_order_list:sorter:0')

    def test_duplicate_elements_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([3, 1, 4, 2, 2, 5, 1])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_duplicate_elements_list:sorter:0')

    def test_negative_numbers_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([-5, -2, -8, -1, -3])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_negative_numbers_list:sorter:0')

    def test_mixed_data_types_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter(['apple', 2, 'banana', 1, 'cherry'])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_mixed_data_types_list:sorter:0')

    def test_large_input_list(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter(list(range(1000, 0, -1)))
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_large_input_list:sorter:0')

    def test_list_with_none_values(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([None, 2, None, 1, None])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_list_with_none_values:sorter:0')

    def test_list_with_nan_values(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([float('nan'), 2, float('nan'), 1, float('nan')])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_list_with_nan_values:sorter:0')

    def test_list_with_complex_numbers(self):
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter([3 + 2j, 1 + 1j, 4 + 3j, 2 + 1j, 5 + 4j])
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_list_with_complex_numbers:sorter:0')

    def test_list_with_custom_class_objects(self):

        class Person:

            def __init__(self, name, age):
                self.name = name
                self.age = age

            def __repr__(self):
                return f"Person('{self.name}', {self.age})"
        input_list = [Person('Alice', 25), Person('Bob', 30), Person('Charlie', 20)]
        expected_output = [Person('Charlie', 20), Person('Alice', 25), Person('Bob', 30)]
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter(input_list)
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_list_with_custom_class_objects:sorter:3')

    def test_list_with_uncomparable_elements(self):
        with self.assertRaises(TypeError):
            gc.disable()
            counter = time.perf_counter_ns()
            return_value = sorter([5, 'apple', 3, [1, 2, 3], 2])
            duration = time.perf_counter_ns() - counter
            gc.enable()
            _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_list_with_uncomparable_elements:sorter:0_0')

    def test_list_with_custom_comparison_function(self):
        input_list = [5, 4, 3, 2, 1]
        expected_output = [5, 4, 3, 2, 1]
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = sorter(input_list, reverse=True)
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'code_to_optimize_path:SorterTestCase.test_list_with_custom_comparison_function:sorter:2')
if __name__ == '__main__':
    unittest.main()"""

    auxillary_functions = []
    module_node = ast.parse(code)
    test_module_path = "code_to_optimize_path"
    function_to_optimize = FunctionToOptimize(
        function_name="sorter", file_path="/tmp/path", parents=[]
    )
    new_module_node = InjectPerfAndLogging(
        function_to_optimize, auxillary_functions, test_module_path
    ).visit(module_node)
    assert ast_unparse(new_module_node).strip("\n") == expected
