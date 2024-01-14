import os.path
import sys
import tempfile

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test


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


def test_perfinjector_only_replay_test():
    code = """import pickle
import pytest
from codeflash.tracing.replay_test import get_next_arg_and_return
from codeflash.validation.equivalence import compare_results
from packagename.ml.yolo.image_reshaping_utils import prepare_image_for_yolo as packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo
def test_prepare_image_for_yolo():
    for arg_val_pkl, return_val_pkl in get_next_arg_and_return('/home/saurabh/packagename/traces/first.trace', 3):
        args = pickle.loads(arg_val_pkl)
        return_val_1= pickle.loads(return_val_pkl)
        ret = packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo(**args)
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
from packagename.ml.yolo.image_reshaping_utils import prepare_image_for_yolo as packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo

def test_prepare_image_for_yolo():
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, iteration_id TEXT, runtime INTEGER, return_value BLOB)')
"""
    if sys.version_info <= (3, 10):
        expected += """    for (arg_val_pkl, return_val_pkl) in get_next_arg_and_return('/home/saurabh/packagename/traces/first.trace', 3):
"""
    else:
        expected += """    for arg_val_pkl, return_val_pkl in get_next_arg_and_return('/home/saurabh/packagename/traces/first.trace', 3):
"""
    expected += """        args = pickle.loads(arg_val_pkl)
        return_val_1 = pickle.loads(return_val_pkl)
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo(**args)
        codeflash_duration = time.perf_counter_ns() - counter
        gc.enable()
        codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)', ('{module_path}', None, 'test_prepare_image_for_yolo', 'packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo', '4_2', codeflash_duration, pickle.dumps(return_value)))
        codeflash_con.commit()
        print(f'#####{module_path}:test_prepare_image_for_yolo:packagename_ml_yolo_image_reshaping_utils_prepare_image_for_yolo:4_2#####{{codeflash_duration}}^^^^^')
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
