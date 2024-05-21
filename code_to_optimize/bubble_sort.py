import time


def sorter(arr):
    for i in range(len(arr)):
        for k in range(len(arr) - 1):
            time.sleep(0.1)
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    return arr


CACHED_TESTS = "import unittest\ndef sorter(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - 1):\n            if arr[j] > arr[j + 1]:\n                temp = arr[j]\n                arr[j] = arr[j + 1]\n                arr[j + 1] = temp\n    return arr\nclass SorterTestCase(unittest.TestCase):\n    def test_empty_list(self):\n        self.assertEqual(sorter([]), [])\n    def test_single_element_list(self):\n        self.assertEqual(sorter([5]), [5])\n    def test_ascending_order_list(self):\n        self.assertEqual(sorter([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])\n    def test_descending_order_list(self):\n        self.assertEqual(sorter([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])\n    def test_random_order_list(self):\n        self.assertEqual(sorter([3, 1, 4, 2, 5]), [1, 2, 3, 4, 5])\n    def test_duplicate_elements_list(self):\n        self.assertEqual(sorter([3, 1, 4, 2, 2, 5, 1]), [1, 1, 2, 2, 3, 4, 5])\n    def test_negative_numbers_list(self):\n        self.assertEqual(sorter([-5, -2, -8, -1, -3]), [-8, -5, -3, -2, -1])\n    def test_mixed_data_types_list(self):\n        self.assertEqual(sorter(['apple', 2, 'banana', 1, 'cherry']), [1, 2, 'apple', 'banana', 'cherry'])\n    def test_large_input_list(self):\n        self.assertEqual(sorter(list(range(1000, 0, -1))), list(range(1, 1001)))\n    def test_list_with_none_values(self):\n        self.assertEqual(sorter([None, 2, None, 1, None]), [None, None, None, 1, 2])\n    def test_list_with_nan_values(self):\n        self.assertEqual(sorter([float('nan'), 2, float('nan'), 1, float('nan')]), [1, 2, float('nan'), float('nan'), float('nan')])\n    def test_list_with_complex_numbers(self):\n        self.assertEqual(sorter([3 + 2j, 1 + 1j, 4 + 3j, 2 + 1j, 5 + 4j]), [1 + 1j, 2 + 1j, 3 + 2j, 4 + 3j, 5 + 4j])\n    def test_list_with_custom_class_objects(self):\n        class Person:\n            def __init__(self, name, age):\n                self.name = name\n                self.age = age\n            def __repr__(self):\n                return f\"Person('{self.name}', {self.age})\"\n        input_list = [Person('Alice', 25), Person('Bob', 30), Person('Charlie', 20)]\n        expected_output = [Person('Charlie', 20), Person('Alice', 25), Person('Bob', 30)]\n        self.assertEqual(sorter(input_list), expected_output)\n    def test_list_with_uncomparable_elements(self):\n        with self.assertRaises(TypeError):\n            sorter([5, 'apple', 3, [1, 2, 3], 2])\n    def test_list_with_custom_comparison_function(self):\n        input_list = [5, 4, 3, 2, 1]\n        expected_output = [5, 4, 3, 2, 1]\n        self.assertEqual(sorter(input_list, reverse=True), expected_output)\nif __name__ == '__main__':\n    unittest.main()"
