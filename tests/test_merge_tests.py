import os

os.environ["CODEFLASH_API_KEY"] = "test-key"
from codeflash.verification.verifier import merge_unit_tests


def test_merge_unit_tests_pytest():
    unit_tests = """
import time
import gc
from code_to_optimize.tsp import tsp
import pytest
import math
import sys
import itertools

def distance_between(city1: tuple, city2: tuple) -> float:
    return math.hypot(city1[0] - city2[0], city1[1] - city2[1])

def test_tsp_decimal_coordinates():
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp([(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)])
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_decimal_coordinates_0')

def test_tsp_large_coordinate_values():
    cities = [(1000000, 1000000), (2000000, 2000000), (3000000, 3000000)]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_large_coordinate_values_1')
    """

    inspired_test = """
import pytest
import math
import sys
import itertools

def distance_between(city1: tuple, city2: tuple) -> float:
    return math.hypot(city1[0] - city2[0], city1[1] - city2[1])

def tsp(cities: list[list[int]]):
    permutations = itertools.permutations(cities)
    min_distance = sys.maxsize
    optimal_route = []
    for permutation in permutations:
        distance = 0
        for i in range(len(permutation) - 1):
            distance += distance_between(permutation[i], permutation[i + 1])
        distance += distance_between(permutation[-1], permutation[0])
        if distance < min_distance:
            min_distance = distance
            optimal_route = permutation
    return (optimal_route, min_distance)

def test_tsp_more_cities():
    cities = [[1, 2], [3, 4], [5, 6], [-3, 4], [0, 0]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_more_cities__inspired_1')

def test_tsp_three_cities():
    cities = [[1, 2], [3, 4], [5, 6]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_three_cities__inspired_1')

def test_tsp_single_city():
    cities = [[1, 2]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_single_city__inspired_1')

def test_tsp_empty_cities():
    cities = []
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_empty_cities__inspired_1')

def test_tsp_duplicate_cities():
    cities = [[1, 2], [3, 4], [1, 2], [3, 4]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_duplicate_cities__inspired_1')

def test_tsp_negative_coordinates():
    cities = [[-1, -2], [-3, -4], [-5, -6]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_negative_coordinates__inspired_1')
    """
    expected = """import pytest
import math
import sys
import itertools
import time
import gc
from code_to_optimize.tsp import tsp
import pytest
import math
import sys
import itertools

def distance_between(city1: tuple, city2: tuple) -> float:
    return math.hypot(city1[0] - city2[0], city1[1] - city2[1])

def test_tsp_decimal_coordinates():
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp([(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)])
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_decimal_coordinates_0')

def test_tsp_large_coordinate_values():
    cities = [(1000000, 1000000), (2000000, 2000000), (3000000, 3000000)]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_large_coordinate_values_1')

def distance_between(city1: tuple, city2: tuple) -> float:
    return math.hypot(city1[0] - city2[0], city1[1] - city2[1])

def tsp(cities: list[list[int]]):
    permutations = itertools.permutations(cities)
    min_distance = sys.maxsize
    optimal_route = []
    for permutation in permutations:
        distance = 0
        for i in range(len(permutation) - 1):
            distance += distance_between(permutation[i], permutation[i + 1])
        distance += distance_between(permutation[-1], permutation[0])
        if distance < min_distance:
            min_distance = distance
            optimal_route = permutation
    return (optimal_route, min_distance)

def test_tsp_more_cities__inspired():
    cities = [[1, 2], [3, 4], [5, 6], [-3, 4], [0, 0]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_more_cities__inspired_1')

def test_tsp_three_cities__inspired():
    cities = [[1, 2], [3, 4], [5, 6]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_three_cities__inspired_1')

def test_tsp_single_city__inspired():
    cities = [[1, 2]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_single_city__inspired_1')

def test_tsp_empty_cities__inspired():
    cities = []
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_empty_cities__inspired_1')

def test_tsp_duplicate_cities__inspired():
    cities = [[1, 2], [3, 4], [1, 2], [3, 4]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_duplicate_cities__inspired_1')

def test_tsp_negative_coordinates__inspired():
    cities = [[-1, -2], [-3, -4], [-5, -6]]
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = tsp(cities)
    duration = time.perf_counter_ns() - counter
    gc.enable()
    _log__test__values(return_value, duration, 'tsp_test_tsp_negative_coordinates__inspired_1')"""
    modified_file = merge_unit_tests(unit_tests, inspired_test, "pytest")
    assert modified_file == expected


def test_merge_tests_unittest():
    unit_tests = """import time
import gc
from tree_ops import get_filtered_clusters
from tree_ops import ClusterTree
import timeout_decorator
import unittest

class TestGetFilteredClusters(unittest.TestCase):

    def setUp(self):
        self.cluster_tree = ClusterTree()
        self.cluster_tree.clusters_dict = {1: {'stability': 10, 'feature1': 5, 'feature2': 7}, 2: {'stability': 8, 'feature1': 3, 'feature2': 9}, 3: {'stability': 6, 'feature1': 2, 'feature2': 4}, 4: {'stability': 4, 'feature1': 6, 'feature2': 8}, 5: {'stability': 2, 'feature1': 1, 'feature2': 3}}
        self.cluster_tree.field_indices = {'feature1': 0, 'feature2': 1}
        self.cluster_tree.ordered_ids = [1, 2, 3, 4, 5]

    @timeout_decorator.timeout(15)
    def test_get_filtered_clusters_scenario3(self):
        filters = {'feature1': [3, 6], 'feature2': [5, 9]}
        expected_result = {1: {'stability': 10, 'feature1': 5, 'feature2': 7}, 4: {'stability': 4, 'feature1': 6, 'feature2': 8}}
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = get_filtered_clusters(self.cluster_tree, filters)
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'get_filtered_clusters_test_get_filtered_clusters_scenario3_2')
if __name__ == '__main__':
    unittest.main()"""

    inspired_test = """import unittest


class MockClusterTree:

    def __init__(self, clusters_dict, field_indices, stability_column, ordered_ids):
        self.clusters_dict = clusters_dict
        self.field_indices = field_indices
        self.stability_column = stability_column
        self.ordered_ids = ordered_ids

    def filter_cluster(self, node_id, filters):
        pass

    def get_children(self, node_id):
        pass

    def compute_subtree_stability(self, node_id, stabilities):
        pass

    def bfs_from_cluster_tree(self, node_id):
        pass

class TestGetFilteredClusters(unittest.TestCase):

    def setUp(self):
        self.cluster_tree = MockClusterTree(clusters_dict={}, field_indices={}, stability_column=None, ordered_ids=[])

    @timeout_decorator.timeout(15)
    def test_get_filtered_clusters(self):
        filters = {'feature1': [0, 10], 'feature2': [5, 15]}
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = get_filtered_clusters(self.cluster_tree, filters)
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'get_filtered_clusters_test_get_filtered_clusters_1')

    @timeout_decorator.timeout(15)
    def test_get_filtered_clusters_with_clusters(self):
        filters = {'feature1': [0, 10], 'feature2': [5, 15]}
        self.cluster_tree.filter_cluster = MagicMock(return_value=True)
        self.cluster_tree.get_children = MagicMock(return_value=[1, 2, 3])
        self.cluster_tree.compute_subtree_stability = MagicMock(return_value=20)
        self.cluster_tree.bfs_from_cluster_tree = MagicMock(return_value=[1, 2, 3])
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = get_filtered_clusters(self.cluster_tree, filters)
        duration = time.perf_counter_ns() - counter
        gc.enable()
        _log__test__values(return_value, duration, 'get_filtered_clusters_test_get_filtered_clusters_with_clusters_5')
if __name__ == '__main__':
    unittest.main()"""

    expected = """import unittest
import time
import gc
from tree_ops import get_filtered_clusters
from tree_ops import ClusterTree
import timeout_decorator
import unittest

class TestGetFilteredClusters(unittest.TestCase):

    def setUp(self):
        self.cluster_tree = ClusterTree()
        self.cluster_tree.clusters_dict = {1: {'stability': 10, 'feature1': 5, 'feature2': 7}, 2: {'stability': 8, 'feature1': 3, 'feature2': 9}, 3: {'stability': 6, 'feature1': 2, 'feature2': 4}, 4: {'stability': 4, 'feature1': 6, 'feature2': 8}, 5: {'stability': 2, 'feature1': 1, 'feature2': 3}}
        self.cluster_tree.field_indices = {'feature1': 0, 'feature2': 1}
        self.cluster_tree.ordered_ids = [1, 2, 3, 4, 5]

    @timeout_decorator.timeout(15)
    def test_get_filtered_clusters_scenario3(self):
        filters = {'feature1': [3, 6], 'feature2': [5, 9]}
        expected_result = {1: {'stability': 10, 'feature1': 5, 'feature2': 7}, 4: {'stability': 4, 'feature1': 6, 'feature2': 8}}
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = get_filtered_clusters(self.cluster_tree, filters)
"""
    expected += """        duration = time.perf_counter_ns() - counter\n"""
    expected += """        gc.enable()
        _log__test__values(return_value, duration, 'get_filtered_clusters_test_get_filtered_clusters_scenario3_2')

"""
    expected += """class MockClusterTree:\n"""
    expected += """
    def __init__(self, clusters_dict, field_indices, stability_column, ordered_ids):
        self.clusters_dict = clusters_dict
        self.field_indices = field_indices
        self.stability_column = stability_column
        self.ordered_ids = ordered_ids

    def filter_cluster(self, node_id, filters):
        pass

    def get_children(self, node_id):
        pass

    def compute_subtree_stability(self, node_id, stabilities):
        pass

    def bfs_from_cluster_tree(self, node_id):
        pass

class TestGetFilteredClustersInspired(unittest.TestCase):

    def setUp(self):
        self.cluster_tree = MockClusterTree(clusters_dict={}, field_indices={}, stability_column=None, ordered_ids=[])

    @timeout_decorator.timeout(15)
    def test_get_filtered_clusters(self):
        filters = {'feature1': [0, 10], 'feature2': [5, 15]}
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = get_filtered_clusters(self.cluster_tree, filters)
"""
    expected += """        duration = time.perf_counter_ns() - counter\n"""
    expected += """        gc.enable()
        _log__test__values(return_value, duration, 'get_filtered_clusters_test_get_filtered_clusters_1')

    @timeout_decorator.timeout(15)
    def test_get_filtered_clusters_with_clusters(self):
        filters = {'feature1': [0, 10], 'feature2': [5, 15]}
        self.cluster_tree.filter_cluster = MagicMock(return_value=True)
        self.cluster_tree.get_children = MagicMock(return_value=[1, 2, 3])
        self.cluster_tree.compute_subtree_stability = MagicMock(return_value=20)
        self.cluster_tree.bfs_from_cluster_tree = MagicMock(return_value=[1, 2, 3])
        gc.disable()
        counter = time.perf_counter_ns()
        return_value = get_filtered_clusters(self.cluster_tree, filters)
"""
    expected += """        duration = time.perf_counter_ns() - counter\n"""
    expected += """        gc.enable()
        _log__test__values(return_value, duration, 'get_filtered_clusters_test_get_filtered_clusters_with_clusters_5')
"""
    expected += """if __name__ == '__main__':
    unittest.main()"""

    modified_file = merge_unit_tests(unit_tests, inspired_test, "unittest")
    assert modified_file == expected
