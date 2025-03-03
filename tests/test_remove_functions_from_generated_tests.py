from pathlib import Path

import pytest

from codeflash.code_utils.remove_generated_tests import remove_functions_from_generated_tests
from codeflash.models.models import GeneratedTests, GeneratedTestsList


def test_simple_removal():
    generated_test_source = """def test_empty_list():
    # Test sorting an empty list
    codeflash_output = sorter([])
    # Outputs were verified to be equal to the original implementation

def test_single_element():
    # Test sorting a list with a single element
    codeflash_output = sorter([1])
    # Outputs were verified to be equal to the original implementation
    
def test_sorted_list():
    # Test sorting an already sorted list
    codeflash_output = sorter([1, 2, 3, 4, 5])
    # Outputs were verified to be equal to the original implementation"""
    generated_tests = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="",
        behavior_file_path=Path("test_sorter.py"),
        perf_file_path=Path("test_sorter.py"),
        instrumented_perf_test_source="",
    )
    generated_tests_list = GeneratedTestsList(generated_tests=[generated_tests])
    functions_to_remove = ["test_single_element"]

    expected = """def test_empty_list():
    # Test sorting an empty list
    codeflash_output = sorter([])
    # Outputs were verified to be equal to the original implementation


def test_sorted_list():
    # Test sorting an already sorted list
    codeflash_output = sorter([1, 2, 3, 4, 5])
    # Outputs were verified to be equal to the original implementation"""

    generated_tests = remove_functions_from_generated_tests(generated_tests_list, functions_to_remove)

    assert generated_tests_list.generated_tests[0].generated_original_test_source == expected


def test_multiple_removals():
    generated_test_source = """def test_empty_list():
    # Test sorting an empty list
    codeflash_output = sorter([])
    # Outputs were verified to be equal to the original implementation

def test_single_element():
    # Test sorting a list with a single element
    codeflash_output = sorter([1])
    # Outputs were verified to be equal to the original implementation

def test_sorted_list():
    # Test sorting an already sorted list
    codeflash_output = sorter([1, 2, 3, 4, 5])
    # Outputs were verified to be equal to the original implementation"""
    generated_tests = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="",
        behavior_file_path=Path("test_sorter.py"),
        perf_file_path=Path("test_sorter.py"),
        instrumented_perf_test_source="",
    )
    generated_tests_list_1 = GeneratedTestsList(generated_tests=[generated_tests])
    functions_to_remove = ["test_single_element", "test_sorted_list"]

    expected = """def test_empty_list():
    # Test sorting an empty list
    codeflash_output = sorter([])
    # Outputs were verified to be equal to the original implementation


"""
    generated_tests_1 = remove_functions_from_generated_tests(generated_tests_list_1, functions_to_remove)
    assert generated_tests_1.generated_tests[0].generated_original_test_source == expected

    functions_to_remove = ["test_single_element", "test_empty_list"]

    expected = """
def test_sorted_list():
    # Test sorting an already sorted list
    codeflash_output = sorter([1, 2, 3, 4, 5])
    # Outputs were verified to be equal to the original implementation"""

    generated_tests_2 = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="",
        behavior_file_path=Path("test_sorter.py"),
        perf_file_path=Path("test_sorter.py"),
        instrumented_perf_test_source="",
    )

    generated_tests_list_2 = GeneratedTestsList(generated_tests=[generated_tests_2])

    generated_tests_2 = remove_functions_from_generated_tests(generated_tests_list_2, functions_to_remove)
    assert generated_tests_list_2.generated_tests[0].generated_original_test_source == expected


def test_remove_complex_functions():
    generated_test_source = """def test_list_with_complex_numbers():
    # Test with a list containing complex numbers
    with pytest.raises(TypeError):
        sorter([3 + 2j, 1 + 1j, 4 + 0j, 2 + 3j])
    with pytest.raises(TypeError):
        sorter([0 + 1j, -1 + 0j, 3 + 3j, -2 + 2j])
    # Outputs were verified to be equal to the original implementation


def test_list_with_custom_objects():
    # Test with a list containing custom objects
    class CustomObject:
        def __init__(self, value):
            self.value = value
            # Outputs were verified to be equal to the original implementation

        def __lt__(self, other):
            return self.value < other.value
            # Outputs were verified to be equal to the original implementation

        def __gt__(self, other):
            return self.value > other.value
            # Outputs were verified to be equal to the original implementation

    codeflash_output = sorter([CustomObject(3), CustomObject(1), CustomObject(2)])
    codeflash_output = sorter([3, CustomObject(1), 4, CustomObject(2)])
    # Outputs were verified to be equal to the original implementation


def test_list_with_mixed_orderable_and_non_orderable_types():
    # Test with a list containing a mix of orderable and non-orderable types
    with pytest.raises(TypeError):
        sorter([1, "a", 3.5, None])
    with pytest.raises(TypeError):
        sorter([True, 1, "string", [1, 2]])
    # Outputs were verified to be equal to the original implementation"""

    generated_tests = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="",
        behavior_file_path=Path("test_sorter.py"),
        perf_file_path=Path("test_sorter.py"),
        instrumented_perf_test_source="",
    )
    generated_tests_list = GeneratedTestsList(generated_tests=[generated_tests])
    functions_to_remove = ["test_list_with_custom_objects"]

    expected = """def test_list_with_complex_numbers():
    # Test with a list containing complex numbers
    with pytest.raises(TypeError):
        sorter([3 + 2j, 1 + 1j, 4 + 0j, 2 + 3j])
    with pytest.raises(TypeError):
        sorter([0 + 1j, -1 + 0j, 3 + 3j, -2 + 2j])
    # Outputs were verified to be equal to the original implementation



def test_list_with_mixed_orderable_and_non_orderable_types():
    # Test with a list containing a mix of orderable and non-orderable types
    with pytest.raises(TypeError):
        sorter([1, "a", 3.5, None])
    with pytest.raises(TypeError):
        sorter([True, 1, "string", [1, 2]])
    # Outputs were verified to be equal to the original implementation"""

    generated_tests = remove_functions_from_generated_tests(generated_tests_list, functions_to_remove)
    assert generated_tests.generated_tests[0].generated_original_test_source == expected


def test_keep_parametrized_tests():
    generated_test_source = """def test_empty_list():
    # Test sorting an empty list
    codeflash_output = sorter([])
    # Outputs were verified to be equal to the original implementation

def test_single_element():
    # Test sorting a list with a single element
    codeflash_output = sorter([1])
    # Outputs were verified to be equal to the original implementation
    
@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(5000))), list(range(5000))),
    ],
)
def test_sort_parametrized(input, expected_output):
    output = sorter(input)
    assert output == expected_output

def test_sorted_list():
    # Test sorting an already sorted list
    codeflash_output = sorter([1, 2, 3, 4, 5])
    # Outputs were verified to be equal to the original implementation"""
    generated_tests = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="",
        behavior_file_path=Path("test_sorter.py"),
        perf_file_path=Path("test_sorter.py"),
        instrumented_perf_test_source="",
    )
    generated_tests_list = GeneratedTestsList(generated_tests=[generated_tests])
    functions_to_remove = ["test_empty_list", "test_sort_parametrized"]

    expected = """
def test_single_element():
    # Test sorting a list with a single element
    codeflash_output = sorter([1])
    # Outputs were verified to be equal to the original implementation
    
@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(5000))), list(range(5000))),
    ],
)
def test_sort_parametrized(input, expected_output):
    output = sorter(input)
    assert output == expected_output

def test_sorted_list():
    # Test sorting an already sorted list
    codeflash_output = sorter([1, 2, 3, 4, 5])
    # Outputs were verified to be equal to the original implementation"""

    generated_tests = remove_functions_from_generated_tests(generated_tests_list, functions_to_remove)
    assert generated_tests_list.generated_tests[0].generated_original_test_source == expected


@pytest.mark.skip("We don't handle the edge case where the parametrized test appears right after the test to remove")
def test_keep_parametrized_test2():
    generated_test_source = """def test_empty_list():
    # Test sorting an empty list
    codeflash_output = sorter([])
    # Outputs were verified to be equal to the original implementation

@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(5000))), list(range(5000))),
    ],
)
def test_sort_parametrized(input, expected_output):
    output = sorter(input)
    assert output == expected_output
    
def test_single_element():
    # Test sorting a list with a single element
    codeflash_output = sorter([1])
    # Outputs were verified to be equal to the original implementation

def test_sorted_list():
    # Test sorting an already sorted list
    codeflash_output = sorter([1, 2, 3, 4, 5])
    # Outputs were verified to be equal to the original implementation"""
    generated_tests = GeneratedTests(
        generated_original_test_source=generated_test_source,
        instrumented_behavior_test_source="",
        behavior_file_path=Path("test_sorter.py"),
        perf_file_path=Path("test_sorter.py"),
        instrumented_perf_test_source="",
    )
    generated_tests_list = GeneratedTestsList(generated_tests=[generated_tests])
    functions_to_remove = ["test_empty_list", "test_sort_parametrized"]

    expected = """
@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(5000))), list(range(5000))),
    ],
)
def test_sort_parametrized(input, expected_output):
    output = sorter(input)
    assert output == expected_output
    
def test_single_element():
    # Test sorting a list with a single element
    codeflash_output = sorter([1])
    # Outputs were verified to be equal to the original implementation

def test_sorted_list():
    # Test sorting an already sorted list
    codeflash_output = sorter([1, 2, 3, 4, 5])
    # Outputs were verified to be equal to the original implementation"""

    generated_tests = remove_functions_from_generated_tests(generated_tests_list, functions_to_remove)
    assert generated_tests_list.generated_tests[0].generated_original_test_source == expected
