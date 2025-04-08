import socket
from unittest.mock import Mock

import pytest

from code_to_optimize.bubble_sort_picklepatch import bubble_sort_with_unused_socket, bubble_sort_with_used_socket


def test_bubble_sort_with_unused_socket():
    mock_socket = Mock()
    # Test case 1: Regular unsorted list
    data_container = {
        'numbers': [5, 2, 9, 1, 5, 6],
        'socket': mock_socket
    }

    result = bubble_sort_with_unused_socket(data_container)

    # Check that the result is correctly sorted
    assert result == [1, 2, 5, 5, 6, 9]

def test_bubble_sort_with_used_socket():
    mock_socket = Mock()
    # Test case 1: Regular unsorted list
    data_container = {
        'numbers': [5, 2, 9, 1, 5, 6],
        'socket': mock_socket
    }

    result = bubble_sort_with_used_socket(data_container)

    # Check that the result is correctly sorted
    assert result == [1, 2, 5, 5, 6, 9]

