import socket

from code_to_optimize.bubble_sort_picklepatch_test_unused_socket import bubble_sort_with_unused_socket
from code_to_optimize.bubble_sort_picklepatch_test_used_socket import bubble_sort_with_used_socket

def test_socket_picklepatch(benchmark):
    s1, s2 = socket.socketpair()
    data = {
        "numbers": list(reversed(range(500))),
        "socket": s1
    }
    benchmark(bubble_sort_with_unused_socket, data)

def test_used_socket_picklepatch(benchmark):
    s1, s2 = socket.socketpair()
    data = {
        "numbers": list(reversed(range(500))),
        "socket": s1
    }
    benchmark(bubble_sort_with_used_socket, data)