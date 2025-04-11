
from codeflash.benchmarking.codeflash_trace import codeflash_trace


@codeflash_trace
def bubble_sort_with_unused_socket(data_container):
    # Extract the list to sort, leaving the socket untouched
    numbers = data_container.get('numbers', []).copy()

    return sorted(numbers)

@codeflash_trace
def bubble_sort_with_used_socket(data_container):
    # Extract the list to sort, leaving the socket untouched
    numbers = data_container.get('numbers', []).copy()
    socket = data_container.get('socket')
    socket.send("Hello from the optimized function!")
    return sorted(numbers)
