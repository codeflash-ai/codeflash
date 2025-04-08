def bubble_sort_with_unused_socket(data_container):
    """
    Performs a bubble sort on a list within the data_container. The data container has the following schema:
    - 'numbers' (list): The list to be sorted.
    - 'socket' (socket): A socket

    Args:
        data_container: A dictionary with at least 'numbers' (list) and 'socket' keys

    Returns:
        list: The sorted list of numbers
    """
    # Extract the list to sort, leaving the socket untouched
    numbers = data_container.get('numbers', []).copy()

    # Classic bubble sort implementation
    n = len(numbers)
    for i in range(n):
        # Flag to optimize by detecting if no swaps occurred
        swapped = False

        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Swap if the element is greater than the next element
            if numbers[j] > numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
                swapped = True

        # If no swapping occurred in this pass, the list is sorted
        if not swapped:
            break

    return numbers


def bubble_sort_with_used_socket(data_container):
    """
    Performs a bubble sort on a list within the data_container. The data container has the following schema:
    - 'numbers' (list): The list to be sorted.
    - 'socket' (socket): A socket

    Args:
        data_container: A dictionary with at least 'numbers' (list) and 'socket' keys

    Returns:
        list: The sorted list of numbers
    """
    # Extract the list to sort and socket
    numbers = data_container.get('numbers', []).copy()
    socket = data_container.get('socket')

    # Track swap count
    swap_count = 0

    # Classic bubble sort implementation
    n = len(numbers)
    for i in range(n):
        # Flag to optimize by detecting if no swaps occurred
        swapped = False

        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Swap if the element is greater than the next element
            if numbers[j] > numbers[j + 1]:
                # Perform the swap
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
                swapped = True
                swap_count += 1

        # If no swapping occurred in this pass, the list is sorted
        if not swapped:
            break

    # Send final summary
    summary = f"Bubble sort completed with {swap_count} swaps"
    socket.send(summary.encode())

    return numbers