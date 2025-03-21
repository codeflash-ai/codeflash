

def sorter_deps(arr):
    n = len(arr)
    for i in range(n):
        # Introduce a flag that will allow us to exit early if no swaps occur
        swapped = False
        # Only iterate to the unsorted portion of the list
        for j in range(n - 1 - i):
            # Inline the comparison function
            if arr[j] > arr[j + 1]:
                # Inline the swap
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # If no swaps occurred in the last inner loop, the array is sorted
        if not swapped:
            break
    return arr
