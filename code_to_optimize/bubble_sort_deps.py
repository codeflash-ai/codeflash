def sorter_deps(arr):
    n = len(arr)
    for i in range(n):
        # We use a flag to check if the array is already sorted
        swapped = False
        # Reduce the range of j, since the last i elements are already sorted
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                # Swap without a helper function
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # If no elements were swapped in the inner loop, break
        if not swapped:
            break
    return arr
