def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    # Optimized bubble sort: on each pass, the largest elements settle at the end.
    for i in range(n):
        swapped = False
        # Reduce inner loop range for already-sorted tail
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                # Swap elements directly (no temp variable needed)
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # Early exit if no swaps, array is already sorted
        if not swapped:
            break
    print(f"result: {arr}")
    return arr
