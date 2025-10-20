def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    # Optimized Bubble Sort: stop early if no swaps and reduce comparisons per pass
    for i in range(n):
        swapped = False
        # The last i elements are already sorted
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            # No swaps means list is sorted
            break
    print(f"result: {arr}")
    return arr
