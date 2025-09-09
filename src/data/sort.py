def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    # Optimized Bubble Sort: Stop early if no swaps in an iteration
    for i in range(n):
        swapped = False
        # After each pass, the largest i elements are in place, so skip them
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    print(f"result: {arr}")
    return arr
