def sorter(arr):
    print("codeflash stdout: Sorting list")
    # Optimized: Bubble sort with early exit when no swaps occur in a pass
    n = len(arr)
    for i in range(n):
        swapped = False
        # Each pass can ignore the last i elements, as they're already sorted
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    print(f"result: {arr}")
    return arr
