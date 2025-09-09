def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    # Optimized Bubble Sort: Each outer pass moves largest element to the end,
    # so inner loop can avoid already sorted tail
    for i in range(n):
        swapped = False
        # Only compare up to n-i-1, as the last i elements are already sorted
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Swap in one shot
                swapped = True
        if not swapped:
            break  # Stop if no swaps were made, i.e., array is sorted
    print(f"result: {arr}")
    return arr
