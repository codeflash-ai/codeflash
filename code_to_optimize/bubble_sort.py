def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):  # don't check the sorted tail
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Use tuple swap
                swapped = True
        if not swapped:
            break  # No swaps means already sorted
    print(f"result: {arr}")
    return arr
