def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        # Each pass can skip the last i elements, which are already sorted
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            # If no swaps occurred, the array is already sorted
            break
    print(f"result: {arr}")
    return arr
