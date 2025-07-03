def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        # Last i elements are in place already
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Faster swap
                swapped = True
        if not swapped:
            break  # List is sorted
    print(f"result: {arr}")
    return arr
