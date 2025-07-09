def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        # Stop after n - i - 1, as the last i elements are already sorted
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Faster swap
                swapped = True
        if not swapped:
            break  # List is sorted; no need for further passes
    print(f"result: {arr}")
    return arr
