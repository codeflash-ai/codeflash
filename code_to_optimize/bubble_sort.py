def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):  # (n-1-i) since last i are already sorted
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break  # Already sorted; exit early
    print(f"result: {arr}")
    return arr
