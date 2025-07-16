def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        # After each i, the last i elements are in place
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                # Swap elements directly, no temp necessary
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # If no two elements were swapped by inner loop, then array is sorted
        if not swapped:
            break
    print(f"result: {arr}")
    return arr
