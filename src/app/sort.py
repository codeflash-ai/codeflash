def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    # Optimized bubble sort with early exit and reduced inner loop range
    for i in range(n):
        swapped = False
        # After each outer loop pass, the largest element is at the end
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    print(f"result: {arr}")
    return arr
