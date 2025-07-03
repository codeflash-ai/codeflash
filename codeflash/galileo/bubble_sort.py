def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    arr_ = arr  # local variable for faster access
    for i in range(n):
        swapped = False
        maxj = n - i - 1
        for j in range(maxj):
            a, b = arr_[j], arr_[j + 1]
            if a > b:
                arr_[j], arr_[j + 1] = b, a
                swapped = True
        if not swapped:  # No swaps means array is sorted
            break
    print(f"result: {arr}")
    return arr
