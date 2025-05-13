def sorter(arr):
    print("codeflash stdout: Sorting list")
    # Use Python's efficient built-in sort (Timsort), which is much faster than bubble sort.
    arr.sort()
    print(f"result: {arr}")
    return arr
