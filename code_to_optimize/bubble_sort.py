def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Use Python's built-in Timsort which is much faster than bubble sort
    print(f"result: {arr}")
    return arr
