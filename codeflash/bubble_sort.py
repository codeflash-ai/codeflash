def sorter(arr):
    print("codeflash stdout: Sorting list")
    # Uses Python's built-in Timsort algorithm, much faster than insertion sort or bubble sort
    arr.sort()
    print(f"result: {arr}")
    return arr
