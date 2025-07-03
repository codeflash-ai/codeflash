def sorter(arr):
    print("codeflash stdout: Sorting list")
    # Using built-in sort (Timsort) for much faster performance
    arr.sort()
    print(f"result: {arr}")
    return arr
