def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Use Python's built-in in-place sort (TimSort, O(n log n))
    print(f"result: {arr}")
    return arr
