def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Use Python's built-in Timsort which is highly optimized (O(n log n))
    print(f"result: {arr}")
    return arr
