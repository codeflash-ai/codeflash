def sorter(arr):
    print("codeflash stdout: Sorting list")
    # Using Python's built-in sort for better performance (Timsort, O(n log n))
    arr.sort()
    print(f"result: {arr}")
    return arr
