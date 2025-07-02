def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Use built-in Timsort for very fast, O(n log n) sort
    print(f"result: {arr}")
    return arr
