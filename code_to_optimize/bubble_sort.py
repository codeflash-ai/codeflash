def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # much faster built-in sort (Timsort, O(n log n))
    print(f"result: {arr}")
    return arr
