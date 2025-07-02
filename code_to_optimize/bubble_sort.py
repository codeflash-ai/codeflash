def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Use Timsort for fast in-place sorting
    print(f"result: {arr}")
    return arr
