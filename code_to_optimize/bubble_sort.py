def sorter(arr):
    print("codeflash stdout: Sorting list")

    # Using Python's built-in sort method, which is optimized (Timsort)
    arr.sort()

    print(f"result: {arr}")
    return arr
