def sorter(arr):
    # Print sorting notice
    print("codeflash stdout: Sorting list")

    # Use Timsort for optimal performance
    arr.sort()

    # Print sorted result
    print(f"result: {arr}")
    return arr
