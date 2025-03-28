def sorter(arr):
    # Informing about the sort operation
    print("codeflash stdout: Sorting list")

    # Using Python's built-in timsort for efficiency
    arr.sort()

    # Displaying the sorted list
    print(f"result: {arr}")
    return arr
