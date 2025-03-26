def sorter(arr):
    print("codeflash stdout: Sorting list")

    # Using built-in sort method which uses Timsort with a time complexity of O(n log n)
    arr.sort()

    print(f"result: {arr}")
    return arr
