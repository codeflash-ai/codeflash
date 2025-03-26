def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Using Timsort under the hood for better performance
    print(f"result: {arr}")
    return arr
