def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # use the built-in highly optimized sort instead of bubble sort
    print(f"result: {arr}")  # this preserves your output
    return arr
