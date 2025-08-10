def mysorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Use Python's highly optimized Timsort
    print(f"result: {arr}")
    return arr


mysorter([5, 4, 3, 2, 1])
