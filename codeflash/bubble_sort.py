def mysorter(arr):
    arr.sort()  # Use Python's highly optimized Timsort
    print(f"codeflash stdout: Sorting list\nresult: {arr}")
    return arr


mysorter([5, 4, 3, 2, 1])
