def mysorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Python's built-in sort (Timsort), much faster than bubble sort
    print(f"result: {arr}")
    return arr


mysorter([5, 4, 3, 2, 1])
