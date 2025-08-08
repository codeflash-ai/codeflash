def mysorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Use built-in Timsort for much faster sorting
    print(f"result: {arr}")
    return arr


mysorter([5, 4, 3, 2, 1])
