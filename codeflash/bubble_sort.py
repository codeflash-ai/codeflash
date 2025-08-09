def mysorter(arr):
    print("codeflash stdout: Sorting list")
    # Use the built-in sort for faster performance
    arr.sort()
    print(f"result: {arr}")
    return arr


mysorter([5, 4, 3, 2, 1])
