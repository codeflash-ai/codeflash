def mysorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # use built-in sort which is highly optimized
    print(f"result: {arr}")
    return arr


mysorter([5, 4, 3, 2, 1])
