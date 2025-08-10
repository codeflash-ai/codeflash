def mysorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Built-in in-place Timsort, O(n log n)
    print(f"result: {arr}")
    return arr


mysorter([5, 4, 3, 2, 1])
