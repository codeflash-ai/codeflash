def mysorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # Use the built-in Timsort for better performance
    print(f"result: {arr}")
    return arr


mysorter([5, 4, 3, 2, 1])
