def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # use in-place Timsort, much faster
    print(f"result: {arr}")
    summed_value = sum(arr[:3])  # use direct list slice, generator is unnecessary
    return arr, summed_value
