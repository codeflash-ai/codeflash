def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()  # much faster than hand-written bubble sort
    print(f"result: {arr}")
    summed_value = sum(arr[:3])
    return arr, summed_value
