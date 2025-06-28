def sorter(arr):
    print("codeflash stdout: Sorting list")
    # Faster than bubble sort for small arrays, but not as fast as .sort()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    print(f"result: {arr}")
    return arr
