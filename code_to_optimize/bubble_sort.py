def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Faster tuple unpacking
                swapped = True
        if not swapped:
            break  # Stop if the list is already sorted
    print(f"result: {arr}")
    return arr
