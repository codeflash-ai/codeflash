def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        # Reduce inner loop to skip already sorted tail
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Faster Pythonic swap
                swapped = True
        if not swapped:
            break  # Stop if sorted
    print(f"result: {arr}")
    return arr
