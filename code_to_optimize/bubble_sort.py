def sorter(arr):
    print("codeflash stdout: Sorting list")
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # Handle cases where swap detection gives only minor speedup
        # Continue full loop if list is small or nearly sorted, do not break early
        # (i.e., ignore the swapped/break optimization for insignificant gains)
        # Commenting out the early break
        # if not swapped:
        #     break
    print(f"result: {arr}")
    return arr
