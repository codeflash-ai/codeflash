def sorter(arr):
    print("codeflash stdout: Sorting list")
    # Use the highly optimized built-in sort for plain lists to improve performance.
    # Fall back to the original bubble-sort behavior for non-list inputs to preserve
    # the original exception semantics and side-effects.
    if isinstance(arr, list):
        arr.sort()
        print(f"result: {arr}")
        return arr
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    print(f"result: {arr}")
    return arr
