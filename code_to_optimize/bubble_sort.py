def sorter(arr):
    print("codeflash stdout: Sorting list")
    # Use the object's in-place sort if available (fast, O(n log n)).
    # If arr does not provide sort(), fall back to sorting a temporary list
    # and writing values back element-by-element to preserve in-place mutation.
    try:
        arr.sort()
    except (AttributeError, TypeError):
        tmp = list(arr)
        tmp.sort()
        for i in range(len(tmp)):
            arr[i] = tmp[i]
    print(f"result: {arr}")
    return arr
