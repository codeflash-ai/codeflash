def sorter(arr):
    n = len(arr)
    while True:
        swapped = False
        # After every full pass, the largest item is at the end
        newn = 0  # store last index involved in a swap
        for i in range(1, n):
            if arr[i - 1] > arr[i]:
                arr[i - 1], arr[i] = arr[i], arr[i - 1]  # in-place swap
                swapped = True
                newn = i  # last swap position
        n = newn  # after this, items after newn are already sorted
        if not swapped:
            break
    return arr
