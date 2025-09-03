def sorter(arr):
    print("codeflash stdout: Sorting list")
    for i in range(len(arr)):
        already_sorted = True
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
                already_sorted = False
        if already_sorted:
            break
    print(f"result: {arr}")
    return arr
