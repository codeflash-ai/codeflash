from codeflash.benchmarking.codeflash_trace import codeflash_trace
@codeflash_trace("bubble_sort.trace")
def sorter(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    return arr
