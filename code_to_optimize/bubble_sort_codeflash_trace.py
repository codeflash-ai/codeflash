from codeflash.benchmarking.codeflash_trace import codeflash_trace
@codeflash_trace
def sorter(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    return arr

@codeflash_trace
def recursive_bubble_sort(arr, n=None):
    # Initialize n if not provided
    if n is None:
        n = len(arr)

    # Base case: if n is 1, the array is already sorted
    if n == 1:
        return arr

    # One pass of bubble sort - move the largest element to the end
    for i in range(n - 1):
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]

    # Recursively sort the remaining n-1 elements
    return recursive_bubble_sort(arr, n - 1)

class Sorter:
    @codeflash_trace
    def __init__(self, arr):
        self.arr = arr
    @codeflash_trace
    def sorter(self, multiplier):
        for i in range(len(self.arr)):
            for j in range(len(self.arr) - 1):
                if self.arr[j] > self.arr[j + 1]:
                    temp = self.arr[j]
                    self.arr[j] = self.arr[j + 1]
                    self.arr[j + 1] = temp
        return self.arr * multiplier

    @staticmethod
    @codeflash_trace
    def sort_static(arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr

    @classmethod
    @codeflash_trace
    def sort_class(cls, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr
