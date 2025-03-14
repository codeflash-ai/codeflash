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
