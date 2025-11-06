import sys


class BubbleSorter:
    def __init__(self, x=0):
        self.x = x

    def sorter(self, arr):
        print("codeflash stdout : BubbleSorter.sorter() called")
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        print("stderr test", file=sys.stderr)
        return arr

    @classmethod
    def sorter_classmethod(cls, arr):
        print("codeflash stdout : BubbleSorter.sorter_classmethod() called")
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        print("stderr test classmethod", file=sys.stderr)
        return arr

    @staticmethod
    def sorter_staticmethod(arr):
        print("codeflash stdout : BubbleSorter.sorter_staticmethod() called")
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        print("stderr test staticmethod", file=sys.stderr)
        return arr