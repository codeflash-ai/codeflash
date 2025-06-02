def hi():
    pass


class BubbleSortClass:
    def __init__(self):
        pass

    def sorter(self, arr):
        # Use Python's highly optimized built-in sort for faster performance
        arr.sort()
        return arr

    def helper(self, arr, j):
        return arr[j] > arr[j + 1]
