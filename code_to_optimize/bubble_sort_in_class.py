def hi():
    pass


class BubbleSortClass:
    def __init__(self):
        pass

    def sorter(self, arr):
        n = len(arr)
        for i in range(n):
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def helper(self, arr, j):
        return arr[j] > arr[j + 1]
