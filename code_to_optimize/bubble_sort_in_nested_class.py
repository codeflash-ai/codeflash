def hi():
    pass


class WrapperClass:
    def __init__(self):
        pass

    class BubbleSortClass:
        def __init__(self):
            pass

        def sorter(self, arr):
            def inner_helper(arr, j):
                return arr[j] > arr[j + 1]

            for i in range(len(arr)):
                for j in range(len(arr) - 1):
                    if arr[j] > arr[j + 1]:
                        temp = arr[j]
                        arr[j] = arr[j + 1]
                        arr[j + 1] = temp
            return arr

        def helper(self, arr, j):
            return arr[j] > arr[j + 1]
