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
        return arr
