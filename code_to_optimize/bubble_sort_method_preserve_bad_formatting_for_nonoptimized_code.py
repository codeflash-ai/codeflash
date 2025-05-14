import sys


def lol():
    print(       "lol" )









class BubbleSorter:
    def __init__(self, x=0):
        self.x = x

    def lol(self):
        print(       "lol" )








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
