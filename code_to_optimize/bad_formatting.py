import sys


def lol():
    print(       "lol" )









class BubbleSorter:
    def __init__(self, x=0):
        self.x = x

    def lol(self):
        print(       "lol" )








    def sorter  (self, arr):
        
        
        print       ("codeflash stdout : BubbleSorter.sorter() called")
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in                    range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j],     arr[j + 1] =        arr[j + 1], arr[j]  # Faster swap
                    swapped =   True
            if not swapped:
                break
        print           ("stderr test", file=sys.stderr)
        return arr