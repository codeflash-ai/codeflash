def problem_p02607():
    import sys

    from collections import deque

    import numpy as np

    import math

    sys.setrecursionlimit(10**6)

    def S():
        return sys.stdin.readline().rstrip()

    def SL():
        return list(map(str, sys.stdin.readline().rstrip().split()))

    def I():
        return int(sys.stdin.readline().rstrip())

    def IL():
        return list(map(int, sys.stdin.readline().rstrip().split()))

    def Main():

        n = I()

        a = list(IL())

        c = 0

        for rep in range(0, n, 2):

            if a[rep] % 2:

                c += 1

        print(c)

        return

    if __name__ == "__main__":

        Main()


problem_p02607()
