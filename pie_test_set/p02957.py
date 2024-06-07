def problem_p02957(input_data):
    import math, string, itertools, fractions, heapq, collections, re, array, bisect, sys, random, time, queue, copy

    sys.setrecursionlimit(10**7)

    inf = 10**20

    mod = 10**9 + 7

    dd = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    ddn = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    def LI():
        return [int(x) for x in sys.stdin.readline().split()]

    def LI_():
        return [int(x) - 1 for x in sys.stdin.readline().split()]

    def I():
        return int(sys.stdin.readline())

    def LS():
        return sys.stdin.readline().split()

    def S():
        return eval(input_data)

    def main():

        a, b = LI()

        x = max(a, b) - min(a, b)

        if x % 2 == 1:

            return "IMPOSSIBLE"

        return max(a, b) - x // 2

    # main()

    return main()
