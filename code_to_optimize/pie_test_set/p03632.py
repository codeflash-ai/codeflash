def problem_p03632():
    import sys

    sys.setrecursionlimit(4100000)

    import math

    import itertools

    INF = float("inf")

    from heapq import heapify, heappop, heappush

    def main():

        a, b, c, d = list(map(int, input().split()))

        tmp = set(list(range(a, b + 1))) & set(list(range(c, d + 1)))

        print((max(0, len(tmp) - 1)))

    if __name__ == "__main__":

        main()


problem_p03632()
