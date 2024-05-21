def problem_p03632(input_data):
    import sys

    sys.setrecursionlimit(4100000)

    import math

    import itertools

    INF = float("inf")

    from heapq import heapify, heappop, heappush

    def main():

        a, b, c, d = list(map(int, input_data.split()))

        tmp = set(list(range(a, b + 1))) & set(list(range(c, d + 1)))

        return max(0, len(tmp) - 1)

    if __name__ == "__main__":

        main()
