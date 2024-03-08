def problem_p02548():
    import sys, os, math, bisect, itertools, collections, heapq, queue, copy, array

    # from scipy.sparse.csgraph import csgraph_from_dense, floyd_warshall

    # from decimal import Decimal

    # from collections import defaultdict, deque

    sys.setrecursionlimit(10000000)

    ii = lambda: int(sys.stdin.buffer.readline().rstrip())

    il = lambda: list(map(int, sys.stdin.buffer.readline().split()))

    fl = lambda: list(map(float, sys.stdin.buffer.readline().split()))

    iln = lambda n: [int(sys.stdin.buffer.readline().rstrip()) for _ in range(n)]

    iss = lambda: sys.stdin.buffer.readline().decode().rstrip()

    sl = lambda: list(map(str, sys.stdin.buffer.readline().decode().split()))

    isn = lambda n: [sys.stdin.buffer.readline().decode().rstrip() for _ in range(n)]

    lcm = lambda x, y: (x * y) // math.gcd(x, y)

    MOD = 10**9 + 7

    INF = float("inf")

    def main():

        if os.getenv("LOCAL"):

            sys.stdin = open("input.txt", "r")

        N = ii()

        ret = 0

        for a in range(1, N):

            b = N // a

            if N - (a * b) <= 0:

                ret += b - 1

            else:

                ret += b

        print(ret)

    if __name__ == "__main__":

        main()


problem_p02548()
