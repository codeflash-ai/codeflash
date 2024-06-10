def problem_p03016(input_data):
    from collections import defaultdict, deque, Counter

    from heapq import heappush, heappop, heapify

    import math

    import bisect

    import random

    from itertools import permutations, accumulate, combinations, product

    import sys

    import string

    from bisect import bisect_left, bisect_right

    from math import factorial, ceil, floor

    from operator import mul

    from functools import reduce

    sys.setrecursionlimit(2147483647)

    INF = 10**20

    def LI():
        return list(map(int, sys.stdin.buffer.readline().split()))

    def I():
        return int(sys.stdin.buffer.readline())

    def LS():
        return sys.stdin.buffer.readline().rstrip().decode("utf-8").split()

    def S():
        return sys.stdin.buffer.readline().rstrip().decode("utf-8")

    def IR(n):
        return [I() for i in range(n)]

    def LIR(n):
        return [LI() for i in range(n)]

    def SR(n):
        return [S() for i in range(n)]

    def LSR(n):
        return [LS() for i in range(n)]

    def SRL(n):
        return [list(S()) for i in range(n)]

    def MSRL(n):
        return [[int(j) for j in list(S())] for i in range(n)]

    mod = 1000000007

    def mat_mul(a, b):

        I, J, K = len(a), len(b[0]), len(b)

        c = [[0] * J for _ in range(I)]

        for i in range(I):

            for j in range(J):

                for k in range(K):

                    c[i][j] += a[i][k] * b[k][j]

                c[i][j] %= m

        return c

    def mat_pow(x, n):

        y = [[0] * len(x) for _ in range(len(x))]

        for i in range(len(x)):

            y[i][i] = 1

        while n > 0:

            if n & 1:

                y = mat_mul(x, y)

            x = mat_mul(x, x)

            n >>= 1

        return y

    l, a, b, m = LI()

    d0 = 0

    ret = [[0], [a], [1]]

    for i in range(1, 19):

        if 10**i - 1 - a < 0:

            continue

        d1 = min((10**i - 1 - a) // b + 1, l)

        mat = [[10**i, 1, 0], [0, 1, b], [0, 0, 1]]

        ret = mat_mul(mat_pow(mat, d1 - d0), ret)

        if d1 == l:

            break

        d0 = d1

    return ret[0][0]
