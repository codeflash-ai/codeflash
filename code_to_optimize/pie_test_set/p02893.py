def problem_p02893():
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

    INF = 10**13

    def LI():
        return list(map(int, sys.stdin.readline().split()))

    def I():
        return int(sys.stdin.readline())

    def LS():
        return sys.stdin.readline().rstrip().split()

    def S():
        return sys.stdin.readline().rstrip()

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

    mod = 998244353

    n = I()

    x = S()

    rx = x.replace("1", "2").replace("0", "1").replace("2", "0")

    D = []

    for i in range(1, int(n**0.5) + 1):

        if n % i == 0:

            if n // i % 2:

                D += [i]

            if i != n // i and i % 2:

                D += [n // i]

    D.sort()

    bx = int(x, 2)

    cnt = [0] * len(D)

    ans = 0

    for k in range(len(D)):

        t = x[: D[k]]

        rt = rx[: D[k]]

        f = (t + rt) * (n // D[k] // 2) + t

        cnt[k] = cnt[k] + int(int(f, 2) <= bx) + int(t, 2)

        cnt[k] %= mod

        ans += cnt[k] * D[k] * 2

        for kk in range(k + 1, len(D)):

            if D[kk] % D[k] == 0:

                cnt[kk] -= cnt[k]

                cnt[kk] %= mod

        ans %= mod

    print((ans % mod))


problem_p02893()
