def problem_p02965():
    from collections import defaultdict, deque, Counter

    from heapq import heappush, heappop, heapify

    import math

    from bisect import bisect_left, bisect_right

    import random

    from itertools import permutations, accumulate, combinations

    import sys

    import string

    INF = float("inf")

    def LI():
        return list(map(int, sys.stdin.readline().split()))

    def I():
        return int(sys.stdin.readline())

    def LS():
        return sys.stdin.readline().split()

    def S():
        return sys.stdin.readline().strip()

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

    n, m = LI()

    total = m * 3

    fac = [1] * (total + n + 1)

    inv = [1] * (total + n + 1)

    for i in range(total + n):

        fac[i + 1] = fac[i] * (i + 1) % mod

    inv[total + n] = pow(fac[-1], mod - 2, mod)

    for j in range(total + n - 1, -1, -1):

        inv[j] = inv[j + 1] * (j + 1) % mod

    def comb(n, r):

        if r > n:

            return 0

        return fac[n] * inv[n - r] * inv[r] % mod

    ans = comb(total + n - 1, n - 1)

    for i in range(m + 2, min(n + 1, total + 1)):

        if (total - i) % 2 == 0:

            ans -= comb(n, i) * comb(n + (total - i) // 2 - 1, n - 1) % mod

            ans %= mod

    ret = 0

    for i in range(m):

        ret = (ret + comb(i + n - 2, n - 2)) % mod

    ans -= (ret * n) % mod

    print((ans % mod))


problem_p02965()
