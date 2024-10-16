def problem_p03797(input_data):
    import bisect
    import math
    import random
    import string
    import sys
    from bisect import bisect_left, bisect_right
    from collections import Counter, defaultdict, deque
    from functools import reduce
    from heapq import heapify, heappop, heappush
    from itertools import accumulate, combinations, permutations, product
    from math import ceil, factorial, floor
    from operator import mul

    sys.setrecursionlimit(2147483647)

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

    mod = 1000000007

    n, m = LI()

    if n * 2 >= m:

        return m // 2

    else:

        return n + (m - (n * 2)) // 4
