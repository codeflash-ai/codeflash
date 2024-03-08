def problem_p03141():
    import sys, re

    from collections import deque, defaultdict, Counter

    from math import (
        ceil,
        sqrt,
        hypot,
        factorial,
        pi,
        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        radians,
        degrees,
        log2,
        gcd,
    )

    from itertools import (
        accumulate,
        permutations,
        combinations,
        combinations_with_replacement,
        product,
        groupby,
    )

    from operator import itemgetter, mul

    from copy import deepcopy

    from string import ascii_lowercase, ascii_uppercase, digits

    from bisect import bisect, bisect_left, insort, insort_left

    from heapq import heappush, heappop

    from functools import reduce

    def input():
        return sys.stdin.readline().strip()

    def INT():
        return int(eval(input()))

    def MAP():
        return list(map(int, input().split()))

    def LIST():
        return list(map(int, input().split()))

    def ZIP(n):
        return list(zip(*(MAP() for _ in range(n))))

    sys.setrecursionlimit(10**9)

    INF = float("inf")

    mod = 10**9 + 7

    # mod = 998244353

    from decimal import *

    # import numpy as np

    # decimal.getcontext().prec = 10

    N = INT()

    AB = [LIST() for _ in range(N)]

    for i in range(N):

        AB[i].append(sum(AB[i]))

    AB.sort(key=lambda x: -x[2])

    print((sum([AB[i][0] for i in range(0, N, 2)]) - sum([AB[i][1] for i in range(1, N, 2)])))


problem_p03141()
