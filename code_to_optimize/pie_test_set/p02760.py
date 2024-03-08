def problem_p02760():
    import sys, re

    from collections import deque, defaultdict, Counter

    from math import ceil, sqrt, hypot, factorial, pi, sin, cos, radians

    from itertools import accumulate, permutations, combinations, product, groupby

    from operator import itemgetter, mul

    from copy import deepcopy

    from string import ascii_lowercase, ascii_uppercase, digits

    from bisect import bisect, bisect_left

    from fractions import gcd

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

    import numpy as np

    A1 = LIST()

    A2 = LIST()

    A3 = LIST()

    N = INT()

    B = np.ones(N)

    for i in range(N):

        B[i] = INT()

    B = list(B)

    def bingo_array(A, B):

        for i in range(3):

            for x in B:

                if A[i] == x:

                    A[i] *= 0

    bingo_array(A1, B)

    bingo_array(A2, B)

    bingo_array(A3, B)

    if sum(A1) == 0 or sum(A2) == 0 or sum(A3) == 0:

        print("Yes")

    elif A1[0] + A2[0] + A3[0] == 0 or A1[1] + A2[1] + A3[1] == 0 or A1[2] + A2[2] + A3[2] == 0:

        print("Yes")

    elif A1[0] + A2[1] + A3[2] == 0 or A1[2] + A2[1] + A3[0] == 0:

        print("Yes")

    else:

        print("No")


problem_p02760()
