def problem_p03136():
    import sys, re, os

    from collections import deque, defaultdict, Counter

    from math import ceil, sqrt, hypot, factorial, pi, sin, cos, radians

    from itertools import permutations, combinations, product, accumulate

    from operator import itemgetter, mul

    from copy import deepcopy

    from string import ascii_lowercase, ascii_uppercase, digits

    from fractions import gcd

    def input():
        return sys.stdin.readline().strip()

    def INT():
        return int(eval(input()))

    def MAP():
        return list(map(int, input().split()))

    def S_MAP():
        return list(map(str, input().split()))

    def LIST():
        return list(map(int, input().split()))

    def S_LIST():
        return list(map(str, input().split()))

    sys.setrecursionlimit(10**9)

    INF = float("inf")

    mod = 10**9 + 7

    N = INT()

    L = LIST()

    m = max(L)

    for i in range(N):

        a = L[i]

        # b = L[:i-1] + L[i:]

        # print(L[:i-1] + L[i:])

        if a == m:

            if sum(L) - a > a:

                print("Yes")

                break

            else:

                print("No")

                break


problem_p03136()
