def problem_p03726():
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

    n = I()

    G = [[] for _ in range(n)]

    for _ in range(n - 1):

        a, b = LI()

        G[a - 1] += [b - 1]

        G[b - 1] += [a - 1]

    def dfs(v, visited):
        """

        :param v: X側の未マッチングの頂点の1つ

        :param visited: 空のsetを渡す（外部からの呼び出し時）

        :return: 増大路が見つかればTrue

        """

        for u in G[v]:

            if u in visited:

                continue

            visited.add(u)

            if matched[u] == -1 or dfs(matched[u], visited):

                matched[u] = v

                return True

        return False

    cnt = 0

    matched = [-1] * n

    for i in range(n):

        cnt += int(dfs(i, set()))

    if cnt == n:

        print("Second")

    else:

        print("First")


problem_p03726()
