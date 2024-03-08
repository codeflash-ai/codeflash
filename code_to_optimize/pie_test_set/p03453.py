def problem_p03453():
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

    from functools import reduce

    sys.setrecursionlimit(2147483647)

    INF = 10**15

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

    n, m = LI()

    s, t = LI()

    G = [[] for _ in range(n)]

    for i in range(m):

        a, b, c = LI()

        G[a - 1] += [(b - 1, c)]

        G[b - 1] += [(a - 1, c)]

    def dijkstra(graph, start=0):

        route_cnt = [start] * n

        route_cnt[start] = 1

        que = [(0, start)]

        dist = [INF] * n

        dist[start] = 0

        while que:

            min_dist, u = heappop(que)

            if min_dist > dist[u]:

                continue

            for v, c in graph[u]:

                if dist[u] + c < dist[v]:

                    dist[v] = dist[u] + c

                    route_cnt[v] = route_cnt[u]

                    heappush(que, (dist[u] + c, v))

                elif dist[u] + c == dist[v]:

                    route_cnt[v] = (route_cnt[v] + route_cnt[u]) % mod

        return route_cnt, dist

    route_cnt, dist = dijkstra(G, s - 1)

    r_route_cnt, r_dist = dijkstra(G, t - 1)

    total_dist = dist[t - 1]

    ret = 0

    for i in range(n):

        if dist[i] == r_dist[i]:

            ret = ret + (route_cnt[i] * r_route_cnt[i] % mod) ** 2 % mod

    for u in range(n):

        for v, c in G[u]:

            if dist[u] < total_dist / 2 < dist[v] and dist[u] + c + r_dist[v] == total_dist:

                ret = (ret + route_cnt[u] ** 2 % mod * r_route_cnt[v] ** 2 % mod) % mod

    print(((route_cnt[t - 1] ** 2 - ret) % mod))


problem_p03453()
