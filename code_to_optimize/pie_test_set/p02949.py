def problem_p02949():
    #!/usr/bin/env python3

    # ABC137 E

    import sys

    import math

    import bisect

    sys.setrecursionlimit(1000000000)

    from heapq import heappush, heappop

    from collections import defaultdict

    from itertools import accumulate

    from collections import Counter

    from collections import deque

    from operator import itemgetter

    from itertools import permutations

    mod = 10**9 + 7

    def I():
        return int(sys.stdin.readline())

    def LI():
        return list(map(int, sys.stdin.readline().split()))

    def Bellman_Ford(edges, num_v, start):

        # start → iの最短距離

        # start:始点, num_v:頂点数, edges:[辺の始点, 辺の終点, 辺のコスト]

        dist = [-float("inf") for _ in range(num_v)]

        # dist[i] := s→iの最短距離

        dist[start] = 0

        flg = 0

        for i in range(num_v):

            update = False

            for edge in edges:

                e_from, e_to, e_cost = edge

                if e_from != -float("inf") and visited[e_to] and dist[e_to] < dist[e_from] + e_cost:

                    dist[e_to] = dist[e_from] + e_cost

                    update = True

            if not update:

                flg = 1

                break

        return dist, flg

    n, m, p = LI()

    edge = []

    f = defaultdict(dict)

    for _ in range(m):

        a, b, c = LI()

        edge.append([a - 1, b - 1, c - p])

        f[b - 1][a - 1] = 1

    visited = [False] * n

    def dfs(x):

        visited[x] = True

        for t in list(f[x].keys()):

            if not visited[t]:

                dfs(t)

    dfs(n - 1)

    d, f = Bellman_Ford(edge, n, 0)

    if not f:

        print((-1))

    else:

        print((max(0, d[-1])))


problem_p02949()
