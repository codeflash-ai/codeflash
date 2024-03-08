def problem_p03142():
    from collections import deque

    import numpy as np

    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    sys.setrecursionlimit(10**7)

    N, M = map(int, input().split())

    AB = [tuple(map(int, input().split())) for _ in range(N + M - 1)]

    e_from = [[] for _ in range(N + 1)]

    e_to = [[] for _ in range(N + 1)]

    for a, b in AB:

        e_from[a].append(b)

        e_to[b].append(a)

    add = np.zeros(N + 1, dtype=int)

    rank = [-1] * (N + 1)

    # まず根を探す

    for i in range(1, N + 1):

        add[i] = len(e_to[i])

        if len(e_to[i]) == 0:

            root = i

    # BFS

    rank[root] = 0

    node = deque([root])

    while node:

        s = node.popleft()

        add[e_from[s]] -= 1

        for t in e_from[s]:

            if add[t] == 0:

                rank[t] = s

                node.append(t)

    print(*rank[1:], sep="\n")


problem_p03142()
