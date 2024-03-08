def problem_p03647():
    from collections import defaultdict, deque

    INF = 10**12

    N, M = list(map(int, input().split()))

    E = defaultdict(list)

    for _ in range(M):

        a, b = list(map(int, input().split()))

        E[a - 1].append(b - 1)

        E[b - 1].append(a - 1)

    dist = [INF] * N

    q = deque([(0, 0)])

    while q:

        v, d = q.popleft()

        dist[v] = d

        for u in E[v]:

            if dist[u] == INF:

                q.append((u, d + 1))

    print(("POSSIBLE" if dist[N - 1] == 2 else "IMPOSSIBLE"))


problem_p03647()
