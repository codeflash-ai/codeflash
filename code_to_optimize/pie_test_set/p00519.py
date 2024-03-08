def problem_p00519():
    from collections import deque

    import heapq

    def dijkstra(s, g, m):

        color = [0] * n

        dis = [float("inf")] * n

        dis[s] = 0

        heapq.heappush(pq, [0, s])

        while len(pq) != 0:

            t, u = heapq.heappop(pq)

            color[u] = 2

            if dis[u] < t:

                continue

            for v in g[u]:

                if color[v] != 2:

                    if dis[u] + m[u][v] < dis[v]:

                        dis[v] = dis[u] + m[u][v]

                        color[v] = 1

                        heapq.heappush(pq, [dis[v], v])

        return dis

    n, k = list(map(int, input().split()))

    cr = [list(map(int, input().split())) for _ in range(n)]

    g = [[] * n for _ in range(n)]

    for i in range(k):

        a, b = list(map(int, input().split()))

        g[a - 1].append(b - 1)

        g[b - 1].append(a - 1)

    d = [[0] * n for _ in range(n)]

    for i in range(n):

        q = deque([i])

        visited = [0] * n

        visited[i] = 1

        while len(q) != 0:

            p = q.popleft()

            for j in g[p]:

                if visited[j] == 0:

                    visited[j] = 1

                    q.append(j)

                    d[i][j] = d[i][p] + 1

    cost = [[float("inf")] * n for _ in range(n)]

    newg = [[] * n for _ in range(n)]

    for i in range(n):

        for j in range(n):

            if d[i][j] <= cr[i][1]:

                cost[i][j] = cr[i][0]

                newg[i].append(j)

    pq = []

    dis = dijkstra(0, newg, cost)

    print((dis[n - 1]))


problem_p00519()
