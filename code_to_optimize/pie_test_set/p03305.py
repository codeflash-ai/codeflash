def problem_p03305():
    import heapq

    import sys

    input = sys.stdin.readline

    INF = 10**18

    n, m, s, t = list(map(int, input().split()))

    edge = [list(map(int, input().split())) for _ in range(m)]

    graph_s = [[] for _ in range(n)]

    graph_t = [[] for _ in range(n)]

    for e in edge:

        graph_s[e[0] - 1].append([e[1] - 1, e[2]])

        graph_s[e[1] - 1].append([e[0] - 1, e[2]])

        graph_t[e[0] - 1].append([e[1] - 1, e[3]])

        graph_t[e[1] - 1].append([e[0] - 1, e[3]])

    dist_s = [INF] * n

    dist_t = [INF] * n

    dist_s[s - 1] = 0

    dist_t[t - 1] = 0

    heap_s = [[0, s - 1]]

    heapq.heapify(heap_s)

    heap_t = [[0, t - 1]]

    heapq.heapify(heap_t)

    while heap_s:

        cost, node = heapq.heappop(heap_s)

        if dist_s[node] < cost:

            continue

        for adj, adjcost in graph_s[node]:

            if dist_s[node] + adjcost < dist_s[adj]:

                dist_s[adj] = dist_s[node] + adjcost

                heapq.heappush(heap_s, [dist_s[adj], adj])

    while heap_t:

        cost, node = heapq.heappop(heap_t)

        if dist_t[node] < cost:

            continue

        for adj, adjcost in graph_t[node]:

            if dist_t[node] + adjcost < dist_t[adj]:

                dist_t[adj] = dist_t[node] + adjcost

                heapq.heappush(heap_t, [dist_t[adj], adj])

    dist = INF

    ans = []

    for i in range(n):

        dist = min(dist, dist_t[-1 - i] + dist_s[-1 - i])

        ans.append(10**15 - dist)

    for i in range(n):

        print((ans[-1 - i]))


problem_p03305()
