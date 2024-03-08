def problem_p00485():
    from heapq import heappop as pop

    from heapq import heappush as push

    INF = 10**18

    class edge:

        def __init__(self, to, cost):

            self.to = to

            self.cost = cost

    # V, E, r = map(int,input().split())

    N, M, K = list(map(int, input().split()))

    G = [[] for i in range(N)]

    # G[i]...頂点iからの辺list、(行き先、コスト)

    d = [INF for i in range(N)]

    # d[i]...スタートからみた頂点iまでの最短距離

    def dijkstra(lst):

        que = []

        for s in lst:

            d[s] = 0

            push(que, (0, s))

        while len(que):

            p = pop(que)

            v = p[1]

            if d[v] < p[0]:

                continue

            for i in range(len(G[v])):

                e = G[v][i]

                if d[e.to] > d[v] + e.cost:

                    d[e.to] = d[v] + e.cost

                    push(que, (d[e.to], e.to))

    for i in range(M):

        s, t, c = list(map(int, input().split()))

        s -= 1

        t -= 1

        G[s].append(edge(t, c))

        G[t].append(edge(s, c))

    lst = [int(eval(input())) - 1 for i in range(K)]

    dijkstra(lst)

    anss = []

    append = anss.append

    for i in range(N):

        for e in G[i]:

            x = d[i] + d[e.to] + e.cost

            if x % 2:

                append(x // 2 + 1)

            else:

                append(x // 2)

    print((max(anss)))


problem_p00485()
