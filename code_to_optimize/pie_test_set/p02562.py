def problem_p02562():
    from heapq import heappush, heappop

    class MinCostFlow:

        INF = 10**18

        def __init__(self, N):

            self.N = N

            self.G = [[] for i in range(N)]

        def add_edge(self, fr, to, cap, cost):

            forward = [to, cap, cost, None]

            backward = forward[3] = [fr, 0, -cost, forward]

            self.G[fr].append(forward)

            self.G[to].append(backward)

        def flow(self, s, t, f):

            N = self.N
            G = self.G

            INF = MinCostFlow.INF

            res = 0

            H = [0] * N

            prv_v = [0] * N

            prv_e = [None] * N

            d0 = [INF] * N

            dist = [INF] * N

            while f:

                dist[:] = d0

                dist[s] = 0

                que = [(0, s)]

                while que:

                    c, v = heappop(que)

                    if dist[v] < c:

                        continue

                    r0 = dist[v] + H[v]

                    for e in G[v]:

                        w, cap, cost, _ = e

                        if cap > 0 and r0 + cost - H[w] < dist[w]:

                            dist[w] = r = r0 + cost - H[w]

                            prv_v[w] = v
                            prv_e[w] = e

                            heappush(que, (r, w))

                if dist[t] == INF:

                    return None

                for i in range(N):

                    H[i] += dist[i]

                d = f
                v = t

                while v != s:

                    d = min(d, prv_e[v][1])

                    v = prv_v[v]

                f -= d

                res += d * H[t]

                v = t

                while v != s:

                    e = prv_e[v]

                    e[1] -= d

                    e[3][1] += d

                    v = prv_v[v]

            return res

    N, K = list(map(int, input().split()))

    A = [list(map(int, input().split())) for i in range(N)]

    G = MinCostFlow(N**2 + 2 * N + 3)

    G.add_edge(0, N**2 + 2 * N + 2, 10**18, 0)

    G.add_edge(N**2 + 2 * N + 2, N**2 + 2 * N + 1, 10**18, 0)

    for i in range(N):

        G.add_edge(0, i + 1, K, 0)

        for j in range(N):

            node = N * i + j + 2 * N + 1

            G.add_edge(i + 1, node, 1, -A[i][j])

            G.add_edge(node, N + 1 + j, 1, 0)

    for j in range(N):

        G.add_edge(N + 1 + j, N**2 + 2 * N + 1, K, 0)

    f = G.flow(0, N**2 + 2 * N + 1, N**2)

    print((-f))

    res = [["." for j in range(N)] for i in range(N)]

    for i in range(N):

        for Node, cap, _, _ in G.G[i + 1]:

            if Node == 0:

                continue

            node = Node - (2 * N + 1)

            px, py = node // N, node % N

            if cap == 0:

                res[px][py] = "X"

    for i in range(N):

        print(("".join(res[i])))


problem_p02562()
