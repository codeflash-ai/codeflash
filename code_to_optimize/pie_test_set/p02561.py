def problem_p02561():
    # Dinic's algorithm

    # https://tjkendev.github.io/procon-library/python/max_flow/dinic.htmlより

    from collections import deque

    class Dinic:

        def __init__(self, N):

            self.N = N

            self.G = [[] for i in range(N)]

        def add_edge(self, fr, to, cap):

            forward = [to, cap, None]

            forward[2] = backward = [fr, 0, forward]

            self.G[fr].append(forward)

            self.G[to].append(backward)

        def add_multi_edge(self, v1, v2, cap1, cap2):

            edge1 = [v2, cap1, None]

            edge1[2] = edge2 = [v1, cap2, edge1]

            self.G[v1].append(edge1)

            self.G[v2].append(edge2)

        def bfs(self, s, t):

            self.level = level = [None] * self.N

            deq = deque([s])

            level[s] = 0

            G = self.G

            while deq:

                v = deq.popleft()

                lv = level[v] + 1

                for w, cap, _ in G[v]:

                    if cap and level[w] is None:

                        level[w] = lv

                        deq.append(w)

            return level[t] is not None

        def dfs(self, v, t, f):

            if v == t:

                return f

            level = self.level

            for e in self.it[v]:

                w, cap, rev = e

                if cap and level[v] < level[w]:

                    d = self.dfs(w, t, min(f, cap))

                    if d:

                        e[1] -= d

                        rev[1] += d

                        return d

            return 0

        def flow(self, s, t):

            flow = 0

            INF = 10**9 + 7

            G = self.G

            while self.bfs(s, t):

                (*self.it,) = list(map(iter, self.G))

                f = INF

                while f:

                    f = self.dfs(s, t, INF)

                    flow += f

            return flow

    N, M = list(map(int, input().split()))

    S = [list(eval(input())) for i in range(N)]

    dc = Dinic(N * M + 2)

    s = N * M

    t = N * M + 1

    for i in range(N):

        for j in range(M):

            if (i + j) % 2 == 0:

                dc.add_edge(s, M * i + j, 1)

            else:

                dc.add_edge(M * i + j, t, 1)

    for i in range(N):

        for j in range(M):

            if j + 1 < M and S[i][j] == "." and S[i][j + 1] == ".":

                u, v = M * i + j, M * i + j + 1

                if (i + j) % 2 == 1:

                    u, v = v, u

                dc.add_edge(u, v, 1)

            if i + 1 < N and S[i][j] == "." and S[i + 1][j] == ".":

                u, v = M * i + j, M * (i + 1) + j

                if (i + j) % 2 == 1:

                    u, v = v, u

                dc.add_edge(u, v, 1)

    print((dc.flow(s, t)))

    for u in range(N * M + 2):

        for v, cap, _ in dc.G[u]:

            ui, uj = divmod(u, M)

            vi, vj = divmod(v, M)

            if (ui + uj) % 2 == 0 and cap == 0 and u != s and u != t and v != s and v != t:

                if ui + 1 == vi:

                    S[ui][uj] = "v"

                    S[vi][vj] = "^"

                elif ui == vi + 1:

                    S[ui][uj] = "^"

                    S[vi][vj] = "v"

                elif uj + 1 == vj:

                    S[ui][uj] = ">"

                    S[vi][vj] = "<"

                elif uj == vj + 1:

                    S[ui][uj] = "<"

                    S[vi][vj] = ">"

    for res in S:

        print(("".join(res)))


problem_p02561()
