def problem_p02373():
    class Edge:

        def __init__(self, dst, weight):

            self.dst, self.weight = dst, weight

        def __lt__(self, e):

            return self.weight > e.weight

    class Graph:

        def __init__(self, V):

            self.V = V

            self.E = [[] for _ in range(V)]

        def add_edge(self, src, dst, weight):

            self.E[src].append(Edge(dst, weight))

    class HeavyLightDecomposition:

        def __init__(self, g, root=0):

            self.g = g

            self.vid, self.head, self.heavy, self.parent = (
                [0] * g.V,
                [-1] * g.V,
                [-1] * g.V,
                [-1] * g.V,
            )

            self.dfs(root)

            self.bfs(root)

        def dfs(self, root):

            stack = [(root, -1)]

            sub, max_sub = [1] * self.g.V, [(0, -1)] * self.g.V

            used = [False] * self.g.V

            while stack:

                v, par = stack.pop()

                if not used[v]:

                    used[v] = True

                    self.parent[v] = par

                    stack.append((v, par))

                    stack.extend((e.dst, v) for e in self.g.E[v] if e.dst != par)

                else:

                    if par != -1:

                        sub[par] += sub[v]

                        max_sub[par] = max(max_sub[par], (sub[v], v))

                    self.heavy[v] = max_sub[v][1]

        def bfs(self, root=0):

            from collections import deque

            k, que = 0, deque([root])

            while que:

                r = v = que.popleft()

                while v != -1:

                    self.vid[v], self.head[v] = k, r

                    for e in self.g.E[v]:

                        if e.dst != self.parent[v] and e.dst != self.heavy[v]:

                            que.append(e.dst)

                    k += 1

                    v = self.heavy[v]

        def lca(self, u, v):

            while self.head[u] != self.head[v]:

                if self.vid[u] > self.vid[v]:

                    u, v = v, u

                v = self.parent[self.head[v]]

            else:

                if self.vid[u] > self.vid[v]:

                    u, v = v, u

            return u

    N = int(eval(input()))

    g = Graph(N)

    for i in range(N):

        for c in map(int, input().split()[1:]):

            g.add_edge(i, c, 1)

            g.add_edge(c, i, 1)

    hld = HeavyLightDecomposition(g)

    Q = int(eval(input()))

    for _ in range(Q):

        u, v = list(map(int, input().split()))

        print((hld.lca(u, v)))


problem_p02373()
