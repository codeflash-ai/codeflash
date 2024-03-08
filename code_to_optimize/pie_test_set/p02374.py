def problem_p02374():
    #!/usr/bin/env python3

    # GRL_5_D: Tree - Range Query on Tree

    class Edge:

        __slots__ = ("v", "w")

        def __init__(self, v, w):

            self.v = v

            self.w = w

        def either(self):

            return self.v

        def other(self, v):

            if v == self.v:

                return self.w

            else:

                return self.v

    class Graph:

        def __init__(self, v):

            self.v = v

            self._edges = [[] for _ in range(v)]

        def add(self, e):

            self._edges[e.v].append(e)

            self._edges[e.w].append(e)

        def adj(self, v):

            return self._edges[v]

        def edges(self):

            for es in self._edges:

                for e in es:

                    yield e

    class RAQ:
        """Segment Tree"""

        def __init__(self, n, initial=0):

            size = 1

            while size < n:

                size *= 2

            self.size = 2 * size - 1

            self.data = [initial] * self.size

        def add(self, i, j, v):

            def _add(r, lo, hi):

                if hi < i or lo > j:

                    return

                elif i <= lo and hi <= j:

                    self.data[r] += v

                else:

                    mid = lo + (hi - lo) // 2

                    _add(r * 2 + 1, lo, mid)

                    _add(r * 2 + 2, mid + 1, hi)

            return _add(0, 0, self.size // 2)

        def get(self, i):

            def _get(r, lo, hi, v):

                v += self.data[r]

                if lo == hi:

                    return v

                mid = lo + (hi - lo) // 2

                if mid >= i:

                    return _get(r * 2 + 1, lo, mid, v)

                else:

                    return _get(r * 2 + 2, mid + 1, hi, v)

            return _get(0, 0, self.size // 2, 0)

    class PathSum:

        def __init__(self, graph, root):

            self.seg = RAQ(graph.v * 2)

            self._in = [0] * graph.v

            self._out = [0] * graph.v

            self.dfs(graph, root)

        def dfs(self, graph, root):

            visited = [False] * graph.v

            stack = [root]

            i = 0

            while stack:

                v = stack.pop()

                if not visited[v]:

                    visited[v] = True

                    self._in[v] = i

                    i += 1

                    stack.append(v)

                    for e in graph.adj(v):

                        w = e.other(v)

                        if not visited[w]:

                            stack.append(w)

                else:

                    self._out[v] = i

                    i += 1

        def add(self, v, val):

            i = self._in[v]

            j = self._out[v]

            self.seg.add(i, j, val)

        def get(self, v):

            return self.seg.get(self._in[v])

    def run():

        n = int(eval(input()))

        g = Graph(n)

        for i in range(n):

            k, *cs = [int(i) for i in input().split()]

            if k > 0:

                for j in cs:

                    g.add(Edge(i, j))

        raq = PathSum(g, 0)

        q = int(eval(input()))

        for _ in range(q):

            com, *args = [int(i) for i in input().split()]

            if com == 0:

                u, val = args

                raq.add(u, val)

            elif com == 1:

                (u,) = args

                print((raq.get(u)))

            else:

                raise ValueError("invalid command")

    if __name__ == "__main__":

        run()


problem_p02374()
