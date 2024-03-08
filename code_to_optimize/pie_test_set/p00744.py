def problem_p00744():
    import collections

    import math

    class Dinic:
        """Dinic Algorithm: find max-flow

        complexity: O(EV^2)

        used in GRL6A(AOJ)

        """

        class edge:

            def __init__(self, to, cap, rev):

                self.to, self.cap, self.rev = to, cap, rev

        def __init__(self, V, E, source, sink):
            """V: the number of vertexes

            E: adjacency list

            source: start point

            sink: goal point

            """

            self.V = V

            self.E = [[] for _ in range(V)]

            for fr in range(V):

                for to, cap in E[fr]:

                    self.E[fr].append(self.edge(to, cap, len(self.E[to])))

                    self.E[to].append(self.edge(fr, 0, len(self.E[fr]) - 1))

            self.maxflow = self.dinic(source, sink)

        def dinic(self, source, sink):
            """find max-flow"""

            INF = float("inf")

            maxflow = 0

            while True:

                self.bfs(source)

                if self.level[sink] < 0:

                    return maxflow

                self.itr = [0] * self.V

                while True:

                    flow = self.dfs(source, sink, INF)

                    if flow > 0:

                        maxflow += flow

                    else:

                        break

        def dfs(self, vertex, sink, flow):
            """find augmenting path"""

            if vertex == sink:

                return flow

            for i in range(self.itr[vertex], len(self.E[vertex])):

                self.itr[vertex] = i

                e = self.E[vertex][i]

                if e.cap > 0 and self.level[vertex] < self.level[e.to]:

                    d = self.dfs(e.to, sink, min(flow, e.cap))

                    if d > 0:

                        e.cap -= d

                        self.E[e.to][e.rev].cap += d

                        return d

            return 0

        def bfs(self, start):
            """find shortest path from start"""

            que = collections.deque()

            self.level = [-1] * self.V

            que.append(start)

            self.level[start] = 0

            while que:

                fr = que.popleft()

                for e in self.E[fr]:

                    if e.cap > 0 and self.level[e.to] < 0:

                        self.level[e.to] = self.level[fr] + 1

                        que.append(e.to)

    while True:

        M, N = list(map(int, input().split()))

        if M == 0 and N == 0:

            break

        blue, red = [], []

        while True:

            for x in input().split():

                blue.append(int(x))

            if len(blue) == M:

                break

        while True:

            for x in input().split():

                red.append(int(x))

            if len(red) == N:

                break

        V = M + N + 2

        edge = [set() for _ in range(V)]

        for i, b in enumerate(blue):

            if b != 1:

                for j, r in enumerate(red):

                    if r % b == 0:

                        edge[i].add((M + j, 1))

            for j in range(2, int(math.sqrt(b)) + 1):

                if b % j == 0:

                    for k, r in enumerate(red):

                        if r % j == 0 or r % (b // j) == 0:

                            edge[i].add((M + k, 1))

        for i in range(M):

            edge[M + N].add((i, 1))

        for j in range(N):

            edge[M + j].add((M + N + 1, 1))

        d = Dinic(V, edge, M + N, M + N + 1)

        print((d.maxflow))


problem_p00744()
