def problem_p02902():
    INF = float("inf")

    from collections import defaultdict

    from heapq import heappop, heappush

    class Graph(object):

        def __init__(self):

            self.graph = defaultdict(list)

        def __len__(self):

            return len(self.graph)

        def add_edge(self, From, To, cost=1):

            self.graph[From].append((To, cost))

        def get_nodes(self):

            return list(self.graph.keys())

    class Dijkstra(object):

        def __init__(self, graph, start):

            self.g = graph.graph

            self.dist = defaultdict(lambda: INF)

            self.dist[start] = 0

            self.prev = defaultdict(lambda: None)

            self.Q = []

            heappush(self.Q, (self.dist[start], start))

            while self.Q:

                dist_u, u = heappop(self.Q)

                if self.dist[u] < dist_u:

                    continue

                for v, cost in self.g[u]:

                    alt = dist_u + cost

                    if self.dist[v] > alt:

                        self.dist[v] = alt

                        self.prev[v] = u

                        heappush(self.Q, (alt, v))

        def shortest_distance(self, goal):

            return self.dist[goal]

        def shortest_path(self, goal):

            path = []

            node = goal

            while node is not None:

                path.append(node)

                node = self.prev[node]

            return path[::-1]

    N, M = list(map(int, input().split()))

    g = Graph()

    for i in range(M):

        a, b = list(map(int, input().split()))

        a -= 1

        b -= 1

        g.add_edge(a, b)

        g.add_edge(a, b + N)

        g.add_edge(a + N, b)

    ans = INF

    j = -1

    for i in range(N):

        d = Dijkstra(g, i).dist

        if d[i + N] < ans:

            ans = int(d[i + N])

            j = i

    if ans != INF:

        path = Dijkstra(g, j).shortest_path(j + N)[:-1]

        print(ans)

        for node in path:

            print((node + 1))

    else:

        print((-1))


problem_p02902()
