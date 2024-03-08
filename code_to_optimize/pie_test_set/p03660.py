def problem_p03660():
    from collections import defaultdict

    from heapq import heappop, heappush

    N = int(eval(input()))

    class Graph(object):

        def __init__(self):

            self.graph = defaultdict(list)

        def __len__(self):

            return len(self.graph)

        def add_edge(self, src, dst, weight=1):

            self.graph[src].append((dst, weight))

        def get_nodes(self):

            return list(self.graph.keys())

    class Dijkstra(object):

        def __init__(self, graph, start):

            g = graph.graph

            self.dist = defaultdict(lambda: float("inf"))

            self.dist[start] = 0

            self.prev = defaultdict(lambda: None)

            Q = []

            heappush(Q, (self.dist[start], start))

            while Q:

                dist_u, u = heappop(Q)

                if self.dist[u] < dist_u:

                    continue

                for v, weight in g[u]:

                    alt = dist_u + weight

                    if self.dist[v] > alt:

                        self.dist[v] = alt

                        self.prev[v] = u

                        heappush(Q, (alt, v))

        def shortest_distance(self, goal):

            return self.dist[goal]

        def shortest_path(self, goal):

            path = []

            node = goal

            while node is not None:

                path.append(node)

                node = self.prev[node]

            return path[::-1]

    g = Graph()

    for _ in range(N - 1):

        a, b = list(map(int, input().split()))

        g.add_edge(a - 1, b - 1)

        g.add_edge(b - 1, a - 1)

    d1 = Dijkstra(g, 0)

    d2 = Dijkstra(g, N - 1)

    c1, c2 = 0, 0

    for i in range(N):

        if d1.shortest_distance(i) <= d2.shortest_distance(i):

            c1 += 1

        else:

            c2 += 1

    if c1 > c2:

        print("Fennec")

    else:

        print("Snuke")


problem_p03660()
