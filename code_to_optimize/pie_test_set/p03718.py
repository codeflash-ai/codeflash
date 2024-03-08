def problem_p03718():
    import sys

    input = sys.stdin.readline

    from scipy.sparse.csgraph import dijkstra

    H, W = list(map(int, input().split()))

    # start = 0

    # rows = 1,2,...,H

    # cols = H+1,...,H+W

    # goal = H+W+1

    INF = 10**9

    start = 0

    goal = H + W + 1

    V = H + W + 2

    graph = [[0] * V for _ in range(V)]

    edges = []  # 隣接リスト

    for i in range(H):

        row = eval(input())

        for j, cell in enumerate(row):

            if cell == "o":

                graph[1 + i][1 + H + j] = 1

                graph[1 + H + j][1 + i] = 1

        s = row.find("S")

        t = row.find("T")

        if s != -1:

            graph[start][1 + i] = INF

            graph[start][1 + H + s] = INF

        if t != -1:

            graph[1 + i][goal] = INF

            graph[1 + H + t][goal] = INF

    # あとは max flow を求めればよい

    def max_flow(graph):

        f = 0

        while True:

            if f > 200:

                return -1

            dist, pred = dijkstra(graph, indices=start, unweighted=True, return_predecessors=True)

            if dist[goal] > INF:

                return f

            f += 1

            after = goal

            while after != start:

                before = pred[after]

                graph[before][after] -= 1

                graph[after][before] += 1

                after = before

    answer = max_flow(graph)

    print(answer)


problem_p03718()
