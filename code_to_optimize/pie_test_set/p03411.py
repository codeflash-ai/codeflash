def problem_p03411():
    from collections import deque

    N = int(eval(input()))

    R = [list(map(int, input().split())) for _ in range(N)]

    B = [list(map(int, input().split())) for _ in range(N)]

    graph = [[] for _ in range(2 * N + 2)]  # 残余グラフを構成　0を始点，1を終点とする

    for i in range(N):

        forward = [2 + i, 1, None]  # [行き先，流量，逆辺の情報]

        forward[2] = backward = [0, 0, forward]

        graph[0].append(forward)

        graph[2 + i].append(backward)

    for i in range(N):

        forward = [1, 1, None]

        forward[2] = backward = [2 + N + i, 0, forward]

        graph[2 + N + i].append(forward)

        graph[1].append(backward)

    for i in range(N):

        for j in range(N):

            if R[i][0] < B[j][0] and R[i][1] < B[j][1]:

                forward = [2 + N + j, 1, None]

                forward[2] = backward = [2 + i, 0, forward]

                graph[2 + i].append(forward)

                graph[2 + N + j].append(backward)

    flow = 0

    while True:

        queue = deque([0])

        level = [None for _ in range(2 * N + 2)]

        level[0] = 0

        while queue:

            node = queue.popleft()

            for adj, adjcap, _ in graph[node]:

                if adjcap > 0 and level[adj] is None:

                    level[adj] = level[node] + 1

                    queue.append(adj)

        if level[1] is None:

            break

        stack = [0]

        visited = [0 for _ in range(2 * N + 2)]

        visited[0] = 1

        parent = [None for _ in range(2 * N + 2)]

        parent[0] = 0

        # *it, = map(iter, graph)

        while stack:

            node = stack.pop()

            for adj, adjcap, adjrev in graph[node]:

                if not visited[adj] and adjcap > 0 and level[node] < level[adj]:

                    visited[adj] = 1

                    parent[adj] = node

                    stack.append(adj)

        if visited[1]:

            flow += 1

            node = 1

            prev = parent[node]

            while node != 0:

                for g in graph[prev]:

                    adj, adjcap, adjrev = g

                    if adj == node:

                        g[1] = 0

                        adjrev[1] = 1

                node = prev

                prev = parent[node]

    print(flow)


problem_p03411()
