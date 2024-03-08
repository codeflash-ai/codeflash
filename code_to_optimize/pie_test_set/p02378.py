def problem_p02378():
    # Acceptance of input

    import sys

    file_input = sys.stdin

    X, Y, E = list(map(int, file_input.readline().split()))

    v_num = X + Y + 2

    adj_mat = [[0] * (v_num) for i in range(v_num)]

    for line in file_input:

        x, y = list(map(int, line.split()))

        adj_mat[x][X + y] = 1

    s = X + Y

    t = s + 1

    for i in range(X):

        adj_mat[s][i] = 1

    for i in range(X, s):

        adj_mat[i][t] = 1

    # Dinic's algorithm

    import collections

    # BFS for residual capacity network to construct level graph

    def bfs(start, goal, parent):

        unvisited = [True] * v_num

        queue = collections.deque()

        queue.append(start)

        unvisited[start] = False

        while queue:

            u = queue.popleft()

            for v, r_capacity in enumerate(adj_mat[u]):

                if unvisited[v] and (r_capacity > 0):

                    parent[v].append(u)

                    queue.append(v)

                    unvisited[v] = False

        return unvisited[goal]

    # DFS for level graph and construct blocking flow

    def dfs(goal, path, parent, blocking_flow):

        v = path[-1]

        if v == goal:

            aug_path_flow = 10000

            for x, y in zip(path[1:], path[:-1]):

                aug_path_flow = min(aug_path_flow, adj_mat[x][y])

            for x, y in zip(path[1:], path[:-1]):

                adj_mat[x][y] -= aug_path_flow

                adj_mat[y][x] += aug_path_flow

            blocking_flow[0] += aug_path_flow

        else:

            for u in parent[v]:

                path.append(u)

                dfs(goal, path, parent, blocking_flow)

                path.pop()

    def dinic(source, sink):

        max_flow = 0

        parent = [[] for i in range(v_num)]

        while not bfs(source, sink, parent):

            blocking_flow = [0]

            path = [sink]

            # search in reverse direction from sink

            dfs(source, path, parent, blocking_flow)

            max_flow += blocking_flow[0]

            parent = [[] for i in range(v_num)]

        return max_flow

    # output

    print((dinic(s, t)))


problem_p02378()
