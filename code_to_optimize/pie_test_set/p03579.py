def problem_p03579():
    import sys

    sys.setrecursionlimit(100000)

    N, M = list(map(int, input().split()))

    edges = [list(map(int, input().split())) for _ in range(M)]

    graph = [[] for _ in range(N)]

    for x, y in edges:

        graph[x - 1].append(y - 1)

        graph[y - 1].append(x - 1)

    def dfs(v, c):

        # c: color = 1 or -1

        node[v] = c

        for i in graph[v]:

            if node[i] == c:

                return False

            if node[i] == 0 and not dfs(i, -c):

                return False

        return True

    node = [0] * N

    if dfs(0, 1):

        x = sum(v + 1 for v in node) // 2

        print((x * (N - x) - M))

    else:

        print((N * (N - 1) // 2 - M))


problem_p03579()
