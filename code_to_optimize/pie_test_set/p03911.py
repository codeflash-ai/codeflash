def problem_p03911():
    N, M = list(map(int, input().split()))

    K = []

    L = []

    for _ in range(N):

        k, *l = list(map(int, input().split()))

        K.append(k)

        L.append(l)

    graph = [[] for _ in range(N + M)]

    for i in range(N):

        for l in L[i]:

            graph[i].append(N + l - 1)

            graph[N + l - 1].append(i)

    visited = [0 for _ in range(N + M)]

    visited[0] = 1

    stack = [0]

    while stack:

        node = stack.pop()

        for adj in graph[node]:

            if not visited[adj]:

                visited[adj] = 1

                stack.append(adj)

    print(("YES" if all(visited[:N]) else "NO"))


problem_p03911()
