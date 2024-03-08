def problem_p03044():
    N = int(eval(input()))

    from collections import deque

    tree = [{} for _ in range(N)]

    color = [-1] * N

    for i in range(N - 1):

        u, v, w = list(map(int, input().split()))

        tree[u - 1][v - 1] = w

        tree[v - 1][u - 1] = w

    queue = deque()

    queue.append((0, 0))

    while queue:

        v, w = queue.pop()

        color[v] = w % 2

        for i in list(tree[v].keys()):

            if color[i] == -1:

                queue.append((i, w + tree[v][i]))

    for c in color:

        print(c)


problem_p03044()
