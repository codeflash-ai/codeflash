def problem_p03600():
    def Warshall_Floyd(edges, N):

        for k in range(N):

            for i in range(N):

                for j in range(N):

                    edges[i][j] = min(edges[i][j], edges[i][k] + edges[k][j])

        return edges

    n = int(eval(input()))

    arr = [list(map(int, input().split())) for _ in range(n)]

    edge = []

    for i in range(n):

        for j in range(n):

            if i >= j:

                continue

            edge.append((arr[i][j], (i, j)))

    edge = sorted(edge, key=lambda x: x[0])

    g = [[float("inf")] * n for _ in range(n)]

    for i in range(n):

        for j in range(n):

            g[i][j] = arr[i][j]

    d = Warshall_Floyd(g, n)

    ans = 0

    for w, (a, b) in edge:

        if d[a][b] < w:

            print((-1))

            break

        else:

            tmp = 10**10

            for k in range(n):

                if k == a or k == b:

                    continue

                tmp = min(tmp, d[a][k] + d[k][b])

            if w == tmp:

                continue

            else:

                ans += w

    else:

        print(ans)


problem_p03600()
