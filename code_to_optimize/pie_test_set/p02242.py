def problem_p02242():
    # coding:utf-8

    inf = 1000000

    n = int(eval(input()))

    M = [[inf for i in range(n)] for j in range(n)]

    vwMtrx = []

    for i in range(n):

        data = list(map(int, input().split()))

        u = data[0]

        k = data[1]

        col = []

        for j in range(0, 2 * k, 2):

            v, w = data[j + 2], data[j + 3]

            M[u][v] = w

            col.append([v, w])

        vwMtrx.append(col)

    def dijkstra(s):

        color = ["white" for i in range(n)]

        d = [inf for i in range(n)]

        p = [-1 for i in range(n)]

        d[s] = 0

        Q = [[s, 0]]

        while Q != []:

            u = min(Q, key=lambda x: x[1])

            Q = [List for List in Q if List != u]

            u = u[0]

            color[u] = "black"

            vwList = vwMtrx[u]

            for vw in vwList:

                v, w = vw[0], vw[1]

                if color[v] != "black" and M[u][v] != inf:

                    if d[u] + M[u][v] < d[v]:

                        d[v] = d[u] + M[u][v]

                        p[v] = u

                        color[v] = "gray"

                        Q.append([v, d[v]])

        return d

    dList = dijkstra(0)

    for i, d in enumerate(dList):

        print((i, d))


problem_p02242()
