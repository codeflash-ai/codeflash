def problem_p00449():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0526

    TLE

    """

    import sys

    from sys import stdin

    from heapq import heappush, heappop

    from collections import namedtuple

    input = stdin.readline

    def warshall_floyd(d, V):

        for k in range(V):

            for i in range(V):

                for j in range(V):

                    new_cost = d[i][k] + d[k][j]

                    if new_cost < d[i][j]:

                        d[i][j] = new_cost

    def main1(args):

        while True:

            n, k = list(map(int, input().split()))

            if n == 0 and k == 0:

                break

            d = [[float("inf")] * (n + 1) for _ in range(n + 1)]

            # for i in range(n+1):

            #    d[i][i] = 0

            updated = False

            for _ in range(k):

                data = [int(x) for x in input().split()]

                if data[0] == 0:

                    if updated:

                        warshall_floyd(d, n + 1)

                        updated = False

                    f, t = data[1], data[2]

                    if d[f][t] == float("inf"):

                        print((-1))

                    else:

                        print((d[f][t]))

                else:

                    f, t, c = data[1], data[2], data[3]

                    if c < d[f][t]:

                        d[f][t] = c

                        updated = True

                    if c < d[t][f]:

                        d[t][f] = c

                        updated = True

    edge = namedtuple("edge", ["t", "c"])

    def dijkstra(s, G):

        pq = []

        d = [float("inf")] * len(G)

        d[s] = 0

        heappush(pq, (0, s))

        while pq:

            c, v = heappop(pq)

            if d[v] < c:

                continue

            for e in G[
                v
            ]:  #  ?????????v?????£??\??????????????±????????¨?????????????????°?????§?????????????????´??°

                if (
                    d[e.t] > d[v] + e.c
                ):  #  ???????????§??\??£?????????????????????????????????v????????±????????????????????????????????????????????°???????????????????????´??°

                    d[e.t] = d[v] + e.c

                    heappush(
                        pq, (d[e.t], e.t)
                    )  #  python???heapq??????????°???????????????????pop?????????????????????????????§???-1??????????????????

        return d

    def main(args):

        while True:

            n, k = list(map(int, input().split()))

            if n == 0 and k == 0:

                break

            d = [[] for _ in range(n + 1)]

            costs = [[float("inf")] * (n + 1) for _ in range(n + 1)]

            updated = False

            for _ in range(k):

                data = [int(x) for x in input().split()]

                if data[0] == 0:

                    f, t = data[1], data[2]

                    if updated:

                        costs[f] = dijkstra(f, d)

                    if costs[f][t] == float("inf"):

                        print((-1))

                    else:

                        print((costs[f][t]))

                else:

                    f, t, c = data[1], data[2], data[3]

                    d[f].append(edge(t, c))

                    d[t].append(edge(f, c))

                    updated = True

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00449()
