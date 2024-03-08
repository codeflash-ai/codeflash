def problem_p03099():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(10**9)

    from bisect import bisect_left, bisect_right

    class MinCostFlow:

        def __init__(self, n):

            self.n = n

            self.edges = [[] for i in range(n)]

        def add_edge(self, fr, to, cap, cost):

            self.edges[fr].append([to, cap, cost, len(self.edges[to])])

            self.edges[to].append([fr, 0, -cost, len(self.edges[fr]) - 1])

        def MinCost(self, source, sink, flow):

            inf = 10**15 + 1

            n, E = self.n, self.edges

            prev_v, prev_e = [0] * n, [0] * n

            mincost = 0

            while flow:

                dist = [inf] * n

                dist[source] = 0

                flag = True

                while flag:

                    flag = False

                    for v in range(n):

                        if dist[v] == inf:

                            continue

                        Ev = E[v]

                        for i in range(len(Ev)):

                            to, cap, cost, rev = Ev[i]

                            if cap > 0 and dist[v] + cost < dist[to]:

                                dist[to] = dist[v] + cost

                                prev_v[to], prev_e[to] = v, i

                                flag = True

                if dist[sink] == inf:

                    return 1

                f = flow

                v = sink

                while v != source:

                    f = min(f, E[prev_v[v]][prev_e[v]][1])

                    v = prev_v[v]

                flow -= f

                mincost += f * dist[sink]

                v = sink

                while v != source:

                    E[prev_v[v]][prev_e[v]][1] -= f

                    rev = E[prev_v[v]][prev_e[v]][3]

                    E[v][rev][1] += f

                    v = prev_v[v]

            return mincost

    n = int(eval(input()))

    J = []

    L_org, D_org = [1] * n, [1] * n

    for _ in range(n):

        x, y, v = list(map(int, input().split()))

        J.append((x, y, v))

    m = int(eval(input()))

    T = []

    for _ in range(m):

        t, a, b = input().split()

        a, b = int(a), int(b)

        T.append((t, a, b))

        if t == "L":

            L_org[b] = a + 1

        elif t == "D":

            D_org[b] = a + 1

    for i in range(1, n):

        L_org[i] = max(L_org[i - 1], L_org[i])

        D_org[i] = max(D_org[i - 1], D_org[i])

    def solve(k):

        L, D = L_org[:k], D_org[:k]

        R, U = [100] * k, [100] * k

        for t, a, b in T:

            if k - b - 1 >= 0:

                if t == "R":

                    R[k - b - 1] = a - 1

                elif t == "U":

                    U[k - b - 1] = a - 1

        for i in range(k - 2, -1, -1):

            R[i] = min(R[i], R[i + 1])

            U[i] = min(U[i], U[i + 1])

        solver = MinCostFlow(2 * n + 2 * k + 2)

        for i in range(1, k + 1):

            solver.add_edge(0, i, 1, 0)

            solver.add_edge(2 * n + k + i, 2 * n + 2 * k + 1, 1, 0)

        for i in range(n):

            v = J[i][2]

            solver.add_edge(k + i + 1, n + k + i + 1, 1, -v)

        for i in range(n):

            x, y = J[i][0], J[i][1]

            l = bisect_right(L, x)

            r = bisect_left(R, x) + 1

            d = bisect_right(D, y)

            u = bisect_left(U, y) + 1

            for j in range(r, l + 1):

                solver.add_edge(j, k + i + 1, 1, 0)

            for j in range(u, d + 1):

                solver.add_edge(n + k + i + 1, 2 * n + k + j, 1, 0)

        return -solver.MinCost(0, 2 * n + 2 * k + 1, k)

    ans = 0

    k = 1

    while True:

        tmp = solve(k)

        ans = max(ans, tmp)

        if tmp == -1 or k == n:

            break

        k += 1

    print(ans)


problem_p03099()
