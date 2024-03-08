def problem_p01812():
    import math, string, itertools, fractions, heapq, collections, re, array, bisect, sys, random, time, copy, functools

    sys.setrecursionlimit(10**7)

    inf = 10**20

    eps = 1.0 / 10**10

    mod = 10**9 + 7

    dd = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    ddn = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]

    def LI():
        return [int(x) for x in sys.stdin.readline().split()]

    def LI_():
        return [int(x) - 1 for x in sys.stdin.readline().split()]

    def LF():
        return [float(x) for x in sys.stdin.readline().split()]

    def LS():
        return sys.stdin.readline().split()

    def I():
        return int(sys.stdin.readline())

    def F():
        return float(sys.stdin.readline())

    def S():
        return input()

    def pf(s):
        return print(s, flush=True)

    def main():

        n, m, k = LI()

        d = LI_()

        v = [LI_() for _ in range(n)]

        dd = collections.defaultdict(lambda: None)

        for i in range(m):

            dd[d[i]] = i

        vv = []

        for c in d:

            vv.append([dd[v[c][i]] for i in range(k)])

        vvv = [[vv[i][j] for i in range(m)] for j in range(k)]

        u = set()

        m2 = 2**m

        u.add(m2 - 1)

        q = [(m2 - 1, 1)]

        ii = [2**_ for _ in range(m)]

        # print('vv',vv)

        while q:

            qd, qk = q.pop(0)

            # print('q', qd,qk)

            qdi = [di for di in range(m) if qd & ii[di]]

            for vi in range(k):

                t = 0

                vvi = vvv[vi]

                for di in qdi:

                    if not vvi[di] is None:

                        t |= ii[vvi[di]]

                # print('vit',vi,t)

                if t in u:

                    continue

                if t == 0:

                    return qk

                u.add(t)

                q.append((t, qk + 1))

        return -1

    print(main())


problem_p01812()
