def problem_p01753():
    import math, string, itertools, fractions, heapq, collections, re, array, bisect, sys, random, time, copy, functools

    sys.setrecursionlimit(10**7)

    inf = 10**20

    eps = 1.0 / 10**10

    mod = 998244353

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

        n, q = LI()

        a = [LI() for _ in range(n)]

        b = [LI() for _ in range(q)]

        rr = []

        def k(a, b):

            return sum([(a[i] - b[i]) ** 2 for i in range(3)]) ** 0.5

        def f(a, b, c, r):

            ab = k(a, b)

            ac = k(a, c)

            bc = k(b, c)

            if ac <= r or bc <= r:

                return True

            at = ac**2 - r**2

            bt = bc**2 - r**2

            t = max(at, 0) ** 0.5 + max(bt, 0) ** 0.5

            return ab >= t - eps

        for x1, y1, z1, x2, y2, z2 in b:

            tr = 0

            ta = (x1, y1, z1)

            tb = (x2, y2, z2)

            for x, y, z, r, l in a:

                if f(ta, tb, (x, y, z), r):

                    tr += l

            rr.append(tr)

        return "\n".join(map(str, rr))

    print(main())


problem_p01753()
