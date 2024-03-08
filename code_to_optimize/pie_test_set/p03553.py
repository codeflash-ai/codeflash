def problem_p03553():
    import math, string, itertools, fractions, heapq, collections, re, array, bisect, sys, random, time, copy, functools

    sys.setrecursionlimit(10**7)

    inf = 10**20

    eps = 1.0 / 10**15

    mod = 10**9 + 7

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

    def divisions(n):

        sq = int(math.sqrt(n) + 1)

        d = collections.defaultdict(int)

        while n % 2 == 0:

            n //= 2

            d[2] += 1

        i = 3

        while n > 1 and sq >= i:

            if n % i == 0:

                n //= i

                d[i] += 1

            else:

                i += 2

        if n > 1:

            d[n] += 1

        r = [1]

        for k, v in d.items():

            for c in r[:]:

                for i in range(1, v + 1):

                    r.append(c * (k**i))

        return sorted(r)

    def main():

        n = I()

        a = LI()

        s = set()

        for i in range(n, 0, -1):

            d = divisions(i)

            ld = len(d)

            for j in range(1, 2**ld):

                c = []

                ff = True

                for k in range(ld):

                    if j & (1 << k):

                        f = True

                        for e in c:

                            if d[k] % e == 0:

                                f = False

                                ff = False

                                break

                        if f:

                            c.append(d[k])

                    if not ff:

                        break

                if ff:

                    s.add(tuple(c + [n + 1]))

        b = sorted(list(s), reverse=True)

        for c in b:

            t = 0

            for j in range(1, n + 1):

                f = False

                for e in c:

                    if j % e == 0:

                        f = True

                        break

                if f:

                    t += a[j - 1]

            if t < 0:

                for j in range(1, n + 1):

                    f = False

                    for e in c:

                        if j % e == 0:

                            f = True

                            break

                    if f:

                        a[j - 1] = 0

        return sum(a)

    print(main())


problem_p03553()
